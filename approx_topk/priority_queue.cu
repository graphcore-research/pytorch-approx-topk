
// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <limits>

#include <torch/extension.h>

#include <c10/macros/Macros.h>

namespace approx_topk
{
  using namespace at;
  using namespace at::native;

  namespace
  {
    template <typename T>
    struct AddOp
    {
      __device__ __forceinline__ T operator()(T const &lhs, T const &rhs)
      {
        return (lhs + rhs);
      }
    };

    template <typename T, typename IndexType, int J>
    __device__ void insertIntoQueues(
        T v, IndexType index,
        T *valueQueue, IndexType *indexQueue,
        bool largest)
    {
      // The smallest (or largest) item is at the start of the queue. We walk down the
      // queue inserting the new item (if possible), and shifting the existing, smaller,
      // items towards the front of the queue.
#pragma unroll
      for (IndexType i = 0; i < J; i++)
      {
        if ((largest && valueQueue[i] > v) || (!largest && valueQueue[i] < v))
          break;
        if (i > 0)
        {
          valueQueue[i - 1] = valueQueue[i];
          indexQueue[i - 1] = indexQueue[i];
        }
        valueQueue[i] = v;
        indexQueue[i] = index;
      }
    }

    template <typename T, typename IndexType, int Dim, int J>
    __global__ void priorityQueueTopk(
        at::cuda::detail::TensorInfo<const T, IndexType> input,
        IndexType inputSliceSize,
        IndexType k, // aka `k`
        bool largest,
        bool interleaved,

        IndexType numInputSlices,
        IndexType inputWithinSliceStride,

        at::cuda::detail::TensorInfo<T, IndexType> topK,
        IndexType topKWithinSliceStride,

        at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
        IndexType indicesWithinSliceStride)
    {
      IndexType sliceIndex = blockIdx.x;
      IndexType bucketIndex = blockIdx.y * 32 + threadIdx.x;
      IndexType numBuckets = k == 0 ? 1 : k / J;
      if (sliceIndex >= numInputSlices || bucketIndex >= numBuckets)
      {
        return;
      }

      IndexType outputBucketSize = k / numBuckets;
      // If the number of buckets divides the input slice size then we just equally
      // divide the input slice between the buckets.
      // If the number of buckets does not exactly divide the slice size then we round
      // the bucket size down leaving some remainder, r, of the slice. In order to cover
      // the remainder, we increase the size of the first r buckets by one.
      IndexType baseInputBucketSize = inputSliceSize / numBuckets;
      IndexType remainder = inputSliceSize - baseInputBucketSize * numBuckets;
      IndexType inputBucketSize = baseInputBucketSize;
      if (bucketIndex < remainder)
      {
        inputBucketSize += 1;
      }

      IndexType previousBigBuckets = min(bucketIndex, remainder);
      IndexType previousNormalBuckets =
          bucketIndex > remainder ? bucketIndex - remainder : 0;
      IndexType inputBucketOffset =
          previousBigBuckets * (baseInputBucketSize + 1) + previousNormalBuckets * baseInputBucketSize;
      IndexType inputStartIndex =
          at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(sliceIndex, input);
      if (interleaved)
      {
        inputStartIndex += bucketIndex;
        inputWithinSliceStride *= numBuckets;
      }
      else
      {
        inputStartIndex += inputBucketOffset;
      }

      const T *inputStart = &input.data[inputStartIndex];

      T valueQueue[J];
      IndexType indexQueue[J];
#pragma unroll
      for (IndexType i = 0; i < J; i++)
      {
        if (largest)
          valueQueue[i] = std::numeric_limits<T>::lowest();
        else
          valueQueue[i] = std::numeric_limits<T>::max();
      }

      for (IndexType i = 0; i < inputBucketSize; i++)
      {
        T v = doLdg(&inputStart[i * inputWithinSliceStride]);
        insertIntoQueues<T, IndexType, J>(v, i, valueQueue, indexQueue, largest);
      }

      IndexType valuesOutputStartIndex =
          at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(sliceIndex, topK);
      IndexType indicesOutputStartIndex =
          at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(sliceIndex, indices);
      valuesOutputStartIndex += bucketIndex * outputBucketSize;
      indicesOutputStartIndex += bucketIndex * outputBucketSize;
      T *valuesOutputStart = &topK.data[valuesOutputStartIndex];
      int64_t *indicesOutputStart = &indices.data[indicesOutputStartIndex];

#pragma unroll
      for (IndexType i = 0; i < J; i++)
      {
        IndexType topKOffset = i * topKWithinSliceStride;
        IndexType indexOffset = i * indicesWithinSliceStride;
        valuesOutputStart[topKOffset] = valueQueue[i];

        IndexType trueIndex;
        if (interleaved)
          trueIndex = bucketIndex + indexQueue[i] * numBuckets;
        else
          trueIndex = inputBucketOffset + indexQueue[i];
        indicesOutputStart[indexOffset] = trueIndex;
      }
    };

    template <typename T, typename IndexType, int Dim, int J>
    void launchKernel(
        at::cuda::detail::TensorInfo<const T, IndexType> input,
        IndexType inputSliceSize,
        IndexType k, // aka `k`
        bool largest,
        bool interleaved,

        IndexType numInputSlices,
        IndexType inputWithinSliceStride,

        at::cuda::detail::TensorInfo<T, IndexType> topK,
        IndexType topKWithinSliceStride,

        at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
        IndexType indicesWithinSliceStride)
    {
      // We use the x dimension of the grid for batches provided by the user.
      // There is then one thread per bucket. We group these into warps of 32, and put
      // each warp in its own block.
      // 2^31 - 1 = the max grid size in the x dimension, from compute capability 3.0.
      TORCH_CHECK(k <= inputSliceSize, "topk k must not be larger than topk size");
      TORCH_INTERNAL_ASSERT(numInputSlices < 2 ^ 31 - 1, "Too many slices for topk");
      IndexType numBuckets = k == 0 ? 1 : k / J;
      int warp_size = at::cuda::warp_size();
      IndexType blockY = at::ceil_div((int64_t)numBuckets, (int64_t)warp_size);
      dim3 grid(numInputSlices, blockY, 1);
      dim3 block(warp_size);

      priorityQueueTopk<T, IndexType, Dim, J><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          input,
          inputSliceSize,
          k,
          largest,
          interleaved,
          numInputSlices,
          inputWithinSliceStride,
          topK,
          topKWithinSliceStride,
          indices,
          indicesWithinSliceStride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  void priority_queue_topk(
      const Tensor &self, int64_t k, int64_t j, int64_t dim, bool largest,
      bool interleaved, const Tensor &values, const Tensor &indices)
  {
    TensorArg input_arg{self, "xs", 1}, topK_arg{values, "valuesOutput", 2},
        indices_arg{indices, "indicesOutput", 3};
    checkAllSameGPU(__func__, {input_arg, topK_arg, indices_arg});

    int numDims = self.dim();
    numDims = numDims == 0 ? 1 : numDims;
    TORCH_CHECK(numDims <= MAX_DIMS, "input tensor has too many dimensions");
    int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);

    TORCH_CHECK(k == 0 || k % j == 0, "topk j must divide k");
    TORCH_CHECK(k == 0 || j > 0, "topk j must be > 0")
    if (k == 0 && j == 0)
    {
      // This is valid but no work needs to be done. It's better to early exit than
      // instantiate another template for J=0 for this no-op configuration.
      return;
    }

    auto input = self.contiguous();
    // static_cast is required to ensure that the correct type (INDEX_T)
    // is provided to the kernel for the arguments.
#define RUN_J(INDEX_T, DIM, J)                                   \
  launchKernel<scalar_t, INDEX_T, DIM, J>(                       \
      inputInfo,                                                 \
      static_cast<INDEX_T>(sliceSize),                           \
      static_cast<INDEX_T>(k),                                   \
      largest,                                                   \
      interleaved,                                               \
      static_cast<INDEX_T>(numInputSlices),                      \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]), \
      topKInfo,                                                  \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),   \
      indicesInfo,                                               \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]));

    // J has to be a statically known template parameter so that the priorty queue can
    // be kept in registers rather than local memory.
#define RUN_K(INDEX_T, DIM)              \
  if (j == 1)                            \
  {                                      \
    RUN_J(INDEX_T, DIM, 1);              \
  }                                      \
  else if (j == 2)                       \
  {                                      \
    RUN_J(INDEX_T, DIM, 2);              \
  }                                      \
  else if (j == 3)                       \
  {                                      \
    RUN_J(INDEX_T, DIM, 3);              \
  }                                      \
  else if (j == 4)                       \
  {                                      \
    RUN_J(INDEX_T, DIM, 4);              \
  }                                      \
  else                                   \
  {                                      \
    TORCH_CHECK(false, "j must be < 5"); \
  }

#define RUN_DIM(INDEX_T) \
  if (allDims == 1)      \
  {                      \
    RUN_K(INDEX_T, 1);   \
  }                      \
  else if (allDims == 2) \
  {                      \
    RUN_K(INDEX_T, 2);   \
  }                      \
  else if (allDims == 3) \
  {                      \
    RUN_K(INDEX_T, 3);   \
  }                      \
  else                   \
  {                      \
    RUN_K(INDEX_T, -1);  \
  }

#define RUN_T(INDEX_T) \
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "topk_out_cuda", [&] { \
    at::cuda::detail::TensorInfo<const scalar_t, INDEX_T> inputInfo =     \
      at::cuda::detail::getTensorInfo<const scalar_t, INDEX_T>(input);    \
    at::cuda::detail::TensorInfo<scalar_t, INDEX_T> topKInfo =            \
      at::cuda::detail::getTensorInfo<scalar_t, INDEX_T>(values);         \
    at::cuda::detail::TensorInfo<int64_t, INDEX_T> indicesInfo =          \
      at::cuda::detail::getTensorInfo<int64_t, INDEX_T>(indices);         \
    /* tensorInfoLegacyIfScalar*/                                         \
    if (!input.dim()) {                                                   \
      inputInfo.dims = 1;                                                 \
      inputInfo.sizes[0] = 1;                                             \
      inputInfo.strides[0] = 1;                                           \
      topKInfo.dims = 1;                                                  \
      topKInfo.sizes[0] = 1;                                              \
      topKInfo.strides[0] = 1;                                            \
      indicesInfo.dims = 1;                                               \
      indicesInfo.sizes[0] = 1;                                           \
      indicesInfo.strides[0] = 1;                                         \
    }                                                                     \
    /* We use these structures solely to find the offset to */            \
    /* each slice we are operating on */                                  \
    inputInfo.sizes[dim] = 1;                                             \
    topKInfo.sizes[dim] = 1;                                              \
    indicesInfo.sizes[dim] = 1;                                           \
    /* stash the stride of dim because it can be accidentally collapsed */ \
    auto strideTopK = topKInfo.strides[dim];                              \
    auto strideIndices = indicesInfo.strides[dim];                        \
    /* Collapse all other dims */                                         \
    int collapseInputDim = inputInfo.collapseDims(dim);                   \
    int collapseTopKDim = topKInfo.collapseDims(dim);                     \
    int collapseIndicesDim = indicesInfo.collapseDims(dim);               \
    /* restore stride in case it was collapsed */                         \
    topKInfo.strides[collapseTopKDim] = strideTopK;                       \
    indicesInfo.strides[collapseIndicesDim] = strideIndices;              \
    int64_t numInputSlices = 1;                                           \
    for (int i = 0; i < inputInfo.dims; ++i) {                            \
      numInputSlices *= inputInfo.sizes[i];                               \
    }                                                                     \
                                                                          \
    /* This is used as a template parameter to calculate indices. */      \
    /* We only specialize it if all collapsed dim sizes are the */        \
    /* same; otherwise, we use -1 which is the specialization */          \
    /* parameter for arbitrary dimensions */                              \
    int allDims = inputInfo.dims;                                         \
    if (topKInfo.dims != allDims || indicesInfo.dims != allDims) {        \
      allDims = -1;                                                       \
    }                                                                     \
                                                                          \
    RUN_DIM(INDEX_T); });

    // the below is safe with 0-dimensional tensors because it is based on
    // TensorInfo which implicitly expands to 1-dimensional.
    if (input.numel() > 0)
    {
      // Based on required index size, run the algorithm with the
      // appropriate index type
      if (at::cuda::detail::canUse32BitIndexMath(input) &&
          at::cuda::detail::canUse32BitIndexMath(values) &&
          at::cuda::detail::canUse32BitIndexMath(indices))
      {
        RUN_T(uint32_t);
      }
      else
      {
        RUN_T(uint64_t);
      }
    }
#undef RUN_T
#undef RUN_DIM
#undef RUN_K
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
  {
    m.def("priority_queue_topk", &priority_queue_topk);
  }

} // approx_topk
