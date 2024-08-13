// Radix-select top-k that splits a large top-k into buckets to improve parallelism.
// Largely copied from Pytorch:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/TensorTopK.cu
//
// The original PyTorch implementation computes an exact top-k across multiple SMs. This
// implementation only supports computing an exact top-k on a single SM, but allows an
// approximate top-k to be computed by splitting the sequence into buckets and computing
// an exact top-k of each bucket individually.

// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>

#include <torch/extension.h>

#include <c10/macros/Macros.h>

namespace pytorch_topk
{
  using namespace at;
  using namespace at::native;

  template <typename T>
  struct AddOp
  {
    __device__ __forceinline__ T operator()(T const &lhs, T const &rhs)
    {
      return (lhs + rhs);
    }
  };

  template <typename T, typename IndexType, int Dim>
  C10_LAUNCH_BOUNDS_1(1024)
  __global__ void gatherTopK(at::cuda::detail::TensorInfo<const T, IndexType> input,
                             IndexType inputSliceSize,
                             IndexType outputSliceSize, // aka `k`
                             IndexType j,
                             bool largest,

                             IndexType numInputSlices,
                             IndexType inputWithinSliceStride,

                             at::cuda::detail::TensorInfo<T, IndexType> topK,
                             IndexType topKWithinSliceStride,

                             at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
                             IndexType indicesWithinSliceStride)
  {
    // Indices are limited to integer fp precision, so counts can fit in
    // int32, regardless of IndexType
#if defined(USE_ROCM)
    __shared__ int smem[64];
#else
    __shared__ int smem[32]; // one per each warp, up to warp limit
#endif
    IndexType sliceIndex = blockIdx.x;
    IndexType bucketIndex = blockIdx.y;
    IndexType numBuckets = outputSliceSize == 0 ? 1 : outputSliceSize / j;
    if (sliceIndex >= numInputSlices || bucketIndex >= numBuckets)
    {
      return;
    }

    IndexType outputBucketSize = outputSliceSize / numBuckets;
    // If the number of buckets divides the input slice size then we just equally divide
    // the input slice between the buckets.
    // If the number of buckets does not exactly divide the slice size then we round the
    // bucket size down leaving some remainder, r, of the slice. In order to cover the
    // remainder, we increase the size of the first r buckets by one.
    IndexType baseInputBucketSize = inputSliceSize / numBuckets;
    IndexType remainder =
        inputSliceSize - baseInputBucketSize * (inputSliceSize / baseInputBucketSize);
    IndexType thisBlockInputBucketSize = baseInputBucketSize;
    if (bucketIndex < remainder)
    {
      thisBlockInputBucketSize += 1;
    }

    // Find the start offset of the slice in the input and outputs.
    IndexType inputStartIndex =
        at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(sliceIndex, input);
    IndexType valuesOutputStartIndex =
        at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(sliceIndex, topK);
    IndexType indicesOutputStartIndex =
        at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(sliceIndex, indices);

    // Work out the start position of this block's bucket in the input.
    IndexType previousBigBuckets = min(bucketIndex, remainder);
    IndexType previousNormalBuckets =
        bucketIndex > remainder ? bucketIndex - remainder : 0;
    IndexType inputBucketOffset =
        previousBigBuckets * (baseInputBucketSize + 1) + previousNormalBuckets * baseInputBucketSize;

    // Adjust the indices to point at the start of the bucket within the slice.
    inputStartIndex += inputBucketOffset;
    valuesOutputStartIndex += bucketIndex * outputBucketSize;
    indicesOutputStartIndex += bucketIndex * outputBucketSize;

    const T *inputStart = &input.data[inputStartIndex];
    T *valuesOutputStart = &topK.data[valuesOutputStartIndex];
    int64_t *indicesOutputStart = &indices.data[indicesOutputStartIndex];

    // Find the j-th highest element in this block's slice and bucket.
    T topKValue;
    topKValue = static_cast<T>(0);
    radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType>(
        inputStart, j, largest,
        thisBlockInputBucketSize, inputWithinSliceStride,
        smem, &topKValue);
    const auto topKConverted = at::native::TopKTypeConfig<T>::convert(topKValue);

    // Every value that is strictly less/greater than `pattern`
    // (depending on sort dir) in sorted int format is in the top-K.
    // The top-K value itself might not be unique.
    //
    // Since there are a variable number of elements that we see that
    // are within the top-k, we don't know at what index to write out
    // the resulting values.
    // In order to get this, we perform an exclusive prefix sum of
    // `hasTopK`. This will return the resulting index into which we
    // need to write the result, if a thread has a result.

    // All threads need to participate in the loop and the prefix sum,
    // but not necessarily in the load; hence loop bounds being rounded
    // up to a multiple of the block dim.
    IndexType numIterations = round_up(thisBlockInputBucketSize, (IndexType)blockDim.x);
    IndexType writeIndexStart = 0;

    for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x)
    {
      bool inRange = (i < thisBlockInputBucketSize);
      T v =
          inRange ? doLdg(&inputStart[i * inputWithinSliceStride]) : static_cast<T>(0);
      const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
      bool hasTopK;
      if (largest)
      {
        hasTopK = inRange && (convertedV > topKConverted);
      }
      else
      {
        hasTopK = inRange && (convertedV < topKConverted);
      }

      int index;
      int carry;
      at::cuda::exclusiveBinaryPrefixScan<int, true>(
          smem, hasTopK, &index, &carry, AddOp<int>());

      if (hasTopK)
      {
        int writeIndex = writeIndexStart + index;
        CUDA_KERNEL_ASSERT(writeIndex < j);

        IndexType topKOffset = writeIndex * topKWithinSliceStride;
        IndexType indexOffset = writeIndex * indicesWithinSliceStride;

        valuesOutputStart[topKOffset] = v;
        indicesOutputStart[indexOffset] = i + inputBucketOffset;
      }

      writeIndexStart += carry;
    }

    // We need to fill in the rest with actual == top-K values.
    // The number that we need is j - writeIndexStart. There might be more than that
    // number available, in which case we have to choose the first seen set. We do this
    // via a prefix sum to calculate indices for writing results.
    CUDA_KERNEL_ASSERT(j >= writeIndexStart);
    IndexType topKRemaining = j - writeIndexStart;

    for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x)
    {
      bool inRange = (i < thisBlockInputBucketSize);
      T v =
          inRange ? doLdg(&inputStart[i * inputWithinSliceStride]) : static_cast<T>(0);
      const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
      bool hasTopK = inRange && (convertedV == topKConverted);

      int index;
      int carry;
      at::cuda::exclusiveBinaryPrefixScan<int, true>(
          smem, hasTopK, &index, &carry, AddOp<int>());

      if (hasTopK && index < topKRemaining)
      {
        int writeIndex = writeIndexStart + index;
        CUDA_KERNEL_ASSERT(writeIndex < j);

        IndexType topKOffset = writeIndex * topKWithinSliceStride;
        IndexType indexOffset = writeIndex * indicesWithinSliceStride;

        valuesOutputStart[topKOffset] = v;
        indicesOutputStart[indexOffset] = i + inputBucketOffset;
      }

      if (carry >= topKRemaining)
      {
        break;
      }

      topKRemaining -= carry;
      writeIndexStart += carry;
    }
  };

  template <typename T, typename IndexType, int Dim>
  void launch(
      at::cuda::detail::TensorInfo<const T, IndexType> input,
      IndexType inputSliceSize,
      IndexType outputSliceSize, // aka `k`
      IndexType j,
      bool largest,

      IndexType numInputSlices,
      IndexType inputWithinSliceStride,

      at::cuda::detail::TensorInfo<T, IndexType> topK,
      IndexType topKWithinSliceStride,

      at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
      IndexType indicesWithinSliceStride)
  {
    // We use the x dimension of the grid for batches provided by the user, and the y
    // dimension for buckets.
    // 2^31 - 1 = the max grid size in the x dimension, from compute capability 3.0.
    TORCH_INTERNAL_ASSERT(numInputSlices < 2 ^ 31 - 1, "Too many slices for topk");
    TORCH_CHECK(outputSliceSize >= 0, "topk k cannot be negative");
    TORCH_CHECK(outputSliceSize == 0 || j > 0, "topk j must be > 0");
    TORCH_CHECK(outputSliceSize == 0 || outputSliceSize % j == 0,
                "topk j must divide k");
    IndexType numBuckets = outputSliceSize == 0 ? 1 : outputSliceSize / j;
    IndexType minInputBucketSize = inputSliceSize / numBuckets;
    dim3 grid = dim3(numInputSlices, numBuckets, 1);

    int warp_size = at::cuda::warp_size();
    int num_threads = std::min(
        at::ceil_div((int64_t)minInputBucketSize, (int64_t)warp_size) * (int64_t)warp_size,
        (int64_t)1024);
    dim3 block(num_threads);

    gatherTopK<T, IndexType, Dim><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input,
        inputSliceSize,
        outputSliceSize,
        j,
        largest,
        numInputSlices,
        inputWithinSliceStride,
        topK,
        topKWithinSliceStride,
        indices,
        indicesWithinSliceStride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  void launch_gather_topk_kernel(
      const Tensor &self, int64_t k, int64_t j, int64_t dim, bool largest,
      const Tensor &values, const Tensor &indices)
  {
    TensorArg input_arg{self, "xs", 1}, topK_arg{values, "valuesOutput", 2},
        indices_arg{indices, "indicesOutput", 3};
    checkAllSameGPU(__func__, {input_arg, topK_arg, indices_arg});

    int numDims = self.dim();
    numDims = numDims == 0 ? 1 : numDims;
    TORCH_CHECK(numDims <= MAX_DIMS, "input tensor has too many dimensions");
    int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);

    auto input = self.contiguous();
    // static_cast is required to ensure that the correct type (INDEX_T)
    // is provided to the kernel for the arguments.
#define RUN_K(INDEX_T, DIM)                                      \
  pytorch_topk::launch<scalar_t, INDEX_T, DIM>(                  \
      inputInfo,                                                 \
      static_cast<INDEX_T>(sliceSize),                           \
      static_cast<INDEX_T>(k),                                   \
      static_cast<INDEX_T>(j),                                   \
      largest,                                                   \
      static_cast<INDEX_T>(numInputSlices),                      \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]), \
      topKInfo,                                                  \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),   \
      indicesInfo,                                               \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]));

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
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
  {
    m.def("topk", &launch_gather_topk_kernel);
  }

} // pytorch_topk