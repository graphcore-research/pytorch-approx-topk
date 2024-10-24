// Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <limits>
#include <tuple>

#include <torch/extension.h>

#include <c10/macros/Macros.h>

namespace approx_topk
{
  using namespace at;
  using namespace at::native;

  namespace
  {

    template <typename IndexType>
    __device__ std::tuple<IndexType, IndexType, IndexType> calculateBucket(
        IndexType bucketId, IndexType inputSliceSize, IndexType numBuckets,
        bool interleaved)
    {
      // If the number of buckets divides the input slice size then we just equally
      // divide the input slice between the buckets.
      // If the number of buckets does not exactly divide the slice size then we round
      // the bucket size down leaving some remainder, r, of the slice. In order to cover
      // the remainder, we increase the size of the first r buckets by one.
      auto baseInputBucketSize = inputSliceSize / numBuckets;
      auto remainder = inputSliceSize - baseInputBucketSize * numBuckets;
      auto inputBucketSize = baseInputBucketSize;
      if (bucketId < remainder)
        inputBucketSize += 1;

      IndexType inputStartIndex, inputIndexStride;
      if (interleaved)
      {
        inputStartIndex = bucketId;
        inputIndexStride = numBuckets;
      }
      else
      {
        auto previousBigBuckets = min(bucketId, remainder);
        auto previousNormalBuckets = bucketId > remainder ? bucketId - remainder : 0;
        inputStartIndex =
            previousBigBuckets * (baseInputBucketSize + 1) + previousNormalBuckets * baseInputBucketSize;
        inputIndexStride = 1;
      }
      return {inputBucketSize, inputStartIndex, inputIndexStride};
    }

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
    __device__ void findJLargest(
        at::cuda::detail::TensorInfo<const T, IndexType> input, IndexType sliceIndex,
        IndexType offset, IndexType length, IndexType stride, bool largest,
        T *values, IndexType *indices)
    {
      offset +=
          at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(sliceIndex, input);
      const T *inputStart = &input.data[offset];

#pragma unroll
      for (IndexType i = 0; i < J; i++)
      {
        if (largest)
          values[i] = std::numeric_limits<T>::lowest();
        else
          indices[i] = std::numeric_limits<T>::max();
      }

      for (IndexType i = 0; i < length; i++)
      {
        T v = doLdg(&inputStart[i * stride]);
        insertIntoQueues<T, IndexType, J>(v, i, values, indices, largest);
      }
    }

    template <typename IndexType, int J>
    __device__ void correctIndices(
        IndexType *largestIndices,
        IndexType indexCorrectionOffset, IndexType indexCorrectionStride)
    {
#pragma unroll
      for (IndexType i = 0; i < J; i++)
      {
        largestIndices[i] =
            indexCorrectionOffset + largestIndices[i] * indexCorrectionStride;
      }
    }

    template <typename T, typename IndexType, int Dim, int J>
    __device__ void writeOutputs(
        T *largestValues, IndexType *largestIndices,
        IndexType sliceIndex, IndexType offset,

        at::cuda::detail::TensorInfo<T, IndexType> topK,
        IndexType topKWithinSliceStride,

        at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
        IndexType indicesWithinSliceStride)
    {
      IndexType valuesOffset =
          offset + at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(sliceIndex, topK);
      IndexType indicesOffset =
          offset + at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(sliceIndex, indices);
      T *valuesOutputStart = &topK.data[valuesOffset];
      int64_t *indicesOutputStart = &indices.data[indicesOffset];

#pragma unroll
      for (IndexType i = 0; i < J; i++)
      {
        valuesOutputStart[i * topKWithinSliceStride] = largestValues[i];
        indicesOutputStart[i * indicesWithinSliceStride] = largestIndices[i];
      }
    }

    template <typename T, typename IndexType, int Dim, int J>
    __global__ void threadTopk(
        at::cuda::detail::TensorInfo<const T, IndexType> input,
        IndexType inputSliceSize,
        IndexType k,
        bool largest,
        bool interleaved,

        IndexType numInputSlices,
        IndexType inputWithinSliceStride,

        at::cuda::detail::TensorInfo<T, IndexType> topK,
        IndexType topKWithinSliceStride,

        at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
        IndexType indicesWithinSliceStride)
    {
      IndexType sliceIndex = blockIdx.y;
      IndexType bucketId = blockIdx.x * 32 + threadIdx.x;
      IndexType numBuckets = k / J;
      if (sliceIndex >= numInputSlices || bucketId >= numBuckets)
        return;

      auto [inputBucketSize, inputStartIndex, inputIndexStride] =
          calculateBucket(bucketId, inputSliceSize, numBuckets, interleaved);
      IndexType inputOffset = inputStartIndex * inputWithinSliceStride;
      IndexType inputStride = inputWithinSliceStride * inputIndexStride;

      T largestValues[J];
      IndexType largestIndices[J];
      findJLargest<T, IndexType, Dim, J>(
          input, sliceIndex, inputOffset, inputBucketSize, inputStride, largest,
          largestValues, largestIndices);
      correctIndices<IndexType, J>(largestIndices, inputStartIndex, inputIndexStride);

      IndexType outputBucketSize = k / numBuckets;
      IndexType outputOffset = bucketId * outputBucketSize;
      writeOutputs<T, IndexType, Dim, J>(
          largestValues, largestIndices,
          sliceIndex, outputOffset,
          topK, topKWithinSliceStride, indices, indicesWithinSliceStride);
    }

    struct Descending
    {
      template <typename T>
      __device__ bool operator()(const T &lhs, const T &rhs)
      {
        return lhs > rhs;
      }
    };

    template <typename T, typename IndexType, int Dim, int J, int NumThreads>
    __global__ void blockTopk(
        at::cuda::detail::TensorInfo<const T, IndexType> input,
        IndexType inputSliceSize,
        IndexType k,
        bool largest,
        bool interleaved,

        IndexType numInputSlices,
        IndexType inputWithinSliceStride,

        at::cuda::detail::TensorInfo<T, IndexType> topK,
        IndexType topKWithinSliceStride,

        at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
        IndexType indicesWithinSliceStride)
    {
      IndexType sliceIndex = blockIdx.y;
      IndexType bucketId = blockIdx.x;
      IndexType threadId = threadIdx.x;
      IndexType numBuckets = k / J;
      if (sliceIndex >= numInputSlices || bucketId >= numBuckets)
        return;

      auto [bucketSize, bucketStartIndex, bucketIndexStride] =
          calculateBucket(bucketId, inputSliceSize, numBuckets, interleaved);
      IndexType numThreads = min((int64_t)blockDim.x, (int64_t)bucketSize);

      IndexType inputStartIndex = bucketStartIndex + threadId * bucketIndexStride;
      IndexType inputOffset = inputStartIndex * inputWithinSliceStride;
      IndexType inputLength = bucketSize / numThreads;
      if (threadId < bucketSize - inputLength * numThreads)
        inputLength += 1;
      IndexType inputIndexStride = bucketIndexStride * numThreads;
      IndexType inputStride = inputWithinSliceStride * inputIndexStride;

      // We might launch more threads than needed in which case we don't want to process
      // any inputs. We can't just early exit because this thread's queues will still
      // take part in the sort later, thus we need to make sure we initialise our
      // queues otherwise uninitialised values will end up in the output.
      if (threadId >= bucketSize)
        inputLength = 0;

      T largestValues[J];
      IndexType largestIndices[J];
      findJLargest<T, IndexType, Dim, J>(
          input, sliceIndex, inputOffset, inputLength, inputStride, largest,
          largestValues, largestIndices);
      correctIndices<IndexType, J>(largestIndices, inputStartIndex, inputIndexStride);

      using BlockMergeSort = cub::BlockMergeSort<T, NumThreads, J, IndexType>;
      __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;
      BlockMergeSort(temp_storage_shuffle).Sort(largestValues, largestIndices, Descending());

      // The first thread contains the top-j, thus is the only one to write.
      if (threadId > 0)
        return;

      IndexType outputBucketSize = k / numBuckets;
      IndexType outputOffset = bucketId * outputBucketSize;
      writeOutputs<T, IndexType, Dim, J>(
          largestValues, largestIndices,
          sliceIndex, outputOffset,
          topK, topKWithinSliceStride, indices, indicesWithinSliceStride);
    }

    template <typename T, typename IndexType, int Dim, int J>
    void launchKernel(
        at::cuda::detail::TensorInfo<const T, IndexType> input,
        IndexType inputSliceSize,
        IndexType k,
        bool largest,
        bool interleaved,
        bool multithreadBuckets,

        IndexType numInputSlices,
        IndexType inputWithinSliceStride,

        at::cuda::detail::TensorInfo<T, IndexType> topK,
        IndexType topKWithinSliceStride,

        at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
        IndexType indicesWithinSliceStride)
    {
      // FIXME: This should be the device the tensors are on, not the current device.
      const auto kMaxGridSize = at::cuda::getCurrentDeviceProperties()->maxGridSize;
      auto warp_size = at::cuda::warp_size();
      // We use the y dimension of the grid for batches provided by the user.
      TORCH_CHECK(numInputSlices < kMaxGridSize[1], "Too many slices for topk");
      TORCH_CHECK(k <= inputSliceSize, "topk: k must not be larger than topk size");
      IndexType numBuckets = k / J;

      if (multithreadBuckets)
      {
        TORCH_CHECK(numBuckets <= kMaxGridSize[0], "topk: too many buckets")
        dim3 grid(numBuckets, numInputSlices, 1);
        const uint kNumThreads = 64;
        dim3 block(kNumThreads);
        blockTopk<T, IndexType, Dim, J, kNumThreads><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
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
      }
      else
      {
        // We use one thread per bucket and group them into one warp per block.
        IndexType gridX = at::ceil_div((int64_t)numBuckets, (int64_t)warp_size);
        TORCH_CHECK(gridX <= kMaxGridSize[0], "topk: too many buckets")
        dim3 grid(gridX, numInputSlices, 1);
        dim3 block(warp_size);
        threadTopk<T, IndexType, Dim, J><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
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
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  void priority_queue_topk(
      const Tensor &self, int64_t k, int64_t j, // int64_t k_mult,
      int64_t dim, bool largest, bool interleaved, bool multithread_buckets,
      const Tensor &values, const Tensor &indices)
  {
    TensorArg input_arg{self, "xs", 1}, topK_arg{values, "valuesOutput", 2},
        indices_arg{indices, "indicesOutput", 3};
    checkAllSameGPU(__func__, {input_arg, topK_arg, indices_arg});

    int numDims = self.dim();
    numDims = numDims == 0 ? 1 : numDims;
    TORCH_CHECK(numDims <= MAX_DIMS, "topk: input tensor has too many dimensions");
    int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);

    // k=0 is valid but no work needs to be done, so exit early to simplify the rest
    // of the implementation.
    if (k == 0)
      return;
    TORCH_CHECK(k % j == 0, "topk: j must divide k");

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
      multithread_buckets,                                       \
      static_cast<INDEX_T>(numInputSlices),                      \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]), \
      topKInfo,                                                  \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),   \
      indicesInfo,                                               \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]));

    // J has to be a statically known template parameter so that the priorty queue can
    // be kept in registers rather than local memory.
#define RUN_K(INDEX_T, DIM)                       \
  if (j == 1)                                     \
  {                                               \
    RUN_J(INDEX_T, DIM, 1);                       \
  }                                               \
  else if (j == 2)                                \
  {                                               \
    RUN_J(INDEX_T, DIM, 2);                       \
  }                                               \
  else if (j == 3)                                \
  {                                               \
    RUN_J(INDEX_T, DIM, 3);                       \
  }                                               \
  else if (j == 4)                                \
  {                                               \
    RUN_J(INDEX_T, DIM, 4);                       \
  }                                               \
  else                                            \
  {                                               \
    TORCH_CHECK(false, "topk: j must 0 < j < 5"); \
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
