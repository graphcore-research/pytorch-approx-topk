// Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <limits>
#include <tuple>

#include <torch/extension.h>

#include <c10/macros/Macros.h>

namespace approx_topk {
using namespace at;
using namespace at::native;

namespace {

template <typename IndexType>
__device__ std::tuple<IndexType, IndexType, IndexType> CalculateBucket(
    IndexType bucket_id,
    IndexType input_slice_size,
    IndexType num_buckets,
    bool interleaved) {
  // If the number of buckets divides the input slice size then we just equally
  // divide the input slice between the buckets.
  // If the number of buckets does not exactly divide the slice size then we
  // round the bucket size down leaving some remainder, r, of the slice. In
  // order to cover the remainder, we increase the size of the first r buckets
  // by one.
  auto base_input_bucket_size = input_slice_size / num_buckets;
  auto remainder = input_slice_size - base_input_bucket_size * num_buckets;
  auto input_bucket_size = base_input_bucket_size;
  if (bucket_id < remainder)
    input_bucket_size += 1;

  IndexType input_start_index, input_index_stride;
  if (interleaved) {
    input_start_index = bucket_id;
    input_index_stride = num_buckets;
  } else {
    auto previous_big_buckets = min(bucket_id, remainder);
    auto previous_normal_buckets =
        bucket_id > remainder ? bucket_id - remainder : 0;
    input_start_index = previous_big_buckets * (base_input_bucket_size + 1) +
        previous_normal_buckets * base_input_bucket_size;
    input_index_stride = 1;
  }
  return {input_bucket_size, input_start_index, input_index_stride};
}

template <typename T, typename IndexType, int J>
__device__ void InsertIntoQueues(
    T v,
    IndexType index,
    T* value_queue,
    IndexType* index_queue,
    bool largest) {
  // The smallest (or largest) item is at the start of the queue. We walk down
  // the queue inserting the new item (if possible), and shifting the existing,
  // smaller, items towards the front of the queue.
#pragma unroll
  for (IndexType i = 0; i < J; i++) {
    if ((largest && value_queue[i] > v) || (!largest && value_queue[i] < v))
      break;
    if (i > 0) {
      value_queue[i - 1] = value_queue[i];
      index_queue[i - 1] = index_queue[i];
    }
    value_queue[i] = v;
    index_queue[i] = index;
  }
}

template <typename T, typename IndexType, int Dim, int J>
__device__ void FindJLargest(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    IndexType slice_index,
    IndexType offset,
    IndexType length,
    IndexType stride,
    bool largest,
    T* values,
    IndexType* indices) {
  offset += at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(
      slice_index, input);
  const T* kInputStart = &input.data[offset];

#pragma unroll
  for (IndexType i = 0; i < J; i++) {
    if (largest)
      values[i] = std::numeric_limits<T>::lowest();
    else
      indices[i] = std::numeric_limits<T>::max();
  }

  for (IndexType i = 0; i < length; i++) {
    T v = doLdg(&kInputStart[i * stride]);
    InsertIntoQueues<T, IndexType, J>(v, i, values, indices, largest);
  }
}

template <typename IndexType, int J>
__device__ void CorrectIndices(
    IndexType* largest_indices,
    IndexType index_correction_offset,
    IndexType index_correction_stride) {
#pragma unroll
  for (IndexType i = 0; i < J; i++) {
    largest_indices[i] =
        index_correction_offset + largest_indices[i] * index_correction_stride;
  }
}

template <typename T, typename IndexType, int Dim, int J>
__device__ void WriteOutputs(
    T* largest_values,
    IndexType* largest_indices,
    IndexType slice_index,
    IndexType offset,

    at::cuda::detail::TensorInfo<T, IndexType> top_k,
    IndexType top_k_within_slice_stride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indices_within_slice_stride) {
  IndexType values_offset = offset +
      at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(
                                slice_index, top_k);
  IndexType indices_offset = offset +
      at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(
                                 slice_index, indices);
  T* values_output_start = &top_k.data[values_offset];
  int64_t* indices_output_start = &indices.data[indices_offset];

#pragma unroll
  for (IndexType i = 0; i < J; i++) {
    values_output_start[i * top_k_within_slice_stride] = largest_values[i];
    indices_output_start[i * indices_within_slice_stride] = largest_indices[i];
  }
}

template <typename T, typename IndexType, int Dim, int J>
__global__ void ThreadTopK(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    IndexType input_slice_size,
    IndexType k,
    bool largest,
    bool interleaved,

    IndexType num_input_slices,
    IndexType input_within_slice_stride,

    at::cuda::detail::TensorInfo<T, IndexType> top_k,
    IndexType top_k_within_slice_stride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indices_within_slice_stride) {
  IndexType slice_index = blockIdx.y;
  IndexType bucket_id = blockIdx.x * 32 + threadIdx.x;
  IndexType num_buckets = k / J;
  if (slice_index >= num_input_slices || bucket_id >= num_buckets)
    return;

  auto [input_bucket_size, input_start_index, input_index_stride] =
      CalculateBucket(bucket_id, input_slice_size, num_buckets, interleaved);
  IndexType input_offset = input_start_index * input_within_slice_stride;
  IndexType input_stride = input_within_slice_stride * input_index_stride;

  T largest_values[J];
  IndexType largest_indices[J];
  FindJLargest<T, IndexType, Dim, J>(
      input,
      slice_index,
      input_offset,
      input_bucket_size,
      input_stride,
      largest,
      largest_values,
      largest_indices);
  CorrectIndices<IndexType, J>(
      largest_indices, input_start_index, input_index_stride);

  IndexType output_bucket_size = k / num_buckets;
  IndexType output_offset = bucket_id * output_bucket_size;
  WriteOutputs<T, IndexType, Dim, J>(
      largest_values,
      largest_indices,
      slice_index,
      output_offset,
      top_k,
      top_k_within_slice_stride,
      indices,
      indices_within_slice_stride);
}

struct Descending {
  template <typename T>
  __device__ bool operator()(const T& lhs, const T& rhs) {
    return lhs > rhs;
  }
};

template <typename T, typename IndexType, int Dim, int J, int NumThreads>
__global__ void BlockTopK(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    IndexType input_slice_size,
    IndexType k,
    bool largest,
    bool interleaved,

    IndexType num_input_slices,
    IndexType input_within_slice_stride,

    at::cuda::detail::TensorInfo<T, IndexType> top_k,
    IndexType top_k_within_slice_stride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indices_within_slice_stride) {
  IndexType slice_index = blockIdx.y;
  IndexType bucket_id = blockIdx.x;
  IndexType thread_id = threadIdx.x;
  IndexType num_buckets = k / J;
  if (slice_index >= num_input_slices || bucket_id >= num_buckets)
    return;

  auto [bucket_size, bucket_start_index, bucket_index_stride] =
      CalculateBucket(bucket_id, input_slice_size, num_buckets, interleaved);
  IndexType num_threads = min((int64_t)blockDim.x, (int64_t)bucket_size);

  IndexType input_start_index =
      bucket_start_index + thread_id * bucket_index_stride;
  IndexType input_offset = input_start_index * input_within_slice_stride;
  IndexType input_length = bucket_size / num_threads;
  if (thread_id < bucket_size - input_length * num_threads)
    input_length += 1;
  IndexType input_index_stride = bucket_index_stride * num_threads;
  IndexType input_stride = input_within_slice_stride * input_index_stride;

  // We might launch more threads than needed in which case we don't want to
  // process any inputs. We can't just early exit because this thread's queues
  // will still take part in the sort later, thus we need to make sure we
  // initialise our queues otherwise uninitialised values will end up in the
  // output.
  if (thread_id >= bucket_size)
    input_length = 0;

  T largest_values[J];
  IndexType largest_indices[J];
  FindJLargest<T, IndexType, Dim, J>(
      input,
      slice_index,
      input_offset,
      input_length,
      input_stride,
      largest,
      largest_values,
      largest_indices);
  CorrectIndices<IndexType, J>(
      largest_indices, input_start_index, input_index_stride);

  __syncthreads();
  using BlockMergeSort = cub::BlockMergeSort<T, NumThreads, J, IndexType>;
  __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;
  BlockMergeSort(temp_storage_shuffle)
      .Sort(largest_values, largest_indices, Descending());

  // The first thread contains the top-j, thus is the only one to write.
  if (thread_id > 0)
    return;

  IndexType output_bucket_size = k / num_buckets;
  IndexType output_offset = bucket_id * output_bucket_size;
  WriteOutputs<T, IndexType, Dim, J>(
      largest_values,
      largest_indices,
      slice_index,
      output_offset,
      top_k,
      top_k_within_slice_stride,
      indices,
      indices_within_slice_stride);
}

template <typename IndexType>
bool ShouldUseMultithreadBuckets(
    IndexType num_input_slices,
    IndexType input_slice_size,
    IndexType buckets_per_slice) {
  IndexType total_buckets = num_input_slices * buckets_per_slice;
  IndexType bucket_size = input_slice_size / buckets_per_slice;
  // TODO: Get actual values for GPU in use.
  int n_sms = 120;
  int threads_per_warp = 32;
  bool lots_of_buckets = total_buckets >= n_sms * threads_per_warp;
  bool small_buckets = bucket_size < 64;
  return !(lots_of_buckets || small_buckets);
}

template <typename T, typename IndexType, int Dim, int J>
void LaunchKernel(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    IndexType input_slice_size,
    IndexType k,
    bool largest,
    bool interleaved,
    c10::optional<bool> multithread_buckets,

    IndexType num_input_slices,
    IndexType input_within_slice_stride,

    at::cuda::detail::TensorInfo<T, IndexType> top_k,
    IndexType top_k_within_slice_stride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indices_within_slice_stride) {
  // FIXME: This should be the device the tensors are on, not the current
  // device.
  const auto kMaxGridSize = at::cuda::getCurrentDeviceProperties()->maxGridSize;
  auto warp_size = at::cuda::warp_size();
  // We use the y dimension of the grid for batches provided by the user.
  TORCH_CHECK(num_input_slices < kMaxGridSize[1], "Too many slices for topk");
  TORCH_CHECK(
      k <= input_slice_size, "topk: k must not be larger than topk size");
  IndexType num_buckets = k / J;

  bool should_use_multithread_buckets =
      multithread_buckets.value_or(ShouldUseMultithreadBuckets<IndexType>(
          num_input_slices, input_slice_size, num_buckets));
  if (should_use_multithread_buckets) {
    TORCH_CHECK(num_buckets <= kMaxGridSize[0], "topk: too many buckets")
    dim3 grid(num_buckets, num_input_slices, 1);
    const uint k_num_threads = 64;
    dim3 block(k_num_threads);
    BlockTopK<T, IndexType, Dim, J, k_num_threads>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            input,
            input_slice_size,
            k,
            largest,
            interleaved,
            num_input_slices,
            input_within_slice_stride,
            top_k,
            top_k_within_slice_stride,
            indices,
            indices_within_slice_stride);
  } else {
    // We use one thread per bucket and group them into one warp per block.
    IndexType grid_x = at::ceil_div((int64_t)num_buckets, (int64_t)warp_size);
    TORCH_CHECK(grid_x <= kMaxGridSize[0], "topk: too many buckets")
    dim3 grid(grid_x, num_input_slices, 1);
    dim3 block(warp_size);
    ThreadTopK<T, IndexType, Dim, J>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            input,
            input_slice_size,
            k,
            largest,
            interleaved,
            num_input_slices,
            input_within_slice_stride,
            top_k,
            top_k_within_slice_stride,
            indices,
            indices_within_slice_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

void PriorityQueueTopK(
    const Tensor& self,
    int64_t k,
    int64_t j, // int64_t k_mult,
    int64_t dim,
    bool largest,
    bool interleaved,
    c10::optional<bool> multithread_buckets,
    const Tensor& values,
    const Tensor& indices) {
  TensorArg input_arg{self, "xs", 1}, top_k_arg{values, "values_output", 2},
      indices_arg{indices, "indices_output", 3};
  checkAllSameGPU(__func__, {input_arg, top_k_arg, indices_arg});

  int num_dims = self.dim();
  num_dims = num_dims == 0 ? 1 : num_dims;
  TORCH_CHECK(
      num_dims <= MAX_DIMS, "topk: input tensor has too many dimensions");
  int64_t slice_size = self.dim() == 0 ? 1 : self.size(dim);

  // k=0 is valid but no work needs to be done, so exit early to simplify the
  // rest of the implementation.
  if (k == 0)
    return;
  TORCH_CHECK(k % j == 0, "topk: j must divide k");

  auto input = self.contiguous();
  // static_cast is required to ensure that the correct type (INDEX_T)
  // is provided to the kernel for the arguments.
#define RUN_J(INDEX_T, DIM, J)                                      \
  LaunchKernel<scalar_t, INDEX_T, DIM, J>(                          \
      input_info,                                                   \
      static_cast<INDEX_T>(slice_size),                             \
      static_cast<INDEX_T>(k),                                      \
      largest,                                                      \
      interleaved,                                                  \
      multithread_buckets,                                          \
      static_cast<INDEX_T>(num_input_slices),                       \
      static_cast<INDEX_T>(input_info.strides[collapse_input_dim]), \
      top_k_info,                                                   \
      static_cast<INDEX_T>(top_k_info.strides[collapse_top_k_dim]), \
      indices_info,                                                 \
      static_cast<INDEX_T>(indices_info.strides[collapse_indices_dim]));

  // J has to be a statically known template parameter so that the priorty queue
  // can be kept in registers rather than local memory.
#define RUN_K(INDEX_T, DIM)                       \
  if (j == 1) {                                   \
    RUN_J(INDEX_T, DIM, 1);                       \
  } else if (j == 2) {                            \
    RUN_J(INDEX_T, DIM, 2);                       \
  } else if (j == 3) {                            \
    RUN_J(INDEX_T, DIM, 3);                       \
  } else if (j == 4) {                            \
    RUN_J(INDEX_T, DIM, 4);                       \
  } else {                                        \
    TORCH_CHECK(false, "topk: j must 0 < j < 5"); \
  }

#define RUN_DIM(INDEX_T)      \
  if (all_dims == 1) {        \
    RUN_K(INDEX_T, 1);        \
  } else if (all_dims == 2) { \
    RUN_K(INDEX_T, 2);        \
  } else if (all_dims == 3) { \
    RUN_K(INDEX_T, 3);        \
  } else {                    \
    RUN_K(INDEX_T, -1);       \
  }

#define RUN_T(INDEX_T)                                                         \
  AT_DISPATCH_ALL_TYPES_AND2(                                                  \
      at::ScalarType::Half,                                                    \
      at::ScalarType::BFloat16,                                                \
      input.scalar_type(),                                                     \
      "topk_out_cuda",                                                         \
      [&] {                                                                    \
        at::cuda::detail::TensorInfo<const scalar_t, INDEX_T> input_info =     \
            at::cuda::detail::getTensorInfo<const scalar_t, INDEX_T>(input);   \
        at::cuda::detail::TensorInfo<scalar_t, INDEX_T> top_k_info =           \
            at::cuda::detail::getTensorInfo<scalar_t, INDEX_T>(values);        \
        at::cuda::detail::TensorInfo<int64_t, INDEX_T> indices_info =          \
            at::cuda::detail::getTensorInfo<int64_t, INDEX_T>(indices);        \
        /* tensorInfoLegacyIfScalar*/                                          \
        if (!input.dim()) {                                                    \
          input_info.dims = 1;                                                 \
          input_info.sizes[0] = 1;                                             \
          input_info.strides[0] = 1;                                           \
          top_k_info.dims = 1;                                                 \
          top_k_info.sizes[0] = 1;                                             \
          top_k_info.strides[0] = 1;                                           \
          indices_info.dims = 1;                                               \
          indices_info.sizes[0] = 1;                                           \
          indices_info.strides[0] = 1;                                         \
        }                                                                      \
        /* We use these structures solely to find the offset to */             \
        /* each slice we are operating on */                                   \
        input_info.sizes[dim] = 1;                                             \
        top_k_info.sizes[dim] = 1;                                             \
        indices_info.sizes[dim] = 1;                                           \
        /* stash the stride of dim because it can be accidentally collapsed */ \
        auto stride_top_k = top_k_info.strides[dim];                           \
        auto stride_indices = indices_info.strides[dim];                       \
        /* Collapse all other dims */                                          \
        int collapse_input_dim = input_info.collapseDims(dim);                 \
        int collapse_top_k_dim = top_k_info.collapseDims(dim);                 \
        int collapse_indices_dim = indices_info.collapseDims(dim);             \
        /* restore stride in case it was collapsed */                          \
        top_k_info.strides[collapse_top_k_dim] = stride_top_k;                 \
        indices_info.strides[collapse_indices_dim] = stride_indices;           \
        int64_t num_input_slices = 1;                                          \
        for (int i = 0; i < input_info.dims; ++i) {                            \
          num_input_slices *= input_info.sizes[i];                             \
        }                                                                      \
                                                                               \
        /* This is used as a template parameter to calculate indices. */       \
        /* We only specialize it if all collapsed dim sizes are the */         \
        /* same; otherwise, we use -1 which is the specialization */           \
        /* parameter for arbitrary dimensions */                               \
        int all_dims = input_info.dims;                                        \
        if (top_k_info.dims != all_dims || indices_info.dims != all_dims) {    \
          all_dims = -1;                                                       \
        }                                                                      \
                                                                               \
        RUN_DIM(INDEX_T);                                                      \
      });

  // the below is safe with 0-dimensional tensors because it is based on
  // TensorInfo which implicitly expands to 1-dimensional.
  if (input.numel() > 0) {
    // Based on required index size, run the algorithm with the
    // appropriate index type
    if (at::cuda::detail::canUse32BitIndexMath(input) &&
        at::cuda::detail::canUse32BitIndexMath(values) &&
        at::cuda::detail::canUse32BitIndexMath(indices)) {
      RUN_T(uint32_t);
    } else {
      RUN_T(uint64_t);
    }
  }
#undef RUN_T
#undef RUN_DIM
#undef RUN_K
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("PriorityQueueTopK", &PriorityQueueTopK);
}

} // namespace approx_topk
