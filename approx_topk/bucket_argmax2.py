"""A technique with k=1 for each bucket."""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton import cdiv

from approx_topk.bucket_argmax import _batch_contiguous, _min_value

K_PER_BUCKET = 2


@triton.jit
def _topk_triton2_kernel__parallel_bk(
    xs_ptr,
    values_out_ptr,
    indices_out_ptr,
    b: int,
    k: int,
    n: int,
    n_chunk: int,
    xs_stride: int,
    K_PER_BUCKET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    n_buckets = k // K_PER_BUCKET

    # block_size -> n_jobs
    # "job" - one bucket top-2
    pidx = tl.program_id(axis=0).to(tl.int64)
    job_idx = BLOCK_SIZE * pidx + tl.arange(0, BLOCK_SIZE)

    seq_idx, bucket_idx = job_idx // n_buckets, job_idx % n_buckets
    xs_ptr += seq_idx * xs_stride
    if INTERLEAVED:
        bucket_stride, i_stride = 1, n_buckets
    else:
        bucket_stride, i_stride = n_chunk, 1

    # Max value
    mask = (seq_idx < b) & (bucket_idx * bucket_stride < n)
    max_value1 = tl.load(
        xs_ptr + bucket_idx * bucket_stride, mask=mask, other=PAD_VALUE
    )
    max_i1 = tl.zeros((BLOCK_SIZE,), tl.int64)

    # Second max value
    max_value2 = tl.full((BLOCK_SIZE,), PAD_VALUE, max_value1.dtype)
    max_i2 = tl.full((BLOCK_SIZE,), -1, tl.int64)

    for i in tl.range(1, n_chunk):
        mask = (seq_idx < b) & (bucket_idx * bucket_stride + i * i_stride < n)
        block = tl.load(
            xs_ptr + bucket_idx * bucket_stride + i * i_stride,
            mask=mask,
            other=PAD_VALUE,
        )

        is_max1 = max_value1 < block
        is_max2 = ~is_max1 & (max_value2 < block)

        # If > max1, update max2 to old max1, then update max1
        mask1 = mask & is_max1
        max_value2 = tl.where(mask1, max_value1, max_value2)
        max_i2 = tl.where(mask1, max_i1, max_i2)
        max_value1 = tl.where(mask1, block, max_value1)
        max_i1 = tl.where(mask1, i, max_i1)

        # If <= max1 & > max2, just update max2
        mask2 = mask & is_max2
        max_value2 = tl.where(mask2, block, max_value2)
        max_i2 = tl.where(mask2, i, max_i2)

    max_index1 = bucket_idx * bucket_stride + max_i1 * i_stride
    max_index2 = bucket_idx * bucket_stride + max_i2 * i_stride
    tl.store(
        values_out_ptr + seq_idx * k + bucket_idx * K_PER_BUCKET + 0,
        max_value1,
        mask=(seq_idx < b),
    )
    tl.store(
        indices_out_ptr + seq_idx * k + bucket_idx * K_PER_BUCKET + 0,
        max_index1,
        mask=(seq_idx < b),
    )
    tl.store(
        values_out_ptr + seq_idx * k + bucket_idx * K_PER_BUCKET + 1,
        max_value2,
        mask=(seq_idx < b),
    )
    tl.store(
        indices_out_ptr + seq_idx * k + bucket_idx * K_PER_BUCKET + 1,
        max_index2,
        mask=(seq_idx < b),
    )


def topk_triton2(
    xs: Tensor,
    k: int,
    dim: int,
    interleaved: bool,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    if k % 2 != 0:
        raise NotImplementedError

    dim = dim % xs.ndim  # convert negative dims to positive
    if dim != xs.ndim - 1:
        values, indices = topk_triton2(
            xs.movedim(dim, -1),
            k=k,
            dim=-1,
            block_size=block_size,
            interleaved=interleaved,
        )
        return values.movedim(-1, dim), indices.movedim(-1, dim)

    xs = _batch_contiguous(xs)
    n = xs.shape[-1]
    b = xs.nelement() // n
    n_buckets = k // K_PER_BUCKET
    values = torch.empty(xs.shape[:-1] + (k,), device="cuda", dtype=xs.dtype)
    indices = torch.empty(values.shape, device="cuda", dtype=torch.int64)
    _topk_triton2_kernel__parallel_bk[(cdiv(b * n_buckets, block_size),)](
        xs_ptr=xs,
        values_out_ptr=values,
        indices_out_ptr=indices,
        b=b,
        k=k,
        n=n,
        n_chunk=cdiv(n, n_buckets),
        xs_stride=xs.stride(-2),
        K_PER_BUCKET=K_PER_BUCKET,
        BLOCK_SIZE=block_size,
        PAD_VALUE=_min_value(xs.dtype),
        INTERLEAVED=interleaved,
    )
    return values, indices
