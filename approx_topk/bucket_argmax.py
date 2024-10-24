# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

"""A technique with k=1 for each bucket."""

from typing import Any, Literal

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton import cdiv


def _min_value(dtype: torch.dtype) -> Any:
    return torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min


def topk_torch(
    xs: Tensor, k: int, dim: int, interleaved: bool
) -> tuple[Tensor, Tensor]:
    dim = dim % xs.ndim  # convert negative dims to positive
    pad_value = (
        torch.finfo(xs.dtype).min
        if xs.dtype.is_floating_point
        else torch.iinfo(xs.dtype).min
    )
    xs_pad = torch.nn.functional.pad(
        xs,
        pad=(0, 0) * len(xs.shape[dim + 1 :]) + (0, -xs.shape[dim] % k),
        value=pad_value,
    )
    krange = torch.arange(0, k, dtype=torch.long, device=xs.device).view(
        (k,) + (1,) * (xs.ndim - dim - 1)
    )
    if interleaved:
        max_ = xs_pad.unflatten(dim, (-1, k)).max(dim=dim)
        max_.indices.mul_(k).add_(krange)
    else:
        max_ = xs_pad.unflatten(dim, (k, -1)).max(dim=dim + 1)
        n_per_bucket = xs_pad.shape[dim] // k
        max_.indices.add_(n_per_bucket * krange)
    return (max_.values, max_.indices)


def _batch_contiguous(x: Tensor) -> Tensor:
    """Ensure `x` is contiguous for all dimensions except the last pair.

    E.g. for a tensor of shape (2, 3, 4):

       (*,  *, 2) is not batch-contiguous
       (12, 4, 1) is batch-contiguous (and contiguous)
       (15, 5, 1) is batch-contiguous (but not contiguous)
       (13, 4, 1) is not batch-contiguous
       ( 4, 8, 1) is not batch-contiguous
    """
    stride = x.stride()
    if stride[-1] != 1:
        return x.contiguous()
    if len(stride) >= 2:
        if stride[-2] < x.shape[-1]:
            return x.contiguous()
        expected = stride[-2]
        for i in range(len(stride) - 2, -1, -1):
            if stride[i] != expected:
                return x.contiguous()
            expected *= x.shape[i]
    return x


@triton.jit
def _topk_triton_kernel__parallel_bk(
    xs_ptr,
    values_out_ptr,
    indices_out_ptr,
    b: int,
    k: int,
    n: int,
    n_chunk: int,
    xs_stride: int,
    BLOCK_SIZE: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    pidx = tl.program_id(axis=0).to(tl.int64)
    bk_idx = BLOCK_SIZE * pidx + tl.arange(0, BLOCK_SIZE)
    b_idx, k_idx = bk_idx // k, bk_idx % k
    xs_ptr += b_idx * xs_stride
    if INTERLEAVED:
        k_stride, i_stride = 1, k
    else:
        k_stride, i_stride = n_chunk, 1

    mask = (b_idx < b) & (k_idx * k_stride < n)
    max_value = tl.load(xs_ptr + k_idx * k_stride, mask=mask, other=PAD_VALUE)
    max_i = tl.zeros((BLOCK_SIZE,), tl.int64)
    for i in tl.range(1, n_chunk):
        mask = (b_idx < b) & (k_idx * k_stride + i * i_stride < n)
        block = tl.load(xs_ptr + k_idx * k_stride + i * i_stride, mask=mask, other=PAD_VALUE)
        mask &= max_value < block
        max_value = tl.where(mask, block, max_value)
        max_i = tl.where(mask, i, max_i)

    max_index = k_idx * k_stride + max_i * i_stride
    tl.store(values_out_ptr + b_idx * k + k_idx, max_value, mask=(b_idx < b))
    tl.store(indices_out_ptr + b_idx * k + k_idx, max_index, mask=(b_idx < b))


@triton.jit
def _topk_triton_kernel__parallel_bkn(
    xs_ptr,
    values_out_ptr,
    indices_out_ptr,
    xs_stride: int,
    n_stride: int,
    b: int,
    k: int,
    n: int,
    BLOCK_BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    idx = tl.program_id(axis=0) * BLOCK_BK + tl.arange(0, BLOCK_BK)
    b_idx, k_idx = idx // k, idx % k
    if INTERLEAVED:
        k_stride, i_stride = 1, k
    else:
        k_stride, i_stride = n_stride, 1

    ni = tl.arange(0, BLOCK_N)
    n_idx = k_idx[:, None] * k_stride + ni[None, :] * i_stride
    data = tl.load(
        xs_ptr + b_idx[:, None] * xs_stride + n_idx,
        mask=(b_idx[:, None] < b) & (n_idx < n) & (ni < n_stride),
        other=PAD_VALUE,
    )
    max_value, max_i = tl.max(data, axis=1, return_indices=True)
    max_index = k_idx * k_stride + max_i * i_stride
    tl.store(values_out_ptr + b_idx * k + k_idx, max_value, mask=(b_idx < b))
    tl.store(indices_out_ptr + b_idx * k + k_idx, max_index, mask=(b_idx < b))


def topk_triton(
    xs: Tensor,
    k: int,
    dim: int,
    interleaved: bool,
    block_size: int,
    kernel: Literal["bk", "bkn"],
) -> tuple[Tensor, Tensor]:
    dim = dim % xs.ndim  # convert negative dims to positive
    if dim != xs.ndim - 1:
        values, indices = topk_triton(
            xs.movedim(dim, -1),
            k=k,
            dim=-1,
            kernel=kernel,
            block_size=block_size,
            interleaved=interleaved,
        )
        return values.movedim(-1, dim), indices.movedim(-1, dim)

    xs = _batch_contiguous(xs)
    n = xs.shape[-1]
    b = xs.nelement() // n
    values = torch.empty(xs.shape[:-1] + (k,), device="cuda", dtype=xs.dtype)
    indices = torch.empty(values.shape, device="cuda", dtype=torch.int64)
    if kernel == "bk":
        _topk_triton_kernel__parallel_bk[(cdiv(b * k, block_size),)](
            xs_ptr=xs,
            values_out_ptr=values,
            indices_out_ptr=indices,
            b=b,
            k=k,
            n=n,
            n_chunk=cdiv(n, k),
            xs_stride=xs.stride(-2),
            BLOCK_SIZE=block_size,
            PAD_VALUE=_min_value(xs.dtype),
            INTERLEAVED=interleaved,
        )
    elif kernel == "bkn":
        n_stride = cdiv(n, k)
        _topk_triton_kernel__parallel_bkn[(cdiv(b * k, block_size),)](
            xs_ptr=xs,
            values_out_ptr=values,
            indices_out_ptr=indices,
            xs_stride=xs.stride(-2),
            n_stride=n_stride,
            b=b,
            k=k,
            n=n,
            BLOCK_BK=block_size,
            BLOCK_N=triton.next_power_of_2(n_stride),
            PAD_VALUE=_min_value(xs.dtype),
            INTERLEAVED=interleaved,
        )
    else:
        raise ValueError(f"Unknown kernel {kernel!r}")
    return values, indices
