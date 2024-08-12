"""A technique with k=1 for each bucket."""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton import cdiv

from . import autobucket


def _argmax_topk(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    assert k == 1
    return torch.max(xs, dim=dim, keepdim=True)


topk_autobucket = autobucket.bucket(_argmax_topk, k_per_bucket=1)
topk_autobucket.__name__ = "topk_autobucket"
topk_autobucket.__module__ = _argmax_topk.__module__


def topk_torch(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    dim = dim % xs.ndim  # convert negative dims to positive
    xs_bucketed = torch.nn.functional.pad(
        xs,
        pad=(0, 0) * len(xs.shape[dim + 1 :]) + (0, -xs.shape[dim] % k),
        value=torch.finfo(xs.dtype).min,
    ).unflatten(dim, (k, -1))
    max_ = xs_bucketed.max(dim=dim + 1)
    indices = max_.indices.add_(
        xs_bucketed.shape[dim + 1]
        * torch.arange(0, k, dtype=max_.indices.dtype, device=xs.device)
    )
    return (max_.values, indices)


@triton.jit
def _topk_triton_kernel(
    xs_ptr,
    values_out_ptr,
    indices_out_ptr,
    b: int,
    k: int,
    n: int,
    n_chunk: int,
    xs_stride: int,
    BLOCK_SIZE: tl.constexpr,
):
    pidx = tl.program_id(axis=0).to(tl.int64)
    bk_idx = BLOCK_SIZE * pidx + tl.arange(0, BLOCK_SIZE)
    b_idx, k_idx = bk_idx // k, bk_idx % k
    xs_ptr += b_idx * xs_stride + k_idx * n_chunk

    max_value = tl.load(xs_ptr, mask=(b_idx < b))
    max_index = tl.zeros((BLOCK_SIZE,), tl.int64)
    for i in tl.range(1, n_chunk):
        mask = (b_idx < b) & (k_idx * n_chunk + i < n)
        block = tl.load(xs_ptr + i, mask=mask)
        mask &= max_value < block
        max_value = tl.where(mask, block, max_value)
        max_index = tl.where(mask, i, max_index)

    max_index += k_idx * n_chunk
    tl.store(values_out_ptr + b_idx * k + k_idx, max_value, mask=(b_idx < b))
    tl.store(indices_out_ptr + b_idx * k + k_idx, max_index, mask=(b_idx < b))


def topk_triton(xs: Tensor, k: int, dim: int, block_size: int) -> tuple[Tensor, Tensor]:
    dim = dim % xs.ndim  # convert negative dims to positive
    if dim != xs.ndim - 1:
        values, indices = topk_triton(
            xs.movedim(dim, -1), k=k, dim=-1, block_size=block_size
        )
        return values.movedim(-1, dim), indices.movedim(-1, dim)

    xs = xs.contiguous()
    values = torch.empty(xs.shape[:-1] + (k,), device="cuda", dtype=xs.dtype)
    indices = torch.empty(values.shape, device="cuda", dtype=torch.int64)

    n = xs.shape[-1]
    b = xs.nelement() // n
    _topk_triton_kernel[(cdiv(b * k, block_size),)](
        xs_ptr=xs,
        values_out_ptr=values,
        indices_out_ptr=indices,
        b=b,
        k=k,
        n=n,
        n_chunk=cdiv(n, k),
        xs_stride=xs.stride(-2),
        BLOCK_SIZE=block_size,
    )
    return values, indices
