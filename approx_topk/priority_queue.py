# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

import math

import torch
from torch import Tensor

from approx_topk.cuda_extensions import load_cuda_extension


@torch.library.custom_op("approx_topk::topk", mutates_args=())
def topk(
    xs: Tensor,
    k: int,
    dim: int,
    j: int | None = None,
    k_mult: int = 1,
    interleaved: bool = True,
    multithread_buckets: bool | None = False,
) -> tuple[Tensor, Tensor]:
    """Computes a top-k. This is exact if j is None (default), or otherwise approximate.

    :param j: if not None, xs is split into k/j buckets, and then the top-j is computed
              for each bucket.
    :param multithread_buckets: If True, use a warp of threads to process each bucket.
                                If False, use a single thread for each bucket.
                                If None, decide using a heuristic
                                (see shouldUseMultithreadBuckets in the kernel)
    """
    if dim < 0:
        dim = xs.ndim + dim
    if j is None:
        j = k
    if j < k and dim != xs.ndim - 1:
        raise NotImplementedError("Heap topk only currently works on the last dim")

    if k_mult == 1:
        k0 = k
    elif k_mult > 1:
        k0 = k * k_mult
    else:
        raise ValueError(f"k_mult must be >=1, was {k_mult}")

    impl = load_cuda_extension("priority_queue.cu", compile_mode="optimize")

    output_shape = _get_output_shape(xs, k0, dim)
    stage_1_values = torch.empty(output_shape, device=xs.device, dtype=xs.dtype)
    # FIXME: Output can be int32 or int64 depending on whether kernel decides if it can
    #        do 32 bit indexing.
    # Oscar: Unsure how to deal with this properly yet.
    stage_1_indices = torch.empty(output_shape, dtype=torch.int64, device=xs.device)

    largest = True
    impl.priority_queue_topk(
        xs,
        k0,
        j,
        dim,
        largest,
        interleaved,
        multithread_buckets,
        stage_1_values,
        stage_1_indices,
    )

    if k0 == k:
        return stage_1_values, stage_1_indices
    else:
        stage_2_values, stage_2_indices = torch.topk(
            stage_1_values, k, dim, largest, sorted=False
        )
        return stage_2_values, stage_1_indices.gather(dim, stage_2_indices)


@topk.register_fake
def _(
    xs: Tensor,
    k: int,
    dim: int,
    j: int | None = None,
    k_mult: int = 1,
    interleaved: bool = True,
    multithread_buckets: bool | None = False,
) -> tuple[Tensor, Tensor]:
    output_shape = _get_output_shape(xs, k, dim)
    values = torch.empty(output_shape, device=xs.device, dtype=xs.dtype)
    indices = torch.empty(output_shape, dtype=torch.int64, device=xs.device)
    return values, indices


def _get_output_shape(xs: Tensor, k: int, dim: int) -> list[int]:
    output_shape = list(xs.shape)
    output_shape[dim] = k
    return output_shape
