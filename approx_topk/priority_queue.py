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
    j: int = 1,
    k_mult: int = 1,
    interleaved: bool = True,
    multithread_buckets: bool | None = False,
) -> tuple[Tensor, Tensor]:
    """Computes an approximate top-k.

    The algorithm is as follows:
        - Split the input into b = k_mult * k/j buckets
          (either be contiguous or interleaved, see the `interleaved` parameter)
        - Take the top j values per bucket
        - If b*j>k: take a second top-k of the results

    :param j: number of items to take per bucket.
        The implementation currently requires j in {1,2,4}, and k mod j = 0.

    :param k_mult: controls the number of buckets that the input is split into.
        You might want to increase the number of buckets in order to to
        (a) increase the recall, or
        (b) increase the amount of parallelism, if k/j is small.
        However, if k_mult > 1 then a second top-k stage is required, which
        can slow things down for larger k.

    :param multithread_buckets: If True, use a warp of threads to process each bucket.
                                If False, use a single thread for each bucket.
                                If None, decide using a heuristic
                                (see shouldUseMultithreadBuckets in the kernel)
    """
    if dim < 0:
        dim = xs.ndim + dim
    if j < k and dim != xs.ndim - 1:
        raise NotImplementedError("This topk only currently works on the last dim")

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
