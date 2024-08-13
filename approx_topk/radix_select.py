from functools import cache

import torch
from torch import Tensor
from torch.utils import cpp_extension


def topk(xs: Tensor, k: int, dim: int, j: int | None = None) -> tuple[Tensor, Tensor]:
    """Computes a top-k. This is exact if j is None (default), or otherwise approximate.

    :param j: if not None, xs is split into k/j buckets, and then the top-j is computed
              for each bucket.
    """
    if dim < 0:
        dim = xs.ndim + dim

    impl = _get_impl()

    output_shape = list(xs.shape)
    output_shape[dim] = k
    values = torch.empty(output_shape, device=xs.device, dtype=xs.dtype)
    # FIXME: Output can be int32 or int64 depending on whether kernel decides if it can
    #        do 32 bit indexing.
    # Oscar: Unsure how to deal with this properly yet.
    indices = torch.empty(output_shape, dtype=torch.int64, device=xs.device)

    largest = True
    if j is None:
        j = k
    impl.topk(xs, k, j, dim, largest, values, indices)
    return values, indices


@cache
def _get_impl():
    # TODO: Work out how to package the C code properly.
    return cpp_extension.load(
        name="radix_select_topk",
        sources="approx_topk/radix_select.cu",
    )
