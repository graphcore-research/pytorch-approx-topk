from functools import cache

import torch
from torch import Tensor
from torch.utils import cpp_extension


def topk(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    assert xs.is_cuda
    assert k <= xs.shape[dim]
    if dim < 0:
        dim = xs.ndim + dim

    impl = _get_impl()

    output_shape = list(xs.shape)
    output_shape[dim] = k
    values = torch.empty(output_shape, device=xs.device)
    # FIXME: Output can be int32 or int64 depending on whether kernel decides if it can
    #        do 32 bit indexing.
    # Oscar: Unsure how to deal with this properly yet.
    indices = torch.empty(output_shape, dtype=torch.int64, device=xs.device)

    largest = True
    impl.topk(xs, k, dim, largest, values, indices)
    return values, indices


@cache
def _get_impl():
    # TODO: Work out how to package the C code properly.
    return cpp_extension.load(
        name="radix_select_topk",
        sources="approx_topk/exact_methods/radix_select_topk.cu",
    )
