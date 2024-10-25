# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

"""Wrapper around torch.topk() to match our topk Protocol,
as well as the reference bucketed top-k implementation using torch.topk
"""

import torch
from torch import Tensor

from approx_topk.autobucket import bucket


def topk(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    return torch.topk(xs, k, dim, sorted=False)


def bucket_topk(
    xs: Tensor,
    k: int,
    dim: int,
    k_mult: int,
    k_per_bucket: int,
    interleaved: bool,
) -> tuple[Tensor, Tensor]:
    return bucket(
        topk,
        k_mult=k_mult,
        k_per_bucket=k_per_bucket,
        interleaved=interleaved,
    )(xs, k, dim)
