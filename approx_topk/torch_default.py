"""Wrapper around torch.topk() to match our topk Protocol."""

import torch
from torch import Tensor


def topk(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    return torch.topk(xs, k, dim, sorted=False)
