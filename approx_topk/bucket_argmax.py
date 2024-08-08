"""A technique with k=1 for each bucket."""

import torch
from torch import Tensor

from . import autobucket

def _argmax_topk(xs: Tensor, k: int, dim: int):
    assert k == 1
    return torch.max(xs, dim=dim, keepdim=True)

topk = autobucket.bucket(_argmax_topk, k_per_bucket=1)
topk.__name__ = "topk"
topk.__module__ = _argmax_topk.__module__
