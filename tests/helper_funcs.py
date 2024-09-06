import torch
from torch import Tensor


def assert_close_up_to_permutation(x: Tensor, y: Tensor, dim: int = -1) -> None:
    torch.testing.assert_close(x.sort(dim=dim)[0], y.sort(dim=dim)[0])
