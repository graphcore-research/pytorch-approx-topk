from typing import Protocol

from torch import Tensor


class ExactTopK(Protocol):
    @staticmethod
    def __call__(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]: ...
