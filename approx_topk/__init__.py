# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

from typing import Protocol

from torch import Tensor


class Topk(Protocol):
    @staticmethod
    def __call__(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]: ...
