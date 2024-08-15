from functools import cache
from pathlib import Path
from typing import Literal

from torch.utils import cpp_extension

CompileMode = Literal["default", "optimize", "debug"]


@cache
def load_cuda_extension(file_name: str, compile_mode: CompileMode):
    if compile_mode == "default":
        nvcc_flags = []
    elif compile_mode == "optimize":
        nvcc_flags = ["-O3"]
    elif compile_mode == "debug":
        nvcc_flags = ["-g", "-G"]
    nvcc_flags += ["--generate-line-info"]

    return cpp_extension.load(
        name="radix_select_topk",
        # TODO: Work out how to package the C code properly.
        sources=[str(Path("approx_topk") / file_name)],
        extra_cuda_cflags=nvcc_flags,
    )
