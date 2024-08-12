"""Run an experiment to measure runtime."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Callable

import torch
import tqdm
from torch import Tensor
from torch.cuda import Event

from approx_topk import TopK, bucket_argmax, radix_select, torch_default


def benchmark_gpu(
    fn: Callable[[], Any],
    steps: int,
    pre_fn: Callable[[], Any] = lambda: None,
    cuda_graphs: bool = True,
    warmup_steps: Optional[int] = None,
) -> list[float]:
    """Generic benchmarking function with {events, CUDAGraphs} for tight measurement.

    pre_fn: outside measurement scope, called before each call to fn()

    warmup_steps: defaults to `steps`

    returns: list of `steps` durations, measured in seconds
    """
    if cuda_graphs:
        # Pre-cudagraphs warmup
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            pre_fn()
            fn()
        torch.cuda.current_stream().wait_stream(stream)

        # Capture
        pre_fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()

        fn = graph.replay

    # Timing warmup
    for _ in range(steps if warmup_steps is None else warmup_steps):
        pre_fn()
        fn()

    # Measure
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(steps)
    ]
    for start, end in events:
        pre_fn()
        start.record()
        fn()
        end.record()
    events[-1][-1].synchronize()
    return [start.elapsed_time(end) / 1e3 for start, end in events]


@dataclass
class Experiment:
    method: TopK
    args: dict[str, Any]
    compile: Optional[str]
    cuda_graphs: bool
    batch_size: int
    topk_size: int
    k: int
    dtype: torch.dtype
    n_warmup: int = 16
    n_outer: int = 16
    n_inner: int = 128
    n_inner_inputs: Optional[int] = None

    def save(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        method = d.pop("method")
        d["method"] = f"{method.__module__}.{method.__name__}"
        d["dtype"] = str(d.pop("dtype")).replace("torch.", "")
        d.update(d.pop("args"))
        return d


def run(config: Experiment) -> list[float]:
    method = config.method
    if config.compile:
        torch.compiler.reset()
        method = torch.compile(
            method, mode=config.compile, fullgraph=True, dynamic=False
        )

    assert config.n_inner_inputs is None or config.n_inner_inputs <= config.n_inner
    xs = torch.zeros(
        config.n_inner if config.n_inner_inputs is None else config.n_inner_inputs,
        config.batch_size,
        config.topk_size,
        dtype=config.dtype,
        device="cuda",
    )

    def inner_loop_fn() -> list[Tensor]:
        return [
            method(xs[i % xs.shape[0]], k=config.k, dim=1, **config.args)[0]
            for i in range(config.n_inner)
        ]

    return benchmark_gpu(
        inner_loop_fn,
        steps=config.n_outer,
        pre_fn=lambda: torch.nn.init.normal_(xs),
        cuda_graphs=config.cuda_graphs,
        warmup_steps=config.n_warmup,
    )


def sweep(configs: list[Experiment], out: Path) -> None:
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for config in tqdm.tqdm(configs):
            print(
                json.dumps(
                    dict(
                        **config.save(),
                        duration=run(config),
                        device=torch.cuda.get_device_name(),
                    )
                ),
                file=f,
                flush=True,
            )


def fake_topk_sum(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    y = xs.sum(dim=dim, keepdim=True)  # ignore k, just run a sum
    return y, torch.zeros(y.shape, dtype=torch.long, device=y.device)


if __name__ == "__main__":
    sweep(
        [
            Experiment(
                method=method,
                args=args,
                compile=compile,
                cuda_graphs=True,
                batch_size=32,
                topk_size=topk_size,
                k=topk_size // 8,
                dtype=torch.float16,
                n_inner_inputs=n_inner_inputs,
            )
            for n_inner_inputs in [
                None,  # large input size (main memory)
                1,  # .. small input size (L2 cache)
            ]
            for compile in [
                None,
                "default",
                "max-autotune-no-cudagraphs",
            ]
            for method, args in [
                (fake_topk_sum, {}),
                (torch_default.topk, {}),
                # (radix_select.topk, {}),
                # (bucket_argmax.topk_autobucket, {}),
                (bucket_argmax.topk_torch, {}),
                (bucket_argmax.topk_triton, dict(block_size=128)),
            ]
            for topk_size in [2**n for n in [12, 14, 16, 18]]
            if not (compile and method is radix_select.topk)
        ],
        Path("results/measure_speed.jsonl"),
    )
