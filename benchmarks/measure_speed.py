# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

"""Run an experiment to measure runtime."""

import itertools
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cupy as cp
import torch
import tqdm
from pylibraft.common import Handle
from pylibraft.matrix import select_k as raft_select_k
from torch import Tensor

from approx_topk import Topk, bucket_argmax, priority_queue, torch_default


@dataclass(frozen=True)
class Experiment(ABC):
    cuda_graphs: bool
    batch_size: int
    topk_size: int
    k: int
    dtype: torch.dtype
    n_warmup: int = 16
    n_outer: int = 16
    n_inner: int = 128
    n_inner_inputs: Optional[int] = None

    @abstractmethod
    def save(self) -> dict[str, Any]: ...

    @abstractmethod
    def run(self) -> list[float]: ...


@dataclass(frozen=True, kw_only=True)
class PyTorchExperiment(Experiment):
    method: Topk
    args: dict[str, Any]
    compile: Optional[str]

    def save(self) -> dict[str, Any]:
        d = asdict(self)
        method = d.pop("method")
        d["method"] = f"{method.__module__}.{method.__name__}"
        d["dtype"] = str(d.pop("dtype")).replace("torch.", "")
        return d

    def run(self) -> list[float]:
        method = self.method
        if self.compile:
            torch.compiler.reset()
            method = torch.compile(
                method, mode=self.compile, fullgraph=True, dynamic=False
            )

        assert self.n_inner_inputs is None or self.n_inner_inputs <= self.n_inner
        xs = torch.zeros(
            self.n_inner if self.n_inner_inputs is None else self.n_inner_inputs,
            self.batch_size,
            self.topk_size,
            dtype=self.dtype,
            device="cuda",
        )

        def inner_loop_fn() -> list[Tensor]:
            return [
                method(xs[i % xs.shape[0]], k=self.k, dim=1, **self.args)[0]
                for i in range(self.n_inner)
            ]

        return benchmark_pytorch(
            inner_loop_fn,
            steps=self.n_outer,
            pre_fn=lambda: torch.nn.init.normal_(xs),
            cuda_graphs=self.cuda_graphs,
            warmup_steps=self.n_warmup,
        )


def benchmark_pytorch(
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


@dataclass(frozen=True, kw_only=True)
class RaftExperiment(Experiment):
    def __post_init__(self):
        if self.dtype != torch.float32:
            raise ValueError("Raft only supports float32")

    def save(self) -> dict[str, Any]:
        d = asdict(self)
        d["method"] = "raft"
        d["dtype"] = str(d.pop("dtype")).replace("torch.", "")
        d["args"] = {}
        d["compile"] = None
        return d

    def run(self) -> list[float]:
        xs = cp.empty(
            (
                self.n_inner if self.n_inner_inputs is None else self.n_inner_inputs,
                self.batch_size,
                self.topk_size,
            ),
            dtype=cp.float32,
        )
        values = cp.empty(xs.shape[1:], dtype=cp.float32)
        indices = cp.empty(xs.shape[1:], dtype=cp.int64)

        cupy_stream = cp.cuda.Stream()
        raft_handle = Handle(cupy_stream.ptr)

        def pre_fn():
            xs[:, :, :] = cp.random.standard_normal(xs.shape)

        def fn():
            for i in range(self.n_inner):
                raft_select_k(
                    xs[i % xs.shape[0]],
                    self.k,
                    select_min=False,
                    distances=values,
                    indices=indices,
                    handle=raft_handle,
                )

        if self.cuda_graphs:
            cupy_stream.begin_capture()
            fn()
            graph = cupy_stream.end_capture()
            fn = graph.launch

        for _ in range(self.n_outer if self.n_warmup is None else self.n_warmup):
            fn()

        events = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(self.n_outer)]
        for start, end in events:
            pre_fn()
            start.record(stream=cupy_stream)
            fn()
            end.record(stream=cupy_stream)
        cupy_stream.synchronize()
        return [cp.cuda.get_elapsed_time(start, end) / 1e3 for start, end in events]


def sweep(configs: Iterable[Experiment], out: Path) -> None:
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        iterator = tqdm.tqdm(configs)
        for config in iterator:
            iterator.set_description(
                " ".join(f"{k}={v}" for k, v in config.save().items())
            )
            print(
                json.dumps(
                    dict(
                        **config.save(),
                        duration=config.run(),
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
    experiments: list[Experiment] = []
    ns = [2**n for n in [16, 15, 14, 13, 12]]
    batch_sizes = [128, 32]
    for n, batch_size in itertools.product(ns, batch_sizes):
        for k in [64, n // 8]:
            experiments += [
                RaftExperiment(
                    cuda_graphs=True,
                    batch_size=batch_size,
                    topk_size=n,
                    k=k,
                    dtype=torch.float32,
                )
            ]
            experiments += [
                PyTorchExperiment(
                    method=method,
                    args=args,
                    compile="default" if method is bucket_argmax.topk_torch else None,
                    cuda_graphs=True,
                    batch_size=batch_size,
                    topk_size=n,
                    k=k,
                    dtype=torch.float32,
                )
                for km in [1]
                for mtb in [False, True]
                for method, args in [
                    (torch_default.topk, {}),
                    (
                        priority_queue.topk,
                        dict(j=1, k_mult=km, multithread_buckets=mtb),
                    ),
                    (
                        priority_queue.topk,
                        dict(j=2, k_mult=km, multithread_buckets=mtb),
                    ),
                    (
                        priority_queue.topk,
                        dict(j=4, k_mult=km, multithread_buckets=mtb),
                    ),
                    (bucket_argmax.topk_torch, dict(interleaved=True)),
                ]
                if method is priority_queue.topk or (km == 1 and mtb == False)
                if k * km < n
            ]

    sweep(experiments, Path("results/measure_speed.jsonl"))
