# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

"""Run an experiment to measure runtime."""

import dataclasses
import itertools
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cupy as cp
import torch
import tqdm
from pylibraft.common import Handle
from pylibraft.matrix import select_k as raft_select_k
from torch import Tensor

from approx_topk import Topk, priority_queue
from approx_topk.experimental import bucket_argmax


@dataclass(frozen=True, kw_only=True)
class Experiment(ABC):
    cuda_graphs: bool
    batch_size: int
    topk_size: int
    k: int
    dtype: torch.dtype
    n_outer: int
    n_inner: int
    n_warmup: int = 16
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
        # Have to do this rather than use dataclasses.asdict() because
        # priority_queue.topk is a CustomOpDef and can't be deepcopied, which is
        # required for asdict() to work. Of course, now have to be careful not to mutate
        # any of the values.
        d = {k: v for k, v in self.__dict__.items()}
        method = d.pop("method")
        if method == priority_queue.topk:
            d["method"] = "approx_topk.priority_queue.topk"
        else:
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
        torch.cuda._sleep(10_000_000)
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
        d = dataclasses.asdict(self)
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
        values = cp.empty((self.n_inner, self.batch_size, self.k), dtype=cp.float32)
        indices = cp.empty((self.n_inner, self.batch_size, self.k), dtype=cp.int64)

        def pre_fn(stream: cp.cuda.Stream):
            with stream:
                xs[:, :, :] = cp.random.standard_normal(xs.shape)

        def fn(stream: cp.cuda.Stream):
            handle = Handle(stream=stream.ptr)
            for i in range(self.n_inner):
                j = i % xs.shape[0]
                raft_select_k(
                    xs[j],
                    select_min=False,
                    distances=values[i],
                    indices=indices[i],
                    handle=handle,
                )

        return benchmark_cupy(fn, self.n_outer, pre_fn, self.cuda_graphs, self.n_warmup)


def benchmark_cupy(
    fn: Callable[[cp.cuda.Stream], Any],
    steps: int,
    pre_fn: Callable[[cp.cuda.Stream], Any] = lambda: None,
    cuda_graphs: bool = True,
    warmup_steps: Optional[int] = None,
) -> list[float]:
    main_stream = cp.cuda.Stream(non_blocking=True)
    # We need to set the pytorch stream so _sleep() below uses the correct stream.
    prev_torch_stream = torch.cuda.current_stream()
    torch.cuda.set_stream(torch.cuda.ExternalStream(main_stream.ptr))

    if cuda_graphs:
        main_stream.synchronize()
        warmup_stream = cp.cuda.Stream()
        pre_fn(warmup_stream)
        fn(warmup_stream)
        warmup_stream.synchronize()

        capture_stream = cp.cuda.Stream()
        pre_fn(capture_stream)
        capture_stream.begin_capture()
        fn(capture_stream)
        graph = capture_stream.end_capture()
        capture_stream.synchronize()
        fn = graph.launch

    for _ in range(warmup_steps):
        fn(main_stream)

    events = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(steps)]
    for start, end in events:
        pre_fn(main_stream)
        torch.cuda._sleep(10_000_000)
        start.record(stream=main_stream)
        fn(main_stream)
        end.record(stream=main_stream)
    main_stream.synchronize()
    times = [cp.cuda.get_elapsed_time(start, end) / 1e3 for start, end in events]

    torch.cuda.set_stream(prev_torch_stream)

    return times


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


def torch_topk(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    return torch.topk(xs, k, dim, sorted=False)


if __name__ == "__main__":
    ns = [2**n for n in [16, 15, 14, 13, 12]]
    batch_sizes = [128, 32]
    cuda_graph_configs = [
        dict(cuda_graphs=False, n_outer=512, n_inner=1),
        dict(cuda_graphs=True, n_outer=16, n_inner=128),
    ]

    experiments: list[Experiment] = []
    for n, batch_size, cuda_graph_config in itertools.product(
        ns, batch_sizes, cuda_graph_configs
    ):
        for k in [64, n // 8]:
            experiments += (
                [
                    RaftExperiment(
                        batch_size=batch_size,
                        topk_size=n,
                        k=k,
                        dtype=torch.float32,
                        **cuda_graph_config,
                    )
                ]
                # Do not run RAFT with cuda graphs enabled if k > 256. For k > 256 it
                # uses the radix select kernel which does not support cuda graphs.
                # For k <=256 it uses the warp select kernel which does work.
                if k <= 256 or not cuda_graph_config["cuda_graphs"]
                else []
            )
            experiments += [
                PyTorchExperiment(
                    method=method,
                    args=args,
                    compile="default"
                    if method is bucket_argmax.topk_torch
                    or method is priority_queue.topk
                    else None,
                    batch_size=batch_size,
                    topk_size=n,
                    k=k,
                    dtype=torch.float32,
                    **cuda_graph_config,
                )
                for km in [1]
                for mtb in [False, True]
                for method, args in [
                    (torch_topk, {}),
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
