"""Run an experiment to measure runtime."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import tqdm
from torch.cuda import Event

from approx_topk import TopK, bucket_argmax
from approx_topk.exact_methods import radix_select, torch_default


@dataclass
class Experiment:
    method: TopK
    args: dict[str, Any]
    compile: Optional[str]
    batch_size: int
    topk_size: int
    k: int
    dtype: torch.dtype
    n_outer: int = 50
    n_inner: int = 20
    n_warmup: int = 200

    def save(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        method = d.pop("method")
        d["method"] = f"{method.__module__}.{method.__name__}"
        d["dtype"] = str(d.pop("dtype")).replace("torch.", "")
        d.update(d.pop("args"))
        return d



def run(config: Experiment) -> list[float]:
    durations: list[float] = []

    method = lambda xs: config.method(xs, k=config.k, dim=1, **config.args)
    if config.compile:
        torch.compiler.reset()
        method = torch.compile(method, mode=config.compile)

    xs = torch.randn(config.batch_size, config.topk_size, dtype=config.dtype, device="cuda")
    for _ in range(config.n_warmup):
        method(xs)

    for _ in range(config.n_outer):
        xs = torch.randn(config.batch_size, config.topk_size, dtype=config.dtype, device="cuda")
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)
        start_event.record()
        for _ in range(config.n_inner):
            method(xs)
        end_event.record()
        end_event.synchronize()
        durations.append(start_event.elapsed_time(end_event) / 1e3)

    return durations


def sweep(configs: list[Experiment], out: Path) -> None:
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for config in tqdm.tqdm(configs):
            print(json.dumps(dict(**config.save(), duration=run(config))), file=f, flush=True)


if __name__ == "__main__":
    sweep([Experiment(
            method=method,
            args={},
            compile=compile,
            batch_size=32,
            topk_size=topk_size,
            k=topk_size//8,
            dtype=torch.float32,
        )
        for compile in [None, "default", "reduce-overhead", "max-autotune"]
        for method in [radix_select.topk, torch_default.topk, bucket_argmax.topk]
        # for topk_size in [2**n for n in range(10, 16+1, 1)]
        for topk_size in range(1024, 16384 + 1, 1024)
    ], Path("results/measure_speed.jsonl"))
