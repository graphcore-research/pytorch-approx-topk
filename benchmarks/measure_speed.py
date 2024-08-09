"""Run an experiment to measure runtime."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import tqdm
from torch import Tensor
from torch.cuda import Event

from approx_topk import TopK, bucket_argmax, radix_select, torch_default


@dataclass
class Experiment:
    method: TopK
    args: dict[str, Any]
    compile: Optional[str]
    batch_size: int
    topk_size: int
    k: int
    dtype: torch.dtype
    n_warmup: int = 16
    n_outer: int = 16
    n_inner: int = 128

    def save(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        method = d.pop("method")
        d["method"] = f"{method.__module__}.{method.__name__}"
        d["dtype"] = str(d.pop("dtype")).replace("torch.", "")
        d.update(d.pop("args"))
        return d


def fake_topk_sum(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    y = xs.sum(dim=dim, keepdim=True)  # ignore k, just run a sum
    return y, torch.zeros(y.shape, dtype=torch.long, device=y.device)


def run(config: Experiment) -> list[float]:
    durations: list[float] = []

    def _inner_loop_fn(xs: Tensor) -> list[Tensor]:
        return [
            config.method(xs[i], k=config.k, dim=1, **config.args)[0]
            for i in range(config.n_inner)
        ]

    if config.compile:
        torch.compiler.reset()
        _inner_loop_fn = torch.compile(
            _inner_loop_fn, mode=config.compile, fullgraph=True, dynamic=False
        )

    xs = torch.randn(
        config.n_inner,
        config.batch_size,
        config.topk_size,
        dtype=config.dtype,
        device="cuda",
    )
    for _ in range(config.n_warmup):
        _inner_loop_fn(xs)

    for _ in range(config.n_outer):
        xs = torch.randn(
            config.n_inner,
            config.batch_size,
            config.topk_size,
            dtype=config.dtype,
            device="cuda",
        )
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)
        start_event.record()
        _inner_loop_fn(xs)
        end_event.record()
        end_event.synchronize()
        durations.append(start_event.elapsed_time(end_event) / 1e3)

    return durations


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


if __name__ == "__main__":
    sweep(
        [
            Experiment(
                method=method,
                args=args,
                compile=compile,
                batch_size=32,
                topk_size=topk_size,
                k=topk_size // 8,
                dtype=torch.float32,
            )
            for compile in [
                None,
                "default",
                "reduce-overhead",
                # "max-autotune",
            ]
            for method, args in [
                (fake_topk_sum, {}),
                (torch_default.topk, {}),
                (radix_select.topk, {}),
                (bucket_argmax.topk_autobucket, {}),
                (bucket_argmax.topk_torch, {}),
                (bucket_argmax.topk_triton, dict(block_size=128)),
            ]
            for topk_size in [2**n for n in [12, 14, 16]]
            if not (compile and method is radix_select.topk)
            if not (compile == "reduce-overhead" and method is bucket_argmax.topk_triton)
        ],
        Path("results/measure_speed.jsonl"),
    )
