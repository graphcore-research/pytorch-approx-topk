from pathlib import Path

import torch

import approx_topk.bucket_argmax
import benchmarks.measure_speed as B

xps = [
    B.Experiment(
        method=method,
        args=dict(interleaved=interleaved, **args),
        compile=compile,
        cuda_graphs=True,
        batch_size=32,
        topk_size=topk_size,
        k=topk_size // topk_ratio,
        dtype=torch.float16,
    )
    for interleaved in [False, True]
    for topk_ratio in [int(r * 2**n) for n in range(1, 4 + 1) for r in [1, 1.5]]
    for topk_size in [
        int(r * 2**n + offset)
        for n in range(10, 16 + 1)
        for r in [1, 1.25, 1.5, 1.75]
        for offset in [-1, 0, 1, 16]
    ]
    for method, compile, args in [
        (approx_topk.bucket_argmax.topk_torch, "default", {}),
        (
            approx_topk.bucket_argmax.topk_triton,
            None,
            dict(block_size=128, kernel="bk"),
        ),
        (
            approx_topk.bucket_argmax.topk_triton,
            None,
            dict(block_size=64, kernel="bkn"),
        ),
    ]
]
B.sweep(xps, Path("results/20240814_sweep_sequence_length.jsonl"))
