"""Compares the time taken by different topk methods."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.cuda import Event
from tqdm import tqdm

from approx_topk import TopK
from approx_topk.exact_methods import radix_select, torch_default


def run_config(
    method: TopK, batch_size: int, topk_size: int, k: int
) -> tuple[float, float]:
    n_seqs = 50
    n_repeats = 20

    durations: list[float] = []
    for _ in range(n_seqs):
        xs = torch.randn(batch_size, topk_size).cuda()

        # Warm up
        method(xs, k, dim=1)
        method(xs, k, dim=1)

        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)
        start_event.record()
        for _ in range(n_repeats):
            method(xs, k, dim=1)
        end_event.record()
        end_event.synchronize()
        durations.append(start_event.elapsed_time(end_event))

    ds = torch.tensor(durations)
    return torch.mean(ds).item(), torch.std(ds).item()


methods = {
    "radix whole sequence": radix_select.topk,
    "torch default": torch_default.topk,
}
batch_sizes = [32]
topk_sizes = list(range(1_000, 20_000, 1_000))

for method_name, method in methods.items():
    for batch_size in batch_sizes:
        ks = [round(topk_size / 8) for topk_size in tqdm(topk_sizes)]
        means_stds = [
            run_config(method, batch_size, topk_size, k)
            for topk_size, k in zip(topk_sizes, ks)
        ]
        means, stds = zip(*means_stds)
        label = f"{method_name} batch_size={batch_size}"
        plt.errorbar(topk_sizes, means, yerr=stds, label=label)

plt.xlabel("topk size")
plt.ylabel("duration (milliseconds)")
plt.legend()

plt.tight_layout()
figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)
plt.savefig(figure_dir / "benchmark.png")
plt.close()
