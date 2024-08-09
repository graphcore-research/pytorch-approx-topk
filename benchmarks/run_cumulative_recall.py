"""Computes the cumulative recall of the top k of different approximate methods.

Cumulative recall at j = the probability that the approximate method retrieves all of
the top-j elements.
"""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from approx_topk import TopK, radix_select
from approx_topk.autobucket import bucket

n_repeats = 100
n = 16 * 32 * 40
distributions = {
    "uniform": lambda: torch.rand(n_repeats, n),
    "log normal": lambda: torch.randn(n_repeats, n).exp(),
}
k_ratios = [8, 16]
methods = {"exact": radix_select.topk}
for per_bucket in [1, 2, 4, 8, 16, 32]:
    methods[f"{per_bucket} per bucket"] = bucket(radix_select.topk, per_bucket)


def compute_cumulative_recall(
    distribution: Callable[[], Tensor], k_ratio: int, method: TopK
) -> tuple[list[int], list[float]]:
    assert n % k_ratio == 0
    k = n // k_ratio
    xs = distribution().cuda()
    true_top, _ = torch.topk(xs, k, dim=-1, sorted=True)
    approx_top, _ = method(xs, k, dim=-1)
    approx_top = approx_top.sort(dim=-1, descending=True).values
    coverage = (true_top == approx_top).float().mean(dim=0).tolist()
    js = list(range(1, k + 1))
    return js, coverage


fig, axes = plt.subplots(
    len(distributions),
    len(k_ratios),
    figsize=(10, 10),
    squeeze=False,
    sharex="col",
    sharey=True,
)

for ax_row, (dist_name, dist) in zip(axes, distributions.items()):
    ax_row[0].set_ylabel(dist_name)
    for ax, k_ratio in zip(ax_row, k_ratios):
        for method_name, method in methods.items():
            print(f"Running {dist_name}, k=1/{k_ratio}, {method_name}")
            ratios, recall = compute_cumulative_recall(dist, k_ratio, method)
            ax.plot(ratios, recall, label=method_name)

axes[0, 0].legend()
for ax, k_ratio in zip(axes[0], k_ratios):
    ax.set_title(f"k = n/{k_ratio}")
for ax in axes[-1]:
    ax.set_xlabel("j out of k")
for ax in axes.flatten():
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="x", which="both")

plt.tight_layout()
figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)
plt.savefig(figure_dir / "cumulative_recall.png")
plt.close()
