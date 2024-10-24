# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

"""Benchmarks the top-50 and top-k recall for different values of k."""

from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from torch import Tensor

from approx_topk import Topk, priority_queue

n_repeats = 100
n = 16 * 32 * 40
distributions = {
    "uniform": lambda: torch.rand(n_repeats, n),
    "log normal": lambda: torch.randn(n_repeats, n).exp(),
}
methods = {"exact": torch.topk}
for per_bucket in [1, 2, 4]:
    methods[f"{per_bucket} per bucket"] = partial(priority_queue.topk, j=per_bucket)


def compute_recall(
    distribution: Callable[[], Tensor], method: Topk, k: int
) -> tuple[float, float]:
    xs = distribution().cuda()
    true_top, _ = torch.topk(xs, k, dim=-1, sorted=True)
    approx_top, _ = method(xs, k, dim=-1)
    approx_top = approx_top.sort(dim=-1, descending=True).values
    top_50 = ((approx_top >= true_top[:, 49:50]).sum(-1) / 50).mean().item()
    top_k = ((approx_top >= true_top[:, k - 1 : k]).sum(-1) / k).mean().item()
    return top_50, top_k


def plot_recalls(
    distribution: Callable[[], Tensor], method: Topk, ax: list[Axes], **plot_kwargs
) -> None:
    ks = list(range(1024, n, 128))
    top_50_recall, top_k_recall = zip(
        *[compute_recall(distribution, method, k) for k in ks]
    )
    ax[0].plot(ks, top_50_recall, **plot_kwargs)
    ax[1].plot(ks, top_k_recall, **plot_kwargs)


fig, axes = plt.subplots(
    len(distributions),
    ncols=2,
    figsize=(10, 10),
    squeeze=False,
    sharex="col",
    sharey=True,
)

for ax_row, (dist_name, dist) in zip(axes, distributions.items()):
    ax_row[0].set_ylabel(dist_name)
    for method_name, method in methods.items():
        print(f"Running {dist_name}, {method_name}")
        plot_recalls(dist, method, ax_row.tolist(), label=method_name)

axes[0, 0].legend()
axes[0, 0].set_title("top 50 recall")
axes[0, 1].set_title("top k recall")
# for ax, k_ratio in zip(axes[0], k_ratios):
#     ax.set_title(f"k = n/{k_ratio}")
for ax in axes[-1]:
    ax.set_xlabel("k")
for ax in axes.flatten():
    ax.grid(axis="x", which="both")

plt.tight_layout()
figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)
plt.savefig(figure_dir / "recall.png")
plt.close()
