"""Computes and plots the memory bandwidth as measured in measure_speed.py."""

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rc("axes", **{"spines.top": False, "spines.right": False})
matplotlib.rc("legend", frameon=False)


def name(s: pd.Series) -> str:
    method = re.sub(r"\b(experimental|approx_topk|topk|__main__)\b", "", s.method)
    method = re.sub(r"(^\.+)|(\.+$)", "", method)
    args = ", ".join(f"{k}={v}" for k, v in s.args.items())
    return f"{method}({args})".replace("()", "")


df = (
    pd.read_json("results/measure_speed.jsonl", lines=True)
    .pipe(
        lambda d: d.assign(
            input_mode=d.n_inner_inputs.apply(lambda x: "small" if x == 1 else "large"),
            compile=d.compile.fillna("eager"),
            method=d.apply(name, axis=1),
            k_ratio=[
                "small" if k == 64 else topk_size // k
                for _, topk_size, k in d[["topk_size", "k"]].itertuples()
            ],
            duration=d.duration.apply(np.mean) / d.n_inner,
            duration_stderr=d.duration.apply(lambda d: np.std(d) / np.sqrt(len(d)))
            / d.n_inner,
        )
    )
    .pipe(
        lambda d: d.assign(
            bandwidth=d.dtype.apply(dict(float32=4, float16=2).__getitem__)
            * d.topk_size
            * d.batch_size
            / d.duration,
        )
    )
)
print(df.head())
print(df["method"].unique())

plt.figure(figsize=(10, 5))
(input_mode,) = df.input_mode.unique()
g = sns.relplot(
    data=df,
    y="bandwidth",
    x="topk_size",
    hue="method",
    col="k_ratio",
    row="batch_size",
    style="compile",
    kind="line",
    height=3,
    aspect=1.2,
)
for ax in g.axes.flatten():
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e9:.0f} GB/s")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x//1024:.0f} k")
    ax.set_xlabel("n")
    ax.set_ylabel("Read bandwidth")
g.figure.suptitle(f"Input={input_mode}", y=1.05)
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
g.figure.savefig(figures_dir / "bandwidth.png", dpi=300)
