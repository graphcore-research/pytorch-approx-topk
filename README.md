# PyTorch Approx Top-k

Approximate algorithms for computing top-k faster on machine learning accelerators, by using bucketing to increase parallelism.
Rather than computing a single top-k over the sequence:
1. split the sequence into $b$ interleaved buckets
2. take $k_b$ elements from each bucket
3. if $k_b \cdot b > k$: take a final top-k

You can get pretty nice speedups (e.g. several times) with little loss in recall!
See our paper for detailed benchmarks and analysis of the cost/quality trade-off:

_[
  Approximate Top-k for Increased Parallelism;
  O Key, L Ribar, A Cattaneo, L Hudlass-Galley, D Orr
](https://oscarkey.github.io/approx-topk.html)_

The implementation is quite fast, but we welcome any contributions from CUDA experts.
In Figure 1, we compare against `torch.argmax()`, which is a reasonable upper-bound on how fast this kernel could be. There's still room for improvement!

## Using the library
Requires: Python >3.10, PyTorch >=2.4, Ninja (`ninja-build`), CUDA toolkit matching your version of PyTorch
```sh
pip install git+https://github.com/graphcore-research/pytorch-approx-topk.git
```

Usage:

```py
from approx_topk import topk as approx_topk
import torch

x = torch.randn(128, int(2**20), device="cuda")
values, indices = approx_topk(x, k=int(2**16), dim=-1, j=2, k_mult=1)
```
(the kernel is compiled on first use, which might take a while)

Note that, when comparing to the paper, `j` is $k_b$ and `k_mult` is $k_b \cdot b / k$.

## Repository highlights
- `approx_topk.priority_queue`: main CUDA kernel supporting $k_b \in \{1,2,4\}$, implemented using a priority queue algorithm
- `approx_topk.experimental.bucketed_argmax`: implementations for $k_b=1$ only, using `torch.argmax()` and custom Triton kernels
- `benchmarks.measure_speed`: benchmarks speed of our implementation vs exact top-ks (Figure 1 in paper)
  - requires additional dependencies, see below
- `notebooks`: experimental results notebooks (theoretical performance analysis, figure plotting)

## Reproducing benchmarks + development
To set up the environment, install the dependencies:
- CUDA toolkit 12.4
- Ninja (`ninja-build`)
- Python 3.11
- Python Poetry

Then run `poetry install --with benchmarks`

To make it easier to install the CUDA dependencies, we provide an [Apptainer](https://apptainer.org/) image recipe in [environment.simg](environment.simg):
- Build: `apptainer build environment.sif environment.simg`
- Run:
  - `apptainer exec --nv environment.sif python benchmarks/measure_speed.py`
  - `apptainer exec --nv environment.sif python benchmarks/plot_bandwidth.py`

Code tools:
- Type checking: `mypy --ignore-missing-imports -p approx_topk`
- Formatting Python: `ruff format **/*.py`
- Formatting CUDA: `clang-format -i **/*.cu`

## License

Copyright (c) 2024 Graphcore Ltd and Oscar Key. Licensed under the MIT License.
