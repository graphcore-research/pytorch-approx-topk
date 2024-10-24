# PyTorch Approx Topk

An alpha implementation of the bucketed top-k algorithm using a priority queue.

Requires: Python 3.11, CUDA toolkit 12.1.
```sh
pip install git+https://github.com/graphcore-research/pytorch-approx-topk.git
```

Usage:

```py
from approx_topk.priority_queue import topk as approx_topk
import torch

x = torch.randn(128, int(2**20), device="cuda")
values, indices = approx_topk(x, k=int(2**16), dim=-1, j=2, k_mult=1)
```

Note that `j` is $k_b$ and `k_mult` is $k_b \cdot b / k$.

Repository highlights:

- `approx_top/` PyTorch library code
  - [priority_queue.py](approx_topk/priority_queue.py) custom priority queue implementation (also [priority_queue.cu](approx_topk/priority_queue.cu))
  - [bucket_argmax.py](approx_topk/bucket_argmax.py) $k_b\!=\!1$ torch & triton implementations
- `benchmarks/` benchmarking scripts
  - [measure_speed.py](benchmarks/measure_speed.py) main benchmarking script for measuring runtime/bandwidth as in Figure 1
- `notebooks/` experimental results notebooks (including work-in-progress results)
  - [20240912-benchmarks-3090.ipynb](notebooks/20240912-benchmarks-3090.ipynb) example of visualising memory bandwidth results

## Development

To set up the environment, install the dependencies:
- CUDA toolkit 12.1
- Ninja (`ninja-build`)
- Python 3.11
- Python Poetry

Then run `poetry install --with benchmarks`

## License

Copyright (c) 2024 Graphcore Ltd and Oscar Key. Licensed under the MIT License.
