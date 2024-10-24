# PyTorch Approx Topk

An alpha implementation of the bucketed top-k algorithm using a priority queue.

To do before public release:
- integration as a PyTorch op, to support autograd and torch.compile()
- improved heuristic to decide whether to launch thread-per-bucket or block-per-bucket kernel
- improvements to api

Contents:
- `approx_topk/priority_queue.cu` and `approx_topk/priority_queue.py`: the priority queue kernel
- `approx_topk/bucket_argmax.py`: implementation of bucketed top-k for `b_k=1` using `torch.argmax()`
- `benchmarks/measure_speed.py`: script for measuring runtime/bandwidth as in Figure 1
- `notebooks/20240912-benchmarks-3090.ipynb`: notebook for looking at bandwidth results

To set up the environment, install the dependencies:
- CUDA toolkit 12.1
- Ninja (`ninja-build`)
- Python 3.12
- Python Poetry

Then run `poetry install --with benchmarks`


