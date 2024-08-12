from functools import partial

import pytest
import torch

from approx_topk.bucket_argmax import topk_torch, topk_triton


@pytest.mark.parametrize("topk", [topk_torch, partial(topk_triton, block_size=8)])
def test_bucket_argmax(topk):
    inputs = torch.stack([100 * torch.arange(0, 20), -100 * torch.arange(0, 20)]).to(
        dtype=torch.float32, device="cuda"
    )
    values, indices = topk(inputs, k=3, dim=1)
    assert torch.equal(
        values,
        torch.tensor(
            [[600, 1300, 1900], [0, -700, -1400]], dtype=torch.float32, device="cuda"
        ),
    )
    assert torch.equal(indices, torch.tensor([[6, 13, 19], [0, 7, 14]], device="cuda"))
