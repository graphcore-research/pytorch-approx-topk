from functools import partial

import pytest
import torch
import numpy as np
import triton
from approx_topk.bucket_argmax import topk_torch, topk_triton, _batch_contiguous


def test_batch_contiguous():
    x = torch.arange(3 * 4 * 5).view(3, 4, 5)
    for t, same_instance in [
        (x, True),
        (x[:, :, :-1], True),
        (x[:, :, ::2], False),
        (x[:, :-1, :], False),
        (x.swapdims(0, 1), False),
    ]:
        t2 = _batch_contiguous(t)
        if same_instance:
            assert t2 is t
        else:
            assert torch.equal(t, t2)
            assert t2.is_contiguous()
        # Invariants
        stride = t2.stride()
        assert stride[-1] == 1
        expected_stride = [1, stride[-2]]
        for dim in range(t.ndim - 2, 0, -1):
            expected_stride.append(expected_stride[-1] * t.shape[dim])
        assert (*stride[:-1], 1) == tuple(expected_stride[::-1])


@pytest.mark.parametrize(
    "topk",
    [
        topk_torch,
        partial(topk_triton, block_size=1, kernel="bk"),
        partial(topk_triton, block_size=8, kernel="bk"),
        partial(topk_triton, block_size=1, kernel="bkn"),
        partial(topk_triton, block_size=8, kernel="bkn"),
    ],
)
def test_bucket_argmax(topk):
    inputs = torch.stack([100 * torch.arange(0, 20), -100 * torch.arange(0, 20)]).to(
        dtype=torch.float32, device="cuda"
    )

    # Regular
    values, indices = topk(inputs, k=3, dim=1, interleaved=False)
    assert torch.equal(
        values.cpu(), torch.tensor([[600.0, 1300, 1900], [0, -700, -1400]])
    )
    assert torch.equal(indices.cpu(), torch.tensor([[6, 13, 19], [0, 7, 14]]))

    # Transposed
    values_t, indices_t = topk(inputs.T, k=3, dim=0, interleaved=False)
    assert torch.equal(values_t.T, values)
    assert torch.equal(indices_t.T, indices)

    # Interleaved
    values_i, indices_i = topk(inputs, k=3, dim=1, interleaved=True)
    assert torch.equal(
        values_i.cpu(), torch.tensor([[1800.0, 1900, 1700], [0, -100, -200]])
    )
    assert torch.equal(indices_i.cpu(), torch.tensor([[18, 19, 17], [0, 1, 2]]))


@pytest.mark.parametrize("kernel", ["bk", "bkn"])
def test_bucket_argmax_triton_fuzz(kernel: str):
    for seed in np.random.SeedSequence(123).generate_state(20):
        rng = np.random.RandomState(seed)
        torch.manual_seed(rng.randint(100000))
        b = 2 ** rng.randint(1, 4 + 1) if rng.rand() < 0.5 else rng.randint(1, 2**4 + 1)
        n = (
            2 ** rng.randint(1, 10 + 1)
            if rng.rand() < 0.5
            else rng.randint(1, 2**10 + 1)
        )
        # Certain values of k create 'unavoidable' padding
        # k = max(1, n//rng.randint(1, 32 + 1)) if rng.rand() < 0.5 else rng.randint(1, n + 1)
        k = triton.cdiv(n, rng.randint(1, 32 + 1))
        block_size = 2 ** rng.randint(0, 4 + 1)

        x = torch.randint(-(2**15), 2**15, (b, n), device="cuda")
        expected_values, expected_indices = topk_torch(x, k=k, dim=1, interleaved=False)
        actual_values, actual_indices = topk_triton(
            x,
            k=k,
            dim=1,
            block_size=block_size,
            kernel=kernel,
            interleaved=False,
        )

        print(f"seed={seed} | b={b} n={n} k={k} block_size={block_size}")
        print("expected", expected_values, "actual", actual_values)
        assert torch.equal(
            expected_values.sort(dim=1).values, actual_values.sort(dim=1).values
        )
        assert torch.equal(
            expected_indices.sort(dim=1).values, actual_indices.sort(dim=1).values
        )
