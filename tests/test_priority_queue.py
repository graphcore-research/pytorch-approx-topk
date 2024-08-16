import pytest
import torch
from torch import Generator

from approx_topk.priority_queue import topk


@pytest.mark.parametrize("k", [0, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test__one_bucket__no_batch__equal_to_built_in(k: int, dtype) -> None:
    xs = torch.randn(1001, dtype=dtype, **rng_kwargs(234))
    values, indices = topk(xs, k, dim=0)

    expected_values, expected_indices = torch.topk(xs, k, dim=0)

    assert torch.allclose(values.sort().values, expected_values.sort().values)
    assert torch.allclose(indices.sort().values, expected_indices.sort().values)


@pytest.mark.parametrize("k", [0, 4, 7])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32])
def test__one_bucket__batched__equal_to_built_in(k: int, dim: int, dtype) -> None:
    xs = torch.rand((30, 25, 16), dtype=dtype, **rng_kwargs(4242))
    values, indices = topk(xs, k, dim)

    expected_values, expected_indices = torch.topk(xs, k, dim)

    assert torch.allclose(values.sort(dim).values, expected_values.sort(dim).values)
    assert torch.allclose(indices.sort(dim).values, expected_indices.sort(dim).values)


@pytest.mark.parametrize("dim", [-1, -2])
def test__negative_dim__does_not_crash(dim: int) -> None:
    xs = torch.randn((128, 256), **rng_kwargs(99))
    topk(xs, k=10, dim=dim)


def test__call_twice_on_same_data__does_not_crash() -> None:
    xs = torch.randn((128,), **rng_kwargs(56))
    topk(xs, k=8, dim=0)
    topk(xs, k=8, dim=0)


@pytest.mark.parametrize("n", [10, 16])
def test__bucketed__no_batch__bucket_size_one__equal_to_exact(n: int) -> None:
    xs = torch.randn(n, **rng_kwargs(856))
    # k=n with j=1 ensures that the buckets will have size one.
    k = n
    values, indices = topk(xs, k, 0, j=1)

    expected_values, expected_indices = torch.topk(xs, k, dim=0)

    assert torch.allclose(values.sort().values, expected_values.sort().values)
    assert torch.allclose(indices.sort().values, expected_indices.sort().values)


@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test__bucketed__batched__bucket_size_one__equal_to_exact(dim: int, dtype) -> None:
    xs = torch.randn((16, 4, 12), dtype=dtype, **rng_kwargs(23))
    # k=topk size with j=1 ensures that the buckets will have size one.
    k = xs.size(dim)
    values, indices = topk(xs, k, dim, j=1)

    expected_values, expected_indices = torch.topk(xs, k, dim)

    assert torch.allclose(values.sort(dim).values, expected_values.sort(dim).values)
    assert torch.allclose(indices.sort(dim).values, expected_indices.sort(dim).values)


def test__bucketed__topk_ideally_distributed__equal_to_exact() -> None:
    # We set up 4 buckets for a sequence length of 17:
    # [0 1 2 3 4] [5 6 7 8] [9 10 11 12] [13 14 15 16]
    # We then take two from each bucket.
    k = 8
    j = 2
    xs = torch.full((2, 17), fill_value=0.0, device="cuda")
    # Ensure each bucket contains two of the top k.
    xs[0, 0] = 11.0
    xs[0, 4] = 12.0
    xs[0, 5] = 13.0
    xs[0, 8] = 14.0
    xs[0, 9] = 15.0
    xs[0, 12] = 16.0
    xs[0, 13] = 17.0
    xs[0, 16] = 18.0
    xs[1, 2] = 21.0
    xs[1, 3] = 22.0
    xs[1, 6] = 23.0
    xs[1, 7] = 24.0
    xs[1, 10] = 25.0
    xs[1, 11] = 26.0
    xs[1, 14] = 27.0
    xs[1, 15] = 28.0

    values, indices = topk(xs, k, dim=1, j=j)
    expected_values, expected_indices = torch.topk(xs, k, dim=1, largest=True)

    assert torch.allclose(values.sort(dim=1).values, expected_values.sort(dim=1).values)
    assert torch.allclose(
        indices.sort(dim=1).values, expected_indices.sort(dim=1).values
    )


def rng_kwargs(seed: int):
    return dict(generator=Generator(device="cuda").manual_seed(seed), device="cuda")
