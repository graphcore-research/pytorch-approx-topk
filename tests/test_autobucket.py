# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

import pytest
import torch
from torch import Tensor

from approx_topk.experimental.autobucket import bucket
from tests.helper_funcs import assert_close_up_to_permutation


def topk(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    return torch.topk(xs, k, dim, sorted=False)


def test__bucket__k_not_divisible_by_k_per_bucket__raises() -> None:
    bucketed_topk = bucket(topk, k_mult=1, k_per_bucket=12, interleaved=True)
    xs = torch.randn(1024)
    with pytest.raises(NotImplementedError):
        bucketed_topk(xs, 13, dim=0)


# NOTE: Only works for interleaved=False
def test__bucket__top_k_values_ideally_distributed__equal_to_exact_top_k() -> None:
    bucketed_topk = bucket(topk, k_mult=1, k_per_bucket=2, interleaved=False)
    k = 8
    # Make sure input size is not divisible by number of buckets (in this case 4) to
    # test padding code.
    xs = torch.full((2, 1025), fill_value=0.0)
    # Ensure each bucket contains two of the top k.
    xs[0, 1] = 10.0
    xs[0, 2] = 11.0
    xs[0, 300] = 11.0
    xs[0, 301] = 12.0
    xs[0, 600] = 11.0
    xs[0, 601] = 12.0
    xs[0, 900] = 13.0
    xs[0, 901] = 12.0
    xs[1, 5] = 10.0
    xs[1, 6] = 11.0
    xs[1, 302] = 11.0
    xs[1, 305] = 12.0
    xs[1, 610] = 11.0
    xs[1, 611] = 12.0
    xs[1, 920] = 13.0
    xs[1, 921] = 12.0

    values, indices = bucketed_topk(xs, k, dim=1)
    expected_values, expected_indices = topk(xs, k, dim=1)

    assert_close_up_to_permutation(values, expected_values, dim=1)
    assert_close_up_to_permutation(indices, expected_indices, dim=1)


@pytest.mark.parametrize("interleaved", [True, False])
def test__bucket__only_one_bucket__equal_to_exact_top_k(interleaved) -> None:
    k = 128
    bucketed_topk = bucket(topk, k_mult=1, k_per_bucket=k, interleaved=interleaved)
    xs = torch.randn(1024)

    values, indices = bucketed_topk(xs, k, dim=0)
    expected_values, expected_indices = topk(xs, k, dim=0)

    assert_close_up_to_permutation(values, expected_values)
    assert_close_up_to_permutation(indices, expected_indices)


@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("k_mult", [1, 2, 4])
@pytest.mark.parametrize("k_per_bucket", [1, 2, 4])
def test__bucket__one_k_per_bucket__does_not_crash(
    interleaved, k_mult, k_per_bucket
) -> None:
    bucketed_topk = bucket(
        topk,
        k_mult=k_mult,
        k_per_bucket=k_per_bucket,
        interleaved=interleaved,
    )
    xs = torch.randn(5, 32, 1000)

    values, indices = bucketed_topk(xs, k=16, dim=-1)

    assert values.shape == (5, 32, 16)
    assert indices.shape == (5, 32, 16)


# === ADDITIONAL TESTS ===
def test_bucket_non_interleaved() -> None:
    # b1 = {0, 1, 2, 3}, b2 = {4, 5, 6}, b3 = {7, 8, 9}
    xs = torch.arange(10)
    topk = bucket(torch.topk, k_mult=1, k_per_bucket=1, interleaved=False)
    values, indices = topk(xs, k=3, dim=-1)
    expected = torch.tensor([3, 6, 9])

    assert_close_up_to_permutation(values, expected)
    assert_close_up_to_permutation(indices, expected)


def test_bucket_interleaved() -> None:
    # b1 = {0, 3, 6, 9}, b2 = {1, 4, 7, PAD}, b3 = {2, 5, 8, PAD}
    xs = torch.arange(10)
    topk = bucket(torch.topk, k_mult=1, k_per_bucket=1, interleaved=True)
    values, indices = topk(xs, k=3, dim=-1)
    expected = torch.tensor([9, 7, 8])

    assert_close_up_to_permutation(values, expected)
    assert_close_up_to_permutation(indices, expected)


def test_bucket_k_mult_not_one() -> None:
    xs = torch.arange(16)
    topk = bucket(torch.topk, k_mult=2, k_per_bucket=1, interleaved=False)
    values, indices = topk(xs, k=4, dim=-1)
    expected = torch.tensor([15, 13, 11, 9])

    assert_close_up_to_permutation(values, expected)
    assert_close_up_to_permutation(indices, expected)


def test_bucket_k_mult_not_one_interleaved() -> None:
    xs = torch.arange(16)
    topk = bucket(torch.topk, k_mult=2, k_per_bucket=1, interleaved=True)
    values, indices = topk(xs, k=4, dim=-1)
    expected = torch.tensor([15, 14, 13, 12])

    assert_close_up_to_permutation(values, expected)
    assert_close_up_to_permutation(indices, expected)
