import pytest
import torch

from approx_topk import torch_default
from approx_topk.autobucket import bucket


def test__bucket__k_not_divisible_by_k_per_bucket__raises() -> None:
    bucketed_topk = bucket(torch_default.topk, k_per_bucket=12, interleaved=True)
    xs = torch.randn(1024)
    with pytest.raises(NotImplementedError):
        bucketed_topk(xs, 13, dim=0)


# NOTE: Old test, only works for interleaved=False
def test__bucket__top_k_values_ideally_distributed__equal_to_exact_top_k() -> None:
    bucketed_topk = bucket(torch_default.topk, k_per_bucket=2, interleaved=False)
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
    expected_values, expected_indices = torch_default.topk(xs, k, dim=1)

    assert torch.allclose(values.sort().values, expected_values.sort().values)
    assert torch.allclose(indices.sort().values, expected_indices.sort().values)


@pytest.mark.parametrize("interleaved", [True, False])
def test__bucket__only_one_bucket__equal_to_exact_top_k(interleaved) -> None:
    k = 128
    bucketed_topk = bucket(torch_default.topk, k_per_bucket=k, interleaved=interleaved)
    xs = torch.randn(1024)

    values, indices = bucketed_topk(xs, k, dim=0)
    expected_values, expected_indices = torch_default.topk(xs, k, dim=0)

    assert torch.allclose(values.sort().values, expected_values.sort().values)
    assert torch.allclose(indices.sort().values, expected_indices.sort().values)


@pytest.mark.parametrize("interleaved", [True, False])
def test__bucket__one_k_per_bucket__does_not_crash(interleaved) -> None:
    bucketed_topk = bucket(torch_default.topk, k_per_bucket=1, interleaved=interleaved)
    xs = torch.randn(32, 1000)

    values, indices = bucketed_topk(xs, k=16, dim=-1)

    assert values.shape == (32, 16)
    assert indices.shape == (32, 16)
