import torch

from approx_topk import TopK
from approx_topk.exact_methods import radix_select


def test__radix_select__no_batch__equal_to_built_in() -> None:
    _test(radix_select.topk, xs_shape=(256,), dim=0)


def test__radix_select__with_batch__equal_to_built_in() -> None:
    _test(radix_select.topk, xs_shape=(256, 512, 128), dim=0)
    _test(radix_select.topk, xs_shape=(256, 512, 128), dim=1)
    _test(radix_select.topk, xs_shape=(256, 512, 128), dim=2)


def test__radix_select__call_twice_on_same_data__does_not_crash() -> None:
    xs = torch.randn((128,), device="cuda:0")
    radix_select.topk(xs, k=128, dim=0)
    radix_select.topk(xs, k=128, dim=0)


def test__radix_select__negative_dim__does_not_crash() -> None:
    xs = torch.randn((128, 256), device="cuda:0")
    radix_select.topk(xs, k=128, dim=-1)


def _test(method: TopK, xs_shape: tuple[int, ...], dim: int) -> None:
    for k in [0, 12, 50, 100, xs_shape[dim]]:
        xs = torch.randn(xs_shape, device="cuda:0")
        values, indices = method(xs, k, dim)

        expected_values, expected_indices = torch.topk(xs, k, dim, sorted=False)

        assert torch.allclose(values.sort().values, expected_values.sort().values)
        assert torch.allclose(indices.sort().values, expected_indices.sort().values)
