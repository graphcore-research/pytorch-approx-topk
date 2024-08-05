import torch

from approx_topk.exact_methods import ExactTopK, radix_select


def test__radix_select__no_batch__equal_to_built_in() -> None:
    _test(radix_select.topk, xs_shape=(256,), dim=0)


def test__radix_select__with_batch__equal_to_built_in() -> None:
    _test(radix_select.topk, xs_shape=(256, 512, 128), dim=0)
    _test(radix_select.topk, xs_shape=(256, 512, 128), dim=1)
    _test(radix_select.topk, xs_shape=(256, 512, 128), dim=2)


def _test(method: ExactTopK, xs_shape: tuple[int, ...], dim: int) -> None:
    for k in [0, 12, 50, 100, xs_shape[dim]]:
        xs = torch.randn(xs_shape, device="cuda:0")
        values, indices = method(xs, k, dim)

        expected_values, expected_indices = torch.topk(xs, k, dim, sorted=False)

        assert torch.allclose(values.sort().values, expected_values.sort().values)
        assert torch.allclose(indices.sort().values, expected_indices.sort().values)
