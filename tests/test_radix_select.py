import torch

from approx_topk import TopK, radix_select


def test__one_bucket_and_no_batch__equal_to_built_in() -> None:
    _test_one_bucket(radix_select.topk, xs_shape=(256,), dim=0)


def test__one_bucket_with_batch__equal_to_built_in() -> None:
    _test_one_bucket(radix_select.topk, xs_shape=(256, 512, 128), dim=0)
    _test_one_bucket(radix_select.topk, xs_shape=(256, 512, 128), dim=1)
    _test_one_bucket(radix_select.topk, xs_shape=(256, 512, 128), dim=2)


def _test_one_bucket(method: TopK, xs_shape: tuple[int, ...], dim: int) -> None:
    for k in [0, 12, 50, 100, xs_shape[dim]]:
        xs = torch.randn(xs_shape, device="cuda:0")
        values, indices = method(xs, k, dim)

        expected_values, expected_indices = torch.topk(xs, k, dim, sorted=False)

        assert torch.allclose(values.sort().values, expected_values.sort().values)
        assert torch.allclose(indices.sort().values, expected_indices.sort().values)


def test__call_twice_on_same_data__does_not_crash() -> None:
    xs = torch.randn((128,), device="cuda:0")
    radix_select.topk(xs, k=128, dim=0)
    radix_select.topk(xs, k=128, dim=0)


def test__negative_dim__does_not_crash() -> None:
    xs = torch.randn((128, 256), device="cuda:0")
    radix_select.topk(xs, k=128, dim=-1)


def test__bucketed_and_top_k_values_ideally_distributed__equal_to_exact_top_k() -> None:
    k = 8
    j = 2
    xs = torch.full((2, 1024), fill_value=0.0, device="cuda")
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

    values, indices = radix_select.topk(xs, k, dim=1, j=j)
    expected_values, expected_indices = torch.topk(xs, k, dim=1, largest=True)

    assert torch.allclose(values.sort().values, expected_values.sort().values)
    assert torch.allclose(indices.sort().values, expected_indices.sort().values)
