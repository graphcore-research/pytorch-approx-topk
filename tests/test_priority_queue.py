# Copyright (c) 2024 Graphcore Ltd and Oscar Key. All rights reserved.

import pytest
import torch
from torch import Generator, Tensor

from approx_topk.autobucket import bucketed_torch_topk
from approx_topk.priority_queue import topk
from tests.helper_funcs import assert_close_up_to_permutation


@pytest.mark.parametrize("k", [0, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__one_bucket__no_batch__equal_to_built_in(
    k: int, dtype, interleaved: bool, multithread_buckets: bool
) -> None:
    xs = torch.randn(1001, dtype=dtype, **rng_kwargs(234))
    values, indices = topk(
        xs, k, dim=0, interleaved=interleaved, multithread_buckets=multithread_buckets
    )

    expected_values, expected_indices = torch.topk(xs, k, dim=0)

    assert torch.allclose(values.sort().values, expected_values.sort().values)
    assert torch.allclose(indices.sort().values, expected_indices.sort().values)


@pytest.mark.parametrize("k", [0, 2, 4])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__one_bucket__batched__equal_to_built_in(
    k: int, dim: int, dtype, interleaved: bool, multithread_buckets: bool
) -> None:
    xs = torch.rand((30, 25, 16), dtype=dtype, **rng_kwargs(4242))
    values, indices = topk(
        xs, k, dim, interleaved=interleaved, multithread_buckets=multithread_buckets
    )

    expected_values, expected_indices = torch.topk(xs, k, dim)

    assert torch.allclose(values.sort(dim).values, expected_values.sort(dim).values)
    assert torch.allclose(indices.sort(dim).values, expected_indices.sort(dim).values)


@pytest.mark.parametrize("dim", [-1, -2])
def test__negative_dim__does_not_crash(dim: int) -> None:
    xs = torch.randn((128, 256), **rng_kwargs(99))
    topk(xs, k=4, dim=dim)


def test__call_twice_on_same_data__does_not_crash() -> None:
    xs = torch.randn((128,), **rng_kwargs(56))
    topk(xs, k=4, dim=0)
    topk(xs, k=4, dim=0)


@pytest.mark.parametrize("n", [10, 16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__bucketed__no_batch__bucket_size_one__equal_to_exact(
    n: int, interleaved: bool, multithread_buckets: bool
) -> None:
    xs = torch.randn(n, **rng_kwargs(856))
    # k=n with j=1 ensures that the buckets will have size one.
    k = n
    values, indices = topk(
        xs, k, 0, j=1, interleaved=interleaved, multithread_buckets=multithread_buckets
    )

    expected_values, expected_indices = torch.topk(xs, k, dim=0)

    assert torch.allclose(values.sort().values, expected_values.sort().values)
    assert torch.allclose(indices.sort().values, expected_indices.sort().values)


@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__bucketed__batched__bucket_size_one__equal_to_exact(
    dim: int, dtype, interleaved: bool, multithread_buckets: bool
) -> None:
    xs = torch.randn((16, 4, 12), dtype=dtype, **rng_kwargs(23))
    # k=topk size with j=1 ensures that the buckets will have size one.
    k = xs.size(dim)
    values, indices = topk(
        xs,
        k,
        dim,
        j=1,
        interleaved=interleaved,
        multithread_buckets=multithread_buckets,
    )

    expected_values, expected_indices = torch.topk(xs, k, dim)

    assert torch.allclose(values.sort(dim).values, expected_values.sort(dim).values)
    assert torch.allclose(indices.sort(dim).values, expected_indices.sort(dim).values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__bucketed__batched__k_mult_means_bucket_size_one__equal_to_exact(
    dtype, interleaved: bool, multithread_buckets: bool
) -> None:
    xs = torch.randn((32, 1024), dtype=dtype, **rng_kwargs(23))
    dim = -1
    k = 64
    # Setting k_mult=16 should mean we end up with 1024 buckets, so bucket size 1.
    values, indices = topk(
        xs,
        k,
        dim,
        k_mult=16,
        j=1,
        interleaved=interleaved,
        multithread_buckets=multithread_buckets,
    )

    expected_values, expected_indices = torch.topk(xs, k, dim)

    assert torch.allclose(values.sort(dim).values, expected_values.sort(dim).values)
    assert torch.allclose(indices.sort(dim).values, expected_indices.sort(dim).values)


@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__bucketed__not_interleaved__topk_ideally_distributed__equal_to_exact(
    multithread_buckets: bool,
) -> None:
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

    values, indices = topk(
        xs, k, dim=1, j=j, interleaved=False, multithread_buckets=multithread_buckets
    )
    expected_values, expected_indices = torch.topk(xs, k, dim=1, largest=True)

    assert torch.allclose(values.sort(dim=1).values, expected_values.sort(dim=1).values)
    assert torch.allclose(
        indices.sort(dim=1).values, expected_indices.sort(dim=1).values
    )


@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__bucketed__not_interleaved__topk_ideally_distributed__long_sequence__equal_to_exact(
    multithread_buckets: bool,
) -> None:
    n = 10_000
    k = 8
    j = 2
    xs = torch.zeros((2, n), device="cuda")
    # The bucket size is n / k = 1250, thus we set elements at a stride of half that to
    # put two elements in each bucket.
    stride = 1250 // 2
    xs[0, tuple(range(0, n, stride))] = torch.rand(n // stride, **rng_kwargs(1)) + 2.0
    xs[1, tuple(range(50, n, stride))] = torch.rand(n // stride, **rng_kwargs(1)) + 2.0
    # xs[1, -1] = 10.0

    values, indices = topk(
        xs, k, dim=1, j=j, interleaved=False, multithread_buckets=multithread_buckets
    )
    expected_values, expected_indices = torch.topk(xs, k, dim=1, largest=True)

    assert torch.allclose(values.sort(dim=1).values, expected_values.sort(dim=1).values)
    assert torch.allclose(
        indices.sort(dim=1).values, expected_indices.sort(dim=1).values
    )


@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__bucketed__interleaved__topk_ideally_distributed__equal_to_exact(
    multithread_buckets: bool,
) -> None:
    # We start with a sequence of length 17:
    # [0 1 2 3] [4 5 6 7] [8 9 10 11] [12 13 14 15] 16
    # Once interleaved, we end up with the following buckets
    # [0 4 8 12 16] [1 5 9 13] [2 6 10 14] [3 7 11 15]
    # We then take two from each bucket.
    k = 8
    j = 2
    xs = torch.full((2, 17), fill_value=0.0, device="cuda")
    # Ensure each bucket contains two of the top k.
    xs[0, 0] = 11.0
    xs[0, 16] = 12.0
    xs[0, 1] = 13.0
    xs[0, 13] = 14.0
    xs[0, 2] = 15.0
    xs[0, 14] = 16.0
    xs[0, 3] = 17.0
    xs[0, 15] = 18.0
    xs[1, 4] = 21.0
    xs[1, 8] = 22.0
    xs[1, 5] = 23.0
    xs[1, 9] = 24.0
    xs[1, 6] = 25.0
    xs[1, 10] = 26.0
    xs[1, 7] = 27.0
    xs[1, 11] = 28.0

    values, indices = topk(
        xs, k, dim=1, j=j, interleaved=True, multithread_buckets=multithread_buckets
    )
    expected_values, expected_indices = torch.topk(xs, k, dim=1, largest=True)

    assert torch.allclose(values.sort(dim=1).values, expected_values.sort(dim=1).values)
    assert torch.allclose(
        indices.sort(dim=1).values, expected_indices.sort(dim=1).values
    )


@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("k_per_bucket", [1, 2, 4])
@pytest.mark.parametrize("multithread_buckets", [False, True])
def test__equal_to_reference_bucketed_torch_topk(
    interleaved: bool, k_per_bucket: int, multithread_buckets: bool
) -> None:
    torch.manual_seed(100)
    xs = torch.randn(5, 32, 512, device="cuda")
    values, indices = topk(
        xs,
        k=64,
        dim=-1,
        j=k_per_bucket,
        interleaved=interleaved,
        multithread_buckets=multithread_buckets,
    )
    expected_values, expected_indices = bucketed_torch_topk(
        xs,
        k=64,
        dim=-1,
        k_mult=1,
        k_per_bucket=k_per_bucket,
        interleaved=interleaved,
    )
    assert_close_up_to_permutation(values, expected_values)
    assert_close_up_to_permutation(indices, expected_indices)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("multithread_buckets", [None, False, True])
def test__passes_op_check(
    dtype, interleaved: bool, multithread_buckets: None | bool
) -> None:
    xs = torch.randn(32, 128, **rng_kwargs(0))
    torch.library.opcheck(
        topk,
        args=(xs,),
        kwargs=dict(
            k=12,
            dim=-1,
            j=1,
            interleaved=interleaved,
            multithread_buckets=multithread_buckets,
        ),
    )


compiled_topk = torch.compile(topk, fullgraph=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("multithread_buckets", [None, False, True])
def test__torch_compile__equal_to_uncompiled(
    dtype, interleaved: bool, multithread_buckets: None | bool
) -> None:
    xs = torch.randn(32, 128, **rng_kwargs(0))
    kwargs = dict(
        xs=xs,
        k=12,
        dim=-1,
        j=2,
        interleaved=interleaved,
        multithread_buckets=multithread_buckets,
    )
    expected_values, expected_indices = topk(**kwargs)
    compiled_values, compiled_indices = compiled_topk(**kwargs)

    assert torch.allclose(expected_values, compiled_values)
    assert torch.allclose(expected_indices, compiled_indices)


def rng_kwargs(seed: int):
    return dict(generator=Generator(device="cuda").manual_seed(seed), device="cuda")
