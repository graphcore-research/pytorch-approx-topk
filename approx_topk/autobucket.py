import torch
from torch import Tensor
from torch.nn.functional import pad

from approx_topk import TopK


def bucket(exact_method: TopK, k_per_bucket: int, interleaved: bool) -> TopK:
    """Construct an approximate, parallelised top k by bucketing an exact method.

    The resulting method is quite slow because of launching additional kernels.
    """

    def bucketed(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
        if k % k_per_bucket != 0:
            raise NotImplementedError

        dim = dim % xs.ndim
        if dim != xs.ndim - 1:
            values, indices = bucketed(xs.movedim(dim, -1), k=k, dim=-1)
            return values.movedim(-1, dim), indices.movedim(-1, dim)

        n = xs.size(-1)
        n_buckets = k // k_per_bucket

        n_pad = -n % n_buckets
        pad_value = (
            torch.finfo(xs.dtype).min
            if xs.dtype.is_floating_point
            else torch.iinfo(xs.dtype).min
        )

        if interleaved:
            xs_pad = pad(xs, pad=(0, n_pad), value=pad_value)
            values, indices = exact_method(
                xs_pad.unflatten(-1, (-1, n_buckets)).transpose(-1, -2),
                k=k_per_bucket,
                dim=-1,
            )
            idx_offset = torch.arange(
                0, n_buckets, dtype=torch.int64, device=xs.device
            ).unsqueeze(-1)
            indices.mul_(n_buckets).add_(idx_offset)
        else:
            xs_pad_b = torch.empty(
                size=(*xs.shape[:-1], n + n_pad),
                dtype=xs.dtype,
                device=xs.device,
            ).unflatten(-1, (n_buckets, -1))

            mask = torch.full(
                size=xs_pad_b.shape, fill_value=True, device=xs_pad_b.device
            )
            r = n % n_buckets or n_buckets
            mask[..., r:, -1] = False

            xs_pad_b[mask] = xs.flatten()
            xs_pad_b[~mask] = pad_value

            values, indices = exact_method(xs_pad_b, k=k_per_bucket, dim=-1)
            idx_offset = mask.sum(dim=-1, keepdim=True).cumsum(dim=-2)
            indices[..., 1:, :].add_(idx_offset[..., :-1, :])

        return values.flatten(-2), indices.flatten(-2)

    return bucketed
