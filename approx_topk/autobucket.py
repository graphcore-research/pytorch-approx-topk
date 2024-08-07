import math

import torch
import torch.nn.functional as F
from torch import Tensor

from approx_topk import TopK


def bucket(exact_method: TopK, k_per_bucket: int) -> TopK:
    """Construct an approximate, parallelised top k by bucketing an exact method.

    The resulting method is quite slow because of launching additional kernels.
    """

    def bucketed(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
        if k % k_per_bucket != 0:
            raise NotImplementedError
        if dim != -1 and dim != xs.ndim - 1:
            raise NotImplementedError

        n_buckets = k // k_per_bucket
        base_shape = xs.shape[:-1]
        topk_size = xs.shape[-1]

        bucket_size = int(math.floor(topk_size / n_buckets))
        sliced_size = bucket_size * n_buckets
        bucketed_xs = xs[..., :sliced_size].reshape(*base_shape, n_buckets, bucket_size)

        values, indices = exact_method(bucketed_xs, k_per_bucket, dim=-1)

        indices_correction = (
            torch.arange(0, sliced_size, bucket_size, device=indices.device)
            .unsqueeze(-1)
            .expand_as(indices)
        )
        corrected_indices = indices + indices_correction
        return values.reshape(*base_shape, k), corrected_indices.reshape(*base_shape, k)

    return bucketed
