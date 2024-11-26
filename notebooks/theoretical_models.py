from math import log2
from typing import Any, Iterable

import numpy as np
import scipy.stats


class recall:
    def model(*, k: int, n: int, m: int, b: int, k_b: int) -> float:
        return float(
            (k_b + scipy.stats.binom.cdf(k_b - 1, np.arange(k_b, k), p=1 / b).sum()) / k
        )

    def simulation(
        *, k: int, n: int, m: int, b: int, k_b: int, reps: int
    ) -> tuple[float, float]:
        samples = np.array(
            [
                (np.random.choice(n, size=n, replace=False) < k)
                .reshape(b, -1)
                .sum(-1)
                .clip(max=k_b)
                .sum()
                / k
                for _ in range(reps)
            ]
        )
        return float(samples.mean()), float(samples.std())

    def simulation_birthday(
        *, k: int, n: int, m: int, b: int, k_b: int, reps: int
    ) -> tuple[float, float]:
        samples = np.array(
            [
                np.bincount(np.random.randint(0, b, k), minlength=b).clip(max=k_b).sum()
                / k
                for _ in range(reps)
            ]
        )
        return float(samples.mean()), float(samples.std())


class cost:
    @staticmethod
    def topk(*, k: int, n: int, m: int) -> int: ...

    @classmethod
    def approx_topk(cls, *, k: int, n: int, m: int, b: int, k_b: int) -> int:
        return cls.topk(k=k_b, n=n // b, m=m * b) + (k_b * b > k) * cls.topk(
            k=k, n=k_b * b, m=m
        )


class cost_basic(cost):
    @staticmethod
    def topk(*, k: int, n: int, m: int) -> int:
        return m * n * (log2(k) + 1)


class cost_serial(cost):
    @staticmethod
    def insertion(*, k: int, n: int, m: int) -> int:
        return m * n * (3 * k - 1)

    @staticmethod
    def radix(*, k: int, n: int, m: int) -> int:
        return m * n * (4 * log2(n) + 4)

    @classmethod
    def topk(cls, **kwargs: Any) -> int:
        return min(cls.insertion(**kwargs), cls.radix(**kwargs))


class cost_hybrid(cost):
    @staticmethod
    def insertion(*, k: int, n: int, m: int) -> int:
        return n * (3 * k - 1)

    @staticmethod
    def radix(*, k: int, n: int, m: int) -> int:
        return n * (4 * log2(n) + 4)

    @classmethod
    def topk(cls, **kwargs: Any) -> int:
        return min(cls.insertion(**kwargs), cls.radix(**kwargs))


class cost_parallel(cost):
    @staticmethod
    def scan(*, k: int, n: int, m: int) -> int:
        return k * (2 * log2(n) + 3)

    @staticmethod
    def radix(*, k: int, n: int, m: int) -> int:
        return log2(n) * (2 * log2(n) + 16)

    @classmethod
    def topk(cls, **kwargs: Any) -> int:
        return min(cls.scan(**kwargs), cls.radix(**kwargs))


COST_MODELS = [cost_basic, cost_serial, cost_hybrid, cost_parallel]


def _test_knm(*, k: int, n: int, m: int, reps: int) -> Iterable[dict[str, Any]]:
    exact_args = dict(k=k, n=n, m=m)
    yield dict(
        algorithm="exact",
        **exact_args,
        recall_model=1.0,
        recall_simulation=1.0,
        recall_simulation_n=reps,
        recall_simulation_std=0.0,
        **{c.__name__: c.topk(**exact_args) for c in COST_MODELS},
    )
    for b in [2**i for i in range(1, int(log2(n)) + 1)]:
        for k_b in [2**i for i in range(0, int(log2(k)) + 1)]:
            if b * k_b >= k and (k_b <= n // b):
                approx_args = dict(**exact_args, b=b, k_b=k_b)
                sim_mean, sim_std = recall.simulation(**approx_args, reps=reps)
                yield dict(
                    algorithm="approx",
                    **approx_args,
                    recall_model=recall.model(**approx_args),
                    recall_simulation=sim_mean,
                    recall_simulation_n=reps,
                    recall_simulation_std=sim_std,
                    **{c.__name__: c.approx_topk(**approx_args) for c in COST_MODELS},
                )


def _run_test_knm(args: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_test_knm(**args))


if __name__ == "__main__":
    import json
    import multiprocessing
    from pathlib import Path

    import tqdm

    settings = [
        dict(k=k, n=n, m=1, reps=64)
        for n in [2**i for i in range(10, 20 + 1, 2)]
        for k in [2**i for i in range(0, int(log2(n)), 1)]
    ]
    with (
        Path("theoretical_models.jsonl").open("w") as f,
        multiprocessing.Pool() as pool,
    ):
        for results in tqdm.tqdm(
            pool.imap(_run_test_knm, settings, chunksize=1), total=len(settings)
        ):
            for result in results:
                print(json.dumps(result), file=f, flush=True)
