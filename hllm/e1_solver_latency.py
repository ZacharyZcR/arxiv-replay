"""E1: Does the Hungarian solver really cost 0.008 ms at N=50?

The paper (Sec 5.5) reports a single point: LAPJV at N=50 costs 0.008 ms on CPU,
under 0.03% of the 28 ms end-to-end latency. It never shows how that scales,
even though its own internal dataset goes up to N=150. We measure the curve.

This uses SciPy's modified-JV solver; e1b compares dedicated LAPJV
implementations, which are ~2.5x faster and reproduce the paper's number.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

RNG = np.random.default_rng(0)
SIZES = [10, 25, 50, 100, 150, 250, 500, 1000, 2000]


def score_matrix(n, kind):
    """Cost matrices the solver might actually see."""
    if kind == "uniform":
        return RNG.random((n, n))
    if kind == "lowrank":
        # item quality x position bias, plus interaction noise -- closer to a
        # learned item-position affinity matrix than iid uniform
        s = RNG.normal(size=(n, 1))
        w = RNG.normal(size=(1, n))
        return s @ w + 0.3 * RNG.normal(size=(n, n))
    raise ValueError(kind)


def bench(fn, mats, repeats):
    for m in mats[:3]:
        fn(m)  # warmup
    times = []
    for _ in range(repeats):
        for m in mats:
            t0 = time.perf_counter()
            fn(m)
            times.append((time.perf_counter() - t0) * 1e3)
    return np.array(times)


def main():
    print(f"{'N':>6} {'kind':>8} {'hungarian_ms':>14} {'p99_ms':>9} "
          f"{'argsort_ms':>11} {'ratio':>7} {'%of 28ms':>9}")
    for kind in ("uniform", "lowrank"):
        for n in SIZES:
            trials = max(3, min(50, 200_000 // (n * n)))
            mats = [score_matrix(n, kind) for _ in range(trials)]
            repeats = max(1, 2000 // (trials * max(1, n // 25)))

            hung = bench(lambda m: linear_sum_assignment(m, maximize=True), mats, repeats)
            # the degenerate alternative: if the matrix were rank-1, sorting by
            # row score would give the same permutation at O(N log N)
            sort = bench(lambda m: np.argsort(-m.sum(axis=1)), mats, repeats)

            med, p99 = np.median(hung), np.percentile(hung, 99)
            print(f"{n:>6} {kind:>8} {med:>14.4f} {p99:>9.4f} "
                  f"{np.median(sort):>11.4f} {med / np.median(sort):>7.1f}x "
                  f"{100 * med / 28:>8.3f}%")


if __name__ == "__main__":
    main()
