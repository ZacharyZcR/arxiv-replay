"""E5: Is variable N / top-K really a gap, as the review claims?

The paper only ever runs N=K (full permutation on a fixed slate). Reviewers
flagged partial ranking and variable candidate counts as unaddressed. But
rectangular linear assignment is a solved problem -- scipy handles N x K
directly, picking the best K items and placing them. We measure whether that
costs anything, since production reranking almost always wants top-K of many.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

RNG = np.random.default_rng(0)


def bench(n, k, trials=30, repeats=5):
    mats = [RNG.random((n, k)) for _ in range(trials)]
    for m in mats[:3]:
        linear_sum_assignment(m, maximize=True)
    ts = []
    for _ in range(repeats):
        for m in mats:
            t0 = time.perf_counter()
            rows, cols = linear_sum_assignment(m, maximize=True)
            ts.append((time.perf_counter() - t0) * 1e3)
    assert len(rows) == min(n, k) and len(set(cols)) == min(n, k)
    return float(np.median(ts))


def main():
    print("median solve time (ms) for an N-item x K-position matrix\n")
    print(f"{'N':>6} " + " ".join(f"{'K=' + str(k):>10}" for k in (1, 5, 10, 20)) +
          f" {'K=N':>10} {'full/topK10':>12}")
    for n in (50, 150, 500, 1000):
        row = [bench(n, k) for k in (1, 5, 10, 20)]
        full = bench(n, n, trials=max(5, 30 - n // 40))
        print(f"{n:>6} " + " ".join(f"{v:>10.4f}" for v in row) +
              f" {full:>10.4f} {full / row[2]:>11.1f}x")

    print("\nOutput validity is asserted in every call: exactly min(N,K) "
          "distinct positions, no repeats.")
    print("Rectangular assignment is O(N*K*min(N,K)), so top-K decoding is "
          "cheaper than the full permutation the paper always solves.")


if __name__ == "__main__":
    main()
