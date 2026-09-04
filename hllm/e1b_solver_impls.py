"""E1b: Is the paper's 0.008 ms at N=50 reachable, and by which implementation?

Sec 5.5 reports LAPJV at N=50 costing 0.008 ms on CPU. SciPy's modified-JV
solver is 5x slower than that on our machine, so we check whether a dedicated
LAPJV C++ implementation closes the gap -- and how each scales past N=50.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

import lap
import lapjv as lapjv_mod

RNG = np.random.default_rng(0)
SIZES = [50, 100, 150, 250, 500, 1000]

SOLVERS = {
    "scipy_mJV": lambda C: linear_sum_assignment(C),
    "lap_JV": lambda C: lap.lapjv(C),
    "lapjv_cpp": lambda C: lapjv_mod.lapjv(C),
}


def bench(fn, mats, repeats):
    for m in mats[:3]:
        fn(m.copy())
    ts = []
    for _ in range(repeats):
        for m in mats:
            c = m.copy()
            t0 = time.perf_counter()
            fn(c)
            ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    print("median solve time (ms), cost-minimisation on iid uniform matrices\n")
    print(f"{'N':>6} " + " ".join(f"{k:>11}" for k in SOLVERS) + f" {'paper':>8}")
    for n in SIZES:
        trials = max(3, min(40, 100_000 // (n * n)))
        mats = [RNG.random((n, n)) for _ in range(trials)]
        repeats = max(1, 1500 // (trials * max(1, n // 50)))
        row = [bench(f, mats, repeats) for f in SOLVERS.values()]
        paper = "0.008" if n == 50 else "-"
        print(f"{n:>6} " + " ".join(f"{v:>11.4f}" for v in row) + f" {paper:>8}")

    print("\nFastest implementation as a share of the paper's 28 ms budget:")
    for n in SIZES:
        mats = [RNG.random((n, n)) for _ in range(max(3, min(40, 100_000 // (n * n))))]
        best = min(bench(f, mats, max(1, 1500 // (len(mats) * max(1, n // 50))))
                   for f in SOLVERS.values())
        print(f"  N={n:>5}: {best:>8.4f} ms  ({100 * best / 28:>7.3f}% of 28 ms)")


if __name__ == "__main__":
    main()
