"""E2: When is the Hungarian step doing anything a sort could not?

hLLM's novelty claim rests on ranking being a bipartite assignment problem.
But if the learned score matrix factorises as M_ij = s_i * w_j (item quality
times position bias), the rearrangement inequality makes the optimal assignment
identical to sorting items by s -- O(N log N), no solver, no permutation
machinery. So the claim only has content when the matrix carries genuine
item-position interaction. We sweep interaction strength and measure when the
two decoders diverge.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau

RNG = np.random.default_rng(0)
N = 50
TRIALS = 300
EPSILONS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def build(n, eps):
    """M = rank-1 (item quality x monotone position bias) + interaction noise."""
    s = RNG.normal(size=n)
    w = np.linspace(1.0, 0.1, n)            # earlier positions worth more
    return np.outer(s, w) + eps * RNG.normal(size=(n, n)), s


def rank_of_items(perm_cols):
    """perm_cols[i] = position assigned to item i -> ranking order of items."""
    return np.argsort(perm_cols)


def main():
    print(f"N={N}, {TRIALS} trials per point\n")
    print(f"{'eps':>6} {'kendall_tau':>12} {'exact_match':>12} {'top1_match':>11} "
          f"{'obj_gain_%':>11}")
    for eps in EPSILONS:
        taus, exact, top1, gains = [], 0, 0, []
        for _ in range(TRIALS):
            M, s = build(N, eps)

            rows, cols = linear_sum_assignment(M, maximize=True)
            hung_order = rank_of_items(cols)

            sort_order = np.argsort(-s)      # the O(N log N) shortcut
            sort_cols = np.empty(N, dtype=int)
            sort_cols[sort_order] = np.arange(N)

            taus.append(kendalltau(hung_order, sort_order).statistic)
            exact += int(np.array_equal(hung_order, sort_order))
            top1 += int(hung_order[0] == sort_order[0])

            hung_obj = M[rows, cols].sum()
            sort_obj = M[np.arange(N), sort_cols].sum()
            gains.append(100 * (hung_obj - sort_obj) / abs(sort_obj))

        print(f"{eps:>6.2f} {np.mean(taus):>12.4f} {exact / TRIALS:>11.1%} "
              f"{top1 / TRIALS:>10.1%} {np.mean(gains):>11.3f}")

    print("\neps=0 is the rank-1 case: assignment provably reduces to argsort.")


if __name__ == "__main__":
    main()
