"""E3: The baseline the paper never ran -- isolate the decoder.

hLLM's Amazon Beauty gain (AUC .5122 -> .6168 at the same 0.6B scale) is
credited to the whole package: distillation + LoRA + attention head + Hungarian
decoding. The paper never isolates the last term. Here the score matrix is held
fixed and only the decoder is swapped, so any difference is the solver's own
contribution.

Two decoders on the same M:
  hungarian  optimal assignment of sum_i M[i, pos(i)]      (the paper's)
  argsort    sort items by row mean, O(N log N), no solver

Ground truth mimics Amazon Beauty as reported: one relevant item per slate
(the paper's NDCG@1 and Recall@1 are numerically identical, which forces this).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

RNG = np.random.default_rng(0)
N = 50
SLATES = 2000
SIGNAL = 1.2                      # how well the backbone separates the gold item
SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
W = np.linspace(1.0, 0.1, N)      # position value, monotone decreasing


def slate(sigma):
    gold = RNG.integers(N)
    u = RNG.normal(size=N)
    u[gold] += SIGNAL
    M = np.outer(u, W) + sigma * RNG.normal(size=(N, N))
    return M, gold


def hungarian_order(M):
    _, cols = linear_sum_assignment(M, maximize=True)
    return np.argsort(cols)


def sort_order(M):
    return np.argsort(-M.mean(axis=1))


def ndcg_at(order, gold, k):
    hit = np.where(order[:k] == gold)[0]
    return 0.0 if len(hit) == 0 else 1.0 / np.log2(hit[0] + 2)


def sweep():
    print(f"Rank-1 signal + iid cell noise. N={N}, {SLATES} slates each.\n")
    print(f"{'sigma':>6} | {'NDCG@10':>17} | {'R@1':>15} | {'identical':>9}")
    print(f"{'':>6} | {'hung':>8} {'argsort':>8} | {'hung':>7} {'argsort':>7} |")
    print("-" * 58)
    for sigma in SIGMAS:
        nd = {"h": [], "a": []}
        r1 = {"h": 0, "a": 0}
        same = 0
        for _ in range(SLATES):
            M, gold = slate(sigma)
            oh, oa = hungarian_order(M), sort_order(M)
            same += int(np.array_equal(oh, oa))
            for key, o in (("h", oh), ("a", oa)):
                nd[key].append(ndcg_at(o, gold, 10))
                r1[key] += int(o[0] == gold)
        print(f"{sigma:>6.2f} | {np.mean(nd['h']):>8.4f} {np.mean(nd['a']):>8.4f} | "
              f"{r1['h'] / SLATES:>7.4f} {r1['a'] / SLATES:>7.4f} | {same / SLATES:>8.1%}")


def degenerate_case():
    """If the head scores each item independently (the paper's Linear Probe
    variant), M has no column variation and every permutation is optimal."""
    print("\nItem-independent head: M_ij = u_i, no position variation.")
    objs, spread = [], []
    for _ in range(200):
        u = RNG.normal(size=N)
        M = np.repeat(u[:, None], N, axis=1)
        rows, cols = linear_sum_assignment(M, maximize=True)
        objs.append(M[rows, cols].sum())
        perm = RNG.permutation(N)
        spread.append(abs(M[rows, cols].sum() - M[np.arange(N), perm].sum()))
    print(f"  objective of Hungarian vs a random permutation, max gap: "
          f"{max(spread):.2e}")
    print("  -> every permutation ties; the solver's output carries no ranking "
          "information.")


if __name__ == "__main__":
    sweep()
    degenerate_case()
