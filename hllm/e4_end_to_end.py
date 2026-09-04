"""E4: The ablation the paper is missing -- head architecture vs decoder.

hLLM couples two changes: a head emitting an N x K item-position matrix, and a
Hungarian decoder that reads a permutation off it. The paper reports the pair
together and never separates them, so its Amazon Beauty gain cannot be
attributed. We train both heads on identical data with a shared encoder
architecture and equal budget, then decode the matrix head four ways:

  A  pointwise + argsort     no matrix, no solver
  B  matrix + hungarian      hLLM as published
  C  matrix + argsort        same trained matrix, solver removed (row mean)
  D  matrix + top-column     same matrix, ranked by fitness for position 0
  E  matrix + greedy         legal permutation, greedy not optimal
                             (the "repair" style decoding hLLM argues against)

B vs C/D asks whether the solver earns anything over reading the matrix
directly. B vs E asks whether *optimal* assignment beats a cheap legal one.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_rel

torch.set_num_threads(4)

N, D_IN, D = 20, 16, 64
TRAIN, TEST = 8000, 2000
EPOCHS, BATCH = 20, 128
REL_NOISE = float(os.environ.get("REL_NOISE", "0.1"))
SEEDS = [0, 1, 2]


def make_data(rng, n_slates, w):
    """Relevance depends on the item AND the slate it sits in, so a
    context-free scorer cannot solve it -- this is what reranking means."""
    x = rng.normal(size=(n_slates, N, D_IN)).astype(np.float32)
    contrast = np.linalg.norm(x - x.mean(axis=1, keepdims=True), axis=-1)
    rel = x @ w + 0.8 * contrast + REL_NOISE * rng.normal(size=(n_slates, N))
    return torch.from_numpy(x), torch.from_numpy(rel.astype(np.float32))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_IN, D)
        layer = nn.TransformerEncoderLayer(D, 4, 128, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, 2)

    def forward(self, x):
        return self.enc(self.proj(x))


class Pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.out = Encoder(), nn.Linear(D, 1)

    def forward(self, x):
        return self.out(self.enc(x)).squeeze(-1)


class MatrixHead(nn.Module):
    """Slot-Query variant: learnable position embeddings score every item."""

    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.slots = nn.Parameter(torch.randn(N, D) * 0.02)

    def forward(self, x):
        return self.enc(x) @ self.slots.T / D ** 0.5


def sinkhorn_log(m, iters=20, tau=0.5):
    z = m / tau
    for _ in range(iters):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    return z


def perm_target(rel):
    order = torch.argsort(rel, dim=1, descending=True)
    p = torch.zeros(rel.shape[0], N, N)
    p.scatter_(1, order.unsqueeze(-1).expand(-1, -1, N),
               torch.eye(N).expand(rel.shape[0], -1, -1))
    return p


def train(model, x, rel, kind):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    tgt = perm_target(rel) if kind == "matrix" else F.softmax(rel, dim=1)
    for _ in range(EPOCHS):
        idx = torch.randperm(len(x))
        for b in range(0, len(x), BATCH):
            sel = idx[b:b + BATCH]
            out = model(x[sel])
            if kind == "matrix":
                loss = -(tgt[sel] * sinkhorn_log(out)).sum(dim=(1, 2)).mean()
            else:
                loss = -(tgt[sel] * F.log_softmax(out, dim=1)).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def greedy_order(m):
    """Fill positions left to right, taking the best item still unused."""
    n = m.shape[0]
    taken = np.zeros(n, dtype=bool)
    out = np.empty(n, dtype=int)
    for j in range(n):
        col = np.where(taken, -np.inf, m[:, j])
        out[j] = int(np.argmax(col))
        taken[out[j]] = True
    return out


def ndcg(order, rel, k):
    """Per-slate NDCG@k, so configs can be compared as paired samples."""
    g = np.take_along_axis(rel, order, axis=1)[:, :k]
    disc = 1.0 / np.log2(np.arange(k) + 2)
    ideal = -np.sort(-rel, axis=1)[:, :k]
    return (g * disc).sum(1) / (ideal * disc).sum(1)


def run_seed(seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=D_IN).astype(np.float32)
    xtr, rtr = make_data(rng, TRAIN, w)
    xte, rte = make_data(rng, TEST, w)
    rel = rte.numpy()

    pw = train(Pointwise(), xtr, rtr, "pointwise")
    mh = train(MatrixHead(), xtr, rtr, "matrix")
    with torch.no_grad():
        s, M = pw(xte).numpy(), mh(xte).numpy()

    res, lat, obj = {}, {}, {}

    t0 = time.perf_counter()
    res["A pointwise + argsort"] = np.argsort(-s, axis=1)
    lat["A pointwise + argsort"] = (time.perf_counter() - t0) / TEST * 1e3

    t0 = time.perf_counter()
    hung = np.empty((TEST, N), dtype=int)
    for i in range(TEST):
        _, cols = linear_sum_assignment(M[i], maximize=True)
        hung[i] = np.argsort(cols)
    res["B matrix + hungarian"] = hung
    lat["B matrix + hungarian"] = (time.perf_counter() - t0) / TEST * 1e3

    t0 = time.perf_counter()
    res["C matrix + argsort"] = np.argsort(-M.mean(axis=2), axis=1)
    lat["C matrix + argsort"] = (time.perf_counter() - t0) / TEST * 1e3

    t0 = time.perf_counter()
    res["D matrix + top-column"] = np.argsort(-M[:, :, 0], axis=1)
    lat["D matrix + top-column"] = (time.perf_counter() - t0) / TEST * 1e3

    t0 = time.perf_counter()
    res["E matrix + greedy"] = np.stack([greedy_order(M[i]) for i in range(TEST)])
    lat["E matrix + greedy"] = (time.perf_counter() - t0) / TEST * 1e3

    # assignment objective actually attained, to confirm B is the optimum
    for name, o in res.items():
        if name.startswith("A"):
            continue
        pos = np.argsort(o, axis=1)
        obj[name] = float(np.mean(M[np.arange(TEST)[:, None], np.arange(N)[None, :], pos]
                                  .sum(axis=1)))

    gold = np.argmax(rel, axis=1)
    rows, per_slate = {}, {}
    for name, o in res.items():
        nd10 = ndcg(o, rel, 10)
        per_slate[name] = nd10
        rows[name] = (nd10.mean(), ndcg(o, rel, 5).mean(),
                      float(np.mean(o[:, 0] == gold)), lat[name], obj.get(name, np.nan))
    agree = float(np.mean((res["B matrix + hungarian"] == res["E matrix + greedy"]).all(axis=1)))
    return rows, agree, per_slate


def main():
    all_rows, agrees, pooled = [], [], {}
    for seed in SEEDS:
        print(f"seed {seed} ...", flush=True)
        r, a, ps = run_seed(seed)
        all_rows.append(r)
        agrees.append(a)
        for k, v in ps.items():
            pooled.setdefault(k, []).append(v)
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}

    names = list(all_rows[0])
    print(f"\n{len(SEEDS)} seeds, N={N}, {TRAIN} train / {TEST} test slates, "
          f"{EPOCHS} epochs, relevance noise {REL_NOISE}\n")
    print(f"{'config':>23} {'NDCG@10':>15} {'NDCG@5':>15} {'R@1':>14} "
          f"{'decode_ms':>10} {'assign_obj':>11}")
    for name in names:
        v = np.array([[*all_rows[s][name]] for s in range(len(SEEDS))])
        m, sd = v.mean(0), v.std(0)
        print(f"{name:>23} {m[0]:>8.4f}+-{sd[0]:<6.4f} {m[1]:>7.4f}+-{sd[1]:<6.4f} "
              f"{m[2]:>7.4f}+-{sd[2]:<5.4f} {m[3]:>10.4f} {m[4]:>11.3f}")
    print(f"\nB and E decode to the identical permutation on "
          f"{np.mean(agrees):.1%} of slates")

    # seed-level std is larger than the gaps above, so compare per-slate and
    # paired: every config ranked the same test slates.
    print(f"\npaired NDCG@10 over {len(pooled['B matrix + hungarian'])} slates "
          f"(all seeds pooled)")
    ref = "B matrix + hungarian"
    for name in names:
        if name == ref:
            continue
        d = pooled[ref] - pooled[name]
        t, pval = ttest_rel(pooled[ref], pooled[name])
        print(f"  {ref[:1]} - {name[:1]}: delta {d.mean():+.4f} "
              f"[95% CI {d.mean() - 1.96 * d.std() / len(d) ** .5:+.4f}, "
              f"{d.mean() + 1.96 * d.std() / len(d) ** .5:+.4f}]  "
              f"t={t:>8.2f}  p={pval:.2e}")


if __name__ == "__main__":
    main()
