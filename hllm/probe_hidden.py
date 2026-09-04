"""Diagnostic: is the gold item even linearly recoverable from the backbone?

hLLM reads its score matrix off frozen prefill hidden states. If a plain linear
probe on those states cannot find the gold item, no head or solver on top of
them will either, and the problem is upstream of anything the paper proposes.

Extract once with the backbone frozen, then try heads cheaply.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_rerank as L


@torch.no_grad()
def extract(split, limit):
    tok = AutoTokenizer.from_pretrained(L.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        L.MODEL, dtype=torch.bfloat16).to(L.DEV).eval()
    data = L.load(f"data/{split}.jsonl")[:limit]
    H = np.empty((len(data), L.N_CAND, model.config.hidden_size), dtype=np.float16)
    g = np.empty(len(data), dtype=np.int64)
    t0 = time.perf_counter()
    for i, slate in enumerate(data):
        ids, mask, cidx, _ = L.encode(tok, slate)
        h = model(input_ids=ids, attention_mask=mask,
                  output_hidden_states=True).hidden_states[-1]
        H[i] = h[0, cidx].float().cpu().numpy().astype(np.float16)
        g[i] = slate["gold"]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(data)}  {(time.perf_counter() - t0) / (i + 1):.3f}s/slate",
                  flush=True)
    np.save(f"data/h_{split}.npy", H)
    np.save(f"data/g_{split}.npy", g)
    print(f"saved {H.shape} to data/h_{split}.npy")


class Linear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, h):
        return self.f(h).squeeze(-1)


class Attn(nn.Module):
    """Two bidirectional layers, so candidates can compare against each other."""

    def __init__(self, d, inner=512, layers=2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, inner))
        layer = nn.TransformerEncoderLayer(inner, 8, inner * 2, batch_first=True,
                                           dropout=0.0, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(inner, 1)

    def forward(self, h):
        return self.out(self.enc(self.proj(h))).squeeze(-1)


def probe(kind, epochs, lr, batch=64):
    Htr = torch.from_numpy(np.load("data/h_train.npy")).float()
    gtr = torch.from_numpy(np.load("data/g_train.npy"))
    Hev = torch.from_numpy(np.load("data/h_eval.npy")).float().to(L.DEV)
    gev = torch.from_numpy(np.load("data/g_eval.npy")).to(L.DEV)

    d = Htr.shape[-1]
    net = (Linear(d) if kind == "linear" else Attn(d)).to(L.DEV)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    for ep in range(epochs):
        perm = torch.randperm(len(Htr))
        tot = 0.0
        for b in range(0, len(Htr), batch):
            sel = perm[b:b + batch]
            s = net(Htr[sel].to(L.DEV))
            loss = F.cross_entropy(s, gtr[sel].to(L.DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(sel)
        with torch.no_grad():
            se = net(Hev)
            order = se.argsort(dim=1, descending=True)
            r1 = (order[:, 0] == gev).float().mean().item()
            r10 = (order[:, :10] == gev[:, None]).any(1).float().mean().item()
        print(f"  ep{ep} train_loss {tot / len(Htr):.4f}  "
              f"eval R@1 {r1:.4f}  R@10 {r10:.4f}")
    print(f"{kind} probe: {n_par / 1e6:.2f}M params, random baseline R@1=0.02 R@10=0.20")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--kind", choices=["linear", "attn"], default="linear")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    if a.extract:
        extract("train", a.train_n)
        extract("eval", 10**9)
    else:
        probe(a.kind, a.epochs, a.lr)
