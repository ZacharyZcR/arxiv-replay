"""hLLM vs autoregressive decoding on the same backbone, same LoRA, same signal.

This is the ablation the paper never runs. Both modes share Qwen3-0.6B, the
same LoRA config, the same slates and the same supervision (the gold item
belongs at rank 1). Only the output head and the decoder differ:

  ar    generate ordinals left to right, one forward pass per emitted token
  hllm  read an N x K matrix off the prefill hidden states, decode it with the
        Hungarian algorithm in one pass

Everything the paper claims that we can check without a 32B teacher lands here:
the real speed-up, whether quality holds, and how often autoregressive decoding
emits an illegal permutation.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
N_CAND = 50
DEV = "cuda"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_prompt(slate):
    """Return prompt text plus the char offset ending each candidate line."""
    parts = ["Recent purchases:\n"]
    for t in slate["history"]:
        parts.append(f"- {t}\n")
    parts.append("\nCandidates:\n")
    ends = []
    for i, t in enumerate(slate["candidates"]):
        parts.append(f"{i + 1}. {t}\n")
        ends.append(sum(len(p) for p in parts) - 1)
    parts.append("\nBest match:")
    return "".join(parts), ends


def cand_token_index(offsets, ends):
    """Last token whose span ends at or before each candidate line end."""
    idx, j = [], 0
    for e in ends:
        while j + 1 < len(offsets) and offsets[j + 1][1] <= e:
            j += 1
        idx.append(j)
    return idx


class SlotHead(nn.Module):
    """Slot-Query variant: K learnable position embeddings score N items.

    No item-item interaction. Under a causal backbone candidate i cannot see
    candidates i+1..N either, so nothing in this path compares candidates.
    """

    def __init__(self, d, k=N_CAND, inner=512):
        super().__init__()
        self.proj = nn.Linear(d, inner)
        self.norm = nn.LayerNorm(inner)
        self.slots = nn.Parameter(torch.randn(k, inner) * 0.05)

    def forward(self, h):                       # h: (B, N, d)
        return self.norm(self.proj(h)) @ self.slots.T


class AttnHead(nn.Module):
    """Self-Attention (L=2) variant -- the paper's best head.

    Bidirectional attention over the N extracted states, so every candidate is
    compared against every other before the item-position matrix is read out.
    Sized to match the paper's ~4.2M head.
    """

    def __init__(self, d, k=N_CAND, inner=512, layers=2):
        super().__init__()
        self.proj = nn.Linear(d, inner)
        layer = nn.TransformerEncoderLayer(inner, 8, inner * 2, batch_first=True,
                                           dropout=0.0, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(inner)
        self.slots = nn.Parameter(torch.randn(k, inner) * 0.05)

    def forward(self, h):
        return self.norm(self.enc(self.proj(h))) @ self.slots.T


HEADS = {"slot": SlotHead, "attn": AttnHead}


def sinkhorn_log(m, iters=20, tau=0.2):
    z = m / tau
    for _ in range(iters):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    return z


def encode(tok, slate, device=DEV):
    text, ends = build_prompt(slate)
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    idx = cand_token_index(enc["offset_mapping"][0].tolist(), ends)
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device),
            torch.tensor(idx, device=device), text)


def ndcg_at(order, gold, k):
    hit = np.where(np.asarray(order[:k]) == gold)[0]
    return 0.0 if len(hit) == 0 else 1.0 / np.log2(hit[0] + 2)


def train_hllm(model, tok, data, head, epochs, lr, loss_kind="ce", accum=16,
                log_every=200):
    # the head is randomly initialised and needs a larger step than the LoRA
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr},
         {"params": list(head.parameters()), "lr": lr * 3}])
    step, run, t0 = 0, 0.0, time.perf_counter()
    for ep in range(epochs):
        np.random.shuffle(data)
        for slate in data:
            ids, mask, cidx, _ = encode(tok, slate)
            h = model(input_ids=ids, attention_mask=mask,
                      output_hidden_states=True).hidden_states[-1]
            m = head(h[0, cidx].unsqueeze(0).float())
            gold = torch.tensor([slate["gold"]], device=m.device)
            # With one relevant item per slate there is no full target
            # permutation to distil, so the supervision is "gold belongs at
            # rank 1". "ce" states that directly on the rank-1 column; the
            # paper's Sinkhorn relaxation ("sinkhorn") states it through a
            # doubly-stochastic relaxation, whose gradient is weak while alpha
            # is still near-uniform.
            loss = 0.0
            if loss_kind in ("ce", "both"):
                loss = loss + F.cross_entropy(m[0, :, 0].unsqueeze(0), gold)
            if loss_kind in ("sinkhorn", "both"):
                loss = loss - sinkhorn_log(m)[0, gold.item(), 0]
            (loss / accum).backward()
            step += 1
            if step % accum == 0:
                opt.step()
                opt.zero_grad()
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % log_every == 0:
                print(f"  ep{ep} step {step} loss(ema) {run:.4f} "
                      f"({(time.perf_counter() - t0) / step:.3f}s/step)", flush=True)
    return head


def train_ar(model, tok, data, epochs, lr, accum=16, log_every=200):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    step, run, t0 = 0, 0.0, time.perf_counter()
    for ep in range(epochs):
        np.random.shuffle(data)
        for slate in data:
            text, _ = build_prompt(slate)
            target = f" {slate['gold'] + 1}"
            p_ids = tok(text, return_tensors="pt")["input_ids"].to(DEV)
            t_ids = tok(target, return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(DEV)
            ids = torch.cat([p_ids, t_ids], dim=1)
            # only the target positions need logits; materialising all ~1200 x
            # 151669 of them costs 5x the time and nearly fills the card
            n = t_ids.shape[1]
            logits = model(input_ids=ids, logits_to_keep=n + 1).logits[:, :-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   t_ids.reshape(-1))
            (loss / accum).backward()
            step += 1
            if step % accum == 0:
                opt.step()
                opt.zero_grad()
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % log_every == 0:
                print(f"  ep{ep} step {step} loss(ema) {run:.4f} "
                      f"({(time.perf_counter() - t0) / step:.3f}s/step)", flush=True)


@torch.no_grad()
def eval_hllm(model, tok, head, data, topk):
    """Decode the same matrix two ways.

    hungarian is what the paper proposes. argsort just sorts items by their
    rank-1 column score -- no solver, O(N log N). Any quality difference
    between them is the combinatorial machinery's own contribution, which the
    paper never isolates.
    """
    rows, rows_sort, solver_ms = [], [], []
    for slate in data:
        ids, mask, cidx, _ = encode(tok, slate)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = model(input_ids=ids, attention_mask=mask,
                  output_hidden_states=True).hidden_states[-1]
        m = head(h[0, cidx].unsqueeze(0).float())[0].cpu().numpy()
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        t1 = time.perf_counter()
        # rectangular N x topk: rows holds the selected item indices, which are
        # NOT 0..N-1. Ordering by the assigned column gives the ranking.
        rows_i, cols = linear_sum_assignment(m[:, :topk], maximize=True)
        t_solve = time.perf_counter() - t1
        order = rows_i[np.argsort(cols)]

        t2 = time.perf_counter()
        order_s = np.argsort(-m[:, 0])[:topk]
        t_sort = time.perf_counter() - t2

        solver_ms.append(t_solve * 1e3)
        rows.append((order, slate["gold"], (t_fwd + t_solve) * 1e3, True))
        rows_sort.append((order_s, slate["gold"], (t_fwd + t_sort) * 1e3, True))
    return rows, rows_sort, float(np.mean(solver_ms))


@torch.no_grad()
def eval_ar(model, tok, data, topk):
    """Greedy-decode ordinals one token at a time, exactly as the paper's
    baseline does. Illegal output (repeats / out of range) is counted, not
    repaired."""
    rows = []
    for slate in data:
        text, _ = build_prompt(slate)
        ids = tok(text, return_tensors="pt")["input_ids"].to(DEV)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=topk * 3, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3

        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        nums = [int(x) - 1 for x in re.findall(r"\d+", gen)]
        legal = [n for n in nums if 0 <= n < N_CAND]
        seen, order = set(), []
        for n in legal:
            if n not in seen:
                seen.add(n)
                order.append(n)
        valid = len(order) >= topk and len(nums) == len(legal)
        order += [i for i in range(N_CAND) if i not in seen]   # pad to full
        rows.append((order[:topk], slate["gold"], dt, valid))
    return rows, 0.0


def report(name, rows, topk, solver_ms):
    r1 = np.mean([o[0] == g for o, g, _, _ in rows])
    rk = np.mean([g in o for o, g, _, _ in rows])
    nd = np.mean([ndcg_at(o, g, topk) for o, g, _, _ in rows])
    lat = np.array([d for _, _, d, _ in rows])
    valid = np.mean([v for _, _, _, v in rows])
    print(f"\n{name}")
    print(f"  R@1     {r1:.4f}")
    print(f"  R@{topk:<4} {rk:.4f}")
    print(f"  NDCG@{topk:<2} {nd:.4f}")
    print(f"  latency median {np.median(lat):.1f} ms   mean {lat.mean():.1f} ms")
    if solver_ms:
        print(f"  of which solver: {solver_ms:.4f} ms "
              f"({100 * solver_ms / np.median(lat):.3f}%)")
    print(f"  legal permutation: {valid:.1%}")
    return {"r1": float(r1), f"r@{topk}": float(rk), "ndcg": float(nd),
            "median_ms": float(np.median(lat)), "legal": float(valid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ar", "hllm"], required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-n", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--head", choices=["slot", "attn"], default="attn")
    ap.add_argument("--loss", choices=["ce", "sinkhorn", "both"], default="ce")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--out", default="")
    ap.add_argument("--save", default="", help="dir to save LoRA + head for later probing")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mode={args.mode}  head={args.head}  loss={args.loss}  "
          f"LoRA trainable {trainable / 1e6:.2f} M")

    train = load("data/train.jsonl")
    ev = load("data/eval.jsonl")
    if args.train_n:
        train = train[:args.train_n]
    if args.eval_n:
        ev = ev[:args.eval_n]
    print(f"train {len(train)}  eval {len(ev)}  topk {args.topk}")

    if args.mode == "hllm":
        head = HEADS[args.head](model.config.hidden_size).to(DEV)
        print(f"head params {sum(p.numel() for p in head.parameters()) / 1e6:.2f} M")
        model.train()
        train_hllm(model, tok, train, head, args.epochs, args.lr, args.loss,
                   args.accum)
        model.eval()
        head.eval()
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            model.save_pretrained(args.save)
            torch.save(head.state_dict(), f"{args.save}/head.pt")
            print(f"saved to {args.save}")
        rows, rows_sort, solver = eval_hllm(model, tok, head, ev, args.topk)
    else:
        model.train()
        train_ar(model, tok, train, args.epochs, args.lr, args.accum)
        model.eval()
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            model.save_pretrained(args.save)
            print(f"saved to {args.save}")
        rows, solver = eval_ar(model, tok, ev, args.topk)
        rows_sort = None

    res = report(f"{args.mode} + hungarian (topk={args.topk})", rows, args.topk, solver)
    if args.mode == "hllm":
        res_sort = report(f"{args.mode} + argsort (same matrix, no solver)",
                          rows_sort, args.topk, 0.0)
        res["argsort"] = res_sort
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": args.mode, "head": getattr(args, "head", None),
                       "loss": getattr(args, "loss", None),
                       "topk": args.topk, **res}, f, indent=2)


if __name__ == "__main__":
    main()
