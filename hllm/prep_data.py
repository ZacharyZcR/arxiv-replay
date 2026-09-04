"""Build Amazon Beauty rerank slates in the paper's shape.

The paper reports 21245 train / 1118 eval slates, 50 candidates each, with
NDCG@1 numerically equal to Recall@1 -- which only holds when a slate contains
exactly one relevant item. So: leave-one-out per user, the held-out item plus
49 sampled negatives, shuffled. Our 5-core dump has 9880 users rather than the
paper's ~22k, so slate counts are smaller; the structure is the same.
"""
import html
import json
import re
import numpy as np
import pandas as pd

N_CAND = 50
MAX_HIST = 5
TITLE_WORDS = 16
EVAL_FRAC = 0.05
SEED = 0


def title_of(text):
    m = re.search(r"###Title###\s*\n(.+?)(?:\n\n###|$)", text, re.S)
    t = html.unescape(m.group(1) if m else text).strip().replace("\n", " ")
    return " ".join(t.split()[:TITLE_WORDS])


def main():
    rng = np.random.default_rng(SEED)
    reviews = pd.read_parquet("data/reviews_5core.parquet")
    meta = pd.read_parquet("data/metadata_5core.parquet")

    titles = {a: title_of(t) for a, t in zip(meta.asin, meta.text)}
    all_items = np.array([a for a in meta.asin if titles[a]])
    print(f"{len(all_items)} items with titles, {len(reviews)} users")

    slates = []
    for _, row in reviews.iterrows():
        seq = [a for a in row.asin if a in titles]
        if len(seq) < 2:
            continue
        hist, gold_item = seq[:-1][-MAX_HIST:], seq[-1]
        seen = set(seq)

        negs = []
        while len(negs) < N_CAND - 1:
            pick = all_items[rng.integers(len(all_items), size=N_CAND)]
            negs += [a for a in pick if a not in seen and a not in negs]
        cands = negs[:N_CAND - 1] + [gold_item]
        rng.shuffle(cands)

        slates.append({
            "history": [titles[a] for a in hist],
            "candidates": [titles[a] for a in cands],
            "gold": cands.index(gold_item),
        })

    rng.shuffle(slates)
    n_eval = int(len(slates) * EVAL_FRAC)
    for name, part in (("eval", slates[:n_eval]), ("train", slates[n_eval:])):
        with open(f"data/{name}.jsonl", "w") as f:
            for s in part:
                f.write(json.dumps(s) + "\n")
        print(f"{name}: {len(part)} slates")

    ex = slates[0]
    print(f"\nexample: hist={len(ex['history'])} cands={len(ex['candidates'])} "
          f"gold={ex['gold']}")
    print(f"  hist[0]: {ex['history'][0][:70]}")
    print(f"  cand[gold]: {ex['candidates'][ex['gold']][:70]}")


if __name__ == "__main__":
    main()
