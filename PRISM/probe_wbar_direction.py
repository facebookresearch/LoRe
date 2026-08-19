"""
What does the learned shared reward direction prefer, concretely?

interpret_wbar.py showed the population direction is nearly orthogonal to Skywork's reward head
and barely aligned with the 11-concept library. This script asks the question directly: score real
responses by their projection onto that direction and look at what comes out on top.

Two parts:

1. SURFACE CORRELATES, checked first and deliberately. This project has already been burned once by
   a reward direction that turned out to be measuring a formatting artifact rather than preference
   (chosen rendered as prose, rejected as a Python list-repr). So before reading any examples, the
   direction is regressed against cheap surface features -- response length, newlines, list markers,
   markdown, digits -- with the pretrained head as a reference. A direction that is mostly length
   would show up here and nowhere else.

2. WITHIN-PROMPT EXTREMES. For each turn, all responses share a prompt, so ranking within a turn
   controls for topic. Prints the turns where the direction most strongly prefers one response over
   another, and the turns where it most disagrees with the pretrained head.

CPU-only.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoReV2                                             # noqa: E402
from community_alignment_lore import build_user_diffs, unit          # noqa: E402
from embed_community_alignment import text_key                       # noqa: E402
from rm_head_utils import load_reward_head                           # noqa: E402


def surface_features(text):
    return {
        "chars": len(text),
        "words": len(text.split()),
        "newlines": text.count("\n"),
        "list_markers": len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", text)),
        "markdown_bold": text.count("**"),
        "digits": sum(c.isdigit() for c in text),
        "questions": text.count("?"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=4)
    ap.add_argument("--chars", type=int, default=420, help="chars of each response to print")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}

    users = build_user_diffs(pairs, emb_lookup, 50, 0.2, gen, split_by_turn=True, val_frac=0.2)
    uids = sorted(users)
    m = LoReV2(len(uids), 4096, args.rank, lam_pop=args.lam_pop, lam_d=args.lam_d,
               num_iterations=args.iters, learning_rate=args.lr, verbose=False)
    m.train([users[u][0] for u in uids], val=[users[u][1] for u in uids])
    head = unit(load_reward_head().reshape(-1).float())
    wbar = unit((m.V.detach() @ m.wbar.detach()).float())
    if float(wbar @ head) < 0:
        wbar = -wbar

    # ---- score every distinct (prompt, response) we have an embedding for ----
    texts = {}
    for p in pairs:
        for side in ("chosen", "rejected"):
            k = text_key(p["prompt"], p[side])
            if k in emb_lookup and k not in texts:
                texts[k] = (p["prompt"], p[side], (p["conversation_id"], p["turn"]))
    keys = list(texts)
    E = torch.stack([emb_lookup[k].float() for k in keys])
    s_w = (E @ wbar).numpy()
    s_h = (E @ head).numpy()
    print(f"{len(keys)} distinct (prompt, response) items scored\n")

    # ---- 1. surface correlates ----
    feats = [surface_features(texts[k][1]) for k in keys]
    names = list(feats[0])
    print("=== surface correlates (Pearson r with the response score) ===")
    print("a direction that is mostly length/formatting shows up here\n")
    print(f"{'feature':<14} | {'wbar':>8} | {'base_head':>9}")
    print("-" * 38)
    for n in names:
        v = np.array([f[n] for f in feats], dtype=float)
        if v.std() < 1e-9:
            continue
        print(f"{n:<14} | {np.corrcoef(v, s_w)[0,1]:>+8.3f} | {np.corrcoef(v, s_h)[0,1]:>+9.3f}")
    print(f"\n{'agreement':<14} | corr(wbar, base_head) over responses = "
          f"{np.corrcoef(s_w, s_h)[0,1]:+.3f}")

    # ---- 2. within-prompt extremes ----
    by_turn = defaultdict(list)
    for i, k in enumerate(keys):
        by_turn[texts[k][2]].append(i)
    multi = {t: ix for t, ix in by_turn.items() if len(ix) >= 2}

    def show(title, ranked, hi_key, lo_key):
        print(f"\n\n{'='*78}\n{title}\n{'='*78}")
        for t, _ in ranked[:args.n_examples]:
            ix = multi[t]
            hi = max(ix, key=hi_key); lo = min(ix, key=lo_key)
            prompt = texts[keys[hi]][0]
            print(f"\nPROMPT: {prompt[:200]}")
            for tag, j in (("PREFERRED", hi), ("DISPREFERRED", lo)):
                body = " ".join(texts[keys[j]][1].split())
                print(f"\n  [{tag}]  wbar {s_w[j]:+.2f}   head {s_h[j]:+.2f}   "
                      f"({len(texts[keys[j]][1])} chars)")
                print(f"    {body[:args.chars]}{'...' if len(body) > args.chars else ''}")
            print("\n" + "-" * 78)

    spread = sorted(((t, s_w[ix].max() - s_w[ix].min()) for t, ix in multi.items()),
                    key=lambda x: -x[1])
    show("A. Turns where the learned direction most strongly prefers one response",
         spread, lambda j: s_w[j], lambda j: s_w[j])

    # disagreement: wbar's favourite is the head's least favourite
    def disagree(t):
        ix = multi[t]
        best_w = max(ix, key=lambda j: s_w[j])
        rank_h = sorted(ix, key=lambda j: s_h[j])
        return rank_h.index(best_w) / max(1, len(ix) - 1)   # 0 = head hates wbar's pick
    dis = sorted(((t, disagree(t)) for t in multi), key=lambda x: x[1])
    show("B. Turns where the two directions disagree most "
         "(the learned direction's pick is the pretrained head's least favourite)",
         dis, lambda j: s_w[j], lambda j: s_w[j])


if __name__ == "__main__":
    main()
