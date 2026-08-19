"""
Behavioural concept alignment: does the learned reward direction RANK responses the way each
concept probe does?

interpret_wbar.py measured cosine between direction vectors. That is the wrong test in this space.
Embeddings occupy a narrow cone rather than filling 4096 dimensions, so two directions can be
near-orthogonal as vectors (cos = 0.039 for wbar vs the pretrained head) while ranking real data
almost identically (r = 0.777 over actual responses). Vector angle does not answer the
interpretability question; score correlation does.

Two views, both reported:

  RESPONSE view  -- correlate scores over individual (prompt, response) items. "Does this direction
                    like the same responses a helpfulness probe likes?" Includes prompt-driven
                    variance, since every response carries its prompt's embedding.

  DIFF view      -- correlate scores over (chosen - rejected) differences. This is the
                    decision-relevant quantity: pairwise accuracy depends only on the sign of the
                    difference, and the shared prompt cancels. Where the two views disagree, the
                    diff view is the one that bears on why the direction wins.

Null: the 95th percentile of |r| between the direction and random unit directions scored on the
same data, which absorbs whatever correlation the cone geometry induces by itself.

CPU-only.
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoReV2                                             # noqa: E402
from community_alignment_lore import build_user_diffs, unit          # noqa: E402
from embed_community_alignment import text_key                       # noqa: E402
from rm_head_utils import load_reward_head                           # noqa: E402


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--concept_vectors", default=os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                                              "concept_vectors.pt"))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_null", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results",
                                                  "community_alignment", "concept_score_align.csv"))
    args = ap.parse_args()

    cvs = torch.load(args.concept_vectors, weights_only=False)
    concepts = list(cvs)
    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    keys_all = list(blob["keys"])
    kidx = {k: i for i, k in enumerate(keys_all)}
    E = blob["emb"].float()
    emb_lookup = {k: E[i] for i, k in enumerate(keys_all)}
    head = unit(load_reward_head().reshape(-1).float())

    # index pairs once so diff scores can be taken from response scores directly
    ci, ri = [], []
    for p in pairs:
        kc, kr = text_key(p["prompt"], p["chosen"]), text_key(p["prompt"], p["rejected"])
        if kc in kidx and kr in kidx:
            ci.append(kidx[kc]); ri.append(kidx[kr])
    ci, ri = np.array(ci), np.array(ri)
    print(f"{E.shape[0]} responses | {len(ci)} pairs | {len(concepts)} concepts\n")

    C = torch.stack([unit(cvs[c].float()) for c in concepts], dim=1)      # [4096, n_concepts]
    S_c = (E @ C).numpy()                                                 # concept scores
    s_h = (E @ head).numpy()

    # null: random unit directions scored on the same data
    g = torch.Generator().manual_seed(0)
    R = F.normalize(torch.randn(4096, args.n_null, generator=g), dim=0)
    S_r = (E @ R).numpy()

    rows, per_seed = [], {c: {"resp": [], "diff": []} for c in concepts}
    head_row = {}
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        users = build_user_diffs(pairs, emb_lookup, 50, 0.2, gen, split_by_turn=True, val_frac=0.2)
        uids = sorted(users)
        m = LoReV2(len(uids), 4096, args.rank, lam_pop=args.lam_pop, lam_d=args.lam_d,
                   num_iterations=args.iters, learning_rate=args.lr, verbose=False)
        m.train([users[u][0] for u in uids], val=[users[u][1] for u in uids])
        wbar = unit((m.V.detach() @ m.wbar.detach()).float())
        if float(wbar @ head) < 0:
            wbar = -wbar
        s_w = (E @ wbar).numpy()
        d_w = s_w[ci] - s_w[ri]
        for j, c in enumerate(concepts):
            per_seed[c]["resp"].append(r(s_w, S_c[:, j]))
            per_seed[c]["diff"].append(r(d_w, S_c[ci, j] - S_c[ri, j]))
        if not head_row:
            for j, c in enumerate(concepts):
                head_row[c] = (r(s_h, S_c[:, j]),
                               r(s_h[ci] - s_h[ri], S_c[ci, j] - S_c[ri, j]))
        print(f"seed {seed}: corr(wbar, head) resp {r(s_w, s_h):+.3f} | "
              f"diff {r(d_w, s_h[ci] - s_h[ri]):+.3f}")

    # null PER CONCEPT: each concept vector sits differently relative to the embedding cone, so a
    # single shared threshold would be wrong. For each concept, the 95th percentile of |r| between
    # that concept's scores and n_null random directions' scores.
    # Report the null DISTRIBUTION, not just its 95th percentile. A direction can sit far above the
    # random median -- i.e. genuinely more concept-aligned than chance -- while still failing a
    # 5% threshold, and reporting only "below p95" would call that "not aligned", which is wrong.
    null_resp_c, null_diff_c, null_med_c, null_dist_c = {}, {}, {}, {}
    for j, c in enumerate(concepts):
        dc = S_c[ci, j] - S_c[ri, j]
        nr = np.array([abs(r(S_r[:, k], S_c[:, j])) for k in range(args.n_null)])
        nd = np.array([abs(r(S_r[ci, k] - S_r[ri, k], dc)) for k in range(args.n_null)])
        null_resp_c[c] = float(np.quantile(nr, 0.95))
        null_diff_c[c] = float(np.quantile(nd, 0.95))
        null_med_c[c] = float(np.median(nd))
        null_dist_c[c] = nd
    print(f"\nnull |r| computed per concept vs {args.n_null} random directions "
          f"(response {min(null_resp_c.values()):.2f}-{max(null_resp_c.values()):.2f}, "
          f"diff {min(null_diff_c.values()):.2f}-{max(null_diff_c.values()):.2f})\n")

    print("DIFF view (decision-relevant). pctile = where the direction falls inside the random")
    print("distribution; * = clears the 95th percentile.\n")
    print(f"{'concept':<14} | {'wbar':>16} | {'pctile':>6} | {'head':>7} | {'pctile':>6} | "
          f"{'null med':>8} | {'null p95':>8}")
    print("-" * 88)
    order = sorted(concepts, key=lambda c: -abs(np.mean(per_seed[c]["diff"])))
    for c in order:
        a = np.array(per_seed[c]["resp"]); b = np.array(per_seed[c]["diff"])
        null_resp, null_diff, nd = null_resp_c[c], null_diff_c[c], null_dist_c[c]
        pw = float((nd < abs(b.mean())).mean() * 100)
        ph = float((nd < abs(head_row[c][1])).mean() * 100)
        sb = "*" if abs(b.mean()) > null_diff else " "
        sh = "*" if abs(head_row[c][1]) > null_diff else " "
        print(f"{c:<14} | {b.mean():+.3f}{sb} +/- {b.std():.3f} | {pw:>5.1f}% | "
              f"{head_row[c][1]:+.3f}{sh} | {ph:>5.1f}% | {null_med_c[c]:>8.3f} | {null_diff:>8.3f}")
        rows.append({"concept": c, "wbar_response_r": round(a.mean(), 4),
                     "wbar_response_std": round(a.std(), 4),
                     "head_response_r": round(head_row[c][0], 4),
                     "wbar_diff_r": round(b.mean(), 4), "wbar_diff_std": round(b.std(), 4),
                     "wbar_diff_pctile": round(pw, 1),
                     "head_diff_r": round(head_row[c][1], 4), "head_diff_pctile": round(ph, 1),
                     "null_diff_median": round(null_med_c[c], 4),
                     "null_response": round(null_resp, 4), "null_diff": round(null_diff, 4)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {args.out}")
    print("DIFF is the decision-relevant column: pairwise accuracy depends only on the sign of the "
          "difference, and the shared prompt cancels there.")


if __name__ == "__main__":
    main()
