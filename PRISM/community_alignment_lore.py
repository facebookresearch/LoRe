"""
Run LoRe on Community Alignment -- the dataset that actually has enough data per user.

Why this dataset: synthetic_recovery.py found a sharp phase transition at ~50 pairs/user (below ~25
LoRe cannot recover even clean planted axes; at 100+ it recovers them near-perfectly). PRISM has ~15
pairs/user, which is why it nulled out. Community Alignment (English) has a MEDIAN of 201 pairs/user
with 941 users above 100 -- comfortably inside the regime where LoRe demonstrably works.

So this is the real test of the question PRISM could not answer: with adequate data per user, does
LoRe find genuine personalization structure that beats a single global reward direction?

Reported, all on HELD-OUT pairs (split per user, so test prompts are unseen):
  * base_rm        : Skywork's true reward head -- the bar LoRe must clear
  * global_meandiff: one shared direction fitted on everyone
  * personal       : each user's own mean-diff direction
  * other_user     : a random OTHER user's direction (the key control; if personal ~= other_user
                     there is no user-specific signal, only a shared quality axis)
  * LoRe test acc at several ranks, plus basis collapse metrics

Read-out:
  * LoRe > base_rm AND personal > other_user -> real personalization exists here; PRISM's null was a
    data-volume artifact and LoRe works when the data supports it.
  * LoRe ~= base_rm and personal ~= other_user -> even with ~200 pairs/user there is no separable
    per-user structure, which would point at preference data/representation rather than data volume.

CPU-only once embeddings exist.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoRe_regularized, LoReV2                           # noqa: E402
from rm_head_utils import load_reward_head                           # noqa: E402
from synthetic_recovery import collapse_metrics, eval_acc            # noqa: E402
from embed_community_alignment import text_key                       # noqa: E402


def unit(v):
    return v / (v.norm() + 1e-8)


def acc(D, direction):
    return (D @ direction > 0).float().mean().item()


def build_user_diffs(pairs, emb_lookup, min_pairs, test_frac, gen, split_by_turn=True,
                     max_pairs=None, val_frac=0.0):
    """{user: (train_diffs [n,4096], val_diffs [v,4096] or None, test_diffs [m,4096])}, per user.

    val_frac carves a THIRD split out of the same turn-level grouping, for tuning regularization
    strengths and learning rates. Anything selected on test is not a held-out number any more, and
    the effect sizes here (~0.02) are small enough that tuning on test would manufacture them.
    val_frac=0.0 (the default) draws exactly the same train/test split as before, so previously
    committed results still reproduce.

    split_by_turn is essential for a valid held-out set. Each turn yields 3 pairs (the preferred
    response vs each of the other 3), so those pairs share the SAME prompt and the SAME chosen
    response. Splitting by pair would put siblings on both sides, letting a trained model memorise
    "for this prompt, response_d wins" and score the test sibling for free -- inflating LoRe (which
    trains) while leaving base_rm (which does not) untouched, i.e. an unfair comparison. Grouping by
    (conversation, turn) keeps all siblings on one side, so test prompts are genuinely unseen.
    """
    by_user = defaultdict(lambda: defaultdict(list))   # user -> turn key -> [diffs]
    missing = 0
    for p in pairs:
        kc, kr = text_key(p["prompt"], p["chosen"]), text_key(p["prompt"], p["rejected"])
        if kc not in emb_lookup or kr not in emb_lookup:
            missing += 1
            continue
        turn_key = (p.get("conversation_id"), p.get("turn")) if split_by_turn else len(
            by_user[p["user_id"]])
        by_user[p["user_id"]][turn_key].append(emb_lookup[kc] - emb_lookup[kr])
    if missing:
        print(f"[warn] {missing} pairs skipped (embedding not found -- partial embed run?)")

    out = {}
    for u, groups in by_user.items():
        keys = list(groups)
        if max_pairs:
            # trim whole TURNS (never split a turn) until under the budget, so the leak-free
            # property is preserved while sweeping how much data each user gets
            kept, total_kept = [], 0
            for k in keys:
                if total_kept >= max_pairs:
                    break
                kept.append(k); total_kept += len(groups[k])
            keys = kept
        total = sum(len(groups[k]) for k in keys)
        if total < min_pairs:
            continue
        perm = torch.randperm(len(keys), generator=gen).tolist()
        n_test_groups = max(1, int(round(test_frac * len(keys))))
        te_keys = {keys[i] for i in perm[:n_test_groups]}
        n_val_groups = int(round(val_frac * len(keys))) if val_frac > 0 else 0
        va_keys = {keys[i] for i in perm[n_test_groups:n_test_groups + n_val_groups]}
        tr = [d for k in keys if k not in te_keys and k not in va_keys for d in groups[k]]
        va = [d for k in keys if k in va_keys for d in groups[k]]
        te = [d for k in keys if k in te_keys for d in groups[k]]
        if not tr or not te or (n_val_groups and not va):
            continue
        out[u] = (torch.stack(tr), torch.stack(va) if va else None, torch.stack(te))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--min_pairs", type=int, default=50,
                    help="drop users below this many usable pairs (phase transition is ~50)")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="cap pairs per user (trims whole turns). Sweep this to test whether more "
                         "data per user helps -- the real-data analogue of synthetic_recovery.py")
    ap.add_argument("--split_by_pair", action="store_true",
                    help="LEAKY: split train/test by pair, so sibling pairs from the same turn "
                         "(same prompt, same chosen) land on both sides. Only for measuring how "
                         "much that leakage inflates results")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--model", choices=["vanilla", "v2"], default="vanilla",
                    help="vanilla = LoRe_regularized; v2 = LoReV2 (signed wbar+delta weights, "
                         "reward-space ridge, no column dropping)")
    ap.add_argument("--val_frac", type=float, default=0.0,
                    help="carve a validation split (turn-level) for tuning. Required for v2 "
                         "early stopping; leave at 0 to reproduce the committed vanilla runs")
    ap.add_argument("--lam_pop", type=float, default=0.01, help="v2 only: ridge on ||V wbar||^2")
    ap.add_argument("--lam_d", type=float, default=0.01, help="v2 only: ridge on ||V delta_u||^2")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 0.5 for vanilla (as published), 1e-2 for v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="run several seeds and report mean +/- std. Each seed redraws BOTH the "
                         "turn-level train/test split and the LoRe init, so the spread covers the "
                         "two things that actually vary. Effect sizes here are small (~0.02), so a "
                         "single seed cannot distinguish a real gap from split noise")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results",
                                                  "community_alignment", "lore_results.csv"))
    ap.add_argument("--save_model", default=None,
                    help="v2 only: save the learned matrices (V, wbar, per-user delta) for each "
                         "(seed, rank) to this .pt path, for offline analysis of the population "
                         "direction V@wbar and the per-user weights wbar+delta_u")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-2 if args.model == "v2" else 0.5
    seeds = args.seeds if args.seeds else [args.seed]

    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}
    print(f"{len(pairs)} pairs, {len(emb_lookup)} embeddings")
    head = unit(load_reward_head().reshape(-1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["seed", "rank", "alpha", "train_acc", "test_acc",
                                      "min_abs_basis_cos", "bases_kept", "base_rm", "global",
                                      "personal", "other_user", "n_users", "median_pairs",
                                      "wbar_only"])
    w.writeheader()

    refs = defaultdict(list)          # reference direction accuracies, per seed
    lore = defaultdict(list)          # rank -> [(test_acc, vs_base, collapse), ...] per seed
    saved_models = []                 # populated only when --save_model is set (v2)
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)

        users = build_user_diffs(pairs, emb_lookup, args.min_pairs, args.test_frac, gen,
                                 split_by_turn=not args.split_by_pair,
                                 max_pairs=args.max_pairs, val_frac=args.val_frac)
        uids = sorted(users)
        if not uids:
            print("No users met the threshold; nothing to run."); return
        npairs = [users[u][0].shape[0] + users[u][2].shape[0] for u in uids]
        train_feats = [users[u][0] for u in uids]
        val_feats = [users[u][1] for u in uids] if args.val_frac > 0 else None
        test_feats = [users[u][2] for u in uids]
        if seed == seeds[0]:
            print(f"{len(uids)} users with >= {args.min_pairs} pairs "
                  f"(median {int(np.median(npairs))} pairs/user, max {max(npairs)})\n")

        # ---- reference directions on held-out pairs ----
        global_dir = unit(torch.cat(train_feats, 0).mean(0))
        base_acc, glob_acc, pers_acc, other_acc = [], [], [], []
        for i, u in enumerate(uids):
            te = test_feats[i]
            base_acc.append(acc(te, head))
            glob_acc.append(acc(te, global_dir))
            pers_acc.append(acc(te, unit(train_feats[i].mean(0))))
            j = random.choice([k for k in range(len(uids)) if k != i])
            other_acc.append(acc(te, unit(train_feats[j].mean(0))))
        m = lambda x: float(np.mean(x))
        for name, v in (("base_rm", base_acc), ("global", glob_acc),
                        ("personal", pers_acc), ("other_user", other_acc)):
            refs[name].append(m(v))

        # ---- LoRe at several ranks ----
        for K in args.ranks:
            if args.model == "v2":
                model = LoReV2(len(train_feats), 4096, K, lam_pop=args.lam_pop,
                               lam_d=args.lam_d, num_iterations=args.iters,
                               learning_rate=args.lr, verbose=False)
                Wk, Vk = model.train(train_feats, val=val_feats)
            else:
                anchor = load_reward_head().reshape(-1, 1)
                model = LoRe_regularized(anchor, args.alpha, len(train_feats), 4096, K,
                                         args.iters, args.lr)
                Wk, Vk = model.train(train_feats)
            tr, te = eval_acc(Wk, Vk, train_feats), eval_acc(Wk, Vk, test_feats)
            # v2 ablation: zero the deltas and keep only the shared population direction. Without
            # this, "v2 beats base_rm" is ambiguous between real personalization and simply
            # having fitted a better GLOBAL reward direction than Skywork's untrained head. If
            # wbar_only ~= test_acc, the deltas are decorative and the gain is not personalization.
            wbar_only = ""
            if args.model == "v2":
                wbar_only = round(eval_acc(
                    model.wbar.detach().unsqueeze(0).repeat(len(train_feats), 1),
                    Vk, test_feats), 4)
            mc = collapse_metrics(model.V.detach().cpu())
            if args.save_model and args.model == "v2":
                V = model.V.detach().cpu()
                wbar = model.wbar.detach().cpu()
                delta = model.delta.detach().cpu()
                saved_models.append({
                    "seed": seed, "rank": K, "uids": list(uids),
                    "V": V, "wbar": wbar, "delta": delta,          # raw learned matrices
                    "wbar_dir": (V @ wbar),                        # [4096] shared population direction
                    "lam_pop": args.lam_pop, "lam_d": args.lam_d,
                    "lr": args.lr, "min_pairs": args.min_pairs,
                })
            lore[K].append((tr, te, te - m(base_acc), mc))
            w.writerow({"seed": seed, "rank": K, "alpha": args.alpha, "train_acc": round(tr, 4),
                        "test_acc": round(te, 4), "min_abs_basis_cos": round(mc, 4),
                        "bases_kept": Vk.shape[1], "base_rm": round(m(base_acc), 4),
                        "global": round(m(glob_acc), 4), "personal": round(m(pers_acc), 4),
                        "other_user": round(m(other_acc), 4), "n_users": len(uids),
                        "median_pairs": int(np.median(npairs)), "wbar_only": wbar_only})
            f.flush()
    f.close()

    if args.save_model:
        if not saved_models:
            print("[warn] --save_model set but nothing saved (requires --model v2)", file=sys.stderr)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
            torch.save({"records": saved_models, "pairs": os.path.abspath(args.pairs),
                        "emb": os.path.abspath(args.emb)}, args.save_model)
            print(f"Saved {len(saved_models)} learned matrices (seed x rank) -> {args.save_model}")

    ms = lambda v: (float(np.mean(v)), float(np.std(v)))
    print(f"\n=== held-out reference directions (mean +/- std over {len(seeds)} seeds) ===")
    for name in ("base_rm", "global", "personal", "other_user"):
        mu, sd = ms(refs[name])
        print(f"  {name:<12}: {mu:.4f} +/- {sd:.4f}")
    pg = np.array(refs["personal"]) - np.array(refs["global"])
    po = np.array(refs["personal"]) - np.array(refs["other_user"])
    print(f"  personal - global    : {pg.mean():+.4f} +/- {pg.std():.4f}")
    print(f"  personal - other_user: {po.mean():+.4f} +/- {po.std():.4f}   "
          f"(>0 and larger than its std => real user-specific signal)")

    print(f"\n=== LoRe (alpha={args.alpha}), mean +/- std over {len(seeds)} seeds ===")
    print(f"{'rank':>5} | {'test':>16} | {'vs base_rm':>18} | {'collapse':>16}")
    print("-" * 66)
    for K in args.ranks:
        te = [r[1] for r in lore[K]]; vb = [r[2] for r in lore[K]]; mc = [r[3] for r in lore[K]]
        print(f"{K:>5} | {np.mean(te):.4f} +/- {np.std(te):.4f} | "
              f"{np.mean(vb):+.4f} +/- {np.std(vb):.4f} | {np.mean(mc):.4f} +/- {np.std(mc):.4f}")
    print(f"\nSaved {args.out}")
    print("Read-out: a gap only counts if it is larger than its std across seeds. LoRe beating "
          "base_rm AND personal > other_user => real personalization at this data volume; both flat "
          "=> the limitation is the preference data, not volume.")


if __name__ == "__main__":
    main()
