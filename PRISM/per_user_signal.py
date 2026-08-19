"""
Is there any *learnable per-user* preference signal in PRISM, above a global reward direction?

This is the upstream question Ifesi's alpha sweep implies but doesn't directly test: LoRe never beats
the base RM (~0.59) at any alpha/rank, and with alpha=0 the free bases OVERFIT (train 0.999 / test
0.570) instead of generalizing. That strongly suggests there is no generalizable per-user structure.
This script tests that head-on, per user, on held-out prompts.

For each SEEN user (present in both train and test) with enough pairs, we build a personal preference
direction from their TRAIN (chosen-rejected) diffs and score their TEST diffs (unseen prompts). We
compare four directions on each user's test pairs:

  1. true_head        : Skywork score.weight  (the general RM; personalization must beat this to matter)
  2. global_meandiff  : mean (chosen-rejected) over ALL users' train diffs (one shared direction)
  3. personal_meandiff: mean (chosen-rejected) over THIS user's train diffs
  4. other_user       : personal direction fit on a RANDOM *different* user (the key control)

Read-out:
  * personal >> global AND personal >> other_user  -> real, user-specific, learnable signal.
  * personal ~= other_user ~= global               -> the "personal" fit only recovers generic quality;
                                                       no per-user signal (any gain is just overfitting
                                                       or the shared axis). This is what alpha=0 predicts.

CPU-only; reads the cached embeddings. Run on the CORRECTED embeddings for the real answer -- on the
buggy ones the shared formatting artifact dominates and this mostly measures that.
"""
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from rm_head_utils import load_reward_head  # noqa: E402

MIN_TRAIN, MIN_TEST = 5, 3   # a user needs at least this many train / test pairs to be included
SEED = 0


def diffs_by_user(dataset, seen_value, split_name):
    """{user_id: [num_pairs, 4096] tensor of (chosen - rejected) diffs} for the given slice."""
    g = defaultdict(list)
    for ex in dataset:
        i = ex.get("extra_info", {})
        if i.get("seen") == seen_value and i.get("split") == split_name and i.get("user_id"):
            c = torch.tensor(i["chosen_conv_embedding"], dtype=torch.float32)
            r = torch.tensor(i["rejected_conv_embedding"], dtype=torch.float32)
            g[i["user_id"]].append(c - r)
    return {u: torch.stack(v) for u, v in g.items()}


def unit(v):
    return v / (v.norm() + 1e-8)


def acc(D, direction):
    return (D @ direction > 0).float().mean().item()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(SCRIPT_DIR, "..", "data", "prism")
    ap.add_argument("--train", default=os.path.join(default_dir, "train_embeddings.pkl"),
                    help="path to (corrected) train_embeddings.pkl")
    ap.add_argument("--test", default=os.path.join(default_dir, "test_embeddings.pkl"),
                    help="path to (corrected) test_embeddings.pkl")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"Loading embeddings:\n  train: {args.train}\n  test:  {args.test}")
    tr = torch.load(args.train, weights_only=False)
    te = torch.load(args.test, weights_only=False)

    train_u = diffs_by_user(tr, True, "train")   # seen users, train prompts
    test_u = diffs_by_user(te, True, "test")     # seen users, test prompts (unseen prompts)

    # global shared direction from ALL seen-train diffs
    all_train = torch.cat([D for D in train_u.values()], dim=0)
    global_dir = unit(all_train.mean(0))
    head = unit(load_reward_head().reshape(-1))

    users = [u for u in train_u
             if u in test_u and train_u[u].shape[0] >= MIN_TRAIN and test_u[u].shape[0] >= MIN_TEST]
    print(f"{len(users)} seen users with >= {MIN_TRAIN} train and >= {MIN_TEST} test pairs "
          f"(of {len(train_u)} seen users)")
    if not users:
        print("No users meet the pair thresholds; nothing to test."); return

    per_user = {"true_head": [], "global_meandiff": [], "personal_meandiff": [], "other_user": []}
    npairs = []
    for u in users:
        Dtr, Dte = train_u[u], test_u[u]
        npairs.append((Dtr.shape[0], Dte.shape[0]))
        personal = unit(Dtr.mean(0))
        # control: a personal direction fit on a RANDOM different qualifying user's train diffs
        other = random.choice([x for x in users if x != u])
        other_dir = unit(train_u[other].mean(0))

        per_user["true_head"].append(acc(Dte, head))
        per_user["global_meandiff"].append(acc(Dte, global_dir))
        per_user["personal_meandiff"].append(acc(Dte, personal))
        per_user["other_user"].append(acc(Dte, other_dir))

    med_tr = int(np.median([a for a, _ in npairs]))
    med_te = int(np.median([b for _, b in npairs]))
    print(f"median pairs per user: {med_tr} train / {med_te} test\n")

    print(f"{'direction':>18} | {'mean test acc':>13} | {'std':>6}")
    print("-" * 46)
    for k in ("true_head", "global_meandiff", "personal_meandiff", "other_user"):
        a = np.array(per_user[k])
        print(f"{k:>18} | {a.mean():>13.4f} | {a.std():>6.4f}")

    # The decisive contrast: personal vs its two null baselines, paired across users.
    p = np.array(per_user["personal_meandiff"])
    g = np.array(per_user["global_meandiff"])
    o = np.array(per_user["other_user"])
    print(f"\npersonal - global      : {(p - g).mean():+.4f}  (per-user paired mean)")
    print(f"personal - other_user  : {(o * 0 + p - o).mean():+.4f}  (vs random-other-user control)")
    print("\nRead-out: if BOTH gaps are clearly positive -> real learnable per-user signal. If both are "
          "~0 (or negative) -> the personal fit just recovers the shared quality axis; no personalization.")


if __name__ == "__main__":
    main()
