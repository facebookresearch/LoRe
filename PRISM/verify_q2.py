"""
Verify, on CPU from already-saved artifacts, the "why is accuracy so high / why does
it climb with K" question. This does NOT retrain anything, does NOT touch a GPU, and
does NOT modify train_basis.py or utils.py. It only READS:
  - PRISM/reproduced_matrices-pruned/PRISM_V_lore_K_{K}_alpha_10000.0.pt   (you made these)
  - PRISM/reproduced_matrices-pruned/PRISM_W_lore_seen_{K}_10000.0.pt
  - PRISM/data/prism/{train,test}_embeddings.pkl

It reuses the SAME accuracy definition as the real pipeline (utils.py evaluate_model,
lines 28-37): score = X @ V @ w, accuracy = fraction of scores > 0, where
X = (chosen_embedding - rejected_embedding).

Run:  uv run python PRISM/verify_q2.py
"""
import json, os
import torch
import torch.nn.functional as F
from collections import defaultdict

HERE   = os.path.dirname(os.path.abspath(__file__))
MATDIR = os.path.join(HERE, "reproduced_matrices-pruned")
KS     = [1, 5, 10, 15, 20, 25, 50]

loadV = lambda K: torch.load(f"{MATDIR}/PRISM_V_lore_K_{K}_alpha_10000.0.pt",
                             map_location="cpu", weights_only=False).float()
loadW = lambda K: torch.load(f"{MATDIR}/PRISM_W_lore_seen_{K}_10000.0.pt",
                             map_location="cpu", weights_only=False).float()


def load_diffs(split):
    """Return (per-user dict of stacked X, and one big stacked X) for a split."""
    fn = "train_embeddings.pkl" if split == "train" else "test_embeddings.pkl"
    data = torch.load(os.path.join(HERE, "data", "prism", fn), weights_only=False)
    by_user, allX = defaultdict(list), []
    for rec in data:
        ei = rec["extra_info"]
        if ei.get("split") != split:
            continue
        c = torch.as_tensor(ei["chosen_conv_embedding"], dtype=torch.float32)
        r = torch.as_tensor(ei["rejected_conv_embedding"], dtype=torch.float32)
        x = c - r
        allX.append(x)
        if ei.get("seen") and ei.get("user_id"):
            by_user[ei["user_id"]].append(x)
    return {u: torch.stack(v) for u, v in by_user.items()}, torch.stack(allX)


def main():
    ids = json.load(open(f"{MATDIR}/seen_train_user_ids.json"))
    row_of = {u: i for i, u in enumerate(ids)}

    print("loading embeddings (a few seconds)...")
    test_by_user, Xte = load_diffs("test")
    _,            Xtr = load_diffs("train")

    # ---- 1. reproduce the PNG's seen-user test-accuracy curve, per K ----
    print("\n[1] seen-user test accuracy per K  (should climb like the PNG's blue line):")
    users = [u for u in test_by_user if u in row_of]
    print(f"    (averaged over {len(users)} seen users)")
    for K in KS:
        V, W = loadV(K), loadW(K)
        accs = [((test_by_user[u] @ V @ W[row_of[u]]) > 0).float().mean().item()
                for u in users]
        print(f"    K={K:<3} ncol={V.shape[1]:<2} acc={sum(accs)/len(accs):.4f}")

    # ---- 2. the ceiling: ONE generic direction = mean(chosen-rejected) ----
    d_opt = F.normalize(Xtr.mean(0), dim=0)          # learned on TRAIN only
    print(f"\n[2] ONE direction = mean(chosen-rejected) from train, tested on test:")
    print(f"    accuracy = {(Xte @ d_opt > 0).float().mean():.4f}"
          f"   <-- task is ~trivially separable by a single generic direction")

    # ---- 3. how much of that one direction does each learned V capture? ----
    # The right per-user reward direction is V @ w (a MIX of columns), not column 0.
    # Mean over users of cos(user reward dir, optimal dir) should track [1]'s accuracy.
    print("\n[3] mean alignment of users' actual reward directions (V @ w) with the")
    print("    optimal direction — this should CLIMB with K like [1]:")
    ids2 = json.load(open(f"{MATDIR}/seen_train_user_ids.json"))
    for K in KS:
        V, W = loadV(K), loadW(K)
        dirs = F.normalize(V @ W.T, dim=0)           # [F, num_users], each col = a user's reward dir
        mean_cos = (dirs.T @ d_opt).mean().item()    # signed cosine to optimal, averaged over users
        print(f"    K={K:<3} mean cos(V@w, opt)={mean_cos:+.4f}")
    print("    (tiny, near-orthogonal cosines — yet a sliver of this dominant signal is enough)")

    # ---- 4. the cosine trap: V_1 vs V_25 look identical but classify differently ----
    v1, v25 = loadV(1)[:, 0], loadV(25)[:, 0]
    flip = (torch.sign(Xte @ v1) != torch.sign(Xte @ v25)).float().mean().item()
    print("\n[4] why cosine-between-bases misled me:")
    print(f"    cos(V_1, V_25) = {F.cosine_similarity(v1, v25, dim=0).item():.4f} (look identical)")
    print(f"    yet they disagree on {flip*100:.1f}% of test pairs' signs "
          f"(acc .78 vs .97). X is huge-norm, so a ~1 degree rotation flips many signs.")


if __name__ == "__main__":
    main()
