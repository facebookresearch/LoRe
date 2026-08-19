"""
E3: Dimensionality ablation -- is a single direction good "because 4096-d is huge", or because
the PRISM preference signal is genuinely low-dimensional?

The concern being tested: "In a massive 4096-dimensional space, it's mathematically easy to find a single
'lazy' global direction that separates 98% of these non-overlapping pairs. This doesn't match my
intuition."

Test: project the (chosen - rejected) embedding diffs down to d' dimensions (top-d' right singular
vectors of the TRAIN diff matrix -- UNCENTERED, because the mean diff IS the preference signal),
then fit a single K=1 direction in that d'-space and measure TRAIN and TEST accuracy.

Interpretation:
  * If accuracy stays high even at small d' (e.g. d'=10 or 50)  -> "high-dim makes it easy" is
    FALSE. The signal is genuinely low-dimensional; a single direction separates because the
    preference is ~1-D (general response quality), not because the space is big.
  * If accuracy only appears at large d' and collapses as d' shrinks toward the sample size
    -> the overparameterization / lazy-separability story gains support.

We fit on SEEN-TRAIN and evaluate on SEEN-TEST (unseen prompts) so a memorizing separator would
show train >> test. References printed: the mean-diff direction, and the TRUE reward head.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from rm_head_utils import load_reward_head  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DIMS = [2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4096]


def pooled_diffs(dataset, seen_value, split_name):
    """All (chosen - rejected) diffs for the given slice, pooled into one [N, 4096] matrix."""
    rows = []
    for example in dataset:
        info = example.get("extra_info", {})
        if info.get("seen") == seen_value and info.get("split") == split_name and info.get("user_id"):
            c = torch.tensor(info["chosen_conv_embedding"], dtype=torch.float32)
            r = torch.tensor(info["rejected_conv_embedding"], dtype=torch.float32)
            rows.append(c - r)
    return torch.stack(rows)  # [N, 4096]


def fit_direction(D, iters=1500, lr=0.05):
    """Fit a single direction v maximizing mean log-sigmoid(D @ v). D: [N, d']. Returns v: [d',1]."""
    d = D.shape[1]
    v = torch.zeros(d, 1, requires_grad=True)
    # init at the mean-diff direction (a strong, sensible starting point)
    with torch.no_grad():
        mu = D.mean(0, keepdim=True).t()
        v.copy_(mu / (mu.norm() + 1e-8))
    opt = torch.optim.Adam([v], lr=lr)
    Dn = D / (D.std() + 1e-8)  # scale for stable logits
    for _ in range(iters):
        opt.zero_grad()
        loss = -F.logsigmoid(Dn @ v).mean()
        loss.backward()
        opt.step()
    return v.detach()


def acc(D, v):
    return (D @ v > 0).float().mean().item()


def main():
    print("Loading cached PRISM embeddings...")
    train_emb = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                        "train_embeddings.pkl"), weights_only=False)
    test_emb = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                       "test_embeddings.pkl"), weights_only=False)

    D_train = pooled_diffs(train_emb, True, "train")
    D_test = pooled_diffs(test_emb, True, "test")
    print(f"Train diffs: {tuple(D_train.shape)}   Test diffs: {tuple(D_test.shape)}")

    # References
    score_w = load_reward_head()
    head_train, head_test = acc(D_train, score_w), acc(D_test, score_w)
    mu = D_train.mean(0, keepdim=True).t()
    mu = mu / (mu.norm() + 1e-8)
    mu_train, mu_test = acc(D_train, mu), acc(D_test, mu)
    print(f"\nReference: TRUE reward head   -> train {head_train:.4f} | test {head_test:.4f}")
    print(f"Reference: mean-diff direction-> train {mu_train:.4f} | test {mu_test:.4f}")

    # DECISIVE CONTROL: remove the shared offset (center the diffs by the TRAIN mean).
    # If all the separability lives in the single shared direction, the centered residual is
    # unlearnable (~chance). This is the real evidence that the signal is ~1-D & global, not a
    # high-dimensional artifact and not per-user personalization.
    train_mean = D_train.mean(0, keepdim=True)
    Dtr_res, Dte_res = D_train - train_mean, D_test - train_mean
    v_res = fit_direction(Dtr_res)
    print(f"Control: RESIDUAL (offset removed) best K=1 -> train {acc(Dtr_res, v_res):.4f} "
          f"| test {acc(Dte_res, v_res):.4f}   (~chance => all signal is the shared offset)")
    print(f"  cos(mean-diff dir, TRUE reward head) = "
          f"{F.cosine_similarity(mu.squeeze(), score_w.squeeze(), dim=0).item():+.4f}")
    cone = (D_train @ mu / (D_train.norm(dim=1, keepdim=True) + 1e-8))
    print(f"  cosine(each train diff, mean-diff dir): mean {cone.mean():.3f}, "
          f"frac>0 {(cone > 0).float().mean():.4f}  (diffs live in one half-space/cone)")

    # Top-d' right singular vectors of the UNCENTERED train diff matrix.
    print("\nComputing SVD basis of train diffs (uncentered)...")
    # economy SVD: D_train = U S Vh ; columns of Vh.T are the principal diff directions
    _, S, Vh = torch.linalg.svd(D_train, full_matrices=False)
    print(f"  top singular values: {[round(x,1) for x in S[:5].tolist()]}  "
          f"(s1 dominates -> low-rank signal)")

    print("\n" + "=" * 66)
    print(f"{'dim d':>7} | {'K=1 train acc':>13} | {'K=1 test acc':>12} | {'train-test gap':>14}")
    print("-" * 66)
    rows = []
    for d in DIMS:
        d = min(d, D_train.shape[1])
        P = Vh[:d, :].t()                    # [4096, d] projection
        Dtr = D_train @ P                    # [N, d]
        Dte = D_test @ P
        v = fit_direction(Dtr)
        a_tr, a_te = acc(Dtr, v), acc(Dte, v)
        rows.append((d, a_tr, a_te))
        print(f"{d:>7} | {a_tr:>13.4f} | {a_te:>12.4f} | {a_tr - a_te:>14.4f}")

    # Save CSV + plot
    out_dir = os.path.join(SCRIPT_DIR, "..", "results", "dim_ablation")
    os.makedirs(out_dir, exist_ok=True)
    import csv
    with open(os.path.join(out_dir, "dim_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "k1_train_acc", "k1_test_acc"])
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = [r[0] for r in rows]
        plt.figure(figsize=(8, 5))
        plt.plot(ds, [r[2] for r in rows], "o-", label="K=1 test acc (unseen prompts)")
        plt.plot(ds, [r[1] for r in rows], "s--", label="K=1 train acc", alpha=0.6)
        plt.axhline(head_test, color="green", ls=":", label=f"true reward head test ({head_test:.2f})")
        plt.axhline(0.5, color="gray", ls=":", label="chance")
        plt.xscale("log")
        plt.xlabel("embedding dimension d' (PCA of chosen-rejected diffs)")
        plt.ylabel("preference accuracy")
        plt.title("E3: does a single direction need high dimension? (PRISM, seen users)")
        plt.ylim(0.45, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dim_ablation_plot.png"), dpi=150)
        print(f"\nSaved plot + csv to {out_dir}")
    except Exception as ex:
        print(f"(plot skipped: {ex!r}) -- csv still written to {out_dir}")


if __name__ == "__main__":
    main()
