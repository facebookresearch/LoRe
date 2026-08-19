"""
Corrected "lazy direction" / "basis collapse" experiments.

Why this rewrite exists
-----------------------
An earlier version PRINTED "alpha=1538" / "alpha=384" while actually passing alpha=0.0, and
justified those numbers by scaling alpha with the sample size N. Both were wrong:
  * The printed condition did not match the executed one (the run shows Alpha=0.0000).
  * LoRe_regularized's loss is  nll.mean() + alpha * mean(1 - cos) : BOTH terms are means, so the
    reg/nll balance does NOT depend on N. The meaningful scaling is alpha/K (reg is averaged over
    the K basis columns), not alpha/N.

It also only reported TRAIN accuracy on random labels. With N < d (2000 < 4096) EVERY labeling is
linearly separable through the origin (Cover's theorem), so 100% train accuracy on noise is
guaranteed and proves nothing on its own. The property that separates "lazy memorization" from
"real signal" is GENERALIZATION: memorized noise gives test ~50%; a real low-dimensional signal
gives test ~ train. So every condition here reports TRAIN and held-out TEST, and we add a
real-label control -- that contrast is the actual answer.

Experiment A: memorization vs generalization (K=1)
  A1 random labels, alpha=0    -> expect train ~100%, TEST ~50%   (memorizes, does NOT generalize)
  A2 real labels,   alpha=0    -> expect train high, TEST high    (generalizes => real signal)
  A3 random labels, alpha=1e4  -> the anchor penalty blocks memorization (train well below 100%)
Experiment B: artificial conflict (K=2, alpha=0) -- capacity demo that LoRe CAN learn distinct
  bases when two users disagree on identical prompts. Not proof that conflict-absence is THE cause
  of the PRISM collapse (that is argued by dim_ablation.py + retrain_correct_anchor.py).
"""
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
import utils                                    # noqa: E402
from rm_head_utils import load_reward_head      # noqa: E402

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ITERS, LR, SEED = 3000, 5.0, 42


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def acc(diffs, v):
    """Fraction of labeled diffs with positive score under direction v [4096]."""
    return (diffs @ v > 0).float().mean().item()


def fit_k1(train_diffs, alpha, V_sft):
    """Fit a single direction (K=1) on labeled diffs [N,4096] via the paper's optimizer."""
    set_seed(SEED)
    m = utils.LoRe_regularized(num_classes=1, num_features=4096, num_basis_vectors=1,
                               num_iterations=ITERS, learning_rate=LR, alpha=alpha, V_sft=V_sft)
    m.train([train_diffs])
    return m.V.detach()[:, 0]


def load_diffs(sample, flip):
    """Build (chosen-rejected) diffs; if flip, assign each pair a fixed random +/-1 label."""
    d = []
    for ex in sample:
        c = torch.tensor(ex['extra_info']['chosen_conv_embedding'], dtype=torch.float32, device=device)
        r = torch.tensor(ex['extra_info']['rejected_conv_embedding'], dtype=torch.float32, device=device)
        diff = c - r
        if flip and random.random() < 0.5:
            diff = -diff                        # random label, fixed for this pair
        d.append(diff)
    return torch.stack(d)


def main():
    print("Loading true reward anchor (score.weight) + embeddings...")
    V_sft = load_reward_head(device=device)     # real head; used only in the alpha>0 condition
    train_emb = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism", "train_embeddings.pkl"),
                           map_location='cpu', weights_only=False)

    # One fixed sample, split into train / held-out test.
    set_seed(SEED)
    sample = random.sample(train_emb, 2600)
    tr_s, te_s = sample[:2000], sample[2000:]

    print("\n" + "=" * 80)
    print("EXPERIMENT A: memorization vs generalization (K=1) -- TRAIN and held-out TEST")
    print("=" * 80)
    print(f"{'condition':>34} | {'train_acc':>9} | {'test_acc':>8}")
    print("-" * 60)

    # A1: random labels, no anchor -> memorizes train, fails test
    set_seed(SEED); tr = load_diffs(tr_s, flip=True)
    set_seed(SEED + 1); te = load_diffs(te_s, flip=True)
    v = fit_k1(tr, alpha=0.0, V_sft=V_sft)
    print(f"{'A1 random labels, alpha=0':>34} | {acc(tr, v):>9.4f} | {acc(te, v):>8.4f}")

    # A2 (CONTROL): real labels, no anchor -> generalizes => real signal, not memorization
    tr = load_diffs(tr_s, flip=False)
    te = load_diffs(te_s, flip=False)
    v = fit_k1(tr, alpha=0.0, V_sft=V_sft)
    print(f"{'A2 real labels,   alpha=0 (control)':>34} | {acc(tr, v):>9.4f} | {acc(te, v):>8.4f}")

    # A3: random labels WITH the real anchor penalty -> memorization is blocked
    set_seed(SEED); tr = load_diffs(tr_s, flip=True)
    set_seed(SEED + 1); te = load_diffs(te_s, flip=True)
    v = fit_k1(tr, alpha=1e4, V_sft=V_sft)
    print(f"{'A3 random labels, alpha=1e4':>34} | {acc(tr, v):>9.4f} | {acc(te, v):>8.4f}")

    print("\nReading: A1 train~100% / test~50% = memorization (does NOT generalize).")
    print("A2 train and TEST both high = the real PRISM signal genuinely generalizes -> it is a")
    print("real low-dimensional axis, NOT a lazy high-dim memorization artifact. A3: the anchor")
    print("penalty prevents memorizing noise (the alpha>0 result an earlier run discarded).")

    print("\n" + "=" * 80)
    print("EXPERIMENT B: artificial conflict injection (K=2, alpha=0) -- capacity demo")
    print("=" * 80)
    set_seed(SEED)
    conf = random.sample(train_emb, 500)
    a, b = [], []
    for ex in conf:
        c = torch.tensor(ex['extra_info']['chosen_conv_embedding'], dtype=torch.float32, device=device)
        r = torch.tensor(ex['extra_info']['rejected_conv_embedding'], dtype=torch.float32, device=device)
        a.append(c - r); b.append(r - c)        # user A prefers chosen, user B the opposite
    feats = [torch.stack(a), torch.stack(b)]

    set_seed(SEED)
    m = utils.LoRe_regularized(num_classes=2, num_features=4096, num_basis_vectors=2,
                               num_iterations=ITERS, learning_rate=LR, alpha=0.0, V_sft=V_sft)
    Wk, Vk = m.train(feats)
    Vn = F.normalize(m.V.detach(), p=2, dim=0)
    cos = torch.dot(Vn[:, 0], Vn[:, 1]).item()
    accs = np.mean([(feats[i] @ (m.V.detach() @ F.softmax(m.W.detach()[i], 0)) > 0).float().mean().item()
                    for i in range(2)])
    print(f"cosine between the 2 learned bases: {cos:+.4f}   train acc: {accs:.4f}")
    print("Negative cosine => forced conflict makes LoRe learn 2 distinct bases (capacity exists).")
    print("This is a capacity demo, NOT proof that conflict-absence causes the PRISM collapse.")


if __name__ == "__main__":
    main()
