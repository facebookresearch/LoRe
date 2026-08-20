"""Shared primitives for the Community Alignment (CA) SAE interpretability experiments.

This is the CA analogue of sae/experiments/common.py. It answers the same question
-- is the LoRe v2 shared population direction v_pop a distinct learned axis or just
the base Skywork reward head? -- but on Community Alignment (~200 pairs/user) instead
of PRISM (~15 pairs/user). CA is the "real test": synthetic_recovery.py found LoRe
needs ~50+ pairs/user before it can recover a planted axis, and PRISM sits below that.

Everything CA-specific lives in this folder. The pure-vector scoring primitives
(concept cosines, null threshold, pair accuracy, score correlation, residual) are
imported from the PRISM SAE experiments' common.py so there is ONE source of truth --
in particular the sign-flip fix (no head-reorientation of fitted directions) is
inherited, not re-implemented.

Inputs it expects (repo-relative):
  data/community_alignment/pairs.json          list of {prompt, chosen, rejected, user_id, conversation_id, turn}
  data/community_alignment/embeddings.pt       {"emb": [N,4096], "keys": [sha1(prompt\\x00response)]}
  results/community_alignment/ca_v_pop.pt       fitted v2 basis (CA_K10_seed42_lam.01_.01): V, wbar, delta, v_pop_unit
  data/prism/concept_vectors.pt                 11 concept vectors (Skywork-space, dataset-independent)
  base reward head via PRISM/rm_head_utils.load_reward_head()

The shared reward direction for CA is the STORED v_pop_unit = unit(V @ wbar) from the
committed fit, used in its natural (data) orientation -- NOT re-oriented to the head.
Re-orienting a near-orthogonal direction to the head flips its sign on a coin toss and
makes pair accuracy report 1 - acc (the sign-flip bug the reviewer caught). The fit
already orients wbar toward the chosen side, so v_pop_unit points the right way.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRISM_DIR = os.path.join(REPO_ROOT, "PRISM")
SAE_EXP_DIR = os.path.join(REPO_ROOT, "sae", "experiments")
for _p in (REPO_ROOT, PRISM_DIR, SAE_EXP_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

# Reuse the PRISM SAE experiments' pure-vector primitives (one source of truth;
# inherits the sign-flip fix). These are dataset-independent.
import common as SC  # noqa: E402
from common import (  # noqa: E402,F401
    resolve_device,
    unit,
    load_head,
    load_concepts,
    null_threshold,
    concept_cosines,
    residual,
    direction_pair_accuracy,
    score_correlation,
    ensure_dirs,
    shared_direction,
)

DEFAULT_PAIRS = os.path.join(REPO_ROOT, "data", "community_alignment", "pairs.json")
DEFAULT_EMB = os.path.join(REPO_ROOT, "data", "community_alignment", "embeddings.pt")
DEFAULT_CA_BASIS = os.path.join(REPO_ROOT, "results", "community_alignment", "ca_v_pop.pt")
DEFAULT_CONFIG_KEY = "CA_K10_seed42_lam.01_.01"   # K=10, lam_pop=lam_d=0.01, seed=42 -- matches PRISM basis_v2.pt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# --- CA fitted basis ----------------------------------------------------------

def load_ca_basis(config_key: str = DEFAULT_CONFIG_KEY, path: str = DEFAULT_CA_BASIS):
    """Return (V [4096,K], wbar [1,K], v_pop_unit [4096], config dict) for a CA fit.

    wbar is reshaped [1, K] so mean_users(W) == wbar and shared_direction(V, wbar)
    == unit(V @ wbar) with no other code change (same trick as the PRISM adapter).
    v_pop_unit is the committed unit(V @ wbar) in its natural data orientation.
    """
    m = torch.load(path, map_location="cpu", weights_only=False)
    if config_key not in m:
        raise KeyError(f"{config_key!r} not in {path}; have {list(m)}")
    e = m[config_key]
    V = e["V"].float()
    wbar = e["wbar"].float().reshape(1, -1)
    v_pop = SC.unit(e["v_pop_unit"].float())
    cfg = {k: e[k] for k in (
        "K", "lam_pop", "lam_d", "seed", "lr", "iters", "dataset",
        "min_pairs", "val_frac", "test_frac", "n_users") if k in e}
    cfg["uids"] = list(e["uids"])
    return V, wbar, v_pop, cfg


# --- CA per-user preference diffs (leak-free turn-level split) -----------------
#
# text_key and build_user_diffs are VERBATIM copies of PRISM/embed_community_alignment.py
# and PRISM/community_alignment_lore.py. They are inlined (not imported) only to avoid
# pulling in that module's `transformers` dependency, which is not needed for analysis
# and is not installed locally. Keeping them byte-identical guarantees the split here is
# the exact one the committed CA fit was trained/held-out on.

def text_key(prompt, response):
    """Stable id for a (prompt, response) pair so embeddings can be de-duplicated and looked up."""
    h = hashlib.sha1()
    h.update(prompt.encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(response.encode("utf-8", "ignore"))
    return h.hexdigest()


def build_user_diffs(pairs, emb_lookup, min_pairs, test_frac, gen, split_by_turn=True,
                     max_pairs=None, val_frac=0.0):
    """{user: (train_diffs, val_diffs or None, test_diffs)} -- verbatim from
    community_alignment_lore.build_user_diffs. Leak-free split by (conversation, turn)."""
    by_user = defaultdict(lambda: defaultdict(list))
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


def load_ca_diffs(config: dict, pairs_path: str = DEFAULT_PAIRS, emb_path: str = DEFAULT_EMB):
    """Reproduce the EXACT train/val/test split of the committed CA fit.

    Uses the fit's own config (seed, min_pairs, test_frac, val_frac) so the held-out
    pairs here are the same ones the fit never saw. Returns (uids, train_diffs,
    val_diffs, test_diffs) where each *_diffs is a list of [m_i, 4096] (chosen -
    rejected) tensors, one per user, aligned to sorted(uids).
    """
    with open(pairs_path) as f:
        pairs = json.load(f)
    blob = torch.load(emb_path, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}

    gen = torch.Generator().manual_seed(int(config["seed"]))
    users = build_user_diffs(
        pairs, emb_lookup,
        min_pairs=int(config["min_pairs"]),
        test_frac=float(config["test_frac"]),
        gen=gen,
        split_by_turn=True,
        val_frac=float(config.get("val_frac", 0.0)),
    )
    uids = sorted(users)
    train = [users[u][0] for u in uids]
    val = [users[u][1] for u in uids]
    test = [users[u][2] for u in uids]
    return uids, train, val, test
