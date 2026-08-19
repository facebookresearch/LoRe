"""Shared primitives for the three SAE interpretability experiments.

Everything here is PRISM-based and self-contained: it reads only the artifacts
regenerated this session and writes nothing outside sae/experiments/.

Inputs it expects (repo-relative):
  PRISM/basis_matrices.pt            (key PART2_K10_seed42: V [4096,K], W [users,K])
  data/prism/concept_vectors.pt      (11 concepts, each a raw [4096] vector)
  PRISM/data/prism/{train,test}_embeddings.pkl
  sae/checkpoints/d3/model.pt        (D3 TopK SAE)
  base reward head via PRISM/rm_head_utils.load_reward_head()

The "shared reward direction" for the PRISM vanilla-LoRe basis is defined as
    v_pop = unit( V @ mean_users(W) )
because basis_matrices.pt stores per-user simplex weights W (softmax'd at save
time), so the population weight is their mean. Swap `shared_direction` if a
different definition is wanted -- nothing else changes.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "PRISM")):
    if _p not in sys.path:
        sys.path.append(_p)

DEFAULT_BASIS = os.path.join(REPO_ROOT, "PRISM", "basis_matrices.pt")
DEFAULT_RUN_KEY = "PART2_K10_seed42"
DEFAULT_CONCEPTS = os.path.join(REPO_ROOT, "data", "prism", "concept_vectors.pt")
DEFAULT_TRAIN_EMB = os.path.join(REPO_ROOT, "PRISM", "data", "prism", "train_embeddings.pkl")
DEFAULT_TEST_EMB = os.path.join(REPO_ROOT, "PRISM", "data", "prism", "test_embeddings.pkl")
DEFAULT_SAE_CKPT = os.path.join(REPO_ROOT, "sae", "checkpoints", "d3", "model.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def resolve_device(choice: str = "auto") -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


# --- base head + basis directions --------------------------------------------

def load_head(device: torch.device | None = None) -> torch.Tensor:
    from rm_head_utils import load_reward_head
    h = load_reward_head().reshape(-1).float()
    h = unit(h)
    return h.to(device) if device is not None else h


def load_basis(run_key: str = DEFAULT_RUN_KEY, path: str = DEFAULT_BASIS):
    m = torch.load(path, map_location="cpu")
    if run_key not in m:
        raise KeyError(f"{run_key!r} not in {path}; have {list(m)[:6]}")
    e = m[run_key]
    return e["V"].float(), e["W"].float()  # V [4096,K], W [users,K] (simplex)


def shared_direction(V: torch.Tensor, W: torch.Tensor, head: torch.Tensor | None = None) -> torch.Tensor:
    """v_pop = unit(V @ mean_users(W)); optionally oriented to agree with head."""
    w_mean = W.mean(dim=0)                 # [K]
    d = unit((V @ w_mean).float())         # [4096]
    if head is not None and float(d @ head) < 0:
        d = -d
    return d


def residual(direction: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Component of `direction` orthogonal to the base head (unit)."""
    return unit(direction - float(direction @ head) * head)


# --- concept library + significance null -------------------------------------

def load_concepts(path: str = DEFAULT_CONCEPTS) -> dict[str, torch.Tensor]:
    cvs = torch.load(path, map_location="cpu", weights_only=False)
    return {c: unit(cvs[c].float()) for c in cvs}


def null_threshold(dim: int = 4096, n: int = 2000, pct: int = 95, seed: int = 0) -> float:
    """95th-percentile |cos| between a fixed vector and n random ones (same as interpret_wbar)."""
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, dim, generator=g)
    v = torch.randn(dim, generator=g)
    c = (F.normalize(R, dim=1) @ unit(v)).abs()
    return float(torch.quantile(c, pct / 100.0))


def concept_cosines(direction: torch.Tensor, concepts: dict[str, torch.Tensor]) -> dict[str, float]:
    d = unit(direction.float().cpu())
    return {c: float(d @ concepts[c]) for c in concepts}


# --- PRISM per-user preference diffs (for held-out accuracy) ------------------

def load_user_diffs(split: str = "test", seen: bool = True,
                    emb_path: str | None = None) -> list[torch.Tensor]:
    """List of [m_i, 4096] tensors of (chosen - rejected) diffs per user.

    Mirrors basis_reproducibility_check.group_embeddings_by_user: filter on
    extra_info.seen / .split, group by user_id, stack chosen-minus-rejected.
    """
    from collections import defaultdict
    if emb_path is None:
        emb_path = DEFAULT_TEST_EMB if split == "test" else DEFAULT_TRAIN_EMB
    dataset = torch.load(emb_path, weights_only=False)
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for ex in dataset:
        info = ex.get("extra_info", {})
        if info.get("seen") == seen and info.get("split") == split:
            uid = info.get("user_id")
            if uid:
                ch = torch.tensor(info["chosen_conv_embedding"], dtype=torch.float32)
                rj = torch.tensor(info["rejected_conv_embedding"], dtype=torch.float32)
                grouped[uid].append(ch - rj)
    return [torch.stack(grouped[u]) for u in sorted(grouped)]


def direction_pair_accuracy(direction: torch.Tensor, user_diffs: list[torch.Tensor]) -> dict:
    """Fraction of pairs where direction ranks chosen above rejected (d . diff > 0).

    Returns overall (pooled over all pairs) and per-user-mean accuracy.
    """
    d = direction.float().cpu()
    per_user, total_correct, total = [], 0, 0
    for X in user_diffs:
        s = (X.float().cpu() @ d) > 0
        per_user.append(float(s.float().mean()))
        total_correct += int(s.sum())
        total += int(s.numel())
    return {
        "overall_pair_acc": total_correct / max(total, 1),
        "per_user_mean_acc": float(sum(per_user) / max(len(per_user), 1)),
        "n_users": len(user_diffs),
        "n_pairs": total,
    }


# --- SAE ----------------------------------------------------------------------

def load_sae(ckpt_path: str = DEFAULT_SAE_CKPT, device: torch.device | None = None):
    """Instantiate TopKSAE from a D3 checkpoint (mirrors evaluate_sae.py)."""
    from sae.src.topk_sae import TopKSAE
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    tr = cfg.get("training", {})
    model = TopKSAE(
        input_dim=int(cfg["input_dim"]),
        dict_size=int(cfg["dict_size"]),
        k=int(cfg["k"]),
        normalize_decoder=bool(tr.get("normalize_decoder", True)),
        aux_k=int(tr.get("aux_k", cfg["k"])),
        sparsity_mode=str(tr.get("sparsity_mode", "topk")),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
