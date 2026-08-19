"""
Load the REAL Skywork reward head (`score.weight`).

Why this file exists
--------------------
Every V_sft / base-RM extraction in this repo (train_basis.py, basis_reproducibility_check.py,
the old test_base_rm_accuracy.py, generate-prism-embeddings.py) loads the model with
`AutoModel.from_pretrained(...)`. For a `LlamaForSequenceClassification` checkpoint that returns
the bare `LlamaModel` backbone: the trained reward head `score.weight` is NOT loaded (it is
silently dropped as an unexpected key). The code then grabs "the last nn.Linear via
named_modules()", which on the bare backbone is `model.layers.31.mlp.down_proj` -- an internal
MLP matrix -- and slices one arbitrary column. That column is NOT the reward direction.

The correct reward direction is `model.score.weight` (shape [1, 4096]), i.e. the linear head that
maps the last-token hidden state to the scalar reward. This module returns it.

Two load paths:
  1. canonical: AutoModelForSequenceClassification -> model.score.weight  (needs the full weights)
  2. lightweight: a ranged HTTP read of just the `score.weight` bytes (~8KB) from the safetensors
     shard on the Hub, so you can get the head without downloading ~15GB. Cached to disk.

The reward for a response is  score.weight @ h_last  (+ bias, which is absent for this head and
cancels in any chosen-vs-rejected difference anyway). Since PRISM embeddings are exactly that
last-token hidden state h_last (see generate-prism-embeddings.py), base-RM preference accuracy is
just  sign( score.weight @ (chosen_emb - rejected_emb) ).
"""
import json
import os
import struct
import urllib.request

import numpy as np
import torch

MODEL_NAME = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
_SHARD = "model-00004-of-00004.safetensors"          # shard holding score.weight
_HUB_URL = f"https://huggingface.co/{MODEL_NAME}/resolve/main/{_SHARD}"
_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reproduced_matrices", "skywork_score_head.pt",
)


def _bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Upcast raw little-endian bfloat16 bytes to float32 (bf16 = top 16 bits of f32)."""
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32)


def _fetch_head_via_range() -> torch.Tensor:
    """Download ONLY the `score.weight` tensor via HTTP range requests (~8KB)."""
    def get_range(start, end):
        req = urllib.request.Request(_HUB_URL, headers={"Range": f"bytes={start}-{end}"})
        return urllib.request.urlopen(req, timeout=60).read()

    hdr_len = struct.unpack("<Q", get_range(0, 7))[0]
    header = json.loads(get_range(8, 8 + hdr_len - 1))
    meta = header["score.weight"]
    assert meta["dtype"] == "BF16" and meta["shape"] == [1, 4096], meta
    s, e = meta["data_offsets"]
    base = 8 + hdr_len
    raw = get_range(base + s, base + e - 1)
    f32 = _bf16_bytes_to_f32(raw).copy()
    return torch.from_numpy(f32).reshape(1, 4096)


# The real Skywork reward head has this norm; used as a light integrity check on any source.
_EXPECTED_NORM = 1.2804
_NORM_TOL = 0.05


def _validate_head(w, source):
    """Sanity-check a candidate reward head so a corrupt cache/fetch can't silently become the
    training anchor. Raises ValueError on a bad tensor."""
    w = w.float().reshape(-1)
    if w.numel() != 4096:
        raise ValueError(f"reward head from {source} has {w.numel()} elems, expected 4096")
    if not torch.isfinite(w).all():
        raise ValueError(f"reward head from {source} contains non-finite values")
    norm = w.norm().item()
    if abs(norm - _EXPECTED_NORM) > _NORM_TOL:
        raise ValueError(
            f"reward head from {source} has ||w||={norm:.4f}, expected ~{_EXPECTED_NORM} "
            f"(+/-{_NORM_TOL}); the cache/fetch is likely stale or corrupt")
    return w.reshape(1, 4096)


def load_reward_head(prefer="auto", cache_path=_DEFAULT_CACHE, device="cpu"):
    """Return the real reward head as a float32 tensor of shape [4096, 1].

    prefer:
      "auto"       -> use cached file if present, else try the lightweight range fetch,
                      else fall back to the full model load.
      "seqclf"     -> force AutoModelForSequenceClassification -> model.score.weight (canonical).
      "range"      -> force the lightweight ranged HTTP fetch.
      "cache"      -> require the cached file at cache_path (raises if it is missing).

    Every source is validated (shape, finiteness, expected norm) before it is returned, so a
    stale/truncated/swapped cache cannot silently become the base-RM measurement or the training
    regularization anchor.
    """
    if prefer == "cache" and not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"prefer='cache' but no cached reward head at {cache_path}. "
            f"Run load_reward_head(prefer='range') or 'seqclf' once to create it.")

    if prefer in ("auto", "cache") and os.path.exists(cache_path):
        w = torch.load(cache_path, map_location="cpu", weights_only=True)
        w = _validate_head(w, f"cache:{cache_path}")
        return w.reshape(-1, 1).to(device)

    if prefer == "seqclf":
        from transformers import AutoModelForSequenceClassification
        m = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, num_labels=1,
            attn_implementation="eager",
        )
        w = _validate_head(m.score.weight.detach(), "AutoModelForSequenceClassification")
        _maybe_cache(w, cache_path)
        return w.reshape(-1, 1).to(device)

    if prefer in ("auto", "range"):
        try:
            w = _validate_head(_fetch_head_via_range(), "range-fetch")
            _maybe_cache(w, cache_path)
            return w.reshape(-1, 1).to(device)
        except Exception as ex:
            if prefer == "range":
                raise
            print(f"[rm_head_utils] range fetch failed ({ex!r}); falling back to full model load")

    # last resort: full canonical load
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, num_labels=1, attn_implementation="eager",
    )
    w = _validate_head(m.score.weight.detach(), "AutoModelForSequenceClassification")
    _maybe_cache(w, cache_path)
    return w.reshape(-1, 1).to(device)


def _maybe_cache(w, cache_path):
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(w.reshape(1, 4096).cpu(), cache_path)
    except Exception:
        pass


if __name__ == "__main__":
    head = load_reward_head()
    print(f"score.weight -> shape {tuple(head.shape)}, ||w|| = {head.norm().item():.4f}")
