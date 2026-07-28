"""
================================================================================
GOAL OF THIS SCRIPT
================================================================================
Test the theory: "as K grows, the single direction that the V columns collapse to
drifts CLOSER to the real Skywork reward head."

To do that we need two things:
  (A) the REAL reward head vector  -> we download & extract it from HuggingFace
  (B) the "collapsed direction" of V at each K  -> we compute it from your saved
      V matrices, then measure the angle (cosine) between (A) and (B) for each K.

If the theory is right, that cosine should climb as K increases.

Nothing here needs a GPU. It only downloads ~one shard of the model (to get one
small vector) and then reads your already-saved V matrices from disk.
================================================================================
"""

import json
import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# ------------------------------------------------------------------------------
# CONFIG: where things are.
# ------------------------------------------------------------------------------
MODEL  = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"   # the HuggingFace repo (like a GitHub repo, but for a model)
KEY    = "score.weight"                               # the name of the reward-head weight inside that repo
ALPHA  = "10000.0"                                    # the alpha value baked into your V filenames
KS     = [1, 5, 10, 15, 20, 25, 50]                  # the K runs you saved

HERE   = os.path.dirname(os.path.abspath(__file__))              # folder this script lives in (.../PRISM)
MATDIR = os.path.join(HERE, "reproduced_matrices-pruned")       # where your saved V matrices live
HEAD_OUT = os.path.join(HERE, "reward_head_score.pt")           # where we'll cache the extracted head


# ==============================================================================
# PART A — download and extract the real reward head (score.weight).
# ==============================================================================
def get_reward_head():
    """
    Returns the real reward head as a 1-D tensor of length 4096.

    A HuggingFace model repo stores its weights across several big '.safetensors'
    shard files, plus a small 'index' file that says which weight is in which
    shard. We use that index so we can download ONLY the shard that holds the
    reward head, instead of the whole ~16GB model.
    """
    # If we already extracted it on a previous run, just reuse the saved copy.
    if os.path.exists(HEAD_OUT):
        head = torch.load(HEAD_OUT, map_location="cpu", weights_only=False).float()
        print(f"[A] reusing cached reward head from {HEAD_OUT}  shape {tuple(head.shape)}")
        return head.reshape(-1)   # flatten to a plain 4096-long vector

    # Step 1: download the tiny index file (a few KB). It's a JSON dict whose
    # 'weight_map' maps every weight name -> the shard file that contains it.
    index_path = hf_hub_download(MODEL, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]
    print(f"[A1] index downloaded. {len(weight_map)} weights listed.")

    # Step 2: look up which shard file holds 'score.weight'.
    if KEY not in weight_map:
        raise SystemExit(f"'{KEY}' not found in index. score-like keys present: "
                         f"{[k for k in weight_map if 'score' in k]}")
    shard = weight_map[KEY]
    print(f"[A2] '{KEY}' lives in shard: {shard}")

    # Step 3: download ONLY that one shard (reuses the local cache if you already
    # pulled the model earlier, so this may be instant).
    shard_path = hf_hub_download(MODEL, shard)
    print(f"[A3] shard ready at: {shard_path}")

    # Step 4: open the shard and pull out just 'score.weight'. safe_open lets us
    # read a single named tensor WITHOUT building the model or loading the rest.
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        score = f.get_tensor(KEY).float()      # shape (1, 4096)
    print(f"[A4] extracted '{KEY}': shape {tuple(score.shape)}")

    # Step 5: save a (4096, 1) copy to disk so future runs skip the download,
    # and return it flattened to a 4096-long vector for the math below.
    torch.save(score.reshape(-1, 1), HEAD_OUT)
    print(f"[A5] saved to {HEAD_OUT}")
    return score.reshape(-1)


# ==============================================================================
# PART B — for each K, find the direction V collapsed to, and compare to the head.
# ==============================================================================
def load_V(K):
    """Load your saved V matrix for a given K. Shape is [4096 features, ncols]."""
    path = f"{MATDIR}/PRISM_V_lore_K_{K}_alpha_{ALPHA}.pt"
    # map_location='cpu' because these were saved on a GPU; weights_only=False
    # because they were saved as full tensors, not a state-dict.
    return torch.load(path, map_location="cpu", weights_only=False).float()


def collapsed_direction(V):
    """
    Return the single 4096-long direction that V's columns collapse onto.

    Because the columns are nearly parallel (that's the 'collapse'), they all point
    the same way. The cleanest way to extract that shared direction is an SVD:
    the top-left singular vector U[:, 0] is the single direction that captures the
    most of V's energy — i.e. the direction the columns pile onto. It comes out
    already normalized to length 1.
    """
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)   # U: [4096, ncols]
    return U[:, 0]                                         # [4096], unit length


def main():
    head = get_reward_head()                 # (A) the real reward head, length 4096
    head = F.normalize(head, dim=0)          # make it unit length so cosine is clean

    print("\n[B] drift of the collapsed V direction toward the real reward head:")
    print("    (cosine near 0 = perpendicular/unrelated; near 1 = same direction)")
    print("    NOTE: the sign of an SVD direction is arbitrary, so what matters is")
    print("    the MAGNITUDE of the cosine and whether it grows with K.\n")

    for K in KS:
        V = load_V(K)                        # [4096, ncols] for this K
        d = collapsed_direction(V)           # the direction its columns collapsed to
        cos = torch.dot(d, head).item()      # cosine between the two unit vectors
        print(f"    K={K:<3} ncols={V.shape[1]:<2} "
              f"cos(collapsed dir, reward head) = {cos:+.4f}   |cos| = {abs(cos):.4f}")

    print("\n    If |cos| climbs with K, that supports the drift theory.")
    print("    If it stays flat/tiny, the collapsed direction is NOT moving toward")
    print("    the real head — it may be drifting toward the data direction instead.")


if __name__ == "__main__":
    main()
