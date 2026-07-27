"""
Interactive inspector for the reproduced LoRe basis/weight matrices.

Lets you (a) print the cosine-similarity table between every basis column and
every other, for a chosen K; (b) print the per-user weight rows WITH their real
user_ids; (c) track ONE user across every K run to see whether their effective
reward changes.

WHY user_ids are recoverable without rerunning: train_basis.py groups seen+train
users in sorted(user_id) order (line 27) and that same order indexes every W file,
for every K. So W row i is always the same person. This script reconstructs that
sorted list from the data pkl and matches it to the W rows (with an assert).

Examples:
  uv run python PRISM/inspect_basis.py                 # list available K + counts
  uv run python PRISM/inspect_basis.py --K 10          # V cosine table + W summary
  uv run python PRISM/inspect_basis.py --K 10 --show-users 20
  uv run python PRISM/inspect_basis.py --user user1434 # track one user across all K
"""
import argparse, json, os
import torch
import torch.nn.functional as F

HERE     = os.path.dirname(os.path.abspath(__file__))
MATDIR   = os.path.join(HERE, "reproduced_matrices-pruned")
DATA_PKL = os.path.join(HERE, "data", "prism", "train_embeddings.pkl")
ID_CACHE = os.path.join(MATDIR, "seen_train_user_ids.json")
ALPHA    = "10000.0"
KS       = [1, 5, 10, 15, 20, 25, 50]


def load_V(K):
    return torch.load(f"{MATDIR}/PRISM_V_lore_K_{K}_alpha_{ALPHA}.pt",
                      map_location="cpu", weights_only=False).float()


def load_W(K):
    return torch.load(f"{MATDIR}/PRISM_W_lore_seen_{K}_{ALPHA}.pt",
                      map_location="cpu", weights_only=False).float()


def seen_train_user_ids():
    """Sorted list of seen+train user_ids = the row order of every W file."""
    if os.path.exists(ID_CACHE):
        with open(ID_CACHE) as f:
            return json.load(f)
    print("(first run: reading data pkl to recover user_ids; this is cached after)")
    data = torch.load(DATA_PKL, weights_only=False)
    ids = set()
    for rec in data:
        ei = rec.get("extra_info", {})
        if ei.get("seen") and ei.get("split") == "train" and ei.get("user_id"):
            ids.add(ei["user_id"])
    ids = sorted(ids)
    with open(ID_CACHE, "w") as f:
        json.dump(ids, f)
    return ids


def print_cosine_table(K):
    V = load_V(K)
    Vn = F.normalize(V, dim=0)
    ncols = Vn.shape[1]
    print(f"\nV for K={K}: shape {tuple(V.shape)}  ({ncols} surviving column(s))")
    if ncols == 1:
        print("  only one column — nothing to compare (already fully collapsed).")
        return
    C = (Vn.T @ Vn)
    hdr = "        " + "".join(f"  c{j:<5}" for j in range(ncols))
    print("  |cos| between basis columns:")
    print(hdr)
    for i in range(ncols):
        row = "".join(f"  {abs(C[i, j].item()):5.2f} " for j in range(ncols))
        print(f"   c{i:<4} {row}")
    iu = torch.triu_indices(ncols, ncols, offset=1)
    offdiag = C[iu[0], iu[1]].abs()
    print(f"  off-diagonal mean |cos| = {offdiag.mean():.4f}  "
          f"(1.00 = all columns are the same direction = collapsed)")


def print_weights(K, n_show, user_ids):
    W = load_W(K)
    assert W.shape[0] == len(user_ids), (
        f"row/user mismatch: W has {W.shape[0]} rows but {len(user_ids)} seen-train "
        f"user_ids — the sorted-order assumption is broken, DON'T trust the mapping.")
    ncols = W.shape[1]
    peak = W.max(dim=1).values
    arg = W.argmax(dim=1)
    print(f"\nW for K={K}: shape {tuple(W.shape)}  "
          f"({W.shape[0]} users x {ncols} columns)")
    print(f"  mean peak weight {peak.mean():.4f}  (1.00 = winner-take-all)")
    counts = torch.bincount(arg, minlength=ncols)
    print("  users maxing on each column: " +
          ", ".join(f"c{j}:{counts[j].item()}" for j in range(ncols)))
    print(f"\n  first {n_show} users (user_id -> weights -> argmax col):")
    for i in range(min(n_show, W.shape[0])):
        w = ", ".join(f"{v:.3f}" for v in W[i].tolist())
        print(f"   {user_ids[i]:>10}  [{w}]   -> c{arg[i].item()}")


def track_user(user_id, user_ids):
    if user_id not in user_ids:
        print(f"user_id '{user_id}' not found among {len(user_ids)} seen-train users.")
        return
    idx = user_ids.index(user_id)
    print(f"\nTracking {user_id} (row {idx}) across all K:")
    dirs = {}
    for K in KS:
        W = load_W(K); V = load_V(K)
        w = W[idx]
        wstr = ", ".join(f"{v:.3f}" for v in w.tolist())
        dirs[K] = F.normalize(V @ w, dim=0)      # this user's 4096-d reward direction
        print(f"  K={K:<3} weights=[{wstr}]  argmax=c{w.argmax().item()}")
    print("\n  Is this user's actual REWARD DIRECTION (V @ w) changing across K?")
    print("  cosine of each K's reward direction vs K=1:")
    for K in KS:
        print(f"    K={K:<3} cos(dir_K, dir_1) = {(dirs[K] @ dirs[1]).item():.4f}")
    print("  (~1.00 everywhere = this user's reward is the same at every K)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, choices=KS, help="which run to inspect")
    ap.add_argument("--show-users", type=int, default=10,
                    help="how many user rows to print (with --K)")
    ap.add_argument("--user", type=str, help="track one user_id across all K")
    args = ap.parse_args()

    user_ids = seen_train_user_ids()

    if args.user:
        track_user(args.user, user_ids)
        return

    if args.K is None:
        print(f"seen-train users: {len(user_ids)}")
        print("available K runs:", KS)
        print("V column counts (surviving after pruning, threshold 1e-2):")
        for K in KS:
            print(f"   K={K:<3} -> {load_V(K).shape[1]} column(s)")
        print("\nrun with  --K <n>  to see a cosine table + weights,"
              "  or  --user <id>  to track a user.")
        return

    print_cosine_table(args.K)
    print_weights(args.K, args.show_users, user_ids)


if __name__ == "__main__":
    main()
