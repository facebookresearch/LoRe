# Copy of train_basis.py, with ONE change: instead of saving the pruned basis
# (only columns whose max user-weight >= 1e-2), we save the FULL V matrix with
# every basis column, plus the kept-mask so you can see which ones survived.
# Nothing about the training itself is changed. Outputs go to ./full_matrices/
# so none of your existing files get overwritten.

import torch
from collections import defaultdict

device = "cuda:0"

def group_embeddings_by_user(train_embeddings, test_embeddings, device):
    def process_dataset(dataset, seen_value, split_name):
        grouped = defaultdict(lambda: {"embeddings": []})
        for example in dataset:
            extra_info = example.get("extra_info", {})
            if extra_info.get("seen") == seen_value and extra_info.get("split") == split_name:
                user_id = extra_info.get("user_id")
                if user_id:
                    chosen = torch.tensor(extra_info["chosen_conv_embedding"], dtype=torch.float32, device=device)
                    rejected = torch.tensor(extra_info["rejected_conv_embedding"], dtype=torch.float32, device=device)
                    grouped[user_id]["embeddings"].append(chosen - rejected)
        sorted_grouped = []
        count = 0
        for user_id in sorted(grouped.keys()):
            count += len(grouped[user_id]["embeddings"])
            sorted_grouped.append(torch.stack(grouped[user_id]["embeddings"]))
        print(count)
        return sorted_grouped

    train_seen = process_dataset(train_embeddings, seen_value=True, split_name="train")
    train_unseen = process_dataset(train_embeddings, seen_value=False, split_name="train")
    test_seen = process_dataset(test_embeddings, seen_value=True, split_name="test")
    test_unseen = process_dataset(test_embeddings, seen_value=False, split_name="test")
    return train_seen, train_unseen, test_seen, test_unseen

train_embeddings = torch.load("data/prism/train_embeddings.pkl")
test_embeddings = torch.load("data/prism/test_embeddings.pkl")

train_seen, train_unseen, test_seen, test_unseen = group_embeddings_by_user(train_embeddings, test_embeddings, device)

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import *
import torch.nn.functional as F

K_list = [0, 1, 5, 10, 15, 20, 25, 50]
alpha_list = [1e4]

N = len(train_seen)
N_unseen = len(train_unseen)
print(N)
print(N_unseen)

from transformers import AutoModel, set_seed

model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="eager",
    num_labels=1,
)

last_linear_layer = None
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module
V_final = last_linear_layer.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)

# Same RNG lock as the original run.
set_seed(42)

# ---- the only new part: train each rank, save FULL (unpruned) V and W ----
outdir = "full_matrices"
os.makedirs(outdir, exist_ok=True)

for alpha in alpha_list:
    for K in K_list:
        if K == 0:
            continue  # rank 0 is just the reference direction, no basis to save
        print("\n==================== Rank :", K, "====================")

        # Build and train exactly like solve_regularized_simplex does.
        num_features = 4096
        model = LoRe_regularized(V_final, alpha, N, num_features, K, 20000, 0.5)
        _W_pruned, _V_pruned = model.train(train_seen)  # runs training; we ignore the pruned return

        # Full, UNPRUNED matrices straight off the trained model.
        full_V = model.V.detach().cpu()                     # [4096, K]  every basis column
        full_W = F.softmax(model.W, dim=1).detach().cpu()   # [N, K]     every user's full recipe

        # Which columns the original code WOULD have kept (max user weight >= 1e-2).
        max_per_basis = full_W.max(dim=0).values            # [K]
        kept_mask = (max_per_basis >= 1e-2)                 # bool [K]
        kept_idx = [i for i in range(K) if bool(kept_mask[i])]

        torch.save(full_V, f"{outdir}/PRISM_V_FULL_K_{K}_alpha_{alpha}.pt")
        torch.save(full_W, f"{outdir}/PRISM_W_FULL_K_{K}_alpha_{alpha}.pt")
        torch.save(kept_mask, f"{outdir}/PRISM_keptmask_K_{K}_alpha_{alpha}.pt")

        # Directional check over ALL K columns (including the dead ones), so you
        # can see whether even the pruned columns point the same way.
        Vn = F.normalize(full_V, dim=0)
        C = Vn.t() @ Vn
        offdiag = C[~torch.eye(K, dtype=torch.bool)].abs()
        print(f"saved FULL V shape {tuple(full_V.shape)} (all {K} columns, none pruned)")
        print(f"columns the weight-threshold WOULD keep: {kept_idx}  ({len(kept_idx)}/{K})")
        print(f"|cos| between ALL {K} columns: min={offdiag.min():.6f} max={offdiag.max():.6f}")

print("\nDone. Full matrices in ./full_matrices/")
