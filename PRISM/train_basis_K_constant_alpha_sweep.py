# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
                    shape = torch.tensor(extra_info["chosen_conv_embedding"]).shape
                    chosen = torch.tensor(extra_info["chosen_conv_embedding"], dtype=torch.float32, device=device)
                    rejected = torch.tensor(extra_info["rejected_conv_embedding"], dtype=torch.float32, device=device)
                    grouped[user_id]["embeddings"].append(chosen - rejected)
        # Stack and sort by user_id
        sorted_grouped = []
        count = 0
        for user_id in sorted(grouped.keys()):
            # print(len(grouped[user_id]["embeddings"]))
            count += len(grouped[user_id]["embeddings"])
            sorted_grouped.append( 
                torch.stack(grouped[user_id]["embeddings"]))
        print(count)
        return sorted_grouped

    # Create all 4 groupings
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
import utils

# All outputs for this experiment (PNG + V/W matrices) go into one descriptive folder.
ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "artifacts", "K1_constant_alpha_sweep")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
# Redirect where run_regularized() saves its matrices. These overrides are backward-
# compatible (defaults in utils.py reproduce the original path), so the original
# train_basis.py is unaffected. Every saved file gets the "K1sweep_" prefix.
utils.SAVE_DIR = ARTIFACT_DIR
utils.SAVE_TAG = "K1sweep_"

# --- Rachel's disentangling experiment: fix K=1, sweep the effective regularization ---
# Original script: swept K with a FIXED alpha=1e4, so effective reg = alpha/K fell as K rose.
# Here: FIX K=1 and set alpha = 1e4/g for each g. At K=1 the effective reg is alpha/K = alpha
# = 1e4/g, which equals the ORIGINAL K=g run's effective reg (1e4/g). So the K=1 run at a given
# g is the twin of the original K=g run in terms of regularization pressure, but with only ONE
# basis. If the K=1 curve reproduces the original K-sweep curve, the gains came from weaker
# regularization, NOT from having more bases.
G_list = [1, 5, 10, 15, 20, 25, 50]   # "equivalent rank" values; each lines up with an original K
BASE_ALPHA = 1e4                       # the alpha the original runs used
K_list = [1]                           # <-- K is now FIXED at 1 for EVERY run
V_final = None

# one alpha per g, ordered to line up with G_list. run_regularized loops alpha (outer) x K (inner),
# so with K_list=[1] the results come back in this same order: results[i] <-> G_list[i].
alpha_list = [BASE_ALPHA / g for g in G_list]

N = len(train_seen)
N_unseen  = len(train_unseen)
print(N)
print(N_unseen)


from transformers import AutoModel

model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="eager",
    num_labels=1,
)

# Initialize a variable to store the last linear layer
last_linear_layer = None
# Iterate over the model's modules
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module
V_final = last_linear_layer.weight[:,0].to(device).to(torch.float32).reshape(-1, 1)

# filename = f"./PRISM/V_ref.pt"
# torch.save(V_final, filename)

train_accuracies_joint, seen_user_unseen_prompts_accuracies_joint, few_shot_train_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot, train_accuracies_joint_std, seen_user_unseen_prompts_accuracies_joint_std, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot_std = run_regularized(K_list, alpha_list, V_final, train_seen, test_seen, 
                train_unseen, test_unseen, N, N_unseen, device)

import matplotlib.pyplot as plt

# Plotting: x-axis is g (the "equivalent rank"). Each point at x=g is a K=1 run with
# alpha=1e4/g, and should be compared to the ORIGINAL PNG's rank=g point. The result
# arrays came back in G_list order (alpha outer x single K), so we plot against G_list.
plt.figure(figsize=(8, 5))
plt.plot(G_list, seen_user_unseen_prompts_accuracies_joint, marker='o', linestyle='-', label="Seen Users")
plt.plot(G_list, unseen_user_unseen_prompts_accuracies_few_shot, marker='o', linestyle='-', label="Unseen Users")
plt.plot(G_list, train_accuracies_joint, marker='o', linestyle='-', label="Train Seen Users")
plt.plot(G_list, few_shot_train_accuracies_few_shot, marker='o', linestyle='-', label="Train Unseen Users Fewshot")
plt.xlabel('g   (all runs are K=1 with alpha=1e4/g;  compare to original rank=g)')
plt.ylabel('Accuracies')
plt.title('K=1 with matched effective regularization (alpha = 1e4 / g)')
plt.xticks(G_list, labels=[str(g) for g in G_list])
plt.legend()

# Save the plot into the artifacts folder alongside the matrices
plt.savefig(os.path.join(ARTIFACT_DIR, 'generalization_accuracy_K1_alpha_sweep.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()