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

K_list = [0, 1, 5, 10, 15, 20, 25, 50]
V_final = None

alpha_list = [1e4]

N = len(train_seen)
N_unseen  = len(train_unseen)
print(N)
print(N_unseen)


# FIXED anchor extraction. The old code used AutoModel + "last nn.Linear", which for a
# LlamaForSequenceClassification checkpoint returns the bare backbone (score.weight is dropped)
# and grabs layers.31.mlp.down_proj -- an arbitrary MLP column, NOT the reward direction.
# load_reward_head() returns the TRUE Skywork reward head score.weight (shape [4096, 1]).
sys.path.append(SCRIPT_DIR)
from rm_head_utils import load_reward_head
V_final = load_reward_head(device=device)

# filename = f"./PRISM/V_ref.pt"
# torch.save(V_final, filename)

# Lock RNG here: after the (non-deterministic) model load and V_final
# extraction, and before run_regularized() solves the bases -- so normal
# training runs are reproducible, not just the dedicated check script.
# Seed 42 = the team-agreed canonical value (matches the reproducibility check).
set_seed(42)

train_accuracies_joint, seen_user_unseen_prompts_accuracies_joint, few_shot_train_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot, train_accuracies_joint_std, seen_user_unseen_prompts_accuracies_joint_std, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot_std = run_regularized(K_list, alpha_list, V_final, train_seen, test_seen, 
                train_unseen, test_unseen, N, N_unseen, device)

import matplotlib.pyplot as plt

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(K_list, seen_user_unseen_prompts_accuracies_joint, marker='o', linestyle='-', label="Seen Users")
plt.plot(K_list, unseen_user_unseen_prompts_accuracies_few_shot, marker='o', linestyle='-', label="Unseen Users")
plt.plot(K_list, train_accuracies_joint, marker='o', linestyle='-', label="Train Seen Users")
plt.plot(K_list, few_shot_train_accuracies_few_shot, marker='o', linestyle='-', label="Train Unseen Users Fewshot")
plt.xlabel('rank')
plt.ylabel('Accuracies')
plt.title('Generalization Accuracy vs. Rank')
plt.xticks(K_list, labels=["ref" if k==0 else str(k) for k in K_list])
plt.legend()

alpha = alpha_list[0]
# Save the plot
plt.savefig(f'./generalization_accuracy_vs_rank_lore_alpha_{alpha}.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()