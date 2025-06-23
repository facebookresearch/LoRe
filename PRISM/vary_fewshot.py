# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import json
import matplotlib.pyplot as plt
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import *

from transformers import AutoModel, AutoTokenizer
import torchinfo

# Load model and tokenizer
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize a variable to store the last linear layer
last_linear_layer = None
# Iterate over the model's modules
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module
# Print the weights and bias of the last linear layer
V_final = last_linear_layer.weight[:,0].to(device).to(torch.float32).reshape(-1, 1)
print(V_final.shape)

file_path = "prism_split_ids_50.json"
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
        print("Data loaded successfully:")
except FileNotFoundError:
    print(f"File not found at {file_path}")
except json.JSONDecodeError:
    print(f"Failed to parse JSON at {file_path}")

train_embeddings = torch.load("prism_train_embeddings_50.pt")
test_embeddings = torch.load("prism_test_embeddings_50.pt")

seen_user_ids = data["seen_user_ids"].keys()
print(len(seen_user_ids))
unseen_user_ids = data["unseen_user_ids"].keys()
print(len(unseen_user_ids))
from sklearn.model_selection import train_test_split
# Join the two dictionaries
all_user_ids = {**data["seen_user_ids"], **data["unseen_user_ids"]}
num_shots = [1, 2, 3]
# Split the joined dictionary into training and testing sets
seen_user_ids, unseen_user_ids = train_test_split(list(all_user_ids.items()), test_size=0.5, random_state=42)
# Convert back to dictionaries
seen_user_ids = dict(seen_user_ids)
unseen_user_ids = dict(unseen_user_ids)
# Split the dictionary into two
seen_user_unseen_dialog_embeddings = {key: test_embeddings[key] for key in seen_user_ids if key in test_embeddings}
seen_user_unseen_dialog_embeddings = dict(sorted(seen_user_unseen_dialog_embeddings.items()))
unseen_user_unseen_dialog_embeddings = {key: value for key, value in test_embeddings.items() if key not in seen_user_ids}
unseen_user_unseen_dialog_embeddings = dict(sorted(unseen_user_unseen_dialog_embeddings.items()))


# Split the dictionary into two
seen_user_seen_dialog_embeddings = {key: train_embeddings[key] for key in seen_user_ids if key in train_embeddings}
seen_user_seen_dialog_embeddings = dict(sorted(seen_user_seen_dialog_embeddings.items()))
unseen_user_seen_dialog_embeddings = {key: value for key, value in train_embeddings.items() if key not in seen_user_ids}
unseen_user_seen_dialog_embeddings = dict(sorted(unseen_user_seen_dialog_embeddings.items()))
# train_embeddings = dict(sorted(train_embeddings.items()))

# create datasets
# train_features = create_dataset_prism_shots(seen_user_seen_dialog_embeddings, shots)
train_features = create_dataset_prism(seen_user_seen_dialog_embeddings)
N = len(train_features)
print(N)
test_features_sparse_unseen = create_dataset_prism(unseen_user_unseen_dialog_embeddings)
N_unseen = len(test_features_sparse_unseen)
print(N_unseen)
train_features_unseen = create_dataset_prism(unseen_user_seen_dialog_embeddings)
# test_features_sparse = create_dataset_prism_shots(seen_user_unseen_dialog_embeddings, shots)

K_list = [50]
alpha_list = [0]
trials = 1

# Initialize lists to store results
train_accuracies_joint = []
seen_user_unseen_prompts_accuracies_joint = []
few_shot_train_accuracies_few_shot = []
unseen_user_unseen_prompts_accuracies_few_shot = []
few_shot_train_accuracies_few_shot_std = []
unseen_user_unseen_prompts_accuracies_few_shot_std = []

for alpha in alpha_list:
    print("alpha : ", alpha)
    
    # Joint Reward and Weights Learning
    for K in K_list:
        print("K : ", K)
        if K == 0:
            V_joint = V_final
            W_joint = [torch.tensor([1.0]).to(device) for i in range(N)]
        else: 
            W_joint, V_joint = solve_regularized(V_final, alpha, train_features, K, num_iterations=500, learning_rate=0.5)
        
        print("Train Performance")
        accuracies_train = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], train_features)
        train_accuracies_joint.append(np.mean(accuracies_train))
    
        for shots in num_shots:
            train_features_unseen = create_dataset_prism_shots(unseen_user_seen_dialog_embeddings, shots)
            # Learn the w on unseen users with few shot interactions
            if K <= 1:
                W_few_shot = [torch.tensor([1.0]).to(device) for i in range(N_unseen)]
            else:
                W_few_shot = learn_multiple_few_shot(train_features_unseen, V_joint.detach(), num_iterations=500, learning_rate=0.1)
            print("Few Shot Train Performance")
            accuracies_few_shot_train = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], train_features_unseen)
            few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
            few_shot_train_accuracies_few_shot_std.append(np.std(accuracies_few_shot_train))
            
            print("Unseen User Unseen Prompts")
            accuracies_unseen_user_unseen_prompts = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], test_features_sparse_unseen)
            unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_user_unseen_prompts))


