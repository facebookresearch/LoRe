# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Where run_regularized() saves the V/W matrices, and an optional filename tag.
# Defaults reproduce the ORIGINAL behavior exactly (same path, no tag), so the
# original train_basis.py is unaffected. An experiment script can override these
# (e.g. utils.SAVE_DIR = "...", utils.SAVE_TAG = "myexp_") to redirect its outputs.
SAVE_DIR = "/checkpoint/ai_society/representative_llms/data/lore/community"
SAVE_TAG = ""


def set_seed(s):
    """Lock every RNG so basis initialization (and the learned bases) is
    deterministic. Call this after loading the backbone model and immediately
    before solving, so the basis init RNG state is not perturbed by earlier,
    non-deterministic model loading."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def simulate_user(reward_tensor, features, w):
    num_prompts = len(reward_tensor)
    feature_diff = []
    for i in range(num_prompts):
        scores = np.dot(reward_tensor[i], w)
        # Find the index of the response with the largest and smallest score
        largest_score_index = np.argmax(scores)
        smallest_score_index = np.argmin(scores)
        feature_diff.append(features[i][largest_score_index] - features[i][smallest_score_index])
    feature_diff = torch.stack(feature_diff, dim=0)
    return feature_diff

def evaluate_model(X, V, w):
    # Compute the expression X @ V @ w
    X = torch.tensor(X, dtype=torch.float32)
    # result = X @ V @ w
    result = X @ V @ w
    # Count the number of positive elements
    num_positive = (result > 0).sum().item()
    # Compute the fraction of positive elements
    fraction_positive = num_positive / result.numel()
    return fraction_positive

def simulate_population(reward_tensor, features, W):
    all_feature_diff = [simulate_user(reward_tensor, features, w) for w in W]
    return torch.stack(all_feature_diff, dim=0)

def generate_popupulation(alpha, N):
    return np.random.dirichlet(alpha, N)

def create_sparse_tensor(dense_tensor, sample_percentage):
    """
    Creates a sparse tensor by randomly sampling entries from a dense tensor.
    Args:
        dense_tensor (torch.Tensor): The input dense tensor.
        sample_percentage (float): The percentage of entries to sample per row.
    Returns:
        torch.Tensor: The resulting sparse tensor.
    """
    # Get the shape of the dense tensor
    N, M, d = dense_tensor.shape
    # Calculate the number of samples per row
    num_samples_per_row = int(sample_percentage * M)
    # Create a list to store the sparse rows
    sparse_rows = []
    # Iterate over each row of the dense tensor
    for i in range(N):
        # Randomly select indices for sampling
        indices = np.random.choice(M, num_samples_per_row, replace=False)
        
        # Sample values from the dense tensor
        values = dense_tensor[i, indices]
        
        # Append the sampled values to the list of sparse rows
        # sparse_rows.append(values.to(device))
        sparse_rows.append(torch.tensor(values, dtype=torch.float32).to(device))
    return sparse_rows

def create_dataset_prism(embeddings):
    sparse_rows = []
    for user_id, dialogs in embeddings.items():
        values = None
        for dialog_id, examples in dialogs.items():
            for i in range(len(examples["chosen"])):
                chosen = torch.tensor(embeddings[user_id][dialog_id]["chosen"][i], dtype=torch.float32, device=device)
                rejected = torch.tensor(embeddings[user_id][dialog_id]["rejected"][i], dtype=torch.float32, device=device)
                diff = (chosen - rejected).reshape(1, -1)
                if values is None:
                    values = diff
                else:
                    values = torch.cat((values, diff), dim=0)
        sparse_rows.append(values)
    return sparse_rows

def create_dataset_prism_shots(embeddings, shots):
    sparse_rows = []
    for user_id, dialogs in embeddings.items():
        values = None
        idx = random.sample([i for i in range(len(dialogs))], shots)
        j = 0
        for dialog_id, examples in dialogs.items():
            if j in idx:
                for i in range(len(examples["chosen"])):
                    chosen = torch.tensor(embeddings[user_id][dialog_id]["chosen"][i], dtype=torch.float32, device=device)
                    rejected = torch.tensor(embeddings[user_id][dialog_id]["rejected"][i], dtype=torch.float32, device=device)
                    diff = (chosen - rejected).reshape(1, -1)
                    if values is None:
                        values = diff
                    else:
                        values = torch.cat((values, diff), dim=0)
            j += 1       
        sparse_rows.append(values)
    return sparse_rows

def learn_multiple(train_features, num_iterations=1000, learning_rate=0.01):
    W_list = []
    V_list = []
    num_features = train_features[0][0].shape[0]
    N = len(train_features)
    for i in range(N):
        am = AlternatingMinimization(1, num_features, 1, num_iterations, learning_rate).to(device)
        w, V = am.train([train_features[i]])
        W_list.append(w[0])
        V_list.append(V.detach())
    return W_list, V_list

def learn_multiple_few_shot(train_features, V, num_iterations=1000, learning_rate=0.01):
    N = len(train_features)
    num_features = train_features[0][0].shape[0]
    fitw = PersonalizeBatch(N, num_features, V.shape[1], num_iterations, learning_rate).to(device)
    W = fitw.train(train_features, V)
    return W

def learn_multiple_few_shot_weighted(alpha, train_features, current_dialog_features, V, num_iterations=1000, learning_rate=0.01):
    N = len(train_features)
    num_features = train_features[0][0].shape[0]
    fitw = PersonalizeBatch_weighted(alpha, N, num_features, V.shape[1], num_iterations, learning_rate).to(device)
    # W = [fitw.train([train_features[i]], V)[0] for i in range(N)] 
    W = fitw.train(train_features, current_dialog_features, V)
    return W

def eval_multiple(W_list, V_list, test_features):
    accuracies = []
    N = len(test_features)
    accuracies = [evaluate_model(test_features[i], V_list[i], W_list[i]) for i in range(N)]
    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Average accuracy: {average_accuracy:.4f}")
    print(f"Standard deviation of accuracy: {std_accuracy:.4f}")
    return accuracies

def solve(train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    num_classes = len(train_features)
    num_features = train_features[0][0].shape[0]
    am = AlternatingMinimization(num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    W, V = am.train(train_features)
    return W, V.detach()

def solve_regularized(V_sft, alpha, train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    num_classes = len(train_features)
    num_features = train_features[0][0].shape[0]
    am = LoRe(V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    W, V = am.train(train_features)
    return W, V.detach()

def solve_regularized_simplex(V_sft, alpha, train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    num_classes = len(train_features)
    num_features = 4096
    am = LoRe_regularized(V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    W, V = am.train(train_features)
    return W, V.detach()

def solve_multi_reward(train_features, num_basis_vectors, num_iterations=1000, learning_rate=0.01):
    num_classes = len(train_features)
    num_features = train_features[0][0].shape[0]
    rm = MultiRewardModel(num_classes, num_features, num_basis_vectors, num_iterations, learning_rate)
    rm.train(train_features)
    return rm

def learn_single_reward(train_features, num_iterations=1000, learning_rate=0.01):
    num_features = train_features[0][0].shape[0]
    N = len(train_features)
    sm = SingleRewardModel(N, num_features, 1, num_iterations, learning_rate).to(device)
    V = sm.train(train_features)
    return V.detach()

def learn_single_reward_regularized(V_ref, alpha, train_features, num_iterations=1000, learning_rate=0.01):
    num_features = train_features[0][0].shape[0]
    N = len(train_features)
    sm = SingleRewardModel_regularized(V_ref, alpha, N, num_features, 1, num_iterations, learning_rate).to(device)
    V = sm.train(train_features)
    return V.detach()

class LoRe_regularized(nn.Module):
    def __init__(
        self, V_sft, alpha, num_classes, num_features, num_basis_vectors,
        num_iterations, learning_rate
    ):
        super().__init__()
        self.V_sft = V_sft.to(device)
        self.V_sft_norm = F.normalize(self.V_sft, dim=0)   # normalize once
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.W = nn.Parameter(torch.rand(num_classes, num_basis_vectors, device=device))
        self.V = nn.Parameter(torch.randn(num_features, num_basis_vectors, device=device))

    # --- NEW: pack once ---
    @staticmethod
    def _prepare_batch(X):
        """
        X: list of length C; X[i] is [m_i, F]
        Returns:
          X_cat: [N, F], y: [N] (values in 0..C-1)
        """
        x_list, y_list = [], []
        for i, x in enumerate(X):
            x_list.append(x)
            y_list.append(torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))
        X_cat = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return X_cat, y

    def _forward_from_packed(self, X_cat, y, alpha_curr):
        # choose which parameter set to freeze for this pass
        V_used = self.V
        # V_used = F.normalize(self.V, dim=0)
        W_logits = self.W

        W_row = F.softmax(W_logits, dim=1)    # [C, B]
        Vw    = V_used @ W_row.T              # [F, C]

        logits_all = (X_cat @ Vw) / 100.0     # [N, C]
        logits = logits_all.gather(1, y.unsqueeze(1)).squeeze(1)
        nll = -F.logsigmoid(logits).mean()

        # V-alignment reg should only act when we're updating V
        reg = 0.0
        if alpha_curr > 0:   
            V_norm = F.normalize(self.V, dim=0)
            V_sft_norm = F.normalize(self.V_sft, dim=0)
            cos_sim = (V_norm * V_sft_norm).sum(dim=0)
            reg = torch.mean(1 - cos_sim)

        # # Diversity reg should only act when we're updating W
        entropy_loss = 0.0
        # entropy_loss = self._diversity_loss_rows()

        return nll, reg, entropy_loss


    # keep a compatibility wrapper (packs every call)
    def forward(self, X, alpha_curr):
        X_cat, y = self._prepare_batch(X)
        return self._forward_from_packed(X_cat, y, alpha_curr)

    def _alpha_at_step(self, step: int) -> float:
        warmup_start = int(0.2 * self.num_iterations)
        warmup_end   = int(0.8 * self.num_iterations)
        if step < warmup_start: return 0.0
        if step >= warmup_end:  return float(self.alpha)
        return float(self.alpha) * (step - warmup_start) / (warmup_end - warmup_start)

    def train(self, X):
        self.to(device)
        X_cat, y = self._prepare_batch(X)
        X_cat = X_cat.to(device, non_blocking=True)
        y     = y.to(device, non_blocking=True)

        optimizer_W = optim.Adam([self.W], lr=self.learning_rate)
        optimizer_V = optim.Adam([self.V], lr=self.learning_rate)

        for step in range(self.num_iterations):
            alpha_curr = self._alpha_at_step(step)

            # ---- Update W: freeze V ----
            optimizer_W.zero_grad()
            nll_W, _, _ = self._forward_from_packed(
                X_cat, y, alpha_curr=0.0)
            
            # loss_W = nll_W + self.entropy_weight * entropy_loss
            nll_W.backward()
            optimizer_W.step()

            # ---- Update V: freeze W ----
            optimizer_V.zero_grad()
            nll_V, reg, _ = self._forward_from_packed(
                X_cat, y, alpha_curr=alpha_curr
            )
            total_loss_V = nll_V + alpha_curr * reg
            total_loss_V.backward()
            optimizer_V.step()

            if (step + 1) == self.num_iterations:
                W_sm = F.softmax(self.W, dim=1)
                print(f"W mean per dim: {W_sm.mean(dim=0).detach().cpu().numpy()}")
                print(f"W std  per dim: {W_sm.std(dim=0).detach().cpu().numpy()}")
                # L2 norms of V columns (parameter) and of the normalized V used in forward
                with torch.no_grad():
                    V_param_norms = torch.linalg.vector_norm(self.V, ord=2, dim=0)
                print(f"||V[:, i]|| (param): {V_param_norms.detach().cpu().numpy()}")

                print(
                    f"Step {step}: "
                    f"NLL(W)={nll_W.item():.4f}, "
                    f"NLL(V)={nll_V.item():.4f}, "
                    f"Reg={float(reg):.4f}, "
                    f"Alpha={alpha_curr:.4f}, "
                )
        
        # ---- Return only directions with min_c softmax(W)[c, i] >= 1e-2 ----
        W_probs = F.softmax(self.W, dim=1)                   # [C, B]
        max_per_basis = W_probs.max(dim=0).values            # [B]
        print(max_per_basis)
        mask = (max_per_basis >= 1e-2)                       # bool[B]

        W_kept = W_probs[:, mask]                            # [C, B_kept]
        V_kept = self.V[:, mask]                             # [F, B_kept]
        num_kept = int(mask.sum().item())
        print(f"Num dimensions kept: {num_kept}/{self.num_basis_vectors} (threshold=1e-2)")

        print(f"W mean per dim: {W_kept.mean(dim=0).detach().cpu().numpy()}")
        print(f"W std  per dim: {W_kept.std(dim=0).detach().cpu().numpy()}")

        return W_kept, V_kept
                
        # return F.softmax(self.W, dim=1), self.V

class LoRe(nn.Module):
    def __init__(self, V_sft, alpha, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate):
        super(LoRe, self).__init__()
        self.V_sft = V_sft
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        # Initialize weight vectors and matrix V
        # self.w = [nn.Parameter(torch.randn(num_basis_vectors)) for _ in range(num_classes)]
        self.W = nn.Parameter(torch.randn(num_classes, num_basis_vectors))
        self.V = nn.Parameter(torch.randn(num_features, num_basis_vectors))
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, X):
        nll = 0 
        # V_w = self.V @ (torch.stack(self.w).T).to(device)
        # V_w = self.V @ (self.W).T 
        V_w = self.V @ (F.softmax(self.W, dim=1)).T 
        # Compute the log-likelihood function
        i = 0
        for x in X:
            # x = torch.tensor(x, dtype=torch.float32)
            # logits =  x @ V_w[:,i] / 100.0
            logits =  x @ V_w[:,i] / 100.0
            # print(logits)
            log_likelihood = torch.log(torch.sigmoid(logits))
            nll +=  ((-log_likelihood.sum()) / len(x))
            # if self.alpha > 0:
            #     probs = F.softmax(self.W[i,:])
            #     entropy = -torch.sum(probs * torch.log(probs))
            #     nll += self.alpha * entropy
            i += 1
        
        reg = 0
        if self.alpha > 0:
            for j in range(self.num_basis_vectors):
                # print(self.V[:,j].shape)
                # print(self.V_sft.shape)
                reg += self.alpha * torch.sum((self.V[:,j] - self.V_sft)**2)
        return nll, reg
    
    
    def train(self, x):
        # Move the model and data to the GPU
        self.to(device)
        # x = [torch.tensor(i, dtype=torch.float32).to(device) for i in x]
        # Train the model using alternating minimization
        for j in range(self.num_iterations):
            # print("Iter : ", j)
            # Update weight vectors
            # for i in range(len(x)):
            self.optimizer.zero_grad()
            loss, reg = self.forward(x)
            regularized_loss = loss + reg
            regularized_loss.backward()
            self.optimizer.step()
            # print(loss.item())
            # if j % 100 == 0:
            #     print("Iter : ", j)
            #     print(loss.item())
        
        # return self.w, self.V
        # return self.W, self.V
        return (F.softmax(self.W, dim=1)), self.V

class PersonalizeBatch(nn.Module):
    def __init__(self, num_classes, num_features, num_basis_vectors, num_iterations, learning_rate):
        super(PersonalizeBatch, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        # Initialize weight vectors and matrix V
        self.w = nn.ParameterList([nn.Parameter(torch.randn(num_basis_vectors)) for _ in range(num_classes)])
        
        # print(self.parameters())
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, X, V):
        nll = 0 
        # Compute the log-likelihood function
        i = 0
        for x in X:
            # V_w = V @ self.w[i]
            V_w = V @ F.softmax(self.w[i]) 
            # x = torch.tensor(x, dtype=torch.float32)
            logits =  x @ V_w / 100.0
            # print(logits)
            log_likelihood = torch.log(torch.sigmoid(logits))
            nll +=  ((-log_likelihood.sum()) / len(x))
            i += 1
        return nll
    
    def train(self, X, V):
        # Train the model using alternating minimization
        for j in range(self.num_iterations):
            
            # Update weight vectors
            # for i in range(len(x)):
            self.optimizer.zero_grad()
            loss = self.forward(X, V)
            loss.backward()
            self.optimizer.step()
            # if j % 100 == 0:
            #     print("Iter : ", j)
            #     print(loss.item())
        
        return [F.softmax(self.w[i]).detach() for i in range(len(X))]

def run(K_list, alpha_list, V_final, train_features, test_features_sparse, 
                       train_features_unseen, test_features_sparse_unseen, N, N_unseen, device):
    """
    Compute accuracies for joint and few-shot learning.

    Parameters:
    K_list (list): List of values for K.
    alpha_list (list): List of values for alpha.
    V_final (tensor): Final value of V.
    train_features (tensor): Training features.
    test_features_sparse (tensor): Test features for seen users.
    train_features_unseen (tensor): Training features for unseen users.
    test_features_sparse_unseen (tensor): Test features for unseen users.
    N (int): Number of seen users.
    N_unseen (int): Number of unseen users.
    device (device): Device to use for computations.

    Returns:
    tuple: Tuple containing 9 numpy arrays with computed accuracies and standard deviations.
    """

    # Initialize lists to store results
    train_accuracies_joint = []
    seen_user_unseen_prompts_accuracies_joint = []
    few_shot_train_accuracies_few_shot = []
    unseen_user_unseen_prompts_accuracies_few_shot = []
    train_accuracies_joint_std = []
    seen_user_unseen_prompts_accuracies_joint_std = []
    few_shot_train_accuracies_few_shot_std = []
    unseen_user_unseen_prompts_accuracies_few_shot_std = []

    for alpha in alpha_list:
        # print("alpha : ", alpha)

        # Joint Reward and Weights Learning
        for K in K_list:
            print("K : ", K)
            if K == 0:
                V_joint = V_final
                W_joint = [torch.tensor([1.0]).to(device) for i in range(N)]
            else: 
                W_joint, V_joint = solve_regularized(V_final, alpha, train_features, K, num_iterations=1000, learning_rate=0.5)

            print("Train Performance")
            accuracies_train = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], train_features)
            train_accuracies_joint.append(np.mean(accuracies_train))
            train_accuracies_joint_std.append(np.std(accuracies_train))

            print("Seen User Unseen Prompts")
            accuracies_seen_user_unseen_prompts = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], test_features_sparse)
            seen_user_unseen_prompts_accuracies_joint.append(np.mean(accuracies_seen_user_unseen_prompts))
            seen_user_unseen_prompts_accuracies_joint_std.append(np.std(accuracies_seen_user_unseen_prompts))

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
            unseen_user_unseen_prompts_accuracies_few_shot_std.append(np.std(accuracies_unseen_user_unseen_prompts))

    fac = 0.25
    train_accuracies_joint = np.array(train_accuracies_joint)
    seen_user_unseen_prompts_accuracies_joint = np.array(seen_user_unseen_prompts_accuracies_joint)
    few_shot_train_accuracies_few_shot = np.array(few_shot_train_accuracies_few_shot)
    unseen_user_unseen_prompts_accuracies_few_shot = np.array(unseen_user_unseen_prompts_accuracies_few_shot)
    train_accuracies_joint_std = fac * np.array(train_accuracies_joint_std)
    seen_user_unseen_prompts_accuracies_joint_std = fac * np.array(seen_user_unseen_prompts_accuracies_joint_std)
    few_shot_train_accuracies_few_shot_std = fac * np.array(few_shot_train_accuracies_few_shot_std)
    unseen_user_unseen_prompts_accuracies_few_shot_std = fac * np.array(unseen_user_unseen_prompts_accuracies_few_shot_std)

    return train_accuracies_joint, seen_user_unseen_prompts_accuracies_joint, few_shot_train_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot, train_accuracies_joint_std, seen_user_unseen_prompts_accuracies_joint_std, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot_std


def sample_shots(train_features_unseen, shots):
    """
    Sample 'shots' number of tensors from each tensor in train_features_unseen.
    Args:
        train_features_unseen (list): A list of tensors.
        shots (int): The number of samples to take from each tensor.
    Returns:
        list: A list of sampled tensors.
    """
    # Check if shots is not greater than the size of any tensor
    # min_size = min(tensor.size(0) for tensor in train_features_unseen)
    # if shots > min_size:
    #     raise ValueError("Shots cannot be greater than the size of any tensor.")
    # Sample shots number of elements from each tensor
    sampled_features = [tensor[torch.randperm(tensor.size(0))[:shots]] for tensor in train_features_unseen]
    return sampled_features

def run_regularized(K_list, alpha_list, V_final, train_features, test_features_sparse, 
                       train_features_unseen, test_features_sparse_unseen, N, N_unseen, device):
    """
    Compute accuracies for joint and few-shot learning.

    Parameters:
    K_list (list): List of values for K.
    alpha_list (list): List of values for alpha.
    V_final (tensor): Final value of V. 
    train_features (tensor): Training features.
    test_features_sparse (tensor): Test features for seen users.
    train_features_unseen (tensor): Training features for unseen users.
    test_features_sparse_unseen (tensor): Test features for unseen users.
    N (int): Number of seen users.
    N_unseen (int): Number of unseen users.
    device (device): Device to use for computations.

    Returns:
    tuple: Tuple containing 9 numpy arrays with computed accuracies and standard deviations.
    """

    # Initialize lists to store results
    train_accuracies_joint = []
    seen_user_unseen_prompts_accuracies_joint = []
    few_shot_train_accuracies_few_shot = []
    unseen_user_unseen_prompts_accuracies_few_shot = []
    train_accuracies_joint_std = []
    seen_user_unseen_prompts_accuracies_joint_std = []
    few_shot_train_accuracies_few_shot_std = []
    unseen_user_unseen_prompts_accuracies_few_shot_std = []

    for alpha in alpha_list:
        print("alpha : ", alpha)

        # Joint Reward and Weights Learning
        for K in K_list:
            print("Rank : ", K)
            if K == 0:
                V_joint = V_final
                W_joint = [torch.tensor([1.0]).to(device) for i in range(N)]
            else: 
                W_joint, V_joint = solve_regularized_simplex(V_final, alpha, train_features, K, num_iterations= 20000, learning_rate=0.5)
            
                # Save V_joint to file
                filename = f"{SAVE_DIR}/{SAVE_TAG}PRISM_V_lore_K_{K}_alpha_{alpha}.pt"
                torch.save(V_joint, filename)
                # Save W_joint to file
                filename = f"{SAVE_DIR}/{SAVE_TAG}PRISM_W_lore_seen_{K}_{alpha}.pt"
                torch.save(W_joint.detach().cpu(), filename)

            print("Train Performance")
            accuracies_train = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], train_features)
            train_accuracies_joint.append(np.mean(accuracies_train))
            train_accuracies_joint_std.append(np.std(accuracies_train))

            print("Seen User Unseen Prompts")
            accuracies_seen_user_unseen_prompts = eval_multiple(W_joint, [V_joint.detach() for i in range(N)], test_features_sparse)
            seen_user_unseen_prompts_accuracies_joint.append(np.mean(accuracies_seen_user_unseen_prompts))
            seen_user_unseen_prompts_accuracies_joint_std.append(np.std(accuracies_seen_user_unseen_prompts))

            # Learn the w on unseen users with few shot interactions
            if K <= 1:
                W_few_shot = [torch.tensor([1.0]).to(device) for i in range(N_unseen)]
            else:
                W_few_shot = learn_multiple_few_shot(train_features_unseen, V_joint.detach(), num_iterations=500, learning_rate=0.5)

            # Save W_joint to file
            # filename = f"checkpoints/W_lore_unseen_{K}.pt"
            # torch.save(torch.stack(W_few_shot).detach().cpu(), filename)

            print("Few Shot Train Performance")
            accuracies_few_shot_train = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], train_features_unseen)
            few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
            few_shot_train_accuracies_few_shot_std.append(np.std(accuracies_few_shot_train))

            print("Unseen User Unseen Prompts")
            accuracies_unseen_user_unseen_prompts = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], test_features_sparse_unseen)
            unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_user_unseen_prompts))
            unseen_user_unseen_prompts_accuracies_few_shot_std.append(np.std(accuracies_unseen_user_unseen_prompts))

    fac = 0.25
    train_accuracies_joint = np.array(train_accuracies_joint)
    seen_user_unseen_prompts_accuracies_joint = np.array(seen_user_unseen_prompts_accuracies_joint)
    few_shot_train_accuracies_few_shot = np.array(few_shot_train_accuracies_few_shot)
    unseen_user_unseen_prompts_accuracies_few_shot = np.array(unseen_user_unseen_prompts_accuracies_few_shot)
    train_accuracies_joint_std = fac * np.array(train_accuracies_joint_std)
    seen_user_unseen_prompts_accuracies_joint_std = fac * np.array(seen_user_unseen_prompts_accuracies_joint_std)
    few_shot_train_accuracies_few_shot_std = fac * np.array(few_shot_train_accuracies_few_shot_std)
    unseen_user_unseen_prompts_accuracies_few_shot_std = fac * np.array(unseen_user_unseen_prompts_accuracies_few_shot_std)

    return train_accuracies_joint, seen_user_unseen_prompts_accuracies_joint, few_shot_train_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot, train_accuracies_joint_std, seen_user_unseen_prompts_accuracies_joint_std, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot_std

def run_few_shot_vary_shots(trials, alpha_list, K_list, num_shots, train_features, train_features_unseen, test_features_sparse_unseen, V_final, N, N_unseen, device):
    all_results = {}
    
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
            train_accuracies_joint = np.mean(accuracies_train)
            
            few_shot_train_accuracies_few_shot_means = []
            few_shot_train_accuracies_few_shot_stds = []
            unseen_user_unseen_prompts_accuracies_few_shot_means = []
            unseen_user_unseen_prompts_accuracies_few_shot_stds = []
            
            for shots in num_shots:
                print("Shots : ", shots)
                few_shot_train_accuracies_few_shot = []
                unseen_user_unseen_prompts_accuracies_few_shot = []
                
                for _ in range(trials):  # Run the experiment 10 times
                    # train_features_unseen = create_dataset_prism_shots(unseen_user_seen_dialog_embeddings, shots)
                    train_features_unseen_shots = sample_shots(train_features_unseen, shots)
                    # Learn the w on unseen users with few shot interactions
                    if K <= 1:
                        W_few_shot = [torch.tensor([1.0]).to(device) for i in range(N_unseen)]
                    else:
                        W_few_shot = learn_multiple_few_shot(train_features_unseen_shots, V_joint.detach(), num_iterations=500, learning_rate=0.1)
                    
                    print("Few Shot Train Performance")
                    accuracies_few_shot_train = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], train_features_unseen_shots)
                    few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
                    
                    print("Unseen User Unseen Prompts")
                    accuracies_unseen_user_unseen_prompts = eval_multiple(W_few_shot, [V_joint.detach() for i in range(N_unseen)], test_features_sparse_unseen)
                    unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_user_unseen_prompts))
                
                few_shot_train_accuracies_few_shot_means.append(np.mean(few_shot_train_accuracies_few_shot))
                few_shot_train_accuracies_few_shot_stds.append(np.std(few_shot_train_accuracies_few_shot))
                unseen_user_unseen_prompts_accuracies_few_shot_means.append(np.mean(unseen_user_unseen_prompts_accuracies_few_shot))
                unseen_user_unseen_prompts_accuracies_few_shot_stds.append(np.std(unseen_user_unseen_prompts_accuracies_few_shot))

    return few_shot_train_accuracies_few_shot_means, few_shot_train_accuracies_few_shot_stds, unseen_user_unseen_prompts_accuracies_few_shot_means, unseen_user_unseen_prompts_accuracies_few_shot_stds


# =====================================================================================
# LoRe v2 -- redesigned parameterization.
#
# Everything below is ADDITIVE. LoRe_regularized above is the vanilla baseline that every
# result in CONTRIBUTIONS.md came from; it is deliberately left untouched so the two can be
# run head-to-head.
#
# Changes vs vanilla:
#   1. user weights are SIGNED and unnormalized (vanilla: softmax -> simplex, so every user
#      had to weight every basis non-negatively, which forbids bases that some users like and
#      others dislike -- exactly the structure personalization needs)
#   2. weights factor as wbar + delta_u: a shared population direction plus individual variation
#   3. regularization is ridge in REWARD-FUNCTION space, not cosine-to-V_sft (the old target
#      pulled toward the pretrained direction, incentivized collapse, and scaled with K)
#   4. no basis-column dropping (meaningless once weights are signed and shrunk toward zero)
#   5. joint Adam over {V, wbar, delta} instead of alternating, no fixed logit rescaling,
#      and early stopping when the objective plateaus
# =====================================================================================

def _pack_users(X):
    """list of [m_i, F] per-user diff tensors -> (X_cat [N, F], uid [N]).

    Same packing as LoRe_regularized._prepare_batch, kept separate so the two models stay
    independent.
    """
    x_list, u_list = [], []
    for i, x in enumerate(X):
        x_list.append(x)
        u_list.append(torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))
    return torch.cat(x_list, dim=0), torch.cat(u_list, dim=0)


class LoReV2(nn.Module):
    """Low-rank personalized reward model with signed weights and a population/individual split.

    Reward for user u on feature diff x:  x @ V @ (wbar + delta_u)

    Loss (every term a mean, so the scale does not drift with K or with the number of users):

        nll = -logsigmoid(x @ V @ (wbar + delta_u)).mean()          pooled over all pairs
        reg = lam_pop * ||V @ wbar||^2 + lam_d * mean_u ||V @ delta_u||^2
        loss = nll + reg

    The penalty is ridge regression in reward-function space: it constrains the reward functions
    the model can express, not the raw parameters. Note this makes it invariant to the rescaling
    V -> cV, w -> w/c, which is intended -- but it also means the individual basis columns are
    identified only up to an invertible transform. Use canonical_variation_axes() below to get
    interpretable axes out of a fitted model.

    The nll pools all pairs, so users with more data contribute more (this matches vanilla LoRe;
    cap pairs per user upstream if you want equal weighting).
    """

    def __init__(self, num_users, num_features, num_basis_vectors, lam_pop=0.01, lam_d=0.01,
                 num_iterations=2000, learning_rate=1e-2, patience=100, tol=1e-5, verbose=True):
        super().__init__()
        self.num_users = num_users
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.lam_pop = lam_pop
        self.lam_d = lam_d
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose

        # Unit-norm basis columns. Vanilla used randn (column norm ~ sqrt(F) ~ 64 at F=4096) and
        # then divided the logits by 100 to compensate. The fixed rescaling is gone here, so the
        # magnitude has to be sane at init instead. _rescale_init() below finishes the job using
        # the actual data.
        V0 = torch.randn(num_features, num_basis_vectors, device=device)
        self.V = nn.Parameter(F.normalize(V0, dim=0))
        self.wbar = nn.Parameter(torch.randn(num_basis_vectors, device=device)
                                 / np.sqrt(num_basis_vectors))
        # deltas start near zero: the population direction should explain what it can before
        # individual variation is invoked
        self.delta = nn.Parameter(1e-3 * torch.randn(num_users, num_basis_vectors, device=device))

        self.stopped_at = None      # iteration training actually stopped at
        self.history = []           # (step, train_loss, val_loss or None)

    # ---- core ----

    def user_weights(self):
        """[C, B] effective per-user weight vectors, wbar + delta_u."""
        return self.wbar.unsqueeze(0) + self.delta

    def reward_dirs(self):
        """[F, C] the reward direction each user's model actually implements."""
        return self.V @ self.user_weights().T

    def _nll(self, X_cat, uid):
        W_eff = self.user_weights()                  # [C, B]
        logits = (X_cat * (W_eff[uid] @ self.V.T)).sum(dim=1)   # [N]
        return -F.logsigmoid(logits).mean()

    def _reg(self):
        pop = (self.V @ self.wbar).pow(2).sum()                       # ||V wbar||^2
        ind = (self.delta @ self.V.T).pow(2).sum(dim=1).mean()        # mean_u ||V delta_u||^2
        return self.lam_pop * pop + self.lam_d * ind

    @torch.no_grad()
    def _rescale_init(self, X_cat, uid):
        """Scale V so the initial logits have unit std.

        Removing the /100.0 logit rescaling only works if the initial reward magnitudes are
        sensible for the data at hand: 4096-dim Skywork embedding diffs have norms in the tens,
        so an unscaled init saturates the sigmoid and kills the gradient. This sets the scale
        from the data once, rather than hard-coding a magic constant.
        """
        logits = (X_cat * (self.user_weights()[uid] @ self.V.T)).sum(dim=1)
        s = logits.std().item()
        if s > 1e-8:
            self.V.mul_(1.0 / s)

    def train(self, X, val=None):
        """Fit on X (list of [m_u, F] per-user diff tensors).

        val: optional list in the same user order. When given, early stopping and the reported
        best model are selected on validation NLL -- the regularization strengths must be tuned
        on held-out data, never on test.

        Returns (W_eff [C, B], V [F, B]) so it drops into eval_acc(W, V, feats) unchanged.
        """
        self.to(device)
        X_cat, uid = _pack_users(X)
        X_cat = X_cat.to(device, non_blocking=True).float()
        uid = uid.to(device, non_blocking=True)
        self._rescale_init(X_cat, uid)

        if val is not None:
            Xv_cat, uidv = _pack_users(val)
            Xv_cat = Xv_cat.to(device, non_blocking=True).float()
            uidv = uidv.to(device, non_blocking=True)

        opt = optim.Adam([self.V, self.wbar, self.delta], lr=self.learning_rate)

        best, best_state, since_best = float("inf"), None, 0
        for step in range(self.num_iterations):
            opt.zero_grad()
            loss = self._nll(X_cat, uid) + self._reg()
            loss.backward()
            opt.step()

            if val is not None:
                with torch.no_grad():
                    monitored = self._nll(Xv_cat, uidv).item()
            else:
                monitored = loss.item()
            self.history.append((step, loss.item(), monitored if val is not None else None))

            # plateau detection: stop once the monitored objective stops improving materially
            if monitored < best - self.tol:
                best, since_best = monitored, 0
                if val is not None:
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                since_best += 1
                if since_best >= self.patience:
                    self.stopped_at = step + 1
                    break
        else:
            self.stopped_at = self.num_iterations

        if val is not None and best_state is not None:
            self.load_state_dict(best_state)

        if self.verbose:
            with torch.no_grad():
                vn = torch.linalg.vector_norm(self.V, ord=2, dim=0)
                dn = torch.linalg.vector_norm(self.delta, ord=2, dim=1)
            print(f"[LoReV2] K={self.num_basis_vectors} lam_pop={self.lam_pop} "
                  f"lam_d={self.lam_d} lr={self.learning_rate} "
                  f"stopped at {self.stopped_at}/{self.num_iterations} iters, "
                  f"{'val' if val is not None else 'train'} obj {best:.4f}")
            print(f"[LoReV2] ||V[:,i]||: min {vn.min():.3f} max {vn.max():.3f} | "
                  f"||wbar||: {self.wbar.norm():.3f} | "
                  f"||delta_u||: min {dn.min():.4f} med {dn.median():.4f} max {dn.max():.4f}")

        return self.user_weights().detach(), self.V.detach()


class PersonalizeDelta(nn.Module):
    """Adapt to unseen users: freeze V and wbar, learn only each new user's delta_u.

    This is the point of the wbar/delta split -- a new user starts at the population reward
    function and we only ever learn how they differ from it, which is both the cheap thing to
    estimate from few shots and the thing we actually want to interpret.

    Replaces PersonalizeBatch, which softmaxed the weights onto the simplex.
    """

    def __init__(self, num_users, num_basis_vectors, lam_d=0.01, num_iterations=500,
                 learning_rate=1e-2, patience=50, tol=1e-5):
        super().__init__()
        self.lam_d = lam_d
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.delta = nn.Parameter(1e-3 * torch.randn(num_users, num_basis_vectors, device=device))

    def train(self, X, V, wbar):
        V = V.detach().to(device)
        wbar = wbar.detach().to(device)
        X_cat, uid = _pack_users(X)
        X_cat = X_cat.to(device).float()
        uid = uid.to(device)

        opt = optim.Adam([self.delta], lr=self.learning_rate)
        best, since_best = float("inf"), 0
        for _ in range(self.num_iterations):
            opt.zero_grad()
            W_eff = wbar.unsqueeze(0) + self.delta
            logits = (X_cat * (W_eff[uid] @ V.T)).sum(dim=1)
            nll = -F.logsigmoid(logits).mean()
            ind = (self.delta @ V.T).pow(2).sum(dim=1).mean()
            loss = nll + self.lam_d * ind
            loss.backward()
            opt.step()

            if loss.item() < best - self.tol:
                best, since_best = loss.item(), 0
            else:
                since_best += 1
                if since_best >= self.patience:
                    break

        return (wbar.unsqueeze(0) + self.delta).detach()


def solve_lore_v2(train_features, num_basis_vectors, lam_pop=0.01, lam_d=0.01,
                  num_iterations=2000, learning_rate=1e-2, val_features=None, verbose=True):
    """Convenience wrapper mirroring solve_regularized_simplex, for LoReV2.

    Note there is no V_sft / alpha: the anchor regularizer is gone by design.
    """
    num_features = train_features[0].shape[1]
    m = LoReV2(len(train_features), num_features, num_basis_vectors, lam_pop=lam_pop,
               lam_d=lam_d, num_iterations=num_iterations, learning_rate=learning_rate,
               verbose=verbose)
    W, V = m.train(train_features, val=val_features)
    return W, V.detach(), m


def varimax(L, gamma=1.0, max_iter=200, tol=1e-6):
    """Varimax rotation of a loading matrix L [n, k] -> rotation R [k, k].

    Rotates toward SPARSE loadings: each user should load heavily on few axes rather than
    moderately on all of them. This is the standard fix for rotational indeterminacy in factor
    analysis, and it is what replaces the identifiability that the simplex constraint used to
    provide for free.
    """
    n, k = L.shape
    R = torch.eye(k, dtype=L.dtype)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lam = L @ R
        G = L.T @ (Lam.pow(3) - (gamma / n) * Lam @ torch.diag(torch.diag(Lam.T @ Lam)))
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        R = U @ Vh
        d = float(S.sum())
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return R


def canonical_variation_axes(V, delta, k=None, rotation="varimax"):
    """Turn a fitted (V, delta) into identifiable axes of user variation.

    The V/w factorization is only identified up to V -> VR, w -> R^-1 w, so the raw basis columns
    are not individually meaningful. What IS identified is the set of per-user reward deviations
    {V @ delta_u}. Their SVD gives orthonormal directions ordered by how much of the across-user
    disagreement each explains -- "clear areas of variation to focus on".

    rotation="varimax" additionally fixes the rotation WITHIN near-degenerate eigenvalue blocks.
    That matters more than it sounds: when user groups sit symmetrically (equal-sized groups each
    caring about one axis), the leading eigenvalues come out nearly equal, and the SVD directions
    are then an essentially arbitrary rotation of the true axes. PCA cannot break that tie; a
    sparsity criterion can, because real users load on few axes. Pass rotation=None for the raw
    principal axes.

    Returns (axes [F, k], singular_values [k], explained_fraction [k]).
    """
    Vc, dc = V.detach().cpu(), delta.detach().cpu()
    D = dc @ Vc.T                                        # [C, F] user reward deviations
    D = D - D.mean(dim=0, keepdim=True)                  # centre: wbar already carries the mean
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    k = k or S.shape[0]
    axes, S_k = Vh[:k].T, S[:k]
    var = S.pow(2)
    expl = (var / var.sum().clamp_min(1e-12))[:k]

    if rotation == "varimax" and k > 1:
        loadings = D @ axes                              # [C, k] user coordinates
        R = varimax(loadings)
        axes = axes @ R
        # re-order by variance explained in the rotated basis (rotation redistributes it)
        rl = loadings @ R
        order = torch.argsort(rl.pow(2).sum(0), descending=True)
        axes, S_k, expl = axes[:, order], S_k[order], expl[order]

    return axes, S_k, expl