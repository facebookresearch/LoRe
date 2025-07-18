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