import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

def compute_null_threshold(concept_vector, num_samples=1000, percentile=95):
    dim = concept_vector.shape[0]
    random_vectors = torch.randn(num_samples, dim, device=concept_vector.device)
    random_vectors = F.normalize(random_vectors, p=2, dim=1)
    concept_normalized = F.normalize(concept_vector.unsqueeze(0), p=2, dim=1)
    similarities = torch.mm(random_vectors, concept_normalized.t()).squeeze()
    sorted_sims, _ = torch.sort(similarities)
    idx = int((percentile / 100.0) * num_samples)
    return sorted_sims[idx].item()

def run_rank_comparison(concept_vectors_path="data/prism/concept_vectors.pt", checkpoints_dir="checkpoints/checkpoints", output_dir="results/rank_comparison"):
    print(f"Loading concept vectors from {concept_vectors_path}...")
    concept_vectors = torch.load(concept_vectors_path, map_location='cpu', weights_only=True)
    concepts = list(concept_vectors.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Computing null distributions...")
    tau_c_dict = {}
    for c in concepts:
        tau_c_dict[c] = compute_null_threshold(concept_vectors[c])
        
    C_mat = torch.stack([concept_vectors[c] for c in concepts], dim=0)
    C_norm = F.normalize(C_mat, p=2, dim=1)
    
    results = []
    
    # Find all trained basis matrices
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "PRISM_V_lore_K_*_alpha_*.pt"))
    # We only care about alpha=10000.0 for comparison (the default regularized version)
    checkpoint_files = [f for f in checkpoint_files if "alpha_10000.0" in f]
    
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate.")
    
    for ckpt in checkpoint_files:
        match = re.search(r"K_(\d+)_", ckpt)
        if not match:
            continue
        K = int(match.group(1))
        
        print(f"\nEvaluating K={K} ({ckpt})...")
        V = torch.load(ckpt, map_location='cpu', weights_only=True)
        B = V.shape[1]
        
        V_norm = F.normalize(V, p=2, dim=0)
        S = torch.mm(V_norm.t(), C_norm.t()) # [B, C]
        
        significant_bases = 0
        covered_concepts = set()
        
        for b in range(B):
            basis_sims = S[b, :]
            has_sig_concept = False
            for c_idx, c in enumerate(concepts):
                sim = basis_sims[c_idx].item()
                if sim >= tau_c_dict[c]:
                    has_sig_concept = True
                    covered_concepts.add(c)
            if has_sig_concept:
                significant_bases += 1
                
        interpretability_score = significant_bases / B if B > 0 else 0
        
        results.append({
            "Rank (K)": K,
            "Significant Bases": significant_bases,
            "Interpretability Score (%)": interpretability_score * 100,
            "Concepts Covered": len(covered_concepts),
            "Total Concepts": len(concepts)
        })
        
    df = pd.DataFrame(results).sort_values(by="Rank (K)")
    df.to_csv(os.path.join(output_dir, "rank_comparison.csv"), index=False)
    
    print("\n=== Rank Comparison Report ===")
    print(df.to_string(index=False))
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Line 1: Interpretability Score
    color = 'tab:blue'
    ax1.set_xlabel('Rank (K)')
    ax1.set_ylabel('Interpretability Score (%)', color=color)
    ax1.plot(df['Rank (K)'], df['Interpretability Score (%)'], marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 105)
    
    ax2 = ax1.twinx()  
    # Line 2: Concepts Covered
    color = 'tab:orange'
    ax2.set_ylabel('Concepts Covered', color=color)  
    ax2.plot(df['Rank (K)'], df['Concepts Covered'], marker='s', color=color, linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, len(concepts) + 1)
    
    plt.title("Basis Interpretability vs. Rank (K)")
    fig.tight_layout()  
    plt.savefig(os.path.join(output_dir, "rank_comparison_plot.png"), dpi=300)
    print(f"\nPlot saved to {output_dir}/rank_comparison_plot.png")

if __name__ == "__main__":
    run_rank_comparison()
