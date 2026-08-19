import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_null_threshold(concept_vector, num_samples=1000, percentile=95):
    """
    Computes the null distribution of cosine similarities by generating random vectors in the same space.
    Returns the given percentile as the significance threshold (tau_c).
    """
    dim = concept_vector.shape[0]
    # Generate random unit vectors
    random_vectors = torch.randn(num_samples, dim, device=concept_vector.device)
    random_vectors = F.normalize(random_vectors, p=2, dim=1)
    
    # Normalize concept vector
    concept_normalized = F.normalize(concept_vector.unsqueeze(0), p=2, dim=1)
    
    # Compute cosine similarities
    similarities = torch.mm(random_vectors, concept_normalized.t()).squeeze()
    
    # Sort and find percentile
    sorted_sims, _ = torch.sort(similarities)
    idx = int((percentile / 100.0) * num_samples)
    
    return sorted_sims[idx].item()

def get_confidence_tier(similarity, tau_c):
    if similarity < tau_c:
        return "Reject"
    elif similarity < 0.15:
        return "Tentative"
    elif similarity < 0.30:
        return "Good"
    else:
        return "Excellent"

def run_alignment_analysis(concept_vectors_path, basis_matrix_path, output_dir="results"):
    print(f"Loading concept vectors from {concept_vectors_path}...")
    concept_vectors = torch.load(concept_vectors_path, map_location='cpu', weights_only=True)
    
    print(f"Loading basis matrix V from {basis_matrix_path}...")
    V = torch.load(basis_matrix_path, map_location='cpu', weights_only=True)
    # V is [4096, B]. We want the rows a_b, which means transposing V to [B, 4096]
    # Wait, in train_basis.py, V is [num_features, num_basis_vectors].
    # So V[:, b] is the b-th basis direction. 
    B = V.shape[1]
    
    os.makedirs(output_dir, exist_ok=True)
    
    concepts = list(concept_vectors.keys())
    C = len(concepts)
    
    # 1. Compute per-concept thresholds
    print("Computing null distributions...")
    tau_c_dict = {}
    for c in concepts:
        tau_c_dict[c] = compute_null_threshold(concept_vectors[c])
        print(f"  {c}: τ_95 = {tau_c_dict[c]:.4f}")
        
    # 2. Compute similarity matrix [B, C]
    # We normalize both to compute cosine similarity
    V_norm = F.normalize(V, p=2, dim=0) # [4096, B]
    
    # Stack concept vectors
    C_mat = torch.stack([concept_vectors[c] for c in concepts], dim=0) # [C, 4096]
    C_norm = F.normalize(C_mat, p=2, dim=1) # [C, 4096]
    
    # Cosine similarity matrix S: [B, C]
    # V_norm.t() is [B, 4096], C_norm.t() is [4096, C]
    # S = V_norm.t() @ C_norm.t() -> shape [B, C]
    S = torch.mm(V_norm.t(), C_norm.t())
    
    # 3. Apply tiered rubric and create report
    report_rows = []
    
    for b in range(B):
        basis_sims = S[b, :]
        for c_idx, c in enumerate(concepts):
            sim = basis_sims[c_idx].item()
            tau_c = tau_c_dict[c]
            tier = get_confidence_tier(sim, tau_c)
            
            report_rows.append({
                "Basis": f"Basis_{b}",
                "Concept": c,
                "Cosine_Sim": sim,
                "Threshold": tau_c,
                "Tier": tier
            })
            
    df = pd.DataFrame(report_rows)
    df.to_csv(os.path.join(output_dir, "alignment_report.csv"), index=False)
    
    # Print summary
    print("\n=== Significant Alignments ===")
    sig_df = df[df["Tier"] != "Reject"].sort_values(by=["Basis", "Cosine_Sim"], ascending=[True, False])
    for b in range(B):
        b_df = sig_df[sig_df["Basis"] == f"Basis_{b}"]
        if len(b_df) > 0:
            print(f"\nBasis {b}:")
            for _, row in b_df.iterrows():
                print(f"  - {row['Concept']}: {row['Cosine_Sim']:>6.3f} ({row['Tier']})")
        else:
            print(f"\nBasis {b}: No significant alignments.")
            
    # 4. Heatmap visualization
    plt.figure(figsize=(12, 8))
    # reshape S to numpy for seaborn
    S_np = S.cpu().numpy()
    
    ax = sns.heatmap(S_np, xticklabels=concepts, yticklabels=[f"Basis {b}" for b in range(B)], 
                     cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Cosine Similarity: Concept Vectors vs LoRe Bases")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_heatmap.png"), dpi=300)
    print(f"\nAnalysis complete! Check {output_dir}/alignment_report.csv and similarity_heatmap.png")

if __name__ == "__main__":
    # Point this to a specific checkpoint you want to evaluate
    run_alignment_analysis(
        concept_vectors_path="data/prism/concept_vectors.pt",
        basis_matrix_path="checkpoints/checkpoints/PRISM_V_lore_K_50_alpha_10000.0.pt", # Defaulting to K=50
        output_dir="results/K_50"
    )
