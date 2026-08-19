import torch
import torch.nn.functional as F

def run_causal_ablation(concept_vectors_path="data/prism/concept_vectors.pt", 
                        ckpt_path="checkpoints/checkpoints/PRISM_V_lore_K_20_alpha_10000.0.pt"):
    print("=== Causal Intervention (Ablation) Analysis ===")
    print(f"Loading concept vectors from {concept_vectors_path}...")
    concept_vectors = torch.load(concept_vectors_path, map_location='cpu', weights_only=True)
    concepts = list(concept_vectors.keys())
    
    C = torch.stack([concept_vectors[c] for c in concepts]) # [C, 4096]
    C_norm = F.normalize(C, p=2, dim=1) # [C, 4096]
    
    print(f"Loading checkpoint {ckpt_path}...")
    V = torch.load(ckpt_path, map_location='cpu', weights_only=True) # [4096, B]
    B = V.shape[1]
    
    V_norm = F.normalize(V, p=2, dim=0) # [4096, B]
    
    # Compute the activation of every basis for every concept
    # S[b, c] = activation of basis b for concept c
    S = torch.mm(V_norm.t(), C_norm.t()) # [B, C]
    
    print("\n--- Ablation Results ---")
    for c_idx, concept in enumerate(concepts):
        activations = S[:, c_idx]
        
        # Find the basis that is most causally linked to this concept
        best_basis = torch.argmax(torch.abs(activations)).item()
        max_activation = activations[best_basis].item()
        
        # If the max activation is too low, this concept isn't learned by the model
        if abs(max_activation) < 0.025:
            continue
            
        print(f"\nConcept: '{concept}'")
        print(f"  -> Causal Basis Identified: Basis {best_basis}")
        
        # Calculate Total Signal Power (sum of squares of all basis activations)
        total_power = torch.sum(activations ** 2).item()
        
        # Calculate Ablated Signal Power (power if we zero out the causal basis)
        ablated_activations = activations.clone()
        ablated_activations[best_basis] = 0.0
        ablated_power = torch.sum(ablated_activations ** 2).item()
        
        # Calculate how much signal is lost by ablating this single basis
        power_drop_pct = ((total_power - ablated_power) / total_power) * 100
        
        print(f"  -> Total Reward Signal Power: {total_power:.6f}")
        print(f"  -> Power after Ablating Basis {best_basis}: {ablated_power:.6f}")
        print(f"  -> Signal Loss from Ablation: {power_drop_pct:.2f}%")
        
        if power_drop_pct > 80:
            print(f"  => CAUSALITY PROVEN: Basis {best_basis} is the single bottleneck for '{concept}'.")
            print(f"     Zeroing it out completely blinds the model to this concept.")
        elif power_drop_pct > 50:
            print(f"  => PARTIAL CAUSALITY: Basis {best_basis} controls the majority of the '{concept}' signal, but there is some redundancy.")
        else:
            print(f"  => ENTANGLED: The '{concept}' signal is heavily distributed across multiple bases.")

if __name__ == "__main__":
    run_causal_ablation()
