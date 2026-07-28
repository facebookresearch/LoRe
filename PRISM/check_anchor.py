"""
Does exactly what train_basis.py does to build V_sft, then prints:
  - the FULL layer that got grabbed (and its name)
  - the ONE column that gets sliced out and glued in as V_sft
  - the REAL reward head, for contrast
The point: V_sft has the right SIZE (4096 numbers) but comes from the wrong layer.
"""
import torch
from transformers import AutoModel, AutoModelForSequenceClassification

model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, num_labels=1)

# same loop as train_basis.py: keep the LAST nn.Linear
last_linear_layer = None
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module
        last_name = name

# same line as train_basis.py:82 — slices ONE column out of the grabbed layer
V_sft = last_linear_layer.weight[:, 0].to(torch.float32).reshape(-1, 1)

print("\n--- what train_basis.py grabs (all values read live from the model) ---")
print("layer grabbed        :", last_name)
print("full grabbed layer   :", tuple(last_linear_layer.weight.shape))
print("V_sft = column 0     :", tuple(V_sft.shape))

# the correct loader keeps the real reward head (this block is NOT in train_basis.py;
# it runs after V_sft is already built and does not touch it)
rm2 = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, num_labels=1)
score = rm2.score
print("\n--- the reward head the code should have used ---")
print("real head 'score'    :", tuple(score.weight.shape))
print(f"\nnote: V_sft came from '{last_name}'.")
