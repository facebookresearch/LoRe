"""
Does exactly what train_basis.py does to build V_sft, then prints the shape of
the vector that gets glued in as V_sft. Nothing else.
"""
import torch
from transformers import AutoModel

model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, num_labels=1)

# same loop as train_basis.py: keep the LAST nn.Linear
last_linear_layer = None
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module
        last_name = name

# same line as train_basis.py:82
V_sft = last_linear_layer.weight[:, 0].to(torch.float32).reshape(-1, 1)

print("layer grabbed :", last_name)
print("V_sft shape   :", tuple(V_sft.shape))
