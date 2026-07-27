"""
Probe what train_basis.py substitutes as its regularizer anchor (V_sft).

train_basis.py (lines 65-82) does:

    from transformers import AutoModel
    rm = AutoModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", ...)
    last_linear_layer = None
    for name, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module          # keeps the LAST Linear it sees
    V_final = last_linear_layer.weight[:, 0]...  # column 0 of that layer = V_sft

This script just prints the characteristics so you can judge for yourself what
that anchor is. No conclusions are drawn.

By default it builds only the model ARCHITECTURE on the meta device (zero memory,
no weights, no GPU) — enough to see WHICH layer the loop lands on and its shape.
Pass --real to actually load the weights (needs the GPU box / lots of RAM) and
print the exact 4096-vector that gets substituted, so you can inspect the values.

Run:  uv run python PRISM/check_anchor.py
      uv run python PRISM/check_anchor.py --real
"""
import argparse
import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

MODEL_NAME = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def empty_weights_ctx():
    try:
        from accelerate import init_empty_weights
        return init_empty_weights()
    except ImportError:
        return torch.device("meta")


def all_linears(model):
    return [(n, tuple(m.weight.shape)) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)]


def probe_meta():
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    print(f"config.architectures : {cfg.architectures}")
    print(f"config.num_labels    : {getattr(cfg, 'num_labels', None)}\n")

    with empty_weights_ctx():
        rm = AutoModel.from_config(cfg)
    lin = all_linears(rm)
    print(f"AutoModel  -> class {type(rm).__name__}")
    print(f"AutoModel  -> total nn.Linear modules: {len(lin)}")
    print("AutoModel  -> last 3 Linear layers in named_modules() order:")
    for n, s in lin[-3:]:
        print(f"                {n}   {s}")
    print(f"AutoModel  -> THE one the loop keeps (last): {lin[-1][0]}   {lin[-1][1]}")
    print(f"AutoModel  -> any module name containing 'score': "
          f"{[n for n, _ in rm.named_modules() if 'score' in n]}\n")

    with empty_weights_ctx():
        rm2 = AutoModelForSequenceClassification.from_config(cfg)
    print(f"AutoModelForSequenceClassification -> class {type(rm2).__name__}")
    print("AutoModelForSequenceClassification -> Linear layers named 'score':")
    for n, s in all_linears(rm2):
        if "score" in n:
            print(f"                {n}   {s}")


def probe_real():
    print("loading real weights (this is the heavy path)...\n")
    rm = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                   attn_implementation="eager", num_labels=1)
    last = None
    for name, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last = (name, module)
    name, module = last
    vsft = module.weight[:, 0].float()   # exactly what train_basis.py:82 substitutes
    print(f"substituted layer : {name}   weight shape {tuple(module.weight.shape)}")
    print(f"V_sft = weight[:, 0] : shape {tuple(vsft.shape)}")
    print(f"  norm      : {vsft.norm().item():.4f}")
    print(f"  mean/std  : {vsft.mean().item():.4e} / {vsft.std().item():.4e}")
    print(f"  first 8   : {vsft[:8].tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="load real weights and print the actual substituted vector")
    args = ap.parse_args()
    print(f"model: {MODEL_NAME}\n")
    probe_real() if args.real else probe_meta()


if __name__ == "__main__":
    main()
