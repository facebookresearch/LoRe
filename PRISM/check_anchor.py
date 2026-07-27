"""
Verify the V_sft anchor bug WITHOUT loading the 8B model.

train_basis.py (lines 65-82) builds its regularizer anchor like this:

    from transformers import AutoModel
    rm = AutoModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", ...)
    last_linear_layer = None
    for name, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module          # <-- keeps the LAST Linear it sees
    V_final = last_linear_layer.weight[:, 0]...  # <-- column 0 of that layer = V_sft

The claim (teammate's bug): AutoModel loads the BASE model and DROPS the reward
head, so "the last Linear" is an internal MLP layer (down_proj), not the reward
head. The anchor everything is regularized toward is therefore a meaningless
internal direction.

This script proves that by building the model's ARCHITECTURE on the meta device
(a zero-memory skeleton — no weights downloaded, no GPU, no 16-32GB of RAM) and
running the EXACT same "grab the last Linear" loop. It also builds the CORRECT
loader (AutoModelForSequenceClassification) to show the real reward head exists.

Run:  uv run python PRISM/check_anchor.py
(Needs network to fetch the ~30KB config.json. No model weights are downloaded.)
"""
import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

MODEL_NAME = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def empty_weights_ctx():
    """Zero-memory model construction. Prefer accelerate; fall back to torch meta."""
    try:
        from accelerate import init_empty_weights
        return init_empty_weights()
    except ImportError:
        return torch.device("meta")


def last_linear(model):
    """Replicates train_basis.py:77-81 exactly — the LAST nn.Linear in iteration order."""
    name_mod = (None, None)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            name_mod = (name, module)
    return name_mod


def main():
    print(f"model: {MODEL_NAME}\n")

    cfg = AutoConfig.from_pretrained(MODEL_NAME)   # tiny: just config.json
    print(f"[0] designed architecture (what the model REALLY is): {cfg.architectures}")
    print("    -> a *ForSequenceClassification* model: its head is a Linear named 'score'.\n")

    # ---- what train_basis.py actually grabs: AutoModel + last Linear ----
    with empty_weights_ctx():
        rm = AutoModel.from_config(cfg)            # BASE model, head dropped, zero memory
    a_name, a_mod = last_linear(rm)
    has_score = any("score" in n for n, _ in rm.named_modules())
    print("[1] what train_basis.py's loop actually grabs (the anchor V_sft):")
    print(f"    last Linear in AutoModel = '{a_name}'  shape {tuple(a_mod.weight.shape)}")
    print(f"    does AutoModel even contain a 'score' reward head?  {has_score}")
    print("    -> if this is an mlp.down_proj and score=False, the anchor is an")
    print("       INTERNAL layer, and V_sft = column 0 of it (train_basis.py:82).\n")

    # ---- the CORRECT loader: the real reward head ----
    with empty_weights_ctx():
        rm2 = AutoModelForSequenceClassification.from_config(cfg)
    heads = [(n, tuple(m.weight.shape)) for n, m in rm2.named_modules()
             if isinstance(m, torch.nn.Linear) and "score" in n]
    print("[2] the reward head the code SHOULD have used:")
    print(f"    AutoModelForSequenceClassification head(s): {heads}")
    print("    -> a (1, 4096) Linear: the actual one-dimensional good/bad ruler.\n")

    # ---- verdict ----
    is_bug = (not has_score) and ("score" not in (a_name or ""))
    print("=" * 60)
    print("VERDICT:", "ANCHOR IS BUGGED (internal layer, not the reward head)"
          if is_bug else "anchor looks like a real head — re-check assumptions")
    print("=" * 60)


if __name__ == "__main__":
    main()
