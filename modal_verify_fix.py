"""
Modal runner: verify that fixing the string-vs-list formatting bug removes basis collapse.

What this checks: "Verify that fixing this bug on the shared code also fixes the basis collapse...
Run a few seeds and check whether LoRe's accuracy is on average better than that of the base RM."

The bug (confirmed on our code):
  * PRISM/prepare.py stored chosen_utterance as a string but rejected_utterance as a LIST.
  * PRISM/generate-prism-embeddings.py dropped that list straight into the chat template, so
    rejected rendered as its Python repr -- ['answer one', 'answer two']<|eot_id|> -- while chosen
    rendered as clean prose. The reward model separates chosen/rejected on that formatting
    fingerprint, and every LoRe basis collapses to that one direction.
Both files are now fixed to embed a single rejected string, formatted identically to chosen.

This runner regenerates the embeddings WITH the fix, then measures, across seeds:
  1. basis collapse  (multiseed_collapse.py: min|cos| between columns, s2/s1, bases kept, test acc)
  2. base-RM accuracy on the FIXED embeddings (test_base_rm_accuracy.py)
So we can answer both questions: did collapse go away, and does LoRe beat the base RM.

The fixed embeddings are committed to the "lore-prism-data" volume at data/prism/*.pkl. To pull
them locally for further CPU analysis:  modal volume get lore-prism-data data/prism/train_embeddings.pkl

Run:  modal run modal_verify_fix.py
"""
import os
import subprocess

import modal

app = modal.App("lore-prism-verify-fix")
volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .pip_install("accelerate>=0.26.0", "ipython")  # prepare.py imports IPython for its excepthook
    .add_local_dir(".", remote_path="/workspace",
                   ignore=["**/*.pkl", "data/prism/*.pkl", ".git", "**/__pycache__"])
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def verify(regen: bool = True):
    import shutil
    os.chdir("/workspace")
    # NOTE: the .add_local_dir(".") mount already creates /workspace/data (it ships the small
    # data/prism JSON/PT files; only *.pkl are ignored). So we CANNOT symlink data -> /vol/data
    # (the dir exists) and must NOT write embeddings into /workspace/data -- that is the container's
    # ephemeral disk and is lost when the container exits. Instead persist the .pkl explicitly on
    # the volume at /vol/prism_embeddings and copy them into place around the analysis steps.
    VOL_PKL = "/vol/prism_embeddings"
    os.makedirs(VOL_PKL, exist_ok=True)
    os.makedirs("data/prism", exist_ok=True)
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")
    pkls = ["train_embeddings.pkl", "test_embeddings.pkl"]

    if regen:
        print("\n[1/4] prepare.py  (regenerate FIXED parquet: rejected as a single string)", flush=True)
        subprocess.run(["python", "PRISM/prepare.py"], check=True)

        print("\n[2/4] generate-prism-embeddings.py  (regenerate FIXED embeddings)", flush=True)
        subprocess.run(["python", "PRISM/generate-prism-embeddings.py"], check=True)
        for p in pkls:  # persist to the volume so --no-regen can reuse them
            shutil.copy(f"data/prism/{p}", f"{VOL_PKL}/{p}")
        volume.commit()
        print(f"  persisted embeddings to {VOL_PKL} on the volume", flush=True)
    else:
        print("\n[skip 1-2] reusing the FIXED embeddings already on the volume", flush=True)
        volume.reload()  # pick up the committed fixed .pkl from a previous run
        for p in pkls:
            if not os.path.exists(f"{VOL_PKL}/{p}"):
                raise FileNotFoundError(
                    f"{VOL_PKL}/{p} not on the volume -- run once WITHOUT --no-regen first.")
            shutil.copy(f"{VOL_PKL}/{p}", f"data/prism/{p}")

    print("\n[3/5] multiseed_collapse.py  (collapse metrics + test acc, several seeds)", flush=True)
    subprocess.run(["python", "PRISM/multiseed_collapse.py"], check=True)

    print("\n[4/5] test_base_rm_accuracy.py  (base-RM accuracy on the FIXED embeddings)", flush=True)
    subprocess.run(["python", "PRISM/test_base_rm_accuracy.py"], check=True)

    print("\n[5/5] per_user_signal.py  (is personalization learnable above the global axis?)", flush=True)
    pu = subprocess.run(["python", "PRISM/per_user_signal.py"], capture_output=True, text=True)
    print(pu.stdout, flush=True)
    if pu.returncode != 0:
        print("per_user_signal stderr:\n" + pu.stderr, flush=True)

    # Persist all results on the volume so they survive a detached run / disconnected client.
    os.makedirs(f"{VOL_PKL}/results", exist_ok=True)
    with open(f"{VOL_PKL}/results/per_user_signal.txt", "w") as fh:
        fh.write(pu.stdout + "\n" + (pu.stderr or ""))
    shutil.copy("results/correct_anchor/multiseed_collapse.csv",
                f"{VOL_PKL}/results/multiseed_collapse.csv")
    volume.commit()

    with open("results/correct_anchor/multiseed_collapse.csv", "rb") as fh:
        return fh.read(), pu.stdout


@app.local_entrypoint()
def main(regen: bool = True):
    # --regen / --no-regen : with --no-regen, reuse the fixed embeddings already committed to the
    # volume and only re-run the (fast, CPU) collapse + base-RM analysis.
    if regen:
        print("Regenerating FIXED PRISM embeddings on Modal and re-checking basis collapse...")
    else:
        print("Reusing committed FIXED embeddings; re-running collapse + base-RM analysis only...")
    csv_bytes, per_user_txt = verify.remote(regen)
    os.makedirs("results/correct_anchor", exist_ok=True)
    with open("results/correct_anchor/multiseed_collapse_fixed.csv", "wb") as fh:
        fh.write(csv_bytes)
    print("\nSynced results/correct_anchor/multiseed_collapse_fixed.csv")
    print(csv_bytes.decode(errors="replace"))
    print("\n=== per-user signal (corrected embeddings) ===")
    print(per_user_txt)
    print("\nFixed embeddings + results are on the volume. If the client disconnected (detached run),")
    print("pull them once the run finishes with:")
    print("  modal volume get lore-prism-data prism_embeddings/train_embeddings.pkl")
    print("  modal volume get lore-prism-data prism_embeddings/test_embeddings.pkl")
    print("  modal volume get lore-prism-data prism_embeddings/results/per_user_signal.txt")
