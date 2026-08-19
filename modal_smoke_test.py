"""
Cheap smoke test for modal_verify_fix.py's volume persistence -- run BEFORE the 2-hour job.

It reproduces the exact prepare -> embed -> copy-to-volume -> commit path from verify(), but
embeds only EMBED_LIMIT=8 examples (so ~1 min of GPU, not 2 hours). Then, in a SEPARATE container,
it reloads the volume and confirms the .pkl are actually there and loadable -- which is precisely
the cross-container persistence that broke before (embeddings written to ephemeral disk, volume
commit saved nothing).

Pass criteria: phase 2 finds both .pkl on the volume and prints their embedding shapes.
If this passes, the real `modal run modal_verify_fix.py` will persist correctly.

Run:  modal run modal_smoke_test.py
"""
import os
import subprocess

import modal

app = modal.App("lore-prism-smoke-test")
volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .pip_install("accelerate>=0.26.0", "ipython")
    .add_local_dir(".", remote_path="/workspace",
                   ignore=["**/*.pkl", "data/prism/*.pkl", ".git", "**/__pycache__"])
)

# Same location verify() uses, so this validates the real path. Use a *_smoke suffix dir so the
# 8-example files never get mistaken for the full embeddings.
VOL_PKL = "/vol/prism_embeddings_smoke"
PKLS = ["train_embeddings.pkl", "test_embeddings.pkl"]


@app.function(image=image, gpu="A10G", timeout=1800,
              volumes={"/vol": volume}, secrets=[modal.Secret.from_name("huggingface")])
def smoke_generate():
    """Phase 1: prepare + embed 8 examples, then persist to the volume EXACTLY as verify() does."""
    import shutil
    os.chdir("/workspace")
    os.makedirs(VOL_PKL, exist_ok=True)
    os.makedirs("data/prism", exist_ok=True)
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")

    print("[1] prepare.py", flush=True)
    subprocess.run(["python", "PRISM/prepare.py"], check=True)

    print("[2] generate-prism-embeddings.py with EMBED_LIMIT=8", flush=True)
    env = {**os.environ, "EMBED_LIMIT": "8"}
    subprocess.run(["python", "PRISM/generate-prism-embeddings.py"], check=True, env=env)

    print("[3] copy to volume + commit", flush=True)
    sizes = {}
    for p in PKLS:
        src = f"data/prism/{p}"
        if not os.path.exists(src):
            raise FileNotFoundError(f"generate did not write {src}")
        shutil.copy(src, f"{VOL_PKL}/{p}")
        sizes[p] = os.path.getsize(f"{VOL_PKL}/{p}")
    volume.commit()
    print(f"  committed {sizes} to {VOL_PKL}", flush=True)
    return sizes


@app.function(image=image, timeout=300, volumes={"/vol": volume})  # CPU-only, fresh container
def smoke_check():
    """Phase 2 (separate container): can we actually read the committed .pkl back?"""
    import torch
    volume.reload()
    results = {}
    for p in PKLS:
        path = f"{VOL_PKL}/{p}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"PERSISTENCE FAILED: {path} not on the volume from a fresh container")
        data = torch.load(path, weights_only=False)
        emb = data[0]["extra_info"]["chosen_conv_embedding"]
        results[p] = (len(data), tuple(torch.as_tensor(emb).shape))
    return results


@app.local_entrypoint()
def main():
    print("Phase 1: prepare + embed 8 examples + persist to volume...")
    sizes = smoke_generate.remote()
    print(f"Phase 1 done, sizes on volume: {sizes}\n")

    print("Phase 2 (fresh container): reload volume and read the .pkl back...")
    results = smoke_check.remote()
    print("\n=== SMOKE TEST RESULT ===")
    for p, (n, shape) in results.items():
        print(f"  {p}: {n} entries, embedding shape {shape}")
    print("\nPASS: embeddings persist across containers. The full modal_verify_fix.py run will save "
          "correctly. Safe to run the 2-hour job.")
