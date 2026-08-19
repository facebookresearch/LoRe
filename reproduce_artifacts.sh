#!/usr/bin/env bash
# Regenerate the three artifacts that are gitignored / not committed, starting
# from the PRISM embeddings you already have. Run from the repository root on a
# GPU box.
#
#   PRISM/basis_matrices.pt          (LoRe basis; key PART2_K10_seed42)
#   data/prism/concept_vectors.pt    (11-concept library, via Skywork)
#   sae/checkpoints/d3/model.pt      (the D3 TopK SAE)
#
# Prereq you must place yourself (the slow GPU embedding step, already done):
#   PRISM/data/prism/train_embeddings.pkl
#   PRISM/data/prism/test_embeddings.pkl
#
# Usage:
#   bash reproduce_artifacts.sh            # everything
#   bash reproduce_artifacts.sh basis      # just PRISM/basis_matrices.pt
#   bash reproduce_artifacts.sh concepts   # just data/prism/concept_vectors.pt
#   bash reproduce_artifacts.sh sae        # just the SAE checkpoint (needs basis)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"

TRAIN_EMB="PRISM/data/prism/train_embeddings.pkl"
TEST_EMB="PRISM/data/prism/test_embeddings.pkl"
BASIS_OUT="PRISM/basis_matrices.pt"
CONCEPTS_OUT="data/prism/concept_vectors.pt"
SAE_CKPT="sae/checkpoints/d3/model.pt"

log() { printf '\n=== %s ===\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Some generator scripts live on the prism-dataset-analysis branch. Fetch them
# into the working tree so this runs on a fresh clone of any branch.
# ---------------------------------------------------------------------------
resolve_ref() {
  for ref in prism-dataset-analysis origin/prism-dataset-analysis; do
    if git rev-parse --verify --quiet "$ref" >/dev/null; then echo "$ref"; return; fi
  done
  die "cannot find the prism-dataset-analysis branch (needed for helper scripts). Run: git fetch origin"
}

fetch_from_branch() {
  # fetch_from_branch <ref> <path> [<path> ...]  -- only if not already present
  local ref="$1"; shift
  for path in "$@"; do
    if [[ ! -f "$path" ]]; then
      mkdir -p "$(dirname "$path")"
      git show "$ref:$path" > "$path" || die "could not fetch $path from $ref"
      echo "fetched $path from $ref"
    fi
  done
}

ensure_helpers() {
  local ref; ref="$(resolve_ref)"
  # basis builder + its imports
  fetch_from_branch "$ref" \
    utils.py \
    PRISM/rm_head_utils.py \
    PRISM/basis_reproducibility_check.py
  # concept builder + its inputs
  fetch_from_branch "$ref" \
    PRISM/compute_concept_vectors.py \
    PRISM/concept_library.py \
    data/prism/contrastive_pairs.json
}

ensure_embeddings() {
  [[ -f "$TRAIN_EMB" ]] || die "missing $TRAIN_EMB -- copy your saved embeddings here first"
  [[ -f "$TEST_EMB" ]]  || die "missing $TEST_EMB -- copy your saved embeddings here first"
  # train_basis / basis builder read the root-relative data/prism/ path; mirror via symlink.
  mkdir -p data/prism
  ln -sf "../../$TRAIN_EMB" data/prism/train_embeddings.pkl
  ln -sf "../../$TEST_EMB"  data/prism/test_embeddings.pkl
}

# ---------------------------------------------------------------------------
# 1. LoRe basis matrix  ->  PRISM/basis_matrices.pt  (key PART2_K10_seed42)
# ---------------------------------------------------------------------------
build_basis() {
  log "1/3  LoRe basis -> $BASIS_OUT"
  ensure_embeddings
  cd PRISM
  $PY basis_reproducibility_check.py \
    --rank-values 10 \
    --rank-seed 42 \
    --seed-values 42 \
    --fixed-K 10 \
    --matrices-out "../$BASIS_OUT"
  cd "$REPO_ROOT"
  [[ -f "$BASIS_OUT" ]] || die "basis build did not produce $BASIS_OUT"
  $PY - "$BASIS_OUT" <<'PYEOF'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu")
assert "PART2_K10_seed42" in m, f"missing run key; have {list(m)[:5]}"
print("basis ok: keys =", list(m.keys()))
PYEOF
}

# ---------------------------------------------------------------------------
# 2. Concept vectors  ->  data/prism/concept_vectors.pt  (needs Skywork model)
# ---------------------------------------------------------------------------
build_concepts() {
  log "2/3  concept vectors -> $CONCEPTS_OUT"
  [[ -f data/prism/contrastive_pairs.json ]] || die "missing data/prism/contrastive_pairs.json"
  # compute_concept_vectors.py uses repo-root-relative paths; run from the root.
  $PY PRISM/compute_concept_vectors.py
  [[ -f "$CONCEPTS_OUT" ]] || die "concept build did not produce $CONCEPTS_OUT"
  echo "concepts ok: $CONCEPTS_OUT"
}

# ---------------------------------------------------------------------------
# 3. D3 SAE checkpoint  ->  sae/checkpoints/d3/model.pt
#    (build tensors -> train; needs the basis for the preservation loss)
# ---------------------------------------------------------------------------
build_sae() {
  log "3/3  D3 SAE -> $SAE_CKPT"
  [[ -f "$BASIS_OUT" ]] || die "missing $BASIS_OUT -- run the basis step first"
  bash sae/run_d3.sh build
  bash sae/run_d3.sh train
  [[ -f "$SAE_CKPT" ]] || die "SAE train did not produce $SAE_CKPT"
  echo "sae ok: $SAE_CKPT"
}

TARGET="${1:-all}"
ensure_helpers
case "$TARGET" in
  basis)    build_basis ;;
  concepts) build_concepts ;;
  sae)      build_sae ;;
  all)      build_basis; build_concepts; build_sae ;;
  *) die "unknown target: $TARGET (use: basis | concepts | sae | all)" ;;
esac

log "done"
echo "basis    : $BASIS_OUT"
echo "concepts : $CONCEPTS_OUT"
echo "sae ckpt : $SAE_CKPT"
