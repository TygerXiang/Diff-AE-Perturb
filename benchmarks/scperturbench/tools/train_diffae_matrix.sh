#!/usr/bin/env bash
# Train the Diff-AE-Perturb conditioning fix: {x0, eps} x {per_cell, batch_mean}.
#
#   EPOCHS=2   bash train_diffae_matrix.sh            # smoke
#   EPOCHS=200 bash train_diffae_matrix.sh            # full matrix
#   ONLY=fix_x0_percell EPOCHS=200 bash ...           # one variant
set -u

EPOCHS="${EPOCHS:-200}"
ONLY="${ONLY:-}"
BASE=/root/diff-project/reproducibility/prepared/splits_unseen_guide_seed7
PREP=/root/diff-project/reproducibility/prepared/gse132080_preprocessed.h5ad
TRAIN_ROOT=/root/perturbench/.scperturbench/trained

train_one () {
  local name="$1" pred="$2" ref="$3"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  local out="$TRAIN_ROOT/$name"
  mkdir -p "$out"
  echo "=== $name (prediction_type=$pred control_reference=$ref epochs=$EPOCHS) ==="
  local start
  start=$(date +%s)
  python /root/Diff-AE-Perturb/scripts/train_with_split.py \
    --data-path "$PREP" \
    --train-path "$BASE/gse132080_train.h5ad" \
    --val-path "$BASE/gse132080_train.h5ad" \
    --test-path "$BASE/gse132080_test.h5ad" \
    --perturbation-col perturbation_label \
    --target-strategy raw \
    --epochs "$EPOCHS" --batch-size 256 --lr 1e-4 --latent-dim 128 \
    --prediction-type "$pred" --control-reference "$ref" \
    --activity-source none --device cuda --seed 7 --save-every "$EPOCHS" \
    --output-dir "$out" > "$out/train.log" 2>&1
  local code=$?
  local secs=$(( $(date +%s) - start ))
  echo "$name exit=$code seconds=$secs"
  tail -n 2 "$out/train.log"
  echo "$name seconds=$secs exit=$code" >> "$TRAIN_ROOT/timing.txt"
}

mkdir -p "$TRAIN_ROOT"
: > "$TRAIN_ROOT/timing.txt"

train_one fix_x0_percell   x0  per_cell
train_one fix_eps_percell  eps per_cell
train_one fix_x0_batchmean x0  batch_mean
train_one fix_eps_batchmean eps batch_mean
train_one legacy_x0_percell x0 encoder_self
echo ALL_DONE
