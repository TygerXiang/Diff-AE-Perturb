#!/usr/bin/env bash
# Score Squidiff under three process_xstart policies from one trained checkpoint.
#
#   squidiff_repro          upstream defaults (sampler.pred -> pred_xstart.clamp(0,))
#   squidiff_repro_clip     Squidiff's own clip_denoised=True branch (clamp(-1,1))
#   squidiff_repro_noclamp  guided-diffusion's original unclamped pred_xstart
#
# The checkpoint is the stable upstream-architecture run (lr_anneal_steps=2e4).  The upstream
# default horizon of 1e5 collapses on this dataset; see reports/.
set -u

CHECKPOINT_DIR="${1:-/root/perturbench/.scperturbench/results/gse132080_6metrics/gse132080/squidiff/seed7/squidiff}"
BASE=/root/diff-project/reproducibility/prepared/splits_unseen_guide_seed7
DATA=/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro
RESULTS=/root/perturbench/.scperturbench/results/gse132080_6metrics/gse132080

if [ ! -f "$CHECKPOINT_DIR/checkpoint/model.pt" ]; then
  echo "missing checkpoint: $CHECKPOINT_DIR/checkpoint/model.pt" >&2
  exit 2
fi

run_variant () {
  local name="$1"
  local policy="$2"
  local out="$RESULTS/$name/seed7"
  mkdir -p "$out"
  echo "=== $name (process_xstart=$policy) ==="
  python /root/perturbench/scperturbench_adapters.py \
    --adapter squidiff_repro \
    --method "$name" \
    --seed 7 \
    --train-path "$BASE/gse132080_train.h5ad" \
    --test-path "$BASE/gse132080_test.h5ad" \
    --squidiff-data-dir "$DATA" \
    --squidiff-train-dir "$CHECKPOINT_DIR" \
    --squidiff-steps 20000 \
    --squidiff-xstart-clamp "$policy" \
    --output-dir "$out" > "$out/adapter.log" 2>&1
  local code=$?
  echo "$name exit=$code"
  tail -n 4 "$out/adapter.log"
}

run_variant squidiff_repro upstream
run_variant squidiff_repro_clip neg1_1
run_variant squidiff_repro_noclamp none
echo ALL_DONE
