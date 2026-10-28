#!/usr/bin/env bash
# Test the conditioning-pathway diagnosis causally: re-run the Diff-AE-Perturb benchmark with
# the per-perturbation latent offsets repaired, everything else frozen.
#
#   diffae_delta_recompute : offsets recomputed from the *trained* encoder (scGen-style, but at
#                            the right time) — this is the one-line architectural fix
#   diffae_delta_x50       : the learned offsets simply multiplied by 50
set -u

BASE=/root/diff-project/reproducibility/prepared/splits_unseen_guide_seed7
CKPT=/root/Diff-AE-Perturb/runs/gse132080_target_200ep/checkpoint_best.pt
RESULTS=/root/perturbench/.scperturbench/results/gse132080_6metrics/gse132080

run_variant () {
  local name="$1"; shift
  local out="$RESULTS/$name/seed7"
  mkdir -p "$out"
  echo "=== $name ==="
  python /root/perturbench/scperturbench_adapters.py \
    --adapter diffae --method "$name" --seed 7 \
    --repo /root/Diff-AE-Perturb --checkpoint "$CKPT" --device cuda --ddim-steps 50 \
    --train-path "$BASE/gse132080_train.h5ad" --test-path "$BASE/gse132080_test.h5ad" \
    --output-dir "$out" "$@" > "$out/adapter.log" 2>&1
  echo "$name exit=$?"
  tail -n 3 "$out/adapter.log"
}

run_variant diffae_delta_recompute --diffae-delta-mode recompute
run_variant diffae_delta_x50       --diffae-delta-mode rescale --diffae-delta-scale 50
echo ALL_DONE
