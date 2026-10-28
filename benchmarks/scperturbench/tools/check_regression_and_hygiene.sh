#!/usr/bin/env bash
# (a) regression: does the historical `diffae` baseline still reproduce after the refactor?
#     (legacy checkpoints get control_reference="per_cell", which is what they were evaluated with)
# (b) hygiene: is the unseen-guide "val" split actually distinct from "train"?
set -u

BASE=/root/diff-project/reproducibility/prepared/splits_unseen_guide_seed7
CKPT=/root/Diff-AE-Perturb/runs/gse132080_target_200ep/checkpoint_best.pt
OUT=/root/perturbench/.scperturbench/results/gse132080_6metrics/gse132080/diffae_regression/seed7
mkdir -p "$OUT"

{
  echo "### split hashes"
  sha256sum "$BASE/gse132080_train.h5ad" "$BASE/gse132080_val.h5ad" "$BASE/gse132080_test.h5ad"
} > /root/perturbench/logs/regression_hygiene.log 2>&1

echo "### diffae regression run" >> /root/perturbench/logs/regression_hygiene.log
python /root/perturbench/scperturbench_adapters.py \
  --adapter diffae --method diffae_regression --seed 7 \
  --repo /root/Diff-AE-Perturb --checkpoint "$CKPT" --device cuda --ddim-steps 50 \
  --train-path "$BASE/gse132080_train.h5ad" --test-path "$BASE/gse132080_test.h5ad" \
  --output-dir "$OUT" >> /root/perturbench/logs/regression_hygiene.log 2>&1
echo "adapter exit=$?" >> /root/perturbench/logs/regression_hygiene.log

python /root/perturbench/scperturbench_metrics.py \
  --result-h5ad "$OUT/result.h5ad" --test-path "$BASE/gse132080_test.h5ad" \
  --deg-table /root/perturbench/.scperturbench/inputs/gse132080_bench/deg_table.json \
  --output-dir "$OUT/score" --method diffae_regression --seed 7 \
  >> /root/perturbench/logs/regression_hygiene.log 2>&1
echo "metrics exit=$?" >> /root/perturbench/logs/regression_hygiene.log
echo ALL_DONE
