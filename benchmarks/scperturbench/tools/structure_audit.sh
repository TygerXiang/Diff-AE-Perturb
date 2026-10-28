#!/usr/bin/env bash
# Structural audit of the Diff-AE-Perturb repo: duplication, drift, and stale artefacts.
set -u
REPO=/home/../root/Diff-AE-Perturb
OUT=/root/perturbench/logs/structure_audit.log
: > "$OUT"

echo "### package copies named 'perturbdiff'" >> "$OUT"
find / -maxdepth 6 -type d -name perturbdiff -not -path '*/proc/*' 2>/dev/null >> "$OUT"
echo "### file sizes per copy (model/modules/losses/data/config)" >> "$OUT"
for d in /root/Diff-AE-Perturb/src/perturbdiff /root/diff-project/src/perturbdiff; do
  echo "-- $d" >> "$OUT"
  ls -l "$d" 2>/dev/null | awk '{print $5, $9}' >> "$OUT"
done
echo "### which files exist only in one copy" >> "$OUT"
comm -3 <(ls /root/Diff-AE-Perturb/src/perturbdiff | sort) <(ls /root/diff-project/src/perturbdiff | sort) >> "$OUT"

echo "### preprocess_gse132080.py copies" >> "$OUT"
find /root -name 'preprocess_gse132080.py' 2>/dev/null >> "$OUT"
echo "### train_with_split.py copies" >> "$OUT"
find /root -name 'train_with_split.py' 2>/dev/null >> "$OUT"
echo "### evaluate_benchmark.py copies" >> "$OUT"
find /root -name 'evaluate_benchmark.py' 2>/dev/null >> "$OUT"

echo "### runs/ top-level entries and apparent size" >> "$OUT"
ls /root/Diff-AE-Perturb/runs 2>/dev/null | head -40 >> "$OUT"
echo "count: $(ls /root/Diff-AE-Perturb/runs 2>/dev/null | wc -l)" >> "$OUT"
du -sh /root/Diff-AE-Perturb/runs 2>/dev/null >> "$OUT"
echo "### checkpoint files in the target run" >> "$OUT"
ls -l /root/Diff-AE-Perturb/runs/gse132080_target_200ep/*.pt 2>/dev/null | awk '{print $5, $9}' >> "$OUT"
echo "### .h5ad prepared copies" >> "$OUT"
find /root/diff-project/reproducibility/prepared -maxdepth 2 -name '*.h5ad' 2>/dev/null | head -20 >> "$OUT"
echo "### reproducibility snapshots that duplicate src/" >> "$OUT"
find /root/Diff-AE-Perturb/reproducibility -maxdepth 3 -type d -name perturbdiff 2>/dev/null >> "$OUT"
echo DONE
