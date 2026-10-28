#!/usr/bin/env bash
# Locate and print the upstream preprocessing that produced Squidiff's perturbation input.
set -u
OUT=/root/perturbench/logs/gears_prep.log
GEARS=/usr/local/anaconda3/envs/gears/lib/python3.9/site-packages/gears
: > "$OUT"
echo "### files in gears package" >> "$OUT"
ls "$GEARS" >> "$OUT" 2>&1
echo "### grep for the preprocessing primitives" >> "$OUT"
grep -rn "normalize_total\|sc.pp.log1p\|highly_variable_genes\|filter_genes\|filter_cells\|perturb_processed" "$GEARS" >> "$OUT" 2>&1
echo "### pertdata.py: function definitions and the preprocess body" >> "$OUT"
grep -n "def \|perturb_processed\|sc.pp\." "$GEARS/pertdata.py" >> "$OUT" 2>&1
echo DONE
