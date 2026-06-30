# Agent Context

This file is the actively maintained model context for this repository copy.
It captures volatile implementation and runtime facts for the `341dae5`
historical snapshot copied from `C:\Users\Admin\myprojects\diff-project`.

## Snapshot Identity

- Source repository: `C:\Users\Admin\myprojects\diff-project`
- Exported commit: `341dae5`
- Commit date: `2026-05-26 22:40:00 +0800`

## Canonical Runtime

- Preferred interpreter:
  `C:\Users\Admin\myprojects\diff-project\.conda\envs\perturbdiff-gse132080\python.exe`
- Python: `3.10.20`
- PyTorch: `2.11.0+cu128`
- CUDA availability in the preferred interpreter: `True`
- Verified GPU on this workstation: `NVIDIA GeForce RTX 5060`

Important reminder:

- Do not use the default system `python` for real training in this copy.
- The system interpreter currently resolves to `torch 2.11.0+cpu`, which will
  silently run training on CPU.

## Available Data

The original project already contains the dataset assets needed for training:

- 10x dataset directory:
  `C:\Users\Admin\myprojects\diff-project\datasets\GSE132080`
- metadata table:
  `C:\Users\Admin\myprojects\diff-project\datasets\GSE132080\GSE132080_cell_identities.csv.gz`
- prepared full `.h5ad`:
  `C:\Users\Admin\myprojects\diff-project\reproducibility\prepared\gse132080_preprocessed.h5ad`
- prepared train split:
  `C:\Users\Admin\myprojects\diff-project\reproducibility\prepared\splits\gse132080_train.h5ad`
- prepared val split:
  `C:\Users\Admin\myprojects\diff-project\reproducibility\prepared\splits\gse132080_val.h5ad`
- prepared test split:
  `C:\Users\Admin\myprojects\diff-project\reproducibility\prepared\splits\gse132080_test.h5ad`

## Current Code Semantics In This Snapshot

This `341dae5` snapshot has now been locally modified for sgRNA-level virtual
strength experiments:

- `TrainingLossConfig` includes:
  - `lambda_direction_align`
- `PerturbationDataset` carries `target_ids`
- `training_losses()` computes `direction_align_loss`
- `train.py` and `train_with_split.py` pass `target_ids` into the model

The implemented direction-alignment behavior in this snapshot is:

- controls are excluded via `target_ids != 0`
- only same-target pairs are used
- only upper-triangular unique pairs are counted
- the loss now directly pulls same-target cosine similarity toward `1`:
  `(1 - cos).mean()`

Important architecture note:

- training-time decoding in this local copy still conditions on `z_sem`
- `PerturbationModule.forward()` still computes:
  - `z_basal = z_sem - delta`
  - `z_pert = z_sem`
- but counterfactual sampling in this local copy has been changed to:
  - build `z_basal` from control input
  - sample with `z_basal + s * delta`

## Local Strength Experiment State

The current local copy contains a sgRNA-level virtual-strength extension:

1. Counterfactual sampling uses `z_basal + s * delta`
2. `s` is not supervised during training
3. after training, each sgRNA receives a virtual `s` by:
   - taking the learned guide embedding norm `||delta_g||`
   - dividing by the norm of that guide family's fully matched sgRNA
4. fully matched guide families are resolved from:
   - `Table_S2_large-scale_library_sequences_revised.txt`
   - `Table_S10_compact_library_sequences_revised.txt`
5. exported guide-scale tables also include:
   - target
   - raw sgRNA sequence
   - fully matched sgRNA family label

## Latest Local Run

Latest completed 200-epoch target-level run:

- run dir:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_target_200ep`
- perturbation column:
  `perturbation_label`
- target strategy:
  `raw`
- epochs:
  `200`
- device:
  `cuda`
- best epoch:
  `198`
- best val loss:
  `0.02314953412860632`
- test loss:
  `0.023286994779482484`

Latest completed 200-epoch sgRNA-level run:

- run dir:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_200ep`
- perturbation column:
  `guide_identity`
- target strategy:
  `raw`
- epochs:
  `200`
- device:
  `cuda`
- best epoch:
  `199`
- best val loss:
  `0.023033942095935345`
- test loss:
  `0.023287909105420113`

Older completed sgRNA-level smoke/baseline run:

- run dir:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_zbasal_sdelta_diralign_nomargin_20ep`
- perturbation column:
  `guide_identity`
- target strategy:
  `raw`
- epochs:
  `20`
- device:
  `cuda`
- best val loss:
  `0.038052564325779676`
- test loss:
  `0.03841410665860772`

Guide-scale export:

- full csv:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_200ep\guide_scale_export\guide_virtual_scales.csv`
- full json:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_200ep\guide_scale_export\guide_virtual_scales.json`
- filtered 128-sgRNA csv:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_200ep\guide_scale_export\guide_virtual_scales_128_sgrnas.csv`
- filtered 128-sgRNA json:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_200ep\guide_scale_export\guide_virtual_scales_128_sgrnas.json`
- csv:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_zbasal_sdelta_diralign_nomargin_20ep\guide_scale_export\guide_virtual_scales.csv`
- json:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_zbasal_sdelta_diralign_nomargin_20ep\guide_scale_export\guide_virtual_scales.json`

Example per-target virtual `s` behavior now looks reasonable:

- `ALDOA`
  - `_00`: `1.0`
  - `_06`: `1.1583`
  - `_13`: `0.4972`
- `BCR`
  - `_00`: `1.0`
  - `_07`: `1.0205`
  - `_15`: `0.2170`
- `HSPA5`
  - `_00`: `1.0`
  - `_08`: `1.2662`
  - `_06`: `0.3943`

## Latest Comparison Result

Formal 200-epoch guide-specific generation comparison:

- comparison json:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_target_vs_sgrna_200ep\comparison.json`
- target-level checkpoint:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_target_200ep\checkpoint_best.pt`
- sgRNA-level checkpoint:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_200ep\checkpoint_best.pt`

Evaluation protocol used in that comparison:

- test split:
  `C:\Users\Admin\myprojects\diff-project\reproducibility\prepared\splits\gse132080_test.h5ad`
- control cells from the test split are used as shared inputs
- for each guide, both models generate counterfactual expression means from the
  same control pool
- guide-specific accuracy is measured against the real test-set mean profile of
  that guide

Important dataset detail:

- the exportable library still contains `128` real targeting sgRNAs
- the prepared test split contains all `128` targeting sgRNA labels
- but one guide has only `1` test cell and one non-guide label `*` is excluded
- the comparison therefore reports `127` evaluated guide rows in
  `comparison.json`

Current conclusion from the 200-epoch comparison:

- sgRNA-level training does **not** show a stable generation-accuracy
  improvement over target-level training in this local setup
- summary metrics are nearly tied, with a slight edge to target-level

Headline numbers from `comparison.json`:

- all evaluated guides (`127`)
  - target mean `pearson_r2`: `0.950923324787982`
  - sgRNA mean `pearson_r2`: `0.9508214757806733`
  - target mean `mse`: `0.015122626215758182`
  - sgRNA mean `mse`: `0.015204287157627417`
- guides with at least `5` test cells (`123`)
  - target mean `pearson_r2`: `0.9534584563650425`
  - sgRNA mean `pearson_r2`: `0.9533673495903316`
  - target mean `mse`: `0.014318739859595435`
  - sgRNA mean `mse`: `0.014388216766462007`
- guides with at least `10` test cells (`84`)
  - target mean `pearson_r2`: `0.9612901746666912`
  - sgRNA mean `pearson_r2`: `0.9610807134108076`
  - target mean `mse`: `0.011943215725410117`
  - sgRNA mean `mse`: `0.012005228791526754`

Per-guide win counts from the same comparison:

- all evaluated guides
  - sgRNA better in `pearson_r2` on `59 / 127`
  - sgRNA lower in `mse` on `56 / 127`
- guides with at least `10` test cells
  - sgRNA better in `pearson_r2` on `35 / 84`
  - sgRNA lower in `mse` on `36 / 84`

## Recommended Training Entry Points

- For 10x data:
  `scripts/train.py`
- For prepared `.h5ad` splits:
  `scripts/train_with_split.py`

For this historical copy, using the prepared split files from the source
repository is the most convenient way to run reproducible smoke or benchmark
training without rebuilding preprocessing outputs locally.

## Output Convention

- Put run outputs under:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs`

## Notes

- This directory is an exported source snapshot, not a Git checkout.
- It does not include the original repository `.git` history unless copied
  separately later.
