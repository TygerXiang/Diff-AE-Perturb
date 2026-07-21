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

### Predicted sgRNA Activity Conditioning Branch

Current active Git branch:

- `codex/predicted-guide-activity-conditioning`

This branch adds predicted sgRNA activity conditioning to the conditional DDIM
path. The default model behavior remains unchanged unless
`ModelConfig.guide_encoder.enabled` is set to `True`.

New implementation pieces:

- `GuideFeatureEncoderConfig` in `src/perturbdiff/config.py`
- `GuideStrengthEncoder` and Table_S8-style guide feature utilities in
  `src/perturbdiff/guide_features.py`
- target-specific `GeneralizedSigmoid` doser in `src/perturbdiff/modules.py`
- `PerturbDiffusionModel.training_losses()` and `sample_counterfactual()` now
  accept optional `guide_features`
- `PerturbationDataset` can retain raw `perturbation_labels` and return
  `guide_features` / `guide_activity_targets` in dataloader batches

Activity conditioning semantics in this branch:

1. `GuideStrengthEncoder` predicts a non-negative guide activity / strength
   scalar from Table_S8 notebook-style `(genome input, sgRNA input)` CNN
   features.
2. `GeneralizedSigmoid(strength, target_id)` maps strength to a target-specific
   scale.
3. The scale multiplies the learned perturbation delta.
4. The resulting condition latent is used through the existing `z_pert` DDIM
   denoiser input.

Training entrypoint in this copy:

- `scripts/train_with_split.py`

Predicted activity flags:

- `--activity-source predicted`
- `--guide-activity-cnn-table <Table_S8_machine_learning_input.txt>`
- optional supervision:
  `--activity-table <Table_S16_perturb-seq_screen_phenotypes.txt>`
- `--activity-col relative_activity_day10`
- `--guide-sequence-length 26`
- `--guide-strength-lambda 0.1`

Important limitation versus `C:\Users\Admin\myprojects\diff-project`:

- this copy supports the predicted CNN activity mode needed for this branch
- it does not yet include the full reference project's direct-oracle activity
  mode, checkpoint-init/freeze-delta flags, or full benchmark script family

### Activity-conditioning branch benchmark state

New lightweight benchmark helper added in this branch:

- `scripts/benchmark_activity_conditioning.py`

Purpose:

- compare guide-level counterfactual generation before/after predicted activity
  conditioning
- use test-set control cells as shared source inputs
- evaluate real guide test cells against generated cells
- report:
  - mean-profile `mse` (lower is better)
  - full-gene `pcc_delta` on mean shifts (higher is better)
  - simplified guide-level `e_distance` (lower is better)

Reference benchmark note:

- the full `C:\Users\Admin\myprojects\diff-project\scripts\evaluate_benchmark.py`
  guide-level benchmark was attempted against this branch by pointing its
  `_REPO` to `C:\Users\Admin\myprojects\Diff-AE-Perturb`
- that full run timed out after 30 minutes before writing outputs
- for this branch, use the lightweight script for fast smoke comparisons, and
  reserve the full reference benchmark for longer unattended runs

Current quick before/after experiment:

- split:
  `C:\Users\Admin\myprojects\diff-project\reproducibility\prepared\splits`
- perturbation column:
  `guide_identity`
- target strategy:
  `raw`
- epochs:
  `20`
- device:
  `cuda`
- predicted activity feature table:
  `C:\Users\Admin\myprojects\diff-project\datasets\GSE132080\41587_2019_387_MOESM3_ESM\Table_S8_machine_learning_input.txt`
- activity supervision table:
  `C:\Users\Admin\myprojects\diff-project\datasets\GSE132080\41587_2019_387_MOESM3_ESM\Table_S16_perturb-seq_screen_phenotypes.txt`
- activity column:
  `relative_activity_day10`

Run outputs:

- no activity:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\benchmark_activity_branch_none_20ep`
- predicted activity:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\benchmark_activity_branch_predicted_day10_20ep`
- lightweight benchmark:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\benchmarks\activity_branch_20ep_light_summary.csv`

Training scalar results:

- no activity:
  - `best_epoch = 20`
  - `best_val_loss = 0.038052`
  - `test_loss = 0.038414`
- predicted activity:
  - `best_epoch = 20`
  - `best_val_loss = 0.063606`
  - `test_loss = 0.063633`
  - note: total loss includes `0.1 * guide_strength_loss`; at epoch 20,
    `val_eval_mse = 0.054792` and `val_strength = 0.088140`

Lightweight guide-level benchmark result:

- output prefix:
  `activity_branch_20ep_light`
- evaluated guides:
  `20`
- DDIM steps:
  `5`
- max control cells:
  `128`

Summary:

- `no_activity_20ep`
  - `mse = 0.010209`
  - `pcc_delta = 0.109135`
  - `e_distance = 1.489972`
- `predicted_activity_20ep`
  - `mse = 0.009001`
  - `pcc_delta = 0.298359`
  - `e_distance = 1.905398`

Current interpretation:

- in this short 20-epoch smoke benchmark, predicted activity improves
  mean-profile fit and mean-shift correlation on the sampled guides
- predicted activity worsens the simplified distribution-distance metric
- because this is a short, limited-guide smoke run with 5-step DDIM, it should
  not be treated as a final benchmark conclusion

### 200ep benchmark update

200ep training runs have now been completed for the same split:

- `runs/benchmark_activity_branch_none_200ep`
- `runs/benchmark_activity_branch_predicted_day10_200ep`

The 200ep benchmark was run in chunks to avoid timeout:

- helper script:
  `scripts/benchmark_activity_conditioning.py`
- chunk size:
  `20` guides
- DDIM steps:
  `5`
- evaluated guides:
  `84`

Combined 200ep guide-level results:

- `no_activity_200ep`
  - `mse = 0.012284`
  - `pcc_delta = 0.073446`
  - `e_distance = 0.976192`
- `predicted_activity_200ep`
  - `mse = 0.006568`
  - `pcc_delta = 0.462830`
  - `e_distance = 0.793576`

Interpretation:

- on this 84-guide chunked benchmark, predicted activity clearly improves
  both mean-profile fit and mean-shift correlation
- it also improves the simplified distribution distance
- the 200ep activity-conditioned model therefore looks materially better than
  the no-activity control on this quick guide-level benchmark
- the result is stronger than the 20ep smoke run and is the current best
  evidence in this branch for predicted activity conditioning helping

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
