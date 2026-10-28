# Agent Context

> 2026-10-28: this file is a historical snapshot and still refers to the deleted
> `C:\Users\Admin\myprojects\diff-project`. The current implementation is
> `src/perturbdiff/`; the benchmark harness is `benchmarks/scperturbench/`.
> See `README.md` for the current layout.

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

## GSE283614 Activity Mapping

Use the reference-screen log2FC table for activity when available:

- reference workbook:
  `C:\Users\Admin\myprojects\diff-project\datasets\GSE283614\1-s2.0-S2213671125001158-mmc2.xlsx`
- sheet:
  `Individual gRNAs`
- source column:
  `Log2(Day 9/Day 0)`
- conversion lives in:
  `src/perturbdiff/gse283614.py::build_log2fc_relative_activity_table`
- formula:
  `screen_depletion_effect = -screen_log2fc_day9_day0`
- relative activity:
  `relative_activity_signed = screen_depletion_effect / reference_depletion_effect`
- model input:
  `relative_activity = max(0, relative_activity_signed)`
- default reference:
  strongest depletion guide in the same reference `Region`
- mapping back to GSE283614:
  pass `media-1.xlsx` as the guide library and join by sgRNA sequence

GSE283614 Table S1 z-scores are a fallback, not the preferred activity source.

- fallback conversion lives in `src/perturbdiff/gse283614.py::build_table_s1_activity_table`
- per state:
  `screen_depletion_effect_STATE = -screen_z_score_STATE`
- relative activity proxy:
  `relative_activity_signed_STATE = screen_depletion_effect_STATE / reference_effect_STATE`
- default reference:
  the strongest depletion effect in the same `(Type, Region)` group, because
  GSE283614 does not currently expose an explicit original sgRNA denominator
- normalized columns:
  `relative_activity_ESC`, `relative_activity_DE`, `relative_activity_mean`
- recommended preprocessing flags:
  `scripts/preprocess_gse283614.py --table-s1-workbook <media-1.xlsx> --log2fc-activity-workbook <1-s2.0-S2213671125001158-mmc2.xlsx>`
- training on GSE283614 direct activity should use `--activity-source direct`
  with `--activity-table ...\gse283614_log2fc_relative_activity.tsv`
  and `--activity-col relative_activity`

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

### Explicit activity condition channel update

Branch `codex/explicit-activity-condition-channel` adds a more explicit DDIM
conditioning path:

- denoiser signature supports:
  `denoiser(x_t, timestep, z_pert, activity_embedding)`
- default predicted-activity semantics on this branch are now:
  - activity does **not** multiply `delta`
  - `delta = delta_base`
  - activity is projected into `activity_embedding` and passed to the denoiser
- legacy folded semantics are still available for old checkpoints / ablations:
  `GuideFeatureEncoderConfig.activity_scales_delta=True`
  or CLI flag `--enable-activity-delta-scaling`

New local 200ep explicit-only run:

- run dir:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\benchmark_activity_branch_predicted_day10_explicit_only_200ep`
- perturbation column:
  `guide_identity`
- target strategy:
  `raw`
- direction align:
  `lambda_direction_align = 0.0`
- activity:
  predicted day10 supervision, explicit denoiser channel only
- best epoch:
  `193`
- best val loss:
  `0.053319`
- test loss:
  `0.054336`

The no-activity 200ep run already satisfies the paired comparison requirement:

- run dir:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\benchmark_activity_branch_none_200ep`
- perturbation column:
  `guide_identity`
- target strategy:
  `raw`
- direction align:
  `lambda_direction_align = 0.0`
- mapping includes per-sgRNA embeddings for the 128 real targeting sgRNAs
  plus the non-guide `*` label

New full six-metric benchmark port:

- script:
  `scripts/evaluate_benchmark.py`
- metrics ported from the reference `diff-project` benchmark:
  - `mse`
  - `e_distance`
  - `pcc_delta`
  - `wasserstein`
  - `kl_divergence`
  - `common_degs`
- guide-level mapping is fixed to prefer the actual guide label before falling
  back to target labels, so sgRNA-level checkpoints use one embedding per guide
- common-DEGs can use the original fallback approximation via:
  `--common-degs-method fallback`

Six-metric explicit-only vs no-activity benchmark:

- output prefix:
  `runs\benchmarks\explicit_only_vs_no_activity_200ep_6metrics_guide_all`
- eval level:
  `guide`
- evaluated guides:
  `127`
- why not 128:
  the checkpoint mapping contains 128 targeting sgRNA embeddings, but the
  prepared test split contains 127 targeting guide labels; the missing mapped
  guide in test is `GNB2L1_GNB2L1_+_180670873.23-P1P2_00`
- min test cells per guide:
  `1`
- DDIM steps:
  `5`
- generated control cells:
  `128`
- metric control cells:
  `211`
- common-DEGs:
  fallback method

Summary:

- `no_activity_200ep`
  - `mse = 0.015533`
  - `e_distance = 1.094355`
  - `pcc_delta = 0.038579`
  - `wasserstein = 0.415086`
  - `kl_divergence = 12.019496`
  - `common_degs = 0.036220`
  - `overall_rank = 1.538`
- `explicit_only_200ep`
  - `mse = 0.006711`
  - `e_distance = 0.723187`
  - `pcc_delta = 0.517564`
  - `wasserstein = 0.430643`
  - `kl_divergence = 12.532225`
  - `common_degs = 0.053858`
  - `overall_rank = 1.462`

Interpretation:

- explicit-only clearly improves mean-profile fit and mean-shift correlation
- explicit-only also improves average e-distance and common-DEGs
- no-activity is better on Wasserstein and KL divergence
- by mean rank across the six metrics, explicit-only is slightly better overall

Embedding/activity correlation analysis:

- script:
  `scripts/analyze_embedding_activity_correlation.py`
- output dir:
  `runs\benchmarks\embedding_activity_correlation_200ep_no_diralign`
- activity:
  `relative_activity_day10`
- both checkpoints use no direction alignment

Results:

- `no_activity_200ep`
  - embedding norm vs activity:
    - Pearson `r = 0.4311`, `p = 3.77e-07`
    - Spearman `rho = 0.5423`, `p = 3.80e-11`
  - within-target embedding norm vs activity rank agreement:
    `0.8377` over `265` usable guide pairs
- `explicit_only_200ep`
  - embedding norm vs activity:
    - Pearson `r = 0.5728`, `p = 1.62e-12`
    - Spearman `rho = 0.6034`, `p = 4.80e-14`
  - predicted activity vs activity:
    - Pearson `r = 0.5127`, `p = 6.13e-10`
    - Spearman `rho = 0.5096`, `p = 8.08e-10`
  - within-target embedding norm vs activity rank agreement:
    `0.8453` over `265` usable guide pairs
  - within-target predicted activity vs activity rank agreement:
    `0.7222` over `252` usable guide pairs

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

## GSE283614 No-activity vs Explicit-activity Benchmark

Final control-correct 200-epoch comparison completed on 2026-07-28.

Data and split:

- prepared combined assigned data:
  `reproducibility/prepared/gse283614/gse283614_combined_assigned.h5ad`
- shape: `14514` cells x `6851` genes
- cell-level stratified split by `guide_identity`, seed `7`:
  - train: `11610`
  - validation: `1452`
  - test: `1452`
- every split contains all `48` guide labels
- train with `--perturbation-col perturbation_label`, not `guide_identity`, so
  both NT and SafeHarbor guides map to control and the model has exactly `33`
  targeting perturbations

Activity definition and coverage:

- source: reference-screen `Log2(Day 9/Day 0)` values
- `screen_depletion_effect = -screen_log2fc_day9_day0`
- `relative_activity_signed = screen_depletion_effect / strongest_same_region_effect`
- model input: `relative_activity = max(0, relative_activity_signed)`
- activity table:
  `reproducibility/prepared/gse283614/gse283614_log2fc_relative_activity.tsv`
- `25 / 33` targeting guides have mapped log2FC activity and `23 / 33` have
  positive activity; missing activity is passed as `0`

Final checkpoints:

- no activity:
  `runs/gse283614_no_activity_controls_fixed_200ep/checkpoint_best.pt`
  - best epoch: `192`
  - best validation loss: `0.0568694926`
  - test loss: `0.0557632291`
- explicit log2FC activity:
  `runs/gse283614_explicit_activity_log2fc_controls_fixed_200ep/checkpoint_best.pt`
  - direct scalar activity, explicit denoiser condition enabled
  - activity delta scaling disabled
  - best epoch: `196`
  - best validation loss: `0.0610632983`
  - test loss: `0.0600633658`

Guide-level generation benchmark:

- final summary:
  `runs/benchmarks/gse283614_no_vs_explicit_log2fc_controls_fixed_200ep_6metrics_summary.csv`
- `33` targeting guides, `128` shared generated test controls, `569` metric
  controls, `5` DDIM steps, fallback common-DEGs
- no activity:
  - `mse = 0.005275`
  - `e_distance = 4.010007`
  - `pcc_delta = 0.057729`
  - `wasserstein = 1.284740`
  - `kl_divergence = 7.950920`
  - `common_degs = 0.072121`
  - `overall_rank = 1.475`
- explicit activity:
  - `mse = 0.004504`
  - `e_distance = 4.545691`
  - `pcc_delta = 0.295953`
  - `wasserstein = 1.251130`
  - `kl_divergence = 8.533606`
  - `common_degs = 0.086667`
  - `overall_rank = 1.525`

Interpretation:

- explicit activity materially improves mean-profile MSE and mean-shift
  correlation; paired guide-level Wilcoxon p-values are `6.21e-5` and
  `8.67e-7`, respectively
- it also improves average Wasserstein and common-DEGs, although the
  common-DEGs difference is not significant (`p = 0.0575`)
- it worsens energy distance and KL divergence, both consistently across
  guides
- on the `25` log2FC-mapped guides, the same tradeoff remains: MSE improves
  from `0.005229` to `0.004514` and PCC-delta from `0.062876` to `0.285278`,
  while energy distance and KL become worse
- the explicit model is preferable when mean response direction/profile is the
  priority, but it is not an across-the-board distributional improvement
- do not use the earlier `gse283614_*_200ep` checkpoints without
  `controls_fixed` for conclusions; those runs treated SafeHarbor labels as
  perturbations during training and are retained only as diagnostics

## GSE283614 Biological Dose Validation

Implemented on 2026-07-28:

- statistical core: `src/perturbdiff/gse283614_biology.py`
- observed analysis: `scripts/analyze_gse283614_biology.py`
- leakage-free fold builder: `scripts/create_gse283614_dose_cv.py`
- gated training runner: `scripts/run_gse283614_dose_cv.py`
- OOF evaluator: `scripts/evaluate_gse283614_dose_cv.py`
- gated full refit and dose scan:
  - `scripts/train_gse283614_full_dose.py`
  - `scripts/scan_gse283614_virtual_dose.py`

Training data now supports `activity_label_col` independently of
`perturbation_col`. The strict dose experiment uses `guide_family` for the two
target embeddings and `guide_identity` to join Table S4 knockdown fractions.
Missing direct-activity values raise an error instead of being filled with 0.

Formal observed-analysis outputs:

- `runs/biological_validation/gse283614_observed`
- complete coverage: 48/48 guides and 33/33 targeting guides
- Table S4 reported model classes match all six expected relationships
- independent refitting does not satisfy the preregistered full gate:
  - NANOG DE differentiation selects quadratic, but its fitted vertex is just
    below the observed dose range
  - OCT4 DE off-target values in the score sheet refit as quadratic, whereas
    `Gene_Score_Model_Selection` reports Hill
- essentiality buffering passes in all four state/TF comparisons after using
  expression-scale-normalized dynamic ranges; the gate requires both the
  expected median direction and a one-sided Mann-Whitney p-value below 0.05
- log2FC sensitivity maps 25/33 targeting guides and correlates with Table S4
  knockdown fraction at Spearman rho 0.6696 (p=0.000251)
- `observed_stage_gate_pass = false`; formal 200-epoch OOF training must remain
  blocked until the score/model discrepancy is resolved

Response-shape gates are strict: quadratic endpoints require downward
curvature and a vertex inside the observed dose range, while Hill endpoints
require an in-range ED50 and Hill coefficient greater than one. The normalized
Hill fit fixes the score asymptotes and estimates ED50 plus cooperativity, which
matches the Table S4 AIC convention more closely than a free-asymptote fit.

The five-fold design was validated in manifest-only mode at:

- `reproducibility/prepared/gse283614/dose_cv/dose_cv_manifest.json`

Every fold contains both NANOG and OCT4 test guides, uses disjoint targeting
guides across train/validation/test, and partitions state-matched control cells
60/20/20 by barcode.

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
