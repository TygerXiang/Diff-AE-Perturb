# PerturbDiff

PerturbDiff is a compact PyTorch implementation of a conditional DDIM model for
single-cell perturbation response prediction. The current model intentionally
removes all perturbation-strength heads, strength losses, ranking losses, anchor
losses, entropy terms, and explicit target-direction regularizers.

The core idea is simple:

```text
z_condition = E(x_control) + delta_g
x_pred = DDIM(z_condition)
```

The perturbation vector `delta_g` is estimated in the scGen style:

```text
delta_g = mean(E(x) | condition = g) - mean(E(x) | control)
```

Perturbation labels can represent target genes or individual sgRNAs. To preserve
sgRNA-level conditions for future strength modeling, use `--target-strategy raw`.

## Model Components

- `SemanticEncoder`: maps expression vectors `x_0 in R^G` to latent vectors
  `z_sem in R^d`.
- `PerturbationModule`: stores one latent delta vector per non-control
  condition; id `0` is reserved for controls and fixed to zero.
- `ConditionalDenoiserMLP`: predicts diffusion noise or clean expression,
  depending on `DiffusionConfig.prediction_type`.
- `GaussianDiffusion`: implements forward noising and deterministic DDIM
  sampling.

## Quick Start

```bash
pip install -e ".[dev]"
python scripts/smoke_train.py
python -m unittest discover -s tests
```

## Training On Prepared Splits

For the GSE132080 prepared splits:

```bash
python scripts/train_with_split.py \
  --data-path reproducibility/prepared/gse132080_preprocessed.h5ad \
  --train-path reproducibility/prepared/splits/gse132080_train.h5ad \
  --val-path reproducibility/prepared/splits/gse132080_val.h5ad \
  --test-path reproducibility/prepared/splits/gse132080_test.h5ad \
  --perturbation-col perturbation_label \
  --target-strategy raw \
  --prediction-type x0 \
  --epochs 200 \
  --batch-size 256 \
  --lr 1e-4 \
  --diffusion-steps 1000 \
  --save-every 20 \
  --output-dir runs/gse132080_cond_ddim_raw_sgrna_200ep
```

Use `--target-strategy first_token` only when you explicitly want to collapse
sgRNAs to target-gene labels.

## Training On Other Data

For 10x-style `matrix.mtx(.gz)` data with a metadata table:

```bash
python scripts/train.py \
  --data-format 10x \
  --data-path datasets/GSE132080 \
  --metadata-path datasets/GSE132080/GSE132080_cell_identities.csv.gz \
  --perturbation-col guide_identity \
  --barcode-col cell_barcode \
  --target-strategy raw \
  --output-dir runs/gse132080
```

For `.h5ad` data:

```bash
python scripts/train.py \
  --data-format h5ad \
  --data-path path/to/train.h5ad \
  --perturbation-col guide_identity \
  --target-strategy raw \
  --output-dir runs/example
```

## GSE283614 Dose-Response Workflow

GSE283614 stores gene expression and CRISPR guide-capture features in the same
10x matrices, so prepare it before training:

```bash
python scripts/preprocess_gse283614.py \
  --data-root C:/Users/Admin/myprojects/diff-project/datasets/GSE283614 \
  --gene-set-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-3.xlsx \
  --table-s1-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-1.xlsx \
  --log2fc-activity-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/1-s2.0-S2213671125001158-mmc2.xlsx \
  --output-dir reproducibility/prepared/gse283614
```

If `--table-s1-workbook` is provided, preprocessing also writes
`gse283614_table_s1_activity.tsv` from the GSE283614 z-score columns. Prefer the
reference-screen log2FC path when `--log2fc-activity-workbook` is available:
it writes `gse283614_log2fc_relative_activity.tsv`, maps rows back to GSE283614
guides by sgRNA sequence, and computes `relative_activity` from
`Log2(Day 9/Day 0)` as `max(0, -log2FC / reference_depletion_effect)`. The
default reference is the strongest depletion guide in the same reference
region.

Create guide-held-out splits to avoid cell-level leakage:

```bash
python scripts/create_h5ad_splits.py \
  --input-h5ad reproducibility/prepared/gse283614/gse283614_esc_assigned.h5ad \
  --output-dir reproducibility/prepared/gse283614/splits_esc \
  --prefix gse283614_esc \
  --split-mode group \
  --group-col guide_identity \
  --group-stratify-col guide_family \
  --stratify-col perturbation_label
```

Train a target-level dose model after creating the Table S4 dose table:

```bash
python scripts/train_with_split.py \
  --data-path reproducibility/prepared/gse283614/gse283614_esc_assigned.h5ad \
  --train-path reproducibility/prepared/gse283614/splits_esc/gse283614_esc_train.h5ad \
  --val-path reproducibility/prepared/gse283614/splits_esc/gse283614_esc_val.h5ad \
  --test-path reproducibility/prepared/gse283614/splits_esc/gse283614_esc_test.h5ad \
  --perturbation-col guide_family \
  --activity-label-col guide_identity \
  --target-strategy raw \
  --control-label control \
  --prediction-type x0 \
  --activity-source direct \
  --activity-table runs/biological_validation/gse283614_observed/gse283614_tf_dosage.tsv \
  --activity-col knockdown_fraction \
  --output-dir runs/gse283614_esc
```

After training, generate guide ranking, state-specific dose-response summaries,
and control-to-perturbation expression changes:

```bash
python scripts/analyze_gse283614_dosage.py \
  --adata-path reproducibility/prepared/gse283614/gse283614_esc_assigned.h5ad \
  --checkpoint runs/gse283614_esc/checkpoint_best.pt \
  --activity-table runs/biological_validation/gse283614_observed/gse283614_tf_dosage.tsv \
  --activity-col knockdown_fraction \
  --gene-set-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-3.xlsx \
  --reference-dosage-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-4.xlsx \
  --output-dir runs/gse283614_esc/dosage
```

### GSE283614 biological dose validation

Use Table S4 TF expression percentages as the primary dose axis. The analysis
requires complete 48-guide / 33-targeting-guide coverage, fits the six reported
linear, quadratic, and Hill relationships, bootstraps guide-level curves, and
tests essentiality buffering without treating cells as independent replicates:

The normalized Hill parameterization fixes the two score asymptotes and fits
ED50 plus the Hill coefficient, matching the AIC convention in Table S4. A
Hill endpoint passes only when ED50 lies inside the observed dose range and the
Hill coefficient is greater than one. Essentiality buffering requires both a
smaller median relative dynamic range and a one-sided Mann-Whitney p-value
below 0.05.

```bash
python scripts/analyze_gse283614_biology.py \
  --adata-path reproducibility/prepared/gse283614/gse283614_combined_assigned.h5ad \
  --dosage-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-4.xlsx \
  --guide-library-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-1.xlsx \
  --gene-set-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-3.xlsx \
  --log2fc-activity-table reproducibility/prepared/gse283614/gse283614_log2fc_relative_activity.tsv \
  --output-dir runs/biological_validation/gse283614_observed \
  --bootstrap 1000
```

Create the five-fold guide-held-out design after inspecting
`observed_biology_summary.json`. Use `--manifest-only` to validate the design
without writing the large h5ad split files:

```bash
python scripts/create_gse283614_dose_cv.py \
  --adata-path reproducibility/prepared/gse283614/gse283614_combined_assigned.h5ad \
  --dosage-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-4.xlsx \
  --guide-library-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-1.xlsx \
  --output-dir reproducibility/prepared/gse283614/dose_cv \
  --manifest-only
```

`run_gse283614_dose_cv.py` blocks formal training unless the observed stage
gate passes. It trains state-specific target-only and target+dose models with
`guide_family` as the perturbation class and `guide_identity` as the separate
activity lookup key. `evaluate_gse283614_dose_cv.py` creates OOF guide scores,
paired bootstrap comparisons, six response-curve fits, and the final pass/fail
summary. Full-data refitting and `scan_gse283614_virtual_dose.py` are separately
gated and explicitly labeled exploratory.

## Minimal Usage

```python
import torch
from perturbdiff import DiffusionConfig, ModelConfig, PerturbDiffusionModel

cfg = ModelConfig(
    n_genes=2000,
    n_perturbations=128,
    latent_dim=128,
    diffusion=DiffusionConfig(prediction_type="x0"),
)
model = PerturbDiffusionModel(cfg)

x0 = torch.randn(32, cfg.n_genes)
perturbation_ids = torch.randint(0, cfg.n_perturbations + 1, (32,))

with torch.no_grad():
    z_sem = model.encoder(x0)
    model.initialize_perturbation_from_latent_shift(z_sem, perturbation_ids)

out = model.training_losses(x0, perturbation_ids)
out["loss"].backward()

samples = model.sample_counterfactual(
    x_ctrl=x0[:4],
    target_ids=torch.tensor([1, 2, 3, 4]),
    steps=25,
)
```

## Notes

The code is written from scratch for this repository. It follows standard
DDPM/DDIM equations and does not copy code from OpenAI guided-diffusion or
improved-diffusion repositories.

When describing this implementation, cite the methodological sources rather
than software unless software was directly reused:

- DDPM: Ho, Jain, and Abbeel, NeurIPS 2020.
- DDIM: Song, Meng, and Ermon, ICLR 2021.
- scGen-style latent vector arithmetic for perturbation deltas.
