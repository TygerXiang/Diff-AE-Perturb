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
