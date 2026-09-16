# GSE132080 diffusion model preliminary results

## Scope

- 128 targeting sgRNAs from `comprehensive_sgrna_table.csv`.
- Expression values are normalized values from the 1,019-gene GSE132080 test matrix.
- Target-gene delta is `guide mean expression - control-cell mean expression`.
- Predicted expression is generated from the sgRNA-level checkpoint with 50 DDIM steps.
- `zsem_000..127` is the encoder output for each guide's observed mean expression profile.
- `sgrna_delta_000..127` is the learned sgRNA perturbation embedding from the checkpoint.

## Coverage

- Rows: 128
- Target-gene expression available: 106
- Target gene absent from 1,019-gene panel: 21
- Guide absent from test split: 1
- Control cells used: 211
- Latent dimensions: 128

## Important interpretation

`target_genomic_sequence` is the 20-nt genomic protospacer, and `pam_sequence` is reported separately. It is not the full genomic span or transcript/CDS of the target gene. `z_sem` is a 128-dimensional cell-expression latent representation, while `sgrna_delta` is the learned 128-dimensional perturbation vector; neither should be interpreted as a single expression measurement.

## Activity input

The primary `gse132080_sgrna_200ep` checkpoint used for this table did not receive activity as a training input (`activity_source` and `conditioning_summary` are absent in its metrics). Its `virtual_scale` is a post-training ratio derived from learned sgRNA embedding norms relative to the fully matched guide.

The separate predicted-activity benchmark uses two 26-nt Table S8 sequences (`genome input` and `sgRNA input`) as a 4 x 26 x 2 one-hot tensor (208 features). A CNN predicts a non-negative guide strength and Table S16 `relative_activity_day10` supplies an auxiliary MSE target with `lambda_strength=0.1`. In the current model implementation the strength is transformed by a target-specific generalized sigmoid, then supplied as an explicit activity embedding to the denoiser when `explicit_condition=True`; optional `activity_scales_delta=True` can additionally multiply the latent perturbation delta.
