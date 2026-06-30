# PerturbDiff preprocessing workflow

This note documents the preprocessing expected before running PerturbDiff on a
Perturb-seq/CRISPRi dataset. It follows the same high-level pattern used in
`Squidiff_reproducibility`: create a clean AnnData object first, encode the
condition/perturbation metadata, select a manageable gene set, save `.h5ad`, and
then train from that prepared file.

## What Squidiff Does

The Squidiff training code assumes data have already been prepared as `.h5ad`.
Its loader reads `adata.X` as the expression matrix and reads `adata.obs["Group"]`
as the condition label. The reproducibility notebooks perform the missing work
upstream:

- read raw or preprocessed single-cell data;
- keep the conditions relevant to the experiment;
- map textual conditions to numeric groups;
- run Scanpy preprocessing and visualization checks such as PCA, neighbors, and
  UMAP;
- choose highly variable genes, sometimes forcing marker genes to remain;
- save train/test `.h5ad` files for the training script.

PerturbDiff differs in the label contract: instead of `Group`, it expects a
target perturbation label that is mapped to `perturbation_ids`, with id `0`
reserved for control/non-targeting cells.

## Recommended PerturbDiff Steps

1. Read the raw count matrix and metadata.

   For 10x-style data, load `matrix.mtx(.gz)`, `barcodes.tsv(.gz)`, and
   `genes.tsv(.gz)` or `features.tsv(.gz)` using Scanpy. Join metadata by cell
   barcode.

2. Clean perturbation labels.

   Keep the original guide label, then derive a target-level label:

   - control/non-targeting guides become `control`;
   - guide labels such as `TP53_sg1` or `TP53_...` become `TP53`;
   - if guide-level modeling is desired, keep the raw guide label instead.

3. Apply cell and gene QC.

   Suggested defaults for an initial run:

   - keep metadata rows marked as good coverage when that column exists;
   - remove multiplets when a `number_of_cells` column exists and is greater
     than `1`;
   - filter cells with too few counts or genes;
   - filter cells with high mitochondrial fraction;
   - filter genes expressed in too few cells;
   - remove mitochondrial and ribosomal genes from the model input.

   Exact thresholds are dataset-dependent. The notebook keeps them as editable
   constants.

4. Normalize and transform.

   Preserve raw counts in `adata.layers["counts"]`, then run total-count
   normalization and `log1p`.

5. Select genes.

   Select highly variable genes. Scanpy can do this directly, but the notebook
   uses a lightweight mean/variance dispersion score to avoid requiring Scanpy
   at execution time. For perturbation modeling, also keep target genes that
   are present in `adata.var_names`; otherwise the model may be trained without
   the expression feature of the perturbed gene itself.

6. Save the prepared `.h5ad`.

   The prepared object should contain:

   - `adata.X`: normalized/log-transformed expression over selected genes;
   - `adata.obs["perturbation_label"]`: cleaned labels where controls are
     named `control`;
   - original metadata columns needed for inspection.

7. Train from the prepared `.h5ad`.

   Use `load_h5ad_perturbation_dataset(..., perturbation_col="perturbation_label",
   target_strategy="raw")`. The loader maps `control` to id `0` and target
   labels to ids `1..K`.

## Notes For GSE132080

The included notebook targets `datasets/GSE132080`, whose metadata contains
`cell_barcode`, `guide_identity`, `good_coverage`, and `number_of_cells`.
Controls can be identified from labels containing `neg_ctrl` or
`non-targeting`. Target labels are derived from the first underscore-delimited
token for non-control guides.

The notebook intentionally avoids importing Scanpy because some Windows Python
environments can be slow to import the full Scanpy stack. It still follows the
same AnnData-based preprocessing contract.

The default notebook settings are intentionally conservative:

- process the full dataset only up to the saved `.h5ad` step;
- train on at most `MAX_TRAIN_CELLS` cells for the first smoke run;
- use a small model and one epoch by default.

After the smoke run succeeds, increase `N_TOP_GENES`, `MAX_TRAIN_CELLS`,
network widths, diffusion steps, and epochs.
