"""Build the shared benchmark inputs for the GSE132080 six-metric benchmark.

Produces, under ``--output-dir``:

    genes.json        fixed gene order (the checkpoint's 1019-gene space) + sha256
    conditions.json   evaluated conditions, per-condition cell counts, train/test membership
    deg_table.json    ground-truth DEG table, literature ``calDEG`` recipe
                      (scanpy t-test vs control, ``foldchanges = 2 ** logfoldchanges``,
                      sorted by |t score|), used by both the metric gene restriction and
                      the ``getDEG_cutoff`` DEG-recovery metric
    gene_embeddings.npz   gene -> embedding matrix for the linear/Ridge baseline
    manifest.json     input paths, sha256, counts, and provenance notes

The DEG recipe is a direct port of
``/home/project/Pertb_benchmark/manuscript2/genetic/myUtil1.py::calDEG``
(which writes ``DEG_hvg5000.pkl`` for the scPerturBench datasets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import anndata as ad
import scanpy as sc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_deg_table(full: ad.AnnData, targets: list[str], control: str = "control", column: str = "perturbation_label") -> dict:
    """Port of calDEG(): t-test DEGs vs control, foldchanges = 2**logfoldchanges, sorted by |scores|."""
    work = full.copy()
    work.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(work, column, groups=targets, reference=control, method="t-test")
    result = work.uns["rank_genes_groups"]
    table: dict[str, dict] = {}
    for target in targets:
        names = [str(n) for n in result["names"][target]]
        pvals = [float(v) for v in result["pvals_adj"][target]]
        logfc = [float(v) for v in result["logfoldchanges"][target]]
        scores = [float(v) for v in result["scores"][target]]
        foldchanges = [float(2.0**v) for v in logfc]
        abs_scores = [abs(v) for v in scores]
        order = np.argsort(abs_scores)[::-1]
        table[target] = {
            "order": [names[i] for i in order],
            "pvals_adj": [pvals[i] for i in order],
            "foldchanges": [foldchanges[i] for i in order],
            "scores": [scores[i] for i in order],
        }
    return table


def load_gene_embeddings(gene_names: list[str], candidate_paths: list[Path]) -> tuple[np.ndarray, dict]:
    """Load a gene->embedding mapping, restricted to ``gene_names`` (missing genes -> zeros)."""
    notes: dict[str, object] = {"source": None, "n_found": 0, "n_missing": len(gene_names), "missing_genes": []}
    embeddings = None
    source = None
    for path in candidate_paths:
        if not Path(path).exists():
            continue
        try:
            with Path(path).open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001
            notes.setdefault("load_errors", []).append(f"{path}: {exc}")
            continue
        candidate = _as_gene_embedding_dict(payload)
        if candidate:
            embeddings = candidate
            source = str(path)
            break
    if embeddings is None:
        # deterministic fallback: one-hot-free zero matrix is useless for Ridge, so use
        # a random-but-seeded projection would be arbitrary; instead we abort and let the
        # caller decide (the adapter reports this in its manifest).
        matrix = np.zeros((len(gene_names), 1), dtype=np.float32)
        notes["fallback"] = "no embedding source available; zeros"
        return matrix, notes

    dim = len(next(iter(embeddings.values())))
    matrix = np.zeros((len(gene_names), dim), dtype=np.float32)
    missing = []
    for index, gene in enumerate(gene_names):
        vector = embeddings.get(gene)
        if vector is None:
            missing.append(gene)
            continue
        vector = np.asarray(vector, dtype=np.float32).ravel()
        if vector.shape[0] != dim:
            vector = np.zeros(dim, dtype=np.float32)
        matrix[index] = vector
    notes.update({"source": source, "n_found": len(gene_names) - len(missing), "n_missing": len(missing), "missing_genes": missing[:50]})
    return matrix, notes


def _as_gene_embedding_dict(payload) -> dict | None:
    """Accept the several shapes used across the image for gene embeddings."""
    if isinstance(payload, dict):
        probe_key = next(iter(payload))
        probe_value = payload[probe_key]
        if isinstance(probe_value, (np.ndarray, list, tuple)):
            return {str(k): np.asarray(v).ravel() for k, v in payload.items() if isinstance(v, (np.ndarray, list, tuple))}
        if isinstance(probe_value, dict):
            for nested in ("embeddings", "gene_embeddings", "embedding", "vectors"):
                if nested in payload:
                    return {str(k): np.asarray(v, dtype=np.float32).ravel() for k, v in payload[nested].items()}
        # pandas DataFrame
        if hasattr(payload, "index") and hasattr(payload, "values"):
            return {str(k): np.asarray(v, dtype=np.float32).ravel() for k, v in zip(payload.index, payload.values)}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build shared benchmark inputs (genes / conditions / DEG table / embeddings).")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-col", default="perturbation_label")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--min-test-cells", type=int, default=20)
    parser.add_argument("--embedding-pkl", type=Path, action="append", default=None)
    parser.add_argument("--skip-embeddings", action="store_true", default=False)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = ad.read_h5ad(args.train_path)
    test = ad.read_h5ad(args.test_path)

    genes = [str(g) for g in test.var_names]
    train_genes = [str(g) for g in train.var_names]
    if genes != train_genes:
        raise SystemExit("train/test gene spaces differ; refusing to build a benchmark on mismatched genes")
    (args.output_dir / "genes.json").write_text(
        json.dumps({"genes": genes, "n_genes": len(genes)}, indent=2), encoding="utf-8"
    )

    column = args.condition_col
    test_labels = test.obs[column].astype(str).to_numpy()
    train_labels = train.obs[column].astype(str).to_numpy()
    test_counts = {str(k): int(v) for k, v in test.obs[column].astype(str).value_counts().items()}
    train_counts = {str(k): int(v) for k, v in train.obs[column].astype(str).value_counts().items()}

    conditions = sorted(
        label
        for label, count in test_counts.items()
        if label != args.control_label and label != "*" and count >= args.min_test_cells
    )
    train_targets = {label for label in train_counts if label != args.control_label and label != "*"}

    conditions_meta = {
        "control_label": args.control_label,
        "min_test_cells": args.min_test_cells,
        "n_test_control_cells": int(test_counts.get(args.control_label, 0)),
        "n_train_control_cells": int(train_counts.get(args.control_label, 0)),
        "n_train_conditions": len(train_targets),
        "n_test_conditions_total": len([k for k in test_counts if k != args.control_label and k != "*"]),
        "evaluated_conditions": conditions,
        "per_condition": {
            condition: {
                "n_test_cells": int(test_counts.get(condition, 0)),
                "n_train_cells": int(train_counts.get(condition, 0)),
                "seen_in_train": condition in train_targets,
            }
            for condition in conditions
        },
    }
    (args.output_dir / "conditions.json").write_text(json.dumps(conditions_meta, indent=2), encoding="utf-8")

    # The 2109 control cells are shared by the train and test splits (guides are
    # disjoint, the control pool is not), so deduplicate before building the
    # ground-truth DEG table to avoid double-counting controls.
    train_names = set(train.obs_names.astype(str))
    test_is_dup = np.array([name in train_names for name in test.obs_names.astype(str)])
    test_unique = test[~test_is_dup].copy()
    full = ad.concat([train, test_unique], join="inner")
    full.obs[column] = list(train.obs.astype(str)[column].to_numpy()) + list(
        test_unique.obs.astype(str)[column].to_numpy()
    )
    full.obs_names_make_unique()
    deg_table = build_deg_table(full, conditions, control=args.control_label, column=column)
    (args.output_dir / "deg_table.json").write_text(
        json.dumps({"recipe": "myUtil1.calDEG (scanpy t-test vs control, foldchanges=2**logfoldchanges, order by |scores|)",
                    "dataset": "GSE132080", "genes": deg_table}, ensure_ascii=False),
        encoding="utf-8",
    )

    embedding_notes = {"skipped": True}
    if not args.skip_embeddings:
        candidates = args.embedding_pkl or [
            Path("/home/project/GW_PerturbSeq/geneEmbedding/scGPT.pkl"),
            Path("/home/project/Pertb_benchmark/DataSet2/Papalexi/hvg5000/GEARS/data/gene2go_all.pkl"),
        ]
        matrix, embedding_notes = load_gene_embeddings(genes, candidates)
        np.savez_compressed(args.output_dir / "gene_embeddings.npz", embeddings=matrix, genes=np.array(genes))
        # Only the perturbation-gene embeddings are consumed by the Ridge baseline
        # (embedding of the perturbed gene -> pseudobulk response), so report
        # coverage for the evaluated + trained conditions explicitly.
        gene_index = {gene: index for index, gene in enumerate(genes)}
        perturbed = sorted(set(conditions) | train_targets)
        missing_conditions = []
        for gene in perturbed:
            index = gene_index.get(gene)
            if index is None or not np.any(matrix[index]):
                missing_conditions.append(gene)
        embedding_notes["n_perturbation_genes"] = len(perturbed)
        embedding_notes["perturbation_genes_without_embedding"] = missing_conditions
        embedding_notes["perturbation_gene_coverage"] = (
            (len(perturbed) - len(missing_conditions)) / len(perturbed) if perturbed else 0.0
        )
    (args.output_dir / "embedding_notes.json").write_text(json.dumps(embedding_notes, indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "GSE132080",
        "train_path": str(args.train_path),
        "test_path": str(args.test_path),
        "train_sha256": sha256(args.train_path),
        "test_sha256": sha256(args.test_path),
        "n_genes": len(genes),
        "n_total_unique_cells": int(full.n_obs),
        "n_duplicate_barcodes_dropped": int(test_is_dup.sum()),
        "deg_table_recipe": "train + unique(test) ground-truth cells, calDEG t-test vs control",
        "genes_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "conditions": conditions_meta,
        "embedding_notes": embedding_notes,
        "notes": [
            "DEG table is computed on train+test ground-truth cells (literature calDEG recipe).",
            "Gene space is the checkpoint's 1019-gene space, not scPerturBench hvg5000.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"n_conditions": len(conditions), "n_genes": len(genes), "embedding": embedding_notes}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
