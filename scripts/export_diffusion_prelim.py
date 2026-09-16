"""Export preliminary diffusion-model results at sgRNA level.

For every targeting sgRNA in a PerturbDiff run this script exports:

* sgRNA protospacer sequence, target gene and the 23 nt genomic context
  (``genome input``) that the paper's activity model consumes;
* the model latent values:
  - ``z_sem`` of the control cell pool ("Z-sem" of the basal reference),
  - ``z_sem`` of each sgRNA's observed mean expression profile,
  - the counterfactual latent ``z_cf = z_sem(control) + delta_g``,
  - the learned sgRNA perturbation vector ``delta_g`` itself;
* observed mean gene expression per sgRNA, plus the control mean profile;
* optional DDIM counterfactual predictions from the same control pool.

Outputs (under ``--output-dir``) are documented in ``--help``/README of the run.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.config import ModelConfig  # noqa: E402
from perturbdiff.data import read_metadata_table  # noqa: E402
from perturbdiff.guide_features import (  # noqa: E402
    build_notebook_cnn_feature_tensor,
    find_record_for_label,
    load_guide_feature_records,
)
from perturbdiff.model import PerturbDiffusionModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--guide-col", default="perturbation_label", help="obs column holding the trained perturbation id label.")
    parser.add_argument("--guide-identity-col", default="guide_identity", help="obs column holding the raw sgRNA identity used for sequence lookup.")
    parser.add_argument(
        "--control-col",
        default=None,
        help="Optional obs column used only to identify control cells (defaults to --guide-col).",
    )
    parser.add_argument("--control-label", default="control")
    parser.add_argument(
        "--granularity",
        choices=["auto", "guide", "condition"],
        default="auto",
        help=(
            "guide: one row per raw sgRNA identity (model delta is shared inside a condition); "
            "condition: one row per trained perturbation label (model delta differs per row)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ml-input-table", type=Path, default=None, help="Table S8 machine learning input (carries the genome target window and mismatch position).")
    parser.add_argument("--library-table", type=Path, action="append", default=[], help="Library sequence table(s).")
    parser.add_argument("--guide-activity-cnn-table", type=Path, default=None, help="Table S8 sequence features required by predicted-activity checkpoints.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--top-genes", type=int, default=15)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> tuple[PerturbDiffusionModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    defaults = ModelConfig(n_genes=config.n_genes, n_perturbations=config.n_perturbations)
    missing = {
        field.name: getattr(defaults, field.name)
        for field in dataclasses.fields(ModelConfig)
        if not hasattr(config, field.name)
    }
    if missing:
        config = dataclasses.replace(config, **missing)
    if getattr(config.guide_encoder, "enabled", False):
        has_explicit_activity = any(
            key.startswith("activity_condition.")
            for key in checkpoint["model_state_dict"]
        )
        config = dataclasses.replace(
            config,
            guide_encoder=dataclasses.replace(
                config.guide_encoder,
                explicit_condition=has_explicit_activity,
                activity_scales_delta=not has_explicit_activity,
            ),
        )
    model = PerturbDiffusionModel(config)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    blocking = [k for k in missing if "signature_gene" not in k and "control_expression" not in k]
    if blocking:
        raise RuntimeError(f"unexpected missing state keys: {blocking}")
    if unexpected:
        raise RuntimeError(f"unexpected state keys: {unexpected}")
    model.to(device).eval()
    return model, checkpoint


def canonical(label: str) -> list[str]:
    text = str(label).strip()
    keys = {text, text.upper()}
    return sorted(keys)


def load_sequence_records(ml_input_table: Path | None, library_tables: list[Path]) -> dict[str, dict]:
    records: dict[str, dict] = {}

    def add(label: str, gene: str, payload: dict) -> None:
        if not label:
            return
        entry = records.setdefault(label, {"label": label})
        for key, value in payload.items():
            if value not in (None, ""):
                entry[key] = value
        for key in canonical(label):
            records.setdefault(key, entry)
        if gene:
            for key in canonical(f"{gene}_{label}"):
                records.setdefault(key, entry)

    for table in library_tables:
        for row in read_metadata_table(table):
            label = str(next(iter(row.values()), "")).strip()
            add(
                label,
                str(row.get("gene", "")).strip(),
                {
                    "gene": str(row.get("gene", "")).strip(),
                    "designed_sequence": str(row.get("sequence", "")).strip().upper().replace("U", "T"),
                    "perfect_match_sgrna": str(row.get("perfect match sgRNA", "")).strip() or label,
                    "mismatch_position": row.get("mismatch 1 position", ""),
                },
            )
    if ml_input_table is not None:
        for row in read_metadata_table(ml_input_table):
            label = str(next(iter(row.values()), "")).strip()
            add(
                label,
                str(row.get("gene", "")).strip(),
                {
                    "gene": str(row.get("gene", "")).strip(),
                    "designed_sequence": str(row.get("sgRNA sequence", "")).strip().upper().replace("U", "T"),
                    "perfect_match_sgrna": str(row.get("perfect match sgRNA", "")).strip() or label,
                    "genome_input": str(row.get("genome input", "")).strip().upper().replace("U", "T"),
                    "sgrna_input": str(row.get("sgRNA input", "")).strip().upper().replace("U", "T"),
                    "paper_mean_relative_gamma": row.get("mean relative gamma", ""),
                    "mismatch_position": row.get("mismatch position", ""),
                },
            )
    # Attach each guide's perfect match sequence, which is the target reference.
    gene_of = {id(value): str(value.get("gene", "")) for value in records.values()}
    for entry in {id(value): value for value in records.values()}.values():
        perfect_label = str(entry.get("perfect_match_sgrna", ""))
        if not perfect_label:
            continue
        gene = gene_of.get(id(entry), "")
        candidates = [perfect_label, f"{perfect_label}_00"]
        if gene:
            candidates.extend([f"{gene}_{perfect_label}", f"{gene}_{perfect_label}_00"])
        for candidate in candidates:
            found = None
            for key in canonical(candidate):
                found = records.get(key)
                if found is not None and found.get("designed_sequence"):
                    break
            if found is not None and found.get("designed_sequence"):
                entry["perfect_match_sequence"] = found["designed_sequence"]
                break
    return records


def resolve(label: str, records: dict[str, dict]) -> dict | None:
    for key in canonical(label):
        if key in records:
            return records[key]
    return None


def parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if np.isfinite(out) else None


def target_window(record: dict) -> tuple[str, str, int]:
    """Return (target window, protospacer part, index of the protospacer inside the window).

    The paper's ``genome input`` is the 26 nt genomic window used by the sgRNA
    activity model; the protospacer sits inside it. The window and the sgRNA
    input differ by a fixed 5' offset that is resolved by prefix matching.
    """
    genome_input = str(record.get("genome_input", "")).upper()
    sgrna_input = str(record.get("sgrna_input", "")).upper()
    if not genome_input:
        return "", "", -1
    offset = -1
    for length in range(20, 7, -1):
        if not sgrna_input:
            break
        offset = sgrna_input.find(genome_input[:length])
        if offset >= 0:
            break
    if offset < 0:
        designed = str(record.get("sequence", "")).upper()
        offset = genome_input.find(designed[:12]) if designed else -1
    if offset < 0:
        offset = 2
    return genome_input, genome_input[offset : offset + 20], offset


def mismatch_columns(record: dict) -> dict[str, object]:
    """Compare each sgRNA sequence against the target segment it was designed to hit.

    The library rows hold the *variant* sgRNA in their ``sequence`` column, so the
    comparison reference is resolved in this order:

    1. the genomic window from the paper's Table S8, when the guide has one
       (the window is shared by every variant derived from the same perfect match);
    2. otherwise the perfect match sgRNA sequence.

    ``distance_to_pam`` is counted from the PAM-proximal end: the base next to the
    PAM has distance 0, the distal base of a 20 nt protospacer has distance 19.
    """
    genome_input, protospacer, offset = target_window(record)
    perfect_match_label = str(record.get("perfect_match_sgrna", "")).upper()
    perfect_match_sequence = str(record.get("perfect_match_sequence", "")).upper()
    variant = str(record.get("designed_sequence", "")).upper()
    # The library's own perfect match guide is the primary reference; the genomic
    # window protospacer only supplies the PAM-side context.
    target_segment = perfect_match_sequence[:20] or protospacer or variant[:20]

    columns: dict[str, object] = {
        "target_segment_20nt": target_segment,
        "sgrna_sequence": variant,
        "sgrna_length": len(variant),
        "target_window_26nt": genome_input,
        "target_window_protospacer": protospacer,
        "target_window_target_row": "",
        "target_window_sgrna_row": "",
        "alignment_target_vs_sgrna": "",
        "n_mismatches_vs_target": "",
        "mismatch_positions_1based": "",
        "mismatch_positions_0based": "",
        "mismatch_distance_to_pam": "",
        "mismatch_label_from_library": "",
        "mismatch_distance_to_pam_from_library": "",
        "library_designed_sgrna": variant,
        "alignment_library_vs_target": "",
        "n_mismatches_library_vs_target": "",
        "paper_mean_relative_gamma": record.get("paper_mean_relative_gamma", ""),
        "perfect_match_sgrna": perfect_match_label,
    }
    if genome_input and protospacer and perfect_match_sequence:
        start = genome_input.find(protospacer)
        if start >= 0:
            columns["target_window_target_row"] = (
                "." * start + perfect_match_sequence[:20] + "." * max(len(genome_input) - start - 20, 0)
            )
            if variant:
                columns["target_window_sgrna_row"] = (
                    "." * start + variant[:20] + "." * max(len(genome_input) - start - 20, 0)
                )
    if not target_segment or not variant or len(target_segment) != len(variant):
        return columns
    columns["alignment_target_vs_sgrna"] = "".join(
        "|" if left == right else "X" for left, right in zip(target_segment, variant)
    )
    positions = [index for index, (left, right) in enumerate(zip(target_segment, variant)) if left != right]
    columns["n_mismatches_vs_target"] = len(positions)
    columns["mismatch_positions_1based"] = ";".join(str(index + 1) for index in positions)
    columns["mismatch_positions_0based"] = ";".join(str(index) for index in positions)
    columns["mismatch_distance_to_pam"] = ";".join(str(len(target_segment) - 1 - index) for index in positions)
    library_position = parse_float(record.get("mismatch_position"))
    if library_position is not None:
        columns["mismatch_label_from_library"] = int(library_position)
        columns["mismatch_distance_to_pam_from_library"] = abs(int(library_position)) - 1
    if protospacer and len(protospacer) == len(perfect_match_sequence[:20]):
        other = [
            index
            for index, (left, right) in enumerate(zip(protospacer, perfect_match_sequence[:20]))
            if left != right
        ]
        columns["alignment_library_vs_target"] = "".join(
            "|" if left == right else "X"
            for left, right in zip(protospacer, perfect_match_sequence[:20])
        )
        columns["n_mismatches_library_vs_target"] = len(other)
    return columns


def main() -> None:
    args = parse_args()
    import anndata as ad

    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, args.device)
    device = torch.device(args.device)
    mapping: dict[str, int] = checkpoint["perturbation_mapping"]
    embedding = model.perturbation.embedding.weight.detach().cpu().numpy()
    latent_dim = embedding.shape[1]

    records = load_sequence_records(args.ml_input_table, args.library_table)
    activity_records = (
        load_guide_feature_records([args.guide_activity_cnn_table])
        if args.guide_activity_cnn_table is not None
        else None
    )

    adata = ad.read_h5ad(args.data_path)
    guide_col = args.guide_col if args.guide_col in adata.obs else "perturbation_label"
    labels = adata.obs[guide_col].astype(str).to_numpy()
    identity_col = (
        args.guide_identity_col
        if args.guide_identity_col and args.guide_identity_col in adata.obs
        else guide_col
    )
    identities = adata.obs[identity_col].astype(str).to_numpy()
    control_col = args.control_col or guide_col
    if control_col not in adata.obs:
        raise KeyError(f"control column {control_col!r} not found in adata.obs")
    control_labels = adata.obs[control_col].astype(str).to_numpy()
    gene_names = [str(g) for g in adata.var_names]
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.asarray(X, dtype=np.float32)

    control_mask = control_labels == args.control_label
    if not control_mask.any():
        raise ValueError(f"no control cells labelled {args.control_label!r}")
    control_cells = X[control_mask]
    control_mean = control_cells.mean(axis=0)

    def encode(batch: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            for start in range(0, batch.shape[0], 256):
                tensor = torch.from_numpy(batch[start : start + 256]).to(device)
                chunks.append(model.encoder(tensor).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    control_latents = encode(control_cells)
    z_sem_control = control_latents.mean(axis=0)
    z_sem_control_mean_expr = encode(control_mean[None, :])[0]

    granularity = args.granularity
    if granularity == "auto":
        distinct = {value for value in identities.tolist() if value not in {"*", args.control_label}}
        granularity = "guide" if len(distinct) > len({label for label in labels.tolist()} - {"*", args.control_label}) else "condition"

    units: list[dict] = []
    if granularity == "guide":
        for identity in sorted({value for value in identities.tolist() if value not in {"*", args.control_label}}):
            mask = identities == identity
            n_cells = int(mask.sum())
            condition = labels[mask][0] if n_cells else identity.split("_", 1)[0]
            units.append({"identity": identity, "condition": str(condition), "mask": mask, "n_cells": n_cells})
    else:
        for label in sorted(set(labels.tolist())):
            if label in {args.control_label, "*"} or label not in mapping:
                continue
            mask = labels == label
            n_cells = int(mask.sum())
            candidates = [value for value in identities[mask].tolist() if value not in {"*", args.control_label}]
            identity = max(set(candidates), key=candidates.count) if candidates else label
            units.append({"identity": identity, "condition": label, "mask": mask, "n_cells": n_cells})

    rows: list[dict] = []
    for unit in units:
        mask = unit["mask"]
        label = unit["condition"]
        identity = unit["identity"]
        n_cells = unit["n_cells"]
        if label not in mapping:
            continue
        perturb_id = int(mapping[label])
        observed = X[mask].mean(axis=0) if n_cells else np.full(len(gene_names), np.nan, dtype=np.float32)
        delta = embedding[perturb_id] if perturb_id < embedding.shape[0] else np.full(latent_dim, np.nan)
        target = str(label).split("_", 1)[0]
        record = resolve(identity, records) or resolve(label, records) or {}
        if not record.get("gene"):
            record = {**record, "gene": target}
        z_sem_observed = encode(observed[None, :])[0] if n_cells else np.full(latent_dim, np.nan)
        z_cf = z_sem_control_mean_expr + delta
        guide_ids = {value for value in identities[mask].tolist()} if n_cells else {identity}
        mismatch = mismatch_columns(record)
        guide_features = None
        strength = scaled_strength = activity_embedding_norm = np.nan
        if model.guide_doser is not None:
            if model.config.guide_encoder.activity_source != "predicted":
                raise ValueError("This exporter currently supports predicted activity checkpoints only")
            if activity_records is None:
                raise ValueError("Predicted activity checkpoint requires --guide-activity-cnn-table")
            activity_record = find_record_for_label(identity, activity_records)
            if activity_record is None:
                feature_array = np.zeros((1, model.config.guide_encoder.feature_dim), dtype=np.float32)
            else:
                feature_array = build_notebook_cnn_feature_tensor(
                    activity_record,
                    sequence_length=model.config.guide_encoder.sequence_length,
                ).reshape(1, -1)
            guide_features = torch.from_numpy(feature_array).to(device)
            with torch.no_grad():
                strength_tensor = model.guide_strength(guide_features)
                target_tensor = torch.tensor([perturb_id], dtype=torch.long, device=device)
                scaled_tensor = model.guide_doser(strength_tensor, target_tensor)
                activity_tensor = model._build_activity_embedding(scaled_tensor, target_tensor)
            strength = float(strength_tensor.item())
            scaled_strength = float(scaled_tensor.item())
            activity_embedding_norm = (
                np.nan if activity_tensor is None else float(torch.linalg.vector_norm(activity_tensor[0]).item())
            )
        rows.append(
            {
                "perturbation_id": perturb_id,
                "guide_label": label,
                "guide_identity": identity,
                "target_gene": str(record.get("gene") or target),
                "n_cells": n_cells,
                "n_guides_in_condition": len(guide_ids),
                "delta": delta,
                "observed_mean": observed,
                "z_sem_observed_mean": z_sem_observed,
                "z_cf_from_control_mean": z_cf,
                "guide_features": guide_features,
                "activity_raw_strength": strength,
                "activity_scaled_strength": scaled_strength,
                "activity_embedding_l2_norm": activity_embedding_norm,
                **mismatch,
            }
        )
    verified = [
        row
        for row in rows
        if row["mismatch_label_from_library"] != ""
        and int(row["n_mismatches_vs_target"] or 0) == 1
        and row["mismatch_distance_to_pam"] != ""
        and int(row["mismatch_distance_to_pam"]) == int(row["mismatch_distance_to_pam_from_library"])
    ]

    order = sorted(range(len(rows)), key=lambda i: -float(np.linalg.norm(rows[i]["delta"])))
    prefix = "sgrna" if granularity == "guide" else "condition"

    # Control-pool Z-sem reference (raw latents only).
    with (args.output_dir / "z_sem_control.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", *[f"zsem_{i:03d}" for i in range(latent_dim)]])
        writer.writerow(["control_pool_mean_latent", *z_sem_control.astype(float).tolist()])
        writer.writerow(["control_mean_expression_latent", *z_sem_control_mean_expr.astype(float).tolist()])
        for index in order:
            row = rows[index]
            writer.writerow(
                [f"sgrna_observed_mean_latent:{row['guide_identity']}", *row["z_sem_observed_mean"].astype(float).tolist()]
            )

    with (args.output_dir / "z_sem_control_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"zsem_{i:03d}" for i in range(latent_dim)])
        for vector in control_latents:
            writer.writerow(vector.astype(float).tolist())

    with (args.output_dir / f"{prefix}_delta_vectors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "perturbation_id",
                "guide_label",
                "guide_identity",
                "target_gene",
                *[f"delta_{i:03d}" for i in range(latent_dim)],
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["perturbation_id"],
                    row["guide_label"],
                    row["guide_identity"],
                    row["target_gene"],
                    *row["delta"].astype(float).tolist(),
                ]
            )

    with (args.output_dir / "z_sem_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "guide_identity", "target_gene", "n_cells", "z_sem_source", *[f"zsem_{i:03d}" for i in range(latent_dim)]])
        writer.writerow(["control", "neg_ctrl", "control", int(control_mask.sum()), "pool_mean", *z_sem_control.astype(float).tolist()])
        for row in rows:
            writer.writerow(
                [row["guide_label"], row["guide_identity"], row["target_gene"], row["n_cells"], "observed_mean_expression", *row["z_sem_observed_mean"].astype(float).tolist()]
            )
        for row in rows:
            writer.writerow(
                [row["guide_label"], row["guide_identity"], row["target_gene"], row["n_cells"], "counterfactual_z_cf", *row["z_cf_from_control_mean"].astype(float).tolist()]
            )

    # Predictions.
    predictions: dict[str, np.ndarray] = {}
    if not args.skip_prediction:
        control_mean_tensor = torch.from_numpy(control_mean).unsqueeze(0).to(device)
        with torch.no_grad():
            for row in rows:
                target_ids = torch.tensor([row["perturbation_id"]], device=device, dtype=torch.long)
                feature_tensor = row["guide_features"]
                if feature_tensor is not None:
                    feature_tensor = feature_tensor.expand(control_mean_tensor.shape[0], -1)
                sample = model.sample_counterfactual(
                    control_mean_tensor,
                    target_ids,
                    guide_features=feature_tensor,
                    steps=args.ddim_steps,
                )
                predictions[row["guide_identity"]] = sample[0].cpu().numpy()

    # Per-sgRNA summary with top differential genes.
    summary_path = args.output_dir / f"{prefix}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "perturbation_id",
                "guide_label",
                "guide_identity",
                "target_gene",
                "target_segment_20nt",
                "target_window_26nt",
                "target_window_target_row",
                "target_window_sgrna_row",
                "sgrna_sequence",
                "alignment_target_vs_sgrna",
                "n_mismatches_vs_target",
                "mismatch_positions_1based",
                "mismatch_positions_0based",
                "mismatch_distance_to_pam",
                "mismatch_label_from_library",
                "mismatch_distance_to_pam_from_library",
                "paper_mean_relative_gamma",
                "activity_raw_strength",
                "activity_scaled_strength",
                "activity_embedding_l2_norm",
                "n_cells",
                "n_guides_in_condition",
                "top_down_genes_observed",
                "top_up_genes_observed",
                "top_down_genes_predicted",
                "top_up_genes_predicted",
            ]
        )
        for row in rows:
            observed = row["observed_mean"]
            delta_expr = observed - control_mean
            order_down = np.argsort(delta_expr)
            down = [gene_names[i] for i in order_down[: args.top_genes] if delta_expr[i] < 0]
            up = [gene_names[i] for i in np.argsort(-delta_expr)[: args.top_genes] if delta_expr[i] > 0]
            predicted = predictions.get(row["guide_identity"])
            if predicted is not None:
                pred_delta = predicted - control_mean
                pred_down = [gene_names[i] for i in np.argsort(pred_delta)[: args.top_genes] if pred_delta[i] < 0]
                pred_up = [gene_names[i] for i in np.argsort(-pred_delta)[: args.top_genes] if pred_delta[i] > 0]
            else:
                pred_down, pred_up = [], []
            writer.writerow(
                [
                    row["perturbation_id"],
                    row["guide_label"],
                    row["guide_identity"],
                    row["target_gene"],
                    row["target_segment_20nt"],
                    row["target_window_26nt"],
                    row["target_window_target_row"],
                    row["target_window_sgrna_row"],
                    row["sgrna_sequence"],
                    row["alignment_target_vs_sgrna"],
                    row["n_mismatches_vs_target"],
                    row["mismatch_positions_1based"],
                    row["mismatch_positions_0based"],
                    row["mismatch_distance_to_pam"],
                    row["mismatch_label_from_library"],
                    row["mismatch_distance_to_pam_from_library"],
                    row["paper_mean_relative_gamma"],
                    row["activity_raw_strength"],
                    row["activity_scaled_strength"],
                    row["activity_embedding_l2_norm"],
                    row["n_cells"],
                    row["n_guides_in_condition"],
                    ";".join(down),
                    ";".join(up),
                    ";".join(pred_down),
                    ";".join(pred_up),
                ]
            )

    # Focused long-format expression table: top differential genes per sgRNA.
    focus_path = args.output_dir / f"{prefix}_top_gene_expression_long.csv"
    with focus_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["guide_label", "target_gene", "gene", "control_mean", "observed_mean", "predicted_mean", "observed_delta", "predicted_delta", "rank_by_abs_observed_delta"])
        for row in rows:
            observed = row["observed_mean"]
            predicted = predictions.get(row["guide_identity"])
            delta_expr = observed - control_mean
            order_abs = np.argsort(-np.abs(delta_expr))[: args.top_genes]
            for gene_index in order_abs:
                writer.writerow(
                    [
                        row["guide_label"],
                        row["target_gene"],
                        gene_names[gene_index],
                        float(control_mean[gene_index]),
                        float(observed[gene_index]),
                        "" if predicted is None else float(predicted[gene_index]),
                        float(delta_expr[gene_index]),
                        "" if predicted is None else float(predicted[gene_index] - control_mean[gene_index]),
                        int(np.where(order_abs == gene_index)[0][0]) + 1,
                    ]
                )

    # Raw expression values: control baseline, observed sgRNA means, model predictions.
    with (args.output_dir / "expression_control_mean.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", args.control_label])
        for gene_index, gene in enumerate(gene_names):
            writer.writerow([gene, float(control_mean[gene_index])])

    matrix_path = args.output_dir / f"{prefix}_expression_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", *[row["guide_identity"] for row in rows]])
        for gene_index, gene in enumerate(gene_names):
            writer.writerow([gene] + [float(row["observed_mean"][gene_index]) for row in rows])

    if predictions:
        pred_path = args.output_dir / f"{prefix}_predicted_expression_matrix.csv"
        with pred_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["gene", *[row["guide_identity"] for row in rows]])
            for gene_index, gene in enumerate(gene_names):
                writer.writerow(
                    [gene] + [float(predictions[row["guide_identity"]][gene_index]) for row in rows]
                )

    manifest = {
        "checkpoint": str(args.checkpoint),
        "data_path": str(args.data_path),
        "guide_col": guide_col,
        "guide_identity_col": identity_col,
        "control_col": control_col,
        "granularity": granularity,
        "n_control_cells": int(control_mask.sum()),
        "n_genes": len(gene_names),
        "n_rows": len(rows),
        "n_sgrnas": len(rows) if granularity == "guide" else None,
        "n_conditions": len({row["guide_label"] for row in rows}),
        "n_target_genes": len({row["target_gene"] for row in rows}),
        "latent_dim": int(latent_dim),
        "activity_source": (
            model.config.guide_encoder.activity_source if model.guide_doser is not None else None
        ),
        "ddim_steps": None if args.skip_prediction else args.ddim_steps,
        "n_rows_with_target_segment": sum(1 for row in rows if row["target_segment_20nt"]),
        "n_rows_with_mismatch": sum(1 for row in rows if int(row["n_mismatches_vs_target"] or 0) > 0),
        "n_rows_with_library_mismatch_label": sum(1 for row in rows if row["mismatch_label_from_library"] != ""),
        "n_single_mismatch_rows_matching_library_distance": len(verified),
        "outputs": sorted(p.name for p in args.output_dir.iterdir() if p.is_file()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
