"""Strict local wrapper around Arc's official VCC 2026 scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

PINNED_RELEASE = "0.18.0"
EXPECTED_RULE_VERSION = 3
EXPECTED_METRICS = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)
DEFAULT_CONTROL = "non-targeting"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dense(path: Path):
    data = ad.read_h5ad(path)
    if sparse.issparse(data.X):
        matrix = data.X
        values = matrix.data
        totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        matrix = np.asarray(data.X)
        values = matrix.ravel()
        totals = matrix.sum(axis=1)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: expression matrix contains NaN or infinity")
    if np.any(values < 0):
        raise ValueError(f"{path}: raw count values must be non-negative")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{path}: raw count values must be integral")
    if np.any(totals > 1_000_000):
        raise ValueError(f"{path}: cell total exceeds 1,000,000")
    return data


def _identity(data, *, path: Path, pert_col: str, control: str):
    if pert_col not in data.obs:
        raise ValueError(f"{path}: obs column {pert_col!r} is missing")
    if data.obs[pert_col].isna().any():
        raise ValueError(f"{path}: perturbation labels contain missing values")
    labels = data.obs[pert_col].astype(str)
    if control not in set(labels):
        raise ValueError(f"{path}: control label {control!r} is missing")
    return tuple(map(str, data.var_names)), frozenset(labels)


def validate_alignment(pred, real, *, prediction_path: Path, reference_path: Path,
                       pert_col: str, control: str):
    genes, labels = _identity(pred, path=prediction_path, pert_col=pert_col, control=control)
    real_genes, real_labels = _identity(real, path=reference_path, pert_col=pert_col, control=control)
    if genes != real_genes:
        raise ValueError(f"{prediction_path}: ordered gene axis differs from reference")
    if labels != real_labels:
        missing = sorted(real_labels - labels)
        extra = sorted(labels - real_labels)
        raise ValueError(
            f"{prediction_path}: perturbation labels differ; missing={missing}, extra={extra}"
        )


def _check_rule(bundle_path: Path):
    from cell_eval2 import competition
    from cell_eval2.real_bundle import read_real_bundle

    bundle = read_real_bundle(bundle_path)
    payload = competition.competition_payload()
    if payload.get("rule_version") != EXPECTED_RULE_VERSION:
        raise RuntimeError(
            f"pinned scorer has rule_version={payload.get('rule_version')}, "
            f"expected {EXPECTED_RULE_VERSION}"
        )
    if tuple(competition.competition_members()) != EXPECTED_METRICS:
        raise RuntimeError("official scorer metric membership differs from VCC 2026 spec")
    if not bundle.manifest.get("rule_digest"):
        raise ValueError(f"{bundle_path}: reference bundle is diagnostic and has no rule digest")
    if bundle.manifest.get("rule_digest") != competition.competition_digest():
        raise ValueError(f"{bundle_path}: reference bundle rule digest does not match pinned scorer")
    with (Path(bundle_path) / "anchor_meta.json").open(encoding="utf-8") as handle:
        anchor_meta = json.load(handle)
    with (Path(bundle_path) / "baseline_meta.json").open(encoding="utf-8") as handle:
        baseline_meta = json.load(handle)
    if anchor_meta.get("rule_version", EXPECTED_RULE_VERSION) != EXPECTED_RULE_VERSION:
        raise ValueError(f"{bundle_path}: anchor metadata is from a different rule version")
    cfg = competition.EvalConfig.from_yaml(Path(bundle_path) / "config.yaml")
    reasons = competition.is_competition_rule(
        cfg,
        members=competition.competition_members(),
        anchor_meta=anchor_meta,
        estimators={
            metric: bundle.manifest.get("estimators", {}).get(metric)
            for metric in EXPECTED_METRICS
        },
    )
    if reasons:
        raise ValueError(f"{bundle_path}: reference bundle does not match VCC 2026: {reasons}")
    return bundle, payload, competition.competition_digest(), anchor_meta, baseline_meta


def score_one(prediction: Path, bundle_path: Path, reference_path: Path, *, context: str, outdir: Path,
              pert_col: str, control: str):
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import aggregate_metrics_wide, compute_metrics, metric_output_names
    from cell_eval2.score import score_metrics

    pred = _load_dense(prediction)
    bundle, payload, rule_digest, anchor_meta, baseline_meta = _check_rule(bundle_path)
    cfg = EvalConfig.from_yaml(Path(bundle_path) / "config.yaml")

    real_path = reference_path
    real = ad.read_h5ad(real_path)
    validate_alignment(pred, real, prediction_path=prediction, reference_path=real_path,
                       pert_col=pert_col, control=control)
    from cell_eval2.cache import fingerprint_adata
    real_fp = fingerprint_adata(real, pert_col=pert_col, strict=True)
    if baseline_meta.get("source_fingerprint_strict") and \
            real_fp != baseline_meta.get("source_fingerprint"):
        raise ValueError(f"{real_path}: content hash does not match official bundle reference")

    if cfg.pert_col != pert_col or cfg.control != control:
        raise ValueError(
            f"{bundle_path}: CLI labels ({pert_col!r}, {control!r}) differ from bundle config "
            f"({cfg.pert_col!r}, {cfg.control!r})"
        )
    if cfg.rule_version != EXPECTED_RULE_VERSION:
        raise ValueError(f"{bundle_path}: config rule_version is not {EXPECTED_RULE_VERSION}")
    if not cfg.cache_strict:
        raise ValueError(f"{bundle_path}: config cache_strict must be enabled for submission scoring")
    result = compute_metrics(pred, real, config=cfg)
    aggregate = aggregate_metrics_wide(result, metrics=metric_output_names(cfg))
    outdir.mkdir(parents=True, exist_ok=True)
    agg_path = outdir / "user_agg.csv"
    aggregate.write_csv(agg_path)
    meta = build_run_meta(cfg, real, pred)
    from cell_eval2.real_bundle import check_submission
    check_submission(bundle.manifest, meta)
    scored = score_metrics(str(agg_path), real_bundle=str(bundle_path), user_meta=meta)
    rows = scored.to_dicts()
    by_metric = {row["metric"]: row for row in rows}
    actual = tuple(sorted(name for name in by_metric if name != "avg_score"))
    if actual != tuple(sorted(EXPECTED_METRICS)):
        raise RuntimeError(f"official scorer returned unexpected metrics: {actual}")
    score_row = by_metric.get("avg_score")
    if score_row is None:
        raise RuntimeError("official scorer did not return avg_score")
    metrics = {
        name: float(row["from_replicate"])
        for name, row in by_metric.items()
        if name != "avg_score"
    }
    if len(metrics) != len(EXPECTED_METRICS) or set(metrics) != set(EXPECTED_METRICS):
        raise RuntimeError("official scorer output is missing or duplicates a required metric")
    official_table = [
        {key: value for key, value in row.items()}
        for row in rows
    ]
    (outdir / "official_scores.json").write_text(
        json.dumps(official_table, indent=2, default=str), encoding="utf-8")
    return {
        "context": context,
        "metrics": metrics,
        "avg_score": float(score_row["from_replicate"]),
        "rule_version": payload["rule_version"],
        "rule_digest": rule_digest,
        "bundle_id": bundle.manifest.get("real_bundle_id"),
        "bundle_path": str(bundle_path.resolve()),
        "bundle_manifest_sha256": _sha256(Path(bundle_path) / "manifest.json"),
        "prediction_path": str(prediction.resolve()),
        "prediction_sha256": _sha256(prediction),
        "real_path": str(real_path.resolve()),
        "real_sha256": _sha256(real_path),
        "scorer_release": PINNED_RELEASE,
        "scorer_version": meta.get("cell_eval2_version"),
        "resolved_de_backend": meta.get("resolved_de_backend"),
        "resolved_device": meta.get("resolved_device"),
        "official_score_table": official_table,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--bundles", nargs="+", type=Path, required=True)
    parser.add_argument("--references", nargs="+", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--pert-col", default="target")
    parser.add_argument("--control", default=DEFAULT_CONTROL)
    args = parser.parse_args(argv)
    if not (len(args.predictions) == len(args.bundles) == len(args.references) == len(args.contexts)):
        parser.error("predictions, bundles, references, and contexts must have equal lengths")
    if len(set(args.contexts)) != len(args.contexts):
        parser.error("context names must be unique")
    for path in [*args.predictions, *args.bundles, *args.references]:
        if not path.exists():
            parser.error(f"input path does not exist: {path}")
    for bundle in args.bundles:
        if not bundle.is_dir() or not (bundle / "manifest.json").is_file():
            parser.error(f"reference bundle must be a directory containing manifest.json: {bundle}")
    if any(not path.is_file() for path in [*args.predictions, *args.references]):
        parser.error("prediction and reference inputs must be files")

    rows = [score_one(pred, bundle, reference, context=name, outdir=args.outdir / name,
                      pert_col=args.pert_col, control=args.control)
            for pred, bundle, reference, name in zip(args.predictions, args.bundles,
                                                      args.references, args.contexts)]
    args.outdir.mkdir(parents=True, exist_ok=True)
    contexts_csv = args.outdir / "contexts.csv"
    flat = []
    for row in rows:
        score_by_metric = {
            score_row["metric"]: score_row["from_replicate"]
            for score_row in row["official_score_table"]
        }
        if len(score_by_metric) != len(EXPECTED_METRICS) + 1:
            raise RuntimeError(f"{row['context']}: unexpected metric rows in official scorer output")
        flat.append({"context": row["context"],
                     **{name: score_by_metric[name] for name in EXPECTED_METRICS},
                     "avg_score": score_by_metric["avg_score"]})
    pd.DataFrame(flat, columns=["context", *EXPECTED_METRICS, "avg_score"]).to_csv(
        contexts_csv, index=False)
    overall = float(np.mean([row["avg_score"] for row in rows]))
    (args.outdir / "overall.json").write_text(
        json.dumps({"overall_avg_score": overall,
                    "context_count": len(rows),
                    "contexts": {row["context"]: row["avg_score"] for row in rows}},
                   indent=2), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorer": "ArcInstitute/cell-eval2",
        "scorer_release": PINNED_RELEASE,
        "scorer_version": rows[0]["scorer_version"],
        "rule_version": EXPECTED_RULE_VERSION,
        "rule_digest": rows[0]["rule_digest"],
        "metric_names": list(EXPECTED_METRICS),
        "aggregate_policy": "equal mean across metrics within context, then contexts",
        "pert_col": args.pert_col,
        "control": args.control,
        "contexts": rows,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"overall_avg_score": overall, "contexts_csv": str(contexts_csv),
                      "manifest": str(args.outdir / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
