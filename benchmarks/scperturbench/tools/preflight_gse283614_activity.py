"""Preflight validation for the GSE283614 measured-activity ablation.

Turns every assumption the ablation relies on into a hard assertion, records the
provenance the report must cite, and exits non-zero on any failed gate.  Run it
before spending GPU time.

Two entry points, merged by the caller:

* ``--part data``     fold integrity, activity-axis resolution and degeneracy.
  Needs anndata + scipy only, so it runs in the pertpyV7 env.
* ``--part model``    the actual Diff-AE-Perturb model construction, the
  guide-encoder sizing that the GSE132080 report documented as buggy upstream,
  and an audited-loader checkpoint round trip.  Needs the repo on sys.path, so
  it runs in the diffae-cu128 env.

The split exists because the repository needs Python >= 3.10 with a GPU-capable
torch, while the metric stack needs pertpy; no single interpreter in this image
has both (see reports/diffae_activity_ablation_gse132080.md section 6.3).

Usage::

    tools/preflight_gse283614_activity.py --part data  --inputs-dir ... --output .../preflight_data.json
    tools/preflight_gse283614_activity.py --part model --inputs-dir ... --output .../preflight_model.json
    tools/preflight_gse283614_activity.py --merge a.json b.json --output preflight.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_LABEL = "control"
# The ablation trains with --perturbation-col guide_family, so the perturbation
# vocabulary is the target family, not the guide.
PERTURBATION_COL = "guide_family"
TARGET_OF = {"NANOG": "NANOG", "POU5F1": "OCT4"}
EXPECTED_FAMILIES = {"control", "NANOG", "OCT4"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GSE283614 ablation inputs.")
    parser.add_argument("--part", choices=("data", "model"))
    parser.add_argument("--merge", nargs="+", type=Path, default=None)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--repo", default="/root/Diff-AE-Perturb")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--states", nargs="+", default=["esc"])
    parser.add_argument("--skip-hash", action="store_true",
                        help="Skip the (slow) train/val/test sha256 verification.")
    return parser.parse_args()


class Report:
    """Accumulate named checks; a failed hard gate is fatal at the end."""

    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.failed: list[str] = []

    def record(self, name: str, passed: bool, **detail: object) -> bool:
        self.checks.append({"name": name, "passed": bool(passed), **detail})
        if not passed:
            self.failed.append(name)
        return bool(passed)

    def info(self, name: str, **detail: object) -> None:
        self.checks.append({"name": name, "passed": True, "informational": True, **detail})

    def payload(self, part: str) -> dict:
        return {
            "part": part,
            "n_checks": len(self.checks),
            "n_failed": len(self.failed),
            "failed": self.failed,
            "all_passed": not self.failed,
            "checks": self.checks,
        }


# --------------------------------------------------------------------------
# key resolution: mirror the target code exactly
# --------------------------------------------------------------------------


def canonical_guide_keys(label: str) -> tuple[str, ...]:
    """Copy of ``perturbdiff.data.canonical_guide_keys``."""

    value = str(label).strip()
    if not value:
        return ("",)
    keys = [value]
    parts = value.split("_")
    if len(parts) >= 3:
        keys.append("_".join(parts[1:]))
    if len(parts) >= 2 and parts[0] == parts[1]:
        keys.append("_".join(parts[1:]))
    unique: list[str] = []
    for key in keys:
        if key and key not in unique:
            unique.append(key)
    return tuple(unique)


def guide_keys(label: str) -> tuple[str, ...]:
    """Copy of ``perturbdiff.guide_activity._guide_keys``."""

    value = str(label).strip()
    if not value:
        return ("",)
    keys = [value]
    parts = value.split("_")
    if len(parts) >= 3:
        keys.append("_".join(parts[1:]))
    if len(parts) >= 2 and parts[0] == parts[1]:
        keys.append("_".join(parts[1:]))
    if value.endswith("-all"):
        keys.append(value.removesuffix("-all"))
    if "-all_" in value:
        keys.append(value.replace("-all_", "_"))
    return tuple(dict.fromkeys(key for key in keys if key))


def identified_keys(label: str, gene: str) -> tuple[str, ...]:
    """Keys a control row needs so ``lookup_activity_for_label`` can find it.

    ``canonical_guide_keys`` is verbatim from ``perturbdiff.data`` and performs
    *only* underscore splitting, so a label such as ``SafeHarborchr1_1007``
    yields ``("SafeHarborchr1_1007",)`` - the bare label is always included, so
    the table simply has to carry one row per control guide.  This helper states
    that requirement explicitly instead of leaving it implicit.
    """

    keys = list(canonical_guide_keys(label))
    if gene:
        keys.extend(canonical_guide_keys(f"{gene}_{label}"))
    return tuple(dict.fromkeys(k for k in keys if k))


def load_activity_keys(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Build both lookups the two resolution paths would build from this table.

    Returns ``(data_lookup, guide_lookup)``.  ``load_guide_activity_lookup`` adds
    ``gene_label`` spellings in addition to the bare label, so the two maps are
    not identical for tables that carry a gene column.
    """

    frame = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    if frame.columns[0] != "guide_identity":
        raise KeyError(f"{path} first column must be guide_identity, got {frame.columns[0]!r}")
    data_lookup: dict[str, float] = {}
    guide_lookup: dict[str, float] = {}
    for row in frame.to_dict("records"):
        label = str(row.get("guide_identity", "")).strip()
        raw = row.get("relative_activity", "")
        if not label or raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "":
            continue
        value = float(raw)
        gene = str(row.get("target_gene", "")).strip()
        for key in canonical_guide_keys(label):
            data_lookup[key] = value
        if gene:
            for key in canonical_guide_keys(f"{gene}_{label}"):
                data_lookup[key] = value
        for key in guide_keys(label):
            guide_lookup[key] = value
        if gene:
            for key in guide_keys(f"{gene}_{label}"):
                guide_lookup[key] = value
    return data_lookup, guide_lookup


# --------------------------------------------------------------------------
# part: data
# --------------------------------------------------------------------------


def run_data_part(args: argparse.Namespace, report: Report) -> None:
    import anndata as ad

    inputs = args.inputs_dir
    summary = json.loads((inputs / "build_summary.json").read_text(encoding="utf-8"))
    report.info("build_summary", **{k: summary[k] for k in ("activity_table_sha256", "deg_built")})
    report.record(
        "activity_table_hash_matches_source",
        summary["activity_table_sha256"]
        == sha256(Path(args.repo) / "reproducibility/prepared/gse283614/gse283614_table_s1_activity.tsv"),
        recorded=summary["activity_table_sha256"],
    )
    coverage = summary["coverage"]

    def coverage_for(state: str) -> dict:
        """The builder keys coverage by the upper-case state name."""

        return coverage[state.upper()]

    def coverage_missing(state: str) -> list[str]:
        return list(coverage_for(state)["missing_guides"])

    for state in [s.lower() for s in args.states]:
        conditions = json.loads((inputs / f"conditions_{state}.json").read_text(encoding="utf-8"))
        activity_path = inputs / f"gse283614_activity_{state}.tsv"
        data_lookup, guide_lookup = load_activity_keys(activity_path)
        prefix = f"{state}:"

        evaluated = list(conditions["evaluated_conditions"])
        per_condition = conditions["per_condition"]
        report.record(f"{prefix}evaluated_conditions_nonempty", len(evaluated) > 0, n=len(evaluated))

        # --- activity axis resolution -------------------------------------
        # Two resolution paths exist in the target code:
        #   * attach_direct_activity_features -> lookup_activity_for_label ->
        #     canonical_guide_keys
        #   * build_guide_features (inference) -> canonical_guide_keys as well
        # So canonical_guide_keys is the one that must work for every guide the
        # model conditions on.  guide_keys is checked too, because
        # load_guide_activity_lookup is what actually builds the table.
        resolved, unresolved = {}, []
        for guide in evaluated:
            hit = next((data_lookup[k] for k in canonical_guide_keys(guide) if k in data_lookup), None)
            if hit is None:
                unresolved.append(guide)
            else:
                resolved[guide] = float(hit)
        report.record(
            f"{prefix}every_evaluated_guide_resolves_activity_key",
            not unresolved,
            n_resolved=len(resolved),
            unresolved=unresolved,
            note="including unmeasured guides: the table writes them as an explicit 0.0 so attach_direct_activity_features can resolve every non-control label",
        )
        # The zero-fill must be explicit and enumerable: the preflight refuses a
        # table where an unmeasured guide is simply absent, because then the
        # trainer raises and the run is not reproducible against the published
        # 2026-07-28 GSE283614 comparison.
        fill_missing = [
            g
            for g in coverage_missing(state)
            if not (g in resolved and resolved[g] == 0.0)
        ]
        report.record(
            f"{prefix}unmeasured_guides_present_as_explicit_zero",
            not fill_missing,
            n_missing_in_source=len(coverage_missing(state)),
            not_resolved_as_zero=fill_missing,
        )
        # An unmeasured guide is legitimate only when it is written as an empty
        # field and therefore becomes 0; it must never be silently absent from
        # both the table and the trainer's control handling.
        measured_guides = sorted(g for g in evaluated if per_condition[g]["activity_measured"])
        unmeasured = sorted(set(evaluated) - set(measured_guides))
        report.record(
            f"{prefix}conditions_measured_flags_match_build_summary",
            set(measured_guides) == set(coverage_for(state)["measured_guides"]) & set(evaluated)
            and set(unmeasured) == set(coverage_for(state)["missing_guides"]) & set(evaluated),
            from_conditions=sorted(measured_guides),
            from_build_summary=sorted(set(coverage_for(state)["measured_guides"]) & set(evaluated)),
        )
        report.info(
            f"{prefix}activity_coverage",
            n_measured=len(measured_guides),
            n_unmeasured=len(unmeasured),
            unmeasured_guides=unmeasured,
            unmeasured_guide_types=sorted({per_condition[g]["guide_type"] for g in unmeasured}),
            missing_value_policy=coverage_for(state)["missing_value_policy"],
            note=(
                "unmeasured guides are written as 0.0, so in the with-arm they are numerically "
                "identical to a guide whose screen depletion really was zero; the report must "
                "split the two groups using conditions_<state>.json::activity_measured"
            ),
        )
        report.record(
            f"{prefix}measured_guides_positive_and_finite",
            all(
                np.isfinite(resolved[g]) and resolved[g] >= 0.0
                for g in measured_guides
            ),
            n=len(measured_guides),
        )
        report.record(
            f"{prefix}measured_guides_match_source_table",
            all(
                abs(resolved[g] - per_condition[g]["activity_raw_or_none"]) < 1e-12
                for g in measured_guides
            ),
            n=len(measured_guides),
        )
        values = np.array([resolved[g] for g in measured_guides], dtype=float)
        report.record(
            f"{prefix}activity_axis_is_not_degenerate",
            values.size >= 5 and len(np.unique(values)) >= 5 and float(values.std()) > 0.05,
            n=int(values.size),
            n_unique=int(len(np.unique(values))),
            sd=float(values.std()) if values.size else 0.0,
        )

        # --- fold integrity ----------------------------------------------
        for fold in conditions["folds"]:
            f = f"fold{fold['fold']}"
            test_guides = list(fold["test_guides"])
            train_guides = list(fold["train_guides"])
            report.record(
                f"{prefix}{f}_held_out_guides_disjoint",
                not (set(test_guides) & set(train_guides)),
                overlap=sorted(set(test_guides) & set(train_guides)),
            )
            if not args.skip_hash:
                # Only the test split is hashed by default: it is small, and it
                # is the file both the adapter's control pool and the metric
                # ground truth come from.  The train split is ~300 MB per fold
                # and is re-hashed by the training manifest instead.
                report.record(
                    f"{prefix}{f}_test_sha256_matches",
                    sha256(Path(fold["test_path"])) == fold["test_sha256"],
                    path=fold["test_path"],
                )

            test = ad.read_h5ad(fold["test_path"], backed="r")
            train = ad.read_h5ad(fold["train_path"], backed="r")
            labels = test.obs["perturbation_label"].astype(str)
            identities = test.obs["guide_identity"].astype(str)
            families = set(test.obs[PERTURBATION_COL].astype(str)) | set(
                train.obs[PERTURBATION_COL].astype(str)
            )
            report.record(
                f"{prefix}{f}_perturbation_vocabulary_is_targets",
                families == EXPECTED_FAMILIES,
                observed=sorted(families),
                expected=sorted(EXPECTED_FAMILIES),
            )
            present = sorted(set(labels) - {CONTROL_LABEL})
            report.record(
                f"{prefix}{f}_file_conditions_match_manifest",
                present == sorted(test_guides),
                from_file=present,
                from_manifest=sorted(test_guides),
            )
            report.record(
                f"{prefix}{f}_has_both_targets",
                {"NANOG", "OCT4"}
                <= {
                    TARGET_OF.get(str(g))
                    for g in test.obs.loc[labels != CONTROL_LABEL, "target_gene"].astype(str).unique()
                },
                targets=sorted(
                    set(test.obs.loc[labels != CONTROL_LABEL, "target_gene"].astype(str))
                ),
            )
            mismatch = sorted(
                {
                    (str(i), str(l))
                    for i, l in zip(identities[labels != CONTROL_LABEL], labels[labels != CONTROL_LABEL])
                    if str(i) != str(l)
                }
            )
            report.record(
                f"{prefix}{f}_guide_identity_equals_perturbation_label",
                not mismatch,
                mismatches=mismatch[:5],
                n_mismatch=len(mismatch),
            )
            # Controls: SafeHarbor and NT must both be treated as control by the
            # trainer's CONTROL_PATTERNS, otherwise they would become
            # perturbations and corrupt the vocabulary.
            control_guides = sorted(set(identities[labels == CONTROL_LABEL]))
            report.record(
                f"{prefix}{f}_control_covers_nt_and_safeharbor",
                any(g.startswith("NT_") for g in control_guides)
                and any(g.startswith("SafeHarbor") for g in control_guides),
                control_guides=control_guides[:4],
                n_control_guides=len(control_guides),
            )

        # --- evaluation protocol agree with the metrics run ----------------
        counts = {
            g: int(
                sum(
                    1
                    for fold in conditions["folds"]
                    if g in set(fold["test_guides"])
                )
            )
            for g in evaluated
        }
        report.record(
            f"{prefix}each_guide_held_out_exactly_once",
            all(v == 1 for v in counts.values()),
            violations={g: v for g, v in counts.items() if v != 1},
        )
        report.record(
            f"{prefix}min_test_cells_satisfied_for_evaluated",
            all(int(per_condition[g]["n_test_cells"]) >= int(conditions["min_test_cells"]) for g in evaluated),
            min_test_cells=int(conditions["min_test_cells"]),
        )
        report.info(
            f"{prefix}excluded_conditions",
            excluded=conditions["excluded_conditions"],
            note="dropped by --min-test-cells; both arms lose exactly the same guides",
        )
        per_fold_measured = {
            f"fold{fold['fold']}": [
                g for g in fold["test_guides"] if g in set(measured_guides)
            ]
            for fold in conditions["folds"]
        }
        report.record(
            f"{prefix}every_fold_has_activity_bearing_test_guides",
            all(len(v) >= 1 for v in per_fold_measured.values()),
            per_fold=per_fold_measured,
        )


# --------------------------------------------------------------------------
# part: model
# --------------------------------------------------------------------------


def run_model_part(args: argparse.Namespace, report: Report) -> None:
    import sys as _sys

    repo_src = str(Path(args.repo) / "src")
    if repo_src not in _sys.path:
        _sys.path.insert(0, repo_src)

    import torch

    from perturbdiff import ModelConfig, PerturbDiffusionModel
    from perturbdiff.config import GuideFeatureEncoderConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        a = torch.randn(128, 128, device=device)
        (a @ a).sum().item()
        report.record("gpu_matmul_executes", True, device=torch.cuda.get_device_name(0))
        report.info("gpu_arch_list", arch_list=list(torch.cuda.get_arch_list()))
    else:
        report.record("gpu_matmul_executes", False, device="cpu", note="no CUDA device visible")

    inputs = args.inputs_dir
    for state in [s.lower() for s in args.states]:
        conditions = json.loads((inputs / f"conditions_{state}.json").read_text(encoding="utf-8"))
        prefix = f"{state}:"
        # The perturbation vocabulary is the target family: control + the two
        # targets (the ablation trains with --perturbation-col guide_family).
        families = sorted({CONTROL_LABEL, "NANOG", "OCT4"})
        mapping = {name: i for i, name in enumerate(families)}
        n_genes = int(json.loads((inputs / "genes.json").read_text(encoding="utf-8"))["n_genes"])

        for arm, enabled in (
            ("activity_off", False),
            ("activity_on", True),
        ):
            guide_encoder = GuideFeatureEncoderConfig(
                enabled=enabled,
                feature_dim=1 if enabled else 0,
                activity_source="direct" if enabled else "predicted",
                explicit_condition=True,
                activity_scales_delta=False,
                architecture="mlp",
                sequence_length=20,
            )
            cfg = ModelConfig(
                n_genes=n_genes,
                n_perturbations=len(mapping),
                latent_dim=128,
                guide_encoder=guide_encoder,
            )
            model = PerturbDiffusionModel(cfg).to(device)
            has_doser = model.guide_doser is not None
            report.record(
                f"{prefix}{arm}_model_builds",
                True,
                n_perturbations=cfg.n_perturbations,
                guide_encoder_enabled=enabled,
                has_guide_doser=has_doser,
            )
            if enabled:
                report.record(
                    f"{prefix}{arm}_doser_covers_target_ids",
                    has_doser and int(model.guide_doser.bias.shape[1]) >= len(mapping),
                    doser_table_size=None if not has_doser else int(model.guide_doser.bias.shape[1]),
                    n_target_ids=len(mapping),
                    note="upstream sizes the doser from n_perturbations; equal here because the perturbation column is already the target family",
                )
            # Exercise the conditioning path itself (forward() only denoises and
            # therefore cannot catch a mis-sized doser or a dead activity
            # channel).  encode_condition is the function that consumes
            # target_ids + guide_features.
            perturbation_ids = torch.full((4,), mapping["OCT4"], dtype=torch.long, device=device)
            guide_features = (
                torch.tensor(
                    [[0.0], [0.25], [0.5], [1.0]], dtype=torch.float32, device=device
                )
                if enabled
                else None
            )
            with torch.no_grad():
                condition = model.encode_condition(
                    torch.randn(4, cfg.n_genes, device=device) * 0.1,
                    perturbation_ids,
                    target_ids=perturbation_ids,
                    guide_features=guide_features,
                )
            report.record(
                f"{prefix}{arm}_conditioning_path_runs",
                condition["z_pert"].shape == (4, cfg.latent_dim),
                z_pert_shape=list(condition["z_pert"].shape),
                activity_embedding_present=condition["activity_embedding"] is not None,
            )
            if enabled:
                embedding = condition["activity_embedding"]
                # A working channel must depend on the activity value.  This is
                # the cheap structural version of the inference-time
                # perturbation probe; the real probe runs after training.
                spread = (
                    float((embedding - embedding.mean(dim=0, keepdim=True)).abs().max())
                    if embedding is not None
                    else 0.0
                )
                report.record(
                    f"{prefix}{arm}_activity_embedding_depends_on_value",
                    embedding is not None and spread > 0.0,
                    max_row_deviation=spread,
                )
                strength = condition["strength"]
                report.record(
                    f"{prefix}{arm}_direct_activity_passthrough",
                    strength is not None
                    and bool(torch.allclose(strength, torch.tensor([0.0, 0.25, 0.5, 1.0], device=device))),
                    strength=None if strength is None else [round(float(v), 4) for v in strength],
                )

        # Audited-loader round trip on the activity arm, using a real save format.
        guide_encoder = GuideFeatureEncoderConfig(
            enabled=True, feature_dim=1, activity_source="direct", explicit_condition=True,
            activity_scales_delta=False, architecture="mlp", sequence_length=20,
        )
        cfg = ModelConfig(n_genes=n_genes, n_perturbations=len(mapping), latent_dim=128, guide_encoder=guide_encoder)
        model = PerturbDiffusionModel(cfg).to(device)
        checkpoint = args.output.parent / f"_preflight_roundtrip_{state}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": cfg,
                "perturbation_mapping": mapping,
                "epoch": 1,
                "val_loss": float("nan"),
            },
            checkpoint,
        )
        try:
            import scperturbench_diffae_loader as loader

            _, loaded_mapping, loader_report = loader.load_model(checkpoint, torch.device("cpu"))
            report.record(
                f"{prefix}audited_loader_restores_doser",
                not loader_report["missing_doser_keys"]
                and loader_report["embedding_rows_in_model"]
                == loader_report["embedding_rows_in_checkpoint"]
                and loaded_mapping == mapping,
                **{
                    k: loader_report[k]
                    for k in (
                        "doser_table_size_in_checkpoint",
                        "doser_table_size_in_model",
                        "missing_doser_keys",
                        "embedding_rows_in_model",
                        "embedding_rows_in_checkpoint",
                    )
                },
            )
        finally:
            checkpoint.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()

    if args.merge:
        merged = {"parts": [], "n_checks": 0, "n_failed": 0, "failed": [], "checks": []}
        for path in args.merge:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            merged["parts"].append(payload.get("part", str(path)))
            merged["checks"].extend(payload.get("checks", []))
            merged["failed"].extend(payload.get("failed", []))
        merged["n_checks"] = len(merged["checks"])
        merged["n_failed"] = len(merged["failed"])
        merged["all_passed"] = not merged["failed"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"PREFLIGHT_MERGED checks={merged['n_checks']} failed={merged['n_failed']}")
        for name in merged["failed"]:
            print(f"  FAILED {name}")
        return 0 if merged["all_passed"] else 1

    if not args.part or not args.inputs_dir:
        raise SystemExit("--part and --inputs-dir are required unless --merge is used")

    report = Report()
    if args.part == "data":
        run_data_part(args, report)
    else:
        run_model_part(args, report)

    payload = report.payload(args.part)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PREFLIGHT part={args.part} checks={payload['n_checks']} failed={payload['n_failed']}")
    for name in payload["failed"]:
        print(f"  FAILED {name}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
