"""Host-side activity-axis context for the GSE283614 report.

Computes three descriptions of the measured-activity axis, none of which is ever
fed to a model:

1. proxy quality - Spearman between the measured screen value and the observed
   held-out single-cell response magnitude, overall and within each target;
2. coverage and degeneracy per state;
3. cross-guide separability - for each target, the mean pairwise L2 distance
   between guide centroids next to each guide's distance to the control centroid.
   This bounds what ANY guide-level conditioning signal could buy: if two guides of
   the same target are nearly indistinguishable in expression space, no scalar can
   separate them.

Mirrors tools/build_activity_context_gse283614.py (which needs the repo's anndata
stack) but runs on the host so the report does not depend on a container round trip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activity-axis context (host-side).")
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--state", default="esc")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-cells-per-guide", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def separability(prep: Path, state: str, seed: int, cap: int) -> dict:
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    folds = sorted((prep / "dose_cv" / state).glob("fold_*"))
    control_blocks: list[np.ndarray] = []
    by_guide: dict[str, list[np.ndarray]] = {}
    guide_target: dict[str, str] = {}
    for fold in folds:
        adata = ad.read_h5ad(fold / f"gse283614_{state}_test.h5ad")
        labels = adata.obs["perturbation_label"].astype(str).to_numpy()
        targets = adata.obs["target_gene"].astype(str).to_numpy()
        matrix = adata.X
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        dense = np.asarray(dense, dtype=np.float64)
        ctrl = np.flatnonzero(labels == "control")
        take = rng.choice(ctrl, min(cap, ctrl.size), replace=False)
        control_blocks.append(dense[np.sort(take)])
        for guide in sorted(set(labels) - {"control"}):
            idx = np.flatnonzero(labels == guide)
            picked = rng.choice(idx, min(cap, idx.size), replace=False)
            by_guide.setdefault(guide, []).append(dense[np.sort(picked)])
            guide_target[guide] = {"NANOG": "NANOG", "POU5F1": "OCT4"}.get(
                str(targets[idx[0]]), str(targets[idx[0]])
            )

    control_mean = np.vstack(control_blocks).mean(axis=0)
    centroids = {g: np.vstack(b).mean(axis=0) for g, b in by_guide.items()}

    out: dict[str, object] = {
        "n_guides": len(centroids),
        "n_controls": int(np.vstack(control_blocks).shape[0]),
    }
    for target in ("NANOG", "OCT4"):
        members = [g for g, t in guide_target.items() if t == target]
        if len(members) < 2:
            continue
        to_control = [float(np.linalg.norm(centroids[g] - control_mean)) for g in members]
        pairwise = [
            float(np.linalg.norm(centroids[a] - centroids[b]))
            for i, a in enumerate(members)
            for b in members[i + 1 :]
        ]
        out[target] = {
            "n_guides": len(members),
            "mean_guide_to_control_l2": float(np.mean(to_control)),
            "mean_pairwise_guide_l2": float(np.mean(pairwise)),
            "min_pairwise_guide_l2": float(np.min(pairwise)),
            "max_pairwise_guide_l2": float(np.max(pairwise)),
            "guide_separation_ratio": float(np.mean(pairwise) / np.mean(to_control)),
            "per_guide_to_control_l2": {g: d for g, d in zip(members, to_control)},
        }
    return out


def main() -> int:
    args = parse_args()
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    conditions = json.loads(
        (args.inputs_dir / f"conditions_{args.state}.json").read_text(encoding="utf-8")
    )
    state_diag = diagnostic["states"][args.state]
    payload = {
        "state": args.state,
        "activity_column": conditions["activity_column"],
        "activity_source": conditions["activity_source"],
        "coverage": {
            "n_evaluated": conditions["n_evaluated"],
            "n_measured": conditions["n_evaluated_measured_activity"],
            "n_unmeasured_written_as_zero": conditions["n_evaluated"]
            - conditions["n_evaluated_measured_activity"],
            "n_excluded_by_min_test_cells": len(conditions["excluded_conditions"]),
            "excluded_conditions": conditions["excluded_conditions"],
            "activity_distribution": state_diag["activity_distribution"],
        },
        "proxy_quality": {
            "definition": (
                "Spearman between the measured screen relative activity and the observed "
                "single-cell response magnitude (L2 from the guide mean to the state control "
                "mean), on held-out ground truth. Diagnostic only; never a model input."
            ),
            "overall": {
                "spearman_rho": state_diag["observed_response_diagnostic"][
                    "spearman_activity_vs_observed_response"
                ],
                "p_value": state_diag["observed_response_diagnostic"]["p_value"],
                "n": state_diag["observed_response_diagnostic"]["n"],
            },
            "within_target": state_diag["observed_response_diagnostic"]["within_target"],
        },
        "coverage_by_fold": {
            name: {
                "n_test_guides": fold["n_test_guides"],
                "n_measured": fold["n_measured"],
                "n_unmeasured": fold["n_unmeasured"],
                "guide_type_counts": fold["guide_type_counts"],
            }
            for name, fold in state_diag["per_fold"].items()
        },
    }
    payload["cross_guide_separability"] = separability(
        args.prep_dir, args.state, args.seed, args.max_cells_per_guide
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
