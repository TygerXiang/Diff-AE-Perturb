"""Assemble the reference context the report must cite alongside each number.

The activity axis is a *proxy*, and the report has to state how good that proxy
is rather than assert it.  This tool computes three independent descriptions,
none of which is fed to any model:

1. **Observed vs screen activity** (from the diagnostic): Spearman correlation
   between measured screen depletion and the held-out single-cell response
   magnitude of the same guide, overall and within each target.
2. **Coverage and degeneracy** per state: how many guides carry a measured value,
   how many are exact zeros, how many are unmeasured and therefore written as 0.
3. **Cross-guide separability of the observed responses**: for each target, the
   mean pairwise L2 distance between guide centroids next to each guide's distance
   to the control centroid.  This bounds what any conditioning signal could
   possibly buy: if two guides of the same target are almost indistinguishable in
   expression space, no guide-level scalar can separate them.

Output: ``reports/assets/gse283614_activity_ablation/activity_context.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the GSE283614 activity-axis context block.")
    parser.add_argument("--inputs-dir", type=Path,
                        default=Path("/root/perturbench/.scperturbench/inputs/gse283614_activity_ablation"))
    parser.add_argument("--diagnostic", type=Path,
                        default=Path("/root/perturbench/.scperturbench/inputs/gse283614_activity_diagnostic.json"))
    parser.add_argument("--prep-dir", type=Path,
                        default=Path("/root/Diff-AE-Perturb/reproducibility/prepared/gse283614"))
    parser.add_argument("--state", default="esc")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-cells-per-guide", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def guide_centroid_separability(prep: Path, state: str, seed: int, cap: int) -> dict:
    import anndata as ad

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
    centroids = {g: np.vstack(blocks).mean(axis=0) for g, blocks in by_guide.items()}

    out: dict[str, object] = {"n_guides": len(centroids), "n_controls": int(np.vstack(control_blocks).shape[0])}
    for target in ("NANOG", "OCT4"):
        members = [g for g, t in guide_target.items() if t == target]
        if len(members) < 2:
            continue
        dist_to_control = [float(np.linalg.norm(centroids[g] - control_mean)) for g in members]
        pairwise = [
            float(np.linalg.norm(centroids[a] - centroids[b]))
            for i, a in enumerate(members)
            for b in members[i + 1 :]
        ]
        out[target] = {
            "n_guides": len(members),
            "mean_guide_to_control_l2": float(np.mean(dist_to_control)),
            "mean_pairwise_guide_l2": float(np.mean(pairwise)),
            "min_pairwise_guide_l2": float(np.min(pairwise)),
            "max_pairwise_guide_l2": float(np.max(pairwise)),
            # < 1 means two guides of the same target are closer to each other than a
            # guide is to control, i.e. the target effect dominates the guide effect.
            "guide_separation_ratio": float(np.mean(pairwise) / np.mean(dist_to_control)),
            "per_guide_to_control_l2": {g: d for g, d in zip(members, dist_to_control)},
        }
    return out


def main() -> int:
    args = parse_args()
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    conditions_path = args.inputs_dir / f"conditions_{args.state}.json"
    conditions = json.loads(conditions_path.read_text(encoding="utf-8"))

    state_diag = diagnostic["states"][args.state]
    payload = {
        "state": args.state,
        "activity_column": conditions["activity_column"],
        "activity_source": conditions["activity_source"],
        "pre_registered_by_the_prompt": False,
        "targets": {
            "n_evaluated": conditions["n_evaluated"],
            "n_measured": conditions["n_evaluated_measured_activity"],
            "n_unmeasured_written_as_zero": conditions["n_evaluated"]
            - conditions["n_evaluated_measured_activity"],
            "n_excluded_by_min_test_cells": len(conditions["excluded_conditions"]),
            "excluded_conditions": conditions["excluded_conditions"],
            "extra_unmeasured": conditions["n_test_conditions_total"] - conditions["n_evaluated"],
        },
        "proxy_quality": {
            "definition": (
                "Spearman correlation between the measured screen relative activity and the "
                "observed single-cell response magnitude (L2 from the guide mean to the state "
                "control mean) of the same guide, on held-out ground truth. Never used as a "
                "model input; it only says how informative the proxy is."
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
        "activity_distribution": state_diag["activity_distribution"],
    }
    payload["cross_guide_separability"] = guide_centroid_separability(
        args.prep_dir, args.state, args.seed, args.max_cells_per_guide
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
