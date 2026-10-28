"""Activity-axis context for the knockdown-stratum fold set.

Unlike ``build_activity_context_gse283614.py`` this resolves each fold's test file
from the conditions JSON rather than assuming a ``dose_cv`` directory layout, so it
works for any fold set.  Proxy-quality numbers come from the matching diagnostic
JSON; cross-guide separability is recomputed from the fold files that experiment
actually used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activity-axis context for a fold set.")
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--state", default="esc")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-cells-per-guide", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def separability(conditions: dict, seed: int, cap: int) -> dict:
    import anndata as ad

    rng = np.random.default_rng(seed)
    control_blocks: list[np.ndarray] = []
    by_guide: dict[str, list[np.ndarray]] = {}
    guide_target: dict[str, str] = {}
    for fold in conditions["folds"]:
        adata = ad.read_h5ad(Path(fold["test_path"]))
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
        }
    return out


def main() -> int:
    args = parse_args()
    conditions = json.loads(
        (args.inputs_dir / f"conditions_{args.state}.json").read_text(encoding="utf-8")
    )
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    diag = diagnostic["states"][args.state]

    # Per-fold knockdown-adjacent facts the report needs: how much activity the
    # test block carries, and how many test guides are unmeasured.
    fold_rows = []
    for fold in conditions["folds"]:
        guides = list(fold["test_guides"])
        measured = [g for g in guides if conditions["per_condition"][g]["activity_measured"]]
        nonzero = [g for g in guides if conditions["per_condition"][g]["activity_input"] > 0]
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "n_test_guides": len(guides),
                "n_measured": len(measured),
                "n_nonzero": len(nonzero),
                "test_cells": int(sum(conditions["per_condition"][g]["n_test_cells"] for g in guides)),
                "min_test_cells": int(min(conditions["per_condition"][g]["n_test_cells"] for g in guides)),
            }
        )

    payload = {
        "state": args.state,
        "fold_set": conditions.get("fold_set", "unspecified"),
        "activity_column": conditions["activity_column"],
        "activity_source": conditions["activity_source"],
        "coverage": {
            "n_evaluated": conditions["n_evaluated"],
            "n_measured": conditions["n_evaluated_measured_activity"],
            "n_unmeasured_written_as_zero": conditions["n_evaluated"]
            - conditions["n_evaluated_measured_activity"],
            "excluded_by_min_test_cells": conditions["excluded_conditions"],
            "activity_distribution": diag["activity_distribution"],
        },
        "per_fold": fold_rows,
        "proxy_quality": {
            "definition": (
                "Spearman between measured screen relative activity and observed single-cell "
                "response magnitude (L2 from the guide mean to the state control mean), on "
                "held-out ground truth. Diagnostic only; never a model input."
            ),
            "overall": {
                "spearman_rho": diag["observed_response_diagnostic"][
                    "spearman_activity_vs_observed_response"
                ],
                "p_value": diag["observed_response_diagnostic"]["p_value"],
                "n": diag["observed_response_diagnostic"]["n"],
            },
            "within_target": diag["observed_response_diagnostic"]["within_target"],
        },
        "cross_guide_separability": separability(conditions, args.seed, args.max_cells_per_guide),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
