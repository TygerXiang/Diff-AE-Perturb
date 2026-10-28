"""H3(b)/T3.2: does the model reproduce the DE dose-response *shape*, or only its level?

The pre-registered H3 test (b) asks whether the predicted response is monotone where
the ground truth is not: "the share of predicted inverted-U responses is at least 30
points below the ground-truth share".  This tool answers it directly.

For every guide evaluated in a design it computes the mean expression profile of
(a) the ground-truth cells and (b) the model's imputed cells, for the activity-on and
activity-off arms, then fits the same linear/quadratic family the paper uses across
guides ordered by remaining TF level (``tf_remaining`` from the paper's dosage table).
A gene whose quadratic term is negative with an interior vertex is an inverted-U
response (peak inside the observed dose range); a positive quadratic term with an
interior vertex is a U response.  Both are non-monotone.

Reads only existing ``result.h5ad`` files; nothing is retrained or rescored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WS = HERE.parent
DIFFAE = Path(r"C:\Users\Admin\myprojects\Diff-AE-Perturb")
DOSAGE = DIFFAE / "runs/biological_validation/gse283614_observed/gse283614_tf_dosage.tsv"

DESIGNS = {
    "dose_cv": WS / ".scperturbench/results/gse283614_activity_ablation",
    "kd3": WS / ".scperturbench/results/gse283614_activity_ablation_kd3",
}
FOLDS = {"dose_cv": [0, 1, 2, 3, 4], "kd3": [0, 1, 2]}
STATES = ["esc", "de"]
ARMS = ["activity_on", "activity_off"]
# Ground-truth share must exceed the predicted share by this many points for H3(b).
GAP_POINTS = 30.0


def fit_shape(x: np.ndarray, y: np.ndarray) -> dict:
    """Linear vs quadratic on x; interior vertex + sign classifies the shape."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 6 or np.ptp(x) == 0:
        return {}
    design_lin = np.column_stack([np.ones_like(x), x])
    design_quad = np.column_stack([np.ones_like(x), x, x ** 2])
    beta_lin, res_lin, *_ = np.linalg.lstsq(design_lin, y, rcond=None)
    beta_quad, res_quad, *_ = np.linalg.lstsq(design_quad, y, rcond=None)
    rss_lin = float(((y - design_lin @ beta_lin) ** 2).sum())
    rss_quad = float(((y - design_quad @ beta_quad) ** 2).sum())
    n = int(x.size)
    aic_lin = n * np.log(max(rss_lin / n, 1e-300)) + 2 * 2
    aic_quad = n * np.log(max(rss_quad / n, 1e-300)) + 2 * 3
    delta = aic_lin - aic_quad
    a = float(beta_quad[2])
    vertex = float(-beta_quad[1] / (2 * a)) if a != 0 else np.nan
    interior = bool(np.isfinite(vertex) and x.min() < vertex < x.max())
    if interior and a < 0:
        shape = "inverted_u"
    elif interior and a > 0:
        shape = "u"
    else:
        shape = "monotone"
    return {"n": n, "delta_aic_linear_minus_quadratic": float(delta),
            "quadratic_coef": a, "vertex_tf_remaining": vertex,
            "interior_vertex": interior, "shape": shape,
            "quadratic_selected": bool(delta > 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", nargs="+", default=["kd3", "dose_cv"])
    parser.add_argument("--output-dir", type=Path,
                        default=WS / "reports/assets/gse283614_de_vs_esc")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import anndata as ad

    dosage = pd.read_csv(DOSAGE, sep="\t")
    dose = (dosage.dropna(subset=["guide_identity"])
            .drop_duplicates("guide_identity")
            .set_index("guide_identity")["tf_remaining"].astype(float).to_dict())

    rows = []
    for design in args.designs:
        root = DESIGNS[design] / "gse283614"
        for state in STATES:
            for arm in ARMS:
                profiles: dict[str, dict[str, np.ndarray]] = {"truth": {}, "prediction": {}}
                for fold in FOLDS[design]:
                    path = root / state / f"fold_{fold}" / arm / "seed7/result.h5ad"
                    if not path.exists():
                        continue
                    result = ad.read_h5ad(path)
                    genes = result.var_names.astype(str).to_numpy()
                    X = result.X
                    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                    X = np.asarray(X, dtype=np.float64)
                    for category, key in (("stimulated", "truth"), ("imputed", "prediction")):
                        mask = (result.obs["Expcategory"].astype(str) == category).to_numpy()
                        if not mask.any():
                            continue
                        guides = result.obs["perturbation"].astype(str).to_numpy()[mask]
                        block = X[mask]
                        for guide in np.unique(guides):
                            sel = guides == guide
                            if sel.sum() < 5:
                                continue
                            profiles[key][str(guide)] = block[sel].mean(axis=0)
                    del result, X
                if not profiles["truth"] or not profiles["prediction"]:
                    continue
                shared = sorted(set(profiles["truth"]) & set(profiles["prediction"]) & set(dose))
                if len(shared) < 6:
                    continue
                x = np.array([dose[g] for g in shared], dtype=float)
                truth = np.vstack([profiles["truth"][g] for g in shared])
                pred = np.vstack([profiles["prediction"][g] for g in shared])
                print(f"[{design}/{state}/{arm}] {len(shared)} shared guides, "
                      f"{truth.shape[1]} genes", flush=True)
                for i, gene in enumerate(genes):
                    for source, matrix in (("truth", truth), ("prediction", pred)):
                        fit = fit_shape(x, matrix[:, i])
                        if not fit:
                            continue
                        rows.append({"design": design, "state": state, "arm": arm,
                                     "gene": str(gene), "source": source, **fit})
    frame = pd.DataFrame(rows)
    out = args.output_dir / "response_shape.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out.name} ({len(frame)} rows)")

    summary = (frame.groupby(["design", "state", "arm", "source"])
               .agg(n_genes=("gene", "nunique"),
                    share_quadratic=("quadratic_selected", "mean"),
                    share_inverted_u=("shape", lambda s: float((s == "inverted_u").mean())),
                    share_u=("shape", lambda s: float((s == "u").mean())),
                    share_nonmonotone=("shape", lambda s: float((s != "monotone").mean())),
                    median_vertex=("vertex_tf_remaining", "median"))
               .reset_index())
    summary["share_nonmonotone_pct"] = 100 * summary["share_nonmonotone"]
    wide = summary.pivot_table(index=["design", "state", "arm"], columns="source",
                               values=["share_nonmonotone_pct", "share_inverted_u"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["gap_nonmonotone_pts"] = (wide["share_nonmonotone_pct_truth"]
                                   - wide["share_nonmonotone_pct_prediction"])
    wide["gap_inverted_u_pts"] = 100 * (wide["share_inverted_u_truth"]
                                        - wide["share_inverted_u_prediction"])
    wide["h3b_truth_exceeds_prediction_by_30pts"] = wide["gap_nonmonotone_pts"] >= GAP_POINTS
    wide.to_csv(args.output_dir / "response_shape_summary.csv", index=False)
    print(f"wrote response_shape_summary.csv ({len(wide)} rows)")

    # Like-for-like DEG-restricted view: the paper's non-monotonicity claim is about
    # DEGs, and the DEG panel differs between states, so repeat the summary on the
    # per-guide DEG cut-off sets (padj <= 0.01 and |FC| outside [0.5, 2]), which is
    # the same set the common_degs metric uses.  ``order`` alone is the full ranking.
    import json

    deg_inputs = {
        "kd3": (WS / ".scperturbench/inputs/gse283614_knockdown3_inputs", "deg_table_kd3"),
        "dose_cv": (WS / ".scperturbench/inputs/gse283614_activity_ablation", "deg_table"),
    }
    rows_deg = []
    for (design, state), block in frame.groupby(["design", "state"]):
        folder, prefix = deg_inputs[design]
        genes: set[str] = set()
        for fold in FOLDS[design]:
            path = folder / f"{prefix}_{state}_fold{fold}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for entry in payload.get("genes", {}).values():
                for gene, padj, fc in zip(entry.get("order", []),
                                          entry.get("pvals_adj", []),
                                          entry.get("foldchanges", [])):
                    if padj is None or fc is None:
                        continue
                    if padj <= 0.01 and (fc >= 2.0 or fc <= 0.5):
                        genes.add(str(gene))
        if not genes:
            continue
        sub = block[block["gene"].isin(genes)]
        for (arm, source), sblock in sub.groupby(["arm", "source"]):
            rows_deg.append({
                "design": design, "state": state, "arm": arm, "source": source,
                "n_deg_genes": int(sblock["gene"].nunique()),
                "share_nonmonotone_pct": 100 * float((sblock["shape"] != "monotone").mean()),
                "share_inverted_u_pct": 100 * float((sblock["shape"] == "inverted_u").mean()),
                "share_quadratic_selected_pct": 100 * float(sblock["quadratic_selected"].mean()),
                "median_vertex_tf_remaining": float(sblock["vertex_tf_remaining"].median()),
            })
    deg = pd.DataFrame(rows_deg)
    if not deg.empty:
        wide_deg = deg.pivot_table(index=["design", "state", "arm"], columns="source",
                                   values=["share_nonmonotone_pct", "share_inverted_u_pct"]).reset_index()
        wide_deg.columns = [f"{a}_{b}" for a, b in wide_deg.columns]
        wide_deg["gap_nonmonotone_pts"] = (wide_deg["share_nonmonotone_pct_truth"]
                                          - wide_deg["share_nonmonotone_pct_prediction"])
        wide_deg["h3b_truth_exceeds_prediction_by_30pts"] = wide_deg["gap_nonmonotone_pts"] >= GAP_POINTS
        deg.to_csv(args.output_dir / "response_shape_summary_deg.csv", index=False)
        wide_deg.to_csv(args.output_dir / "response_shape_summary_deg_wide.csv", index=False)
        print(f"wrote response_shape_summary_deg.csv ({len(deg)} rows)")
        print(wide_deg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(wide.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
