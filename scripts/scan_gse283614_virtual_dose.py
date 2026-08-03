from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.gse283614_biology import fit_response_models, read_gene_sets
from perturbdiff.model import PerturbDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory continuous-dose scan for validated GSE283614 models.")
    parser.add_argument("--adata-path", type=Path, required=True)
    parser.add_argument("--esc-checkpoint", type=Path, required=True)
    parser.add_argument("--de-checkpoint", type=Path, required=True)
    parser.add_argument("--gene-set-workbook", type=Path, required=True)
    parser.add_argument("--oof-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dose-step", type=float, default=0.05)
    parser.add_argument("--max-control-cells", type=int, default=128)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-failed-oof", action="store_true")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> tuple[PerturbDiffusionModel, dict[str, int]]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state["model_config"]
    has_activity_condition = any(key.startswith("activity_condition.") for key in state["model_state_dict"])
    if getattr(cfg.guide_encoder, "enabled", False):
        cfg = replace(
            cfg,
            guide_encoder=replace(
                cfg.guide_encoder,
                explicit_condition=has_activity_condition,
                activity_scales_delta=not has_activity_condition,
            ),
        )
    model = PerturbDiffusionModel(cfg).to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    return model, {str(key): int(value) for key, value in state["perturbation_mapping"].items()}


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def score(mean: np.ndarray, control: np.ndarray, indices: dict[str, list[int]]) -> dict[str, float]:
    delta = {name: float(mean[idx].mean() - control[idx].mean()) for name, idx in indices.items() if idx}
    return {
        "pluripotency_drop": -delta.get("pluripotency", float("nan")),
        "de_gain": delta.get("de", float("nan")),
        "ne_gain": delta.get("ne", float("nan")),
        "essentiality_delta": delta.get("essentiality", float("nan")),
    }


def main() -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("This script requires anndata") from exc
    args = parse_args()
    oof = json.loads(args.oof_summary.read_text(encoding="utf-8"))
    if not bool(oof.get("overall_pass", False)) and not args.allow_failed_oof:
        raise RuntimeError("OOF biological validation did not pass; refusing to present a full-data dose scan")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.adata_path)
    genes = list(adata.var_names.astype(str))
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    gene_sets = read_gene_sets(args.gene_set_workbook, genes)
    indices = {name: [gene_index[gene] for gene in members] for name, members in gene_sets.items()}
    states = adata.obs["cell_state"].astype(str).to_numpy()
    labels = adata.obs["perturbation_label"].astype(str).to_numpy()
    device = torch.device(args.device)
    checkpoints = {"ESC": args.esc_checkpoint, "DE": args.de_checkpoint}
    doses = np.arange(0.0, 1.0 + args.dose_step / 2.0, args.dose_step)
    rows: list[dict[str, object]] = []

    for state, checkpoint in checkpoints.items():
        model, mapping = load_model(checkpoint, device)
        controls = dense(adata.X[(states == state) & (labels == "control")]).astype(np.float32, copy=False)
        rng = np.random.default_rng(stable_seed(args.seed, state, "controls"))
        if len(controls) > args.max_control_cells:
            controls = controls[rng.choice(len(controls), args.max_control_cells, replace=False)]
        control_mean = controls.mean(axis=0)
        x_ctrl = torch.from_numpy(controls).to(device)
        for target in ("NANOG", "OCT4"):
            target_id = mapping.get(target)
            if target_id is None:
                raise KeyError(f"Target {target!r} missing from {checkpoint}")
            ids = torch.full((len(controls),), target_id, dtype=torch.long, device=device)
            generator = torch.Generator(device=device)
            generator.manual_seed(stable_seed(args.seed, state, target, "noise"))
            shared_noise = torch.randn(x_ctrl.shape, generator=generator, device=device, dtype=x_ctrl.dtype)
            for knockdown in doses:
                features = torch.full((len(controls), 1), float(knockdown), device=device, dtype=x_ctrl.dtype)
                with torch.no_grad():
                    prediction = model.sample_counterfactual(
                        x_ctrl,
                        ids,
                        guide_features=features,
                        steps=args.ddim_steps,
                        noise=shared_noise,
                    )
                mean = prediction.detach().cpu().numpy().mean(axis=0)
                rows.append(
                    {
                        "state": state,
                        "dose_target": target,
                        "knockdown_fraction": float(knockdown),
                        "tf_remaining": float(1.0 - knockdown),
                        **score(mean, control_mean, indices),
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "virtual_dose_scan.csv", index=False)
    findings: list[dict[str, object]] = []
    for target in ("NANOG", "OCT4"):
        de = table[(table["state"] == "DE") & (table["dose_target"] == target)]
        optimum = de.loc[de["de_gain"].idxmax()]
        ne_min = float(de["ne_gain"].min())
        ne_span = float(de["ne_gain"].max() - ne_min)
        threshold_rows = de[de["ne_gain"] >= ne_min + 0.5 * ne_span]
        findings.append(
            {
                "dose_target": target,
                "predicted_golden_knockdown": float(optimum["knockdown_fraction"]),
                "predicted_golden_tf_remaining": float(optimum["tf_remaining"]),
                "off_target_half_range_threshold": (
                    None if threshold_rows.empty else float(threshold_rows["knockdown_fraction"].min())
                ),
            }
        )
    oct4_esc = table[(table["state"] == "ESC") & (table["dose_target"] == "OCT4")]
    response = oct4_esc["pluripotency_drop"].to_numpy(dtype=float)
    span = float(response.max() - response.min())
    normalized = (response - response.min()) / span if span > 0 else np.zeros_like(response)
    hill = next(
        row
        for row in fit_response_models(oct4_esc["tf_remaining"], normalized, models=("linear", "hill"))
        if row["model"] == "hill"
    )
    summary = {
        "exploratory": True,
        "oof_overall_pass": bool(oof.get("overall_pass", False)),
        "dose_step": args.dose_step,
        "findings": findings,
        "oct4_esc_ed50_tf_remaining": hill.get("ed50"),
        "oct4_esc_hill_coefficient": hill.get("hill_coefficient"),
        "warning": "This full-data interpolation is model inference and does not replace held-out-guide evidence.",
    }
    (args.output_dir / "virtual_dose_scan_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
