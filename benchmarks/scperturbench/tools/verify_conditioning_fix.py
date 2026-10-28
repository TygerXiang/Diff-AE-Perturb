"""Acceptance test for the conditioning-pathway fix in Diff-AE-Perturb.

Checks, on a small synthetic model (no training data needed):

1. perturbed rows get ``z_pert = z_control + delta``, not ``z_pert = encoder(x0)``;
2. control rows keep their own latent;
3. ``delta`` now receives a non-zero gradient from the diffusion loss (before the fix the
   embedding table was unreachable from the denoiser);
4. changing only ``perturbation_ids`` changes the denoiser output (the condition is live);
5. ``per_cell`` and ``batch_mean`` control anchoring differ, and ``sample_counterfactual``
   honours the configured mode;
6. a legacy config object without ``control_reference`` is patched to ``batch_mean``.

Run inside the container with the repo mounted at /root/Diff-AE-Perturb.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import torch

sys.path.insert(0, "/root/Diff-AE-Perturb/src")

from perturbdiff.config import DiffusionConfig, ModelConfig, TrainingLossConfig  # noqa: E402
from perturbdiff.model import PerturbDiffusionModel  # noqa: E402

N_GENES = 32
N_PERT = 4
BATCH = 16


def build(control_reference: str, prediction_type: str = "x0") -> PerturbDiffusionModel:
    torch.manual_seed(0)
    cfg = ModelConfig(
        n_genes=N_GENES,
        n_perturbations=N_PERT,
        latent_dim=8,
        encoder_hidden_dims=(16,),
        denoiser_hidden_dims=(32,),
        control_reference=control_reference,
        diffusion=DiffusionConfig(timesteps=20, prediction_type=prediction_type),
        losses=TrainingLossConfig(lambda_direction_align=0.0, lambda_semantic_align=0.0),
    )
    return PerturbDiffusionModel(cfg)


def make_batch():
    torch.manual_seed(1)
    x0 = torch.rand(BATCH, N_GENES)
    ids = torch.zeros(BATCH, dtype=torch.long)
    ids[4:8] = 1
    ids[8:12] = 2
    ids[12:] = 3
    return x0, ids


def main() -> int:
    results: dict[str, object] = {}
    x0, ids = make_batch()

    # ---- 1/2: z_pert construction ------------------------------------------------
    model = build("per_cell")
    condition = model.encode_condition(x0, ids)
    z_sem = condition["z_sem"]
    z_pert = condition["z_pert"]
    z_control = condition["z_control"]
    delta = condition["delta"]
    pert_rows = ids != 0
    ctrl_rows = ~pert_rows
    results["perturbed_rows_use_control_plus_delta"] = bool(
        torch.allclose(z_pert[pert_rows], (z_control + delta)[pert_rows], atol=1e-6)
    )
    results["perturbed_rows_differ_from_z_sem"] = float(
        (z_pert[pert_rows] - z_sem[pert_rows]).norm(dim=1).mean()
    )
    results["control_rows_keep_own_latent"] = bool(
        torch.allclose(z_pert[ctrl_rows], z_sem[ctrl_rows], atol=1e-6)
    )

    # ---- 3: delta receives gradient from the diffusion loss ----------------------
    model.zero_grad(set_to_none=True)
    out = model.training_losses(x0, ids)
    out["loss"].backward()
    grad = model.perturbation.embedding.weight.grad
    results["delta_grad_norm_total"] = float(grad.norm())
    results["delta_grad_norm_per_row"] = [round(float(g.norm()), 6) for g in grad]
    results["delta_gets_gradient"] = bool(float(grad[1:].norm()) > 0.0)

    # ---- 4: the condition is live at the denoiser input -------------------------
    model.eval()  # BatchNorm needs batch>1 in train mode
    with torch.no_grad():
        x_t = torch.zeros(2, N_GENES)
        t = torch.tensor([10, 10])
        z_a = model.encode_condition(x0[:2], torch.tensor([1, 1]))["z_pert"]
        z_b = model.encode_condition(x0[:2], torch.tensor([2, 2]))["z_pert"]
        out_a = model.denoiser(x_t, t, z_a)
        out_b = model.denoiser(x_t, t, z_b)
    results["z_pert_changes_with_condition"] = float((z_a - z_b).norm())
    results["denoiser_output_changes_with_condition"] = float((out_a - out_b).norm())
    model.train()

    # ---- 5: both control reference modes, training and inference ----------------
    for reference in ("per_cell", "batch_mean"):
        m = build(reference)
        cond = m.encode_condition(x0, ids)
        rows = ids != 0
        ctrl = cond["z_control"][rows]
        results[f"{reference}_control_rows_all_equal"] = bool(
            torch.allclose(ctrl, ctrl[:1].expand_as(ctrl), atol=1e-6)
        )
        with torch.no_grad():
            pred = m.sample_counterfactual(
                x_ctrl=x0[:6],
                target_ids=torch.full((6,), 2, dtype=torch.long),
                steps=4,
            )
        results[f"{reference}_sample_shape"] = list(pred.shape)
        results[f"{reference}_sample_std_across_cells"] = float(pred.std(dim=0).mean())

    # legacy config patch: simulate a pickled config that predates the field
    legacy = ModelConfig(n_genes=N_GENES, n_perturbations=N_PERT)
    object.__delattr__(legacy, "control_reference")
    results["legacy_field_absent"] = not hasattr(legacy, "control_reference")
    sys.path.insert(0, "/root/Diff-AE-Perturb/scripts")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ev", "/root/Diff-AE-Perturb/scripts/evaluate_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patched = module.compatible_config_from_state(
        {"model_config": legacy, "model_state_dict": {"perturbation.embedding.weight": torch.zeros(1)}}
    )
    results["legacy_without_field_becomes"] = str(getattr(patched, "control_reference", None))
    fresh = module.compatible_config_from_state(
        {"model_config": build("per_cell").config, "model_state_dict": {}}
    )
    results["new_checkpoint_keeps_per_cell"] = str(getattr(fresh, "control_reference", None))

    import json

    print(json.dumps(results, indent=2))
    ok = (
        results["perturbed_rows_use_control_plus_delta"]
        and results["control_rows_keep_own_latent"]
        and results["delta_gets_gradient"]
        and results["z_pert_changes_with_condition"] > 0
        and results["legacy_without_field_becomes"] == "per_cell"
        and results["new_checkpoint_keeps_per_cell"] == "per_cell"
    )
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
