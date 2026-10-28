"""Method adapters that produce the shared ``result.h5ad`` contract.

Every adapter writes one AnnData per (method, seed) containing

    obs['perturbation']  target gene name, or 'control' for the control block
    obs['Expcategory']   'control' | 'stimulated' | 'imputed'
    X                    float32, log-normalised expression, fixed gene order

so that ``scperturbench_metrics.py`` (and the literature's own ``calPerfor``)
can score it unchanged.

Methods
-------
diffae        Diff-AE-Perturb checkpoint (repo: C:\\Users\\Admin\\myprojects\\Diff-AE-Perturb)
linear        linear baseline on gene embeddings, literature ``genetic_baseReg.py`` recipe
              (variants: ridge | train_mean | control_mean)
scgen         scGen 2.1.1 (env ``scarches``, GPU-capable)
cpa           CPA 0.8.3 (env ``cpa``, CPU only: sm_120 has no torch 2.0.1 kernels)
gears         GEARS (env ``gears``, CPU only), cell-level predictions via
              ``gears.inference.evaluate`` exactly like the literature's ``myGears.getPredict``
squidiff      Squidiff 1.0.8 (env ``squidiff``, GPU-capable)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import anndata as ad
from scipy import sparse


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray())
    return np.asarray(x)


def load_genes(path: Path) -> list[str]:
    return [str(g) for g in json.loads(Path(path).read_text(encoding="utf-8"))["genes"]]


class Context:
    """Shared data contract for every adapter."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.train = ad.read_h5ad(args.train_path)
        self.test = ad.read_h5ad(args.test_path)
        self.genes = [str(g) for g in self.test.var_names]
        self.condition_col = args.condition_col
        self.control_label = args.control_label
        self.labels = self.test.obs[self.condition_col].astype(str).to_numpy()
        self.train_labels = self.train.obs[self.condition_col].astype(str).to_numpy()
        self.control_mask = self.labels == self.control_label
        self.control_indices = np.flatnonzero(self.control_mask)

        counts = self.test.obs[self.condition_col].astype(str).value_counts()
        conditions = sorted(
            str(label)
            for label, count in counts.items()
            if str(label) not in {self.control_label, "*"} and int(count) >= args.min_test_cells
        )
        if args.conditions:
            conditions = [c for c in conditions if c in set(args.conditions)]
        if args.max_conditions > 0:
            conditions = conditions[: args.max_conditions]
        self.conditions = conditions

        self.train_targets = sorted(
            {
                str(label)
                for label in self.train_labels
                if str(label) not in {self.control_label, "*"}
            }
        )

    def control_matrix(self) -> np.ndarray:
        return to_dense(self.test[self.control_indices].X).astype(np.float32)

    def n_pred_cells(self, condition: str) -> int:
        n_true = int((self.labels == condition).sum())
        return max(1, min(n_true, self.args.max_pred_cells))

    def sample_control_indices(self, condition: str, n: int, key: int) -> np.ndarray:
        rng = np.random.default_rng(self.args.seed * 100003 + key)
        if n <= self.control_indices.size:
            return rng.choice(self.control_indices, n, replace=False)
        return rng.choice(self.control_indices, n, replace=True)

    def true_cells(self, condition: str) -> np.ndarray:
        mask = self.labels == condition
        return to_dense(self.test[mask].X).astype(np.float32)

    def control_anndata(self, idx: np.ndarray) -> ad.AnnData:
        mask = np.zeros(self.test.n_obs, dtype=bool)
        mask[idx] = True
        return self.test[mask].copy()

    def control_dense_subset(self, idx: np.ndarray) -> np.ndarray:
        return self.control_matrix()[np.isin(self.control_indices, idx)]


def assemble_result(ctx: Context, per_condition: dict[str, np.ndarray]) -> ad.AnnData:
    """Build the shared result AnnData (control + per-condition stimulated/imputed)."""
    blocks = [ctx.control_matrix()]
    categories = ["control"] * blocks[0].shape[0]
    labels = [ctx.control_label] * blocks[0].shape[0]
    for condition in ctx.conditions:
        true = ctx.true_cells(condition)
        pred = np.asarray(per_condition[condition], dtype=np.float32)
        blocks.append(true)
        categories.extend(["stimulated"] * true.shape[0])
        labels.extend([condition] * true.shape[0])
        blocks.append(pred)
        categories.extend(["imputed"] * pred.shape[0])
        labels.extend([condition] * pred.shape[0])
    X = np.vstack(blocks).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = ctx.genes
    adata.obs["perturbation"] = labels
    adata.obs["Expcategory"] = categories
    return adata


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def finish(ctx: Context, args: argparse.Namespace, per_condition: dict[str, np.ndarray], extra: dict) -> int:
    result = assemble_result(ctx, per_condition)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.h5ad"
    result.write_h5ad(result_path)
    write_manifest(
        args.output_dir / "adapter_manifest.json",
        {
            "method": args.method,
            "adapter": args.adapter,
            "seed": args.seed,
            "train_path": str(args.train_path),
            "train_sha256": sha256(args.train_path),
            "test_path": str(args.test_path),
            "test_sha256": sha256(args.test_path),
            "n_genes": len(ctx.genes),
            "conditions": ctx.conditions,
            "n_conditions": len(ctx.conditions),
            "pred_cell_budget": {
                condition: int(np.asarray(per_condition[condition]).shape[0]) for condition in ctx.conditions
            },
            "result_h5ad": str(result_path),
            "result_sha256": sha256(result_path),
            "extra": extra,
        },
    )
    print(f"[{args.method}] wrote {result_path} ({result.shape[0]} cells, {len(ctx.conditions)} conditions)")
    return 0


# --------------------------------------------------------------------------
# adapter: Diff-AE-Perturb
# --------------------------------------------------------------------------


def adapter_diffae(ctx: Context, args: argparse.Namespace) -> int:
    import torch

    repo = Path(args.repo)
    sys.path.insert(0, str(repo / "src"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("pdb_evaluate_benchmark", repo / "scripts" / "evaluate_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    # Prefer the audited loader when available: it restores the trained activity doser, which the
    # upstream loader silently drops for guide-level checkpoints whose target vocabulary is larger
    # than their guide vocabulary (see scperturbench_diffae_loader.py).
    loader = None
    try:
        import scperturbench_diffae_loader as loader  # type: ignore
    except ImportError:
        loader = None
    loader_report: dict[str, object] = {}
    if loader is not None:
        model, mapping, loader_report = loader.load_model(Path(args.checkpoint), device)
    else:
        model, mapping = module.load_model(Path(args.checkpoint), device)
    print(f"[diffae] checkpoint={args.checkpoint} perturbations={len(mapping)} device={args.device}")
    print(f"[diffae] guide_encoder={getattr(model, 'guide_doser', None)}")
    if loader_report:
        print(
            "[diffae] doser table: checkpoint={ckpt} model={model} missing_doser_keys={missing}".format(
                ckpt=loader_report.get("doser_table_size_in_checkpoint"),
                model=loader_report.get("doser_table_size_in_model"),
                missing=loader_report.get("missing_doser_keys"),
            )
        )

    # Predicted-activity checkpoints (guide_encoder.enabled, predictor_backend="table_s8_cnn") do not
    # consume the perturbation embedding alone: the denoiser also receives an activity embedding built
    # from the Table S8 26-mer CNN features.  Without those features the harness cannot evaluate the
    # activity arm at all (build_guide_features raises), so they are threaded through explicitly and
    # recorded in the manifest.  A checkpoint without a guide encoder ignores both arguments.
    guide_encoder_cfg = getattr(getattr(model, "config", None), "guide_encoder", None)
    guide_encoder_enabled = bool(getattr(guide_encoder_cfg, "enabled", False))
    records_by_key = None
    activity_lookup: dict[str, float] = {}
    if guide_encoder_enabled:
        from perturbdiff.data import load_guide_activity_lookup
        from perturbdiff.guide_features import load_guide_feature_records

        if args.diffae_guide_activity_cnn_table is not None:
            records_by_key = load_guide_feature_records([args.diffae_guide_activity_cnn_table])
        if args.diffae_activity_table is not None:
            activity_lookup = load_guide_activity_lookup(
                args.diffae_activity_table,
                activity_col=args.diffae_activity_col,
            )
        print(
            f"[diffae] activity conditioning: source={getattr(guide_encoder_cfg, 'activity_source', '?')} "
            f"backend={getattr(guide_encoder_cfg, 'predictor_backend', '?')} "
            f"feature_dim={getattr(guide_encoder_cfg, 'feature_dim', '?')} "
            f"cnn_records={'loaded' if records_by_key is not None else 'none'} "
            f"activity_scores={len(activity_lookup)}"
        )

    # Diagnostic repair of the conditioning pathway (see
    # reports/diffae_architecture_diagnosis.md).  The trained checkpoint's per-perturbation
    # latent offsets never reach the denoiser during training and are ~50x smaller than the
    # encoder's real latent shifts, so these modes rewrite the offset table before sampling.
    #   checkpoint : leave the learned table untouched (production behaviour)
    #   rescale    : multiply it by --diffae-delta-scale
    #   recompute  : scGen-style offset recomputed from the *trained* encoder
    delta_mode = getattr(args, "diffae_delta_mode", "checkpoint")
    if delta_mode != "checkpoint":
        weight = model.perturbation.embedding.weight.data
        if delta_mode == "rescale":
            scale = float(args.diffae_delta_scale)
            weight.mul_(scale)
            print(f"[diffae] rescaled perturbation offsets by {scale} (row norms {weight.norm(dim=1)[:4].tolist()})")
        else:
            control_rows = ctx.train_labels == ctx.control_label
            control_cells = torch.from_numpy(
                to_dense(ctx.train[control_rows].X).astype(np.float32)
            ).to(weight.device)
            with torch.no_grad():
                z_control = model.encoder(control_cells).mean(dim=0)
            new_weight = torch.zeros_like(weight)
            n_recomputed = 0
            with torch.no_grad():
                for label, index in mapping.items():
                    index = int(index)
                    if index == 0 or label in {ctx.control_label, "*"}:
                        continue
                    rows = ctx.train_labels == label
                    if not rows.any():
                        continue
                    cells = torch.from_numpy(to_dense(ctx.train[rows].X).astype(np.float32)).to(weight.device)
                    new_weight[index] = model.encoder(cells).mean(dim=0) - z_control
                    n_recomputed += 1
            weight.copy_(new_weight)
            print(
                f"[diffae] recomputed {n_recomputed} offsets from the trained encoder "
                f"(median row norm {weight.norm(dim=1).median().item():.4f})"
            )

    control_dense = ctx.control_matrix()
    per_condition: dict[str, np.ndarray] = {}
    skipped_conditions: dict[str, str] = {}
    # The perturbation id must come from the checkpoint's perturbation vocabulary, which is
    # NOT necessarily the evaluated condition.  ``--condition-col perturbation_label`` holds
    # the guide identity in the GSE283614 ablation while training used ``--perturbation-col
    # guide_family``, so the vocabulary is {control, NANOG, OCT4}.  Letting the adapter resolve
    # the guide identity through a first-token guess silently drops every held-out guide whose
    # label has no underscore (``NANOG-1-5``, ``NANOG-pro1``): there is nothing to split, so the
    # guess stays ``NANOG-1-5`` and never matches ``NANOG``.  When the test object carries the
    # column the checkpoint was trained on, use it directly.
    target_col = None
    for candidate in ("guide_family", "target_gene"):
        if candidate in ctx.test.obs:
            target_col = candidate
            break
    target_by_condition: dict[str, str] = {}
    if target_col is not None:
        labels = ctx.test.obs[ctx.condition_col].astype(str)
        targets = ctx.test.obs[target_col].astype(str)
        for condition, target in zip(labels, targets):
            target_by_condition.setdefault(condition, target)
        missing_targets = [c for c in ctx.conditions if c not in target_by_condition]
        if missing_targets:
            raise SystemExit(
                f"[diffae] {target_col!r} does not cover evaluated condition(s): {missing_targets[:5]}"
            )
        print(f"[diffae] resolving perturbation ids via obs[{target_col!r}] (eval_level={args.eval_level})")
    for key, condition in enumerate(ctx.conditions):
        n_pred = ctx.n_pred_cells(condition)
        idx = ctx.sample_control_indices(condition, n_pred, key)
        control_tensor = torch.from_numpy(ctx.control_dense_subset(idx)).to(device)
        pred = module.predict_model(
            model,
            mapping,
            control_tensor,
            condition_label=condition,
            target_label=target_by_condition.get(condition, condition),
            eval_level=args.eval_level,
            records_by_key=records_by_key,
            activity_lookup=activity_lookup,
            sequence_length=args.diffae_guide_sequence_length,
            steps=args.ddim_steps,
        )
        if pred is None:
            skipped_conditions[condition] = "condition unseen by the checkpoint mapping"
            print(f"  {condition}: skipped ({skipped_conditions[condition]})")
            continue
        per_condition[condition] = np.asarray(pred, dtype=np.float32)
        print(f"  {condition}: control={control_tensor.shape[0]} pred={per_condition[condition].shape}")
    if not per_condition:
        raise SystemExit("[diffae] model produced no prediction for any condition")
    if skipped_conditions:
        print(f"[diffae] WARNING: skipped {len(skipped_conditions)} condition(s): {sorted(skipped_conditions)}")
    return finish(
        ctx,
        args,
        per_condition,
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "repo": str(repo),
            "evaluator_sha256": sha256(repo / "scripts" / "evaluate_benchmark.py"),
            "ddim_steps": args.ddim_steps,
            "eval_level": args.eval_level,
            "n_perturbations_in_checkpoint": len(mapping),
            "device": args.device,
            "delta_mode": getattr(args, "diffae_delta_mode", "checkpoint"),
            "delta_scale": getattr(args, "diffae_delta_scale", 1.0),
            "guide_encoder_enabled": guide_encoder_enabled,
            "activity_source": getattr(guide_encoder_cfg, "activity_source", None),
            "activity_predictor_backend": getattr(guide_encoder_cfg, "predictor_backend", None),
            "guide_feature_dim": getattr(guide_encoder_cfg, "feature_dim", None),
            "guide_sequence_length": args.diffae_guide_sequence_length,
            "guide_activity_cnn_table": (
                None
                if args.diffae_guide_activity_cnn_table is None
                else str(args.diffae_guide_activity_cnn_table)
            ),
            "activity_table": None if args.diffae_activity_table is None else str(args.diffae_activity_table),
            "activity_col": args.diffae_activity_col,
            "n_activity_scores_loaded": len(activity_lookup),
            "skipped_conditions": skipped_conditions,
            "checkpoint_loader_report": loader_report,
        },
    )


# --------------------------------------------------------------------------
# adapter: linear baseline
# --------------------------------------------------------------------------


def adapter_linear(ctx: Context, args: argparse.Namespace) -> int:
    """Linear baselines.

    pca_linear    port of the literature's R ``linearModel.r``: PCA-10 gene embedding of the
                  training pseudobulk matrix + ridge-regularised bilinear solve
                  (``solve_y_axb`` with A_ridge = B_ridge = pa$ridge_penalty = 0.1).
                  Requires no external resources and covers every gene.
    ridge         ``genetic_baseReg.py``: Ridge on pretrained scGPT gene embeddings of the
                  perturbed gene. Missing embeddings are filled with the mean embedding and
                  reported in the manifest.
    train_mean    literature ``trainMean``: mean pseudobulk over all training perturbations.
    control_mean  literature ``controlMean``: control pseudobulk.
    """
    from sklearn.linear_model import Ridge

    train_labels = ctx.train_labels
    control_mask = train_labels == ctx.control_label
    baseline = to_dense(ctx.train[control_mask].X).mean(axis=0).astype(np.float64)
    control_std = to_dense(ctx.train[control_mask].X).std(axis=0)
    control_std = np.where(np.isfinite(control_std), control_std, 0.0)

    perturbed = list(ctx.train_targets)
    pseudobulk = np.vstack(
        [to_dense(ctx.train[train_labels == g].X).mean(axis=0) for g in perturbed]
    ).astype(np.float64)
    detail: dict[str, object] = {"n_train_perturbations": len(perturbed)}

    if args.linear_variant == "pca_linear":
        d = args.pca_dim
        ridge = args.pca_ridge
        gene_set = set(ctx.genes)
        missing = [g for g in perturbed if g not in gene_set]
        # R: prcomp_irlba(as.matrix(pseudobulk)) -> $x is indexed by genes, so the gene
        # embedding is the gene's coordinates in condition space.
        matrix = pseudobulk.T  # (n_genes, n_conditions)
        xc = matrix - matrix.mean(axis=0, keepdims=True)
        u, s, _vt = np.linalg.svd(xc, full_matrices=False)
        gene_emb = u[:, :d] * s[:d]  # (n_genes, d) == prcomp_irlba(...)$x
        available = [g for g in perturbed if g in gene_set]
        # Some perturbation genes (here ribosomal genes such as RPS14) are not part of the
        # 1019-gene measured space, so the R script's `colnames(pert_emb)` match would be NA.
        # They fall back to the mean representation of the available training perturbations
        # and are listed explicitly in the manifest.
        fallback = np.vstack([gene_emb[ctx.genes.index(g)] for g in available]).mean(axis=0)
        train_repr = np.vstack(
            [gene_emb[ctx.genes.index(g)] if g in gene_set else fallback for g in perturbed]
        )

        def condition_repr(condition: str) -> np.ndarray:
            return gene_emb[ctx.genes.index(condition)] if condition in gene_set else fallback

        a_matrix = gene_emb  # (n_genes, d)
        b_matrix = train_repr.T  # (d, n_train_conds)
        y = (pseudobulk - baseline[None, :]).T  # (n_genes, n_train_conds)
        center = y.mean(axis=1)  # (n_genes,)
        left = np.linalg.solve(a_matrix.T @ a_matrix + ridge * np.eye(d), a_matrix.T)
        k_matrix = left @ y @ b_matrix.T @ np.linalg.solve(
            b_matrix @ b_matrix.T + ridge * np.eye(d), np.eye(d)
        )
        pert_all = np.vstack([condition_repr(condition) for condition in ctx.conditions]).T  # (d, n_conds)
        preds = a_matrix @ k_matrix @ pert_all + center[:, None] + baseline[:, None]
        means = {condition: preds[:, index] for index, condition in enumerate(ctx.conditions)}
        detail.update(
            {
                "variant": "pca_linear",
                "source": "port of manuscript2/genetic/linearModel.r::solve_y_axb",
                "pca_dim": d,
                "ridge_penalty": ridge,
                "perturbation_genes_not_measured": missing,
                "fallback": "mean representation of available training perturbations",
            }
        )
    elif args.linear_variant == "ridge":
        if args.gene_embeddings is None:
            raise SystemExit("[linear] --gene-embeddings is required for the ridge variant")
        payload = np.load(args.gene_embeddings)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        embedding_genes = [str(g) for g in payload["genes"]]
        gene_index = {gene: index for index, gene in enumerate(embedding_genes)}
        has_embedding = {
            g: (g in gene_index and bool(np.any(embeddings[gene_index[g]]))) for g in ctx.genes
        }
        missing = [g for g in perturbed if not has_embedding.get(g, False)]
        available = [g for g in perturbed if has_embedding.get(g, False)]
        mean_embedding = np.vstack([embeddings[gene_index[g]] for g in available]).mean(axis=0)
        x_train = np.vstack(
            [
                embeddings[gene_index[g]] if has_embedding.get(g, False) else mean_embedding
                for g in perturbed
            ]
        )
        model = Ridge(random_state=42)
        model.fit(x_train, pseudobulk)
        means = {}
        for condition in ctx.conditions:
            vector = (
                embeddings[gene_index[condition]]
                if has_embedding.get(condition, False)
                else mean_embedding
            )
            means[condition] = model.predict(vector[None, :])[0]
        detail.update(
            {
                "variant": "ridge",
                "source": "port of manuscript2/genetic/genetic_baseReg.py",
                "alpha": float(model.alpha),
                "gene_embeddings": str(args.gene_embeddings),
                "n_perturbations_missing_embedding": len(missing),
                "missing_embedding_genes": missing,
                "missing_embedding_policy": "mean of available perturbation-gene embeddings",
            }
        )
    elif args.linear_variant == "train_mean":
        train_mean = pseudobulk.mean(axis=0)
        means = {condition: train_mean for condition in ctx.conditions}
        detail.update({"variant": "train_mean", "source": "literature trainMean"})
    elif args.linear_variant == "control_mean":
        means = {condition: baseline for condition in ctx.conditions}
        detail.update({"variant": "control_mean", "source": "literature controlMean"})
    else:
        raise SystemExit(f"unknown linear variant {args.linear_variant}")

    per_condition: dict[str, np.ndarray] = {}
    for key, condition in enumerate(ctx.conditions):
        n_pred = ctx.n_pred_cells(condition)
        rng = np.random.default_rng(args.seed * 7919 + key)
        mean = np.asarray(means[condition], dtype=np.float64)
        samples = rng.normal(loc=mean, scale=control_std, size=(n_pred, mean.shape[0]))
        per_condition[condition] = samples.astype(np.float32)
    detail["sampling"] = "Normal(pred_mean, control_std), literature generateH5ad recipe (unclipped)"
    return finish(ctx, args, per_condition, detail)


# --------------------------------------------------------------------------
# adapter: scGen
# --------------------------------------------------------------------------


def adapter_scgen(ctx: Context, args: argparse.Namespace) -> int:
    import inspect

    import scgen
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = ctx.train.copy()
    train.obs["condition"] = train.obs[ctx.condition_col].astype(str)
    train.obs["cell_type"] = "all_cells"
    scgen.SCGEN.setup_anndata(train, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(
        train,
        n_hidden=args.scgen_n_hidden,
        n_latent=args.scgen_n_latent,
        n_layers=args.scgen_n_layers,
        dropout_rate=args.scgen_dropout_rate,
    )
    train_kwargs = {"max_epochs": args.epochs, "batch_size": args.scgen_batch_size, "early_stopping": False}
    if "use_gpu" in inspect.signature(model.train).parameters:
        train_kwargs["use_gpu"] = args.device.startswith("cuda")
    else:
        train_kwargs["accelerator"] = "gpu" if args.device.startswith("cuda") else "cpu"
        train_kwargs["devices"] = 1
    model.train(**train_kwargs)

    control_dense = ctx.control_matrix()
    per_condition: dict[str, np.ndarray] = {}
    for key, condition in enumerate(ctx.conditions):
        n_pred = ctx.n_pred_cells(condition)
        idx = ctx.sample_control_indices(condition, n_pred, key)
        ctrl = ctx.control_anndata(idx)
        ctrl.obs["condition"] = ctx.control_label
        ctrl.obs["cell_type"] = "all_cells"
        pred, _delta = model.predict(ctrl_key=ctx.control_label, stim_key=condition, adata_to_predict=ctrl)
        per_condition[condition] = to_dense(pred.X).astype(np.float32)
        print(f"  {condition}: pred={per_condition[condition].shape}")
    _ = control_dense
    return finish(
        ctx,
        args,
        per_condition,
        {
            "scgen_version": getattr(scgen, "__version__", "unknown"),
            "epochs": args.epochs,
            "batch_size": args.scgen_batch_size,
            "n_hidden": args.scgen_n_hidden,
            "n_latent": args.scgen_n_latent,
            "n_layers": args.scgen_n_layers,
            "device": args.device,
        },
    )


# --------------------------------------------------------------------------
# adapter: CPA
# --------------------------------------------------------------------------


def adapter_cpa(ctx: Context, args: argparse.Namespace) -> int:
    import torch

    from cpa import CPA

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    setup_kwargs = dict(
        perturbation_key="condition",
        control_group=ctx.control_label,
        categorical_covariate_keys=["cell_type"],
        is_count_data=False,
    )
    train = ctx.train.copy()
    train.obs["condition"] = train.obs[ctx.condition_col].astype(str)
    train.obs["cell_type"] = "all_cells"
    train.X = to_dense(train.X).astype(np.float32)
    CPA.setup_anndata(train, **setup_kwargs)

    model = CPA(train, recon_loss="gauss", n_latent=args.n_latent, seed=args.seed)
    model.train(max_epochs=args.epochs, batch_size=args.batch_size, plan_kwargs={"lr": args.lr}, check_val_every_n_epoch=1)

    control_dense = ctx.control_matrix()
    per_condition: dict[str, np.ndarray] = {}
    for key, condition in enumerate(ctx.conditions):
        n_pred = ctx.n_pred_cells(condition)
        idx = ctx.sample_control_indices(condition, n_pred, key)
        ctrl = ad.AnnData(X=ctx.control_dense_subset(idx).copy())
        ctrl.var_names = ctx.genes
        ctrl.obs["condition"] = condition
        ctrl.obs["cell_type"] = "all_cells"
        ctrl.obsm["X_true"] = ctrl.X.copy()
        CPA.setup_anndata(ctrl, **setup_kwargs)
        model.predict(ctrl, n_samples=1, return_mean=True)
        stored = ctrl.obsm["CPA_pred"]
        per_condition[condition] = np.asarray(stored, dtype=np.float32).reshape(ctrl.n_obs, len(ctx.genes))
        print(f"  {condition}: pred={per_condition[condition].shape}")
    return finish(
        ctx,
        args,
        per_condition,
        {
            "cpa_version": getattr(__import__("cpa"), "version", "unknown"),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "recon_loss": "gauss",
            "n_latent": args.n_latent,
            "device": "cpu",
            "note": "predictions read from adata.obsm['CPA_pred'] after setting the target condition on control cells",
        },
    )


# --------------------------------------------------------------------------
# adapter: GEARS
# --------------------------------------------------------------------------


def adapter_gears(ctx: Context, args: argparse.Namespace) -> int:
    import shutil

    import torch

    from gears import GEARS, PertData
    from gears.inference import evaluate

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    workdir = Path(args.gears_data_root)
    workdir.mkdir(parents=True, exist_ok=True)
    # PertData downloads gene2go_all.pkl / essential_all_data_pert_genes.pkl /
    # go_essential_all.tar.gz on first use. The image already ships them with the
    # scPerturBench Papalexi dataset, so seed a stable cache directory to stay offline
    # (the dataverse download takes ~40 minutes here).
    seeded = []
    source_dir = Path(args.gears_resource_dir)
    for name in ("gene2go_all.pkl", "essential_all_data_pert_genes.pkl", "go_essential_all.tar.gz"):
        target = workdir / name
        source = source_dir / name
        if not target.exists() and source.exists():
            shutil.copy(source, target)
            seeded.append(name)
    go_dir = workdir / "go_essential_all"
    if not go_dir.exists() and (source_dir / "go_essential_all").is_dir():
        shutil.copytree(source_dir / "go_essential_all", go_dir)
        seeded.append("go_essential_all/")
    print(f"[gears] data root {workdir}; seeded: {seeded or 'already present'}")

    def gears_condition(label: str) -> str:
        return "ctrl" if label == ctx.control_label else f"{label}+ctrl"

    train = ctx.train.copy()
    train.obs["condition"] = [gears_condition(v) for v in ctx.train_labels]
    train.obs["cell_type"] = "all_cells"
    train.var["gene_name"] = list(train.var_names)
    train.X = sparse.csr_matrix(to_dense(train.X).astype(np.float32))
    # Same fix-up the literature wrapper applies before PertData.new_data_process.
    train.uns["log1p"] = {"base": None}
    if "counts" in train.layers:
        del train.layers["counts"]

    pert_data = PertData(str(workdir))
    processed = workdir / "gse132080" / "perturb_processed.h5ad"
    if processed.exists():
        print(f"[gears] reusing cached processed dataset {processed}")
    else:
        pert_data.new_data_process(dataset_name="gse132080", adata=train)
    pert_data.load(data_path=str(workdir / "gse132080"))
    pert_data.prepare_split(split="simulation", seed=args.seed, train_gene_set_size=0.8)
    pert_data.split = "simulation"
    pert_data.set2conditions = {
        "train": ["ctrl"] + [gears_condition(v) for v in ctx.train_targets],
        "val": ["ctrl"] + [gears_condition(v) for v in ctx.train_targets],
        "test": ["ctrl"] + [gears_condition(v) for v in ctx.conditions],
    }
    print(f"[gears] custom split: train={len(pert_data.set2conditions['train'])} test={len(pert_data.set2conditions['test'])}")
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    model = GEARS(pert_data, device="cpu")
    model.model_initialize(hidden_size=args.hidden_size)
    model.train(epochs=args.epochs)

    results = evaluate(model.dataloader["test_loader"], model.best_model, model.config["uncertainty"], model.device)
    pert_cat = np.asarray(results["pert_cat"])
    pred = np.asarray(results["pred"], dtype=np.float32)
    per_condition: dict[str, np.ndarray] = {}
    for condition in ctx.conditions:
        key = gears_condition(condition)
        mask = pert_cat == key
        if not mask.any():
            raise SystemExit(f"[gears] no test predictions for {key}")
        per_condition[condition] = pred[mask]
        print(f"  {condition}: pred={per_condition[condition].shape}")
    return finish(
        ctx,
        args,
        per_condition,
        {
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "device": "cpu",
            "split": "custom set2conditions over the GSE132080 unseen-guide split",
            "gears_resource_dir": str(source_dir),
            "gears_data_root": str(workdir),
            "seeded_resources": seeded,
            "prediction_source": "gears.inference.evaluate(test_loader) -> results['pred'] (cell-level)",
        },
    )


# --------------------------------------------------------------------------
# adapter: Squidiff
# --------------------------------------------------------------------------


def adapter_squidiff(ctx: Context, args: argparse.Namespace) -> int:
    """Train Squidiff on the train split, then sample cells for each test condition.

    Squidiff's defaults run with ``class_cond=False`` and ``use_encoder=True``: the model is a
    DiffAE-style latent diffusion where the semantic vector ``z_sem`` comes from an encoder over
    the clean cell.  Sampling therefore conditions on ``z_mod`` supplied at inference time, which
    is exactly how ``squidiff_on_genepertb.ipynb`` generates for held-out groups.  We follow the
    same route: for each test target gene, ``z_sem`` is the encoder output on that gene's
    *training* cells (all evaluated targets are present in training), then DDIM sampling with
    ``model_kwargs={'z_mod': z_sem}``.

    The official ``TrainLoop`` is reused directly so the training procedure (uniform timestep
    sampling, EMA, LR anneal, microbatching) is unchanged; only the step budget and the DDIM
    respacing are set explicitly and recorded in the manifest.
    """
    import torch

    # Prefer the checkout the user pointed at (mounted read-only at /root/Squidiff) over the
    # package installed in the conda env, so the run is attributable to a concrete source tree.
    repo_root = Path(args.squidiff_repo)
    if (repo_root / "Squidiff" / "script_util.py").exists():
        sys.path.insert(0, str(repo_root))
        for name in list(sys.modules):
            if name == "Squidiff" or name.startswith("Squidiff."):
                del sys.modules[name]
        print(f"[squidiff] using source tree {repo_root}")
    import Squidiff as squidiff_pkg

    from Squidiff import dist_util, logger
    from Squidiff.resample import create_named_schedule_sampler
    from Squidiff.script_util import (
        args_to_dict,
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )
    from Squidiff.scrna_datasets import prepared_data
    from Squidiff.train_util import TrainLoop

    print(f"[squidiff] package={squidiff_pkg.__file__}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment = args.output_dir / "squidiff"
    (experiment / "logs").mkdir(parents=True, exist_ok=True)
    (experiment / "checkpoint").mkdir(parents=True, exist_ok=True)

    group_map = {ctx.control_label: 0}
    for index, gene in enumerate(ctx.train_targets, start=1):
        group_map[gene] = index
    train = ctx.train.copy()
    if args.squidiff_input == "counts":
        if "counts" not in ctx.train.layers:
            raise SystemExit("[squidiff] --squidiff-input counts requires a 'counts' layer")
        train.X = to_dense(ctx.train.layers["counts"]).astype(np.float32)
        print(f"[squidiff] training on raw counts (max={train.X.max():.1f}, mean={train.X.mean():.3f})")
    else:
        train.X = to_dense(train.X).astype(np.float32)
    train.obs["Group"] = np.asarray([group_map.get(label, 0) for label in ctx.train_labels], dtype=np.float32)
    train_path = experiment / "train.h5ad"
    train.write_h5ad(train_path)
    (experiment / "group_map.json").write_text(json.dumps(group_map, indent=2), encoding="utf-8")
    missing_groups = [c for c in ctx.conditions if c not in group_map or group_map[c] == 0]
    if missing_groups:
        raise SystemExit(f"[squidiff] evaluated conditions missing from the training group map: {missing_groups}")

    model_args = dict(model_and_diffusion_defaults())
    model_args.update(
        {
            "gene_size": len(ctx.genes),
            "output_dim": len(ctx.genes),
            "num_layers": args.num_layers,
            "class_cond": False,
            "use_encoder": True,
            "use_drug_structure": False,
            "diffusion_steps": args.diffusion_steps,
            "timestep_respacing": args.timestep_respacing,
            "dropout": 0.0,
        }
    )
    logger.configure(dir=str(experiment / "logs"))
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    device = dist_util.dev()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[squidiff] model params={n_params:,} device={device} steps={args.steps} respacing={args.timestep_respacing}")

    data = prepared_data(
        data_dir=str(train_path),
        control_data_dir=None,
        batch_size=args.squidiff_batch_size,
        use_drug_structure=False,
        comb_num=1,
    )
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.squidiff_batch_size,
        microbatch=-1,
        lr=args.squidiff_lr,
        ema_rate="0.9999",
        log_interval=args.log_interval,
        save_interval=max(args.steps, 1),
        resume_checkpoint=str(experiment / "checkpoint"),
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=0.0,
        lr_anneal_steps=args.steps,
        use_drug_structure=False,
        comb_num=1,
    )
    started = time.time()
    loop.run_loop()
    train_seconds = time.time() - started
    print(f"[squidiff] trained {args.steps} steps in {train_seconds:.1f}s ({train_seconds / max(args.steps, 1):.3f}s/step)")

    model.eval()
    per_condition: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key, condition in enumerate(ctx.conditions):
            n_pred = ctx.n_pred_cells(condition)
            mask = ctx.train_labels == condition
            cells = to_dense(ctx.train[mask].X).astype(np.float32)
            rng = np.random.default_rng(args.seed * 104729 + key)
            pick = rng.choice(cells.shape[0], n_pred, replace=n_pred > cells.shape[0])
            x_start = torch.from_numpy(cells[pick]).to(device)
            z_sem = model.encoder(x_start, label=None, drug_dose=None)
            samples = diffusion.ddim_sample_loop(
                model,
                shape=(n_pred, len(ctx.genes)),
                model_kwargs={"z_mod": z_sem},
                device=device,
            )
            generated = samples.detach().cpu().numpy().astype(np.float32)
            if args.squidiff_input == "counts":
                # Map generated counts back to the shared log-normalised scale with the same
                # transform the benchmark uses for every other method (target_sum=1e4, log1p).
                generated = np.clip(generated, 0.0, None)
                library = generated.sum(axis=1, keepdims=True)
                library[library <= 0] = 1.0
                generated = np.log1p(generated / library * 1e4)
            per_condition[condition] = generated
            print(f"  {condition}: z_sem={tuple(z_sem.shape)} pred={per_condition[condition].shape}")
    return finish(
        ctx,
        args,
        per_condition,
        {
            "train_steps": args.steps,
            "train_seconds": round(train_seconds, 1),
            "seconds_per_step": round(train_seconds / max(args.steps, 1), 4),
            "batch_size": args.squidiff_batch_size,
            "lr": args.squidiff_lr,
            "num_layers": args.num_layers,
            "n_params": n_params,
            "diffusion_steps": args.diffusion_steps,
            "timestep_respacing": args.timestep_respacing,
            "conditioning": "z_mod from model.encoder over the target gene's training cells",
            "input_scale": args.squidiff_input,
            "class_cond": False,
            "checkpoint": str(experiment / "checkpoint" / "model.pt"),
            "device": str(device),
        },
    )


# --------------------------------------------------------------------------
# adapter: Squidiff on its own Squidiff_reproducibility preprocessing
# --------------------------------------------------------------------------


def remap_panel(pred: np.ndarray, panel: list[str], benchmark_genes: list[str], fill: np.ndarray):
    """Map predictions from Squidiff's own gene panel onto the shared evaluation panel."""
    index = {gene: position for position, gene in enumerate(panel)}
    out = np.empty((pred.shape[0], len(benchmark_genes)), dtype=np.float32)
    missing: list[str] = []
    for position, gene in enumerate(benchmark_genes):
        source = index.get(gene)
        if source is None:
            missing.append(gene)
            out[:, position] = fill[position]
        else:
            out[:, position] = pred[:, source]
    return out, missing


def adapter_squidiff_repro(ctx: Context, args: argparse.Namespace) -> int:
    """Squidiff trained and sampled exactly as ``Squidiff_reproducibility`` does.

    Three things differ from :func:`adapter_squidiff`:

    1. the input matrix comes from ``scperturbench_squidiff_prep.py``, i.e. the upstream
       ``prep_pertb_data.ipynb`` recipe (labelled HVG selection + forced genes of interest);
    2. training keeps the upstream schedule: ``lr_anneal_steps=1e5`` and no timestep
       respacing, so sampling runs the full 1000-step DDIM chain;
    3. sampling goes through the upstream ``sample_squidiff.sampler`` object, which reloads
       ``model.pt`` from disk and calls ``sampler.pred`` with ``z_mod``.

    Predictions are then mapped back onto the shared 1019-gene evaluation panel; genes that
    Squidiff's own panel does not contain are filled with the control pseudobulk and listed
    in the manifest.
    """
    import torch

    repo_root = Path(args.squidiff_repo)
    if (repo_root / "sample_squidiff.py").exists():
        sys.path.insert(0, str(repo_root))
        for name in list(sys.modules):
            if name == "Squidiff" or name.startswith("Squidiff.") or name == "sample_squidiff":
                del sys.modules[name]
    import sample_squidiff
    from Squidiff import dist_util, logger
    from Squidiff.resample import create_named_schedule_sampler
    from Squidiff.script_util import (
        args_to_dict,
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )
    from Squidiff.scrna_datasets import prepared_data
    from Squidiff.train_util import TrainLoop

    from scperturbench_squidiff_sampling import sample_with_policy

    data_dir = Path(args.squidiff_data_dir)
    native_train_path = data_dir / "squidiff_train.h5ad"
    panel_payload = json.loads((data_dir / "gene_panel.json").read_text(encoding="utf-8"))
    panel = [str(g) for g in panel_payload["genes"]]
    native_train = ad.read_h5ad(native_train_path)
    if [str(g) for g in native_train.var_names] != panel:
        raise SystemExit("[squidiff_repro] native train gene order disagrees with gene_panel.json")
    n_genes = len(panel)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    reuse_dir = Path(args.squidiff_train_dir) if args.squidiff_train_dir else None
    experiment = reuse_dir if reuse_dir is not None else args.output_dir / "squidiff"
    (experiment / "logs").mkdir(parents=True, exist_ok=True)
    (experiment / "checkpoint").mkdir(parents=True, exist_ok=True)
    checkpoint = experiment / "checkpoint" / "model.pt"
    train_seconds = 0.0
    n_params = 0

    if reuse_dir is None:
        model_args = dict(model_and_diffusion_defaults())
        model_args.update(
            {
                "gene_size": n_genes,
                "output_dim": n_genes,
                "num_layers": args.num_layers,
                "class_cond": False,
                "use_encoder": True,
                "use_drug_structure": False,
                "diffusion_steps": args.diffusion_steps,
                "timestep_respacing": args.squidiff_respacing,
                "dropout": 0.0,
            }
        )
        logger.configure(dir=str(experiment / "logs"))
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(model_args, model_and_diffusion_defaults().keys())
        )
        device = dist_util.dev()
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[squidiff_repro] native_train={native_train.shape} genes={n_genes} params={n_params:,} "
            f"device={device} steps={args.squidiff_steps} respacing={args.squidiff_respacing!r} "
            f"lr_anneal_steps={args.squidiff_steps}"
        )

        data = prepared_data(
            data_dir=str(native_train_path),
            control_data_dir=None,
            batch_size=args.squidiff_batch_size,
            use_drug_structure=False,
            comb_num=1,
        )
        schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
        loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.squidiff_batch_size,
            microbatch=-1,
            lr=args.squidiff_lr,
            ema_rate="0.9999",
            log_interval=args.log_interval,
            save_interval=max(args.squidiff_steps, 1),
            resume_checkpoint=str(experiment / "checkpoint"),
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=schedule_sampler,
            weight_decay=0.0,
            lr_anneal_steps=args.squidiff_steps,
            use_drug_structure=False,
            comb_num=1,
        )
        started = time.time()
        loop.run_loop()
        train_seconds = time.time() - started
        print(
            f"[squidiff_repro] trained {args.squidiff_steps} steps in {train_seconds:.1f}s "
            f"({train_seconds / max(args.squidiff_steps, 1):.3f}s/step)"
        )
        del loop, data, model, diffusion
    else:
        print(f"[squidiff_repro] reusing checkpoint from {checkpoint} (training skipped)")

    if not checkpoint.exists():
        raise SystemExit(f"[squidiff_repro] expected checkpoint at {checkpoint}")
    # Reload through the upstream sampler class: this is exactly the object the notebooks use.
    sampler = sample_squidiff.sampler(
        model_path=str(checkpoint),
        gene_size=n_genes,
        output_dim=n_genes,
        use_drug_structure=False,
    )
    device = next(sampler.model.parameters()).device
    if not n_params:
        n_params = sum(p.numel() for p in sampler.model.parameters())

    native_labels = native_train.obs[args.condition_col].astype(str).to_numpy()
    native_x = to_dense(native_train.X).astype(np.float32)
    fill = ctx.control_matrix().mean(axis=0).astype(np.float32)
    per_condition: dict[str, np.ndarray] = {}
    missing_genes: list[str] = []
    with torch.no_grad():
        for key, condition in enumerate(ctx.conditions):
            n_pred = ctx.n_pred_cells(condition)
            mask = native_labels == condition
            if not mask.any():
                raise SystemExit(f"[squidiff_repro] condition {condition} absent from the Squidiff panel train set")
            cells = native_x[mask]
            rng = np.random.default_rng(args.seed * 104729 + key)
            pick = rng.choice(cells.shape[0], n_pred, replace=n_pred > cells.shape[0])
            x_start = torch.from_numpy(cells[pick]).to(device)
            z_sem = sampler.model.encoder(x_start, label=None, drug_dose=None)
            samples = sample_with_policy(sampler, z_sem, n_genes, args.squidiff_xstart_clamp)
            generated = samples.detach().cpu().numpy().astype(np.float32)
            mapped, missing = remap_panel(generated, panel, ctx.genes, fill)
            for gene in missing:
                if gene not in missing_genes:
                    missing_genes.append(gene)
            per_condition[condition] = mapped
            print(
                f"  {condition}: z_sem={tuple(z_sem.shape)} native={generated.shape} "
                f"mean={generated.mean():.4f} max={generated.max():.3f} "
                f"mapped_mean={mapped.mean():.4f}"
            )
    return finish(
        ctx,
        args,
        per_condition,
        {
            "adapter": "squidiff_repro",
            "preprocessing": "Squidiff_reproducibility prep_* family ported to GSE132080",
            "preprocessing_script": "scperturbench_squidiff_prep.py",
            "native_train_path": str(native_train_path),
            "native_train_sha256": sha256(native_train_path),
            "squidiff_panel_size": n_genes,
            "squidiff_panel_n_hvg": int(len(panel_payload.get("hvg_only_genes", []))),
            "squidiff_forced_genes": panel_payload.get("forced_target_genes")
            or panel_payload.get("target_genes", []),
            "squidiff_panel_recipe": panel_payload.get("recipe"),
            "benchmark_genes_missing_from_panel": missing_genes,
            "missing_gene_fill": "control pseudobulk mean (same value the metric's checkNan would use)",
            "train_steps": args.squidiff_steps,
            "train_seconds": round(train_seconds, 1),
            "seconds_per_step": round(train_seconds / max(args.squidiff_steps, 1), 4),
            "training_skipped": reuse_dir is not None,
            "train_dir": str(experiment),
            "batch_size": args.squidiff_batch_size,
            "lr": args.squidiff_lr,
            "num_layers": args.num_layers,
            "n_params": n_params,
            "diffusion_steps": args.diffusion_steps,
            "timestep_respacing": args.squidiff_respacing or "official (all 1000 steps)",
            "clip_denoised": bool(args.squidiff_xstart_clamp != "upstream"),
            "xstart_clamp_policy": args.squidiff_xstart_clamp,
            "conditioning": "z_mod from sample_squidiff model.encoder over the condition's training cells",
            "sampling_api": "sample_squidiff.sampler.pred",
            "class_cond": False,
            "checkpoint": str(checkpoint),
            "device": str(device),
        },
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one method adapter for the GSE132080 six-metric benchmark.")
    parser.add_argument("--adapter", required=True, choices=["diffae", "linear", "scgen", "cpa", "gears", "squidiff", "squidiff_repro", "primeflow"])
    parser.add_argument("--method", default=None, help="method name used in outputs (defaults to the adapter name)")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-col", default="perturbation_label")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-test-cells", type=int, default=20)
    parser.add_argument("--max-pred-cells", type=int, default=2000)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--threads", type=int, default=10)

    # diffae
    parser.add_argument("--repo", type=Path, default=Path("/root/Diff-AE-Perturb"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eval-level", default="target", choices=["target", "guide"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--diffae-guide-activity-cnn-table",
        type=Path,
        default=None,
        help="Table_S8-style 26-mer guide activity CNN table required by predicted-activity checkpoints",
    )
    parser.add_argument(
        "--diffae-activity-table",
        type=Path,
        default=None,
        help="Optional guide activity supervision table (only used by direct-activity checkpoints)",
    )
    parser.add_argument("--diffae-activity-col", default="relative_activity_day10")
    parser.add_argument("--diffae-guide-sequence-length", type=int, default=26)
    parser.add_argument(
        "--diffae-delta-mode",
        default="checkpoint",
        choices=["checkpoint", "rescale", "recompute"],
        help="diagnostic repair of the per-perturbation latent offsets (see reports/)",
    )
    parser.add_argument("--diffae-delta-scale", type=float, default=50.0)

    # linear
    parser.add_argument("--gene-embeddings", type=Path, default=None)
    parser.add_argument(
        "--linear-variant",
        default="pca_linear",
        choices=["pca_linear", "ridge", "train_mean", "control_mean"],
    )
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument("--pca-ridge", type=float, default=0.1)

    # scgen
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--scgen-batch-size", type=int, default=128)
    parser.add_argument("--scgen-n-hidden", type=int, default=800)
    parser.add_argument("--scgen-n-latent", type=int, default=100)
    parser.add_argument("--scgen-n-layers", type=int, default=2)
    parser.add_argument("--scgen-dropout-rate", type=float, default=0.2)

    # cpa
    parser.add_argument("--n-latent", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    # gears
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument(
        "--gears-resource-dir",
        type=Path,
        default=Path("/home/project/Pertb_benchmark/DataSet2/Papalexi/hvg5000/GEARS/data"),
    )
    parser.add_argument(
        "--gears-data-root",
        type=Path,
        default=Path("/root/perturbench/.scperturbench/inputs/gears_cache"),
    )
    parser.add_argument("--skip-split", action="store_true", default=False)

    # squidiff
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--timestep-respacing", default="ddim50")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--squidiff-lr", type=float, default=1e-4)
    parser.add_argument("--squidiff-batch-size", type=int, default=64)
    parser.add_argument("--squidiff-repo", default="/root/Squidiff")
    parser.add_argument(
        "--squidiff-input",
        default="lognorm",
        choices=["lognorm", "counts"],
        help="scale of the matrix Squidiff is trained on (README documents a count matrix)",
    )

    # squidiff_repro (Squidiff_reproducibility preprocessing + upstream defaults)
    parser.add_argument(
        "--squidiff-data-dir",
        type=Path,
        default=Path("/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro"),
        help="output directory of scperturbench_squidiff_prep.py",
    )
    parser.add_argument("--squidiff-steps", type=int, default=100000, help="upstream lr_anneal_steps default")
    parser.add_argument(
        "--squidiff-train-dir",
        type=Path,
        default=None,
        help="reuse an existing squidiff experiment directory (skips training, samples its model.pt)",
    )
    parser.add_argument(
        "--squidiff-respacing",
        default="",
        help="upstream default is '' (full 1000-step DDIM); pass ddim50 to respace",
    )
    parser.add_argument(
        "--squidiff-xstart-clamp",
        default="upstream",
        choices=["upstream", "none", "neg1_1"],
        help=(
            "process_xstart policy for pred_xstart: 'upstream' keeps Squidiff's unconditional "
            "clamp(0,), 'neg1_1' uses Squidiff's own clip_denoised=True branch, 'none' reverts "
            "to guided-diffusion's unclamped behaviour (see scperturbench_squidiff_sampling.py)"
        ),
    )

    # PrimeFlow (official Norman19 U-Net recipe adapted to this benchmark)
    parser.add_argument("--primeflow-epochs", type=int, default=100)
    parser.add_argument("--primeflow-batch-size", type=int, default=100)
    parser.add_argument("--primeflow-euler-steps", type=int, default=100)
    parser.add_argument("--primeflow-device", default="cuda")
    parser.add_argument("--primeflow-source-ref", default="443295258e364e394b60de27a861d8d1cf3b1fa6")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.method is None:
        args.method = args.adapter
    started = time.time()
    ctx = Context(args)
    print(
        f"[{args.adapter}] train={ctx.train.shape} test={ctx.test.shape} "
        f"conditions={len(ctx.conditions)} control={ctx.control_indices.size} seed={args.seed}",
        flush=True,
    )
    handlers = {
        "diffae": adapter_diffae,
        "linear": adapter_linear,
        "scgen": adapter_scgen,
        "cpa": adapter_cpa,
        "gears": adapter_gears,
        "squidiff": adapter_squidiff,
        "squidiff_repro": adapter_squidiff_repro,
    }
    if args.adapter == "primeflow":
        from scperturbench_primeflow import adapter_primeflow

        handler = adapter_primeflow
    else:
        handler = handlers[args.adapter]
    code = handler(ctx, args)
    print(f"[{args.adapter}] done in {time.time() - started:.1f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
