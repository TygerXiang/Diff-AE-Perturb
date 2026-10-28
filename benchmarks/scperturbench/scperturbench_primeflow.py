"""PrimeFlow adapter for the shared GSE132080 benchmark contract."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scperturbench_adapters import Context, finish, sha256


PRIMEFLOW_EXPERIMENT = "official_norman19_gaussian_source_unet"


def _write_hydra_support_config(workdir: Path) -> tuple[Path, Path]:
    """Supply the configs omitted from the upstream PrimeFlow dependencies."""
    config_root = workdir / "configs"
    configs = {
        "trainer/default.yaml": """_target_: lightning.pytorch.Trainer
min_epochs: 1
max_epochs: 300
accelerator: auto
devices: 1
strategy: auto
precision: 32-true
deterministic: true
use_distributed_sampler: false
log_every_n_steps: 50
enable_checkpointing: true
""",
        "callbacks/default.yaml": """early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: train_loss
  mode: min
  patience: 50
model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  monitor: train_loss
  mode: min
  save_top_k: 1
  save_last: true
  dirpath: ${paths.output_dir}/checkpoints
  filename: epoch_{epoch:03d}
""",
        "logger/default.yaml": "{}\n",
        "hydra/default.yaml": """# @package hydra
run:
  dir: ${hydra:runtime.output_dir}
sweep:
  dir: ${hydra:runtime.output_dir}
  subdir: ${hydra.job.num}
output_subdir: .hydra
""",
        "splitter/saved_split.yaml": """split_path: ???
""",
        "evaluation/default.yaml": """split_value_to_evaluate: val\ntest_data_iter_factory:\n  _target_: perturbench.data.datasets.CounterfactualWithReference.from_anndata\n  _partial_: true\n  max_control_cells_per_covariate: 1000\n  perturbation_key: ${data.data_iter_factory.perturbation_key}\n  perturbation_control_value: ${data.data_iter_factory.perturbation_control_value}\n  perturbation_combination_delimiter: ${data.data_iter_factory.perturbation_combination_delimiter}\n  covariate_keys: ${data.data_iter_factory.covariate_keys}\n  embedding_key: ${data.data_iter_factory.embedding_key}\n  feature_filter_path: ${data.data_iter_factory.feature_filter_path}\n  seed: ${seed}\nchunk_size: 400\nevaluation_pipelines: []\nsave_evaluation: false\nsave_dir: ${paths.output_dir}/evaluation/\nuse_synthetic_controls: false\nuse_control_variance: false\nrun_cell_eval: false\n""",
        "evaluation/h5_all.yaml": """split_value_to_evaluate: val\ntest_data_iter_factory:\n  _target_: perturbench.data.datasets.CounterfactualWithReference.from_anndata\n  _partial_: true\n  max_control_cells_per_covariate: 1000\n  perturbation_key: ${data.data_iter_factory.perturbation_key}\n  perturbation_control_value: ${data.data_iter_factory.perturbation_control_value}\n  perturbation_combination_delimiter: ${data.data_iter_factory.perturbation_combination_delimiter}\n  covariate_keys: ${data.data_iter_factory.covariate_keys}\n  embedding_key: ${data.data_iter_factory.embedding_key}\n  feature_filter_path: ${data.data_iter_factory.feature_filter_path}\n  seed: ${seed}\nchunk_size: 400\nevaluation_pipelines: []\nsave_evaluation: false\nsave_dir: ${paths.output_dir}/evaluation/\nuse_synthetic_controls: false\nuse_control_variance: false\nrun_cell_eval: false\n""",
    }
    for relative, contents in configs.items():
        path = config_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        if not path.is_file():
            raise OSError(f"Failed to materialize PrimeFlow Hydra config: {path}")
    return config_root


def _patch_upstream_config() -> None:
    """Repair two missing package YAML groups from the pinned PrimeFlow wheel."""
    import importlib.util

    spec = importlib.util.find_spec("primeflow")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Cannot locate installed PrimeFlow package")
    config = Path(next(iter(spec.submodule_search_locations))) / "configs" / "data" / "vcc_annbacked_no_controls.yaml"
    source = config.read_text(encoding="utf-8")
    source = source.replace("  - splitter: saved_split\n", "")
    source = source.replace("  - evaluation: default\n", "  - evaluation: h5_all\n")
    config.write_text(source, encoding="utf-8")

    from perturbench.data import utils as perturbench_utils

    parser = Path(perturbench_utils.__file__)
    parser_source = parser.read_text(encoding="utf-8")
    marker = "        uniques = combinations.cat.categories.values\n"
    if "perturbation_counts = {perturbation: int((combinations == perturbation).sum())" not in parser_source:
        parser_source = parser_source.replace(
            marker,
            marker + "        perturbation_counts = {perturbation: int((combinations == perturbation).sum()) for perturbation in uniques}\n",
            1,
        )
        parser.write_text(parser_source, encoding="utf-8")


def _run_command(command: list[str], log_path: Path, label: str) -> None:
    print(f"[primeflow] {label}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        raise RuntimeError(
            f"PrimeFlow {label} exited with status {result.returncode}; "
            f"see {log_path}\n" + "\n".join(tail)
        )


def _write_primeflow_inputs(ctx: Context, workdir: Path, seed: int) -> tuple[Path, Path, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    train = ctx.train.copy()
    train.obs["condition"] = train.obs[ctx.condition_col].astype(str).to_numpy()
    # PrimeFlow's Gaussian-source pipeline encodes perturbation labels only. Keep
    # controls out of this training table so the parser represents them as empty
    # perturbation lists instead of feeding a control token to MultiLabelEncode.
    train = train[train.obs["condition"] != ctx.control_label].copy()

    rng = np.random.default_rng(seed)
    split = pd.Series(index=train.obs_names, dtype=object)
    labels = train.obs["condition"].astype(str).to_numpy()
    for condition in sorted(set(labels)):
        indices = np.flatnonzero(labels == condition)
        shuffled = rng.permutation(indices)
        n_val = max(1, int(round(len(indices) * 0.1))) if len(indices) > 1 else 0
        split.iloc[shuffled] = "train"
        if n_val:
            split.iloc[shuffled[:n_val]] = "val"
    split_path = workdir / "split.csv"
    split.to_csv(split_path, header=False)

    train_path = workdir / "train.h5ad"
    train.write_h5ad(train_path)

    conditions = [
        condition
        for condition in ctx.conditions
        if condition in set(labels) and condition != ctx.control_label
    ]
    if not conditions:
        raise RuntimeError("No evaluated PrimeFlow targets are present in benchmark training data")
    prediction_path = workdir / "prediction_conditions.csv"
    pd.DataFrame({"condition": conditions}).to_csv(prediction_path, index=False)
    return train_path, split_path, prediction_path


def _prediction_arrays(prediction_dir: Path, conditions: list[str], ctx: Context) -> dict[str, np.ndarray]:
    files = sorted(prediction_dir.glob("prediction_rank*_chunk_*.h5ad"))
    if not files:
        raise RuntimeError(f"PrimeFlow produced no prediction files under {prediction_dir}")

    generated: dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    for path in files:
        prediction = ad.read_h5ad(path)
        if list(map(str, prediction.var_names)) != ctx.genes:
            raise RuntimeError(f"PrimeFlow gene order differs from benchmark panel in {path}")
        if "condition" not in prediction.obs:
            raise RuntimeError(f"PrimeFlow prediction metadata lacks 'condition': {path}")
        values = prediction.X.toarray() if hasattr(prediction.X, "toarray") else np.asarray(prediction.X)
        for condition in conditions:
            mask = prediction.obs["condition"].astype(str).to_numpy() == condition
            if mask.any():
                generated[condition].append(np.asarray(values[mask], dtype=np.float32))

    per_condition = {}
    for condition in conditions:
        chunks = generated[condition]
        if not chunks:
            raise RuntimeError(f"PrimeFlow generated no cells for evaluated condition {condition}")
        prediction = np.vstack(chunks)
        budget = ctx.n_pred_cells(condition)
        if prediction.shape[0] < budget:
            raise RuntimeError(
                f"PrimeFlow generated {prediction.shape[0]} cells for {condition}, below shared budget {budget}"
            )
        per_condition[condition] = prediction[:budget]
    return per_condition


def adapter_primeflow(ctx: Context, args) -> int:
    """Train official PrimeFlow U-Net and sample held-out-guide target conditions."""
    if args.primeflow_device != "cuda":
        raise RuntimeError("The registered PrimeFlow benchmark is configured for CUDA execution")
    if args.primeflow_epochs < 1 or args.primeflow_batch_size < 1 or args.primeflow_euler_steps < 2:
        raise ValueError("PrimeFlow epochs/batch size must be positive and Euler steps must be >= 2")

    workdir = args.output_dir / "primeflow"
    _patch_upstream_config()
    train_path, split_path, prediction_path = _write_primeflow_inputs(ctx, workdir / "inputs", args.seed)
    train_run = workdir / "train_run"
    prediction_run = workdir / "prediction_run"
    prediction_dir = prediction_run / "predictions"
    train_run.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    hydra_support = _write_hydra_support_config(workdir)

    common = [
        "data=vcc_annbacked_no_controls",
        f"seed={args.seed}",
        f"data.data.filename={train_path}",
        "data.data_iter_factory.perturbation_key=condition",
        "data.data_iter_factory.perturbation_control_value=__no_control_rows__",
        "data.data_iter_factory.perturbation_combination_delimiter=null",
        "data.data_iter_factory.covariate_keys=[]",
        "data.data_iter_factory.feature_filter_path=null",
        f"data.splitter.split_path={split_path}",
        "data.transform.conf.flow_matching_util.sigma=0.0",
        "+data.loader.ddp_mode=false",
        "data.loader.num_workers=0",
        "model.inject_covariates=false",
        "model.lr=5e-5",
        "model.wd=0.0",
        "+model.grad_accum_batches=2",
        "+model.grad_clip_threshold=5.0",
        "+model.dynamics_model._target_=primeflow.modelcore.nn.unet.UNetModel",
        "+model.dynamics_model._partial_=true",
        "+model.dynamics_model.hidden_dim=256",
        "+model.dynamics_model.num_res_blocks=2",
        "+model.dynamics_model.dropout=0.1",
        "+model.dynamics_model.channel_mult=[1,1,2,2,4]",
        "+model.dynamics_model.conv_resample=true",
        "+model.dynamics_model.num_heads=8",
        "+model.dynamics_model.attention_resolutions=[16]",
        "+model.dynamics_model.n_layers_gene_expression=2",
        "+model.dynamics_model.n_layers_condition=2",
        "+model.dynamics_model.n_layers_time=2",
        "model/lr_scheduler=reduce_on_plateau",
        "model.standard_gaussian_source=true",
        "+model.prob_unconditional=0.2",
        "+model.classifier_guidance_weight=3.0",
        "+model.use_ema=true",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.strategy=auto",
        "trainer.precision=32-true",
        "trainer.deterministic=true",
        "trainer.use_distributed_sampler=false",
        "callbacks.early_stopping.patience=50",
        f"hydra.searchpath=[file://{hydra_support.as_posix()},pkg://perturbench.configs]",
    ]
    train_command = [
        sys.executable,
        "-m",
        "primeflow.modelcore.train",
        *common,
        "train=true",
        "test=false",
        f"data.loader.batch_size={args.primeflow_batch_size}",
        f"trainer.min_epochs={args.primeflow_epochs}",
        f"trainer.max_epochs={args.primeflow_epochs}",
        f"hydra.run.dir={train_run}",
        "hydra.job.chdir=false",
    ]
    stage = "training"
    try:
        _run_command(train_command, workdir / "train.log", stage)

        checkpoints = sorted((train_run / "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            raise RuntimeError(f"PrimeFlow training completed without a checkpoint under {train_run}")
        checkpoint = checkpoints[-1]

        prediction_command = [
            sys.executable,
            "-m",
            "primeflow.modelcore.predict",
            *common,
            f"ckpt_path={checkpoint}",
            "+data/prediction=h5",
            f"data.prediction.prediction_dataframe_path={prediction_path}",
            f"data.prediction.predict_data_iter_factory.max_control_cells_per_covariate={args.max_pred_cells}",
            f"data.prediction.chunk_size={args.primeflow_batch_size}",
            f"model.euler_steps={args.primeflow_euler_steps}",
            "model.solver=euler",
            f"model.prediction_batch_size={args.primeflow_batch_size}",
            f"+output_path={prediction_dir}",
            f"hydra.run.dir={prediction_run}",
            "hydra.job.chdir=false",
        ]
        stage = "prediction"
        _run_command(prediction_command, workdir / "prediction.log", stage)

        conditions = pd.read_csv(prediction_path)["condition"].astype(str).tolist()
        per_condition = _prediction_arrays(prediction_dir, conditions, ctx)
    except Exception as exc:
        failure = {
            "method": "primeflow",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "train_log": str(workdir / "train.log"),
            "prediction_log": str(workdir / "prediction.log"),
            "upstream_ref": args.primeflow_source_ref,
        }
        (workdir / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise

    extra = {
        "upstream": "https://github.com/altoslabs/primeflow",
        "upstream_ref": args.primeflow_source_ref,
        "recipe": "official Norman19 Gaussian-source U-Net flow-matching experiment",
        "task_protocol": "unseen-guide / seen-target; training condition is target gene",
        "benchmark_split_source": "verified Squidiff preprocessing copy; same cells, gene order, and values as canonical benchmark split",
        "train_path": str(train_path),
        "train_sha256": sha256(train_path),
        "split_path": str(split_path),
        "split_sha256": sha256(split_path),
        "prediction_dataframe": str(prediction_path),
        "prediction_dataframe_sha256": sha256(prediction_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epochs": args.primeflow_epochs,
        "batch_size": args.primeflow_batch_size,
        "solver": "euler",
        "euler_steps": args.primeflow_euler_steps,
        "precision": "32-true",
        "classifier_guidance_weight": 3.0,
        "standard_gaussian_source": True,
        "device": args.primeflow_device,
        "train_log": str(workdir / "train.log"),
        "prediction_log": str(workdir / "prediction.log"),
        "n_training_cells": int(ctx.train.n_obs),
        "n_genes": len(ctx.genes),
        "conditions": conditions,
        "primeflow_package_version": _package_version("primeflow"),
        "perturbench_package_version": _package_version("perturbench"),
    }
    return finish(ctx, args, per_condition, extra)


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
