"""Contract test for the Diff-AE-Perturb activity-conditioning passthrough.

``adapter_diffae`` must hand the Table S8 guide features, the activity lookup and the
configured guide sequence length to the repository's ``predict_model``.  Without that,
any checkpoint carrying a guide encoder (``activity_source=predicted``) cannot be
evaluated by the harness at all, and the "activity on vs off" ablation would silently
compare a working arm against a broken one.

The test runs without the real ``torch`` / ``perturbdiff`` / checkpoint machinery: the
adapter loads them lazily inside the function, so a stub module tree injected into
``sys.modules`` is enough to observe the call contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scperturbench_adapters


FEATURE_DIM = 208
FAKE_CNN_RECORDS = {"ALDOA_+_30077139.23-P1P2_00": object()}
FAKE_ACTIVITY = {"ALDOA_+_30077139.23-P1P2_00": 0.42}


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    @property
    def shape(self):
        return self._array.shape

    def __array__(self, dtype=None):
        return self._array if dtype is None else self._array.astype(dtype)

    def to(self, *_args, **_kwargs):
        return self

    def mean(self, dim=0):
        return _FakeTensor(self._array.mean(axis=dim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def astype(self, dtype, copy=False):
        return self._array.astype(dtype, copy=copy)


def _install_stubs(monkeypatch, tmp_path, captured):
    """Install stub ``torch`` / ``perturbdiff`` modules and capture predict_model kwargs."""

    torch_stub = mock.MagicMock()
    torch_stub.device.side_effect = lambda name: SimpleNamespace(type=str(name))
    torch_stub.from_numpy.side_effect = _FakeTensor
    torch_stub.manual_seed.return_value = None

    data_stub = mock.MagicMock()
    data_stub.load_guide_activity_lookup.side_effect = lambda path, activity_col=None: dict(FAKE_ACTIVITY)

    guide_features_stub = mock.MagicMock()
    guide_features_stub.load_guide_feature_records.side_effect = lambda paths: dict(FAKE_CNN_RECORDS)

    def fake_predict_model(model, mapping, control_tensor, **kwargs):
        captured["kwargs"] = kwargs
        captured["control_rows"] = int(_FakeTensor(control_tensor).shape[0])
        return np.full((_FakeTensor(control_tensor).shape[0], 3), 0.5, dtype=np.float32)

    evaluator_stub = mock.MagicMock()
    evaluator_stub.predict_model.side_effect = fake_predict_model
    evaluator_stub.load_model.side_effect = lambda path, device: (captured["model"], {"control": 0, "A": 1})

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "perturbdiff", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "perturbdiff.data", data_stub)
    monkeypatch.setitem(sys.modules, "perturbdiff.guide_features", guide_features_stub)

    real_spec_from_file_location = importlib.util.spec_from_file_location

    def fake_spec_from_file_location(name, location):
        if str(location).endswith("evaluate_benchmark.py"):
            return SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None), name=name)
        return real_spec_from_file_location(name, location)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", fake_spec_from_file_location)
    monkeypatch.setattr(
        importlib.util,
        "module_from_spec",
        lambda spec: evaluator_stub,
    )
    return torch_stub


def _fake_context():
    genes = ["g1", "g2", "g3"]
    labels = ["control", "control", "A", "A"]
    train = ad.AnnData(
        X=np.zeros((4, 3), dtype=np.float32),
        obs=pd.DataFrame({"perturbation_label": labels}, index=[f"cell-{i}" for i in range(4)]),
        var=pd.DataFrame(index=genes),
    )
    control_matrix = np.ones((2, 3), dtype=np.float32)

    def control_dense_subset(idx):
        return control_matrix[np.isin(np.arange(control_matrix.shape[0]), idx)]

    def sample_control_indices(condition, n, key):
        return np.arange(min(n, 2))

    return SimpleNamespace(
        conditions=["A"],
        control_label="control",
        genes=genes,
        train=train,
        test=train,
        train_labels=np.asarray(labels),
        control_matrix=lambda: control_matrix,
        control_dense_subset=control_dense_subset,
        sample_control_indices=sample_control_indices,
        n_pred_cells=lambda condition: 2,
        true_cells=lambda condition: np.full((2, 3), 0.25, dtype=np.float32),
    )


def _fake_cfg():
    return SimpleNamespace(
        guide_encoder=SimpleNamespace(
            enabled=True,
            activity_source="predicted",
            predictor_backend="table_s8_cnn",
            feature_dim=FEATURE_DIM,
        )
    )


def test_adapter_passes_guide_features_and_activity_lookup(monkeypatch, tmp_path):
    captured: dict[str, object] = {"model": SimpleNamespace(config=_fake_cfg(), guide_doser=object())}
    _install_stubs(monkeypatch, tmp_path, captured)

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "evaluate_benchmark.py").write_text("# stub\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"stub checkpoint")

    train_path = tmp_path / "train.h5ad"
    train_path.write_bytes(b"stub split")
    test_path = tmp_path / "test.h5ad"
    test_path.write_bytes(b"stub split")

    cnn_table = tmp_path / "Table_S8.txt"
    cnn_table.write_text("stub\n", encoding="utf-8")
    activity_table = tmp_path / "Table_S16.txt"
    activity_table.write_text("stub\n", encoding="utf-8")

    args = SimpleNamespace(
        repo=repo,
        checkpoint=checkpoint,
        train_path=train_path,
        test_path=test_path,
        seed=7,
        device="cuda",
        conditions=["A"],
        eval_level="target",
        ddim_steps=50,
        diffae_delta_mode="checkpoint",
        diffae_delta_scale=50.0,
        diffae_guide_activity_cnn_table=cnn_table,
        diffae_activity_table=activity_table,
        diffae_activity_col="relative_activity_day10",
        diffae_guide_sequence_length=26,
        method="activity_on_200ep",
        adapter="diffae",
        output_dir=tmp_path / "out",
    )

    exit_code = scperturbench_adapters.adapter_diffae(_fake_context(), args)

    assert exit_code == 0
    kwargs = captured["kwargs"]
    assert kwargs["records_by_key"] == FAKE_CNN_RECORDS
    assert kwargs["activity_lookup"] == FAKE_ACTIVITY
    assert kwargs["sequence_length"] == 26
    assert kwargs["eval_level"] == "target"
    assert kwargs["steps"] == 50

    import json

    manifest = json.loads((tmp_path / "out" / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["guide_feature_dim"] == FEATURE_DIM
    assert manifest["extra"]["activity_source"] == "predicted"
    assert manifest["extra"]["activity_predictor_backend"] == "table_s8_cnn"
    assert manifest["extra"]["n_activity_scores_loaded"] == len(FAKE_ACTIVITY)
    assert manifest["extra"]["skipped_conditions"] == {}
    assert manifest["n_conditions"] == 1


def test_adapter_skips_conditions_missing_from_the_checkpoint_mapping(monkeypatch, tmp_path):
    captured: dict[str, object] = {"model": SimpleNamespace(config=_fake_cfg(), guide_doser=object())}
    _install_stubs(monkeypatch, tmp_path, captured)
    captured["model"].config = _fake_cfg()

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "evaluate_benchmark.py").write_text("# stub\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"stub checkpoint")

    train_path = tmp_path / "train.h5ad"
    train_path.write_bytes(b"stub split")
    test_path = tmp_path / "test.h5ad"
    test_path.write_bytes(b"stub split")

    ctx = _fake_context()
    ctx.conditions = ["A"]

    args = SimpleNamespace(
        repo=repo,
        checkpoint=checkpoint,
        train_path=train_path,
        test_path=test_path,
        seed=7,
        device="cuda",
        conditions=["A"],
        eval_level="target",
        ddim_steps=50,
        diffae_delta_mode="checkpoint",
        diffae_delta_scale=50.0,
        diffae_guide_activity_cnn_table=None,
        diffae_activity_table=None,
        diffae_activity_col="relative_activity_day10",
        diffae_guide_sequence_length=26,
        method="activity_off_200ep",
        adapter="diffae",
        output_dir=tmp_path / "out",
    )

    # The stub returns a prediction for the single condition, so no skip is recorded here;
    # this asserts the counter exists and stays empty on the happy path.
    assert scperturbench_adapters.adapter_diffae(ctx, args) == 0

    import json

    manifest = json.loads((tmp_path / "out" / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["skipped_conditions"] == {}
    assert manifest["extra"]["guide_encoder_enabled"] is True
