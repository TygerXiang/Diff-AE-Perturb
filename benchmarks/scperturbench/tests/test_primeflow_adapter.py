import sys
from types import SimpleNamespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scperturbench_primeflow import _prediction_arrays, _write_primeflow_inputs


def make_context(tmp_path):
    labels = ["control", "control", "A", "A", "B", "B"]
    train = ad.AnnData(
        X=np.arange(18, dtype=np.float32).reshape(6, 3),
        obs=pd.DataFrame({"perturbation_label": labels}, index=[f"cell-{i}" for i in range(6)]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    return SimpleNamespace(
        train=train,
        condition_col="perturbation_label",
        control_label="control",
        conditions=["A", "B"],
        genes=["g1", "g2", "g3"],
        n_pred_cells=lambda _condition: 2,
    )


def test_primeflow_inputs_use_training_cells_and_create_seen_target_conditions(tmp_path):
    ctx = make_context(tmp_path)

    train_path, split_path, prediction_path = _write_primeflow_inputs(ctx, tmp_path / "inputs", 17)

    train = ad.read_h5ad(train_path)
    split = pd.read_csv(split_path, index_col=0, header=None).iloc[:, 0]
    prediction = pd.read_csv(prediction_path)
    assert train.n_obs == ctx.train.n_obs - 2
    assert ctx.control_label not in set(train.obs["condition"].astype(str))
    expected_names = [name for name in ctx.train.obs_names if name not in {"cell-0", "cell-1"}]
    assert train.obs_names.tolist() == expected_names
    assert train.obs["condition"].astype(str).tolist() == ["A", "A", "B", "B"]
    assert set(split) == {"train", "val"}
    assert split.index.tolist() == train.obs_names.tolist()
    assert prediction["condition"].tolist() == ["A", "B"]


def test_primeflow_predictions_are_gene_checked_and_cropped_to_shared_budget(tmp_path):
    ctx = make_context(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction = ad.AnnData(
        X=np.arange(18, dtype=np.float32).reshape(6, 3),
        obs=pd.DataFrame({"condition": ["A", "A", "A", "B", "B", "B"]}),
        var=pd.DataFrame(index=ctx.genes),
    )
    prediction.write_h5ad(prediction_dir / "prediction_rank0_chunk_0.h5ad")

    result = _prediction_arrays(prediction_dir, ["A", "B"], ctx)

    assert result["A"].shape == (2, 3)
    assert result["B"].shape == (2, 3)
    np.testing.assert_array_equal(result["A"], np.arange(9, dtype=np.float32).reshape(3, 3)[:2])


def test_primeflow_predictions_reject_gene_order_mismatch(tmp_path):
    ctx = make_context(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction = ad.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame({"condition": ["A", "A"]}),
        var=pd.DataFrame(index=["g2", "g1", "g3"]),
    )
    prediction.write_h5ad(prediction_dir / "prediction_rank0_chunk_0.h5ad")

    with pytest.raises(RuntimeError, match="gene order differs"):
        _prediction_arrays(prediction_dir, ["A"], ctx)
