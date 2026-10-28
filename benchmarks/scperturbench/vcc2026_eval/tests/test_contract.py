from __future__ import annotations

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


MODULE = Path(__file__).parents[1] / "vcc2026_eval.py"
spec = importlib.util.spec_from_file_location("vcc2026_eval", MODULE)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def write_h5ad(tmp_path, matrix, labels=None, genes=None):
    labels = labels or ["non-targeting", "GENE1"]
    genes = genes or ["GENE1", "GENE2"]
    obs = pd.DataFrame({"target": labels})
    data = ad.AnnData(X=np.asarray(matrix), obs=obs,
                      var=pd.DataFrame(index=genes))
    path = tmp_path / "input.h5ad"
    data.write_h5ad(path)
    return path


@pytest.mark.parametrize("matrix,needle", [
    ([[1, -1], [1, 0]], "non-negative"),
    ([[1.5, 0], [1, 0]], "integral"),
    ([[600_001, 400_000], [1, 0]], "1,000,000"),
    ([[np.nan, 0], [1, 0]], "NaN or infinity"),
])
def test_rejects_invalid_counts(tmp_path, matrix, needle):
    path = write_h5ad(tmp_path, matrix)
    with pytest.raises(ValueError, match=needle):
        wrapper._load_dense(path)


def test_requires_perturbation_column_and_control(tmp_path):
    path = write_h5ad(tmp_path, [[1, 0], [0, 1]], labels=["ctrl", "GENE1"])
    data = wrapper._load_dense(path)
    with pytest.raises(ValueError, match="control label"):
        wrapper._identity(data, path=path, pert_col="target", control="non-targeting")
    with pytest.raises(ValueError, match="obs column"):
        wrapper._identity(data, path=path, pert_col="missing", control="ctrl")


def test_rejects_missing_and_extra_perturbations(tmp_path):
    ref = write_h5ad(tmp_path, [[1, 0], [0, 1]], labels=["non-targeting", "GENE1"])
    pred_data = ad.read_h5ad(ref)
    pred_data.obs["target"] = ["non-targeting", "GENE2"]
    pred = tmp_path / "pred.h5ad"
    pred_data.write_h5ad(pred)
    ref_data = wrapper._load_dense(ref)
    pred_data = wrapper._load_dense(pred)
    with pytest.raises(ValueError, match=r"labels differ; missing=\['GENE1'\], extra=\['GENE2'\]"):
        wrapper.validate_alignment(pred_data, ref_data, prediction_path=pred,
                                   reference_path=ref, pert_col="target",
                                   control="non-targeting")


def test_rejects_reordered_genes(tmp_path):
    ref = write_h5ad(tmp_path, [[1, 0], [0, 1]])
    pred_data = ad.read_h5ad(ref)[:, ::-1].copy()
    pred = tmp_path / "pred.h5ad"
    pred_data.write_h5ad(pred)
    with pytest.raises(ValueError, match="ordered gene axis differs"):
        wrapper.validate_alignment(pred_data, ad.read_h5ad(ref), prediction_path=pred,
                                   reference_path=ref, pert_col="target",
                                   control="non-targeting")


def test_rejects_missing_labels_and_null_labels(tmp_path):
    ref = write_h5ad(tmp_path, [[1, 0], [0, 1]])
    pred_data = ad.read_h5ad(ref)
    pred_data.obs["target"] = ["non-targeting", None]
    pred = tmp_path / "pred.h5ad"
    pred_data.write_h5ad(pred)
    with pytest.raises(ValueError, match="missing values"):
        wrapper._identity(pred_data, path=pred, pert_col="target", control="non-targeting")


def test_requires_existing_inputs_and_aligned_argument_lengths(tmp_path):
    with pytest.raises(SystemExit):
        wrapper.main(["--predictions", str(tmp_path / "missing.h5ad"), "--bundles",
                      str(tmp_path / "bundle"), "--references", str(tmp_path / "ref.h5ad"),
                      "--contexts", "A", "--outdir",
                      str(tmp_path / "out")])


def test_rejects_argument_length_mismatch(tmp_path):
    pred = tmp_path / "pred.h5ad"
    bundle = tmp_path / "bundle"
    ref = tmp_path / "ref.h5ad"
    pred.touch()
    ref.touch()
    bundle.mkdir()
    with pytest.raises(SystemExit):
        wrapper.main([
            "--predictions", str(pred), "--bundles", str(bundle),
            "--references", str(ref), "--contexts", "A", "B",
            "--outdir", str(tmp_path / "out"),
        ])


def test_official_metric_contract():
    assert len(wrapper.EXPECTED_METRICS) == 6
    assert len(set(wrapper.EXPECTED_METRICS)) == 6
    assert wrapper.EXPECTED_RULE_VERSION == 3


def test_rejects_nonexistent_inputs_before_scoring(tmp_path):
    with pytest.raises(SystemExit):
        wrapper.main([
            "--predictions", str(tmp_path / "missing.h5ad"),
            "--bundles", str(tmp_path / "bundle"),
            "--references", str(tmp_path / "reference.h5ad"),
            "--contexts", "A", "--outdir", str(tmp_path / "out"),
        ])
