"""Diagnostics for the GSE132080 prepared split (run inside the container)."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def describe(name: str, path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    names = adata.obs_names.astype(str)
    print(f"[{name}] {path}")
    print(f"  shape={adata.shape} obs_names_unique={names.is_unique} n_dup={len(names) - names.nunique()}")
    print(f"  var_names_unique={adata.var_names.astype(str).is_unique}")
    print(f"  obs cols={list(adata.obs.columns)}")
    print(f"  layers={list(adata.layers.keys())} uns={list(adata.uns.keys())}")
    labels = adata.obs["perturbation_label"].astype(str)
    print(f"  perturbation_label nunique={labels.nunique()}")
    print(f"  first obs_names={names[:3].tolist()}")
    return adata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    args = parser.parse_args()

    train = describe("train", args.train_path)
    test = describe("test", args.test_path)

    train_names = set(train.obs_names.astype(str))
    test_names = set(test.obs_names.astype(str))
    print(f"barcode overlap train∩test = {len(train_names & test_names)}")
    print(f"var equal = {list(train.var_names.astype(str)) == list(test.var_names.astype(str))}")

    try:
        combined = ad.concat([train, test], join="inner")
        print(f"concat OK shape={combined.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"concat FAILED: {type(exc).__name__}: {exc}")

    labels = train.obs["perturbation_label"].astype(str).to_numpy()
    print("train label counts:", dict(zip(*np.unique(labels, return_counts=True))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
