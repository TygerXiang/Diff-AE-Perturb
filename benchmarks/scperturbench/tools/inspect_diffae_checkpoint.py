"""Inspect a Diff-AE-Perturb checkpoint: which conditioning branches actually have weights.

Run inside the container (env `scarches`) with the repo mounted at /root/Diff-AE-Perturb.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

# The checkpoint pickles dataclasses from the project itself.
sys.path.insert(0, "/root/Diff-AE-Perturb/src")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    report: dict = {"top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else str(type(payload))}
    state = payload
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            report["state_dict_under"] = key
            break
    report["n_tensors"] = len(state)
    report["prefixes"] = dict(collections.Counter(k.split(".")[0] for k in state))

    interesting = ("perturbation", "activity", "guide", "denoiser", "encoder")
    shapes = {}
    for name, tensor in state.items():
        if any(token in name for token in interesting) and hasattr(tensor, "shape"):
            if any(token in name for token in ("perturbation", "activity", "guide")):
                shapes[name] = list(tensor.shape)
    report["conditioning_tensor_shapes"] = shapes
    report["denoiser_first_last"] = {
        name: list(tensor.shape)
        for name, tensor in state.items()
        if name.startswith("denoiser") and hasattr(tensor, "shape") and ("0.weight" in name or name.endswith("weight"))
    } if False else {}
    # denoiser input layer width reveals whether activity conditioning was concatenated
    for name, tensor in state.items():
        if name.startswith("denoiser.network.0.") and hasattr(tensor, "shape"):
            report["denoiser_input_layer"] = {name: list(tensor.shape)}

    # numeric magnitude of the learned per-perturbation deltas
    for name, tensor in state.items():
        if name.startswith("perturbation.embedding"):
            report["perturbation_embedding_stats"] = {
                "shape": list(tensor.shape),
                "row_norms": [round(float(v), 5) for v in tensor.norm(dim=1).tolist()],
                "mean_abs": float(tensor.abs().mean()),
            }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
