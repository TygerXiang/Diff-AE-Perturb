"""Print the exact API surface of each method package (run inside the container).

Usage:
    python scperturbench_probe.py --what cpa
    python scperturbench_probe.py --what gears
    python scperturbench_probe.py --what scgen
    python scperturbench_probe.py --what squidiff
"""

from __future__ import annotations

import argparse
import inspect


def dump(name: str, obj, limit: int = 60) -> None:
    print(f"### {name}")
    try:
        print(f"  signature: {inspect.signature(obj)}")
    except (TypeError, ValueError) as exc:
        print(f"  signature: <{exc}>")
    try:
        source = inspect.getsource(obj)
        lines = source.splitlines()
        print(f"  source ({len(lines)} lines):")
        for line in lines[:limit]:
            print(f"    {line}")
    except (TypeError, OSError) as exc:
        print(f"  source: <{exc}>")


def probe_cpa() -> None:
    import cpa

    print(f"cpa version: {getattr(cpa, '__version__', 'n/a')} file={cpa.__file__}")
    print(f"cpa exports: {[n for n in dir(cpa) if not n.startswith('_')]}")
    for name in ("CPA", "BaseCPA"):
        if hasattr(cpa, name):
            cls = getattr(cpa, name)
            print(f"--- class {name} methods: {[m for m in dir(cls) if not m.startswith('_')]}")
            dump(f"{name}.__init__", cls.__init__)
            for method in ("setup_anndata", "train", "predict", "get_latent_representation"):
                if hasattr(cls, method):
                    dump(f"{name}.{method}", getattr(cls, method))
    if hasattr(cpa, "CPA") and hasattr(cpa.CPA, "predict"):
        dump("CPA.predict", cpa.CPA.predict, limit=120)
    if hasattr(cpa, "CPA") and hasattr(cpa.CPA, "custom_predict"):
        dump("CPA.custom_predict", cpa.CPA.custom_predict, limit=60)
    if hasattr(cpa, "CPA") and hasattr(cpa.CPA, "get_pert_embeddings"):
        dump("CPA.get_pert_embeddings", cpa.CPA.get_pert_embeddings)
    if hasattr(cpa, "CPAModule"):
        dump("CPAModule.__init__", cpa.CPAModule.__init__, limit=80)


def probe_gears() -> None:
    import gears
    from gears import GEARS, PertData

    print(f"gears file: {gears.__file__}")
    dump("PertData.prepare_split", PertData.prepare_split, limit=140)
    dump("PertData.load", PertData.load, limit=60)
    dump("PertData.new_data_process", PertData.new_data_process)
    dump("PertData.get_dataloader", PertData.get_dataloader, limit=200)
    dump("GEARS.predict", GEARS.predict, limit=80)
    dump("GEARS.__init__", GEARS.__init__, limit=70)
    dump("GEARS.model_initialize", GEARS.model_initialize, limit=90)
    from gears.inference import evaluate, compute_metrics

    dump("gears.inference.evaluate", evaluate, limit=90)
    dump("gears.inference.compute_metrics", compute_metrics, limit=40)


def probe_scgen() -> None:
    import scgen

    print(f"scgen version: {scgen.__version__} file={scgen.__file__}")
    dump("SCGEN.__init__", scgen.SCGEN.__init__)
    dump("SCGEN.train", scgen.SCGEN.train)
    dump("SCGEN.predict", scgen.SCGEN.predict)
    dump("SCGEN.setup_anndata", scgen.SCGEN.setup_anndata)


def probe_squidiff() -> None:
    import Squidiff
    from Squidiff.MLPModel import MLPModel, EncoderMLPModel

    print(f"Squidiff file: {Squidiff.__file__}")
    dump("MLPModel.__init__", MLPModel.__init__)
    dump("MLPModel.forward", MLPModel.forward)
    dump("EncoderMLPModel.__init__", EncoderMLPModel.__init__)
    dump("EncoderMLPModel.forward", EncoderMLPModel.forward)
    from Squidiff import diffusion

    dump("GaussianDiffusion.ddim_sample_loop", diffusion.GaussianDiffusion.ddim_sample_loop)
    dump("GaussianDiffusion.p_sample_loop", diffusion.GaussianDiffusion.p_sample_loop)
    from Squidiff import train_util

    for name in ("_run_step", "_train_step", "run_loop", "forward_backward"):
        if hasattr(train_util.TrainLoop, name):
            dump(f"TrainLoop.{name}", getattr(train_util.TrainLoop, name), limit=70)
    dump("TrainLoop.__init__", train_util.TrainLoop.__init__, limit=80)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", required=True, choices=["cpa", "gears", "scgen", "squidiff"])
    args = parser.parse_args()
    {
        "cpa": probe_cpa,
        "gears": probe_gears,
        "scgen": probe_scgen,
        "squidiff": probe_squidiff,
    }[args.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
