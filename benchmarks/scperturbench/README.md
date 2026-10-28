# scPerturBench harness

This directory is the version-controlled home of the benchmark harness that used
to live only in `C:\Users\Admin\myprojects\perturbench` (an untracked 390 GB
workspace). It was merged into this repository on 2026-10-28.

The harness compares `Diff-AE-Perturb` against GEARS, scGen, CPA, Squidiff,
PrimeFlow and the linear baselines with six metrics on the same split, gene
order, control pool and seed. The method adapters, metric implementation,
aggregation, diagnostic tools and reports are all tracked here.

## Layout

| Path | Content |
| --- | --- |
| `scperturbench_adapters.py` | Method adapters (`diffae`, `scgen`, `cpa`, `gears`, `squidiff`, `squidiff_repro`, `primeflow`, linear baselines) |
| `scperturbench_metrics.py` | Six-metric scoring (literature port of `calPerformance_genetic.py`) |
| `scperturbench_inputs.py`, `scperturbench_aggregate.py` | Input preparation and table aggregation |
| `scperturbench_*.py` | Focused drivers: probe, sanity, crosscheck, diagnosis, Squidiff prep/sampling, GSE283614 ablations, figures |
| `run-*.ps1` | Podman entry points; `scperturbench-common.ps1` holds the shared runtime root |
| `tools/` | One-off analyses, preflights and claim verifiers. They assume the workspace root contains `.scperturbench/` and `reports/` |
| `tests/` | Adapter contract tests (run on the host) |
| `reports/` | Benchmark and ablation reports plus their `assets/` tables and figures |
| `vcc2026_eval/` | VCC2026 evaluation contract and runner |

## Code vs. runtime data

Git tracks the code and reports, not the artifacts. The runtime root is external
and is selected in this order:

1. `$env:SCPERTURBENCH_ROOT`
2. `C:\Users\Admin\myprojects\perturbench` (legacy workspace, still the default
   on this machine)

It contains:

```
$SCPERTURBENCH_ROOT\
  .scperturbench\   # 38 GB: inputs, trained checkpoints, results, envs, cache
  .local\           # 310 GB: Podman machine storage
  .config\ .podman\ # Podman configuration
  scperturbench_v1.tar.gz
```

`.scperturbench` is deliberately not in git. A directory junction keeps the
relative-path tools working without rewriting them:

```powershell
New-Item -ItemType Junction `
  -Path benchmarks\scperturbench\.scperturbench `
  -Target "$env:SCPERTURBENCH_ROOT\.scperturbench"
```

Recreate the junction after cloning on a new machine, or point the tools at an
explicit path with the `--results-root` / `--inputs-dir` arguments.

## Running

The PowerShell entry points mount:

- this directory at `/root/perturbench`
- `$SCPERTURBENCH_ROOT\.scperturbench` at `/root/perturbench/.scperturbench`
- the repository root at `/root/Diff-AE-Perturb` (read-only)

```powershell
# Verify the Podman machine, image and GPU passthrough
.\verify-scperturbench.ps1 -Gpu

# Six-metric benchmark for one dataset/seed
.\run-benchmark.ps1 -Dataset gse132080 -Seed 7

# Diff-AE-Perturb conditioning ablation (train + predict + score)
.\run-diffae-fix.ps1 -Variants fix_x0_percell,legacy_x0_percell
```

Host-side contract tests:

```powershell
python -m pytest benchmarks/scperturbench/tests -q
```

## Reports

Start with `reports/gse132080_6metrics_benchmark.md` for the main table and
`reports/external_methods_comparison.md` for the four external methods. The
GSE283614 work is in `reports/gse283614_de_vs_esc_why.md`,
`reports/gse283614_activity_ablation.md` and
`reports/gse283614_knockdown3_ablation.md`.

## History

Before this merge the same code was edited in two places: here (untracked) and
in a few `Diff-AE-Perturb/scripts/` copies. That duplication is why the
2026-09-28 structure audit found four copies of `preprocess_gse132080.py` and
five of `train_with_split.py`. Keep new benchmark changes in this directory.
