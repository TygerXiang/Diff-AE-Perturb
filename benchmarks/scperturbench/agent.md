# Agent Context

This directory is the tracked home of the benchmark harness. It was merged from
`C:\Users\Admin\myprojects\perturbench` on 2026-10-28; only code, tools, tests
and reports live here. Podman storage and `.scperturbench` artifacts stay in
`$SCPERTURBENCH_ROOT` (default `C:\Users\Admin\myprojects\perturbench`). See
`README.md` in this directory for the layout and the junction that keeps
relative paths working.

This workspace owns container-first benchmark execution for comparing the
`Diff-AE-Perturb` project with scGen. Do not use the host Python environment
for benchmark conclusions. The Podman image is the source of truth.

## Runtime

- Image: `localhost/scperturbench:squidiff` (derived from the original `:latest` image)
- Podman connection: `scperturbench-root`
- Machine: `scperturbench` (Fedora 42 in WSL2)
- GPU: NVIDIA GeForce RTX 5060
- GPU CDI spec: `/etc/cdi/nvidia.yaml` inside the machine
- Verified command: `.run-scperturbench.ps1 -Gpu`
- Container Python: 3.12.2
- Conda: 25.3.1
- scGen: 2.1.1 in conda environment `scarches`
- Squidiff: 1.0.8 in conda environment `squidiff`
- Squidiff extra runtime dependency: `rdkit 2022.03.2`

The image exposes the GPU through CDI and `/dev/dxg`. The run and verification
scripts add the WSL driver directories to `LD_LIBRARY_PATH` because some image
linker configurations do not find the CDI-mounted NVIDIA libraries otherwise.

Important compatibility note: the installed PyTorch in `scarches` reports CUDA
available and sees the RTX 5060, but warns that the build does not contain an
`sm_120` kernel. Treat GPU visibility and GPU training compatibility as separate
checks. Do not silently claim a GPU benchmark succeeded when PyTorch falls back
or fails at kernel execution.

## Source and Data

- Compared project: `C:\Users\Admin\myprojects\Diff-AE-Perturb`
- Dataset root: `C:\Users\Admin\myprojects\diff-project\datasets`
- Prepared GSE132080 splits:
  `C:\Users\Admin\myprojects\diff-project\reproducibility\prepared\splits`
- Prepared GSE283614 files, when available:
  `C:\Users\Admin\myprojects\Diff-AE-Perturb\reproducibility\prepared\gse283614`
- Outputs should be written under this workspace's `.scperturbench\results`
  or the compared project's `runs` directory only when explicitly requested.

The most direct existing comparison implementation is
`Diff-AE-Perturb\scripts\compare_with_scgen.py`. The six benchmark metrics are
implemented in `Diff-AE-Perturb\scripts\evaluate_benchmark.py`:

1. `mse` (lower is better)
2. `e_distance` / energy distance (lower is better)
3. `pcc_delta` (higher is better)
4. `wasserstein` (lower is better)
5. `kl_divergence` (lower is better)
6. `common_degs` (higher is better)

## Benchmark Rules

- Use the same train/test split, gene order, control pool, condition labels,
  random seed, generated-control budget, and DDIM step count for every method.
- Train scGen and evaluate the Diff-AE-Perturb checkpoint inside the image.
- Use the `scarches` conda environment for scGen and the shared metric code.
- Prefer guide-level evaluation when the Diff-AE-Perturb checkpoint contains
  guide-specific embeddings; otherwise state explicitly that evaluation is at
  target level.
- Record the exact image, conda environment, checkpoint, dataset paths,
  command, seed, device, and all six metrics in the output JSON/CSV.
- Report missing conditions and minimum-cell filtering. Never compare averages
  over different condition sets without labeling the difference.
- A successful CDI probe is not sufficient evidence of successful GPU model
  execution. Capture PyTorch CUDA warnings and kernel errors in the run log.

## Operational Commands

From this directory:

```powershell
.verify-scperturbench.ps1 -Gpu
.run-scperturbench.ps1 -Gpu -Image localhost/scperturbench:squidiff -Command @('-lc', "conda run --no-capture-output -n scarches python -c 'import scgen; print(scgen.__version__)'")
.run-scperturbench.ps1 -Gpu -Image localhost/scperturbench:squidiff -Command @('-lc', "conda run --no-capture-output -n squidiff python -c 'import Squidiff, rdkit; print(Squidiff.__file__, rdkit.__version__)'")
```

The Podman machine's CDI setup can be regenerated after a machine reset with:

```powershell
podman machine start scperturbench
podman machine ssh scperturbench "dnf install -y golang-github-nvidia-container-toolkit"
podman machine ssh scperturbench "mkdir -p /etc/cdi; nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
```

Do not use `--gpus all` as evidence of a valid benchmark until
`verify-scperturbench.ps1 -Gpu` passes and the selected conda environment has
been probed for actual PyTorch device execution.
