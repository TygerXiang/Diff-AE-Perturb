# scPerturBench Podman runtime

This directory contains a local Podman setup for the downloaded image archive:

```text
scperturbench_v1.tar.gz
```

The setup uses a project-local Podman configuration and a dedicated WSL machine
named `scperturbench`. This avoids the broken user-level Podman configuration
under `C:\Users\Admin\.config`.

## First run

Open PowerShell in this directory and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\prepare-scperturbench.ps1 -LoadImage
.\verify-scperturbench.ps1 -Gpu
```

The first command may take a long time because the image archive is large. It
loads the gzip archive directly and does not create a second `.tar` file.

## Start an interactive container

```powershell
.\run-scperturbench.ps1 -Gpu
```

The container starts in `/root/project`. Persistent host directories are:

```text
.scperturbench\inputs   -> /root/project/inputs
.scperturbench\results  -> /root/project/results
.scperturbench\cache    -> /root/.cache
```

## Run a command

```powershell
.\run-scperturbench.ps1 -Gpu -Command @('-lc', 'python --version')
.\run-scperturbench.ps1 -Gpu -Command @('-lc', 'conda env list')
```

## Squidiff environment

The derived image `localhost/scperturbench:squidiff` contains a `squidiff`
Conda environment. It is cloned from `scarches` so the tested PyTorch/CUDA,
Scanpy, anndata, SciPy, scikit-learn, and matplotlib versions stay aligned;
the extra Squidiff runtime dependency is `rdkit=2022.03.2`.

The image is already configured in this workspace. To rebuild it from the
local Squidiff checkout, run:

```powershell
.\configure-squidiff.ps1
```

Verify the package and environment with:

```powershell
.\run-scperturbench.ps1 -Gpu -Image localhost/scperturbench:squidiff -Command @('-lc', 'conda run --no-capture-output -n squidiff python -c "import Squidiff, rdkit, torch; print(Squidiff.__file__, rdkit.__version__, torch.__version__)"')
```

If the archive imports a different image tag, pass it explicitly:

```powershell
.\verify-scperturbench.ps1 -Image <repository>:<tag> -Gpu
.\run-scperturbench.ps1 -Image <repository>:<tag> -Gpu
```

## Current host observations

- Podman 5.6.0 with the WSL provider is installed.
- NVIDIA GPU: GeForce RTX 5060 with 8 GB VRAM.
- A dedicated machine was created with 10 CPUs, 16 GiB RAM, and a 120 GiB
  virtual disk.
- The machine uses the rootful connection `scperturbench-root`.

## GPU CDI setup

The WSL2 driver is exposed through `/dev/dxg` and the WSL driver store. If
`--gpus all` reports an unresolved CDI device, start the machine and generate
the CDI spec inside it:

```powershell
podman machine start scperturbench
podman machine ssh scperturbench "dnf install -y golang-github-nvidia-container-toolkit"
podman machine ssh scperturbench "mkdir -p /etc/cdi; nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
.\verify-scperturbench.ps1 -Gpu
```

The verification and run scripts add the WSL driver directories to
`LD_LIBRARY_PATH`; this is required by some images even when CDI mounts the
libraries successfully. The current setup was verified with an NVIDIA
GeForce RTX 5060 and `localhost/scperturbench:squidiff`.

If WSL reports a localhost proxy warning, it only means that the proxy is not
available inside WSL. Local image loading does not require that proxy.
