[CmdletBinding()]
param(
    [string]$Image = '',
    [switch]$Gpu
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scperturbench-common.ps1')
Set-ScPerturBenchPodmanEnvironment

if ([string]::IsNullOrWhiteSpace($Image)) {
    $Image = Get-ScPerturBenchImage -Connection $ScPerturBenchConnection
}
if ([string]::IsNullOrWhiteSpace($Image)) {
    throw 'No full scPerturBench image was found. Run .\prepare-scperturbench.ps1 -LoadImage first.'
}

$checkScript = @'
set -eu
echo "container=$(hostname)"
echo "user=$(id -un)"
echo "workdir=$(pwd)"
echo "python=$(python --version 2>&1)"
echo "conda=$(conda --version 2>&1)"
test -d /root/project
echo "project=/root/project"
echo "--- conda environments ---"
conda env list
if conda env list | awk '$1 == "squidiff" { found=1 } END { exit !found }'; then
  conda run --no-capture-output -n squidiff python -c 'import rdkit, Squidiff, torch; print("squidiff=", Squidiff.__file__); print("rdkit=", rdkit.__version__, "torch=", torch.__version__, "cuda=", torch.version.cuda)'
fi
echo "--- project files ---"
find /root/project -maxdepth 1 -mindepth 1 -printf "%f\n" 2>/dev/null | sort | head -80
'@

Write-Host "Checking image: $Image"
Invoke-ScPerturBenchPodman -Connection $ScPerturBenchConnection -Arguments @(
    'run',
    '--rm',
    '--entrypoint',
    '/bin/bash',
    $Image,
    '-lc',
    $checkScript
)

if ($Gpu) {
    Write-Host ''
    Write-Host 'Checking NVIDIA GPU passthrough...'
    Invoke-ScPerturBenchPodman -Connection $ScPerturBenchConnection -Arguments @(
        'run',
        '--rm',
        '--gpus',
        'all',
        '--env',
        'LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nv_dispig.inf_amd64_f4c7a2fd13e0f763:/usr/lib/wsl/lib:$LD_LIBRARY_PATH',
        '--entrypoint',
        '/bin/bash',
        $Image,
        '-lc',
        'set -eu; command -v nvidia-smi; nvidia-smi -L; for env in scarches squidiff; do if conda env list | awk -v e="$env" ''$1 == e { found=1 } END { exit !found }''; then conda run --no-capture-output -n "$env" python -c "import torch; print(\"env=\", \"$env\", \"torch=\", torch.__version__, \"cuda=\", torch.version.cuda); print(\"device=\", torch.cuda.get_device_name(0)); print(\"arch_list=\", torch.cuda.get_arch_list()); assert torch.cuda.is_available(); x=torch.randn(128,128,device=\"cuda\"); print(\"kernel_mean=\", float((x@x).mean()))"; fi; done'
    )
}

Write-Host ''
Write-Host 'scPerturBench runtime check passed.'
