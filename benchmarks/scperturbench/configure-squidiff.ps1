[CmdletBinding()]
param(
    [string]$BaseImage = 'localhost/scperturbench:latest',
    [string]$Image = 'localhost/scperturbench:squidiff',
    [string]$SquidiffPath = 'C:\Users\Admin\myprojects\Squidiff'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scperturbench-common.ps1')
Set-ScPerturBenchPodmanEnvironment

$resolvedSquidiff = (Resolve-Path -LiteralPath $SquidiffPath).Path
$container = "scperturbench-squidiff-build-$PID"

if (-not (Test-Path -LiteralPath (Join-Path $resolvedSquidiff 'dist\Squidiff-1.0.8-py3-none-any.whl') -PathType Leaf)) {
    throw "The Squidiff 1.0.8 wheel was not found under '$resolvedSquidiff\dist'."
}

& podman --connection $ScPerturBenchConnection rm -f $container 2>$null | Out-Null
try {
    & podman --connection $ScPerturBenchConnection run `
        --name $container `
        --entrypoint /bin/bash `
        --volume "${resolvedSquidiff}:/opt/Squidiff:ro" `
        $BaseImage `
        -lc 'set -eu
if ! conda env list | awk ''$1 == "squidiff" { found=1 } END { exit !found }''; then
    conda create -n squidiff --clone scarches -y
fi
conda run --no-capture-output -n squidiff python -m pip install --no-deps /opt/Squidiff/dist/Squidiff-1.0.8-py3-none-any.whl
conda install -n squidiff -c conda-forge rdkit=2022.03.2 -y
conda run --no-capture-output -n squidiff python -c "import rdkit, Squidiff, torch, scanpy, anndata, scipy, sklearn, matplotlib; print(\"squidiff-import-ok\", rdkit.__version__, torch.__version__, torch.version.cuda)"'
    if ($LASTEXITCODE -ne 0) {
        throw 'Squidiff environment setup failed.'
    }

    & podman --connection $ScPerturBenchConnection commit $container $Image
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to commit image '$Image'."
    }
} finally {
    & podman --connection $ScPerturBenchConnection rm -f $container 2>$null | Out-Null
}

Write-Host "Configured $Image with conda environment 'squidiff'."
