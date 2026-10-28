[CmdletBinding()]
param(
    [string]$Image = 'localhost/scperturbench:squidiff',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run-bench.ps1'
$dataRoot = if ($env:SCPERTURBENCH_ROOT) {
    Join-Path $env:SCPERTURBENCH_ROOT '.scperturbench'
} else {
    'C:\Users\Admin\myprojects\perturbench\.scperturbench'
}
$prefix = '/root/perturbench/.scperturbench/envs/primeflow'
$logPath = Join-Path $dataRoot 'logs\primeflow_setup.log'
$primeflowRef = '443295258e364e394b60de27a861d8d1cf3b1fa6'
$perturbenchRef = 'c84038bc1ea409aa54f3832cfa6f34f5059adf0c'
$command = @"
mkdir -p /root/perturbench/.scperturbench/logs
exec > '/root/perturbench/.scperturbench/logs/primeflow_setup.log' 2>&1
set -eux
if [ ! -x '$prefix/bin/python' ]; then
  conda create -y -p '$prefix' python=3.12 pip
fi
if ! conda run --no-capture-output -p '$prefix' python -c 'import torch, primeflow, perturbench; assert torch.cuda.is_available(); x=torch.randn(128,128,device="cuda"); x@x' >/dev/null 2>&1; then
  conda run --no-capture-output -p '$prefix' python -m pip install --upgrade pip
  conda run --no-capture-output -p '$prefix' python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  conda run --no-capture-output -p '$prefix' python -m pip install 'primeflow @ git+https://github.com/altoslabs/primeflow@$primeflowRef'
fi
src=/root/perturbench/.scperturbench/cache/perturbench-src
mkdir -p /root/perturbench/.scperturbench/cache
if [ ! -f \"$src/src/perturbench/configs/hydra/default.yaml\" ]; then
  if [ ! -d \"$src/.git\" ]; then
    git init \"$src\"
    git -C \"$src\" remote add origin https://github.com/altoslabs/perturbench.git
  fi
  git -C \"$src\" fetch --depth 1 origin '$perturbenchRef'
  git -C \"$src\" checkout --detach FETCH_HEAD
fi
conda run --no-capture-output -p '$prefix' python -c 'import torch, primeflow, perturbench; print("primeflow=", primeflow.__file__); print("perturbench=", perturbench.__file__); print("torch=", torch.__version__, "cuda=", torch.version.cuda); assert torch.cuda.is_available(); x=torch.randn(128,128,device="cuda"); print("cuda_matmul=", float((x@x).mean()))'
"@

Write-Host "Preparing pinned PrimeFlow environment at $prefix"
$runnerParams = @{
    Image = $Image
    Name = 'setup-primeflow'
    Command = $command
    DryRun = [bool]$DryRun
}
& $runner @runnerParams
if ($LASTEXITCODE -ne 0) {
    if (Test-Path -LiteralPath $logPath) { Get-Content -LiteralPath $logPath -Tail 50 }
    throw "PrimeFlow environment setup failed with exit code $LASTEXITCODE; see $logPath"
}
