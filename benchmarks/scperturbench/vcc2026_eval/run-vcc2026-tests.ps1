[CmdletBinding()]
param(
    [string]$Image = 'localhost/scperturbench:squidiff',
    [string]$PodmanConnection = 'scperturbench-root'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
. (Join-Path $repoRoot 'scperturbench-common.ps1')
Set-ScPerturBenchPodmanEnvironment
$podmanConnectionName = $PodmanConnection
$envRoot = Join-Path $repoRoot '.scperturbench\vcc2026_eval\env'
$cacheRoot = Join-Path $repoRoot '.scperturbench\cache'
New-Item -ItemType Directory -Force -Path $envRoot, $cacheRoot | Out-Null

$py = @'
set -euo pipefail
if [ ! -x /root/vcc2026_eval_env/bin/python ]; then
  conda create -y -p /root/vcc2026_eval_env python=3.12 pip
fi
/root/vcc2026_eval_env/bin/python -m pip install -r /root/project/vcc2026_eval/requirements.lock
/root/vcc2026_eval_env/bin/python -m pytest -q /root/project/vcc2026_eval/tests
'@
$testArgs = @(
    'run', '--rm', '--userns=keep-id', '--security-opt', 'label=disable',
    '-v', "$($PSScriptRoot.Replace('\','/')):/root/project/vcc2026_eval:ro",
    '-v', "$($envRoot.Replace('\','/')):/root/vcc2026_eval_env",
    '-v', "$($cacheRoot.Replace('\','/')):/root/.cache",
    '--workdir', '/root/project'
)
$testArgs += @($Image, '-lc', $py)
Invoke-ScPerturBenchPodman -Connection $podmanConnectionName -Arguments $testArgs
