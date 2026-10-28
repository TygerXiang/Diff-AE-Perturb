[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string[]]$Predictions,
    [Parameter(Mandatory = $true)][string[]]$Bundles,
    [Parameter(Mandatory = $true)][string[]]$References,
    [Parameter(Mandatory = $true)][string[]]$ContextNames,
    [string]$OutputDirectory = '',
    [string]$Image = 'localhost/scperturbench:squidiff',
    [string]$PodmanConnection = 'scperturbench-root',
    [string]$PerturbationColumn = 'target',
    [string]$ControlLabel = 'non-targeting'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
. (Join-Path $repoRoot 'scperturbench-common.ps1')
Set-ScPerturBenchPodmanEnvironment
$podmanConnectionName = $PodmanConnection

if ($Predictions.Count -eq 0 -or
    $Predictions.Count -ne $Bundles.Count -or
    $Predictions.Count -ne $References.Count -or
    $Predictions.Count -ne $ContextNames.Count) {
    throw 'Predictions, Bundles, References, and ContextNames must have the same non-zero length.'
}
if (@($ContextNames | Select-Object -Unique).Count -ne $ContextNames.Count) {
    throw 'ContextNames must be unique.'
}
foreach ($path in $Predictions + $Bundles + $References) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Input path not found: $path" }
}
foreach ($path in $Bundles) {
    if (-not (Test-Path -LiteralPath (Join-Path $path 'manifest.json'))) {
        throw "Bundle manifest.json not found in: $path"
    }
}
$useDefaultOutput = [string]::IsNullOrWhiteSpace($OutputDirectory)
if ($useDefaultOutput) {
    $OutputDirectory = Join-Path $repoRoot '.scperturbench\results\vcc2026_eval'
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$inputRoot = Join-Path $repoRoot ('.scperturbench\inputs\vcc2026_eval\' + [guid]::NewGuid().ToString('N'))
$envRoot = Join-Path $repoRoot '.scperturbench\vcc2026_eval\env'
New-Item -ItemType Directory -Force -Path $OutputDirectory, $inputRoot, $envRoot | Out-Null

$containerPred = @()
$containerBundles = @()
$containerRefs = @()
for ($i = 0; $i -lt $Predictions.Count; $i++) {
    $predDest = Join-Path $inputRoot ("pred_{0}.h5ad" -f $i)
    $refDest = Join-Path $inputRoot ("reference_{0}.h5ad" -f $i)
    $bundleDest = Join-Path $inputRoot ("bundle_{0}" -f $i)
    Copy-Item -LiteralPath $Predictions[$i] -Destination $predDest
    Copy-Item -LiteralPath $References[$i] -Destination $refDest
    Copy-Item -LiteralPath $Bundles[$i] -Destination $bundleDest -Recurse
    $containerPred += "/root/project/inputs/$([IO.Path]::GetFileName($predDest))"
    $containerRefs += "/root/project/inputs/$([IO.Path]::GetFileName($refDest))"
    $containerBundles += "/root/project/inputs/$([IO.Path]::GetFileName($bundleDest))"
}

$containerOut = '/mnt/vcc-output'
$cacheRoot = Join-Path $repoRoot '.scperturbench\cache'
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
$mounts = @(
    '-v', "$($PSScriptRoot.Replace('\','/')):/root/project/vcc2026_eval:ro",
    '-v', "$($inputRoot.Replace('\','/')):/root/project/inputs:ro",
    '-v', "$($OutputDirectory.Replace('\','/')):$containerOut",
    '-v', "$($envRoot.Replace('\','/')):/root/vcc2026_eval_env",
    '-v', "$($cacheRoot.Replace('\','/')):/root/.cache"
)
$py = @'
set -euo pipefail
if [ ! -x /root/vcc2026_eval_env/bin/python ]; then
  conda create -y -p /root/vcc2026_eval_env python=3.12 pip
fi
/root/vcc2026_eval_env/bin/python -m pip install -r /root/project/vcc2026_eval/requirements.lock
/root/vcc2026_eval_env/bin/python - <<'PY'
from cell_eval2 import __version__
from cell_eval2.competition import competition_payload
assert competition_payload()['rule_version'] == 3
print(f'cell-eval2={__version__}; rule_version=3')
PY
'@
$initArgs = @(
    'run', '--rm', '--userns=keep-id', '--security-opt', 'label=disable',
    '--workdir', '/root/project'
)
$initArgs += $mounts
$initArgs += @($Image, '-lc', $py)
Invoke-ScPerturBenchPodman -Connection $podmanConnectionName -Arguments $initArgs

function Quote-BashArgument([string]$Value) {
    $singleQuote = [string][char]39
    $doubleQuote = [string][char]34
    $replacement = $singleQuote + $doubleQuote + $singleQuote + $doubleQuote + $singleQuote
    return $singleQuote + $Value.Replace($singleQuote, $replacement) + $singleQuote
}
$quotedPred = @($containerPred | ForEach-Object { Quote-BashArgument $_ })
$quotedBundles = @($containerBundles | ForEach-Object { Quote-BashArgument $_ })
$quotedRefs = @($containerRefs | ForEach-Object { Quote-BashArgument $_ })
$quotedContexts = @($ContextNames | ForEach-Object { Quote-BashArgument $_ })
$evalCommand = @(
    'set -euo pipefail',
    'export PYTHONPATH=/root/project/vcc2026_eval',
    '/root/vcc2026_eval_env/bin/python /root/project/vcc2026_eval/vcc2026_eval.py',
    '--predictions ' + ($quotedPred -join ' '),
    '--bundles ' + ($quotedBundles -join ' '),
    '--references ' + ($quotedRefs -join ' '),
    '--contexts ' + ($quotedContexts -join ' '),
    '--outdir ' + $containerOut,
    '--pert-col ' + (Quote-BashArgument $PerturbationColumn),
    '--control ' + (Quote-BashArgument $ControlLabel)
) -join "`n"
$evalArgs = @(
    'run', '--rm', '--userns=keep-id', '--security-opt', 'label=disable',
    '--workdir', '/root/project'
)
$evalArgs += $mounts
$evalArgs += @($Image, '-lc', $evalCommand)
Invoke-ScPerturBenchPodman -Connection $podmanConnectionName -Arguments $evalArgs
