[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [string]$CondaEnv = '',
    [string]$Image = 'localhost/scperturbench:squidiff',
    [switch]$Gpu,
    [switch]$DryRun,
    [string]$Name = 'scperturbench-bench'
)

$ErrorActionPreference = 'Stop'
$codeRoot = $PSScriptRoot
$repo     = (Resolve-Path -LiteralPath (Join-Path $codeRoot '..\..')).Path
$runtimeRoot = $env:SCPERTURBENCH_ROOT
if ([string]::IsNullOrWhiteSpace($runtimeRoot)) {
    $runtimeRoot = 'C:\Users\Admin\myprojects\perturbench'
}
$dataRoot = Join-Path $runtimeRoot '.scperturbench'
$data     = 'C:\Users\Admin\myprojects\diff-project'
$squidiff = 'C:\Users\Admin\myprojects\Squidiff'

# The historical `diff-project` host path has been removed from this machine, so the bind mount is
# added only when it exists.  Container-side paths for the older benchmark scripts
# (/root/diff-project/...) therefore require that mount to be restored; the GSE132080 activity
# ablation uses splits and inputs that already live inside this workspace instead.
$inner = $Command
if ($CondaEnv -eq 'primeflow') {
    $inner = "conda run --no-capture-output -p /root/perturbench/.scperturbench/envs/primeflow $Command"
} elseif ($CondaEnv) {
    $inner = "conda run --no-capture-output -n $CondaEnv $Command"
}

$args = @(
    '--connection', 'scperturbench-root',
    'run', '--rm', '--replace',
    '--name', $Name,
    '--ipc=host',
    '--workdir', '/root/perturbench',
    '--entrypoint', '/bin/bash',
    '--volume', "${codeRoot}:/root/perturbench",
    '--volume', "${dataRoot}:/root/perturbench/.scperturbench",
    '--volume', "${repo}:/root/Diff-AE-Perturb:ro"
)
foreach ($optional in @(@{ path = $data; target = '/root/diff-project' }, @{ path = $squidiff; target = '/root/Squidiff' })) {
    if (Test-Path -LiteralPath $optional.path) {
        $args += @('--volume', "$($optional.path):$($optional.target):ro")
    } else {
        Write-Host "[run-bench] skipping missing mount $($optional.path) -> $($optional.target)"
    }
}

if ($Gpu) {
    $args += @(
        '--gpus', 'all',
        '--env', 'LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nv_dispig.inf_amd64_f4c7a2fd13e0f763:/usr/lib/wsl/lib:$LD_LIBRARY_PATH'
    )
}

$args += @($Image, '-lc', $inner)

if ($DryRun) {
    Write-Host ($args -join ' ')
    exit 0
}

& podman @args
exit $LASTEXITCODE
