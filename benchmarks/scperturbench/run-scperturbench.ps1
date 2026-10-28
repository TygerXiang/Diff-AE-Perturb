[CmdletBinding()]
param(
    [string]$Image = '',
    [switch]$Gpu,
    [string[]]$Command
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

$runtimeRoot = Join-Path $ScPerturBenchRoot '.scperturbench'
$resultsPath = Join-Path $runtimeRoot 'results'
$inputsPath = Join-Path $runtimeRoot 'inputs'
$cachePath = Join-Path $runtimeRoot 'cache'
New-Item -ItemType Directory -Force -Path $resultsPath, $inputsPath, $cachePath | Out-Null

$runArguments = @(
    'run',
    '--rm',
    '--replace',
    '--name',
    'scperturbench-session',
    '--ipc=host',
    '--ulimit',
    'nofile=65536:65536',
    '--workdir',
    '/root/project',
    '--entrypoint',
    '/bin/bash',
    '--volume',
    "$resultsPath`:/root/project/results",
    '--volume',
    "$inputsPath`:/root/project/inputs",
    '--volume',
    "$cachePath`:/root/.cache"
)

if ($Gpu) {
    $runArguments += @(
        '--gpus', 'all',
        '--env', 'LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nv_dispig.inf_amd64_f4c7a2fd13e0f763:/usr/lib/wsl/lib:$LD_LIBRARY_PATH'
    )
}

if ($Command -and $Command.Count -gt 0) {
    $runArguments += $Image
    $runArguments += $Command
} else {
    $runArguments += @('-it', $Image, '/bin/bash')
}

Write-Host "Starting $Image on connection $ScPerturBenchConnection"
& podman --connection $ScPerturBenchConnection @runArguments
if ($LASTEXITCODE -ne 0) {
    throw "Container exited with code $LASTEXITCODE."
}
