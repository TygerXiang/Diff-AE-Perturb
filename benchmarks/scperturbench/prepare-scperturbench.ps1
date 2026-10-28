[CmdletBinding()]
param(
    [switch]$LoadImage,
    [int]$Cpus = 10,
    [int]$MemoryMiB = 16384,
    [int]$DiskSizeGiB = 120,
    [string]$ArchivePath = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scperturbench-common.ps1')
Set-ScPerturBenchPodmanEnvironment

if (-not (Get-Command podman -ErrorAction SilentlyContinue)) {
    throw 'Podman was not found in PATH.'
}

$machineExists = $false
& podman machine inspect $ScPerturBenchMachine --format '{{.Name}}' 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    $machineExists = $true
}

if (-not $machineExists) {
    Write-Host "Creating Podman machine '$ScPerturBenchMachine'..."
    & podman machine init `
        --cpus $Cpus `
        --memory $MemoryMiB `
        --disk-size $DiskSizeGiB `
        --rootful `
        $ScPerturBenchMachine
    if ($LASTEXITCODE -ne 0) {
        throw 'Podman machine initialization failed.'
    }
}

$state = (& podman machine inspect $ScPerturBenchMachine --format '{{.State}}' 2>$null | Out-String).Trim()
if ($state.ToLowerInvariant() -ne 'running') {
    Write-Host "Starting Podman machine '$ScPerturBenchMachine'..."
    & podman machine start $ScPerturBenchMachine
    if ($LASTEXITCODE -ne 0) {
        throw 'Podman machine start failed.'
    }
}

Write-Host "Checking connection '$ScPerturBenchConnection'..."
Invoke-ScPerturBenchPodman -Connection $ScPerturBenchConnection -Arguments @(
    'info',
    '--format',
    'host={{.Host.OS}}/{{.Host.Arch}} cpus={{.Host.CPUs}} memory={{.Host.MemTotal}} graphroot={{.Store.GraphRoot}}'
)

$image = Get-ScPerturBenchImage -Connection $ScPerturBenchConnection
if ($LoadImage -and -not $image) {
    $archive = if ([string]::IsNullOrWhiteSpace($ArchivePath)) {
        $ScPerturBenchArchive
    } else {
        (Resolve-Path -LiteralPath $ArchivePath).Path
    }

    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        throw "Image archive not found: $archive"
    }

    Write-Host "Loading image archive: $archive"
    Invoke-ScPerturBenchPodman -Connection $ScPerturBenchConnection -Arguments @(
        'load',
        '--input',
        $archive
    )
    $image = Get-ScPerturBenchImage -Connection $ScPerturBenchConnection
}

Write-Host ''
Write-Host 'Available images:'
Invoke-ScPerturBenchPodman -Connection $ScPerturBenchConnection -Arguments @(
    'images',
    '--format',
    '{{.Repository}}:{{.Tag}} {{.Size}}'
)

if ($image) {
    Write-Host ''
    Write-Host "Selected scPerturBench image: $image"
} elseif ($LoadImage) {
    throw 'The archive loaded, but no scPerturBench image tag was detected.'
} else {
    Write-Host ''
    Write-Host 'No full scPerturBench image was detected. Re-run with -LoadImage.'
}
