$ScPerturBenchCodeRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path

# Runtime root: Podman storage, conda environments and .scperturbench artifacts live
# outside this git checkout. Override it with SCPERTURBENCH_ROOT on another machine.
$ScPerturBenchRoot = $env:SCPERTURBENCH_ROOT
if ([string]::IsNullOrWhiteSpace($ScPerturBenchRoot)) {
    $ScPerturBenchRoot = 'C:\Users\Admin\myprojects\perturbench'
}
if (Test-Path -LiteralPath $ScPerturBenchRoot) {
    $ScPerturBenchRoot = (Resolve-Path -LiteralPath $ScPerturBenchRoot).Path
}
$ScPerturBenchDataRoot = Join-Path $ScPerturBenchRoot '.scperturbench'
$ScPerturBenchMachine = 'scperturbench'
$ScPerturBenchConnection = 'scperturbench-root'
$ScPerturBenchArchive = Join-Path $ScPerturBenchRoot 'scperturbench_v1.tar.gz'

function Set-ScPerturBenchPodmanEnvironment {
    $env:HOME = $ScPerturBenchRoot
    $env:USERPROFILE = $ScPerturBenchRoot
    $env:XDG_CONFIG_HOME = Join-Path $ScPerturBenchRoot '.config'
    $env:XDG_DATA_HOME = Join-Path $ScPerturBenchRoot '.local\share'

    New-Item -ItemType Directory -Force -Path $env:XDG_CONFIG_HOME, $env:XDG_DATA_HOME | Out-Null
}

function Invoke-ScPerturBenchPodman {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Connection,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Set-ScPerturBenchPodmanEnvironment
    & podman --connection $Connection @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Podman command failed with exit code $LASTEXITCODE."
    }
}

function Get-ScPerturBenchImage {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Connection
    )

    Set-ScPerturBenchPodmanEnvironment
    $images = @(
        & podman --connection $Connection images --format '{{.Repository}}:{{.Tag}}' 2>$null
    )
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    $preferred = @(
        'localhost/scperturbench:squidiff',
        'localhost/scperturbench:latest',
        'docker.io/bm2lab/scperturbench:latest',
        'localhost/scperturbench_v1:latest'
    )
    foreach ($candidate in $preferred) {
        if ($images -contains $candidate) {
            return $candidate
        }
    }

    $candidates = @(
        $images |
            Where-Object {
                $_ -match '(?i)scperturbench' -and
                $_ -notmatch '(?i)cpa'
            } |
            Sort-Object -Unique
    )
    if ($candidates.Count -gt 0) {
        return $candidates[0]
    }

    return $null
}
