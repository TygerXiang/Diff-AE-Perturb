[CmdletBinding()]
param(
    [string]$Dataset = 'gse132080',
    [int]$Seed = 7,
    # Stable upstream-architecture checkpoint (lr_anneal_steps=2e4).  Used as the source for
    # all three sampling policies unless -TrainFresh is given.
    [string]$CheckpointDir = '/root/perturbench/.scperturbench/results/gse132080_6metrics/gse132080/squidiff/seed7/squidiff',
    [int]$Steps = 20000,
    [switch]$TrainFresh,
    [switch]$ProbeUpstreamDefault,
    [switch]$SkipPrep,
    [switch]$SkipScoring,
    [switch]$DryRun
)

# Squidiff redone with its own Squidiff_reproducibility preprocessing.
#
#   1. prep                  reproduce Squidiff's prep_* recipe from the raw GSE132080 counts.
#                            Result: identical to the benchmark panel (1019/1019 genes, same
#                            cells, matrices equal to float32 precision) -- see prep_summary.json.
#   2. squidiff_repro        train (or reuse) the upstream architecture and sample through
#                            sample_squidiff.sampler.pred, i.e. process_xstart = pred_xstart.clamp(0,)
#   3. squidiff_repro_clip   same checkpoint, Squidiff's own clip_denoised=True branch
#   4. squidiff_repro_noclamp same checkpoint, guided-diffusion's original unclamped pred_xstart
#
# -ProbeUpstreamDefault runs the upstream lr_anneal_steps=1e5 schedule instead.  That schedule
# collapses on this dataset (loss -> 1.000 by step ~9e3, grad_norm -> 0.02); the run is kept as
# an explicit probe rather than the default because a collapsed checkpoint is not usable.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$runner = Join-Path $root 'run-bench.ps1'
$dataRoot = if ($env:SCPERTURBENCH_ROOT) {
    Join-Path $env:SCPERTURBENCH_ROOT '.scperturbench'
} else {
    'C:\Users\Admin\myprojects\perturbench\.scperturbench'
}
$base = '/root/diff-project/reproducibility/prepared/splits_unseen_guide_seed7'
$dataDir = '/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro'
$resultsRoot = "/root/perturbench/.scperturbench/results/gse132080_6metrics/$Dataset"
$localResults = Join-Path $dataRoot "results\gse132080_6metrics\$Dataset"
$trainPath = "$base/gse132080_train.h5ad"
$testPath = "$base/gse132080_test.h5ad"
$degTable = '/root/perturbench/.scperturbench/inputs/gse132080_bench/deg_table.json'

function Invoke-InPodman {
    param([string]$Name, [string]$CondaEnv, [string]$Command, [switch]$Gpu)
    if ($DryRun) { Write-Host "[$Name] $Command"; return }
    if ($Gpu) { & $runner -CondaEnv $CondaEnv -Gpu -Name $Name -Command $Command }
    else { & $runner -CondaEnv $CondaEnv -Name $Name -Command $Command }
}

if (-not $SkipPrep) {
    Write-Host '=== 1. Squidiff preprocessing (Squidiff_reproducibility port) ==='
    Invoke-InPodman -Name 'squidiff-prep' -CondaEnv 'squidiff' -Command (
        "python /root/perturbench/scperturbench_squidiff_prep.py " +
        "--raw-dir /root/diff-project/datasets/GSE132080 " +
        "--metadata /root/diff-project/datasets/GSE132080/GSE132080_cell_identities.csv.gz " +
        "--train-h5ad $trainPath --test-h5ad $testPath --benchmark-genes /root/perturbench/.scperturbench/inputs/gse132080_bench/genes.json " +
        "--output-dir $dataDir > $dataDir/prep.log 2>&1"
    )
}

$primary = "$resultsRoot/squidiff_repro/seed$Seed"
New-Item -ItemType Directory -Force -Path (Join-Path $localResults "squidiff_repro\seed$Seed") | Out-Null

if ($ProbeUpstreamDefault) {
    Write-Host '=== 2b. probe: upstream lr_anneal_steps=1e5 (expected to collapse) ==='
    Invoke-InPodman -Name 'squidiff-probe-1e5' -CondaEnv 'squidiff' -Gpu -Command (
        "python /root/perturbench/scperturbench_adapters.py --adapter squidiff_repro --method squidiff_probe_1e5 " +
        "--seed $Seed --train-path $trainPath --test-path $testPath --squidiff-data-dir $dataDir " +
        "--squidiff-steps 100000 --log-interval 1000 --output-dir $resultsRoot/squidiff_probe_1e5/seed$Seed " +
        "> $resultsRoot/squidiff_probe_1e5/seed$Seed/adapter.log 2>&1"
    )
}

if ($TrainFresh) {
    Write-Host "=== 2. squidiff_repro: train $Steps steps, sample via sampler.pred ==="
    Invoke-InPodman -Name 'squidiff-repro' -CondaEnv 'squidiff' -Gpu -Command (
        "python /root/perturbench/scperturbench_adapters.py --adapter squidiff_repro --method squidiff_repro " +
        "--seed $Seed --train-path $trainPath --test-path $testPath --squidiff-data-dir $dataDir " +
        "--squidiff-steps $Steps --log-interval 1000 --output-dir $primary > $primary/adapter.log 2>&1"
    )
    $CheckpointDir = "$primary/squidiff"
} else {
    Write-Host '=== 2. squidiff_repro: reuse the stable checkpoint, sample via sampler.pred ==='
    Invoke-InPodman -Name 'squidiff-repro' -CondaEnv 'squidiff' -Gpu -Command (
        "bash /root/perturbench/tools/run_squidiff_sampling_variants.sh $CheckpointDir " +
        "> /root/perturbench/variants.log 2>&1"
    )
}

if (-not $SkipScoring) {
    foreach ($method in @('squidiff_repro', 'squidiff_repro_clip', 'squidiff_repro_noclamp')) {
        Write-Host "=== 4. scoring $method ==="
        $out = "$resultsRoot/$method/seed$Seed"
        Invoke-InPodman -Name "score-$method" -CondaEnv 'pertpyV7' -Command (
            "python /root/perturbench/scperturbench_metrics.py --result-h5ad $out/result.h5ad " +
            "--test-path $testPath --deg-table $degTable --output-dir $out/score --method $method --seed $Seed " +
            "> $out/score.log 2>&1"
        )
    }
    Write-Host '=== 5. aggregating ==='
    Invoke-InPodman -Name 'aggregate' -CondaEnv 'pertpyV7' -Command (
        "python /root/perturbench/scperturbench_aggregate.py --results-root /root/perturbench/.scperturbench/results/gse132080_6metrics " +
        "--dataset $Dataset --output-dir /root/perturbench/.scperturbench/results/gse132080_6metrics/$Dataset/aggregate"
    )
}
