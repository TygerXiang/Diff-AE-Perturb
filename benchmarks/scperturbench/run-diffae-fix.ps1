[CmdletBinding()]
param(
    [string]$Dataset = 'gse132080',
    [int]$Seed = 7,
    [int]$Epochs = 200,
    [string[]]$Variants = @('fix_x0_percell', 'fix_eps_percell', 'fix_x0_batchmean', 'fix_eps_batchmean', 'legacy_x0_percell'),
    [switch]$SkipTrain,
    [switch]$SkipScoring,
    [switch]$DryRun
)

# Diff-AE-Perturb conditioning fix: train the ablation matrix, predict, score with the six
# metrics, and refresh the aggregate table.
#
# Variant naming is <arm>_<prediction-type>_<control-reference>:
#   fix_*      z_pert = z_control + delta is trained (the fix)
#   legacy_*   z_pert = z_sem, i.e. the original leaky conditioning, kept as an explicit
#              ablation so the A/B is run on the same split and the same recipe
#
# Training uses the project's own scripts/train_with_split.py.  Model selection is *not* used:
# this repo's val split is byte-identical to its train split, so checkpoint_epoch_0200.pt
# (final epoch) is the checkpoint that gets evaluated.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$runner = Join-Path $root 'run-bench.ps1'
$dataRoot = if ($env:SCPERTURBENCH_ROOT) {
    Join-Path $env:SCPERTURBENCH_ROOT '.scperturbench'
} else {
    'C:\Users\Admin\myprojects\perturbench\.scperturbench'
}
$repo = '/root/Diff-AE-Perturb'
$base = '/root/diff-project/reproducibility/prepared/splits_unseen_guide_seed7'
$prep = '/root/diff-project/reproducibility/prepared/gse132080_preprocessed.h5ad'
$ckpt = "$repo/runs/gse132080_target_200ep/checkpoint_best.pt"
$trainRoot = '/root/perturbench/.scperturbench/trained'
$resultsRoot = "/root/perturbench/.scperturbench/results/gse132080_6metrics/$Dataset"
$localResults = Join-Path $dataRoot "results\gse132080_6metrics\$Dataset"
$degTable = '/root/perturbench/.scperturbench/inputs/gse132080_bench/deg_table.json'
$testPath = "$base/gse132080_test.h5ad"
$trainPath = "$base/gse132080_train.h5ad"

function Invoke-InPodman {
    param([string]$Name, [string]$CondaEnv, [string]$Command, [switch]$Gpu)
    if ($DryRun) { Write-Host "[$Name] $Command"; return }
    if ($Gpu) { & $runner -CondaEnv $CondaEnv -Gpu -Name $Name -Command $Command }
    else { & $runner -CondaEnv $CondaEnv -Name $Name -Command $Command }
}

function Get-VariantSpec {
    param([string]$Variant)
    # legacy_x0_percell -> prediction x0, control_reference encoder_self
    $reference = if ($Variant -like 'legacy_*') { 'encoder_self' } else { ($Variant -split '_')[2] }
    $prediction = ($Variant -split '_')[1]
    return @{ prediction = $prediction; reference = $reference }
}

if (-not $SkipTrain) {
    foreach ($variant in $Variants) {
        $spec = Get-VariantSpec $variant
        Write-Host "=== train $variant (prediction_type=$($spec.prediction) control_reference=$($spec.reference)) ==="
        Invoke-InPodman -Name "train-$variant" -CondaEnv 'scarches' -Gpu -Command (
            "cd $repo && python scripts/train_with_split.py " +
            "--data-path $prep --train-path $trainPath --val-path $trainPath --test-path $testPath " +
            "--perturbation-col perturbation_label --target-strategy raw " +
            "--epochs $Epochs --batch-size 256 --lr 1e-4 --latent-dim 128 " +
            "--prediction-type $($spec.prediction) --control-reference $($spec.reference) " +
            "--activity-source none --device cuda --seed $Seed --save-every $Epochs " +
            "--output-dir $trainRoot/$variant > $trainRoot/$variant/train.log 2>&1"
        )
    }
}

foreach ($variant in $Variants) {
    $out = "$resultsRoot/$variant/seed$Seed"
    New-Item -ItemType Directory -Force -Path (Join-Path $localResults "$variant\seed$Seed") | Out-Null
    Write-Host "=== predict $variant ==="
    Invoke-InPodman -Name "predict-$variant" -CondaEnv 'scarches' -Gpu -Command (
        "python /root/perturbench/scperturbench_adapters.py --adapter diffae --method $variant " +
        "--seed $Seed --repo $repo --checkpoint $trainRoot/$variant/checkpoint_epoch_$('{0:D4}' -f $Epochs).pt " +
        "--device cuda --ddim-steps 50 --train-path $trainPath --test-path $testPath " +
        "--output-dir $out > $out/adapter.log 2>&1"
    )
}

if (-not $SkipScoring) {
    foreach ($variant in $Variants) {
        $out = "$resultsRoot/$variant/seed$Seed"
        Write-Host "=== score $variant ==="
        Invoke-InPodman -Name "score-$variant" -CondaEnv 'pertpyV7' -Command (
            "python /root/perturbench/scperturbench_metrics.py --result-h5ad $out/result.h5ad " +
            "--test-path $testPath --deg-table $degTable --output-dir $out/score " +
            "--method $variant --seed $Seed > $out/score.log 2>&1"
        )
    }
    Write-Host '=== aggregate ==='
    Invoke-InPodman -Name 'aggregate' -CondaEnv 'pertpyV7' -Command (
        "python /root/perturbench/scperturbench_aggregate.py --results-root /root/perturbench/.scperturbench/results/gse132080_6metrics " +
        "--dataset $Dataset --output-dir /root/perturbench/.scperturbench/results/gse132080_6metrics/$Dataset/aggregate"
    )
}
