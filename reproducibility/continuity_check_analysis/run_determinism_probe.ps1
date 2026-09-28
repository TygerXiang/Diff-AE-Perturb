# Determinism control for the legacy (2a1fb63) implementation.
# The only cuDNN-backed op in the whole model is the 2D convolution inside
# GuideStrengthEncoder. Everything else is Linear / BatchNorm1d, which are
# deterministic on CUDA. So disabling cuDNN isolates the sole nondeterministic
# source and tests whether it explains the residual gap to the historical run.
param(
    [string]$Variant = "A",              # A = default cuDNN, B = cudnn disabled
    [int]$Epochs = 10,
    [int]$Seed = 7
)

$ErrorActionPreference = "Continue"
$py  = "C:\Users\Admin\myprojects\diff-project\.conda\envs\perturbdiff-gse132080\python.exe"
$w   = "C:\Users\Admin\myprojects\Diff-AE-Perturb"
$d   = "C:\Users\Admin\myprojects\diff-project"
$src = "$w\reproducibility\continuity_check"

$tag = if ($Variant -eq "B") { "cudnn_off" } else { "cudnn_on" }
$out = "$w\runs\continuity_check\determinism_${tag}_${Epochs}ep"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# Materialise a code tree whose modules.py disables cuDNN when PERTURBDIFF_NO_CUDNN=1.
if ($Variant -eq "B") {
    $codeDir = "$out\code"
    New-Item -ItemType Directory -Force -Path "$codeDir\src\perturbdiff" | Out-Null
    New-Item -ItemType Directory -Force -Path "$codeDir\scripts" | Out-Null
    Copy-Item "$src\src\perturbdiff\*" "$codeDir\src\perturbdiff\" -Recurse -Force
    Copy-Item "$src\scripts\train_with_split.py" "$codeDir\scripts\" -Force

    # Insert the cuDNN switch AFTER the mandatory `from __future__` line.
    $modPath = "$codeDir\src\perturbdiff\modules.py"
    $lines = Get-Content $modPath
    $futureIdx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^from __future__ import') { $futureIdx = $i; break }
    }
    if ($futureIdx -lt 0) { throw "no __future__ import found in modules.py" }
    $inject = @(
        "",
        "import os as _os",
        "import torch as _torch",
        "if _os.environ.get('PERTURBDIFF_NO_CUDNN') == '1':",
        "    _torch.backends.cudnn.enabled = False"
    )
    $patched = $lines[0..$futureIdx] + $inject + $lines[($futureIdx + 1)..($lines.Count - 1)]
    Set-Content -Path $modPath -Value $patched -Encoding UTF8
    $env:PERTURBDIFF_NO_CUDNN = "1"
    $script = "$codeDir\scripts\train_with_split.py"
} else {
    Remove-Item Env:\PERTURBDIFF_NO_CUDNN -ErrorAction SilentlyContinue
    $script = "$src\scripts\train_with_split.py"
}

Write-Output "variant=$tag epochs=$Epochs seed=$Seed script=$script"
Write-Output "PERTURBDIFF_NO_CUDNN=$($env:PERTURBDIFF_NO_CUDNN)"

& $py $script `
  --data-path "$d\reproducibility\prepared\gse132080_preprocessed.h5ad" `
  --train-path "$d\reproducibility\prepared\splits\gse132080_train.h5ad" `
  --val-path "$d\reproducibility\prepared\splits\gse132080_val.h5ad" `
  --test-path "$d\reproducibility\prepared\splits\gse132080_test.h5ad" `
  --perturbation-col guide_identity `
  --target-strategy raw `
  --output-dir $out `
  --epochs $Epochs --batch-size 256 --lr 1e-4 `
  --latent-dim 128 --encoder-hidden-dims 512,256 --denoiser-hidden-dims 1024,1024 `
  --prediction-type x0 --reconstruction-loss mse `
  --activity-source predicted `
  --guide-activity-cnn-table "$d\datasets\GSE132080\41587_2019_387_MOESM3_ESM\Table_S8_machine_learning_input.txt" `
  --activity-table "$d\datasets\GSE132080\41587_2019_387_MOESM3_ESM\Table_S16_perturb-seq_screen_phenotypes.txt" `
  --activity-col relative_activity_day10 `
  --guide-sequence-length 26 `
  --guide-strength-hidden-dims 64,64 --guide-strength-cnn-channels 32,32 --guide-strength-cnn-kernel-size 4 --guide-strength-cnn-dropout 0.25 `
  --guide-strength-lambda 0.1 `
  --lambda-semantic-align 0.0 --lambda-direction-align 0.0 `
  --device cuda --num-workers 0 --seed $Seed --save-every 1000 2>&1 |
  Tee-Object -FilePath "$out\train.log"
Write-Output "EXIT=$LASTEXITCODE"
