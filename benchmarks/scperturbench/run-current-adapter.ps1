$ErrorActionPreference = 'Stop'
$img = 'localhost/scperturbench:squidiff'
$codeRoot = $PSScriptRoot
$repo = (Resolve-Path -LiteralPath (Join-Path $codeRoot '..\..')).Path
$runtimeRoot = $env:SCPERTURBENCH_ROOT
if ([string]::IsNullOrWhiteSpace($runtimeRoot)) {
    $runtimeRoot = 'C:\Users\Admin\myprojects\perturbench'
}
$dataRoot = Join-Path $runtimeRoot '.scperturbench'
$data = 'C:\Users\Admin\myprojects\diff-project'
$inner = @'
conda run --no-capture-output -n scarches python /root/perturbench/run_current_diff_ae_benchmark.py \
  --repo /root/Diff-AE-Perturb \
  --train-path /root/diff-project/reproducibility/prepared/splits_unseen_guide_seed7/gse132080_train.h5ad \
  --test-path /root/diff-project/reproducibility/prepared/splits_unseen_guide_seed7/gse132080_test.h5ad \
  --checkpoint /root/Diff-AE-Perturb/runs/gse132080_target_200ep/checkpoint_best.pt \
  --output-dir /root/perturbench/.scperturbench/results/gse132080_diff_ae_adapter \
  --output-prefix current_adapter_smoke \
  --device cuda \
  --eval-level guide \
  --target-strategy first_token \
  --ddim-steps 5 \
  --max-control-cells 128 \
  --max-metric-cells 256
'@
$args = @(
    '--connection', 'scperturbench-root',
    'run', '--rm', '--replace',
    '--name', 'scperturbench-adapter',
    '--ipc=host',
    '--workdir', '/root/perturbench',
    '--entrypoint', '/bin/bash',
    '--gpus', 'all',
    '--env', 'LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nv_dispig.inf_amd64_f4c7a2fd13e0f763:/usr/lib/wsl/lib:$LD_LIBRARY_PATH',
    '--volume', "${repo}:/root/Diff-AE-Perturb",
    '--volume', "${codeRoot}:/root/perturbench",
    '--volume', "${dataRoot}:/root/perturbench/.scperturbench"
)
if (Test-Path -LiteralPath $data) {
    $args += @('--volume', "${data}:/root/diff-project:ro")
}
$args += @($img, '-lc', $inner)
& podman @args
if ($LASTEXITCODE -ne 0) { throw "adapter failed with code $LASTEXITCODE" }
