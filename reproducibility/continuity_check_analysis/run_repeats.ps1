# Sequential repeat runs to establish an empirical noise band per architecture family.
# Usage: powershell -ExecutionPolicy Bypass -File run_repeats.ps1
$ErrorActionPreference = "Continue"

$py = "C:\Users\Admin\myprojects\diff-project\.conda\envs\perturbdiff-gse132080\python.exe"
$w  = "C:\Users\Admin\myprojects\Diff-AE-Perturb"
$d  = "C:\Users\Admin\myprojects\diff-project"

$common = @(
    "--data-path", "$d\reproducibility\prepared\gse132080_preprocessed.h5ad",
    "--train-path", "$d\reproducibility\prepared\splits\gse132080_train.h5ad",
    "--val-path", "$d\reproducibility\prepared\splits\gse132080_val.h5ad",
    "--test-path", "$d\reproducibility\prepared\splits\gse132080_test.h5ad",
    "--perturbation-col", "guide_identity",
    "--target-strategy", "raw",
    "--epochs", "200",
    "--batch-size", "256",
    "--lr", "1e-4",
    "--latent-dim", "128",
    "--encoder-hidden-dims", "512,256",
    "--denoiser-hidden-dims", "1024,1024",
    "--prediction-type", "x0",
    "--reconstruction-loss", "mse",
    "--activity-source", "predicted",
    "--guide-activity-cnn-table", "$d\datasets\GSE132080\41587_2019_387_MOESM3_ESM\Table_S8_machine_learning_input.txt",
    "--activity-table", "$d\datasets\GSE132080\41587_2019_387_MOESM3_ESM\Table_S16_perturb-seq_screen_phenotypes.txt",
    "--activity-col", "relative_activity_day10",
    "--guide-sequence-length", "26",
    "--guide-strength-hidden-dims", "64,64",
    "--guide-strength-cnn-channels", "32,32",
    "--guide-strength-cnn-kernel-size", "4",
    "--guide-strength-cnn-dropout", "0.25",
    "--guide-strength-lambda", "0.1",
    "--lambda-semantic-align", "0.0",
    "--lambda-direction-align", "0.0",
    "--device", "cuda",
    "--num-workers", "0",
    "--seed", "7",
    "--save-every", "200"
)

$jobs = @(
    @{ name = "legacy_2a1fb63_repeat2_200ep";    script = "$w\reproducibility\continuity_check\scripts\train_with_split.py" },
    @{ name = "explicit_8179140_repeat2_200ep";  script = "$w\reproducibility\explicit_8179140\scripts\train_with_split.py" },
    @{ name = "current_worktree_repeat2_200ep";  script = "$w\scripts\train_with_split.py" }
)

foreach ($j in $jobs) {
    $out = "$w\runs\continuity_check\$($j.name)"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Output "########## START $($j.name) ##########"
    & $py $j.script @common --output-dir $out 2>&1 |
        Tee-Object -FilePath "$out\train.log" |
        Select-String -Pattern "^(epoch=(1|50|100|150|193|200) |best_epoch=)"
    Write-Output "########## END $($j.name) EXIT=$LASTEXITCODE ##########"
}
