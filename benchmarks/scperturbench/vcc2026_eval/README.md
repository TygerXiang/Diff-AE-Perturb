# VCC 2026 Evaluation Reproduction

This folder runs Arc Institute's official VCC 2026 scorer on prediction
`.h5ad` files. It delegates all six metrics and score normalization to
`cell-eval2`; it does not reimplement them. The locked runtime is
`cell-eval2==0.18.0`, `pdex==0.3.0`, and `pytest==8.4.2`.

The metric specification says the official reference bundles were generated
with `cell-eval2 0.15.0`, but that release is not publicly available from Arc or
PyPI. The wrapper checks that the installed scorer's rule version and digest
match the bundle and loads the bundle's `config.yaml`. This checks scoring
policy compatibility; it cannot claim bit-for-bit package-version identity
with the unavailable 0.15.0 artifact.

## Inputs

For every context, provide one prediction `.h5ad`, its official reference
bundle, and the matching reference `.h5ad`. The bundle contains scoring anchors
and provenance, not the reference expression matrix. Prediction and reference
must have identical ordered genes and perturbation-label sets, including the
control label. Each prediction cell must contain non-negative integer raw
counts with total at most `1,000,000`.

The perturbation label column defaults to `target`; the control label defaults
to `non-targeting`. The launcher stages inputs in a unique directory under
`.scperturbench/inputs/vcc2026_eval/`, outside this source folder. It does not
download competition data.

## Run

From the repository root in PowerShell:

```powershell
.\vcc2026_eval\run-vcc2026-eval.ps1 -Predictions @(
  'C:\data\pred_A.h5ad',
  'C:\data\pred_B.h5ad',
  'C:\data\pred_C.h5ad'
) -Bundles @(
  'C:\data\bundle_A',
  'C:\data\bundle_B',
  'C:\data\bundle_C'
) -References @(
  'C:\data\reference_A.h5ad',
  'C:\data\reference_B.h5ad',
  'C:\data\reference_C.h5ad'
) -ContextNames @('A', 'B', 'C')
```

The existing Podman image and connection from `agent.md` are used. The first
run creates a persistent isolated environment in
`.scperturbench/vcc2026_eval/env/` and installs the locked dependencies. The
launcher mounts source read-only, stages input paths read-only, and writes
results to `.scperturbench/results/vcc2026_eval/` by default. Contract tests
can be run with:

```powershell
.\vcc2026_eval\run-vcc2026-tests.ps1
```

## Results and scoring

The output contains `contexts.csv`, `overall.json`, and `manifest.json`, plus
each context's input aggregate (`user_agg.csv`) and full official result
(`official_scores.json`). The manifest records scorer
version, rule version and digest, bundle identity, input hashes, contexts, and
runtime parameters. The overall score is the equal-weight mean of context
scores; each context score is the official equal-weight mean of its six
normalized members.

The competition normalization is `(score - baseline) / (replicate - baseline)`.
The mean-response baseline maps to zero and the replicate anchor maps to one.
Metric-specific caps and floors are applied inside the official metrics; the
wrapper does not clip the aggregate score.

## References and verification limits

- Adduri et al., *Virtual Cell Challenge 2026: Benchmarking zero-shot
  generalization across cellular contexts*, Cell (2026),
  <https://doi.org/10.1016/j.cell.2026.08.004>.
- Official implementation: <https://github.com/ArcInstitute/cell-eval2>.
- VCC 2026 metric specification:
  <https://github.com/ArcInstitute/cell-eval2/blob/main/docs/vcc2026_metrics/vcc2026-metrics.md>.
- Official metric-specification commit:
  <https://github.com/ArcInstitute/cell-eval2/commit/ba6aa94d606b3c0d80851caa3825080d7b7a66e8>.

Contract tests cover count validation, labels, gene-axis order, and metric
membership. A full scoring run requires the official per-context bundles and
matching reference `.h5ad` files; the wrapper rejects a missing or mismatched
rule digest rather than scoring under uncertain rules.
