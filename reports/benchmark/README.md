# benchmark 报告（从 `perturbench/` 复制留档）

这 6 份报告原本只存在于 `C:\Users\Admin\myprojects\perturbench\reports\`。该目录**不在任何 git 仓库下**，
而且 `perturbench` 不在本项目的会话工作区里（工作区 = `Diff-AE-Perturb`），所以它们既没有版本控制、
也不会出现在文件浏览器里——2026-09-30 就发生过一次「报告文件找不到」。

2026-09-30 把这 6 份 `.md` 原样（字节一致，已用 sha256 校验）复制到本目录，纳入本仓库 git；
**工作副本与全部产物仍以 `perturbench/` 为准**，此处只是留档与可检索。

| 文件 | 内容 |
| --- | --- |
| **`external_methods_comparison.md`** | **只对四个外部方法（GEARS / scGen / CPA / Squidiff）的对比报告**，由 `scripts/build_external_comparison.py` 从 `aggregate/` 生成 |
| `external_methods_comparison.csv` | 上表的机器可读版（method, metric_budget, value, rank_of_19, direction） |
| `aggregate/` | 聚合表原始产物：`summary_wide.csv`（数值）、`rankings.csv`（19 方法 × 10 单元排名）、`report.md`（自动生成的 19 行报告）、`wins.csv`、`summary_stats.csv`、`summary_long.csv`、`excluded_methods.txt` |
| `diffae_architecture_diagnosis.md` | 条件接口诊断：`delta` 从未进入 denoiser、尺度停在初始化值；1019 vs 1022 panel |
| `diffae_conditioning_fix.md` | 四项修复（`z_pert=z_control+delta`、`control_reference`、x0 vs eps、靶基因保留）+ 干净 A/B |
| `diffae_project_structure_audit.md` | 结构审计 S1–S10（重复文件、环境抢占、`runs/` 膨胀、未提交分叉） |
| `gse132080_6metrics_benchmark.md` | 19 行 × 10 个 metric×budget 单元的主表、协议、口径偏差 |
| `scperturbench_conformance_audit.md` | 指标实现与文献 `calPerfor` 的一致性审计 |
| `squidiff_repro_preprocessing.md` | Squidiff 预处理比对与上游采样崩溃的根因 |

未复制的两份是原始训练日志（`squidiff_repro_1e5_collapse_log.txt`、
`..._seed17_log.txt`，各约 10 KB / 8 KB），仍在 `perturbench/reports/` 下。

## 读这些报告时的路径约定

- 报告里写的 `runs/...`、`scripts/...`、`src/perturbdiff/...` 指**本仓库**，
  但注意它们描述的是 2026-09-28 的状态：当时存在的若干 run 目录已在
  `docs/runs_retention.md` 记录的清理中删除（汇总产物在 `reports/assets/run_artifacts/`）。
- 报告里写的 `.scperturbench/...`、`perturbench/...` 指
  `C:\Users\Admin\myprojects\perturbench\`（benchmark 工作区，含 19 个方法的
  checkpoint、逐条件 `result.h5ad` 与聚合 CSV）。
- `diffae_conditioning_fix.md` 里的 `fix_*` / `legacy_*` 训练 run 位于
  `perturbench\.scperturbench\trained\`，不在本仓库。
