# continuity_check 分析脚本（历史留档）

这 8 个脚本原本是 `reproducibility/continuity_check/` 这个 git worktree（commit `2a1fb63` 的检出）
里**未提交**的分析工具，是 2026-09-17 「continuity / 复现性」调查的全部产出代码。2026-09-28 清理
worktree 时把它们单独保留下来，原因有两个：

1. 它们不在任何 commit 里，删掉就真的没有了；
2. `reports/gse132080_implementation_continuity.md` 与 `reports/gse132080_original_vs_current.md`
   里的表格/图都由它们生成。

| 文件 | 作用 |
| --- | --- |
| `write_continuity_report.py` | 汇总 14 个复现 run，产出 `continuity_summary.{csv,json}` 与本节报告 |
| `build_continuity_comparison.py`、`build_final_comparison.py`、`build_two_condition_report.py` | 两条件/多族对比表 |
| `analyze_determinism.py`、`compare_checkpoints.py` | 定点探针与 checkpoint 差异分析 |
| `run_repeats.ps1` | 重复训练各族 run |
| `run_determinism_probe.ps1` | cuDNN on/off 定点探针 |

## 使用前必须改的地方

这些脚本按当时的环境硬编码了路径，运行前必须改：

- `run_repeats.ps1` / `run_determinism_probe.ps1` 顶部的 `$py` 与 `$d`：实测时指向
  `C:\Users\Admin\myprojects\diff-project` 及其项目内 conda 环境
  `diff-project\.conda\envs\perturbdiff-gse132080`。**`diff-project` 已删除**，请改成当前仓库
  路径与可用环境（见 `docs/runs_retention.md`）。
- `write_continuity_report.py` 读取的 run 目录：原先在 `runs/continuity_check/` 下，该目录已随
  2026-09-28 清理删除；汇总产物保存在 `reports/assets/run_artifacts/continuity_check/`。
- 代码快照需要时用 `git worktree add reproducibility/continuity_check 2a1fb63` 重建
  （`reproducibility/explicit_8179140` 同理，commit `8179140`）。

这些脚本保留原样、不做现代化改写：它们记录的是当时的实现，改动会让上面两份报告失去可追溯性。
