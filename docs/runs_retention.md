# `runs/` 保留策略与 2026-09-28 清理

这份文档记录 `runs/`（以及曾经与它并列的两棵重复工作树）里**什么该留、什么可以删、怎么恢复**。
背景见审计报告 `perturbench/reports/diffae_project_structure_audit.md` 的 S1 / S6 / S9：
同名文件多份存在、`runs/` 膨胀到 15 GB、以及一个正在生效的环境抢占。

## 1. 两条规则

**规则一：一次训练只保留 `checkpoint_best.pt` + 最后一个 epoch 的 checkpoint。**

`scripts/train_with_split.py --save-every N` 每 N 个 epoch 落一个 `checkpoint_epoch_NNNN.pt`，
外加 `checkpoint_best.pt`。本仓库所有消费者（`scripts/evaluate_gse283614_*.py`、
`scripts/evaluate_benchmark.py`）只读 `checkpoint_best.pt`；逐 epoch 的 loss 已经在 `metrics.json` 里。
保留「最后一个 epoch」是因为本仓库的 `splits_unseen_guide_seed7/gse132080_val.h5ad` 与 train 是同一份文件
（审计 S2），`best` 是在训练集上选的，做干净对比时应当用固定 epoch 的 checkpoint 而不是 `best`。

**规则二：整体删除「冒烟 run」和「已被后续重训取代的 run」。**

名单写在 `scripts/prune_runs.py::STALE_RUNS` 里——「陈旧」是一个决定，不是启发式，所以逐条列名并写理由。
脚本会先检查是否有被跟踪文件仍指向这些目录（`--check-references`），并且把每个被删 run 的小体积文本产物
（`metrics.json`、`perturbation_*.json`、manifest、图）归档到 `reports/assets/run_artifacts/`。

一键执行（默认 dry-run）：

```powershell
python scripts/prune_runs.py --check-references                        # 只报告，并校验文档引用
python scripts/prune_runs.py --check-references --apply --archive-meta reports/assets/run_artifacts
```

## 2. 2026-09-28 实际删了什么

`runs/` 顶层条目 23 → 15，体积 **14.8 GB → 2.6 GB**（释放 12.2 GB：169 个中间 checkpoint + 8 个 run 目录）。

整体删除的 run：

| run | 体积 | 理由 |
| --- | --- | --- |
| `continuity_check/` | 1.0 GB | 2026-09-17 continuity/determinism 调查的 14 个复现 run；结论已被 2026-09-28 条件接口诊断取代 |
| `gse283614_explicit_activity_log2fc_200ep/` | 801 MB | 已被 `..._controls_fixed_200ep` 重训取代 |
| `gse283614_no_activity_200ep/` | 800 MB | 已被 `..._controls_fixed_200ep` 重训取代 |
| `split_smoke_by_guide/` | 80 MB | 冒烟 split 的 h5ad 转储，可由 `scripts/create_h5ad_splits.py` 重建 |
| `gse132080_zbasal_sdelta_diralign_nomargin_20ep/` | 32 MB | 20 epoch 消融，无任何跟踪文件引用，已被 200 epoch run 取代 |
| `gse132080_split_smoke_gpu_616cb3b/` | 32 MB | 冒烟 run（只训了 1 个 epoch） |
| `gse132080_smoke_616cb3b/`、`smoke_compare_target_vs_sgrna/` | 15 KB | 冒烟/单文件探针 |

同时删除的重复工作树（不在 `runs/` 下，属于审计 S6）：

| 路径 | 说明 |
| --- | --- |
| `reproducibility/continuity_check/` | commit `2a1fb63` 的 **git worktree**，含完整 `src/`、`scripts/`、`reports/` 拷贝 |
| `reproducibility/explicit_8179140/` | commit `8179140` 的 **git worktree** |
| `runs/continuity_check/*/code/` | 训练 run 里再拷一份代码的 `code/` 目录 |

这两个 worktree 曾被误提交为 gitlink（index 里 mode `160000`，但没有 `.gitmodules`），正是
`git status` 里那句 `modified: reproducibility/continuity_check (untracked content)` 的来源；
现在 gitlink 条目已从 index 移除。重建方式：

```powershell
git worktree add reproducibility/continuity_check 2a1fb63
git worktree add reproducibility/explicit_8179140 8179140
```

worktree 里唯一不在任何 commit 中的内容——8 个分析脚本——已原样保存到
`reproducibility/continuity_check_analysis/`（含 `README.md` 说明需要改的硬编码路径）。

清理后，审计 S6 点名的三个文件各只剩一份：
`scripts/preprocess_gse132080.py`、`scripts/train_with_split.py`、`scripts/evaluate_benchmark.py`。

## 3. 保留了什么

- `runs/benchmarks/`（2.7 MB）：embedding / activity 分支的结构分析产物，`scripts/build_visual_report.py` 等直接读。
- `runs/gse132080_target_200ep/`、`gse132080_sgrna_200ep/`：GSE132080 的两个主 run（best + 末 epoch）与 `preliminary/`、`guide_scale_export/`。
- `runs/benchmark_activity_branch_*`：activity 分支消融（best + 末 epoch）。
- `runs/gse283614_*_controls_fixed_200ep/`：GSE283614 的两个正式 run（best + 末 epoch，各约 146 MB）。
- `runs/biological_validation/`（2.0 GB）：GSE283614 剂量 CV / endpoint 插值的 5 折 × 2 条件 checkpoint（best + 末 epoch）与生物学分析产物。
- `reports/assets/run_artifacts/`（1.8 MB, 63 文件）：被删 run 的指标与图，包括 continuity 调查的全部汇总产物。

恢复某个 run 的原始权重只能重训；恢复它的**数字**不需要重训——`metrics.json` 都在。

## 4. 环境：抢占的 `perturbdiff` 已解除

宿主 conda base 里原本有

```
C:\Users\Admin\miniconda3\Lib\site-packages\__editable__.perturbdiff-0.1.0.pth
    → C:\Users\Admin\myprojects\diff-project\src        （diff-project 已删除）
```

它让 `import perturbdiff` 静默解析到**另一个项目**的实现。2026-09-28 已
`python -m pip uninstall -y perturbdiff` 移除。当前状态是**故意不安装**这个包：
`pyproject.toml` 给 pytest 配了 `pythonpath = ["src"]`，`scripts/*.py` 与 `tests/*.py` 也都在
`import perturbdiff` 之前自己 prepend 了 `src`，所以入口都不依赖安装；而没有安装时，
裸 `import perturbdiff` 会**大声失败**，不再可能静默跑到别的实现上。

需要把本仓库装成 editable 时（可选）：

```powershell
cd C:\Users\Admin\myprojects\Diff-AE-Perturb
python -m pip install -e . --no-deps      # 依赖已在环境中，避免解析器动 torch
python scripts/check_perturbdiff_import.py --strict   # 应只有本仓库一份
```

## 5. 已知未处理项

1. `.pytest_cache/` 目录 ACL 异常（连 ACL 都读不到，`Remove-Item`/`rd` 均 Access denied），
   疑似容器写入留下的目录。它已被 `.gitignore` 忽略，需要管理员权限才能删除：
   `takeown /f .pytest_cache /r` 后再删。
2. GSE132080 的 `val` 与 `train` 仍是同一份文件（审计 S2）。本仓库没有改它，因为改划分等于换配方；
   做对比时请用末 epoch checkpoint 并记录，或按 guide 再切一个真正的 val。
3. 本轮 A/B（`fix_x0_*` / `legacy_x0_*` / `fix_eps_*`）的 checkpoint 与分数在
   `perturbench/.scperturbench/`（该目录不在 git 下）。把这几份 benchmark 报告纳入版本控制仍是待办。
