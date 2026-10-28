# Diff-AE-Perturb 项目结构审计（迭代遗留问题）

日期：2026-09-28 · 范围：`C:\Users\Admin\myprojects\Diff-AE-Perturb` 及其关联的 `diff-project`
起因：你问「我的模型架构很复杂，是不是迭代时没管理好项目内容？」

---

## 0. 回答

**是，确实有迭代遗留问题**，而且其中一条是**当前正在生效的隐患**（不是历史遗迹）。但要说清楚两点：

1. 我说的「复杂」本来指的是**条件通路绕了三层**（`PerturbationModule` 返回一个 `z_pert`，`encode_condition` 又把它覆盖一次，`forward` 再消费一次，最后还有一个早退把整段短路）——**这不是模型表达能力的问题，是代码组织的问题**。它在上一轮直接造成了一个 bug：读代码的人（包括我）会以为 `delta` 在给 denoiser 供条件，实际上没有。
2. **结构问题不是模型无效的原因**。模型无效的原因是条件接口与训练目标不匹配（见 `reports/diffae_architecture_diagnosis.md`）。下面这些是「会让同类 bug 反复发生、并且已经让一次实验白跑」的问题。

审计命令与原始输出：`tools/structure_audit.sh` → `logs/structure_audit.log`。

---

## 1. 问题清单（按严重度排序）

### S1（正在生效的隐患）同名 `perturbdiff` 包被另一份实现抢占

镜像与宿主上共有 **4 份** `perturbdiff`：

```
/root/Diff-AE-Perturb/reproducibility/continuity_check/src/perturbdiff     ← 冻结快照
/root/Diff-AE-Perturb/reproducibility/explicit_8179140/src/perturbdiff     ← 冻结快照
/root/Diff-AE-Perturb/src/perturbdiff                                      ← 当前实现
/root/diff-project/src/perturbdiff                                         ← 另一份分叉实现
```

两份 `pyproject.toml` **都用 `name = "perturbdiff"`、`version = "0.1.0"`、`where = ["src"]`**，而且宿主的 conda 环境里有：

```
C:\Users\Admin\miniconda3\Lib\site-packages\__editable__.perturbdiff-0.1.0.pth
    → C:\Users\Admin\myprojects\diff-project\src
```

后果：在宿主机上 `python -c "import perturbdiff"` 解析到的是 **`diff-project` 那一份**（已实测）。两份的差异不只是新旧：

| | Diff-AE-Perturb | diff-project（抢占者） |
|---|---|---|
| `model.py` | 21 499 B | 16 115 B |
| `modules.py` | 7 870 B | 7 043 B |
| `losses.py` | 2 567 B | 4 071 B |
| `config.py` | 2 248 B | 1 523 B |
| 独有文件 | `gse283614*.py` | `dist_util.py` `fp16_util.py` `logger.py` `nn.py` `resample.py` `respace.py`（**Squidiff / guided-diffusion 的五个文件被拷进来了**） |

也就是说抢占者是一个把 Squidiff 的扩散工具箱混进 `perturbdiff` 的混合体。任何**没有**自己 prepend `src` 的入口（notebook、`python -c`、pytest、某些分析脚本）都会静默跑到另一份代码上。

**本轮修复**（不动另一份实现，避免破坏 `diff-project`）：
- `src/perturbdiff/__init__.py` 增加 `__project__ = "Diff-AE-Perturb"` 与 `__project_root__` 标记；
- 新增 `scripts/check_perturbdiff_import.py`：按解析顺序列出所有可被 bare import 命中的 `perturbdiff`，指出哪一个会赢，`--strict` 时不确定就非零退出。**宿主实测：`1. [competing copy] C:\Users\Admin\myprojects\diff-project\src\perturbdiff <-- wins`，exit=1**，问题可见了。

**建议的环境修复**（我没有擅自执行，因为它会影响 `diff-project`）：
```powershell
python -m pip uninstall -y perturbdiff
cd C:\Users\Admin\myprojects\Diff-AE-Perturb ; python -m pip install -e .
```

---

### S2（已造成一次无效实验）`val` 与 `train` 是同一个文件

```
8b680d39ba6c...  splits_unseen_guide_seed7/gse132080_train.h5ad
8b680d39ba6c...  splits_unseen_guide_seed7/gse132080_val.h5ad      ← sha256 完全相同
47cd45231fd7...  splits_unseen_guide_seed7/gse132080_test.h5ad
```

后果：`train_with_split.py` 用 `val_loss` 选 `checkpoint_best.pt`（第 605–617 行），而 val 就是 train，所以**模型选择完全在训练集上做**。`runs/gse132080_target_200ep/metrics.json` 里 `best_epoch=198`（共 200）正是训练损失单调下降的表现，而不是泛化。

**本轮修复**：没有改代码（改选择标准会改变配方），而是**在我自己的对比实验里改用最后一个 epoch 的 checkpoint**（`checkpoint_epoch_0200.pt`）并明确记录，避免选择偏差。**建议**：从 train 里按 guide 再切一个真正的 val（unseen guide 与 test guide 不重叠），或干脆改为按 test 上的固定 epoch 报告。

---

### S3 条件通路三处赋值、含义两套

修复前：
- `PerturbationModule.forward`（`modules.py:91`）返回 `z_pert = z_sem`；
- `PerturbDiffusionModel.encode_condition`（`model.py:183`）在 `guide_doser is not None` 时又把它覆盖成 `z_control + delta`，否则在第 147–152 行**直接 return**，于是 `z_pert` 永远是 `z_sem`；
- `forward` / `training_losses` 消费 `condition["z_pert"]`。

同一个名字在两处被赋成完全不同的东西，其中一条路还是死代码——**这正是那个 bug 能活下来的原因**。

**本轮修复**：`z_pert` 现在**只在 `encode_condition` 一处产生**；`PerturbationModule.forward` 不再返回它（并在 docstring 里说明原因）；早退短路被拆掉，只有 guide/activity 分支是可选的了。

---

### S4 反事实重参数化被一个无关条件挡住

`if self.guide_doser is None: return` —— 也就是说**只要不开 sgRNA/activity 编码器，`z_pert = z_control + delta` 这条路根本走不到**。而 GSE132080 那个 checkpoint 恰恰就是 target-only（`activity_source=none`）。一个功能的可达性被另一个功能的开关决定，是典型的迭代叠加产物。

**本轮修复**：解耦（见 S3）。

---

### S5 初始化时机错误，且函数已被删除但影响仍在

`a8c355f` 的 `initialize_perturbation_from_latent_shift` 用**当时那个随机初始化的编码器**算 `delta = mean(z_sem|g) − mean(z_sem|control)`，在训练开始前调用一次；此后 `delta` 只被尺度不变的 `(1−cos)` 正则触及，于是**永远停在初始化尺度**（实测比训练后编码器的真实潜移位小 ~50×）。HEAD 已经删掉了这个函数（好事），但**已发布的 checkpoint 里带着这个后果**，而且没有任何文档说明。

**本轮修复**：新增 `--control-reference encoder_self` 作为**显式、有名字的遗留消融模式**，把这个失败变成可复现、可对比的对象（而不是一段没人敢碰的历史）；`config.py` / `train_with_split.py` 都有明确注释标为 ablation-only。

---

### S6 文件四份起跳，改一处漏三处

| 文件 | 副本数 | 位置 |
|---|---|---|
| `preprocess_gse132080.py` | **4** | `diff-project/scripts/`、`Diff-AE-Perturb/scripts/`、`reproducibility/{continuity_check,explicit_8179140}/scripts/` |
| `train_with_split.py` | **5** | 上述 2 份 + `reproducibility/{continuity_check,explicit_8179140}/scripts/` + `runs/continuity_check/*/code/scripts/` |
| `evaluate_benchmark.py` | **3** | 上述 2 份 + `reproducibility/explicit_8179140/scripts/` |

其中 `reproducibility/*` 与 `runs/continuity_check/*/code/` 是**故意冻结的快照**（用于复现旧数字），**必须不修改**；但这样一来「当前实现」和「冻结快照」在同一棵树里长得一模一样，很容易改错。第 4 项修复（RPS/RPL 靶基因保留顺序）就同时存在于其中 2 份「当前」副本里，我只改了这两份，并在报告里点名冻结快照不要动。

**建议**：给冻结快照加一个醒目的 `FROZEN_SNAPSHOT.md` 或改成只读/移出工作树；把「当前」实现收敛到单一位置，其余用相对导入或安装包引用。

---

### S7 配置 schema 漂移靠读时打补丁

`compatible_config_from_state()` 在 `evaluate_benchmark.py` 与 `compare_with_scgen.py` 里各有一份（且两份逻辑本来就不同），靠 `hasattr` 逐个补缺失字段（`guide_encoder`、现在的 `control_reference`）。这解释了为什么「旧 checkpoint 没有 guide encoder」这件事要靠运行时打补丁才能读出来。

**本轮修复**：我给两份都补上了 `control_reference` 的兼容分支，并且新字段**写进 checkpoint**（`ModelConfig` 里），这样训练/推理不可能各自用不同的约定。**建议**：给 checkpoint 加一个 `schema_version` 字段，读时按版本迁移，而不是无限 `hasattr`。

---

### S8 静默容错掩盖状态字典不匹配

`load_model()` 用 `load_state_dict(..., strict=False)`。这次实测是 0 missing / 0 unexpected（还好），但 `strict=False` 会让「换架构后旧权重悄悄没加载」这种事故完全无声。

**建议**：改成 `strict=True`，或至少在 `strict=False` 时把 missing/unexpected 的 key 数量写进 manifest 并在非零时报错。

---

### S9 运行目录膨胀

`runs/` 有 **23 个顶层条目、共 15 GB**；单个 `runs/gse132080_target_200ep/` 就存了 **11 个 checkpoint × 16.5 MB**（`checkpoint_epoch_0020..0200` + `checkpoint_best`）。`reproducibility/` 里还有两份完整 `src/` 拷贝。

**建议**：`--save-every` 改成只存 best + last（我这次就用 `--save-every 200`）；把 `runs/` 里只作证据的目录归档压缩，或加 `.gitignore` + 定期清理脚本。

---

### S10 两棵工作树都带着未提交的分叉

审计时 `git status`：

```
Diff-AE-Perturb  M scripts/compare_with_scgen.py  M scripts/evaluate_benchmark.py
                 ?? causal_methods_proposal.md  ?? methods_0923*.docx
diff-project     M agent.md  M methods.md  M scripts/evaluate_benchmark.py
                 M scripts/preprocess_gse132080.py  M scripts/train_with_real_guide_activity.py
                 M src/perturbdiff/{config,guide_features,model}.py
```

也就是说**同名文件在两个仓库里同时存在未提交的、不同的改动**（`evaluate_benchmark.py`、
`preprocess_gse132080.py`、`src/perturbdiff/model.py` 都各有一份在改）。这也是为什么同一件事
（例如「靶基因保留」）需要在多处分别修：没有一个「当前实现」的单一真相源。

**建议**：先各自把工作树里的有效改动 commit 或丢弃，明确哪一份是当前实现；把
`reproducibility/` 下的快照标注为冻结并禁止修改。

---

## 2. 本轮实际改了什么

| 文件 | 改动 | 对应问题 |
|---|---|---|
| `src/perturbdiff/model.py` | `encode_condition` 重写：反事实重参数化**无条件**生效；`z_pert` 只在此处产生；新增 `encoder_self` 显式消融分支；`_build_control_anchor` 支持 `per_cell`/`batch_mean`；`sample_counterfactual` 与训练约定一致 | S3 S4 S5 |
| `src/perturbdiff/modules.py` | `PerturbationModule.forward` 不再返回误导性的 `z_pert` | S3 |
| `src/perturbdiff/config.py` | 新增 `control_reference`（写进 checkpoint，训练/推理不可能不一致） | S7 |
| `scripts/train_with_split.py` | 新增 `--control-reference {per_cell,batch_mean,encoder_self}` | S3 S5 |
| `scripts/evaluate_benchmark.py`、`scripts/compare_with_scgen.py` | 旧 config 兼容补 `control_reference="per_cell"`（保持历史数字不变） | S7 |
| `src/perturbdiff/__init__.py`、`scripts/check_perturbdiff_import.py` | 包身份标记 + 抢占检测（宿主实测能抓出 competing copy） | S1 |
| `scripts/preprocess_gse132080.py`、`diff-project/scripts/preprocess_gse132080.py` | 靶基因豁免 MT/核糖体过滤（修 RPS14/RPS15/RPS18/RPL9 被丢） | 第 4 项需求 |

**未改**（并说明理由）：
- **没删** `diff-project/src/perturbdiff`：它是 `diff-project` 自己的实现，删/改会破坏另一个项目，所以改用「检测 + 报告 + 给出环境修复命令」。
- **没改** `reproducibility/{continuity_check,explicit_8179140}/` 与 `runs/continuity_check/*/code/`：那是冻结快照，改了就没法复现旧数字。
- **没改** val==train 的选择逻辑：改它等于换配方，我改为在对比实验里用 final-epoch checkpoint 并记录。

## 3. 回归保证

重构不能悄悄改变历史结论，所以先做了回归检查：

用重构后的代码重跑历史 `diffae` baseline（同一个 `gse132080_target_200ep/checkpoint_best.pt`、同一 seed、同一 DDIM 50 步），与存档结果逐格对比：

```
metric_budget             value_new   value_old   abs_diff
common_degs@cutoff         0.036309    0.036309     0.0
e_distance@100             2.016981    2.016981     0.0
e_distance@all             1.800163    1.800163     0.0
kl_divergence@100         13.821154   13.821154     0.0
kl_divergence@all         37.431762   37.431762     0.0
kl_divergence_floor@100    1.787214    1.787214     0.0
kl_divergence_floor@all    0.674840    0.674840     0.0
mse@100                    0.143308    0.143308     0.0
mse@all                    0.016483    0.016483     0.0
pcc_delta@100              0.028314    0.028314     0.0
pcc_delta@all              0.023067    0.023067     0.0
wasserstein@100           43.882389   43.882389     0.0
MAX ABS DIFF = 0.0
```

另有功能级验收测试 `tools/verify_conditioning_fix.py`，其中最关键的一条是 **`delta` 现在能从扩散损失拿到梯度**（修复前恒为 0）：

```
perturbed_rows_use_control_plus_delta : true
control_rows_keep_own_latent          : true
delta_gets_gradient                   : true      (per-row L2 = 0.0045 / 0.0039 / 0.0038)
legacy_without_field_becomes          : per_cell
new_checkpoint_keeps_per_cell         : per_cell
ACCEPTANCE: PASS
```
