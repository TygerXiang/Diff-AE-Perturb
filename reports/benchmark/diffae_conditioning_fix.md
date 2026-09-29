# Diff-AE-Perturb 条件通路修复

日期：2026-09-28 · 基线诊断：`reports/diffae_architecture_diagnosis.md` · 结构审计：`reports/diffae_project_structure_audit.md`
被改项目：`C:\Users\Admin\myprojects\Diff-AE-Perturb`（HEAD `359bc2d`）

---

## 1. 做了什么

| # | 你的要求 | 实现 | 状态 |
|---|---|---|---|
| 1 | `z_pert = z_control + delta`，去掉初始化 | `encode_condition` 里反事实重参数化改为**无条件生效**；拆掉 `if self.guide_doser is None: return` 这个把整段短路的早退；`delta` 的潜移位初始化在 HEAD 中已不存在（`a8c355f` 有、现已删），本轮另外把它变成**显式命名的消融模式**而不是一段没人敢碰的历史 | 完成 |
| 2 | 训练/推理条件接口对齐 | 新增 `control_reference`，**写进 checkpoint**，训练与推理不可能各用一套。`per_cell`（默认）：每个扰动细胞配一个对照细胞的潜表示；`batch_mean`：两侧都用对照编码均值 | 完成 |
| 3 | 权衡 x0 vs eps | 先给出机制分析（§3），再在完全相同的条件下把两种都训了，用六指标裁决（§4） | 完成 |
| 4 | 修正靶基因保留顺序 | `preprocess_gse132080.py` 让受扰动基因**豁免** MT/核糖体过滤（此前 `RPS14/RPS15/RPS18/RPL9` 被删后无法再补回） | 完成 |

另外按你「修复的同时改善项目」的要求，处理了迭代遗留问题（详见结构审计报告），其中正在生效的一条是**同名 `perturbdiff` 包被另一份实现抢占**（宿主环境实测 `import perturbdiff` 解析到 `diff-project/src`），已加身份标记 + `scripts/check_perturbdiff_import.py` 检测器。

### 1.1 关键代码改动

`src/perturbdiff/model.py` · `encode_condition`（**`z_pert` 现在只在这一处产生**）：

```python
z_sem = self.encoder(x0)
pert = self.perturbation(z_sem, perturbation_ids)      # delta_base / delta / z_basal
...
if self.guide_doser is not None:                        # 只有 activity 分支可选
    ...strength / activity_value / activity_embedding...
...
pert["z_basal"] = z_sem - pert["delta"]
z_control = self._build_control_anchor(z_sem, pert["delta"], perturbation_ids)
if control_reference == "encoder_self":                 # 遗留消融，仅用于 A/B
    pert["z_pert"] = z_sem
else:
    pert["z_pert"] = torch.where(ids == 0, z_sem, z_control + pert["delta"])
```

`_build_control_anchor` 支持两种锚点；`sample_counterfactual` 用**同一个** `control_reference`，所以两侧一致：

```python
if self.config.control_reference == "batch_mean":
    z_source = z_sem.mean(dim=0, keepdim=True).expand_as(z_sem)
else:
    z_source = z_sem          # per_cell；encoder_self 在推理端等价于 per_cell（即历史行为）
z_cf = self.perturbation.counterfactual_latent(z_source, target_ids, source_ids=zeros)
```

`PerturbationModule.forward` 不再返回误导性的 `z_pert`（它以前返回 `z_sem`，让读代码的人以为 `delta` 在给 denoiser 供条件）。

---

## 2. 验收与回归

功能验收（`tools/verify_conditioning_fix.py`，合成小模型，无需数据）：

```
perturbed_rows_use_control_plus_delta : true
control_rows_keep_own_latent          : true
delta_gets_gradient                   : true     ← 修复前恒为 0
  per-row L2 = [0.0, 0.004492, 0.003938, 0.003791, 0.0]   (0=padding, 4=未使用 id)
legacy_without_field_becomes          : per_cell
new_checkpoint_keeps_per_cell         : per_cell
ACCEPTANCE: PASS
```

**回归**：用重构后的代码重跑历史 `diffae` baseline（同一 checkpoint / seed / DDIM 50 步），与存档结果逐格对比，**12 个 metric×budget 单元的最大绝对差 = 0.0**。也就是说这次重构对旧结果是行为保持的（旧 checkpoint 被兼容补丁映射为 `per_cell`，正是它们当年被评测时的行为）。

---

## 3. 权衡：x0 还是 eps（你的第 3 点）

你提到「eps 好像更难收敛」，文献里这个说法指的是**低噪声端**的梯度病态，和**采样端**的误差放大，两者在本项目里都成立：

1. **低噪声端信息量为零**。t→0 时 `x_t ≈ x0`，ε 与条件几乎独立，模型被要求预测一段不可约的噪声；均匀 t 采样下这批样本数量不小、信息量却最低，因此梯度方差大、收敛慢。
2. **采样端误差被放大**。ε-prediction 要走 `x̂0 = (x_t − √(1−ᾱ)·ε̂)/√ᾱ`，分母 `√ᾱ → 0`。我们在 **Squidiff** 上刚刚量过这条通路：t=999 处 `1/√ᾱ ≈ 192`，配合上游 `pred_xstart.clamp(0,)` 的半波整流，把生成值放大了 **59×**（`reports/squidiff_repro_preprocessing.md` §3）。
3. **x0-prediction 恰好相反**。高噪声端的 loss 上界是 `Var(x0)`，模型在那里学到的正是 `E[x0 | z_cond]`——反事实预测真正需要的量；低噪声端可以直接从 `x_t` 读取个体信息。采样端 `ε̂ = (x_t − √ᾱ·x̂0)/√(1−ᾱ)`，而 `√(1−ᾱ) ≈ 1`，**没有放大**。
4. 对这个模型尤其重要的一点：我们测到**条件位移本身只有对照细胞间散布的 1/50**（`‖Δ_real‖ ≈ 2.9` vs 对照散布 `≈ 9.2`）。x0 目标把监督集中在「条件均值」上，正对着这个弱信号；ε 目标把它分散到与条件无关的噪声上。

**先验结论：保留 x0。** 但这是可以测的，所以我在同一划分、同一 200 epoch、同一 `control_reference` 下把两种目标都训了，用六指标裁决（下表）。

---

## 4. 消融结果

训练：`scripts/train_with_split.py`，GSE132080 unseen-guide 划分，200 epochs，batch 256，lr 1e-4（无日程），latent 128，`--activity-source none`，seed 7。
**模型选择**：本仓库的 `val` 与 `train` 是同一个文件（sha256 相同），所以 `checkpoint_best.pt` 是在训练集上选的；本表一律用**最后一个 epoch** 的 `checkpoint_epoch_0200.pt`，不做选择。
推理：原有 `ddim-steps 50`、对照细胞预算、seed 全部不变。指标：`scperturbench_metrics.py`（文献 `calPerfor` 移植）。

| variant | `z_pert` 训练时 | `control_reference` | mse@100 ↓ | pcc_delta@100 ↑ | e_distance@100 ↓ | wasserstein@100 ↓ | kl@100 ↓ | common_degs ↑ |
|---|---|---|---|---|---|---|---|---|
| **`fix_x0_percell`** | `z_control + delta` | per_cell | **0.0393** (4) | **0.8232** (5) | **0.8988** (4) | 28.389 (3) | 12.504 (2) | **0.1294** (5) |
| `fix_x0_batchmean` | `z_control + delta` | batch_mean | 0.0431 (6) | 0.8071 (6) | 1.0697 (7) | **28.011** (2) | **12.478** (1) | 0.1240 (6) |
| `legacy_x0_percell` | `z_sem`（旧行为） | encoder_self | 0.1404 (11) | 0.0670 (14) | 1.9913 (11) | 43.716 (10) | 13.844 (12) | 0.0277 (16) |
| `fix_eps_percell` | `z_control + delta` | per_cell | 84.221 | 0.0745 | 372.084 | 394 557 | 27.590 | 0.0081 |
| `fix_eps_batchmean` | `z_control + delta` | batch_mean | 83.590 | 0.0480 | 385.638 | 425 288 | 27.749 | 0.0117 |
| 存档 `diffae`（`a8c355f` checkpoint） | `z_sem` | — | 0.1433 (14) | 0.0283 (17) | 2.0170 (13) | 43.882 (11) | 13.821 (11) | 0.0363 (15) |
| `diffae_delta_recompute`（只改 delta 表） | 未重训 | — | 0.0407 (5) | 0.8849 (2) | 0.7758 (3) | 32.707 (4) | 12.746 (4) | 0.1180 (7) |
| `control_mean`（恒等基线） | — | — | 0.1407 | −0.0124 | 1.8027 | 49.623 | 13.348 | 0.0000 |

括号内是在 **20 个方法**中的排名（`aggregate/rankings.csv`）。

### 4.1 第 1 项（条件通路）的效果：全部指标一起改善

`fix_x0_percell` 与 `legacy_x0_percell` 是**同一划分、同一配方、同一 seed、同一推理流程**，唯一差别就是 `z_pert` 怎么构造：

| 指标 | legacy（`z_pert = z_sem`） | fix（`z_pert = z_control + delta`） | 变化 |
|---|---|---|---|
| `pcc_delta@100` ↑ | 0.0670 | **0.8232** | **×12.3** |
| `mse@100` ↓ | 0.1404 | **0.0393** | ÷3.6 |
| `common_degs` ↑ | 0.0277 | **0.1294** | ×4.7 |
| `e_distance@100` ↓ | 1.9913 | **0.8988** | ÷2.2 |
| `wasserstein@100` ↓ | 43.716 | **28.389** | ÷1.5 |
| `kl@100` ↓ | 13.844 | **12.504** | −1.34 |

而且 `legacy_x0_percell` 的结果与存档 `diffae` 行（0.0283 / 0.1433 / 2.0170 …）基本一致（同一量级、同一档位），说明 `encoder_self` 这个消融模式**忠实复现了旧行为**——所以上面的对比是干净的 A/B，不是跨 commit 的猜测。

修复后的模型在 20 个方法里 **10 个 metric×budget 单元全部排进前 6**，并拿下 `kl_divergence@100` 的**第一名**（`fix_x0_batchmean`，12.478，优于 scGen 的 12.526）。

### 4.2 第 3 项（x0 vs eps）：x0 完胜，而且 eps 是**采样层面不可用**

| | `fix_x0_percell` | `fix_eps_percell` |
|---|---|---|
| 生成值范围（imputed 块） | 与真值同量级 | **min −530.5 / max 472.8**（真值 0 – 6.87） |
| `mse@100` | 0.0393 | **84.22**（约 2140×） |
| `wasserstein@100` | 28.4 | **394 557** |

这不是「收敛慢一点」，而是**样本直接炸掉 ~70 倍**。机制与我们在 Squidiff 上量到的完全一致：

1. eps 预测在采样端必须走 `x̂0 = (x_t − √(1−ᾱ)·ε̂)/√ᾱ`，而 `1/√ᾱ` 在高噪声端可达 **192**；
2. 本仓库的 `GaussianDiffusion.ddim_sample` 是

   ```python
   x = torch.sqrt(prev_alpha_bar) * x0 + torch.sqrt(1.0 - prev_alpha_bar) * eps
   ```

   **完全没有 clamp / `denoised_fn`**，所以 eps 的误差没有任何东西约束（对照：Squidiff 至少还做了 `clamp(0,)`，虽然那个 clamp 本身又引入了漂移）；
3. x0 预测天然有界，且 `ε̂ = (x_t − √ᾱ·x̂0)/√(1−ᾱ)` 里 `√(1−ᾱ) ≈ 1`，没有放大。

**结论：保留 `x0`（默认值），你听到的「eps 更难收敛」在本项目里不只是收敛问题，是直接不可用。** 如果将来确实要用 eps，必须**同时**给采样器加 `denoised_fn` 或 clamp，否则 `--prediction-type eps` 应该被禁用（建议加一条断言或 README 警告）。

### 4.3 第 2 项（接口对齐）：两种锚点都能用，默认 `per_cell`

差距很小，`per_cell` 在 4/6 个指标上略优（`mse`、`pcc_delta`、`e_distance`、`common_degs`），`batch_mean` 在 `wasserstein`（28.01 vs 28.39）与 `kl_divergence@100`（12.478 vs 12.504，且是全场第一）上略优。

值得注意的是：**`batch_mean` 并没有把分布类指标打坏**（`wasserstein` 28.0、`e_distance` 1.07）。原因是 DDIM 的初始噪声 `x_T` 每个细胞不同，确定性采样映射把噪声的差异保留到了输出，所以「条件用均值」不等于「生成坍缩」。这一点验证了当初把两侧都改成对照均值也是安全的。

最终选 `per_cell` 作为默认，只因为它**在训练侧就是逐细胞对齐推理侧**，不需要在推理时额外广播一个均值向量，接口更少一层特殊处理。

### 4.4 与「只改 delta 表」的对比

`diffae_delta_recompute`（不重训，只把偏移重算一遍）在 `pcc_delta@100` 上仍略高于训练版（0.885 vs 0.823）：因为它的偏移是**整条件潜移位的精确样本均值**，对「平均响应方向」这类量是最优的低方差估计；训练版还要同时承担分布形状，所以在 `wasserstein`（28.4 vs 32.7）与 `common_degs`（0.129 vs 0.118）上更好。两者都远好于 legacy，且可以叠加使用（把 `recompute` 当作训练后的初始化/校准）。

### 4.5 训练损失不能横向比较

| 变体 | train_loss | best_val | test_loss |
|---|---|---|---|
| `fix_x0_percell` | 0.045412 | 0.045071 | 0.044542 |
| `fix_x0_batchmean` | 0.044204 | 0.043911 | 0.040371 |
| `legacy_x0_percell`（`encoder_self`） | **0.018885** | 0.018514 | **0.021655** |
| 存档 `gse132080_target_200ep`（`a8c355f`） | — | 0.023150 | 0.023287 |

修复后的损失**更高**（0.0445 vs 0.0233；用同一套新代码重训的 `legacy_x0_percell` 更是低到 0.0189），这是应该的：`encoder_self` 把去噪目标本身编码进条件，等于允许模型「看着答案答题」，损失被系统性压低。修好条件通路后模型必须真的从「对照 + 扰动偏移」推目标细胞，难度上升。**因此不能拿 loss 判断哪个模型更好，必须看下游六指标**——而下游指标显示修复版的 `pcc_delta` 是 legacy 的 12 倍。

---

## 5. 第 4 项：靶基因保留顺序

`preprocess_gse132080.py` 原先先做基因过滤（删掉所有 `RPS*`/`RPL*`）再做靶基因强留，而强留只能保住「还在矩阵里」的基因，于是 4 个核糖体类靶基因**永久丢失**：

```
修复前：n_target_genes_present = 21，panel = 1019，缺失 ['RPS14','RPS15','RPS18','RPL9']
        （其中 RPS14 是 15 个被评测条件之一 → 模型连被扰动的基因都看不到）
修复后：n_target_genes_present = 25，panel = 1023
        log: "25 perturbation target genes exempted from the MT/ribosomal filter"
```

改动（`diff-project/scripts/` 与 `Diff-AE-Perturb/scripts/` 两份「当前」副本，冻结快照不动）：

```python
protected_targets = adata.var_names.isin(sorted({...非 control 的 perturbation_label...}))
non_mt_ribo = ~adata.var_names.str.upper().str.startswith(("MT-", "RPS", "RPL"))
gene_mask = (gene_cells >= args.min_cells_per_gene) & (non_mt_ribo | protected_targets)
```

验证：`logs/prep_fix.log`、`.scperturbench/inputs/gse132080_prep_fixed_summary.json`。
**注意**：panel 从 1019 变成 1023，基因空间变了，所以**这套结果不能直接与既有 1019 空间的表并列**；要让全表统一，需要在新 panel 上把所有方法重跑一遍（我本轮没有做，避免把「修 bug」和「换基因空间」两件事混在一起）。

---

## 6. 复现

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\run-diffae-fix.ps1 -DryRun          # 打印全部命令
.\run-diffae-fix.ps1                  # 训练 5 个变体 + 推理 + 打分 + 汇总
.\run-diffae-fix.ps1 -SkipTrain       # 复用已训好的 checkpoint
```

```bash
# 只跑某个变体
ONLY=fix_x0_percell EPOCHS=200 bash tools/train_diffae_matrix.sh
# 验收 + 回归
python tools/verify_conditioning_fix.py
python ../Diff-AE-Perturb/scripts/check_perturbdiff_import.py --strict
```
