# GSE132080 六指标方法对比（scPerturBench 口径）

日期：2026-09-28 · seed 7 · 15 个 held-out guide 条件 · 1019 基因空间
指标实现：`scperturbench_metrics.py`（文献 `calPerfor` 逐条移植；每个方法都用文献代码交叉校验，
条件 × 指标逐值吻合——Squidiff 三个新结果为 75/75，见 §5）

---

## 1. 结论

在统一的 unseen-guide / seen-target 划分、统一 1019 基因空间、统一指标协议下：

- **GEARS、scGen、linear(Ridge) 三个方法在第一梯队**：GEARS 在 `mse@100`、`e_distance@100`、`pcc_delta@100` 上最好；scGen 在 `mse@all`、`kl_divergence`、`wasserstein` 上最好；linear_ridge 在 `common_degs`、`e_distance@all`、`pcc_delta@all` 上最好。
- **CPA 明显偏弱**：`e_distance`、`mse`、`pcc_delta` 都排在后部，只有 `wasserstein` 中等。
- **Diff-AE-Perturb（现有 `gse132080_target_200ep` checkpoint）在 6 个指标上都没有超过简单基线**：`mse`、`pcc_delta`、`e_distance`、`common_degs` 与 `control_mean` / `train_mean` 同级或更差，仅 `kl_divergence@all`、`wasserstein` 略好于 `control_mean`。它的 delta 相关性只有 0.023–0.028，而 GEARS/scGen/linear_ridge 在 0.74–0.90。
  **但这个 checkpoint 的问题已定位且已修好**（`reports/diffae_architecture_diagnosis.md`）：训练时 denoiser 的条件是 `z_pert = z_sem = encoder(x0)`（答案的编码），每个扰动的潜偏移 `delta` **从未进入 denoiser**，且其尺度停留在「用随机初始化编码器算出的初值」，比训练后编码器的真实潜移位小 ~50×；推理时条件接口换成 `enc(对照)+delta`，于是输出 ≈ 对照均值。**只重写 `delta` 表、其余全部冻结**（`diffae_delta_recompute`，偏移按训练好的编码器重算，只用训练细胞），`pcc_delta@100` 从 0.028 → **0.885**（仅次于 GEARS 0.896，高于 scGen/linear_ridge），10 个 metric×budget 单元全部进入前 5。
- **Squidiff 已补齐（见 §5）**：按 `Squidiff_reproducibility` 的预处理重做，并修掉上游采样路径的
  `pred_xstart` 截断后，首次给出可解释的六指标。但即使最好的变体
  `squidiff_repro_clip` 也只在 `kl_divergence@100`（17.16，优于 GEARS 的 20.33）与
  `pcc_delta@100`（0.337，排 6/12）上进入中游，`mse` / `e_distance` / `wasserstein` /
  `common_degs` 四项仍在榜尾，**没有拿到任何一项第一**。上游默认采样策略下的两个变体
  （`squidiff_repro`、`squidiff_repro_noclamp`）仍然不可用。

这也修正了此前的观察：现有 checkpoint 之所以「只改善平均表达轮廓、不改善分布级指标」，**不是**因为 activity/target 条件天生只能做到这一步，而是因为条件偏移 `delta` 从未被接进 denoiser、且尺度停在初始化值。把偏移给对之后，分布级指标（`e_distance` 2.02→0.78、`wasserstein` 43.9→32.7）与差异基因恢复（`common_degs` 0.036→0.118）一起改善。

---

## 2. 协议（与文献一致的部分）

沿用镜像内 `manuscript2/calPerformance_genetic.py::calPerfor`：

1. 指标在 **GT DEG 子集**上计算（按 `|t score|` 排序取 top-N；N=100，以及 "all"=1019 对应文献的 5000）。
2. 分布类指标（`edistance`/`wasserstein`/`sym_kldiv`）先把 imputed / control / stimulated 三组各子采样到 **≤2000 细胞**，`np.random.seed(42)`。
3. `mse` = pseudobulk 均值平方误差；`pearson_distance` 在 `calculateDelta`（减 control 均值）之后计算，本表报告 `pcc_delta = 1 − distance`（越高越好）。
4. `edistance` 用 **sqeuclidean**；`wasserstein` 用 OTT Sinkhorn 正则化 OT cost；`sym_kldiv` 逐基因高斯对称 KL 求均值后 `log2(x+1)`。
5. `common_degs` = 用文献 `getDEG_cutoff` 定义（`p_adj ≤ 0.01 & (FC ≥ 2 | FC ≤ 0.5)`）在 GT 与预测上分别取 DEG 集后的 **F1**（聚合函数未随镜像发布，见审计报告 §4.1）。

**交叉校验**：把我们的 `result.h5ad` 直接喂给镜像内的 `calPerfor`，3 条件 × 5 指标 = **15/15 吻合**（差异仅为文献自身 `round(...,4)`，最大 4.83e-5）；在 scGen 的预测上同样吻合。证据：`.../smoke/linear_pca_linear/seed7/score/crosscheck_literature.json`、`.../gse132080/scgen/seed7/score/crosscheck_literature.json`。

---

## 3. 主结果（15 条件均值，seed 7）

`pcc_delta` / `common_degs` 越高越好，其余越低越好。

| method | mse@100 | pcc_delta@100 | e_distance@100 | wasserstein@100 | kl_divergence@100 | common_degs |
|---|---|---|---|---|---|---|
| **gears** | **0.020826** | **0.895589** | **0.458947** | 35.580166 | 20.325686 | 0.153786 |
| **scgen** | 0.021268 | 0.863893 | 0.908575 | **27.299535** | **12.526211** | 0.198495 |
| **linear_ridge** | 0.024521 | 0.867429 | 0.513460 | 37.535861 | 13.091425 | **0.267264** |
| linear_pca | 0.064553 | 0.677450 | 0.993181 | 41.749075 | 13.191699 | 0.204616 |
| train_mean | 0.120322 | 0.465249 | 1.578395 | 47.505334 | 13.302791 | 0.036538 |
| cpa | 0.135282 | 0.191665 | 3.955410 | 37.033642 | 16.557743 | 0.051805 |
| control_mean | 0.140728 | -0.012392 | 1.802712 | 49.622519 | 13.348100 | 0.000000 |
| **diffae** | 0.143308 | 0.028314 | 2.016981 | 43.882389 | 13.821154 | 0.036309 |
| **diffae_delta_recompute** | 0.040743 | 0.884925 | 0.775817 | 32.707175 | 12.745897 | 0.118034 |
| diffae_delta_x50 | 0.136338 | 0.240854 | 1.965989 | 42.980988 | 13.498325 | 0.056217 |
| **fix_x0_percell** | 0.039266 | 0.823224 | 0.898848 | 28.389225 | 12.503866 | 0.129396 |
| **fix_x0_batchmean** | 0.043085 | 0.807111 | 1.069676 | **28.011030** | **12.477912** | 0.123950 |
| legacy_x0_percell | 0.140407 | 0.066973 | 1.991308 | 43.715528 | 13.844241 | 0.027667 |
| fix_eps_percell | 84.220563 | 0.074539 | 372.083588 | 394557.043750 | 27.590441 | 0.008085 |
| fix_eps_batchmean | 83.589559 | 0.047972 | 385.638328 | 425288.368750 | 27.749022 | 0.011718 |
| squidiff_repro_clip | 1.870213 | 0.337250 | 14.988506 | 276.422846 | 17.156833 | 0.038254 |
| squidiff_repro_noclamp | 63.641286 | 0.077157 | 466.065835 | 629770.379167 | 27.917804 | 0.004040 |
| squidiff（旧：ddim50 + 官方 clamp） | 664.127340 | -0.121157 | 352.566380 | 151812.482813 | 25.089771 | 0.069999 |
| squidiff_repro（完整 1000 步 + 官方 clamp） | 1005.041305 | -0.092636 | 435.705192 | 233915.982292 | 26.011909 | 0.070000 |

> **2026-09-28 追加**。`diffae_delta_recompute` / `diffae_delta_x50` 使用**同一个** Diff-AE-Perturb
> checkpoint，只重写每个扰动的潜偏移表（其余全部冻结）。`fix_*` / `legacy_*` 是**按修复后的代码
> 重训**的消融矩阵（同划分、同 200 epoch、同 seed，只改条件通路与预测目标）：
> `fix_x0_percell` 把 `pcc_delta@100` 从 legacy 的 0.067 提到 **0.823**、`mse` 从 0.140 提到
> **0.039**，并在 20 个方法中 10 个单元全部进前 6，拿下 `kl_divergence@100` 第一
> （`fix_x0_batchmean`）。`fix_eps_*` 是 eps 预测目标，**采样直接炸掉**（生成值 −530…473），
> 判为不可用。完整机制、验收与回归见 **`reports/diffae_conditioning_fix.md`** 与
> **`reports/diffae_architecture_diagnosis.md`**。
>
> `squidiff_repro*` 使用 Squidiff 自己的 `Squidiff_reproducibility` 预处理、上游默认架构、完整
> 1000 步 DDIM，只切换上游代码里 `process_xstart` 的截断策略。上游默认 `lr_anneal_steps=1e5`
> 在本数据上于 step≈9e3 崩坏（loss → 恒等 1.000，两个 seed 都复现），因此评分用 2e4 步的完整
> 退火日程。详见 `reports/squidiff_repro_preprocessing.md`。

全基因预算（`@all`，1019 基因）：

| method | mse@all | pcc_delta@all | e_distance@all | kl_divergence@all |
|---|---|---|---|---|
| scgen | **0.002984** | 0.795696 | 1.221925 | **35.495660** |
| linear_ridge | 0.003488 | **0.803045** | **0.477923** | 39.777511 |
| gears | 0.004495 | 0.744061 | 0.563596 | 41.706032 |
| linear_pca | 0.008478 | 0.557792 | 0.895351 | 39.862383 |
| train_mean | 0.013996 | 0.340420 | 1.289641 | 39.791117 |
| cpa | 0.015812 | 0.153094 | 4.935972 | 38.437993 |
| control_mean | 0.016079 | -0.011666 | 1.440864 | 39.784120 |
| diffae | 0.016483 | 0.023067 | 1.800163 | 37.431762 |

**Top-1 计数**（10 个 metric×budget 单元）：scgen 4、gears 3、linear_ridge 3。

> `kl_divergence` 的绝对值被少数“control 方差为 0”的诱导型基因主导（文献 `epsilon=1e-8`，可达 2^40 量级），因此排序信息有限；`kl_divergence_floor`（std 下限 0.1）是更可读的诊断列，例如 gears 0.618 / scgen 1.697 / squidiff 11.84。完整列见 `aggregate/summary_stats.csv`。

---

## 4. 方法配置（全部记录在各自 `adapter_manifest.json`）

| method | env | device | 配置 |
|---|---|---|---|
| diffae | scarches | GPU | `runs/gse132080_target_200ep/checkpoint_best.pt`（sha256 `8b71467e…`），DDIM 50 步，discrete-target 条件（checkpoint 无 guide encoder 权重） |
| scgen | scarches | GPU | 200 epochs, batch 128, n_hidden 800 / n_latent 100 / 2 层（沿用仓库 `compare_with_scgen.py` 配置） |
| linear_ridge | linearModel | CPU | `genetic_baseReg.py` 移植：scGPT 基因嵌入 + Ridge；4/25 个扰动基因缺嵌入，用可用扰动嵌入均值填充（manifest 记录） |
| linear_pca | linearModel | CPU | `linearModel.r` 移植：pseudobulk PCA-10 基因/扰动嵌入 + ridge 0.1 双线性解（`solve_y_axb`） |
| cpa | cpa | CPU | 100 epochs, batch 128, `recon_loss='gauss'`（log-norm 连续值），预测读 `obsm['CPA_pred']` |
| gears | gears | CPU | 15 epochs, hidden 64；自定义 `set2conditions` 覆盖同一划分；预测取 `gears.inference.evaluate(test_loader)['pred']`（细胞级） |
| train_mean / control_mean | linearModel | CPU | 文献参考基线，按 `generateH5ad` 用 control std 采样 |
| squidiff_repro | squidiff | GPU | Squidiff 自己的 `Squidiff_reproducibility` 预处理（等价于既有 1019 panel），上游架构 57.2M 参数，`lr_anneal_steps=2e4`（上游默认 1e5 会崩坏），完整 1000 步 DDIM，`sample_squidiff.sampler.pred` |
| squidiff_repro_clip | squidiff | GPU | 同上 checkpoint，`ddim_sample_loop(clip_denoised=True)`（Squidiff 自己的另一个分支） |
| squidiff_repro_noclamp | squidiff | GPU | 同上 checkpoint，`process_xstart` 恢复为 guided-diffusion 原版的恒等映射（对照） |
| diffae_delta_recompute | scarches | GPU | 同上 Diff-AE-Perturb checkpoint，**只**把每个扰动的潜偏移表重写为 `mean(z_sem\|c) − mean(z_sem\|对照)`（用训练好的编码器、只用训练细胞）；编码器/denoiser/DDIM 50 步/seed/对照预算全部不变 |
| diffae_delta_x50 | scarches | GPU | 同上 checkpoint，把原偏移表整体 ×50（只改尺度、不改方向，对照） |
| fix_x0_percell / fix_x0_batchmean | scarches | GPU | **按修复后的代码重训**（`--control-reference per_cell` / `batch_mean`，`--prediction-type x0`），200 epoch，用**最后一个 epoch** 的 checkpoint（本仓库 val==train，不做选择） |
| legacy_x0_percell | scarches | GPU | 同配方重训，但 `--control-reference encoder_self`（复现旧的 `z_pert = z_sem` 泄漏行为，作为 A/B 对照） |
| fix_eps_percell / fix_eps_batchmean | scarches | GPU | 同上，但 `--prediction-type eps`；**采样爆炸，判为不可用**（见 `reports/diffae_conditioning_fix.md` §4.2） |

设备差异的原因：RTX 5060 是 `sm_120`，`cpa` / `gears` / `linearModel` env 的 torch（2.0.1+cu117、2.3.0+cu121）无对应 kernel，实测 CUDA kernel 直接报 `no kernel image is available`，因此只能用 CPU；`scarches` / `squidiff`（torch 2.2.2+cu118）可执行 CUDA。

---

## 5. Squidiff 状态（已重做，进入主表）

完整报告：**`reports/squidiff_repro_preprocessing.md`**。要点：

1. **预处理不是瓶颈。** `Squidiff_reproducibility` 的 `prep_pertb_data.ipynb` 族配方
   （QC → 去 MT/RPS/RPL → `normalize_total(1e4)` → `log1p` → `highly_variable_genes(1000)`
   → 手留关注基因）从 raw 10x 复刻后，与本 benchmark 的 1019 基因面板
   **1019/1019 基因、16285/5403 细胞完全相同，矩阵差 max 3.46e-7**。上一轮的输入本来
   就是 Squidiff 自己的预处理结果。
2. **上游默认训练预算崩坏（可复现）。** `lr_anneal_steps=1e5` 两次独立运行都崩坏：seed 7 在
   step 8e3–9e3（loss 0.268 → 1.01 → 恒等 1.000），seed 17 在 step 1.1e4–1.2e4（loss 0.223 →
   0.364 → 恒等 1.000），两次都伴随 grad_norm 0.67 → 0.2 以下的单调衰减（退化为输出 0）。
   改用 2e4 步的完整退火日程可稳定收敛（loss 1.12 → 0.161）。原始日志
   `reports/squidiff_repro_1e5_collapse_log.txt` 与 `..._seed17_log.txt`。
3. **量级爆炸的根因是上游采样代码**：`Squidiff/diffusion.py::p_mean_variance` 的
   `process_xstart` 在 `clip_denoised=False` 时仍执行 `x.clamp(0,)`（guided-diffusion 原版
   是 `return x`）。`pred_xstart` 在高噪声步病态（`1/√ᾱ` 到 ~190），零均值巨尺度量被在 0
   处整流，每步注入 `≈0.4·σ_eps` 的正漂移，1000 步累积把生成值放大 **59×**
   （ALDOA 64 细胞：mean 13.40 vs 真实 0.226；`pred_x0_mean` 恒等于 `pred_x0_absmean`，
   即全部非负）。改成恒等（`none`）后漂移完全消失（mean −0.001），仅剩方差爆炸。
   另测得模型在高 t 只预测 78% 的噪声幅度（`γ≈0.778`），对应 3.17× 的 DDIM 放大。
4. **Squidiff 自己的 `clip_denoised=True` 分支是可用的**：量级正确（‖x‖₂ 21.4 vs 真实
   19.8），但把表达截到 1.0（真实 p99 2.77 / max 5.65），所以偏低；六指标见 §3 表。
5. **交叉校验**：三个新结果各自 **75/75（15 条件 × 5 指标）** 与镜像内文献 `calPerfor`
   逐值吻合（最大差 4.29e-5，来自文献自己的 `round(...,4)`）。过程中修正了
   `scperturbench_crosscheck.py` 里 `kl_divergence` 的列映射（文献对 `sym_kldiv` 会做
   `log2(x+1)`，应对应我们的 `value` 列而不是 `raw_distance`）。主表数值不受影响。

仍未解决的是**方法本身**：Squidiff 在这份数据上即便量级正确，也没有超过 `control_mean` /
`train_mean` 之外的任何简单基线。

---

## 6. 口径偏差（引用结果时必须同时引用）

1. **泛化轴**：unseen guide / **seen target**（沿用 `splits_unseen_guide_seed7`）。15/15 个 test 条件的 target 基因都在训练集中，6 个方法面对同一任务、彼此可比；但这不是 scPerturBench 的 unseen-perturbation 设定。
2. **基因空间**：1019（checkpoint 空间），非 hvg5000；文献 `numDEG=5000` 退化为“全部 1019”。
3. **数据集**：GSE132080 不在论文 12 个数据集内，数值与论文表格不可比。
4. **单 seed**：seed 7，未做多 seed 方差（`summary_stats.csv` 的 `sd` 为空）。
5. **GEARS 的预测细胞来源**：其 test loader 覆盖的是**训练划分**中携带这些 target 基因的细胞（基因被排除在 GEARS 训练集之外，但细胞存在于评测数据集），共 7361 个；其余方法预测的是 held-out guide 条件（3294 个）。评分真值统一为 held-out guide 细胞。这是 GEARS 官方 `myGears.getPredict` 的评测路径，但与被评测细胞集合不同，需要在解释时注意。
6. **CPA 的 `is_count_data=False` + `recon_loss='gauss'`**：因为共享基因空间是 log-norm 而非计数；CPA 官方配置默认 `nb`。
7. **`common_degs` 聚合**：文献定义了 GT DEG 集（`getDEG_cutoff`）但未发布聚合函数，本实现用 F1 并公开。
8. **Squidiff**：见 §5 与 `reports/squidiff_repro_preprocessing.md`。需要随数字一起引用的三项偏差：
   (a) 训练预算用 `lr_anneal_steps=2e4` 的完整退火日程（上游默认 `1e5` 在本数据上于 step≈9e3 崩坏）；
   (b) panel 沿用 notebook 的“手留关注基因”步骤（本数据集上对应 19 个受扰动基因；严格 HVG-only 会恰好丢掉它们）；
   (c) 只有 `squidiff_repro_clip` 可用，且它的 `clip_denoised=True` 把表达截到 `[−1,1]`，所以整体偏低。
9. **指标口径修正**：`scperturbench_crosscheck.py` 的 `kl_divergence` 列映射已从 `raw_distance` 改为 `value`
   （文献对 `sym_kldiv` 做 `log2(perf+1)`）。主表用的是 `value` 列，数值不变；但 `scgen`、`linear_pca`
   等 10 个旧方法 CSV 里的 `kl_divergence` 的 `raw_distance` 诊断列已失效（旧代码把它写成了变换后的值）。

---

## 7. 产物与复现

```
.scperturbench/inputs/gse132080_bench/          genes.json / conditions.json / deg_table.json /
                                                gene_embeddings.npz / manifest.json
.scperturbench/inputs/gse132080_squidiff_repro/ Squidiff 自己的预处理产物（train/test h5ad、
                                                gene_panel.json、prep_summary.json）
.scperturbench/results/gse132080_6metrics/
  gse132080/<method>/seed7/{result.h5ad, adapter_manifest.json, adapter.log,
                            score/{summary.csv, per_condition.csv, full.json,
                                   crosscheck_literature.json}}
  gse132080/aggregate/{summary_long.csv, summary_stats.csv, summary_wide.csv,
                       rankings.csv, wins.csv, report.md}
  smoke/...                                     3 条件 smoke + 文献交叉校验
```

一键复现（需要已加载的 `localhost/scperturbench:squidiff` 镜像）：

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\run-benchmark.ps1 -DryRun                 # 打印全部命令
.\run-benchmark.ps1                         # 全流程：方法 + 参考基线 + 打分 + 汇总
.\run-benchmark.ps1 -Only gears,scgen       # 只跑指定方法
.\run-benchmark.ps1 -Smoke                  # 3 条件冒烟
.\run-squidiff-repro.ps1                    # Squidiff 自己的预处理 + 三种采样策略 + 打分
```

已知耗时（本机 RTX 5060 + 20 vCPU）：scgen 5.6 min、cpa 0.6 min、linear 各 3 s、diffae 3.1 min、gears 47 min（CPU）、squidiff 20000 步 22.7 min；每个方法打分约 10 min（可并行）；Squidiff 一次完整 1000 步采样（15 条件 / 8697 细胞）3–4 min，预处理 4.5 min。
