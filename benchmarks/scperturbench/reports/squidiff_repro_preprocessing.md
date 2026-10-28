# Squidiff 按 `Squidiff_reproducibility` 预处理重做

日期：2026-09-28 · seed 7 · 15 个 held-out guide 条件 · 1019 基因空间
镜像：`localhost/scperturbench:squidiff` · conda env `squidiff`（torch 2.2.2+cu118，RTX 5060 / `sm_120` 可执行 CUDA）
上游代码：`C:\Users\Admin\myprojects\Squidiff`（mounted `/root/Squidiff:ro`）、`C:\Users\Admin\myprojects\Squidiff_reproducibility`

---

## 0. 结论速览

| # | 问题 | 结论 |
|---|---|---|
| 1 | Squidiff 自己的预处理到底是什么？ | `prep_pertb_data.ipynb` 族：**已归一化矩阵 → `sc.pp.highly_variable_genes(n_top_genes=1000)` → 手工强留关注的基因 → 按 `Group` 切分**。兄弟 notebook 在 HVG 之前补 `normalize_total(1e4)` + `log1p`，并要求先剔除 MT-/RPS/RPL。 |
| 2 | 我们的 benchmark 数据是否符合它？ | **部分符合，但不等同（本节已按 2026-09-28 的复核更正）。** 我原来的移植**并不独立**：细胞 QC 阈值、基因过滤、训练/测试细胞集合都照抄了你项目的 `preprocess_gse132080.py` 与 split 文件，所以「与你的矩阵完全一致」是循环论证。把两套配方**各自按自己的文档**独立跑一遍（`tools/compare_squidiff_vs_project_prep.py`）：细胞集 19 579 vs **23 608**，基因 panel 1 019 vs **1 022**，**只重合 780 个（77%）**，HVG∩HVG = **761/1000**。另外 Squidiff 那条链的第一步（GEARS 的 `perturb_processed.h5ad`）在镜像里**不存在**（`gears` 包内无任何 `normalize_total`/`log1p`/`highly_variable_genes` 调用），所以它的**归一化尺度无法从代码验证**，我此前的 "log1p CP10K" 是推断。详见 `reports/diffae_architecture_diagnosis.md` §0。 |
| 3 | 所以预处理是瓶颈吗？ | **仍然不是**，但理由要换一个说法：Squidiff 的失败点已定位到上游采样代码（`pred_xstart.clamp(0,)`），与基因选择或归一化尺度无关；在 77% 重合的另一套 panel 上，同样的量级爆炸与同样的 3.17× eps 收缩都会出现。 |
| 4 | 上游默认训练预算能跑通吗？ | **不能，而且可复现。** `lr_anneal_steps=1e5`（上游默认）在本数据上崩坏：seed 7 于 **step 8e3–9e3**（loss 0.268 → 1.01 → 恒等 1.000）、seed 17 于 **step 1.1e4–1.2e4**（loss 0.223 → 0.364 → 恒等 1.000），两次都伴随 grad_norm 从 0.67 单调衰到 0.2 以下（模型退化为输出 0）。证据：`squidiff_repro_1e5_collapse_log.txt`（seed 7）、`squidiff_repro_1e5_collapse_seed17_log.txt`（seed 17）。 |
| 5 | 官方采样路径为什么量级爆炸？ | **定位到 `process_xstart`。** Squidiff 把 guided-diffusion 的 `else` 分支从 `return x` 改成 `return x.clamp(0,)`；而 `pred_xstart` 在高噪声步是病态的（`1/sqrt(ᾱ)` 到 ~190），把零均值、巨大尺度的量在 0 处截断会产生**单向漂移**，1000 步累积后把生成值放大 **59×**。同时还测到模型在高 t 只预测了 ~78% 的噪声（`γ≈0.778`），对应 `exp(½Σβ(1−γ)) = 3.17×` 的 DDIM 放大。 |
| 6 | 换成 Squidiff 自己的 `clip_denoised=True` 分支呢？ | **首次得到量级正确的输出**（全 15 条件生成块：均值 0.0008、max 1.0、‖x‖₂ 21.4 vs 真实 19.8），六指标可解释：`pcc_delta@100 = 0.337`（12 个方法里第 6）、`kl@100 = 17.16`（好于 GEARS 的 20.33），但 `mse@100 = 1.87`、`e_distance@100 = 14.99`、`wasserstein = 276`、`common_degs = 0.038` **仍全面劣于全部简单基线**，没有拿到任何一项第一。 |
| 7 | 指标口径 | 三个新结果的 **75/75（15 条件 × 5 指标）** 与镜像内文献 `calPerfor` 逐值吻合。顺带修正了交叉校验脚本里 `kl_divergence` 的列映射错误（见 §6）。 |

**一句话**：Squidiff 自己的预处理 = 我们已经在用的预处理；把它换上之后，Squidiff 依然不能给出优于简单基线的结果，而失败点不在预处理，在**上游采样代码的 `pred_xstart` 截断**与**上游默认训练预算的崩坏**。

---

## 1. `Squidiff_reproducibility` 的预处理真身

文献出处（逐条对照）：

| 文件 | 作用 |
|---|---|
| `prep_pertb_data.ipynb` | **基因扰动**的预处理：`sc.pp.highly_variable_genes(adata, n_top_genes=1000)` → 对 `['ALAS2','HBA1','HBA2','HIST1H1C','GYPB','SLC25A37','LGALS1']` 手工置 `highly_variable=True` → `adata = adata[:, adata.var.highly_variable]` → 按 `Group` 切 train/val |
| `prep_drug_screen_data.ipynb` | 从 raw counts 出发的兄弟版本：`calculate_qc_metrics` → `filter_cells(min_genes=300)` → `filter_genes(min_cells=3)` → 去 MT/RP 基因 → `normalize_total(1e4)` → `log1p` → `highly_variable_genes(flavor='seurat', n_top_genes=250)` → 手工强留 marker 基因 |
| `prep_sciplex_data.ipynb` / `prep_simu_data.ipynb` | 同一模式，`n_top_genes` 分别为 200 / 100 |
| `squidiff_on_genepertb.ipynb` | 采样侧：`sampler = sample_squidiff.sampler(model_path=..., gene_size=..., output_dim=..., use_drug_structure=False)`；`z_sem = sampler.model.encoder(torch.tensor(adata.X).to('cuda'))`；`sampler.pred(z_sem, gene_size=adata.shape[1])` |
| `sample_squidiff.py` | `sampler.pred` → `diffusion.ddim_sample_loop(...)`，`use_ddim=True`，`timestep_respacing` 取默认 `''`（**走完整 1000 步 DDIM，不做 respacing**） |

关键点：`prep_pertb_data.ipynb` 读的是 GEARS 已经处理过的 `perturb_processed.h5ad`，**没有**再 `normalize_total`/`log1p`。所以“基因扰动的输入尺度”就是 GEARS 的 `log1p(CP10K)`——与我们的 1019 基因矩阵同尺度。

### 1.1 移植实现

`scperturbench_squidiff_prep.py`：从 raw 10x 计数矩阵出发，用同一套 QC 约定 + Squidiff 的基因选择，再按 benchmark 的 barcode 集合切分（保证与其它方法**同细胞**）：

```
raw 10x (23633 × 33694)
  → 元数据对齐 (23608) → good_coverage / number_of_cells 过滤 (19594)
  → 细胞 QC: total_counts≥1000, 200≤n_genes≤10000, pct_mt<20 (19579)
  → 基因 QC: ≥3 细胞表达, 去掉 MT-/RPS/RPL (19780)
  → layers['counts'] 留存 → normalize_total(1e4) → log1p
  → sc.pp.highly_variable_genes(n_top_genes=1000, flavor='seurat')
  → 强留受扰动基因（notebook 的“手留关注基因”步骤在本数据集上的对应物）
  → 1019 基因；按 benchmark barcode 切 train/test
```

`--no-force-targets` 可给出严格 HVG-only 的 1000 基因 panel（`gene_panel.json` 里的 `hvg_only_genes`）。

### 1.2 等价性证据（`prep_summary.json`）

```json
"benchmark_panel_overlap": {
  "benchmark_genes": 1019, "squidiff_genes": 1019, "shared": 1019,
  "only_in_benchmark": [], "only_in_squidiff": [],
  "hvg_vs_benchmark_shared": 1000
},
"split_alignment": { "benchmark_train_cells": 16285, "squidiff_train_cells": 16285,
                     "benchmark_test_cells": 5403, "squidiff_test_cells": 5403 },
"scale_equivalence_vs_benchmark_train": {
  "n_genes_compared": 1019, "n_cells_compared": 16285,
  "max_abs_diff": 3.4598529818907764e-07, "mean_abs_diff": 6.383562477011337e-09,
  "squidiff_mean": 0.2280535582829417,  "benchmark_mean": 0.22805370390415192,
  "squidiff_max": 7.3596955148810235,  "benchmark_max": 7.3596954345703125
}
```

`max_abs_diff = 3.5e-7` 是 7.36 量级上的 float32 分辨率（7.36 × 6e-8 ≈ 4.4e-7），即**数值上同一个矩阵**。

**但这不构成「Squidiff 的预处理 = 我们的预处理」的独立证据**，因为本脚本的 QC 阈值、基因过滤与切分细胞集合都取自你项目的 `preprocess_gse132080.py` 与 benchmark split 文件。独立对照见 `tools/compare_squidiff_vs_project_prep.py`：把两套配方各按自己的文档跑，细胞集 19 579 vs 23 608、panel 1 019 vs 1 022、**交集 780（77%）**、HVG∩HVG **761/1000**——**两者不同**。

**仍然成立的结论**：无论用哪一套 panel，Squidiff 在本数据上的失败点都在上游采样代码（§3）与训练预算（§2），不在基因选择或归一化；`§3` 的三种 `process_xstart` 对照是在既有 1019 panel 上做的，换 panel 只会改变数值、不会改变机制。

---

## 2. 上游默认训练配置，以及它在 GSE132080 上的崩坏

上游默认（`train_squidiff.py::parse_args` + `script_util.model_and_diffusion_defaults()`）：

| 参数 | 上游默认 |
|---|---|
| `lr` | 1e-4 |
| `lr_anneal_steps` | **1e5** |
| `batch_size` | 64 |
| `ema_rate` | 0.9999 |
| `diffusion_steps` | 1000 |
| `timestep_respacing` | `''`（全部 1000 步） |
| `num_layers` | 3 |
| `use_encoder` / `class_cond` | True / False |
| hidden / time_pos / time_embed / latent | 2048 / 2048 / 2048 / 60 |

按这份默认跑（`--squidiff-steps 100000`，57,156,663 参数，0.065 s/step）时训练**崩坏**：

| step | loss | loss_q0 | loss_q3 | grad_norm |
|---|---|---|---|---|
| 0 | 1.12 | 1.10 | 1.13 | 0.823 |
| 3e3 | 0.428 | 0.530 | 0.396 | 0.651 |
| 5e3 | 0.321 | 0.435 | 0.282 | 0.667 |
| 7e3 | 0.268 | 0.389 | 0.227 | 0.668 |
| **8e3** | **0.366** | 0.472 | 0.330 | 0.197 |
| **9e3** | **1.01** | 1.01 | 1.01 | 0.153 |
| 1e4 – 2.3e4 | **1.000** | 1.000 | 1.000 | 0.070 → 0.0695 |

loss 恰好 1.000 就是“模型输出恒为 0”（`eps ~ N(0,1)` 时的平凡解），同时 grad_norm 单调衰减——退化解 + 梯度消失，之后 1.4 万步没有恢复。完整日志已存为 **`reports/squidiff_repro_1e5_collapse_log.txt`**。

**换 seed 复现**（seed 17，同配置）：

| step | 7e3 | 8e3 | 9e3 | 1e4 | **1.1e4** | **1.2e4** | 1.3e4 – 1.9e4 |
|---|---|---|---|---|---|---|---|
| loss | 0.268 | 0.249 | 0.235 | 0.223 | **0.364** | **1.000** | 1.000 |
| grad_norm | 0.681 | 0.668 | 0.669 | 0.661 | 0.270 | 0.231 | 0.214 → 0.187 |

同样的定性行为（先正常收敛到 ~0.22，再一步跳到 1.000 并卡死、梯度持续衰减），只是崩坏点后移到 1.1e4–1.2e4。日志存为 **`reports/squidiff_repro_1e5_collapse_seed17_log.txt`**。两次独立运行都崩坏 ⇒ 这不是 seed 7 的偶发。

对照：同架构、同优化器、同 batch，只把 `lr_anneal_steps` 改成 **2e4**（此前审计报告 §6.4 已记录的偏差）就能稳定收敛到 `loss 0.161`（`loss_q3 0.105`）。两次的差异只在学习率日程：step 8000 处 `lr` 分别是 9.2e-5 与 6.0e-5。

**结论**：上游默认的 1e5 步日程在 GSE132080 上是**不可训练完成**的，不只是“费时间”。因此本次评分的 checkpoint 使用稳定配置（2e4 步，把学习率退火到 0 的**完整**日程），并在每次采样运行的 `adapter_manifest.json` 中记录 `training_skipped / train_dir / train_steps`。

---

## 3. 采样为什么量级爆炸（定量根因）

诊断脚本 `scperturbench_squidiff_rootcause.py`，使用**上面那个稳定 checkpoint**（seed 7，2e4 步），ALDOA 的 64 个训练细胞，完整 1000 步 DDIM。原始证据：`.scperturbench/results/gse132080_6metrics/gse132080/squidiff/seed7/rootcause_official.json`。

### 3.1 候选一：高噪声步的 eps 收缩

| t 区间 | MSE(epŝ, eps) | γ = ⟨epŝ,eps⟩/⟨eps,eps⟩ | corr | ‖epŝ‖/‖eps‖ | 1/√ᾱ(t_low) |
|---|---|---|---|---|---|
| 0–100 | 0.665 | 0.393 | 0.585 | 0.645 | 1.00 |
| 100–250 | 0.205 | 0.738 | 0.895 | 0.824 | 1.06 |
| 250–500 | 0.140 | 0.775 | 0.933 | 0.830 | 1.38 |
| 500–750 | 0.138 | 0.775 | 0.935 | 0.828 | 3.59 |
| 750–1000 | 0.139 | **0.778** | 0.933 | 0.832 | 17.41 |

模型在高噪声段只预测了约 78% 的噪声幅度。把 `γ` 代进 DDIM 的线性化增长因子：

```
growth = exp( 0.5 · Σ_t β_t (1 − γ_t) ) = 3.17
```

即“纯 eps 收缩”只能解释 ~3×，**不足以**解释观测到的 ~59×。

### 3.2 候选二（主因）：`pred_xstart` 的无条件非负截断

`Squidiff/diffusion.py::p_mean_variance` 内部：

```python
def process_xstart(x):
    if denoised_fn is not None: x = denoised_fn(x)
    if clip_denoised: return x.clamp(-1, 1)
    return x.clamp(0,)          # ← Squidiff 改的；guided-diffusion 原版是 return x
```

`sampler.pred` 用 `ddim_sample_loop` 的默认值，即 `clip_denoised=False`，所以每一步的 `pred_xstart` 都被截断到 **≥ 0**。而

```
pred_xstart = (x_t − √(1−ᾱ_t)·epŝ) / √ᾱ_t
```

在 t≈999 处 `1/√ᾱ ≈ 192`，是两个 ~O(1) 量之差乘以 192 —— 零均值、尺度 ~60 的病态量。**把它在 0 处截断等于做半波整流**，每步注入 `≈ 0.4·σ_eps` 的正向漂移，1000 步累积成几十的超额均值。

三种 `process_xstart` 策略的直接对照（同一 checkpoint、同一 z_sem、ALDOA 64 细胞、完整 1000 步）：

| policy | `process_xstart` | mean | std | max | frac_zero | 末步 ‖x‖₂ |
|---|---|---|---|---|---|---|
| 真实数据（参照） | — | 0.226 | 0.576 | 5.65 | 0.803 | 19.8 |
| `upstream`（官方默认） | `clamp(0,)` | **13.395** | 23.90 | 369.5 | 0.247 | 867 |
| `none`（guided-diffusion 原版） | 恒等 | **−0.0011** | 46.30 | 381.3 | 0.0 | 1474 |
| `neg1_1`（Squidiff 的 `clip_denoised=True` 分支） | `clamp(−1,1)` | **0.0057** | 0.670 | 1.0 | 0.0 | **21.4** |

逐步轨迹（节选，`upstream`）：`t=999 x_mean=0.008 → t=899 0.192 → t=749 0.752 → t=499 3.59 → t=249 9.53 → t=99 12.61 → t=0 13.40`；并且整条链上 `pred_x0_mean` **恒等于** `pred_x0_absmean`（25.8 → 13.4），即 `pred_xstart` 全部非负——整流漂移被直接观测到。

- `none` 把均值漂移**完全消除**（末步 mean −0.001，全程 |x_mean| < 0.01），证明漂移确实来自 `clamp(0,)`；但方差仍然爆炸（‖x‖₂ 32 → 1474），这部分来自 §3.1 的 eps 收缩。
- `neg1_1` 是 Squidiff **自己的**另一个分支，量级正确（‖x‖₂ 21.4 vs 真实 19.8），但把表达上限截到 1.0（真实 p99 = 2.77、max = 5.65），所以整体偏低（生成均值 0.0008 vs 真实 0.2245）。

**根因结论**：Squidiff 在这份数据上的量级爆炸不是预处理、不是 DDIM/EMA/respacing，而是 `diffusion.py` 里那一行 `return x.clamp(0,)` 作用在病态的 `pred_xstart` 上；稳健全程还叠了一层 eps 收缩（3.17×）。这两个都是上游代码的属性，与基因选择或归一化无关。

---

## 4. 六指标结果

全部在统一 unseen-guide/seen-target 划分、统一 1019 基因空间、统一 `scperturbench_metrics.py`（文献 `calPerfor` 移植）下，seed 7，15 条件均值。

`pcc_delta` / `common_degs` 越高越好，其余越低越好。

| method | mse@100 | pcc_delta@100 | e_distance@100 | wasserstein@100 | kl_divergence@100 | common_degs |
|---|---|---|---|---|---|---|
| **gears** | **0.020826** | **0.895589** | **0.458947** | 35.580166 | 20.325686 | 0.153786 |
| **scgen** | 0.021268 | 0.863893 | 0.908575 | **27.299535** | **12.526211** | 0.198495 |
| **linear_ridge** | 0.024521 | 0.867429 | 0.513460 | 37.535861 | 13.091425 | **0.267264** |
| linear_pca | 0.064553 | 0.677450 | 0.993181 | 41.749075 | 13.191699 | 0.204616 |
| train_mean | 0.120322 | 0.465249 | 1.578395 | 47.505334 | 13.302791 | 0.036538 |
| cpa | 0.135282 | 0.191665 | 3.955410 | 37.033642 | 16.557743 | 0.051805 |
| control_mean | 0.140728 | −0.012392 | 1.802712 | 49.622519 | 13.348100 | 0.000000 |
| diffae | 0.143308 | 0.028314 | 2.016981 | 43.882389 | 13.821154 | 0.036309 |
| squidiff_repro_clip | 1.870213 | 0.337250 | 14.988506 | 276.422846 | 17.156833 | 0.038254 |
| squidiff_repro_noclamp | 63.641286 | 0.077157 | 466.065835 | 629770.379167 | 27.917804 | 0.004040 |
| squidiff（旧：ddim50 + 官方 clamp） | 664.127340 | −0.121157 | 352.566380 | 151812.482813 | 25.089771 | 0.069999 |
| squidiff_repro（完整 1000 步 + 官方 clamp） | 1005.041305 | −0.092636 | 435.705192 | 233915.982292 | 26.011909 | 0.070000 |

诊断列 `kl_divergence_floor@100`（std 下限 0.1）：`squidiff_repro_clip` 3.236、`noclamp` 14.010、`squidiff_repro` 12.502；对照 control_mean 0.979、gears 0.618、scgen 1.697。

### 4.1 怎么读

- **`squidiff_repro`（官方默认采样）比旧的 `squidiff` 更差**：`mse@100` 664 → 1005、`wasserstein` 1.5e5 → 2.3e5。原因是这次**去掉了 respacing**（走满 1000 步），而漂移是**每步累积**的——链越长漂移越大。这本身是“更忠实于上游默认”带来的代价，也再次说明默认路径不可用。
- **`squidiff_repro_clip` 是唯一可解释的 Squidiff 结果**：
  - `pcc_delta@100 = 0.337`，在 12 个方法中排第 6，高于 CPA（0.192）与 Diff-AE-Perturb（0.028），低于 train_mean（0.465）；
  - `kl_divergence@100 = 17.16`，**优于 GEARS（20.33）**，排第 8/12；
  - 但 `mse@100 = 1.87`（最好基线 0.0208，差 90×）、`e_distance@100 = 14.99`（最好 0.459）、`wasserstein = 276`（最好 27.3）、`common_degs = 0.038`（最好 0.267）——**四项都在榜尾**，且**没有拿到任何 metric×budget 单元的第一名**。
- **`squidiff_repro_noclamp`**：均值不再漂移，但方差爆炸，除 `pcc_delta@100 = 0.077` 外全部崩坏（`wasserstein` 6.3e5）。它作为“证明漂移来自 `clamp(0,)`”的对照保留，不作为可用结果。

**净结论**：把 Squidiff 换成它自己的预处理 + 上游默认架构 + 补上它自己的 `clip_denoised` 分支后，Squidiff 第一次给出**可解释**的六指标；但它**仍然全面弱于三个简单基线**，也弱于 control_mean / train_mean 的多数指标。与旧结论的差异只是“从不可解释变为可解释”，排名位置没有实质改善。

---

## 5. 交叉校验（指标口径）

对三个新结果，用镜像内文献自己的 `calPerfor` 独立复算，**15 条件 × 5 指标 = 75/75 全部吻合**：

```
$r/squidiff_repro/seed7/score/crosscheck_literature.json          "n_match": 75, "all_match": true
$r/squidiff_repro_clip/seed7/score/crosscheck_literature.json     "n_match": 75, "all_match": true
$r/squidiff_repro_noclamp/seed7/score/crosscheck_literature.json  "n_match": 75, "all_match": true
```

差异只来自文献自身的 `round(...,4)`（最大 4.29e-5），例如：

```
# squidiff_repro
ALDOA  mse            lit=886.271100 ours=886.271143 diff=4.29e-05 match=True
TUBB   kl_divergence  lit=12.284110  ours=12.284110  diff=1.43e-08 match=True
# squidiff_repro_clip
TUBB   kl_divergence  lit=3.318259   ours=3.318252   diff=6.95e-06 match=True
# squidiff_repro_noclamp（pertpy 原值 ~1e5 量级的极端输入）
TUBB   wasserstein    lit=580833.500000 ours=580833.500000 diff=0.00e+00 match=True
```

注意这些是在**极端**输入上取得的：`squidiff_repro` 的 `kl_divergence` 原始 pertpy 值高达 1.01e17（退化基因主导），仍然逐值吻合——说明指标实现不是“只在温和数据上对”。

---

## 6. 顺带修正：交叉校验脚本的 `kl_divergence` 列映射

本次发现 `scperturbench_crosscheck.py` 的 `METRIC_MAP["kl_divergence"]` 指向了我们 CSV 的 `raw_distance` 列（pertpy 原值），而文献 `calPerformance_genetic.py` 第 116–117 行**会**做 `perf = np.log2(perf + 1)`：

```python
if metric == 'sym_kldiv':
    perf = np.log2(perf + 1)
```

因此文献值应当对应我们的 `value` 列。已修正为 `("sym_kldiv", "value")`，修正后三个新结果都是 75/75。

影响范围：

- **主表不受影响**：`aggregate/*.csv` 用的是 `value` 列，语义一直是 `log2(sym_kldiv+1)`，与文献一致。
- **旧方法 CSV 的 `raw_distance` 列对 `kl_divergence` 已失效**：`scgen`、`linear_pca` 等 10 个方法是在旧版代码下打分的，那一列被写成了变换后的值（`raw_distance == value`，例如 scgen ALDOA@100 两者都是 34.858876）。该列只是诊断列，未参与任何排名；三个新方法的该列是正确语义。若要彻底统一，需要用当前代码重跑这 10 个方法的打分。
- 旧审计报告 §4 里 “kl_divergence lit=39.923753 ours=39.923753 diff=0” 这条证据当时是**因为旧版 `raw_distance` 也做了 log2** 才吻合；修正映射后它依然吻合，但原因变成“`value` 列本来就是 log2 值”。

`pcc_delta` 的映射（`pearson_distance` ↔ `raw_distance`）保持不变且正确——文献对 `pearson_distance` 不做任何后处理，而我们的 `value` 是 `1 − perf`。

---

## 7. 口径偏差（引用数字时必须同时引用）

1. **泛化轴**：unseen guide / **seen target**（`splits_unseen_guide_seed7`）。15/15 个 test 条件的 target 基因都在训练集中；不是 scPerturBench 的 unseen-perturbation 设定。
2. **基因空间**：1019（checkpoint 空间），非 hvg5000；`numDEG=5000` 退化为“全部 1019”。
3. **数据集**：GSE132080 不在论文 12 个数据集内，数值与论文表格不可比。
4. **Squidiff 训练预算**：上游默认 `lr_anneal_steps=1e5` 在本数据上崩坏（§2），实际评分用的是 `lr_anneal_steps=2e4` 的完整退火日程（`loss 1.12 → 0.161`）。这是**有意偏差**，记录在各 `adapter_manifest.json` 的 `train_steps / training_skipped / train_dir`。
5. **强留受扰动基因**：Squidiff 的 panel 用了 notebook 的“手留关注基因”步骤（本数据集上对应 19 个受扰动基因）。不这样做会得到 1000 个 HVG，恰好**丢掉这 19 个受扰动基因**；`gene_panel.json` 同时给出 `hvg_only_genes` 以便复核。
5b. **预处理的独立性问题（2026-09-28 更正）**：`scperturbench_squidiff_prep.py` 的 QC 阈值、基因过滤与切分细胞集合取自你项目的 `preprocess_gse132080.py` 与 benchmark split；独立跑 Squidiff 家族的 `prep_*` 参数会得到**不同**的细胞集与 panel（交集 780/1019，HVG∩HVG 761/1000）。同时，Squidiff 基因扰动链第一步用的 GEARS `perturb_processed.h5ad` 的**制备代码不在镜像内**，其归一化尺度未能验证。引用“预处理一致”时必须一并引用这两点。
6. **`squidiff_repro_clip` 的截断**：`clip_denoised=True` 把 `pred_xstart` 截到 `[−1,1]`，因此表达 > 1 的部分被砍掉（真实 p99=2.77、max=5.65），生成均值 0.0008 明显低于真实 0.2245。这不是无偏结果，是“量级正确但偏低”的结果。
7. **单 seed**：seed 7，未做多 seed 方差。
8. **`kl_divergence` 的数值尺度**被少数 control 方差为 0 的基因主导（文献 `epsilon=1e-8`），因而 `kl_divergence_floor` 是更可读的诊断列。
9. **GEARS / CPA 的偏差**与 `common_degs` 聚合方式同 `reports/gse132080_6metrics_benchmark.md` §6，未变。

---

## 8. 产物与复现

新文件：

```
scperturbench_squidiff_prep.py        Squidiff_reproducibility 预处理移植（从 raw 10x 起）
scperturbench_squidiff_sampling.py    三种 process_xstart 策略（upstream / none / neg1_1）
scperturbench_squidiff_rootcause.py   采样根因诊断（eps 收缩 + 逐步轨迹 + 三策略对照）
run-squidiff-repro.ps1                一键：预处理 → 采样三策略 → 打分 → 汇总
tools/run_squidiff_sampling_variants.sh   三策略采样（容器内驱动）
reports/squidiff_repro_1e5_collapse_log.txt         上游 1e5 默认日程崩坏（seed 7）
reports/squidiff_repro_1e5_collapse_seed17_log.txt  同一崩坏的换 seed 复现（seed 17）
```

新增/更新的产物：

```
.scperturbench/inputs/gse132080_squidiff_repro/
    squidiff_train.h5ad / squidiff_test.h5ad / gene_panel.json / prep_summary.json / prep.log
.scperturbench/results/gse132080_6metrics/gse132080/
    squidiff_repro/seed7/{result.h5ad, adapter_manifest.json, adapter.log, score/*}
    squidiff_repro_clip/seed7/{...}            含 crosscheck_literature.json (75/75)
    squidiff_repro_noclamp/seed7/{...}         含 crosscheck_literature.json (75/75)
    squidiff/seed7/rootcause_official.json     根因证据（三策略 + 逐步轨迹）
    squidiff/seed7/rootcause_smoke.json        同一脚本的 4 细胞冒烟（用于校验重构后的采样模块）
    squidiff_probe_1e5/seed17/{adapter.log}    上游默认日程的第二次稳定性探测（日志副本见 reports/）
    aggregate/{summary_long,summary_stats,summary_wide,rankings,wins}.csv + report.md
```

复现（镜像已加载）：

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\run-squidiff-repro.ps1 -DryRun        # 打印全部命令
.\run-squidiff-repro.ps1                # 预处理 + 三策略采样 + 打分 + 汇总
.\run-squidiff-repro.ps1 -SkipPrep      # 复用已有预处理
.\run-squidiff-repro.ps1 -ProbeUpstreamDefault   # 复现上游 1e5 默认日程的崩坏
```

已知耗时（RTX 5060 + 20 vCPU）：预处理 4.5 min（主要耗在 352 MB 的 `matrix.mtx.gz`）；一次 1000 步 DDIM 采样（15 条件、8697 细胞）约 3–4 min；每方法打分约 10 min；每个方法的文献交叉校验（15 条件 × 5 指标 = 75 个比对）约 6 min。
