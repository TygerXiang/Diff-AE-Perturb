# Diff-AE-Perturb vs 四个外部方法（GEARS / scGen / CPA / Squidiff）

本报告是 `reports/benchmark/gse132080_6metrics_benchmark.md`（19 行总表）的**抽取视图**：只保留四个已发表方法与我们的两行，去掉 4 行文献参考基线与 6 行我们自己的消融臂。数值与排名均取自同一份聚合表，未做任何重算。

## 0. 协议

- 种子 7，单一 seed（无多 seed 方差）
- 15 个 held-out guide 条件（unseen guide / **seen target**）
- 1019 基因面板，指标在 GT DEG 子集上算（top-100 与 all=1019）
- 6 个指标全部按文献 `calPerfor` 逐条移植，并与镜像内文献代码交叉校验

## 0.5 结论（数字由下表算出）

- **修复后（`fix_x0_percell`）在 10 个 metric×budget 单元里，只有 1 个（`kl_divergence@100`）超过四个外部方法中最好的那个**，其余 9 个单元由 GEARS、scGen 守住。
- 天花板几乎只在 GEARS 与 scGen 之间移动（GEARS 4 个单元、scGen 6 个）；**CPA、Squidiff 在 10 个单元里一次都没拿到第一**。
- 逐方法看：`fix_x0_percell` 平均每个单元胜过 **2.5/4** 个外部方法；修复前的归档 checkpoint（`diffae`）只有 **1.3/4**。
- 与外部最好的距离（本方法 / 外部最好，1.00 = 追平）：误差类 5 个单元 **1.04×–2.21×**；相关性/差异基因类 5 个单元 **0.65×–1.04×**。
- 结论：修复把我们从「在 4 个指标上与 `control_mean` 同级或更差」拉到「稳定压过 CPA 与 Squidiff、逼近 GEARS/scGen」，但**还不足以说超过第一梯队**——除 `kl_divergence@100` 外没有拿到任何单元的第一。

下限参考：`control_mean` / `train_mean` 在多数单元排 8–13 名之间；修复后的 `fix_x0_percell` 在 10 个单元里全部排进前 6。

## 1. 误差与分布指标（越低越好）

| 方法 | `mse@100` | `mse@all` | `e_distance@100` | `e_distance@all` | `wasserstein@100` |
|---|---|---|---|---|---|
| GEARS | **0.020826** (1) | 0.004495 (3) | **0.458947** (1) | **0.563596** (2) | 35.580166 (5) |
| scGen | 0.021268 (2) | **0.002984** (1) | 0.908575 (5) | 1.221925 (5) | **27.299535** (1) |
| CPA | 0.135282 (9) | 0.015812 (9) | 3.955410 (13) | 4.935972 (13) | 37.033642 (6) |
| Squidiff | 1.870213 (14) | 0.342666 (14) | 14.988506 (14) | 17.907049 (14) | 276.422846 (14) |
| diffae（修复前，归档 checkpoint） | 0.143308 (13) | 0.016483 (12) | 2.016981 (12) | 1.800163 (11) | 43.882389 (11) |
| **fix_x0_percell**（修复后重训） | 0.039266 (4) | 0.005937 (5) | 0.898848 (4) | 1.247116 (6) | 28.389225 (3) |
| _control_mean（下限）_ | 0.140728 (12) | 0.016079 (10) | 1.802712 (9) | 1.440864 (8) | 49.622519 (13) |
| _train_mean（下限）_ | 0.120322 (8) | 0.013996 (8) | 1.578395 (8) | 1.289641 (7) | 47.505334 (12) |

每格 `数值 (排名/19)`；**加粗** = 四个外部方法与我们的两行里最好的那个。

## 2. 相关性与差异基因（越高越好）

| 方法 | `pcc_delta@100` | `pcc_delta@all` | `kl_divergence@100` | `kl_divergence@all` | `common_degs` |
|---|---|---|---|---|---|
| GEARS | **0.895589** (1) | 0.744061 (4) | 20.325686 (14) | 41.706032 (13) | 0.153786 (4) |
| scGen | 0.863893 (4) | **0.795696** (2) | 12.526211 (3) | **35.495660** (1) | **0.198495** (3) |
| CPA | 0.191665 (11) | 0.153094 (10) | 16.557743 (12) | 38.437993 (8) | 0.051805 (11) |
| Squidiff | 0.337250 (9) | 0.131164 (11) | 17.156833 (13) | 49.528323 (14) | 0.038254 (12) |
| diffae（修复前，归档 checkpoint） | 0.028314 (16) | 0.023067 (16) | 13.821154 (10) | 37.431762 (5) | 0.036309 (14) |
| **fix_x0_percell**（修复后重训） | 0.823224 (5) | 0.729793 (5) | **12.503866** (2) | 36.758542 (3) | 0.129396 (5) |
| _control_mean（下限）_ | -0.012392 (17) | -0.011666 (19) | 13.348100 (8) | 39.784120 (10) | 0.000000 (19) |
| _train_mean（下限）_ | 0.465249 (8) | 0.340420 (8) | 13.302791 (7) | 39.791117 (11) | 0.036538 (13) |

每格 `数值 (排名/19)`；**加粗** = 四个外部方法与我们的两行里最好的那个。

## 3. 逐单元判定：我们 vs 四个外部方法

`fix_x0_percell`（修复后重训）与四个外部方法逐一比。比值 <1（误差类）/ >1（相关类）即为胜。

| 指标单元 | 外部最好（方法） | `fix_x0_percell` | 相对外部最好 | 胜过几个外部方法 | 排名/19 |
|---|---|---|---|---|---|
| `mse@100` | GEARS 0.020826 | 0.039266 | 1.89× ❌ | **2/4** | 4 |
| `mse@all` | scGen 0.002984 | 0.005937 | 1.99× ❌ | **2/4** | 5 |
| `e_distance@100` | GEARS 0.458947 | 0.898848 | 1.96× ❌ | **3/4** | 4 |
| `e_distance@all` | GEARS 0.563596 | 1.247116 | 2.21× ❌ | **2/4** | 6 |
| `wasserstein@100` | scGen 27.299535 | 28.389225 | 1.04× ❌ | **3/4** | 3 |
| `pcc_delta@100` | GEARS 0.895589 | 0.823224 | 0.92× ❌ | **2/4** | 5 |
| `pcc_delta@all` | scGen 0.795696 | 0.729793 | 0.92× ❌ | **2/4** | 5 |
| `kl_divergence@100` | scGen 12.526211 | 12.503866 | 1.00× ✅ | **4/4** | 2 |
| `kl_divergence@all` | scGen 35.495660 | 36.758542 | 1.04× ❌ | **3/4** | 3 |
| `common_degs` | scGen 0.198495 | 0.129396 | 0.65× ❌ | **2/4** | 5 |

## 4. `kl_divergence_floor` 诊断列

`kl_divergence` 绝对值被少数 control 方差为 0 的基因主导，此行更可读（std 下限 0.1）：

| 方法 | `kl_divergence_floor@100` | `kl_divergence_floor@all` |
|---|---|---|
| GEARS | 0.617503 | 0.304198 |
| scGen | 1.697149 | 0.758983 |
| CPA | 3.764903 | 1.924392 |
| Squidiff | 3.235991 | 3.762701 |
| diffae（修复前，归档 checkpoint） | 1.787214 | 0.674840 |
| **fix_x0_percell**（修复后重训） | 1.354350 | 0.636528 |

## 5. Squidiff 的另外三个变体（不可用）

Squidiff 上游采样路径的 `process_xstart` 在高噪声步把零均值巨尺度量在 0 处整流，
1000 步累积放大 59×。下表说明为什么主表里只用 `squidiff_repro_clip`：

| 变体 | `mse@100` | `e_distance@100` | `wasserstein@100` | `pcc_delta@100` |
|---|---|---|---|---|
| Squidiff（旧：ddim50 + 官方 clamp） | 664.127340 | 352.566380 | 151,812 | -0.121157 |
| Squidiff（完整 1000 步 + 官方 clamp） | 1005.041305 | 435.705192 | 233,916 | -0.092636 |
| Squidiff（关掉上游截断，对照） | 63.641286 | 466.065835 | 629,770 | 0.077157 |

## 6. 口径偏差（引用这些数字时必须一起引用）

1. **泛化轴**：unseen guide / seen target，不是 scPerturBench 的 unseen-perturbation 设定。
2. **基因空间**：1019（checkpoint 空间），非 HVG5000；文献 `numDEG=5000` 退化为「全部 1019」。
3. **数据集**：GSE132080 不在文献 12 个数据集内，数值与论文表格不可比。
4. **单 seed**：无重复，`summary_stats.csv` 的 `sd` 为空。
5. **GEARS 的评测细胞集合不同**：其 test loader 覆盖训练划分中携带这些 target 基因的细胞（7361 个），其余方法预测 held-out guide 条件（3294 个）；评分真值统一为 held-out guide 细胞。这是 GEARS 官方 `getPredict` 路径，但解释时需要注意。
6. **CPA** 用 `is_count_data=False` + `recon_loss='gauss'`（共享基因空间是 log-norm 而非计数），非官方默认的 `nb`。
7. **Squidiff** 用 `lr_anneal_steps=2e4` 的完整退火日程（上游默认 `1e5` 在本数据上 step≈9e3 崩坏）；只有 `clip_denoised=True` 分支可用，它把表达截到 [−1,1] 所以整体偏低。
8. **设备不同**：GEARS/CPA/linear 在 CPU（RTX 5060 是 `sm_120`，其 torch 无对应 kernel），scGen/Squidiff/Diff-AE-Perturb 在 GPU。
9. **我们的两行不是同一个模型的两次推理**：`diffae` 是归档 checkpoint（条件接口有 bug），`fix_x0_*` 是按修复后的代码重新训练 200 epoch 的结果。
10. **`kl_divergence` 绝对值**被少数 control 方差为 0 的基因主导（可达 2^40 量级），排序信息有限；`kl_divergence_floor`（std 下限 0.1）是更可读的诊断列。

## 7. 出处与复现

```
reports/benchmark/aggregate/summary_wide.csv   数值（19 方法 × 12 列）
reports/benchmark/aggregate/rankings.csv       排名（19 方法 × 10 单元）
reports/benchmark/aggregate/report.md          自动生成的完整 19 行报告
reports/benchmark/aggregate/{wins,summary_stats,summary_long}.csv
reports/benchmark/external_methods_comparison.csv   本报告的机器可读版
```

```powershell
python scripts/build_external_comparison.py
```

逐方法逐条件的原始分（`score/summary.csv`、`per_condition.csv`、`full.json`）与各方法的 `adapter_manifest.json` 在 `C:\Users\Admin\myprojects\perturbench\.scperturbench\results\gse132080_6metrics\`（该目录未纳入版本控制）。
