# scPerturBench 六指标口径审计（GSE132080）

日期：2026-09-28 · 工作区：`C:\Users\Admin\myprojects\perturbench`
镜像：`localhost/scperturbench:squidiff`（Ubuntu 24.04, conda 25.3.1）

---

## 1. 结论速览

| 问题 | 结论 |
|---|---|
| 原来的 `Diff-AE-Perturb/scripts/evaluate_benchmark.py` 六指标是否符合文献口径？ | **不符合。** 6 个指标里只有 `mse` 完全一致，`pcc_delta` 语义等价但符号相反，其余 4 个（`e_distance`、`wasserstein`、`kl_divergence`、`common_degs`）实现不同；且缺少文献的三项协议约束。 |
| 本次实现是否与文献一致？ | **一致。** 用镜像内论文自己的 `calPerfor` 对同一 `result.h5ad` 独立打分，3 个条件 × 5 个 pertpy 指标 = 15/15 全部吻合，差异仅来自文献自身的 `round(..., 4)`（最大绝对偏差 4.83e-5）。证据见 `crosscheck_literature.json`。 |
| 镜像能否跑齐 6 个方法？ | **能，但设备不同。** scGen / Squidiff / Diff-AE-Perturb 可走 GPU；CPA / GEARS / linear 只能用 CPU（`sm_120` 没有对应 kernel）。 |
| 数据是否与论文表可比？ | **不可比。** GSE132080 不在 scPerturBench 的 12 个数据集内，基因空间是 checkpoint 的 1019 而非 hvg5000。我们借用的是**指标定义与计算协议**。 |

---

## 2. 文献真身（镜像内实际代码）

| 作用 | 路径 |
|---|---|
| 六指标计算主函数 | `/home/project/Pertb_benchmark/manuscript2/calPerformance_genetic.py` → `calPerfor` |
| 预处理 / DEG 表生成 | `manuscript2/genetic/myUtil1.py` → `preData`, `calDEG` |
| 距离实现 | `pertpy 0.9.4` 的 `pertpy/tools/_distances/_distances.py` |
| 工具（子采样等） | `calPerformance_genetic.py` 内 `f_subSample`, `checkNan`, `calculateDelta`, `SuppressOutput` |
| 被评测方法 | `manuscript2/genetic/{myGears,linearModel,genetic_baseReg,...}.py` |

`calPerfor` 的真实协议（逐条）：

1. **基因限制**：每个条件只保留 `DEG_hvg5000.pkl` 中按 `|scores|` 排序的 top-N 基因（N=100；wasserstein 只用 100）。
2. **NaN 修补**：`checkNan` 用 control 均值填充。
3. **分组**：`Expcategory ∈ {imputed, stimulated, control}`，`onesided_distances(groupby='Expcategory', selected_group='imputed', groups=['stimulated'])`。
4. **子采样**：分布类指标（`edistance`/`wasserstein`/`sym_kldiv`/`mean_var_distribution`）先把三组各自 `np.random.seed(42)` 子采样到 ≤2000 细胞。
5. **`pearson_distance`**：先 `calculateDelta`（减 control 均值），再算 `1 − pearsonr(mean(Δimputed), mean(Δstimulated))`。
6. **`sym_kldiv`**：逐基因高斯对称 KL 求均值，最后 `log2(x + 1)`。
7. **`mse`**：`‖mean(imputed) − mean(stimulated)‖² / n_genes`（pseudobulk）。
8. **输出**：`round(value, 4)`。

`edistance` 用的是 **sqeuclidean**：

```
2 · mean(sqeuclidean(X,Y)) − mean(sqeuclidean(X,X)) − mean(sqeuclidean(Y,Y))
```

`wasserstein` 用 OTT-JAX 的 Sinkhorn 正则化 OT cost（细胞级，非逐维一维距离）。

---

## 3. 与原 `Diff-AE-Perturb/scripts/evaluate_benchmark.py` 的逐项对照

| 指标 | 原实现 | 文献口径 | 判定 |
|---|---|---|---|
| `mse` | `mean((true_mean − pred_mean)²)` | `MeanSquaredDistance`：同式 | ✅ 一致 |
| `pcc_delta` | `corrcoef(Δtrue, Δpred)` | `1 − pearsonr(Δtrue, Δpred)` | ⚠️ 语义等价、符号相反（原实现“越高越好”，文献是距离“越低越好”） |
| `e_distance` | `2·mean(cdist euclidean) − within`（**非平方**） | `Edistance` 用 **sqeuclidean** | ❌ 不一致（原实现是文献量的非线性变换） |
| `wasserstein` | PCA-50 上逐维 1D `scipy.stats.wasserstein_distance` 取均值 | Sinkhorn 正则化 OT cost（细胞级） | ❌ 不一致 |
| `kl_divergence` | PCA-50 直方图 KL 取均值 | 逐基因高斯对称 KL 均值 + `log2(x+1)` | ❌ 不一致 |
| `common_degs` | 现场 scanpy t-test，top-50，`|logFC| ≥ 0.25` | 预先算好的 GT DEG 表 + `getDEG_cutoff`（`p_adj ≤ 0.01 & (FC ≥ 2 \| FC ≤ 0.5)`） | ❌ 不一致 |

另外缺失的三项协议约束：

- 没有 DEG 基因子集限制（在全 1019 基因上算，与文献的 top-100 差异很大）；
- 没有 `f_subSample`（seed 42，≤2000 细胞/组）；
- 没有 `Expcategory` 分组契约（用“全部细胞 vs 全部细胞”的隐式约定）。

**影响**：原实现的数值不能与 scPerturBench 表里的任何数字并列；其 `e_distance`、`wasserstein`、`kl_divergence` 是不同量纲的量（例如 `wasserstein` 原实现是 ~0.3 量级的“一维距离均值”，文献实现是 ~25–38 量级的 OT cost），因此**旧结论中的跨方法比较不可复用**。

---

## 4. 本次实现（工作区内）

| 文件 | 作用 |
|---|---|
| `scperturbench_metrics.py` | 文献六指标的逐条移植（`calPerfor` 的 7 步协议、pertpy 距离、DEG 表限制、seed-42 子采样），并额外记录 `raw_distance`（未经 `log2`/`1−x` 变换的 pertpy 原值） |
| `scperturbench_inputs.py` | 生成 `genes.json`（1019 基因固定顺序）、`conditions.json`、`deg_table.json`（`calDEG` 同款 t-test 配方）、`gene_embeddings.npz` |
| `scperturbench_adapters.py` | 六个方法的适配器，统一产出 `result.h5ad`（`obs['perturbation']` + `obs['Expcategory']`，可直接喂给文献代码） |
| `scperturbench_crosscheck.py` | 用镜像内的 `calPerfor` 对我们的 `result.h5ad` 独立复算并比对 |
| `scperturbench_aggregate.py` | 汇总为长表/宽表/排名/markdown |
| `run-bench.ps1`, `run-benchmark.ps1` | Podman 执行入口与全流程驱动 |

### 指标口径的显式声明（诚实标注）

1. **`common_degs`**：镜像里定义了 `getDEG_cutoff`（GT DEG 集），但**聚合函数没有随镜像发布**（全镜像 grep 无任何调用点）。本实现用同一 `getDEG_cutoff` 定义在预测侧重算 DEG 集，指标取 **F1**，并同时记录 `precision`/`recall`/`Jaccard`/集合大小。这是唯一一个“配方来自文献、聚合方式由我们定义并公开”的指标。
2. **`kl_divergence` 的数值尺度**：文献 `SymmetricKLDivergence` 用 `std + 1e-8`，在 1019 基因空间的 top-100 DEG 里存在 control 方差为 0 的诱导型基因，导致 KL 被少数基因主导（可达 2^40）。主表保留**文献原值**，同时输出 `kl_divergence_floor`（把 std 下限设为 0.1）作为诊断列，不替换主指标。
3. **`log2(sym_kldiv+1)`** 与 `pcc = 1 − pearson_distance` 的换算在本实现中显式完成，`raw_distance` 列保留 pertpy 原值以便与文献直接比对。

### 交叉校验证据

```
ALDOA  mse              lit=0.028200 ours=0.028219 diff=1.86e-05 match=True
ALDOA  pcc_delta        lit=0.675700 ours=0.675730 diff=3.03e-05 match=True
ALDOA  e_distance       lit=0.572200 ours=0.572171 diff=2.88e-05 match=True
ALDOA  wasserstein      lit=25.727700 ours=25.727661 diff=3.89e-05 match=True
ALDOA  kl_divergence    lit=39.923753 ours=39.923753 diff=0.00e+00 match=True
... (3 条件 × 5 指标，15/15 全部 match)
```

`lit` 的末位抖动来自文献自己的 `round(..., 4)`；`kl_divergence` 在未取整的原始精度上完全相等。

### 4.2 交叉校验脚本的列映射修正（2026-09-28）

做 Squidiff 重做时发现 `scperturbench_crosscheck.py` 的 `METRIC_MAP["kl_divergence"]` 指向了我们
CSV 的 `raw_distance` 列（pertpy 原值），而文献 `calPerformance_genetic.py` 第 116–117 行**会**做：

```python
if metric == 'sym_kldiv':
    perf = np.log2(perf + 1)
```

所以文献值应对应我们的 `value` 列。已改为 `("sym_kldiv", "value")`，修正后三个新的 Squidiff
结果各 **75/75**（15 条件 × 5 指标）逐值吻合，包括 `kl_divergence` 原始 pertpy 值高达 1.01e17 的
病态输入。

影响：

- **主表不受影响**——`aggregate/*.csv` 用的是 `value` 列，语义一直是 `log2(sym_kldiv+1)`，与文献一致。
- 上面 §4 「交叉校验证据」里那条 `kl_divergence lit=39.923753 ours=39.923753 diff=0` 当时之所以吻合，
  是因为**旧版代码把 `raw_distance` 也写成了变换后的值**（`raw_distance == value`）。修正映射后它
  仍然吻合，但理由变成“`value` 列本来就是 log2 值”。
- `scgen`、`linear_pca` 等 10 个在旧版代码下打分的方法，其 `kl_divergence` 的 `raw_distance` 诊断列
  已失效（写成了变换后的值）。该列不参与任何排名；若要让诊断列统一，需用当前代码重跑这 10 个方法的打分。
- `pcc_delta` 的映射（`pearson_distance` ↔ `raw_distance`）保持正确：文献对 `pearson_distance`
  不做后处理，而我们的 `value` 是 `1 − perf`。

---

## 5. 镜像能力核查（实测）

| conda env | python | torch | GPU kernel | 结论 |
|---|---|---|---|---|
| `scarches` | 3.9.19 | 2.2.2+cu118 | ✅ 可执行 | scGen、Diff-AE-Perturb 用 GPU |
| `squidiff` | 3.9.19 | 2.2.2+cu118 | ✅ 可执行 | Squidiff 用 GPU |
| `cpa` | 3.9.16 | 2.0.1+cu117 | ❌ `no kernel image is available` | CPA 用 CPU |
| `gears` | 3.9.16 | 2.0.1+cu117 | ❌ | GEARS 用 CPU |
| `linearModel` | 3.10.14 | 2.3.0+cu121 | ❌ | linear 用 CPU（本来也不训练） |
| `pertpyV7` | 3.12.3 | 2.2.2+cu121 | ❌ | 指标计算用 CPU（`JAX_PLATFORMS=cpu`） |

- RTX 5060 是 `sm_120`；上述失败 env 的 PyTorch 不含 `sm_120` kernel。`sm_120` 警告全部保留在各自日志中。
- pertpy 0.9.4 的五个距离指标在 `JAX_PLATFORMS=cpu` 下全部跑通（JAX CUDA 插件 PJRT 版本不匹配，必须强制 CPU）。
- **文献 linear baseline 是 R 脚本**（`manuscript2/genetic/linearModel.r`，依赖 zellkonverter/glmGamPoi/irlba），镜像内**没有 R**，因此按同一数学式移植为 Python（`solve_y_axb` 的 PCA-10 + ridge 0.1 双线性解），另加 `genetic_baseReg.py` 的 scGPT 嵌入 Ridge 变体。
- GEARS 首次运行会从 dataverse 下载 `gene2go_all.pkl` / `essential_all_data_pert_genes.pkl` / `go_essential_all.tar.gz`（实测 ~25 KB/s，约 40 分钟）；镜像自带这些资源，已改为从本地播种（`--gears-resource-dir`）。
- 镜像自带数据集只有 `DataSet2/Papalexi`、`DataSet2/sciplex3_A549`、`DataSet1/kangCrossCell`；论文其余数据集不在镜像内。

---

## 6. 与文献口径的已知偏差（必须随结果一起引用）

1. **泛化轴**：统一为“unseen guide / seen target”（沿用现有 `splits_unseen_guide_seed7`）。15/15 个 test 条件的 target 基因都在训练集中，6 个方法面对同一任务、彼此可比；但这不是 scPerturBench 的 unseen-perturbation 设定。
2. **基因空间**：1019（checkpoint 空间），不是 hvg5000。文献的 `numDEG=5000` 退化为“全部 1019 基因”，`numDEG=100` 仍有效。
3. **数据集**：GSE132080 不在论文数据集内，数值与论文表格不可比。
4. **Squidiff 预算**：官方默认 `lr_anneal_steps=1e5`；本机实测该日程在本数据上**可复现地崩坏**
   （seed 7：step 8e3–9e3；seed 17：step 1.1e4–1.2e4；loss → 恒等 1.000，grad_norm → 0.2 以下），
   因此实际评分用 `lr_anneal_steps=2e4` 的完整退火日程（记录在 `adapter_manifest.json`）。
   详见 `reports/squidiff_repro_preprocessing.md`。
5. **单 seed**：评测 seed = 7（与 checkpoint 一致）；未做多 seed 方差。
6. **`common_degs`** 的聚合函数（见 §4.1）。
7. **交叉校验的列映射**（见 §4.2）：`kl_divergence` 现已对应 `value` 列；10 个旧方法的
   `raw_distance` 诊断列失效但主表数值不受影响。

---

## 7. 磁盘处置记录

- `localhost/scperturbench:bad`（96.7 GB 名义）已删除；因层与其他镜像共享，`podman system df` 未变化（unique 110.6 GB）。
- `localhost/scperturbench:latest` 保留：它是 `:squidiff` 的父镜像，删除不会释放空间，却会失去重建 squidiff 环境的基线。
- `ext4.vhdx` 302.23 GB。已用 `wsl --manage podman-scperturbench --set-sparse true --allow-unsafe` 打开稀疏；`fstrim` 报告回收 901 GiB，但宿主机 C: 未变（回收需要管理员权限的 `Optimize-VHD -Mode Full` 或 `diskpart compact vdisk`，本会话无提权；`Optimize-VHD` 已确认存在但返回 `You do not have the required permission`）。
- C: 可用空间 56.5 GB；40.36 GB 的 `scperturbench_v1.tar.gz` 是唯一可立即回收的大件（镜像已全部载入，删除不影响运行，但会失去离线重载能力）。
- 如需彻底回收 vhdx，请在**管理员** PowerShell 中执行：
  ```powershell
  $env:HOME='C:\Users\Admin\myprojects\perturbench'
  podman machine stop scperturbench
  wsl --shutdown
  Optimize-VHD -Path "$env:HOME\.local\share\containers\podman\machine\wsl\wsldist\scperturbench\ext4.vhdx" -Mode Full
  podman machine start scperturbench
  ```

