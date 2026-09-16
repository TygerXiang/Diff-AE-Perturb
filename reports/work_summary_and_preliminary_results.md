# 已有工作总结与扩散模型初步结果

本文件总结 `Diff-AE-Perturb` 仓库当前已完成的工作，并汇总扩散模型在 GSE132080 上的初步分析结果。
详细逐条结果见 `reports/diffusion_preliminary_results.md`。

---

## 一、项目定位

PerturbDiff 是一个条件 DDIM（扩散）模型，用于从 control 单细胞表达预测扰动后的单细胞表达：

```text
z_condition = E(x_control) + delta_g
x_pred      = DDIM(z_condition)
```

- `SemanticEncoder`：`x_0 (G 维表达) -> z_sem (128 维 latent)`
- `PerturbationModule`：每个扰动条件一个可学习 latent 偏移 `delta_g`，control 行固定为 0
- `ConditionalDenoiserMLP`：`D(x_t, t, z_cond) -> x_0`
- `GaussianDiffusion`：线性 beta schedule + 确定性 DDIM 采样
- 训练损失：主重建 MSE（可选 GEARS autofocus/direction、semantic alignment、direction alignment）
- 对照约定：所有 control 标签统一映射到 id 0

代码位置：`src/perturbdiff/{model,modules,diffusion,losses,config,data}.py`；
训练入口 `scripts/train_with_split.py`、`scripts/train.py`。

## 二、数据

| 数据集 | 内容 | 本地规模 | 角色 |
| --- | --- | --- | --- |
| GSE132080 | 人类 K562 CRISPRi Perturb-seq，128 条靶向 sgRNA / 25 个 target gene / 10 条 non-targeting | 19,579 细胞 × 1,019 基因（1000 HVG + 强制保留 target gene） | 主训练集 |
| GSE283614 | 人 ESC/DE 中 OCT4、NANOG 增强子扰动 Perturb-seq，48 条 guide capture | 14,514 细胞 × 6,851 基因（合并 assigned） | 跨状态 / 剂量验证 |

- GSE132080 预处理：`scripts/preprocess_gse132080.py`（QC、total-count 归一化、log1p、HVG，`control` 标签归并，`perturbation_label` 保留 target 级标签）
- GSE283614 预处理：`scripts/preprocess_gse283614.py`（基因表达 / guide capture 拆分、guide assignment、ESC/DE 状态标注）
- 划分：`scripts/create_h5ad_splits.py`、`scripts/create_gse283614_dose_cv.py`

## 三、已完成的训练运行

| 运行 | 数据 / 标签 | epoch | best epoch | best val loss | test loss |
| --- | --- | ---: | ---: | ---: | ---: |
| `runs/gse132080_target_200ep` | GSE132080 / target 级 | 200 | 198 | 0.023150 | 0.023287 |
| `runs/gse132080_sgrna_200ep` | GSE132080 / 128 条件 | 200 | 199 | 0.023034 | 0.023288 |
| `runs/benchmark_activity_branch_none_200ep` | GSE132080 / 无 activity | 200 | 199 | 0.023032 | 0.023286 |
| `runs/benchmark_activity_branch_predicted_day10_200ep` | 预测 activity | 200 | 193 | 0.053607 | 0.054527（eval MSE 0.046443） |
| `runs/gse283614_no_activity_controls_fixed_200ep` | GSE283614 / 无 activity | 200 | 192 | 0.056869 | 0.055763 |
| `runs/gse283614_explicit_activity_log2fc_controls_fixed_200ep` | GSE283614 / 显式 log2FC activity | 200 | 196 | 0.061063 | 0.060063 |
| `runs/biological_validation/gse283614_dose_cv/esc/fold_*/seed_7/{target_only,dose}` | 5 折 OOF 剂量模型 | 200 | — | 0.0342 ~ 0.0377 | 0.0341 ~ 0.0389 |

注：`gse132080_target_200ep` 与 `gse132080_sgrna_200ep` 的 loss 几乎相同（0.02329 vs 0.02329），
说明两者都在 target 粒度上学习；两条 sgRNA 级的条件向量并未带来独立增益。

## 四、已有分析结论

1. **target 级 vs sgRNA 级生成对比**（`runs/gse132080_target_vs_sgrna_200ep/comparison.json`，127 条 guide）：
   - target 级 mean Pearson R² = 0.95092，sgRNA 级 = 0.95082；MSE 0.015123 vs 0.015204
   - 在 ≥10 细胞的 84 条 guide 上，R² = 0.96129 vs 0.96108
   - 结论：保留 raw sgRNA 标签本身不能稳定提升生成精度，target 级略有优势
2. **activity 条件的作用**：
   - GSE132080 预测 activity：改善 mean-profile MSE（0.010209 → 0.009001）与 pcc_delta（0.109 → 0.298），但 energy distance 变差（1.490 → 1.905）
   - GSE283614 显式 log2FC activity：MSE 0.005275 → 0.004504，pcc_delta 0.0577 → 0.2960；energy distance 与 KL 变差
   - 结论：activity 条件主要改善平均响应方向与轮廓，对完整分布不一定更准
3. **embedding 结构分析**（`runs/benchmarks/`）：
   - `embedding_activity_correlation_200ep_no_diralign`：guide embedding 范数与 paper activity 的相关性
   - `embedding_sequence_relationship_200ep_no_diralign`：错配位置、GC 含量、rG:dT 等序列特征与 embedding 范数的关系
   - `embedding_direction_clustering_200ep_no_diralign`：同 target guide 方向聚类的逐 target 汇总
4. **GSE283614 生物学剂量验证**（`runs/biological_validation/gse283614_observed`）：
   - 48/48 guide、33/33 靶向 guide 覆盖完整；Table S4 六种关系类型判断一致
   - 但独立重拟合未通过预注册的完整 gate（NANOG DE 二次项顶点略低于观测剂量区间；OCT4 DE 评分表重拟合为二次而记录为 Hill）
   - `observed_stage_gate_pass = false`，因此正式 200 epoch OOF 训练仍被 gate 阻断
   - essentiality buffering 在四个 state/TF 组合上通过（表达尺度归一化动态范围 + 单侧 Mann-Whitney p < 0.05）
   - log2FC 映射 25/33 靶向 guide，与 Table S4 knockdown fraction 的 Spearman rho = 0.6696（p = 0.000251）
5. **虚拟强度**：以 guide embedding 范数 / 同 family fully matched guide 范数定义虚拟剂量 `s`，
   export 见 `runs/gse132080_sgrna_200ep/guide_scale_export/`（128 条 sgRNA 的 `s`、target、序列）

## 五、本次新增的扩散模型初步结果（sgRNA 级）

本次新增导出脚本与报告：

- `scripts/export_diffusion_prelim.py`：从 checkpoint + prepared h5ad 导出 sgRNA 级/条件级结果（target 序列、sgRNA 序列、错配距离、原始表达值与原始 latent）
- `scripts/build_diffusion_prelim_report.py`：生成 Markdown 报告
- `scripts/probe_condition_effect.py`：条件效应强度检查
- `scripts/fetch_target_gene_sequences.py`：可选的 NCBI RefSeq CDS 拉取工具（当前结果未使用）

导出目录：

- sgRNA 级：`runs/gse132080_target_200ep/preliminary/`（127 条 sgRNA × 25 target）
- 条件级：`runs/gse132080_target_200ep/preliminary_condition/`
- 报告：`reports/diffusion_preliminary_results.md`

### 5.1 结果概览

- checkpoint：`runs/gse132080_target_200ep/checkpoint_best.pt`（best epoch 198）
- 数据：GSE132080 test split，1,958 细胞 × 1,019 基因，211 个 control 细胞
- 导出：127 条 sgRNA，25 个 target gene，latent 128 维，DDIM 50 步

所有数值均为原始值，不再包含 L2 等汇总指标：

| 对象 | 量级示例 |
| --- | --- |
| 对照池平均 Z-sem（128 维） | 分量为 -0.1 ~ +0.4 |
| 对照平均表达编码 Z-sem | 分量为 -0.1 ~ +0.2 |
| sgRNA 观测均值 Z-sem | 分量为 -0.5 ~ +0.7 |
| `delta_g`（128 维） | 分量为 -0.01 ~ +0.03；典型范数 0.03（ALDOA）~ 0.18（HSPA5） |

### 5.2 结果表示例（前 4 条 sgRNA，含 target 序列与错配距离）

`target 20 nt` 是同一 family 的 fully matched guide 序列，即该 sgRNA 要识别的目标片段；
`dist to PAM` 从 PAM 端计数（紧邻 PAM = 0），与文库编号满足 `距离 = |编号| - 1`。

| id | target | sgRNA identity | target 20 nt | sgRNA sequence | alignment | dist to PAM | 文库编号 | cells |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 14 | ALDOA | ALDOA_+_30077139.23-P1P2_00 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGACCCCTTCTG` | `\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|` | - | - | 8 |
| 14 | ALDOA | ALDOA_+_30077139.23-P1P2_06 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGATCCCTTCTG` | `\|\|\|\|\|\|\|\|\|\|\|X\|\|\|\|\|\|\|\|` | 8 | -9 | 12 |
| 14 | ALDOA | ALDOA_+_30077139.23-P1P2_07 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGCCCCCTTCTG` | `\|\|\|\|\|\|\|\|\|\|X\|\|\|\|\|\|\|\|\|` | 9 | -10 | 10 |
| 14 | ALDOA | ALDOA_+_30077139.23-P1P2_13 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGACCCCTTTTG` | `\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|X\|\|` | 2 | -3 | 12 |

127 条 sgRNA 全部带 target 20 nt；103 条为单错配变体，其计算出的 PAM 距离与论文文库编号 103/103 完全一致。
完整表见 `reports/diffusion_preliminary_results.md` 第 2 节与
`runs/gse132080_target_200ep/preliminary/sgrna_summary.csv`。

### 5.3 原始表达值（control / 观测 / 模型预测）

- 对照平均表达：`expression_control_mean.csv`（1019 基因，211 个 control 细胞均值）
- 观测表达矩阵：`sgrna_expression_matrix.csv`（1019 基因 × 127 sgRNA 均值）
- 模型生成表达矩阵：`sgrna_predicted_expression_matrix.csv`（同上形状，DDIM 50 步）
- 每 sgRNA top 差异基因长表：`sgrna_top_gene_expression_long.csv`
- 报告第 4 节按 target gene 列出代表 sgRNA 的 top 6 差异基因原始表达值

### 5.4 原始 latent 值

- 对照 Z-sem：`z_sem_control.csv`（control 池均值 latent、control 平均表达 latent，各 128 维原始值）
- 对照细胞逐个 Z-sem：`z_sem_control_cells.csv`（211 × 128）
- 条件 Z-sem：`z_sem_conditions.csv`（每条 sgRNA 的观测 Z-sem 与反事实 `z_cf`，各 128 维原始值）
- sgRNA `delta_g`：`sgrna_delta_vectors.csv`（127 × 128 原始值，无任何 L2 汇总列）

### 5.5 需要提醒的结论边界

- `delta_g` 的分量级在 1e-3 ~ 1e-2（128 维中最大值约 0.03），
- 而对照细胞 Z-sem 的分量级在 0.1 ~ 1 之间，两者相差约 1.5 ~ 2 个数量级；
- 因此 DDIM 生成结果主要由 control 输入决定，条件之间的可见差异较小。
  这属于当前模型（200 epoch、无显式 activity 通道）的真实行为，也是后续引入 guide activity / 强度条件的直接依据。

## 六、尚未完成 / 阻塞项

1. GSE283614 剂量实验的正式 OOF 训练被 `observed_stage_gate_pass = false` 阻断，
   需要先解决 Table S4 记录模型类别与独立重拟合不一致的问题
2. 预测 activity 分支只在 20 epoch / 200 epoch 的轻量 benchmark 上验证，
   尚未与 6 指标完整 benchmark 全量对齐
3. 完整 `evaluate_benchmark.py` 的 guide 级全量 benchmark 曾因超时未产出结果
4. 本次导出的条件效应偏弱，需要在后续加入显式 activity / 剂量条件后重跑同一套导出做对比
