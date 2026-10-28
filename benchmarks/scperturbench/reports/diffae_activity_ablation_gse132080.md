# GSE132080:预测 sgRNA 效率条件的有无消融(六指标)

日期:2026-10-10 · seed 7 · 15 个 held-out guide 条件 · 1019 基因空间 · 指标实现 `scperturbench_metrics.py`(文献 `calPerfor` 移植)

---

## 1. 结论摘要

在同一数据、同一划分、同一训练配方、同一推理与评测协议下,唯一差别是 `--activity-source predicted`(加上预测的 sgRNA 效率条件)与 `none`:

| | 结论 |
|---|---|
| **点估计方向** | 加上活性条件在 **6 个主指标中 4 个更好**:`mse@100` −40.7%、`e_distance@100` −16.0%、`wasserstein@100` −8.1%、`common_degs` +11.8%;`pcc_delta@100` **变差** −2.6%;去掉伪影后的 `kl_divergence_floor@100` 基本持平(−2.0%,g=−0.09) |
| **统计证据** | 15 条件配对检验,**BH 校正后没有任何指标显著**(Wilcoxon q 0.50–0.89;配对 t 检验最强者 `common_degs` p=0.097)。效应量小到中等(Hedges g 0.25–0.43,KL 去伪影后仅 −0.09) |
| **增益的来源** | 平均值由**少数条件的大幅改善**拉动:`mse@100` 上 GATA1 −64%、HSPA5 −73%、EIF2S1 −63%;其余 11 个条件多数**略微变差**(如 POLR2H +100%、RPS14 +55% 的 MSE) |
| **稳健性** | 上述方向在 DEG-100 与全基因两种预算下一致(mse@all −28.0%、pcc_delta@all −4.8%);限定到"活性通道真正有输入"的 12 个条件后**仍无一显著**(BH q 0.33–0.73) |
| **一句话** | **活性条件对平均表达轮廓与整分布距离有一致的改善方向,但幅度小于条件间波动,在 n=15 下无法与噪声区分;同时它牺牲条件间的相对方向精度(`pcc_delta`)。按当前证据不能宣称"加上 sgRNA 效率显著更好"。** |

---

## 2. 实验设计

### 2.1 两条臂

| 臂 | 目录 | `--activity-source` | 其他 |
|---|---|---|---|
| **with** | `activity_on_200ep` | `predicted`(Table S8 26-mer CNN → 强度 → target 特化 sigmoid → 显式活性嵌入) | 完全相同 |
| **without** | `activity_off_200ep` | `none`(无 guide encoder) | 完全相同 |

固定配方(两臂逐字一致):
`--perturbation-col perturbation_label --activity-label-col guide_identity --target-strategy raw --epochs 200 --batch-size 256 --lr 1e-4 --latent-dim 128 --prediction-type x0 --control-reference per_cell --seed 7 --save-every 200`
活性臂额外:`--activity-predictor-backend table_s8_cnn --guide-activity-cnn-table Table_S8 --activity-table Table_S16 --activity-col relative_activity_day10 --guide-sequence-length 26`
训练设备:**GPU**(见 §6)。模型选择:**不做选择**,两臂均用最后一个 epoch(`checkpoint_epoch_0200.pt`)——本仓库 val 与 train 是同一文件,`checkpoint_best.pt` 是在训练集上选的。

### 2.2 为什么扰动嵌入按 target、而活性特征按 guide

原计划按 guide 建嵌入,但实测发现:held-out-guide 划分下,**26 条 test guide 在训练集中 0/3294 个细胞出现**(train 只有 103 条 guide),即所有评测 guide 的嵌入行**永远拿不到梯度**。原 activity 分支能跑通,是因为它用的是**全量数据的 cell 级内划分**(其 `perturbation_mapping.json` 为 129 条、无零细胞 id),而那份全量预处理数据(`gse132080_preprocessed.h5ad`)与本机 `diff-project` 目录都已不存在。

因此改为:
- **扰动嵌入 = 靶基因**(27 行:26 target + control),15 个评测条件的靶基因**全部**在训练集中 → 嵌入全部可训;
- **活性特征 = guide**(Table S8 序列特征按 `guide_identity` 取),每个 target 在训练侧有 **2–5 条不同 guide**,所以"同 target 内的 sgRNA 效率差异"仍有对比信号。

这样既保住了 held-out-guide 评测轴(测试细胞来自训练中从未出现的 guide),又不需要接受"测试 guide 已在训练集见过"的泄霹。

### 2.3 评测协议(与主表 `gse132080_6metrics_benchmark.md` 一致)

- target 级 15 条件,`perturbation_label`;每条件真值细胞数与主表逐条吻合(ALDOA 132 / CAD 312 / DBR1 610 / DUT 334 / EIF2S1 73 / GATA1 89 / GINS1 456 / GNB2L1 28 / HSPA5 120 / HSPA9 81 / HSPE1 275 / POLR1D 173 / POLR2H 157 / RPS14 127 / TUBB 327);control 池 2109。
- DDIM **50** 步,seed 7,生成对照细胞预算 ≤2000/条件。
- 指标:DEG 限制(top-100 by |t| 与 all)、分布指标子采样 ≤2000 细胞(`np.random.seed(42)`)、`pcc_delta` 在 control 中心化后计算、`sym_kldiv` 取 `log2(x+1)`、`common_degs` 用 `getDEG_cutoff` 的 F1。

---

## 3. 六指标主结果(15 条件均值)

| 指标 | without | with | 变化 | 效应量 g | Wilcoxon p | BH q | 配对 t p |
|---|---|---|---|---|---|---|---|
| `mse@100` ↓ | 0.03970 | **0.02353** | **−40.7%** | −0.35 | 0.890 | 0.890 | 0.172 |
| `e_distance@100` ↓ | 0.8997 | **0.7561** | **−16.0%** | −0.25 | 0.762 | 0.890 | 0.327 |
| `wasserstein@100` ↓ | 28.477 | **26.180** | **−8.1%** | −0.42 | 0.454 | 0.757 | 0.107 |
| `pcc_delta@100` ↑ | **0.8219** | 0.8007 | −2.6% | −0.27 | 0.252 | 0.606 | 0.292 |
| `common_degs` ↑ | 0.12654 | **0.14142** | **+11.8%** | +0.43 | 0.101 | 0.503 | 0.097 |
| `kl_divergence_floor@100` ↓ | 1.34533 | **1.31818** | **−2.0%** | −0.09 | 0.762 | 0.890 | 0.712 |

> **图表与上表的差异(有意为之)**:柱状图的 Symmetric KL 面板用的是 **`kl_divergence_floor@100`**(std 下限 0.1),不是报告值 `kl_divergence@100`(12.569 → 12.349)。原因见 §3.1:报告值被**单个零方差基因**支配(高簇里平均 77.6%、三个条件是 100%),其"改善"大部分来自该伪影。去掉伪影后真实差异是 **1.345 → 1.318(−2.0%)**,配对检验 p=0.762、效应量 g=−0.09 —— 基本无差异。报告值那一行仅作为"已知受伪影影响"的存档。

全基因预算(方向一致):

| 指标 | without | with | 变化 | BH q |
|---|---|---|---|---|
| `mse@all` ↓ | 0.005957 | **0.004291** | **−28.0%** | 0.890 |
| `e_distance@all` ↓ | 1.23644 | **1.16997** | −5.4% | 0.890 |
| `pcc_delta@all` ↑ | **0.72904** | 0.69419 | −4.8% | 0.606 |
| `kl_divergence_floor@all` ↓ | **0.62996** | 0.64646 | +2.6% | — |

柱状图(300 dpi PNG + 矢量 PDF,**可直接引用的副本在 `reports/assets/activity_ablation/`**):
`fig1_activity_ablation_6metrics.{png,pdf}` — 标题为 `GSE132080`;6 面板,每面板两柱(±bootstrap 95% CI,按条件重采样 B=10000)+ 每条件个体点 + 配对连线;类别标签统一为 **`with activity` / `without activity`**(左柱蓝 = with,右柱橙 = without);面板只留指标名,无预算/方向/统计小字。
逐条件相对变化:`fig3_per_condition_deltas.{png,pdf}`。
活性通道真正生效子集:`fig2_activity_ablation_6metrics_livechannel.{png,pdf}`。
逐条件原始值:`reports/assets/activity_ablation/ablation_per_condition.csv`;配对统计:`ablation_tests.csv`。

### 3.1 为什么 `kl_divergence` 图上是"一簇高一簇低"(重要,直接影响该指标的可引用性)

柱状图 `Symmetric KL` 面板里 15 个条件分成两簇(约 0.5–3 与约 31–36),**这不是模型或生物学差异,而是该指标的公式伪影**。原因在 `sym_kldiv_with_floor`:

```python
kl = log(y_std/x_std) + (x_std² + (x_mean−y_mean)²) / (2·y_std²) − 0.5
```

当某个基因在**真值**(stimulated)中的标准差恰为 0(该基因在这些细胞里恒为 0),`y_std` 落到 `epsilon = 1e-8` 下限,分子里的 `(x̄−ȳ)²` 会让这一项达到 **1e8–1e9 量级**;再经 `log2(x+1)` 压成约 31–36。实测证据:

| 簇 | 条件 | 报告值(无下限) | 单基因占该条件总值的比例 | 加 std 下限 0.1 后 |
|---|---|---|---|---|
| **高** | ALDOA、DBR1、HSPE1、POLR2H、RPS14 | 31.0–35.7 | **平均 77.6%**,其中 RPS14/POLR2H/HSPE1 是 **100%** | **1.02–1.58** |
| **低** | 其余 10 个条件 | 0.50–3.08 | 平均 10.8% | 0.50–1.88 |

也就是说高簇的"高"**完全由单个零方差基因贡献**(如 RPS14 的 ISG20、POLR2H 的 ICOSLG、HSPE1 的 CREB5);加 0.1 的 std 下限后,高簇均值从 33.49 掉到 **1.34**,低簇从 1.78 到 **1.31**——两簇的差距从 19 倍缩到 **1.03 倍**。

**对结论的影响**:该指标上两臂的"改善"也同样被这一个基因支配。报告值上 `kl@100` 差 −0.220(即 −1.7%),而**加下限后只差 −0.027**(≈−2%,同一量级但绝对值极小)。所以:

- `kl_divergence@100` 的排名信息量很低,**不应作为活性条件有效/无效的主要证据**;本仓库既有主表也早就记录过这一点;
- 更具可比性的是诊断列 `kl_divergence_floor`(std 下限 0.1):**with 1.3182 vs without 1.3453**,方向与报告值一致但幅度小得多;
- 分簇本身主要由"哪个条件恰好有一个零方差基因落进 DEG-100 子集"决定,与预测质量无关。

---

## 4. 关键限定:平均值由少数条件拉动

`mse@100` 逐条件(Δ = with − without,负为好):

| 条件 | Δ | 条件 | Δ |
|---|---|---|---|
| GATA1 | **−0.1432 (−64%)** | HSPE1 | +0.0019 |
| HSPA5 | **−0.0818 (−73%)** | CAD | +0.0030 |
| EIF2S1 | **−0.0490 (−63%)** | TUBB | +0.0038 |
| HSPA9 | −0.0205 (−38%) | POLR1D | +0.0047 |
| GNB2L1 | −0.0010 | DUT | +0.0048 |
| DBR1 | −0.0006 | ALDOA | +0.0064 |
| | | GINS1 | +0.0079 |
| | | RPS14 | +0.0094 |
| | | POLR2H | +0.0116 |

即 **4 个条件改善、11 个条件变差**;均值下降 40.7% 完全由 GATA1/HSPA5/EIF2S1/HSPA9 贡献(它们同时也是 `e_distance` 改善最大的条件)。这也解释了为什么 Wilcoxon(只看符号秩)**看不见**这个差异而配对 t 检验看得见——效应集中在尾部而非普遍平移。

**没有找到可解释该集中性的机制**:把相对效应与"训练侧同 target 活性极差 / 活性标准差 / guide 数 / 平均活性"做 Spearman 相关,**16 个组合全部不显著**(|rho| ≤ 0.38,p ≥ 0.16)。也就是说,观察到的集中性**不能被"该 target 的活性轴信息量更大"解释**,可能是有限样本波动,也可能是尚未识别的数据侧因素。

限定到"活性通道真正有输入"的 12 个条件(GNB2L1/HSPA9/POLR1D 因 Table S8 无 `_00` 行而空转)后,方向不变但**同样无一显著**:`mse@100` g=−0.36(q=0.73)、`pcc_delta@100` g=−0.30(q=0.64)、`wasserstein@100` g=−0.43(q=0.73)、`common_degs` g=+0.37(q=0.64)。即**排除空转条件并不改变结论**,不支持"3 个空转条件稀释了效应"这一解释。

---

## 5. 边界与已知限制(引用数字时必须同时引用)

1. **只有单 seed(7)**,没有多 seed 方差;§3 的不确定性来自 15 个条件间的配对结构,不是训练随机性。
2. **训练细胞来自 1019 基因面板**(`gse132080_squidiff_repro` 的 train 划分,16285 细胞),不是原始全基因预处理数据——后者已随 `diff-project` 目录丢失。两臂口径相同,故 A/B 内部有效;但**不能**与本仓库主表的 19 方法数字直接并列。
3. **泛化轴是 held-out guide / seen target**:测试细胞来自训练中未出现的 guide(0/3294 重叠),但靶基因都见过。因此这是"对未见 guide 的响应幅度外推",不是 unseen-target 泛化。
4. **3/15 个条件活性通道空转**:Table S8 **不存在任何 `_00` 行**(fully-matched 参考 guide 即"零错配"本身),而 GNB2L1、HSPA9、POLR1D 的 held-out guide 恰好都是参考 guide,共 282 细胞拿不到 CNN 特征。这是上游数据限制。敏感性分析见 `fig2` 与 `ablation_tests_livechannel.csv`(12 条件)。另 RAN/RPS15 两族共 8 条 guide 在 Table S8 中也无记录,训练时其特征补零。
5. **`activity_scales_delta=False`**(显式嵌入通道),与上游 200ep 显式分支一致;折叠式条件(`activity_scales_delta=True`)未在本轮评测。
6. **训练/测试损失不可横向比较**:活性臂总损失含 `0.1 × guide_strength_loss`(val_strength ≈0.087),without 臂该项恒为 0,所以 with 臂 `val_loss` 0.0529 与 without 0.0451 不可直接比大小,必须看下游六指标。
7. **`common_degs` 聚合函数文献未发布**,沿用本仓库 F1 实现。
8. **metric 与 budget 的关系**:`wasserstein` 只有 DEG-100(文献如此);`mse`/`pcc_delta`/`e_distance`/`kl` 有 100 与 all 两档;`common_degs` 无 budget。

---

## 6. 本轮修掉的两个上游缺陷与一个环境问题

### 6.1 上游 activity 通路的 target-id 尺寸错误

`PerturbDiffusionModel` 用 `config.n_perturbations` 同时给两样东西定尺寸:**扰动嵌入表**(guide/target 索引)和 **target 特化 doser**(`GeneralizedSigmoid`)。当扰动词表 ≠ target 词表时,`target_ids` 会越界:

```
IndexError: index 104 is out of bounds for dimension 0 with size 104     (guide 级)
```

**只有活性臂会踩**——without 臂从不访问 `target_ids`。若不修,这次 A/B 会变成"坏臂 vs 好臂"。修法:保持两个 id 空间各自独立,只把 doser 尺寸按实际 target 词表放大。自测 `tools/selftest_target_id_fix.py` 先复现 `IndexError: index 120 ... size 104`,再验证修好后 doser 表为 `(1, 129)` 且高 target id 行能拿到梯度。

### 6.2 checkpoint 加载会漏掉训练好的 doser

若默认建模的 doser 尺寸与 checkpoint 不一致,`load_state_dict(strict=False)` 会因 **size mismatch** 报错(strict 只容忍缺失/多余 key)。`scperturbench_diffae_loader.py` 从 checkpoint 自身推断尺寸(`table width − 1`)并只重建 doser,同时**校验扰动嵌入行数与 checkpoint 一致**后才放行,否则拒绝评测。`tools/selftest_diffae_loader.py` 覆盖三种情形(上游式加载失败、审计式加载成功、不可恢复时拒绝)。本轮 manifest 审计:`doser_table_size_in_checkpoint=28 = in_model`、`missing_doser_keys=[]`、`embedding_rows 27 = 27`。

### 6.3 GPU 环境(我此前判断错误,已修正)

我最初只测了 `pertpyV7` 一个环境就宣称"容器里所有 torch 都不支持 sm_120",这是错的。逐环境实测内核:

| env | torch | arch list | 内核实测 |
|---|---|---|---|
| **scarches** | 2.2.2 + **cu118** | …, sm_86, **sm_90**, compute_37 | ✅ MATMUL OK |
| **squidiff** | 2.2.2 + **cu118** | 同上 | ✅ MATMUL OK |
| pertpyV7 | 2.2.2 + cu121 | …, sm_90 | ❌ no kernel image |
| linearModel | 2.3.0 + cu121 | …, sm_90 | ❌ no kernel image |
| gears / cpa | 2.0.1 + cu117 | …, sm_86 | ❌ no kernel image |

历史上能用 GPU 的是 conda 的 `pytorch-2.2.2-py3.9_cuda11.8_cudnn8.7.0_0`(Python 3.9)。卡住本轮的不是 GPU,而是"**Python ≥3.10 + 可执行 GPU 的 torch**"交集为空。为此新建独立环境 `perturbench/.scperturbench/envs/diffae-cu128`(Python 3.12 + `torch 2.11.0+cu128`,arch 含 `sm_120`),实测 MATMUL/CONV2D/AUTOGRAD 全部通过;`pertpyV7` 及其他环境**未改动**,评分工具链零风险。

---

## 7. 产物与复现

```
.scperturbench/inputs/activity_ablation_preflight.json          12 项 preflight 结果
.scperturbench/trained/activity_on_200ep/checkpoint_epoch_0200.pt   sha256 8ecc2ca9…
.scperturbench/trained/activity_off_200ep/checkpoint_epoch_0200.pt
.scperturbench/results/gse132080_activity_ablation/gse132080/
  activity_on_200ep/seed7/{result.h5ad, adapter_manifest.json, adapter.log, score/}
  activity_off_200ep/seed7/{result.h5ad, adapter_manifest.json, adapter.log, score/}
  aggregate/{report.md, rankings.csv, summary_wide.csv, summary_stats.csv, summary_long.csv, wins.csv}
  ablation_analysis/
    ablation_pairwise.csv / ablation_tests.csv / ablation_per_condition.csv / ablation_summary_wide.csv
    ablation_pairwise_livechannel.csv / ablation_tests_livechannel.csv
    figures/fig1_*, fig2_*, fig3_*  (各 .png 300dpi + .pdf 矢量)
```

复现命令(容器内,`scperturbench` machine 已启动):

```powershell
# 0) 环境(一次性)
.\run-bench.ps1 -Name setup-gpu-env -Gpu -Command 'bash /root/perturbench/.scperturbench/work/setup_gpu_env.sh'

# 1) preflight(12 项断言)
.\run-bench.ps1 -Name activity-preflight -Command 'bash /root/perturbench/.scperturbench/work/activity_ablation_preflight.sh'

# 2) 训练两臂(200 epoch,GPU;实测约 7.5 min/臂)
.\run-bench.ps1 -Name ablation-train -Gpu -Command 'bash /root/perturbench/.scperturbench/work/activity_ablation_train.sh'

# 3) 推理 + 六指标打分 + 聚合(推理 15s/臂;打分约 10 min)
.\run-bench.ps1 -Name ablation-infer -Gpu -Command 'bash /root/perturbench/.scperturbench/work/activity_ablation_infer_score.sh'

# 4) 配对统计与出图(宿主机)
python scperturbench_ablation_figure.py --preflight .scperturbench\inputs\activity_ablation_preflight.json
```

自测(均可在容器内独立运行):
`tools/selftest_target_id_fix.py`、`tools/selftest_diffae_loader.py`、`tools/selftest_ablation_figure.py`(合成数据,不依赖真实结果)、`tools/verify_paired_stats.py`(独立复算配对检验)。

---

## 8. 下一步建议

1. **多 seed** 是提升证据强度最直接的一步:当前 g=0.43(`common_degs`)与 g=−0.42(`wasserstein`)在单 seed 下无法定论,seed ∈ {7,17,27} 可把"训练随机性"与"数据异质性"分开。训练成本约 7.5 min/臂,代价很低。
2. **条件数扩到 guide 级**可把 n 从 15 提到 ~127,配对检验功效大幅提升(本仓库已有该口径的 6 指标实现)。
3. 若要解释"为什么只有 GATA1/HSPA5/EIF2S1/HSPA9 受益",建议做**活性扰动实验**:固定 checkpoint,把评测 guide 的活性值替换为同 target 另一 guide 的值,看预测是否随之移动。若几乎不动,说明活性通道在推理时影响很小——这能直接检验"活性条件是否真的被用上"。
