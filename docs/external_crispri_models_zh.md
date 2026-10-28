# 外部 CRISPRi sgRNA 效率模型调研

本文档记录可用于替换当前 Table S8 专用 CNN 的公开候选、证据边界和集成方案。结论面向本仓库的人类 SpCas9/dCas9-KRAB 类 CRISPRi 场景，默认目标是支持新 guide、未见过的 target gene 和跨文库测试。

调研日期：2026-10-03。

## 1. 当前分支的基线和接入约束

当前实现位于 [`src/perturbdiff/guide_features.py`](../src/perturbdiff/guide_features.py) 的 `GuideStrengthEncoder`。`table_s8_cnn` 路径把两条 26 nt 字符串 (`genome input`、`sgRNA input`) 编码为 `(4, 26, 2)`，再用二维 CNN 输出一个非负标量。它没有显式接收 PAM、TSS 距离、靶向链、染色质或细胞类型，因此更接近特定 Table S8 格式的 activity 拟合器。

仓库现有综合表 [`results/comprehensive_sgrna_table.csv`](../results/comprehensive_sgrna_table.csv) 和说明 [`results/README.md`](../results/README.md) 给出了外部模型接入所需的实际边界：总计 179 行（GSE132080 130 行、GSE283614 49 行），其中 176 行有 sgRNA 序列、169 行有靶基因组序列和 PAM，128 行有已整理的实验 activity。未解析的 8 行属于 GSE283614 的非靶向/未分配控制记录，不能作为有基因组靶点的 guide 送入外部 predictor。GSE132080 的坐标按 hg19 处理，GSE283614 的补序列按 hg38 处理；不能在适配器中混用参考版本。GSE132080 还包括完美匹配和有意错配的 guide，需单独评估这两类输入。

外部预测分数进入 PerturbDiff 前必须经过统一的尺度校准。外部模型的 `depletion`、`knockdown fraction`、相对 gamma、概率或回归值不能直接乘到 latent delta。当前已抽象的 [`src/perturbdiff/guide_activity.py`](../src/perturbdiff/guide_activity.py) 和 [`src/perturbdiff/guide_predictor.py`](../src/perturbdiff/guide_predictor.py) 可作为 `GuideInput -> ActivityPrediction` 的边界；预测结果应保留来源、上下文模式、缺失字段和不确定性。

## 2. 候选总览

| 候选 | 任务和物种 | 输入/输出 | 权重和许可证据 | 集成判断 |
|---|---|---|---|---|
| Horlbeck / CRISPRiaDesign | 人类和小鼠 CRISPRi/CRISPRa，核心数据包含 K562；CRISPRi 任务匹配度高 | sgRNA/PAM、序列质量、TSS 方向性距离、可及性/核小体/RNA 特征；输出相对 activity score | 论文 CC BY 4.0；代码仓库未发现明确软件许可证；仓库没有可靠的现代预训练权重，README 建议重新学习模型 | **首选人类基线**，需重实现或按本项目数据重训/校准 |
| EpiCas-DL | 混合 CRISPRi、CRISPRa、CRISPRoff 的人类数据；显式多模态 | 40 bp 基因组窗口、核小体/TSS 通道、ATAC、DNA methylation、RNA expression；CNN 回归或分类 | 论文声明 MIT；GitHub 当前未见明确 license 文件；需核对 checkpoint、scaler 和特征顺序 | **可选 context-enhanced backend**，不应把缺失组学简单置零当作 sequence-only |
| Elia-Lejzerowicz `CNN_CRISPRi_sgRNA_Efficacy` | 公开 notebook 实验实现；物种、泛化和论文出处不充分 | hg38 基因组上下游序列与 sgRNA 的 one-hot，PyTorch 1D CNN；输出回归值 | GitHub 仓库只有 README、notebook 和数据文件；未发现软件许可证或可直接加载 checkpoint | 可作可复现实验参考，**不作为默认模型** |
| Jost/Replogle hCRISPRi-v2 | 人类哺乳动物 CRISPRi library 和跨细胞模型实验 | 经验 guide 排名、dual-sgRNA 和 effector 对照；不是独立 `predict(sequence)` 包 | eLife 论文 CC BY；数据/补充表需按原论文条款使用 | **benchmark/calibration 数据源**，不是 predictor |
| Radzisheuskaya 等设计规则 | 人类 CRISPRi，少量基因和 guide 的实验设计研究 | TSS 位置、FANTOM5/CAGE、开放染色质和序列偏好；规则/回归信号 | NAR 论文 CC BY-NC | 可做透明规则基线和特征，不是预训练通用模型 |
| BarquistLab bacterial model | E. coli CRISPRi depletion screens | mixed-effect random forest，guide effect 与 gene effect 分离；含表达、operon 等基因特征 | 论文 CC BY 4.0；GitHub MIT；仓库包含 pickle 权重 | 仅作为未来 bacterial mode 或方法参考，**不能用于当前人类默认 backend** |
| `qhuang`（身份待确认） | 当前没有足够信息确认具体论文或仓库 | 无法可靠判断输入、输出、物种、训练任务或推理接口 | 未核验 | 收到明确论文 DOI 或仓库 URL 前，不纳入候选比较 |

## 3. 首选候选：Horlbeck / CRISPRiaDesign

### 公开证据

- Horlbeck et al., *eLife* 2016，`10.7554/eLife.19760`：<https://elifesciences.org/articles/19760>；开放全文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC5094855/>。
- 代码：<https://github.com/mhorlbeck/CRISPRiaDesign>。
- 论文数据约 18,380 个 sgRNA activity score，来自 30 个 CRISPRi 和 9 个 CRISPRa screen，覆盖人类和小鼠。方法是 Elastic Net 线性回归，特征约 808 个。

特征同时使用序列/PAM、GC、同聚物和二核苷酸，及与 FANTOM5/CAGE TSS 的方向性距离、DNase/FAIRE/MNase 或核小体可及性、RNA folding/hybridization 等信息。因此它表达了 CRISPRi 的主要机制信号：TSS 相对位置、靶向方向和局部染色质，而不是只拟合一段固定窗口。

### 可行性

**高，建议作为第一阶段人类 baseline。** 它的监督任务和当前目标最接近，Elastic Net 易于审计，推理成本低，能把 `sequence + PAM + TSS distance + optional context` 统一成相对 activity。缺点是公开仓库主要是 Python 2.7 脚本、notebook 和特征流程，没有应直接加载的现代权重；仓库本身也没有清晰的软件 license 文件。正式纳入发行版前应：

1. 固定 hg19/hg38、TSS 版本和 BigWig 预处理版本；
2. 用现代 Python 重实现特征生成与 Elastic Net fit，或只复现实验流程，不复制未授权代码；
3. 在当前训练数据中只训练轻量校准层，冻结外部系数；
4. 把 sequence-only 作为单独训练/验证模型，而不是将可及性列静默填 0；
5. 对完美匹配和 mismatch guide 分层报告结果。

## 4. context-enhanced 候选：EpiCas-DL

### 公开证据

- Yang et al., *Computational and Structural Biotechnology Journal* 2023，`10.1016/j.csbj.2022.11.034`；开放全文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC9763632/>。
- 代码：<https://github.com/yangqianq/EpiCas-DL>。
- 论文报告约 769,918 个 sgRNA、18,975 个基因，含 CRISPRi activity、hCRISPRi-v2、CRISPRi genome、CRISPRi K562，以及 CRISPRoff/CRISPRa 数据。论文声明软件采用 MIT license；GitHub API/仓库当前未见对应 license 文件，不能仅凭仓库页面完成法律确认。

输入是 `11 x 40`：40 bp 基因组序列（9 bp upstream + 23 bp target + 8 bp downstream）、4 个 nucleotide one-hot 通道、4 个核小体/TSS 相关通道、ATAC、DNA methylation 和 RNA expression。网络使用并行 1–5 nt 卷积核、池化和全连接层，既支持 activity regression，也支持高低活性分类。

### 可行性和限制

**中高，适合作为可选增强 backend。** 它比当前 CNN 更通用，数据量也大，但依赖细胞类型的 ATAC、甲基化、表达和核小体/TSS 特征。当前 GSE132080/GSE283614 没有为每条 guide 稳定提供同一细胞状态的完整组学输入；直接把缺失值置零会改变输入分布，不能声称这是合法的 sequence-only 回退。推荐：

- 有匹配细胞状态的组学时使用 `context_enhanced`；
- 缺失组学时切换到单独训练的 reduced/sequence-only 模型；
- 在输出中记录 `context_mode`、缺失特征和外部模型版本；
- 确认 EpiCas-DL 是否存在可加载 checkpoint、特征 scaler、40 bp 坐标约定和完整许可后再作为默认依赖。

由于其训练集合混合了 CRISPRi、CRISPRa 和 CRISPRoff，必须确认调用的是 CRISPRi activity head，且把不同论文的 score 映射到同一相对强度尺度。不能直接把模型概率或混合任务回归值送入 PerturbDiff。

## 5. 其他公开候选

### 5.0 `qhuang` 候选：目前无法唯一定位

当前分支没有引用 `qhuang` 的论文、仓库 URL 或模型文件。对公开论文索引和 GitHub 的 `qhuang + CRISPRi`、`qhuang + sgRNA`、`qhuang + CRISPR` 搜索没有得到能够唯一对应“人类 CRISPRi sgRNA efficacy predictor”的结果。仅凭作者名无法判断它是 CRISPRi、Cas9 cleavage、CRISPRoff 还是数据处理项目；因此不把任何同名项目或二手分数当作该候选的证据。需要补充 DOI、完整仓库 URL 或模型名称后，才能核对训练任务、权重、特征顺序和许可证。

### 5.1 `CNN_CRISPRi_sgRNA_Efficacy`：有代码但证据不足

仓库：<https://github.com/Elia-Lejzerowicz/CNN_CRISPRi_sgRNA_Efficacy>。

公开内容是一个 Colab notebook、`data1-2.xlsx`、`setA.tsv` 和 `setB.tsv`，没有论文引用、发布版本、模型权重或软件 license。notebook 使用一维 PyTorch CNN，将 hg38 基因组中 sgRNA 上下游序列拼接后做 one-hot，再预测按基因归一化的屏幕 target；训练代码只展示随机 train/test split。它缺少标准化推理 API、跨 target/library 评估和可复用权重，因此只能作为实现细节参考，不能作为本项目的外部预训练模型。

### 5.2 Jost/Replogle：benchmark 和 calibration 数据

- Replogle et al., *eLife* 2022，`10.7554/eLife.81856`：<https://elifesciences.org/articles/81856>；开放全文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC9829409/>。

hCRISPRi-v2.1、compact libraries、dual-sgRNA 和 Zim3-dCas9 的实验结果适合测试模型是否能恢复高活性 guide 排名、评估 effector 变化，或作为外部 calibration 标签。论文没有提供一个独立、跨 target 的标准 `sequence -> efficacy` 推理包；不要把 library ranking 直接封装成 predictor。

### 5.3 Radzisheuskaya：设计规则 baseline

- Radzisheuskaya et al., *Nucleic Acids Research* 2016，`10.1093/nar/gkw583`：<https://pmc.ncbi.nlm.nih.gov/articles/PMC5062975/>。

155 条 guide、41 个基因的研究显示 FANTOM5/CAGE TSS 和 guide-TSS 位置是强信号，开放染色质和序列偏好可进一步改善结果。它适合实现成透明的 TSS-distance/PAM/GC/poly-T baseline，或作为校准层特征；样本量和模型形式不足以作为通用预训练 predictor。

### 5.4 BarquistLab：只适合细菌模式

- Yu et al., *Genome Biology* 2024，`10.1186/s13059-023-03153-y`：<https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03153-y>。
- 代码：<https://github.com/BarquistLab/CRISPRi_guide_efficiency_bacteria>。

这是 E. coli depletion screen 的 mixed-effect random forest，加入 gene expression、operon、下游 essential gene 等细菌特征后性能显著提高；sequence-only 的跨验证相关性约 0.21，含 gene features 时约 0.66。MIT 仓库和 pickle 权重有较好的可复现性，但物种、操纵子和 depletion 标签与当前人类 dCas9-KRAB 场景不匹配，不能直接接入默认路径。

## 6. 应排除的直接替代

### Cas9 cleavage/knockout 模型

DeepCas9、DeepCRISPR、DeepSpCas9、Azimuth/Rule Set 2、CRISPOR 和 GuideScan 主要预测 Cas9 切割、indel、knockout 或 off-target。它们的高分不等于 dCas9-KRAB 的转录抑制效率，尤其不能替代 TSS 位置、靶向链、启动子/增强子上下文和染色质可及性。可以将其分数保留为 QC 辅助列，但不能直接输入 `GuideStrengthEncoder`。

### CRISPRoff 和 epigenome-editing 模型

CRISPRoff 更接近长期 DNA methylation/表观遗传沉默，而不是普通 CRISPRi 的即时 knockdown。Batra et al., *eLife* 2023 的 `Predicting the effect of CRISPR-Cas9-based epigenome editing`（`10.7554/eLife.92991`，<https://pmc.ncbi.nlm.nih.gov/articles/PMC12795505/>）主要从 13 种 ENCODE 细胞类型的组蛋白修饰和转录组预测表观遗传编辑后的表达，并不是可替代的 CRISPRi guide efficacy head。它可作为长期沉默倾向或上下文特征来源，不能直接输出当前 `strength`。

## 7. 与 PerturbDiff 的整合方案

### 7.1 统一输入

统一记录至少包含：

```text
sequence              20 nt sgRNA protospacer
pam                   PAM（通常 NGG）
target_sequence       参考基因组 protospacer，可与 sequence 不同
strand                + / -
tss_distance          带方向的 guide-TSS 距离
genomic_context       可选上下游序列
chromatin_features    可选 ATAC、甲基化、核小体、表达等
```

`target_sequence` 与 `sequence` 必须同时保留。对 GSE132080 的 mismatch library，应明确模型是在预测实际 guide 的错配结合/抑制，还是在预测完全匹配 guide 的潜在活性；两者不能混成一个标签。

### 7.2 两阶段训练

1. 外部 predictor 冻结，输出 raw score、normalized score、source、context mode、missing features 和 uncertainty。
2. 在当前项目的 guide activity 表上只训练小型校准层（建议单调 affine/Softplus 或小 MLP），映射到 PerturbDiff 的 `strength`。
3. 保留现有 `GeneralizedSigmoid` 和显式 activity condition；外部分数不直接改变 latent delta，除非另有独立消融实验。
4. 对照 guide 固定为 0 或单独处理；缺失上下文必须显式记录，不能静默填充为有效 activity。
5. checkpoint 元数据保存 predictor source、权重/代码版本、参考基因组、TSS 版本、特征顺序和校准参数。

### 7.3 推荐的分阶段路线

| 阶段 | 默认 backend | 目的 |
|---|---|---|
| 第一阶段 | Horlbeck/CRISPRiaDesign 的现代重实现或按当前数据重训的 Elastic Net | 建立可解释、低依赖的人类 CRISPRi baseline |
| 第二阶段 | EpiCas-DL 的冻结 CRISPRi head | 在有匹配 ATAC/甲基化/表达时增加上下文信息 |
| 校准/验证 | Jost/Replogle library 与当前 128 条有 activity 的 guide | 检查排序、尺度和跨 library 泛化 |
| 未来扩展 | BarquistLab backend | 仅在明确加入细菌数据集时启用 |

### 7.4 当前分支的运行方式

仓库中的 `crispri_ruleset` 是用于接入验证和校准的透明启发式基线，使用序列质量、PAM、poly-T 和可用 TSS 距离；它不是 Horlbeck 已发表系数的复现。它从综合 guide 表读取序列、PAM 和可用上下文，不再要求 Table S8 的 26-mer：

```bash
python scripts/train_with_split.py \
  --activity-source predicted \
  --activity-predictor-backend crispri_ruleset \
  --external-activity-table results/comprehensive_sgrna_table.csv \
  ...
```

若已有外部模型导出的 guide 分数，可使用 `external_score`，并指定分数列：

```bash
python scripts/train_with_split.py \
  --activity-source predicted \
  --activity-predictor-backend external_score \
  --external-activity-table path/to/guide_scores.csv \
  --external-activity-col normalized_score \
  ...
```

两种路径都把外部结果压成一维 score，经过 `FrozenActivityCalibrator` 后才进入
`GeneralizedSigmoid`；control score 固定为 0。`activity_predictor_config.json` 会记录
backend、外部表和分数列。`table_s8_cnn` 仍保留为历史对照，并继续要求
`--guide-activity-cnn-table`。

## 8. 必须完成的比较实验

外部 predictor 本身和接入后的 PerturbDiff 必须分别评估。至少使用以下不泄漏划分：

1. guide-held-out：同一 target 的 guide 不跨 train/test；
2. target-held-out：测试目标基因训练中未出现；
3. library/dataset-held-out：测试文库或数据集完全留出；
4. GSE132080 的 perfect-match 与 mismatch 分层；
5. GSE283614 的 ESC/DE 状态分层，并记录 hg38 上下文完整性。

指标至少包括 guide-level Spearman/Pearson、top-k enrichment、target-level calibration，以及 PerturbDiff 下游 profile MSE/PCC。报告中同时给出：外部分数本身、加入校准层后的分数、对下游 profile 的影响、sequence-only/context-enhanced 差异、缺失上下文的比例。

只有当外部模型在 held-out guide 和 held-out target 上都优于当前 Table S8 CNN，且没有明显牺牲跨 library 性能时，才将其设为默认 backend。否则保留外部模型为可选后端，并继续使用当前 CNN 作为历史基线。

## 9. 结论

当前证据支持的选择是：**Horlbeck/CRISPRiaDesign 重实现作为默认人类 CRISPRi baseline，EpiCas-DL 作为有组学输入时的可选增强后端，Jost/Replogle 作为外部 benchmark/calibration 数据，BarquistLab 仅作为细菌扩展。** `CNN_CRISPRi_sgRNA_Efficacy` 尚无足够的权重、许可和泛化证据，不宜替换默认模型。Cas9 cleavage/knockout 和 CRISPRoff 模型不应直接充当 CRISPRi efficacy 输出。

在完成权重、输入预处理和许可证核对，以及上述 held-out 评估前，不应声称某个外部模型已经优于当前 Table S8 CNN。

## 参考链接

- Horlbeck et al. 2016: <https://doi.org/10.7554/eLife.19760>
- CRISPRiaDesign: <https://github.com/mhorlbeck/CRISPRiaDesign>
- EpiCas-DL paper: <https://doi.org/10.1016/j.csbj.2022.11.034>
- EpiCas-DL code: <https://github.com/yangqianq/EpiCas-DL>
- Jost/Replogle et al. 2022: <https://doi.org/10.7554/eLife.81856>
- Radzisheuskaya et al. 2016: <https://doi.org/10.1093/nar/gkw583>
- Bacterial mixed-effect model: <https://doi.org/10.1186/s13059-023-03153-y>
- Bacterial code: <https://github.com/BarquistLab/CRISPRi_guide_efficiency_bacteria>
- Public CNN notebook: <https://github.com/Elia-Lejzerowicz/CNN_CRISPRi_sgRNA_Efficacy>
- Epigenome-editing model: <https://doi.org/10.7554/eLife.92991>
