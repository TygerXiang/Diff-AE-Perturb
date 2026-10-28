# 引言与 Research Gap 答辩导师评审稿

评审对象：`C:\Users\Admin\Downloads\introduction0930_向凯泽_硕士开题报告v1.docx`  
评审范围：引言正文 6 个主体段落、参考文献和 DOCX 中现有的 12 条导师批注。  
评审方式：逐段逻辑审查、引用与本地 Zotero 全文/元数据核对、方法条件与泛化任务对照。  
原始 DOCX 未修改。

## 一、答辩总体判断

当前引言已经具备清楚的研究方向：从单细胞 CRISPR/Perturb-seq 的高维观测出发，讨论细胞状态、扰动效应和 sgRNA 活性差异，最后引出条件扩散模型。但是它还没有把“为什么必须提出这个模型”证明到答辩可防守的程度。

最大风险集中在第 5 段 research gap：文本把药物剂量、guide 活性和 CRISPRi 敲减效率放在同一层面，又把“现有主流方法主要是离散标签”写成没有检索边界的绝对判断。Song 等 2025 已经展示了从 CRISPRi Perturb-seq 下游表达中估计连续 perturbation-response score，并用于部分扰动和剂量分析；Jost 等 2020 也已经用系统衰减的 sgRNA 活性产生连续表达梯度。因此，论文的创新点不能表述为“首次考虑 guide-level 信息”，而应限定为：在同一条件扩散生成框架中，把细胞状态、target-gene effect 与显式 sgRNA activity 联合建模，并在 held-out guide、连续活性和单细胞分布层面验证其增益。

答辩时建议用以下五句闭环：

1. 本文研究的是 CRISPRi 单细胞转录响应预测，目标是生成给定基线细胞状态和扰动条件下的单细胞表达分布。
2. 现有模型已经分别处理了潜在细胞状态、target/perturbation 表示、药物剂量、组合扰动、知识图谱或跨 context 泛化，但这些条件变量不是同一个概念。
3. 现有实验和分析工作也证明了同一 target 下不同 guide 或不同部分扰动强度会产生连续且异质的响应，因此 target label 不能完全代表实际扰动强度。
4. 在本文纳入比较的公开模型和 CRISPRi 数据中，尚未确认有一个条件生成框架同时把基线细胞状态、target-gene effect 和可测量或可预测的 guide activity 作为联合条件，并针对 held-out guide 的分布生成进行系统验证。
5. 本文将通过 target-only、去除 cell-state、去除 guide activity、离散 activity 与连续 activity 等消融，验证该联合条件是否带来独立收益。

## 二、逐段导师评审

### P0001：CRISPR 筛选背景与单细胞 CRISPR 动机

**段落功能**：介绍 pooled screen、arrayed screen 和单细胞 CRISPR 筛选之间的关系。

**等级：必须修改。**

主要问题：

- 首句主语和因果关系不清。“基于 CRISPR 的基因筛选技术能够检测……”没有说明检测对象是基因功能、调控元件还是细胞表型。
- “传统混合筛选”应统一为 pooled CRISPR screen；“数百条 sgRNA”过窄，文库可以达到数千条甚至全基因组规模。
- “细胞嵌合体”不准确，应改为“池化细胞群”或“混合细胞群”。
- pooled 与 arrayed 是实验设计路线，不是前后替代关系；应先说明筛选目的，再比较通量、表型维度和可追踪性。
- “阵列式筛选能够准确验证基因功能”过度概括，应说明它通常具有更明确的单扰动—表型对应关系，但通量和成本受限。
- “single cell CRISPR”不是全文应固定使用的术语。建议统一为“单细胞 CRISPR 筛选”或“pooled single-cell CRISPR screening”，随后引出 Perturb-seq/CROP-seq。
- [1] 是综述，不能单独承担全部筛选技术史。Datlinger 2017 的 [4] 当前列入参考文献但正文完全未引用，建议在此处或下一段引入。

**建议结构**：先定义筛选问题（从候选基因/调控元件中识别影响表型的扰动），再介绍 pooled screen 的群体富集读出和 arrayed screen 的单扰动读出，最后说明单细胞 CRISPR 将 guide 身份与单细胞分子表型配对，从而把“是否富集”扩展为“每个细胞的转录响应是什么”。

### P0002：Perturb-seq 的信息增益与建模需求

**段落功能**：说明 Perturb-seq 如何扩展表型维度，并引出计算预测。

**等级：必须修改。**

主要问题：

- 需要定义数据单位：单细胞表达矩阵、guide/perturbation identity、细胞状态和基因表达计数如何配对；也需要解释为什么称为 Perturb-seq。
- “前所未有的分辨率”应改为可核验的具体能力：单细胞、全转录组、多扰动并行观测，以及对异质响应的分析能力。
- “数百万个单细胞、数百至数千个基因”必须和具体数据集对应，不能把某一篇 genome-scale CRISPRi Perturb-seq 的规模泛化为所有研究。
- [6] Nelson 2015 支持的是人类遗传证据对已批准药物适应证的支持，不是 Perturb-seq 对药物开发的直接实证。建议改为“可为靶点优先级和机制假设提供候选证据”，或补充直接研究。
- “数据量指数增长”和“多重扰动成本高昂”当前没有直接来源，应改为可验证的规模/成本限制，或删去“指数”这种强表述。
- 从实验数据到预测模型之间少了一步：说明实验只能覆盖有限的 target、guide、cell state 和 context，因此需要模型在未观测条件下生成反事实响应。

**建议补充**：Datlinger 2017 的 CROP-seq 说明 guide 表达可以在单细胞 RNA 测序中被捕获；Dixit 2016、Adamson 2016 和 Replogle 2022 分别支撑 Perturb-seq 的高维读出、细胞状态异质性和 genome-scale 扩展。

### P0003：噪声、guide 活性和细胞异质性

**段落功能**：提出需要建模的三个挑战。

**等级：中高优先级修改。**

优点：三类挑战已经接近本文真正的建模对象，整体逻辑比前两段清楚。

主要问题：

- [8] scVI 可以支持计数分布、潜变量和部分技术因素建模，但不能直接证明“因此必须在潜空间中建模扰动效应”。应把后半句改成作者的方法选择，并给出为什么潜空间适合压缩背景变化的解释。
- 全文应区分 CRISPRi 与 CRISPR knockout。“sgRNA 敲除效率”在 CRISPRi 语境下应改为“sgRNA 抑制效率”或“敲减效率”。
- [9] Horlbeck 2016 支持 guide 序列、靶点位置和活性差异；“局部染色质状态”需要更直接的 guide-design 或 CRISPRi efficacy 文献支持，不能只靠这一篇。
- “不同和响应状态”是明显语病，应改为“不同的响应状态”。
- Adamson 2016 的 UPR 案例需要说明：相同扰动下出现不同 UPR 分支/亚群，说明总体平均表达可能掩盖细胞级机制，而不是简单说明“数据有异质性”。
- “仅使用靶基因身份不足”是推论，应进一步写出可识别性问题：guide 活性、细胞基线状态、转导效率、捕获效率和批次效应可能共同决定观测差异。

**建议新增证据**：Song 2025 直接把 perturbation efficiency、cell type/cell state、cofactor activity 和 on/off-target effects 作为异质性的来源，并对 CRISPRi 部分扰动进行连续 PS 估计；它比单独使用 Adamson 2016 更接近本文 gap。

### P0004：计算模型综述

**段落功能**：定位 scGen、CPA、GEARS、scGPT、STATE 和扩散模型。

**等级：必须修改。**

当前问题不是方法太少，而是每个方法只列出名称和一句能力，无法形成“方法—输入条件—解决的问题—局限—与本文关系”的阅读证据。

建议按下面的顺序展开：

| 方法 | 主要框架 | 条件/输入 | 主要任务 | 与本文 gap 的关系 |
|---|---|---|---|---|
| scGen | VAE 潜空间向量运算 | 离散 perturbation attribute、细胞/批次等 | 细胞状态和研究间的响应迁移 | 说明早期模型将扰动作为属性条件，不等于连续 sgRNA activity |
| CPA | factorized latent representation | perturbation、dose、time、covariate | 剂量、时间、组合、物种/细胞类型泛化 | 有连续 dose/time，但药物或实验 covariate 不能直接等同 CRISPRi guide efficacy |
| GEARS | 知识图谱/基因关系先验 | gene perturbation 与基因关系 | 未见单基因和组合基因扰动 | 有 target-level 先验，但不是 guide sequence/activity 模型 |
| scGPT | 单细胞基础模型 | 预训练表达与任务条件 | 表达建模和多任务迁移 | 不能直接写成已解决 perturbation-response 或 guide-level 问题 |
| STATE | transformer + cell embedding | gene/perturbation 与大规模 cell context | 跨 context 的群体响应预测 | 已处理 cell heterogeneity/context，但当前本地条目是 bioRxiv 预印本，且未等同 guide-level activity |
| scDiffusion | 条件扩散生成 | 条件细胞状态/生物学条件 | 单细胞分布生成 | 说明扩散模型适合分布生成，但未必是遗传 perturbation 专门模型 |
| dbDiffusion | latent diffusion + perturbation embedding | 连续 perturbation-level embedding | 未观测遗传扰动、去偏统计推断 | 连续扰动表示与本文相关，但 embedding 不是 sgRNA sequence/activity |
| scPPDM | diffusion + factorized classifier-free guidance | pre-perturbation state、drug、dose | unseen drug 和 unseen covariate combination | 是 drug-dose 模型，不能作为 CRISPRi guide-level 直接先例 |
| Squidiff | diffusion | 遗传扰动/发育相关条件 | 细胞发育和扰动响应 | 可作为扩散对照，但任务与本文 guide-level 条件不同 |

前置过渡句应解释：单细胞表达是高维、稀疏、带有多峰和细胞间异质性的分布；生成模型的价值在于预测完整响应分布和反事实样本，而不是只预测一个群体均值。扩散模型是否优于 VAE 或线性基线，应由后续实验验证，不能在引言中预设。

### P0005：Research gap

**段落功能**：从文献现状归纳本文研究缺口。

**等级：最高优先级。**

必须修改的地方：

- “现有主流方法主要将遗传扰动视为离散类别标签”过于绝对。scGen 的确使用离散 perturbation attribute，但 CPA 已有连续 dose/time，dbDiffusion 使用连续 perturbation embedding，Song 2025 也能从下游表达估计连续 perturbation-response score。
- “外部预先测定的剂量信息”混淆了药物 dose 与 CRISPRi guide efficacy。前者是外加药物浓度，后者是 guide 对靶基因表达抑制的实际强度，二者不能在同一术语下描述。
- “细胞语义状态、扰动方向、guide-level 活性”必须操作化：细胞状态是基线表达或其 latent representation；target effect 是 target identity/embedding 或目标特异的 effect vector；guide activity 是实测 efficacy 标量、由序列模型预测的活性，或两者之一。
- “难以直接从表达矩阵中观测”应改为“通常不能仅凭单个表达矩阵把 guide efficacy 与基线状态、转导效率和下游响应完全分离”，并说明需要多 guide、阴性对照、重复或外部效率标签。
- “当前仍缺乏一个……”需要给出检索范围和比较矩阵。没有矩阵支撑时，应使用“尚未确认”“在本文纳入的方法中未见”而不是领域绝对断言。
- 本段当前没有正文引文，至少应连接 scGen、CPA、guide-efficacy 实验文献、Song 2025 和扩散/benchmark 文献。

**建议直接替换为以下版本**：

> 现有单细胞扰动预测方法已经分别从不同角度建模细胞状态、扰动身份、药物剂量、组合关系、知识先验或跨细胞环境泛化，但这些条件变量不能直接互换。对于 CRISPRi，靶向同一基因的不同 sgRNA 可能产生不同程度的表达抑制，实验中的部分扰动和未完全转导也会使“guide 身份”与“实际扰动强度”不再等价。已有工作能够从下游表达中估计连续扰动响应，或通过系统设计的低活性 sgRNA 形成表达梯度，但这些分析方法与单细胞反事实生成模型解决的是不同问题。在本文纳入比较的公开模型和 CRISPRi 数据中，尚未确认有一个条件扩散生成框架同时将基线细胞状态、target-gene effect 和可测量或可预测的 sgRNA activity 作为联合条件，并针对 held-out guide 的单细胞分布生成进行系统评估。因此，本文将把上述三类信息作为可分离的条件输入，并通过 target-only、去除 cell-state、去除 guide activity 以及离散/连续 activity 对照，检验 guide-level 条件是否带来独立的预测收益。

这版 gap 的优点是把“已有工作已经覆盖的内容”和“本文仍要验证的联合条件”分开，也把文献事实、作者推断和实验假设分开。

### P0006：本文模型与贡献表述

**段落功能**：把 gap 映射到拟议模型。

**等级：必须修改。**

主要问题：

- “sgRNA 敲除效率”应统一为“sgRNA 抑制/敲减效率”。
- “为每个靶基因学习一个潜空间方向向量”意味着目标基因必须在训练中出现；若实验包含 unseen target，需要基因序列、基因网络或预训练 embedding，否则无法泛化。
- 必须说明 guide activity 的来源：外部实验测量、序列特征预测，还是二者都支持；还要说明训练集、缺失值和潜在标签泄漏。
- “生成接近真实的单细胞分布”和“保留异质性”是结果性承诺，应改成研究目标或待验证假设，并给出评价指标。
- 需要明确任务拆分：当前最适合的主任务是 seen-target/unseen-guide；如果还要做 unseen-target 或 cross-context，应另列任务并改变模型接口。

**建议直接替换为以下版本**：

> 本文拟提出并评估一个面向 CRISPRi 单细胞转录响应预测的条件扩散框架。模型将基线细胞表达编码为细胞状态表示，将目标基因表示为 target-level 条件，并将实测或由 guide 序列特征预测得到的 sgRNA 抑制效率作为连续 guide-level 条件。模型的主要研究假设是：在 target 相同而 guide 活性不同的情况下，显式引入 guide-level 条件能够改善对响应幅度和细胞间异质性的预测。研究将以 held-out guide/seen-target 为主要泛化任务，并与 target-only、连续条件移除和简单生成基线比较；模型效果将通过表达误差、delta/DE 恢复、分布相似性、异质性保持和 activity 分层校准进行评估。若进一步开展 unseen-target 任务，则需要补充目标基因的序列或网络先验，并单独报告该任务的可识别性假设。

## 三、文献与引用审计

### 当前正文引用问题

| 条目 | 当前状态 | 评审结论 |
|---|---|---|
| [1] Cheng et al. | 正文用于单细胞 CRISPR 总述 | 可保留，但应修正书目年份/卷期一致性；Zotero/PDF 显示 Adv. Sci. 2023, 10, 2204484，online 2022 可作为补充信息 |
| [2] Dixit et al. | Perturb-seq 奠基文献 | 可支撑高维单细胞读出；建议与 Datlinger 2017 并用 |
| [3] Adamson et al. | UPR 异质响应案例 | 可保留，但要明确它证明的是同一扰动下的分支响应，而非 guide efficacy 模型 |
| [4] Datlinger et al. | 参考文献中存在，正文未出现 | 必须在 Perturb-seq/CROP-seq 定义段使用；Zotero 有元数据但无本地 PDF 附件 |
| [5] Replogle et al. | genome-scale Perturb-seq | 需核对原文中的规模数字和实验类型，避免泛化为所有 CRISPRi 数据 |
| [6] Nelson et al. | 用于药物开发动机 | 论断超出文献范围；限定为遗传证据对药物靶点/适应证验证的支持，或换成直接 Perturb-seq 研究 |
| [7] Bunne et al. | virtual-cell 宏观动机 | 可保留为观点/综述性动机，不承担“指数增长”和成本的实证论断 |
| [8] Lopez et al. / scVI | 用于潜变量和噪声 | 可支撑生成模型处理技术因素，但不能直接支撑 CRISPR 扰动建模结论 |
| [9] Horlbeck et al. | guide 活性差异 | 可支撑序列/位置与活性差异；局部染色质状态需补更直接文献 |
| [10] scGen | 潜空间扰动偏移 | 可支撑离散 perturbation attribute 与 latent arithmetic；不要把它描述为连续 guide activity 模型 |
| [11] CPA | dose/time/covariate | 原文确实支持 dose、time、cell type、species 和组合泛化，但 dose/time 不等于 sgRNA efficacy |
| [12] GEARS | 知识图谱/组合扰动 | 可支撑 target-level 和组合泛化；不支持 guide-level 条件结论 |
| [13] scGPT | 基础模型 | 应单独说明预训练表达模型及其任务，不要和 GEARS、STATE 统称为同类方法 |
| [14] STATE | 条目信息不完整 | Zotero 本地 PDF 已存在；应补为 2025 bioRxiv 预印本，DOI 10.1101/2025.06.26.661135，并标注未同行评审 |
| [15] PertDiffBench | 2026 bioRxiv | 是基准/评测资源，不应与预测方法并列；正文需标注 preprint |
| [16] scDiffusion | 扩散生成 | 需限定为相应的条件单细胞生成任务，不能自动等同遗传 perturbation-response 模型 |
| [17] dbDiffusion | 未见扰动/连续 embedding | 可作为连续 perturbation representation 对照，但其 embedding 来自 perturbation-level 信息，不是 sgRNA sequence/activity |
| [18] scPPDM | 条目不完整 | Zotero 有 PDF；应标为 arXiv:2510.11726v1（2025-10-08），明确是 drug-dose 条件，不是 CRISPRi guide-level 模型 |
| [19] Squidiff | 扩散模型 | 可作为发育/扰动扩散对照，需说明与本文任务不同 |

### 建议新增或补齐的核心文献

以下条目已经在 Zotero 中有可核验记录或 PDF，优先级最高：

1. Jost et al. *Titrating gene expression using libraries of systematically attenuated CRISPR guide RNAs*. **Nature Biotechnology** 38, 355–364 (2020). DOI: 10.1038/s41587-019-0387-5。它直接证明 sgRNA 活性可以被系统调谐，并形成连续表达和表型梯度。
2. Song et al. *Decoding heterogeneous single-cell perturbation responses*. **Nature Cell Biology** 27, 493–504 (2025). DOI: 10.1038/s41556-025-01626-9。它直接支持部分 perturbation、连续 PS、perturbation efficiency 与 cell-state/cofactor 共同影响响应。
3. Viñas Torné et al. *Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation*. **Nature Biotechnology** (2025). DOI: 10.1038/s41587-025-02777-8。它支持在 research gap 中加入“评估协议可能高估模型性能”的一层。
4. Li et al. *A systematic comparison of single-cell perturbation response prediction models*. **Science Advances** 12, eaed3414 (2026). DOI: 10.1126/sciadv.aed3414。其摘要报告 13 个方法、25 个数据集、3 类任务和 24 个指标，并指出跨细胞类型没有方法稳定泛化，基础模型存在方差压缩和低估组合协同效应的问题。
5. Ahlmann-Eltze, Huber & Anders. *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines*. **Nature Methods** 22, 1657–1661 (2025). DOI: 10.1038/s41592-025-02772-6。适合支撑“复杂模型需要和简单基线及严格指标比较”。

对于 P0001 的技术史，如果篇幅允许，建议继续核验 Shalem et al. 2014、Wang et al. 2014、Gilbert et al. 2014 和 Norman et al. 2019，分别补充 genome-scale CRISPR、CRISPRi/a 和组合 Perturb-seq 的原始研究。它们应只用于技术背景和实验范式，不直接承担本文的 guide-level 生成模型 gap。

### 需要用户后续补入本地全文的条目

- Datlinger et al. 2017：Zotero 中已有元数据和 DOI 10.1038/nmeth.4177，但当前没有 PDF 附件。若要逐页核对 CROP-seq 的技术细节，请补 PDF。
- Cui et al. scGPT 2024：当前未发现本地全文附件。涉及 scGPT 的具体任务、训练数据和 perturbation 结果时，请补 DOI 对应 PDF。
- Mixscape/Mixscale 与 SC-VAE 原文：Song 2025 中已讨论其与部分扰动/效率建模的关系，但本地未找到完整原文，暂不把它们写成本文 gap 的确定反例或支持证据。

## 四、建议的验证与消融清单

为了让 gap 与方法贡献形成闭环，methods 和实验部分至少应明确：

| 实验 | 目的 |
|---|---|
| target-only baseline | 判断只用 target identity 是否已经足够 |
| target + cell-state | 判断基线细胞状态是否改善响应预测 |
| target + guide activity | 判断 guide-level 条件是否独立贡献信息 |
| target + cell-state + guide activity | 验证完整联合条件 |
| 离散 guide label vs 连续 activity | 检验连续条件是否优于 one-hot guide identity |
| activity 分层校准 | 检查预测响应强度是否随 activity 单调变化，避免只得到平均表达 |
| held-out guide / seen target | 验证本文最直接的 guide 泛化任务 |
| unseen target（若开展） | 只有在加入 gene sequence/network prior 后才可作为独立任务 |
| 简单线性/均值/ scGen/CPA/GEARS 基线 | 防止把扩散模型复杂度误当作性能来源 |
| 分布级指标与 DE/delta 指标 | 避免仅用 expression-level MSE 掩盖方差压缩和异质性丢失 |

## 五、交付状态与后续修改顺序

本文件已经完成文献检索、Zotero 元数据核验、正文逐段评审和导师批注整合。当前没有修改原始 DOCX，也没有向 Zotero 写入条目。结合工作区已有 benchmark 口径，本文拟方法的主要泛化轴应明确写为 **unseen guide / seen target**：测试 guide 未在训练中出现，但对应 target gene 在训练中出现；如果未来要声称 unseen-target 泛化，必须增加 gene sequence/network prior 并单独设计任务。

推荐修改顺序：

1. 先重写 P0001–P0002，定义 pooled/arrayed/Perturb-seq 和实验数据单位；
2. 再重写 P0003，统一 CRISPRi、guide efficacy、细胞状态和异质性术语；
3. 将 P0004 改成方法对照段，避免只列模型名称；
4. 用本报告 P0005 的五句闭环替换现有 gap；
5. 等 methods 和数据划分确定后，再压缩 P0006，避免过早承诺模型性能；
6. 最后统一参考文献元数据，并补 Datlinger/scGPT/Mixscale/SC-VAE 的全文附件。

## 参考来源与核验范围

本评审使用了 DOCX 正文及其 12 条未解决导师批注，并通过本地 Zotero MCP 核验了 Jost 2020、Song 2025、Systema 2025、Li 2026、Ahlmann-Eltze 2025、STATE、scPPDM、Datlinger 2017 等条目的元数据和可用全文。页码没有写入评审定位，因为原始 DOCX 没有可靠的分页标记，当前环境也无法使用 LibreOffice 渲染；正文定位采用 P0001–P0006 和段落首句，最终排版后再核对页码。
