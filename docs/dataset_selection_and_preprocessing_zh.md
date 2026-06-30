# 数据集选择与预处理说明

本文档面向 `PerturbDiff` 的单细胞扰动建模任务，概述 `GSE132080`、`GSE257547` 与 `GSE283614` 的数据特征、对本项目的适配性，以及推荐的预处理流程。三者均属于公开单细胞扰动数据，但在扰动类型、测序模态与生物学问题上各有侧重，因此更适合分别承担主训练、强度建模与跨状态泛化验证等不同角色。

## 一、数据集适配性评估

| 数据集 | 研究主题 | 对本项目的适配性 | 建议角色 |
| --- | --- | --- | --- |
| `GSE132080` | 采用一系列逐步减弱的 CRISPR guide RNA 对人类基因表达进行滴定，并以 Perturb-seq 读出转录响应 | 适配性最高。该数据集具有清晰的 guide 标签、对照/扰动结构和连续强度变化，最适合作为主训练集 | 主训练集、强度头校准、方法学对照 |
| `GSE257547` | 通过 CRISPRa/CRISPRi 对转录因子剂量进行渐进调控，并采用 ECCITE-seq 获取转录组、sgRNA 与样本标记信息 | 很适合建模连续剂量响应，但表达矩阵为靶向 panel，和全转录组 Perturb-seq 不完全同构 | 辅助训练集、剂量响应验证、强度头外部验证 |
| `GSE283614` | 围绕 `OCT4` 与 `NANOG` 增强子库开展 Perturb-seq，在胚胎干细胞与 definitive endoderm 中比较扰动响应 | 适合检验细胞状态依赖性与增强子扰动的一般化能力，但实验设计更复杂，建议分状态处理 | 外部验证集、跨细胞状态泛化评估 |

从 `PerturbDiff` 的建模目标出发，`GSE132080` 更接近本项目的核心任务，因为它提供了相对规整的扰动标签与可解释的连续扰动梯度；`GSE257547` 最适合用于补充连续强度监督，尤其适合检验模型是否能够捕捉非线性剂量效应；`GSE283614` 更适合检验模型在不同细胞状态与不同扰动层级上的稳健性。若希望构建分层评估体系，推荐将 `GSE132080` 作为训练集，将 `GSE257547` 作为剂量响应外部验证集，并将 `GSE283614` 作为跨状态迁移测试集。

## 二、脚本步骤与参数对照

本项目已经新增 `scripts/preprocess_gse132080.py`，其处理顺序与参考脚本一致，但将原先 notebook 中的参数显式化，便于记录和复现。以下步骤与脚本参数一一对应。

| 步骤 | 脚本行为 | 对应参数 |
| --- | --- | --- |
| 1. 读取原始矩阵与 metadata | 从 `datasets/GSE132080/` 读取 `matrix.mtx.gz`、`barcodes.tsv.gz`、`genes.tsv.gz`，并将 metadata 按条形码拼接到 `adata.obs` | `--data-dir`、`--metadata-path`、`--barcode-col`、`--perturbation-col` |
| 2. 统一扰动标签 | 将 `neg_ctrl`、`non-targeting`、`non_targeting` 映射为 `control`，其余标签默认取第一个下划线前的 token | `--clean-perturbation-col`，内部清洗规则 |
| 3. 依据实验注释做预筛选 | 如存在 `good_coverage`，仅保留为真的细胞；如存在 `number_of_cells`，仅保留 `<= 1` 的细胞 | 直接读取 metadata 字段，无额外参数 |
| 4. 细胞级 QC | 计算 `total_counts`、`n_genes_by_counts`、`pct_counts_mt`，并按阈值过滤细胞 | `--min-counts`、`--min-genes`、`--max-genes`、`--max-pct-mt` |
| 5. 基因级 QC | 过滤在过少细胞中表达的基因，并去除线粒体与核糖体基因 | `--min-cells-per-gene` |
| 6. 标准化与对数变换 | 将原始计数保存在 `adata.layers["counts"]`，再执行总量归一化和 `log1p` | `--target-sum` |
| 7. 高变基因选择 | 以 `scanpy.pp.highly_variable_genes` 选取高变基因，并可保留目标基因 | `--n-top-genes`、`--hvg-flavor`、`--skip-target-gene-retention` |
| 8. 保存结果 | 输出 `.h5ad` 与 JSON 统计摘要 | `--output-h5ad`、`--summary-json` |

三个数据集的分工也建议与脚本保持一致：`GSE132080` 作为主训练集，使用上述 QC 与 HVG 参数；`GSE257547` 若纳入后续实验，可先沿用同样的条形码对齐、对照归一和标准化流程，再根据其靶向 panel 的大小适当放宽 `--min-cells-per-gene` 与 `--n-top-genes`；`GSE283614` 若纳入跨状态评估，则建议将 ESC 和 DE 分开预处理，并保持同一组标准化和 HVG 逻辑，以便做更干净的比较。

## 三、数据集层面的建议

本研究整合三个公开单细胞扰动数据集，分别对应 CRISPR guide 滴定、转录因子剂量调节以及增强子扰动三类任务。`GSE132080` 包含人类细胞中以系统性减弱 CRISPR guide 为核心的 Perturb-seq 数据，具有较为清晰的 `cell_barcode` 与 `guide_identity` 注释，适合作为主训练集。`GSE257547` 采用 ECCITE-seq 获取 cDNA、guide-derived oligo（GDO）与 hashtag oligo（HTO）等多模态信息，其中 cDNA 表达矩阵适合用于转录响应建模，而 GDO 与 HTO 适合用于扰动归属和样本来源识别。`GSE283614` 则包含胚胎干细胞与 definitive endoderm 两个状态下的 Perturb-seq 数据，围绕 `OCT4` 与 `NANOG` 增强子扰动展开，适合用于研究细胞状态依赖的扰动效应。

在 `GSE132080` 上，推荐采用与脚本一致的默认参数：`MIN_COUNTS=1000`、`MIN_GENES=200`、`MAX_GENES=10000`、`MAX_PCT_MT=20.0`、`MIN_CELLS_PER_GENE=3`、`N_TOP_GENES=1000`、`TARGET_SUM=1e4`。这些阈值与参考 notebook 保持一致，既能够去除低质量细胞，又不会在该数据集上过度压缩扰动响应信号。若后续数据扩展到 `GSE257547` 或 `GSE283614`，则建议在相同的归一化与 `log1p` 逻辑下，结合 panel 大小和目标基因覆盖情况再微调 `MIN_CELLS_PER_GENE` 与 `N_TOP_GENES`，避免把关键扰动靶点筛掉。

在表达矩阵处理上，推荐先按细胞进行总量标准化，再执行 `log1p` 转换，以减弱测序深度差异并提高优化稳定性。若模型需要保留原始计数，可将原始矩阵另存于 `adata.layers["counts"]`，而将标准化后的矩阵用于主训练。基因选择方面，建议在满足最低表达阈值后选取高变基因，并强制保留目标扰动基因及其已知响应标记，以避免模型在训练过程中丢失关键的因果锚点。跨数据集联合训练时，应优先选择三者的公共基因集合，再按任务需求补充特异性 marker 或目标基因。

## 四、可直接写入论文的方法学表述

本研究使用三个公开单细胞扰动数据集构建训练与验证体系。`GSE132080` 为人类细胞中基于逐步减弱 CRISPR guide RNA 的 Perturb-seq 数据，包含原始计数矩阵及细胞条形码和扰动注释信息，适合作为主训练集以学习扰动标签与转录响应之间的映射。`GSE257547` 通过 CRISPRa/CRISPRi 对转录因子剂量进行渐进调控，并结合 ECCITE-seq 获取转录组与 guide 信息，可用于刻画连续扰动强度及其非线性下游效应。`GSE283614` 则围绕 `OCT4` 与 `NANOG` 增强子库开展单细胞 Perturb-seq，并在胚胎干细胞与 definitive endoderm 两种细胞状态下比较扰动响应，适合用于评估模型在跨状态情形下的泛化能力。

在预处理上，首先依据细胞条形码、guide 标识及质量控制字段完成样本对齐，并将对照细胞统一标记为 `control`。随后对低质量细胞、低丰度基因与异常样本进行过滤；对表达矩阵进行按细胞归一化和对数转换；再根据分析目标选取高变基因，并在跨数据集设置下保留共同基因与关键目标基因。对于 `GSE257547` 与 `GSE283614` 这类具有更复杂实验设计或靶向表达面板的数据，应在划分训练、验证与测试集时显式区分数据集来源、细胞状态及扰动层级，以减少批次效应对模型评估的干扰。

## 参考链接

- `GSE132080` GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132080
- `GSE257547` GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE257547
- `GSE283614` GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE283614
- `GSE132080` 原始论文: https://www.nature.com/articles/s41587-019-0387-5
- `GSE257547` 相关论文: https://elifesciences.org/articles/100555
- `GSE283614` 相关研究页: https://www.biorxiv.org/content/10.1101/2025.08.07.669196
