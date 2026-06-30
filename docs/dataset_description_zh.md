# 数据集描述与预处理流程

本文档描述本项目当前本地保存并可用于建模的两个单细胞扰动数据集：`GSE132080` 和 `GSE283614`。其中 `GSE132080` 已有明确的每细胞 guide 注释文件；`GSE283614` 的本地文件为 10x 风格矩阵，表达特征和 CRISPR guide capture 特征在同一个矩阵中，需要在预处理时从 guide capture 计数中推断每个细胞的 guide 身份。

## 一、GSE132080

### 1. 数据集基本信息

`GSE132080` 是人类 K562 细胞中的 CRISPRi Perturb-seq 数据。实验通过多个 CRISPR guide RNA 对一组 target gene 进行转录抑制，并在单细胞 RNA-seq 层面读取扰动后的转录组响应。该数据集适合用于学习 sgRNA / target gene 条件与单细胞表达响应之间的映射。

本地文件位于 `datasets/GSE132080/`：

| 文件 | 内容 |
| --- | --- |
| `matrix.mtx.gz` | 原始 UMI count 矩阵 |
| `barcodes.tsv.gz` | 细胞 barcode |
| `genes.tsv.gz` | 基因列表 |
| `GSE132080_cell_identities.csv.gz` | 每细胞 guide 注释与质量控制信息 |

原始表达矩阵包含 `23,633` 个 barcode 和 `33,694` 个基因；metadata 中有 `23,608` 条细胞注释。预处理时按 `cell_barcode` 将表达矩阵与 metadata 对齐。

### 2. target gene 与 sgRNA

metadata 中的扰动字段为 `guide_identity`。本地统计显示，该字段共有 `139` 个不同标签：

- `128` 个靶向 sgRNA；
- `10` 个 non-targeting / negative control sgRNA；
- `1` 个 `*` 标签。

靶向 sgRNA 覆盖 `25` 个 target gene：

`ALDOA`、`ATP5E`、`BCR`、`CAD`、`CDC23`、`COX11`、`DBR1`、`DUT`、`EIF2S1`、`GATA1`、`GINS1`、`GNB2L1`、`HSPA5`、`HSPA9`、`HSPE1`、`MTOR`、`POLR1D`、`POLR2H`、`RAN`、`RPL9`、`RPS14`、`RPS15`、`RPS18`、`SEC61A1`、`TUBB`。

该数据集不是每个 target gene 只有一个 sgRNA，而是每个 target gene 对应多个 sgRNA。绝大多数 target gene 有 `5` 个 sgRNA，`BCR`、`GATA1` 和 `RPS14` 各有 `6` 个 sgRNA，因此总靶向 sgRNA 数为 `128`。对照组包含 `10` 个 non-targeting sgRNA。

guide 名称中包含 target gene、链方向、基因组位置、promoter 注释和 guide 编号等信息，例如：

```text
DUT_DUT_+_48624411.23-P1P2_01
POLR1D_POLR1D_+_28196016.23-P1_07
SEC61A1_SEC61A1_-_127771295.23-P1_00
```

这说明 sgRNA 设计与 target gene 附近的转录调控区域相关，并通过不同位置或不同编号的 guide 产生不同程度的 CRISPRi 抑制效果。训练时可以将 guide 折叠为 target gene 级别标签，也可以保留完整 sgRNA 标签进行 guide-level 条件建模。

### 3. `*` guide 标签核查

`guide_identity == "*"` 不是本项目预处理脚本清洗造成的，而是原始 metadata 文件 `GSE132080_cell_identities.csv.gz` 中已经存在的标签。原始文件中共有 `311` 行细胞的 `guide_identity` 为 `*`。这些行仍然包含 `read_count`、`UMI_count`、`coverage`、`gemgroup`、`good_coverage` 和 `number_of_cells` 等字段，因此它们更像是原始 guide assignment 中未明确归属或异常归属的细胞，而不是表达矩阵读取或标签清洗时产生的新标签。

在这 `311` 个 `*` 细胞中，只有 `7` 个满足 `good_coverage == True`，也只有 `7` 个满足 `number_of_cells == 1`。默认预处理流程会先根据 coverage 和单细胞过滤去除绝大多数 `*` 细胞；在当前默认预处理 summary 中最终仍保留 `7` 个 `*` 标签细胞。因此，如果后续训练需要严格的扰动标签，建议将 `*` 视为 unknown / unassigned guide，并在训练前显式过滤，或单独映射到 `unknown` 类而不要并入 `control`。

### 4. 细胞与基因规模

| 阶段 | 细胞数 | 基因数 | 说明 |
| --- | ---: | ---: | --- |
| 原始 10x 矩阵 | 23,633 | 33,694 | `matrix.mtx.gz`、`barcodes.tsv.gz`、`genes.tsv.gz` |
| 原始 metadata | 23,608 | - | `GSE132080_cell_identities.csv.gz` |
| 默认预处理结果 | 19,579 | 1,019 | `gse132080_preprocessed.h5ad`，1000 HVG 并额外保留 target gene |
| paper-style 预处理结果 | 23,608 | 2,000 | `gse132080_paper_style_hvg2000.h5ad`，保留 2000 HVG |

默认预处理结果中，清洗后的 target-level 扰动包含 `25` 个 target gene 和 `control`。其中有 `21` 个 target gene 的表达特征在基因 QC 后仍存在于矩阵中，并在 HVG 选择后被强制保留；最终基因数为 `1,019`，即 `1,000` 个 HVG 加上额外保留的 target gene。

### 5. 默认预处理流程

默认预处理脚本为 `scripts/preprocess_gse132080.py`，主要步骤如下：

1. 读取 `matrix.mtx.gz`、`barcodes.tsv.gz` 和 `genes.tsv.gz`，构建 AnnData 对象。
2. 读取 `GSE132080_cell_identities.csv.gz`，按 `cell_barcode` 与表达矩阵对齐。
3. 清洗扰动标签：non-targeting / negative control 统一为 `control`；非对照 guide 默认取第一个下划线前的 token 作为 target-level 标签。
4. 若存在 `good_coverage` 字段，仅保留覆盖质量为真的细胞。
5. 若存在 `number_of_cells` 字段，仅保留 `number_of_cells <= 1` 的单细胞。
6. 进行细胞级 QC：默认保留 `total_counts >= 1000`、`n_genes_by_counts >= 200`、`n_genes_by_counts <= 10000` 且 `pct_counts_mt <= 20.0` 的细胞。
7. 进行基因级 QC：保留至少在 `3` 个细胞中表达的基因，并去除线粒体和核糖体基因。
8. 将原始 count 保存到 `adata.layers["counts"]`。
9. 对每个细胞进行 total-count normalization，默认 `target_sum = 1e4`。
10. 执行 `log1p` 转换。
11. 使用 Scanpy 的 `highly_variable_genes` 选择默认 `1000` 个高变基因。
12. 额外保留表达矩阵中存在的 target gene，避免关键扰动靶点被 HVG 筛选丢弃。
13. 保存 `.h5ad` 文件和 JSON summary。

默认参数为：

```text
MIN_COUNTS = 1000
MIN_GENES = 200
MAX_GENES = 10000
MAX_PCT_MT = 20.0
MIN_CELLS_PER_GENE = 3
N_TOP_GENES = 1000
TARGET_SUM = 1e4
HVG_FLAVOR = seurat
```

## 二、GSE283614

### 1. 数据集基本信息

`GSE283614` 是人类细胞中的 single-cell Perturb-seq 数据。GEO 记录显示，该数据集的实验设计为在 embryonic stem cell（ESC）和 definitive endoderm（DE）两个状态下，对 `OCT4` 和 `NANOG` enhancer perturbation library 进行单细胞 Perturb-seq。该数据集适合用于评估模型在不同细胞状态下对增强子扰动响应的泛化能力。

本地文件位于 `datasets/GSE283614/`，分为 `ESC` 和 `DE` 两个子目录：

| 状态 | 文件 | 内容 |
| --- | --- | --- |
| ESC | `GSE283614_ESC_perturb-seq_matrix.mtx.gz` | ESC 原始矩阵 |
| ESC | `GSE283614_ESC_perturb-seq_barcodes.tsv.gz` | ESC 细胞 barcode |
| ESC | `GSE283614_ESC_perturb-seq_features.tsv_2.gz` | ESC feature 注释 |
| DE | `matrix.mtx.gz` | DE 原始矩阵 |
| DE | `barcodes.tsv.gz` | DE 细胞 barcode |
| DE | `features.tsv.gz` | DE feature 注释 |

本地矩阵维度如下：

| 状态 | barcode 数 | feature 数 | Gene Expression feature | CRISPR Guide Capture feature | nonzero entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| ESC | 12,844 | 33,586 | 33,538 | 48 | 62,837,632 |
| DE | 34,012 | 33,586 | 33,538 | 48 | 120,651,804 |

因此，`GSE283614` 本地总计包含 `46,856` 个细胞 barcode；每个状态都有 `33,538` 个 gene expression feature 和 `48` 个 guide capture feature。与 `GSE132080` 不同，本地目录中没有单独的每细胞 guide assignment metadata，guide 身份需要从矩阵中的 `CRISPR Guide Capture` feature 计数推断。

### 2. guide / target 内容

`GSE283614` 的 feature 文件中包含 `48` 个 CRISPR guide capture feature。按名称统计，这些 guide 主要分为：

- `16` 个 `OCT4` 相关 guide；
- `17` 个 `NANOG` 相关 guide；
- `7` 个 non-targeting control guide；
- `8` 个 SafeHarbor control guide。

ESC 和 DE 的 guide 名称表达略有差异。例如 ESC 中部分 guide 写作 `OCT4+1-38`、`NANOG+1-1`、`NT_Control_1`，而 DE 中对应名称可能写作 `OCT4_1-38`、`NANOG_1-1`、`NT_Control-1`。因此跨状态合并前需要统一 guide 名称格式，例如将 `+`、`_` 和 control 编号中的连字符差异标准化。

从 guide 名称可以看出，该数据集不是直接扰动 coding gene 本身，而是围绕 `OCT4` 和 `NANOG` 的 enhancer / promoter 相关区域构建扰动库；同时包含 non-targeting 和 SafeHarbor guide 作为对照。因此，建模时的 target 可以有两种口径：

- enhancer-level / guide-level：每个 guide 单独作为扰动条件；
- locus-level：将 `OCT4` 相关 guide 合并为一类，将 `NANOG` 相关 guide 合并为一类，并将 non-targeting / SafeHarbor 作为对照。

更推荐先保留 guide-level 标签，因为同一 enhancer 或 promoter 区域内不同 guide 可能造成不同强度或方向的转录响应。

### 3. GSE283614 预处理建议

由于 `GSE283614` 的 gene expression 和 guide capture 在同一个矩阵中，预处理时应先按 feature type 拆分矩阵：

1. 读取 `matrix.mtx.gz`、`barcodes.tsv.gz` 和 `features.tsv.gz`，构建 AnnData 对象。
2. 根据 feature 文件第三列 `feature_type` 区分 `Gene Expression` 和 `CRISPR Guide Capture`。
3. 将 `Gene Expression` 子矩阵作为表达矩阵，用于后续 QC、归一化、HVG 选择和模型训练。
4. 将 `CRISPR Guide Capture` 子矩阵用于每细胞 guide assignment：可根据每个细胞 guide UMI 的最大值、第二大值、总 guide UMI、最大 guide 占比等指标，判定 single-guide、multi-guide、no-guide 或 ambiguous 细胞。
5. 将 non-targeting control 和 SafeHarbor guide 统一标记为 `control` 或分别保留为两个 control 子类。
6. 对 ESC 和 DE 分别完成 guide assignment 和表达矩阵 QC；如果要跨状态训练，再合并两个状态，并在 `obs` 中保留 `cell_state` 字段。
7. 进行细胞级 QC：计算 `total_counts`、`n_genes_by_counts`、`pct_counts_mt`，去除低质量细胞和高线粒体比例细胞。
8. 进行基因级 QC：去除低表达基因、线粒体基因和核糖体基因。
9. 保存原始 count 到 `adata.layers["counts"]`。
10. 执行 total-count normalization 和 `log1p` 转换。
11. 选择高变基因，并强制保留 `POU5F1` 和 `NANOG` 等关键 readout gene；其中 `OCT4` 在表达矩阵中的标准基因名为 `POU5F1`。
12. 保存每个状态的 `.h5ad` 文件，以及合并后的 `.h5ad` 文件和 preprocessing summary。

需要特别注意的是，`GSE283614` 的本地文件目前不能像 `GSE132080` 那样直接读取现成的 `guide_identity` 列。若不先从 `CRISPR Guide Capture` feature 中推断 guide assignment，就不能准确得到每个 guide 对应的细胞数，也不能直接用于需要扰动标签的训练。

## 三、训练使用建议

对于 `GSE132080`，可以直接使用已生成的 `perturbation_label` 或原始 `guide_identity` 进行训练。如果希望保留 sgRNA-level 条件，使用：

```bash
--target-strategy raw
```

如果希望将多个 sgRNA 合并到 target gene 级别，使用：

```bash
--target-strategy first_token
```

对于 `GSE283614`，建议先新增一个专门的预处理脚本，完成 expression / guide capture 拆分、guide assignment、ESC/DE 状态标注、QC 和 HVG 选择。预处理完成后，训练输入应至少包含：

- `adata.X`：归一化并 `log1p` 后的 gene expression 矩阵；
- `adata.obs["guide_identity"]`：从 guide capture 推断出的 guide-level 标签；
- `adata.obs["perturbation_label"]`：清洗后的模型条件标签；
- `adata.obs["cell_state"]`：`ESC` 或 `DE`；
- `adata.layers["counts"]`：原始 gene expression count。

## 四、参考链接

- `GSE132080` GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132080
- `GSE283614` GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE283614
- `GSE283614` preprint: https://www.biorxiv.org/content/10.1101/2025.08.07.669196
