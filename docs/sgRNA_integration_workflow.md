# sgRNA 综合表构建与序列坐标系方法学

本文档记录 `results/comprehensive_sgrna_table.csv` 的数据来源、序列坐标系约定，
以及为保证正确性所做的验证。

---

## 1. 数据来源

### GSE132080（Dixit et al. 2016, K562 CRISPRi Perturb-seq）
`datasets/GSE132080/41587_2019_387_MOESM3_ESM/`（原记录为 `diff-project` 下同名目录，
该仓库已删除，数据已复制到本仓库）

| 文件 | 用途 |
|---|---|
| `Table_S2_large-scale_library_sequences_revised.txt` | **参考 protospacer**（每家族 `_00` 行）+ 已发表错配位置 + CFD 分数，119,510 行 |
| `Table_S8_machine_learning_input.txt` | **PAM 来源**（26 bp 基因组窗口）+ 每 guide 实验 gamma，26,248 行 |
| `guide_virtual_scales.csv` | 模型导出的 embedding / virtual scale |

### GSE283614（ESC/DE 增强子–启动子筛选）
| 文件 | 用途 |
|---|---|
| `reproducibility/prepared/gse283614/gse283614_table_s1_activity.tsv` | guide 序列、hg38 坐标、活性 |
| `reproducibility/prepared/gse283614/gse283614_de_guide_assignments.csv` | guide 分配、靶基因 |

### UCSC hg38 / hg19 sequence API
补齐已发表补充表未覆盖的序列（结果缓存于 `results/.ucsc_cache/`）。

---

## 2. 序列坐标系约定（关键）

### 2.1 Table S8 `genome input` 窗口布局

经全量验证，`genome input` 与 `sgRNA input` 均为 **26 bp**，且**已经按 guide 自身方向书写**
（正链、负链 guide 都是直接匹配，无需反向互补）：

```
索引:  0   1  |  2 ................................ 21 | 22  23  24 | 25
内容: 上游侧翼  |        20 bp protospacer            |    PAM     | 额外 1 bp
```

- guide 在窗口中的偏移**恒为 2**（26,248/26,248 行）
- 窗口 `[22:25]` 在 **100%** 的行中以 `GG` 结尾 → 确认该位置即 PAM

### 2.2 ⚠️ 参考 protospacer 必须取自 Table S2 的 `_00` 行

**这是本项目最重要的一处修正。**

早期实现用 S8 窗口的 `[2:22]` 作为参考序列，导致：
- 大量 guide 在**第 0 位碱基**被误判为错配
- 全局额外错配全部集中在 `-20`（即 guide 第 0 位），共 28,008 例
- 与已发表错配标注仅 **48.4%** 一致

事实：S8 窗口 protospacer 的**第 0 位碱基对部分家族不可靠**。例如：

| 家族 | S8 窗口 `[2:22]` | Table S2 `_00` |
|---|---|---|
| `AARS2_+_44281027.23-P1P2` | `CAGTGGCAGCTGCAGCCCGG` | `GAGTGGCAGCTGCAGCCCGG` |
| `AATF_-_35306286.23-P1P2` | `AAGTGGCCGGTCCAGAGCTG` | `GAGTGGCCGGTCCAGAGCTG` |

二者**仅第 0 位不同**。以 Table S2 `_00` 为参考后：

> **117,010 / 117,010 个 guide 的错配调用与已发表 Table S2 完全一致（agreement = 1.000，0 例分歧）**

因此最终实现：
- **参考 protospacer（靶位点序列）← Table S2 `_00` 行**
- **PAM ← Table S8 窗口 `[22:25]`**
- S8 窗口的 protospacer 仅作为 `genome_window_protospacer` 保留，并用
  `window_pos0_discrepancy` 标记其与参考序列不一致的情况（GSE132080 中 61 例）

### 2.3 错配位置换算

论文列 `mismatch position` 使用负值，`-1` 表示紧邻 PAM 的碱基。设 `i` 为 guide
内 0-based 位置（自 5′ 端）：

```
position_from_pam = -(20 - i)
distance_to_pam   = 19 - i = |position_from_pam| - 1
```

实例验证（`ALDOA_+_30077139.23-P1P2`，参考 `GGTCACCAGGACCCCTTCTG`）：

| guide | 序列 | 错配 idx | position_from_pam | distance_to_pam |
|---|---|---|---|---|
| `_06` | `GGTCACCAGGATCCCTTCTG` | 11 | -9 | 8 |
| `_07` | `GGTCACCAGGCCCCCTTCTG` | 10 | -10 | 9 |
| `_13` | `GGTCACCAGGACCCCTTTTG` | 17 | -3 | 2 |
| `_14` | `GGTCACCAGGACCGCTTCTG` | 13 | -7 | 6 |

`distance_to_pam = 0` 表示错配就在 PAM 旁边。

### 2.4 ⚠️ 基因组版本

- **GSE132080 的 guide 标签坐标是 hg19**（2016 年论文）
- **GSE283614 是 hg38**

判定依据：`RAN_+_131356438` 在 hg19 chr12 命中（`hg19` 中 RAN 位于
chr12:131,356,125–131,361,620），在 hg38 chr1–12 全部未命中。
`RPS15_-_1438413` 在 hg19 / hg38 均命中且 PAM 一致（`AGG`），属该位点跨版本保守。

---

## 3. 残余缺口与补齐

| 缺口 | 处理 |
|---|---|
| RAN、RPS15 两家族不在 Table S8 中（10 个 guide 缺 PAM） | UCSC **hg19** 取序列定位，PAM 分别为 `TGG`、`AGG` |
| GSE283614 41 个 guide 无靶序列 | UCSC **hg38** 双向搜索定位，全部解析出 strand + 有效 PAM |
| GSE283614 `NT_Control_*` / `unassigned`（8 行） | 非靶向对照，本身无基因组靶点，保持空缺 |

定位方式不依赖链的假设：在坐标 ±80 bp 窗口内**双链搜索** protospacer，
命中后按所在链取 PAM。

---

## 4. 最终覆盖

```
rows = 179 (130 GSE132080 + 49 GSE283614)   columns = 66
real guides = 177

GSE132080 (n=128)          GSE283614 (n=49)
  参考 protospacer  128       靶序列 + PAM      41
  PAM 3-mer         128       targeting 解析    33/33
  错配注释          128       非靶向对照         8
  与已发表一致      103/103 (100%)
```

---

## 5. 验证结果

**GSE132080 错配调用**
- 与已发表 Table S2 标注一致率：**103/103 = 100.0%**
- 错配数分布：0 错配 25 个、1 错配 103 个
- 距 PAM 距离：mean 9.64 / median 10 / range 0–18
- 分布：0–4 bp 16.5%、5–9 bp 28.2%、10–14 bp 42.7%、15–19 bp 12.6%
- PAM 组成：AGG 51、CGG 41、GGG 26、TGG 10（全部以 GG 结尾 ✓）

**相关性（GSE132080, n=95）**

| 比较 | Pearson r | Spearman ρ |
|---|---|---|
| `virtual_scale` vs 实验活性 | +0.606 (p=7.4e-11) | **+0.680** (p=3.6e-14) |
| `embedding_norm` vs 实验活性 | +0.436 (p=1.0e-05) | +0.524 (p=5.1e-08) |
| 错配距 PAM vs 实验活性 | +0.450 (p=4.8e-06) | **+0.509** (p=1.4e-07) |
| GC 含量 vs 实验活性 | +0.062 (n.s.) | +0.052 (n.s.) |

**GSE283614（n=33）**：GC 含量 vs 活性 r = +0.364 (p=0.037), ρ = +0.400

---

## 6. 生物学解读

1. **模型学到了有效的 sgRNA 强度表征**：`virtual_scale` 与实验活性 ρ = 0.68。
2. **错配位置效应符合 CRISPR 机制**：错配距 PAM 越远活性越高（ρ = +0.51），
   即 PAM 近端 seed 区错配危害最大。模型隐式学到了这一位置依赖性。
3. **PAM 近端错配的模型标度确实更低**：如 `RAN_+_131356438.23-P1P2_12`
   错配距 PAM 仅 0 bp，`virtual_scale` = 0.612（该家族最低）。
4. 序列组成（GC）与活性无显著关系。

---

## 7. 脚本

| 脚本 | 作用 |
|---|---|
| `scripts/build_comprehensive_table.py` | **主脚本**：离线构建综合表（含 UCSC 结果合并） |
| `scripts/fetch_missing_sequences.py` | 从 UCSC 补齐缺失序列（带磁盘缓存） |
| `scripts/analyze_final_table.py` | 完整性与相关性验证报告 |
| `scripts/investigate_seq_alignment.py` | 序列坐标系约定的实证推导 |
| `scripts/diagnose_mismatch_calls.py` | 错配不一致问题定位 |
| `scripts/diagnose_reference_source.py` | 参考序列来源的 100% 一致性验证 |

---

## 8. 陷阱备忘

1. **不要用 S8 窗口做错配调用** — 第 0 位碱基不可靠；用 Table S2 `_00`。
2. **S8 窗口无需反向互补** — 已按 guide 方向书写。
3. **GSE132080 是 hg19，GSE283614 是 hg38**。
4. **guide 标签有重复基因名**：`ALDOA_ALDOA_+_...` 需归一化为 `ALDOA_+_...`
   才能与补充表匹配。
5. **补充表首列无表头**，读入后需重命名。
6. `distance_to_pam` 为 0 是合法的（紧邻 PAM），不要当缺失值处理。
