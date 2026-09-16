# sgRNA 综合分析表 — 使用说明

## 核心结果文件

| 文件 | 内容 |
|---|---|
| `results/comprehensive_sgrna_table.csv` | ⭐ 主结果表，179 行 × 66 列 |
| `results/final_summary_report.txt` | 完整性验证 + 相关性分析报告 |
| `results/ucsc_fetched_sequences.csv` | 从 UCSC 补齐的序列记录（含来源与基因组版本） |

## 覆盖情况

**177 个真实 guide**（130 GSE132080 + 49 GSE283614，另含 control/`*` 占位行）

| 字段 | 覆盖 |
|---|---|
| sgRNA sequence | 176/177 (99.4%) |
| target genomic sequence | 169/177 (95.5%) |
| PAM sequence | 169/177 (95.5%) |
| mismatch 注释 | 128/177 |
| model embedding / virtual scale | 128/177 |
| 实验 activity | 128/177 |

未解析的 8 行全部是 **GSE283614 non-targeting control**（`NT_Control_*`、`unassigned`），本身没有基因组靶点，属于正常空缺。

### GSE132080 — 完全匹配 ✅
- 128/128 参考 protospacer
- 128/128 PAM 3-mer
- **错配调用与已发表 Table S2 标注 103/103 = 100% 一致**
- 25 个完美匹配 guide（0 错配）+ 103 个单错配 guide

### GSE283614 — 完全匹配 ✅
- 33/33 targeting guide 全部解析出 hg38 靶序列 + PAM
- 41 个 guide 含坐标的均已补齐（其余为非靶向对照）

## 关键字段

### 序列
| 字段 | 含义 |
|---|---|
| `sgrna_sequence` | guide 自身的 20 bp 序列 |
| `target_genomic_sequence` | 参考 protospacer（基因组靶位点，20 bp，已按 guide 方向） |
| `pam_sequence` | 3 bp PAM |
| `reference_source` | 参考序列来源 |
| `pam_source` | PAM 来源：`TableS8_genome_window` / `UCSC_hg19` / `UCSC_hg38` |

### 错配与 PAM 距离
| 字段 | 含义 |
|---|---|
| `n_mismatches` | 错配数 |
| `mismatch_positions_from_5p` | 错配位置（0-based，从 guide 5′ 端数） |
| `mismatch_positions_from_pam` | 论文惯例负值，`-1` = 紧邻 PAM 的那一位 |
| `mismatch_distance_to_pam` | **距 PAM 的碱基数，0 = 紧邻 PAM** |
| `min_distance_to_pam` | 最近错配到 PAM 的距离 |
| `mismatch_ref_bases` / `mismatch_alt_bases` | 参考碱基 / 变异碱基 |
| `mismatch_validated_vs_table_s2` | 是否与已发表 Table S2 一致 |

三者换算关系（已验证）：

```
guide_index (from 5') = 20 - |mismatch_position_from_pam|
distance_to_pam        = 19 - guide_index = |position_from_pam| - 1
```

### 模型与实验
| 字段 | 含义 |
|---|---|
| `virtual_scale` | 模型学到的相对活性标度 |
| `embedding_norm` | sgRNA embedding L2 范数 |
| `mean_relative_gamma` | GSE132080 实验活性 |
| `relative_activity_mean` | GSE283614 实验活性 |

## 使用示例

```python
import pandas as pd
df = pd.read_csv('results/comprehensive_sgrna_table.csv')

# 某个家族的全部 guide（含错配位置与 PAM 距离）
aldoa = df[df.guide_identity.astype(str).str.startswith('ALDOA')]
print(aldoa[['guide_identity', 'sgrna_sequence', 'target_genomic_sequence',
             'pam_sequence', 'n_mismatches', 'mismatch_positions_from_pam',
             'min_distance_to_pam', 'virtual_scale', 'mean_relative_gamma']])

# 错配位置与实验活性的关系
import matplotlib.pyplot as plt
d = df[(df.dataset == 'GSE132080') & df.min_distance_to_pam.notna()]
plt.scatter(d.min_distance_to_pam, d.mean_relative_gamma, alpha=.6)
plt.xlabel('mismatch distance to PAM (bp)'); plt.ylabel('activity')
plt.savefig('mismatch_vs_activity.png', dpi=300)
```

## 关键发现

**模型学到有效的 sgRNA 强度表征**（GSE132080, n=95）
- `virtual_scale` vs 实验活性：r = +0.606, ρ = **+0.680**
- `embedding_norm` vs 实验活性：r = +0.436, ρ = +0.524

**错配位置效应符合 CRISPR 机制**（n=95）
- 错配距 PAM 越远，活性越高：r = +0.450, ρ = **+0.509**
- 即 PAM 近端（seed 区）错配危害更大
- 距 PAM 分布：0–4 bp 16.5%、5–9 bp 28.2%、10–14 bp 42.7%、15–19 bp 12.6%

**PAM 组成**：AGG 51、CGG 41、GGG 26、TGG 10

**GSE283614**：GC 含量与活性弱正相关 r = +0.364 (p = 0.037)

## 重新生成

```bash
python scripts/build_comprehensive_table.py      # 主表（离线，用已发表补充表）
python scripts/fetch_missing_sequences.py        # 补齐缺失序列（需联网，结果已缓存）
python scripts/build_comprehensive_table.py      # 重新合并
python scripts/analyze_final_table.py            # 验证报告
```

---

**数据版本**: v2.0（修正参考序列来源、补齐全部靶序列与 PAM）
**模型**: GSE132080 sgRNA-level 200 epoch
