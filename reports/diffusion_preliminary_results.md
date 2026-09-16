# PerturbDiff 扩散模型初步结果（GSE132080 sgRNA 级）

本报告由 `scripts/export_diffusion_prelim.py` 的导出结果自动生成，
全部数值都是原始值（表达值、Z-sem latent、delta_g latent），不含 L2 等汇总指标。

## 1. 结果来源

| 项目 | 值 |
| --- | --- |
| checkpoint | `runs\gse132080_target_200ep\checkpoint_best.pt` |
| 数据 | `C:\Users\Admin\myprojects\diff-project\reproducibility\prepared\splits\gse132080_test.h5ad` |
| 扰动标签列 | `perturbation_label` |
| sgRNA 身份列 | `guide_identity` |
| 导出粒度 | `guide` |
| 对照细胞数 | 211 |
| 基因数 | 1019 |
| 行数 / 条件数 | 127 / 25 |
| latent 维度 | 128 |
| DDIM 步数 | 50 |
| 带 target 序列的行数 | 127 / 127 |
| 含错配的行数 | 103 / 127 |
| 单错配且与文库 PAM 距离一致的行数 | 103 / 103 |

模型：`z_condition = E(x_control) + delta_g`，`x_pred = DDIM(z_condition)`。

## 2. sgRNA 序列、target 序列与错配位置

`target_segment_20nt` 是该 sgRNA 设计要识别的 20 nt 目标序列，取自文库中同一 family 的 fully matched guide
（即 Table S10/S2 的 `_00` 行），因此每个 target 的多条变体共享同一段 target，可直接横向比较错配位置。
`alignment_target_vs_sgrna` 用 `|` 表示匹配、`X` 表示错配。
`mismatch_distance_to_pam` 从 PAM 端计数：紧邻 PAM 的碱基为 0，20 nt 中最远端为 19；
`mismatch_label_from_library` 是论文文库原始编号（-1 ~ -19），两者关系为 `距离 = |编号| - 1`。

校验：103 条单错配 guide 的计算距离与文库编号完全一致，
103 / 127 条带错配，其余为 fully matched guide。

`target_window_26nt` 与 `target_window_target_row` / `target_window_sgrna_row` 是论文 Table S8 的基因组窗口及对齐视图；
该窗口的 protospacer 相对 sgRNA 有一条固定的 2 nt 5' 偏移，只作为 PAM 侧上下文参考，不参与上面的错配判定。

| id | target | sgRNA identity | target 20 nt (5'->3') | sgRNA (5'->3') | alignment | 错配数 | 错配位置(1-based) | dist to PAM | 文库编号 | cells |
| ---: | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| 14 | ALDOA | ALDOA_ALDOA_+_30077139.23-P1P2_00 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGACCCCTTCTG` | `||||||||||||||||||||` | 0 | - | - | - | 8 |
| 14 | ALDOA | ALDOA_ALDOA_+_30077139.23-P1P2_06 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGATCCCTTCTG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 12 |
| 14 | ALDOA | ALDOA_ALDOA_+_30077139.23-P1P2_07 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGCCCCCTTCTG` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 10 |
| 14 | ALDOA | ALDOA_ALDOA_+_30077139.23-P1P2_13 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGACCCCTTTTG` | `|||||||||||||||||X||` | 1 | 18 | 2 | -3 | 12 |
| 14 | ALDOA | ALDOA_ALDOA_+_30077139.23-P1P2_14 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGACCGCTTCTG` | `|||||||||||||X||||||` | 1 | 14 | 6 | -7 | 11 |
| 8 | ATP5E | ATP5E_ATP5E_-_57607036.23-P1P2_00 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCAGGGGCACTCTGT` | `||||||||||||||||||||` | 0 | - | - | - | 14 |
| 8 | ATP5E | ATP5E_ATP5E_-_57607036.23-P1P2_01 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCTGGGGCACTCTGT` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 14 |
| 8 | ATP5E | ATP5E_ATP5E_-_57607036.23-P1P2_04 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCAGGAGCACTCTGT` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 20 |
| 8 | ATP5E | ATP5E_ATP5E_-_57607036.23-P1P2_14 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCAGGGGCACTGTGT` | `||||||||||||||||X|||` | 1 | 17 | 3 | -4 | 12 |
| 8 | ATP5E | ATP5E_ATP5E_-_57607036.23-P1P2_16 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCAGGGGCGCTCTGT` | `|||||||||||||X||||||` | 1 | 14 | 6 | -7 | 11 |
| 23 | BCR | BCR_BCR_+_23523092.23-P1P2_00 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGGGCCCGTCTCAG` | `||||||||||||||||||||` | 0 | - | - | - | 13 |
| 23 | BCR | BCR_BCR_+_23523092.23-P1P2_04 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGAGCCCGTCTCAG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 8 |
| 23 | BCR | BCR_BCR_+_23523092.23-P1P2_05 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGCGCCCGTCTCAG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 21 |
| 23 | BCR | BCR_BCR_+_23523092.23-P1P2_07 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGGGCTCGTCTCAG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 8 |
| 23 | BCR | BCR_BCR_+_23523092.23-P1P2_13 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGGGCCCATCTCAG` | `|||||||||||||X||||||` | 1 | 14 | 6 | -7 | 12 |
| 23 | BCR | BCR_BCR_+_23523092.23-P1P2_15 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGGGCCCGTCGCAG` | `||||||||||||||||X|||` | 1 | 17 | 3 | -4 | 11 |
| 7 | CAD | CAD_CAD_+_27440280.23-P1P2_00 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGAGAGAAGCCGGGCG` | `||||||||||||||||||||` | 0 | - | - | - | 8 |
| 7 | CAD | CAD_CAD_+_27440280.23-P1P2_03 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGTGAGAAGCCGGGCG` | `||||||X|||||||||||||` | 1 | 7 | 13 | -14 | 10 |
| 7 | CAD | CAD_CAD_+_27440280.23-P1P2_06 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGAGAGTAGCCGGGCG` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 12 |
| 7 | CAD | CAD_CAD_+_27440280.23-P1P2_07 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGAGCGAAGCCGGGCG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 28 |
| 7 | CAD | CAD_CAD_+_27440280.23-P1P2_13 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGAGAGAAGCCTGGCG` | `|||||||||||||||X||||` | 1 | 16 | 4 | -5 | 16 |
| 12 | CDC23 | CDC23_CDC23_-_137548987.23-P1P2_00 | `GACAGCCACCGGGACCATGG` | `GACAGCCACCGGGACCATGG` | `||||||||||||||||||||` | 0 | - | - | - | 9 |
| 12 | CDC23 | CDC23_CDC23_-_137548987.23-P1P2_02 | `GACAGCCACCGGGACCATGG` | `GACAGCTACCGGGACCATGG` | `||||||X|||||||||||||` | 1 | 7 | 13 | -14 | 7 |
| 12 | CDC23 | CDC23_CDC23_-_137548987.23-P1P2_04 | `GACAGCCACCGGGACCATGG` | `GACAGCCAACGGGACCATGG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 12 |
| 12 | CDC23 | CDC23_CDC23_-_137548987.23-P1P2_08 | `GACAGCCACCGGGACCATGG` | `GACAGCCATCGGGACCATGG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 17 |
| 12 | CDC23 | CDC23_CDC23_-_137548987.23-P1P2_11 | `GACAGCCACCGGGACCATGG` | `GACAGCCACCGGGACCACGG` | `|||||||||||||||||X||` | 1 | 18 | 2 | -3 | 24 |
| 1 | COX11 | COX11_COX11_+_53045977.23-P1P2_00 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGGCGTCCTGGATGG` | `||||||||||||||||||||` | 0 | - | - | - | 19 |
| 1 | COX11 | COX11_COX11_+_53045977.23-P1P2_03 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGTCGTCCTGGATGG` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 30 |
| 1 | COX11 | COX11_COX11_+_53045977.23-P1P2_04 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGGCGCCCTGGATGG` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 12 |
| 1 | COX11 | COX11_COX11_+_53045977.23-P1P2_05 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGGCGTCTTGGATGG` | `||||||||||||X|||||||` | 1 | 13 | 7 | -8 | 11 |
| 1 | COX11 | COX11_COX11_+_53045977.23-P1P2_10 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGGCGTCCCGGATGG` | `|||||||||||||X||||||` | 1 | 14 | 6 | -7 | 4 |
| 17 | DBR1 | DBR1_DBR1_+_137893744.23-P1P2_00 | `GTTTGCAGGAGTCTACACCC` | `GTTTGCAGGAGTCTACACCC` | `||||||||||||||||||||` | 0 | - | - | - | 30 |
| 17 | DBR1 | DBR1_DBR1_+_137893744.23-P1P2_01 | `GTTTGCAGGAGTCTACACCC` | `GATTGCAGGAGTCTACACCC` | `|X||||||||||||||||||` | 1 | 2 | 18 | -19 | 21 |
| 17 | DBR1 | DBR1_DBR1_+_137893744.23-P1P2_05 | `GTTTGCAGGAGTCTACACCC` | `GTTTGCAGGAGTGTACACCC` | `||||||||||||X|||||||` | 1 | 13 | 7 | -8 | 9 |
| 17 | DBR1 | DBR1_DBR1_+_137893744.23-P1P2_07 | `GTTTGCAGGAGTCTACACCC` | `GTTTGCAGGGGTCTACACCC` | `|||||||||X||||||||||` | 1 | 10 | 10 | -11 | 18 |
| 17 | DBR1 | DBR1_DBR1_+_137893744.23-P1P2_08 | `GTTTGCAGGAGTCTACACCC` | `GTTTGCAGTAGTCTACACCC` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 14 |
| 5 | DUT | DUT_DUT_+_48624411.23-P1P2_00 | `GCGAGCGAGGAGACCACCGG` | `GCGAGCGAGGAGACCACCGG` | `||||||||||||||||||||` | 0 | - | - | - | 22 |
| 5 | DUT | DUT_DUT_+_48624411.23-P1P2_01 | `GCGAGCGAGGAGACCACCGG` | `GCCAGCGAGGAGACCACCGG` | `||X|||||||||||||||||` | 1 | 3 | 17 | -18 | 32 |
| 5 | DUT | DUT_DUT_+_48624411.23-P1P2_07 | `GCGAGCGAGGAGACCACCGG` | `GCGAGCGAGGAGCCCACCGG` | `||||||||||||X|||||||` | 1 | 13 | 7 | -8 | 10 |
| 5 | DUT | DUT_DUT_+_48624411.23-P1P2_08 | `GCGAGCGAGGAGACCACCGG` | `GCGAGCGAGGAGGCCACCGG` | `||||||||||||X|||||||` | 1 | 13 | 7 | -8 | 30 |
| 5 | DUT | DUT_DUT_+_48624411.23-P1P2_10 | `GCGAGCGAGGAGACCACCGG` | `GCGAGCGAGGAGACCAACGG` | `||||||||||||||||X|||` | 1 | 17 | 3 | -4 | 10 |
| 22 | EIF2S1 | EIF2S1_EIF2S1_-_67827080.23-P1P2_00 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGAAGCGCACGCTGAGG` | `||||||||||||||||||||` | 0 | - | - | - | 11 |
| 22 | EIF2S1 | EIF2S1_EIF2S1_-_67827080.23-P1P2_01 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGAAACGCACGCTGAGG` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 9 |
| 22 | EIF2S1 | EIF2S1_EIF2S1_-_67827080.23-P1P2_02 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGCAGCGCACGCTGAGG` | `|||||X||||||||||||||` | 1 | 6 | 14 | -15 | 14 |
| 22 | EIF2S1 | EIF2S1_EIF2S1_-_67827080.23-P1P2_06 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGAAGCGCGCGCTGAGG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 12 |
| 22 | EIF2S1 | EIF2S1_EIF2S1_-_67827080.23-P1P2_07 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGAAGCGCTCGCTGAGG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 6 |
| 2 | GATA1 | GATA1_GATA1_-_48645022.23-P1P2_00 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTGCCACATCCCCA` | `||||||||||||||||||||` | 0 | - | - | - | 3 |
| 2 | GATA1 | GATA1_GATA1_-_48645022.23-P1P2_03 | `GTGAGCTTGCCACATCCCCA` | `GTGCGCTTGCCACATCCCCA` | `|||X||||||||||||||||` | 1 | 4 | 16 | -17 | 11 |
| 2 | GATA1 | GATA1_GATA1_-_48645022.23-P1P2_04 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTACCACATCCCCA` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 6 |
| 2 | GATA1 | GATA1_GATA1_-_48645022.23-P1P2_06 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTGCGACATCCCCA` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 10 |
| 2 | GATA1 | GATA1_GATA1_-_48645022.23-P1P2_08 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTTCCACATCCCCA` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 5 |
| 2 | GATA1 | GATA1_GATA1_-_48645022.23-P1P2_12 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTGCCACATCCGCA` | `|||||||||||||||||X||` | 1 | 18 | 2 | -3 | 8 |
| 3 | GINS1 | GINS1_GINS1_-_25388381.23-P1P2_00 | `GGACTAGAACGAAAGGAGTG` | `GGACTAGAACGAAAGGAGTG` | `||||||||||||||||||||` | 0 | - | - | - | 3 |
| 3 | GINS1 | GINS1_GINS1_-_25388381.23-P1P2_03 | `GGACTAGAACGAAAGGAGTG` | `GGACTATAACGAAAGGAGTG` | `||||||X|||||||||||||` | 1 | 7 | 13 | -14 | 15 |
| 3 | GINS1 | GINS1_GINS1_-_25388381.23-P1P2_06 | `GGACTAGAACGAAAGGAGTG` | `GGACTAGAACGGAAGGAGTG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 16 |
| 3 | GINS1 | GINS1_GINS1_-_25388381.23-P1P2_08 | `GGACTAGAACGAAAGGAGTG` | `GGACTAGAGCGAAAGGAGTG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 25 |
| 3 | GINS1 | GINS1_GINS1_-_25388381.23-P1P2_14 | `GGACTAGAACGAAAGGAGTG` | `GGACTAGAACGAAAGGAGCG` | `||||||||||||||||||X|` | 1 | 19 | 1 | -2 | 16 |
| 9 | GNB2L1 | GNB2L1_GNB2L1_+_180670873.23-P1P2_02 | `GTGCAAGGCGGCGGCAGGAG` | `GTGCAAGACGGCGGCAGGAG` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 8 |
| 9 | GNB2L1 | GNB2L1_GNB2L1_+_180670873.23-P1P2_07 | `GTGCAAGGCGGCGGCAGGAG` | `GTGCAAGGCGGGGGCAGGAG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 12 |
| 9 | GNB2L1 | GNB2L1_GNB2L1_+_180670873.23-P1P2_08 | `GTGCAAGGCGGCGGCAGGAG` | `GTGCAAGGTGGCGGCAGGAG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 58 |
| 9 | GNB2L1 | GNB2L1_GNB2L1_+_180670873.23-P1P2_13 | `GTGCAAGGCGGCGGCAGGAG` | `GTGCAAGGCGGCGGCGGGAG` | `|||||||||||||||X||||` | 1 | 16 | 4 | -5 | 9 |
| 19 | HSPA5 | HSPA5_HSPA5_+_128003624.23-P1P2_00 | `GAGCCGAGTAGGCGACGGTG` | `GAGCCGAGTAGGCGACGGTG` | `||||||||||||||||||||` | 0 | - | - | - | 6 |
| 19 | HSPA5 | HSPA5_HSPA5_+_128003624.23-P1P2_01 | `GAGCCGAGTAGGCGACGGTG` | `GAACCGAGTAGGCGACGGTG` | `||X|||||||||||||||||` | 1 | 3 | 17 | -18 | 10 |
| 19 | HSPA5 | HSPA5_HSPA5_+_128003624.23-P1P2_04 | `GAGCCGAGTAGGCGACGGTG` | `GAGCCGAGAAGGCGACGGTG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 10 |
| 19 | HSPA5 | HSPA5_HSPA5_+_128003624.23-P1P2_06 | `GAGCCGAGTAGGCGACGGTG` | `GAGCCGAGTAGACGACGGTG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 22 |
| 19 | HSPA5 | HSPA5_HSPA5_+_128003624.23-P1P2_08 | `GAGCCGAGTAGGCGACGGTG` | `GAGCCGAGTGGGCGACGGTG` | `|||||||||X||||||||||` | 1 | 10 | 10 | -11 | 16 |
| 25 | HSPA9 | HSPA9_HSPA9_-_137911079.23-P1P2_00 | `GGAGCTGCGCGATGCGGTGG` | `GGAGCTGCGCGATGCGGTGG` | `||||||||||||||||||||` | 0 | - | - | - | 9 |
| 25 | HSPA9 | HSPA9_HSPA9_-_137911079.23-P1P2_02 | `GGAGCTGCGCGATGCGGTGG` | `GGAGTTGCGCGATGCGGTGG` | `||||X|||||||||||||||` | 1 | 5 | 15 | -16 | 10 |
| 25 | HSPA9 | HSPA9_HSPA9_-_137911079.23-P1P2_04 | `GGAGCTGCGCGATGCGGTGG` | `GGAGCTGCGCAATGCGGTGG` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 13 |
| 25 | HSPA9 | HSPA9_HSPA9_-_137911079.23-P1P2_07 | `GGAGCTGCGCGATGCGGTGG` | `GGAGCTGCGGGATGCGGTGG` | `|||||||||X||||||||||` | 1 | 10 | 10 | -11 | 8 |
| 25 | HSPA9 | HSPA9_HSPA9_-_137911079.23-P1P2_08 | `GGAGCTGCGCGATGCGGTGG` | `GGAGCTGCTCGATGCGGTGG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 5 |
| 20 | HSPE1 | HSPE1_HSPE1_+_198365089.23-P1P2_00 | `GGAGACTCGCAGTCCGGCCC` | `GGAGACTCGCAGTCCGGCCC` | `||||||||||||||||||||` | 0 | - | - | - | 19 |
| 20 | HSPE1 | HSPE1_HSPE1_+_198365089.23-P1P2_01 | `GGAGACTCGCAGTCCGGCCC` | `GGAGACACGCAGTCCGGCCC` | `||||||X|||||||||||||` | 1 | 7 | 13 | -14 | 22 |
| 20 | HSPE1 | HSPE1_HSPE1_+_198365089.23-P1P2_02 | `GGAGACTCGCAGTCCGGCCC` | `GGAGACTGGCAGTCCGGCCC` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 17 |
| 20 | HSPE1 | HSPE1_HSPE1_+_198365089.23-P1P2_03 | `GGAGACTCGCAGTCCGGCCC` | `GGTGACTCGCAGTCCGGCCC` | `||X|||||||||||||||||` | 1 | 3 | 17 | -18 | 15 |
| 20 | HSPE1 | HSPE1_HSPE1_+_198365089.23-P1P2_14 | `GGAGACTCGCAGTCCGGCCC` | `GGAGACTCGCAGTCCTGCCC` | `|||||||||||||||X||||` | 1 | 16 | 4 | -5 | 11 |
| 21 | MTOR | MTOR_MTOR_+_11322547.23-P1P2_00 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGCCTGAAGCGG` | `||||||||||||||||||||` | 0 | - | - | - | 39 |
| 21 | MTOR | MTOR_MTOR_+_11322547.23-P1P2_05 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGCTTGAAGCGG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 16 |
| 21 | MTOR | MTOR_MTOR_+_11322547.23-P1P2_06 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGGCTGAAGCGG` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 11 |
| 21 | MTOR | MTOR_MTOR_+_11322547.23-P1P2_07 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGTCTGAAGCGG` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 6 |
| 21 | MTOR | MTOR_MTOR_+_11322547.23-P1P2_10 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGCCTGAAGCAG` | `||||||||||||||||||X|` | 1 | 19 | 1 | -2 | 8 |
| 4 | POLR1D | POLR1D_POLR1D_+_28196016.23-P1_00 | `GGGAAGCAAGGACCGACCGA` | `GGGAAGCAAGGACCGACCGA` | `||||||||||||||||||||` | 0 | - | - | - | 21 |
| 4 | POLR1D | POLR1D_POLR1D_+_28196016.23-P1_01 | `GGGAAGCAAGGACCGACCGA` | `GGGAAGCCAGGACCGACCGA` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 24 |
| 4 | POLR1D | POLR1D_POLR1D_+_28196016.23-P1_03 | `GGGAAGCAAGGACCGACCGA` | `GGTAAGCAAGGACCGACCGA` | `||X|||||||||||||||||` | 1 | 3 | 17 | -18 | 13 |
| 4 | POLR1D | POLR1D_POLR1D_+_28196016.23-P1_07 | `GGGAAGCAAGGACCGACCGA` | `GGGAAGCAAGGAGCGACCGA` | `||||||||||||X|||||||` | 1 | 13 | 7 | -8 | 33 |
| 4 | POLR1D | POLR1D_POLR1D_+_28196016.23-P1_08 | `GGGAAGCAAGGACCGACCGA` | `GGGAAGCAGGGACCGACCGA` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 26 |
| 13 | POLR2H | POLR2H_POLR2H_+_184081251.23-P1P2_00 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGAGAGCAGCAGA` | `||||||||||||||||||||` | 0 | - | - | - | 6 |
| 13 | POLR2H | POLR2H_POLR2H_+_184081251.23-P1P2_07 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGAGTGCAGCAGA` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 9 |
| 13 | POLR2H | POLR2H_POLR2H_+_184081251.23-P1P2_08 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGCGAGCAGCAGA` | `|||||||||X||||||||||` | 1 | 10 | 10 | -11 | 6 |
| 13 | POLR2H | POLR2H_POLR2H_+_184081251.23-P1P2_11 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGAGAGCAGCGGA` | `|||||||||||||||||X||` | 1 | 18 | 2 | -3 | 7 |
| 13 | POLR2H | POLR2H_POLR2H_+_184081251.23-P1P2_12 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGAGAGCAGGAGA` | `||||||||||||||||X|||` | 1 | 17 | 3 | -4 | 12 |
| 11 | RAN | RAN_RAN_+_131356438.23-P1P2_00 | `GGCGGTCGCTGCGCTTAGGG` | `GGCGGTCGCTGCGCTTAGGG` | `||||||||||||||||||||` | 0 | - | - | - | 11 |
| 11 | RAN | RAN_RAN_+_131356438.23-P1P2_02 | `GGCGGTCGCTGCGCTTAGGG` | `GGCGGCCGCTGCGCTTAGGG` | `|||||X||||||||||||||` | 1 | 6 | 14 | -15 | 24 |
| 11 | RAN | RAN_RAN_+_131356438.23-P1P2_03 | `GGCGGTCGCTGCGCTTAGGG` | `GGGGGTCGCTGCGCTTAGGG` | `||X|||||||||||||||||` | 1 | 3 | 17 | -18 | 12 |
| 11 | RAN | RAN_RAN_+_131356438.23-P1P2_04 | `GGCGGTCGCTGCGCTTAGGG` | `GGCGGTCGCGGCGCTTAGGG` | `|||||||||X||||||||||` | 1 | 10 | 10 | -11 | 14 |
| 11 | RAN | RAN_RAN_+_131356438.23-P1P2_12 | `GGCGGTCGCTGCGCTTAGGG` | `GGCGGTCGCTGCGCTTAGGT` | `|||||||||||||||||||X` | 1 | 20 | 0 | -1 | 14 |
| 18 | RPL9 | RPL9_RPL9_+_39460483.23-P1P2_00 | `GGATGTTTCTGTGCTCGTGG` | `GGATGTTTCTGTGCTCGTGG` | `||||||||||||||||||||` | 0 | - | - | - | 12 |
| 18 | RPL9 | RPL9_RPL9_+_39460483.23-P1P2_01 | `GGATGTTTCTGTGCTCGTGG` | `GGATGATTCTGTGCTCGTGG` | `|||||X||||||||||||||` | 1 | 6 | 14 | -15 | 9 |
| 18 | RPL9 | RPL9_RPL9_+_39460483.23-P1P2_04 | `GGATGTTTCTGTGCTCGTGG` | `GGATGTTTCAGTGCTCGTGG` | `|||||||||X||||||||||` | 1 | 10 | 10 | -11 | 7 |
| 18 | RPL9 | RPL9_RPL9_+_39460483.23-P1P2_05 | `GGATGTTTCTGTGCTCGTGG` | `GGATGTTTCGGTGCTCGTGG` | `|||||||||X||||||||||` | 1 | 10 | 10 | -11 | 8 |
| 18 | RPL9 | RPL9_RPL9_+_39460483.23-P1P2_07 | `GGATGTTTCTGTGCTCGTGG` | `GGATGTTTCTGCGCTCGTGG` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 17 |
| 10 | RPS14 | RPS14_RPS14_+_149829238.23-P1P2_00 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCGGGCGCGACAATC` | `||||||||||||||||||||` | 0 | - | - | - | 14 |
| 10 | RPS14 | RPS14_RPS14_+_149829238.23-P1P2_01 | `GAGGCCCGGGCGCGACAATC` | `GAGACCCGGGCGCGACAATC` | `|||X||||||||||||||||` | 1 | 4 | 16 | -17 | 15 |
| 10 | RPS14 | RPS14_RPS14_+_149829238.23-P1P2_02 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCTGGCGCGACAATC` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 13 |
| 10 | RPS14 | RPS14_RPS14_+_149829238.23-P1P2_04 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCGCGCGCGACAATC` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 22 |
| 10 | RPS14 | RPS14_RPS14_+_149829238.23-P1P2_08 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCGGGCTCGACAATC` | `|||||||||||X||||||||` | 1 | 12 | 8 | -9 | 9 |
| 10 | RPS14 | RPS14_RPS14_+_149829238.23-P1P2_13 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCGGGCGCGACAGTC` | `|||||||||||||||||X||` | 1 | 18 | 2 | -3 | 8 |
| 24 | RPS15 | RPS15_RPS15_-_1438413.23-P1P2_00 | `GACCAAAGCGATCTCTTCTG` | `GACCAAAGCGATCTCTTCTG` | `||||||||||||||||||||` | 0 | - | - | - | 9 |
| 24 | RPS15 | RPS15_RPS15_-_1438413.23-P1P2_01 | `GACCAAAGCGATCTCTTCTG` | `GACCAAACCGATCTCTTCTG` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 23 |
| 24 | RPS15 | RPS15_RPS15_-_1438413.23-P1P2_02 | `GACCAAAGCGATCTCTTCTG` | `GACCAAGGCGATCTCTTCTG` | `||||||X|||||||||||||` | 1 | 7 | 13 | -14 | 9 |
| 24 | RPS15 | RPS15_RPS15_-_1438413.23-P1P2_07 | `GACCAAAGCGATCTCTTCTG` | `GACCAAAGCGGTCTCTTCTG` | `||||||||||X|||||||||` | 1 | 11 | 9 | -10 | 9 |
| 24 | RPS15 | RPS15_RPS15_-_1438413.23-P1P2_12 | `GACCAAAGCGATCTCTTCTG` | `GACCAAAGCGATCTCTTGTG` | `|||||||||||||||||X||` | 1 | 18 | 2 | -3 | 8 |
| 15 | RPS18 | RPS18_RPS18_+_33239917.23-P1P2_00 | `GCTGCGATGCCGCTGGATCA` | `GCTGCGATGCCGCTGGATCA` | `||||||||||||||||||||` | 0 | - | - | - | 4 |
| 15 | RPS18 | RPS18_RPS18_+_33239917.23-P1P2_01 | `GCTGCGATGCCGCTGGATCA` | `GCTGCAATGCCGCTGGATCA` | `|||||X||||||||||||||` | 1 | 6 | 14 | -15 | 8 |
| 15 | RPS18 | RPS18_RPS18_+_33239917.23-P1P2_02 | `GCTGCGATGCCGCTGGATCA` | `GCTGGGATGCCGCTGGATCA` | `||||X|||||||||||||||` | 1 | 5 | 15 | -16 | 5 |
| 15 | RPS18 | RPS18_RPS18_+_33239917.23-P1P2_04 | `GCTGCGATGCCGCTGGATCA` | `GCTGCGATCCCGCTGGATCA` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 13 |
| 15 | RPS18 | RPS18_RPS18_+_33239917.23-P1P2_08 | `GCTGCGATGCCGCTGGATCA` | `GCTGCGATTCCGCTGGATCA` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 23 |
| 16 | SEC61A1 | SEC61A1_SEC61A1_-_127771295.23-P1_00 | `GGCACTGACGTGTCTCTCGG` | `GGCACTGACGTGTCTCTCGG` | `||||||||||||||||||||` | 0 | - | - | - | 20 |
| 16 | SEC61A1 | SEC61A1_SEC61A1_-_127771295.23-P1_01 | `GGCACTGACGTGTCTCTCGG` | `GGCACTGTCGTGTCTCTCGG` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 7 |
| 16 | SEC61A1 | SEC61A1_SEC61A1_-_127771295.23-P1_02 | `GGCACTGACGTGTCTCTCGG` | `GGCGCTGACGTGTCTCTCGG` | `|||X||||||||||||||||` | 1 | 4 | 16 | -17 | 19 |
| 16 | SEC61A1 | SEC61A1_SEC61A1_-_127771295.23-P1_03 | `GGCACTGACGTGTCTCTCGG` | `GGTACTGACGTGTCTCTCGG` | `||X|||||||||||||||||` | 1 | 3 | 17 | -18 | 16 |
| 16 | SEC61A1 | SEC61A1_SEC61A1_-_127771295.23-P1_04 | `GGCACTGACGTGTCTCTCGG` | `GGCACTGAAGTGTCTCTCGG` | `||||||||X|||||||||||` | 1 | 9 | 11 | -12 | 10 |
| 6 | TUBB | TUBB_TUBB_+_30688126.23-P1_00 | `GCGGCAGGAAGGTTCTGAGA` | `GCGGCAGGAAGGTTCTGAGA` | `||||||||||||||||||||` | 0 | - | - | - | 16 |
| 6 | TUBB | TUBB_TUBB_+_30688126.23-P1_01 | `GCGGCAGGAAGGTTCTGAGA` | `GCAGCAGGAAGGTTCTGAGA` | `||X|||||||||||||||||` | 1 | 3 | 17 | -18 | 7 |
| 6 | TUBB | TUBB_TUBB_+_30688126.23-P1_03 | `GCGGCAGGAAGGTTCTGAGA` | `GCGGCAGCAAGGTTCTGAGA` | `|||||||X||||||||||||` | 1 | 8 | 12 | -13 | 5 |
| 6 | TUBB | TUBB_TUBB_+_30688126.23-P1_06 | `GCGGCAGGAAGGTTCTGAGA` | `GCGGCAGGACGGTTCTGAGA` | `|||||||||X||||||||||` | 1 | 10 | 10 | -11 | 9 |
| 6 | TUBB | TUBB_TUBB_+_30688126.23-P1_10 | `GCGGCAGGAAGGTTCTGAGA` | `GCGGCAGGAAGGTTCAGAGA` | `|||||||||||||||X||||` | 1 | 16 | 4 | -5 | 18 |

### 按 target gene 分组对比

同一 target gene 下的多条 sgRNA 共享同一段 target 20 nt，下表把每条 sgRNA 与 target 的差异并排显示。

**ALDOA**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| ALDOA_ALDOA_+_30077139.23-P1P2_00 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGACCCCTTCTG` | `||||||||||||||||||||` | - | - | - |
| ALDOA_ALDOA_+_30077139.23-P1P2_06 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGATCCCTTCTG` | `|||||||||||X||||||||` | 8 | -9 | 0.8283 |
| ALDOA_ALDOA_+_30077139.23-P1P2_07 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGCCCCCTTCTG` | `||||||||||X|||||||||` | 9 | -10 | 0.6401 |
| ALDOA_ALDOA_+_30077139.23-P1P2_13 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGACCCCTTTTG` | `|||||||||||||||||X||` | 2 | -3 | 0.1409 |
| ALDOA_ALDOA_+_30077139.23-P1P2_14 | `GGTCACCAGGACCCCTTCTG` | `GGTCACCAGGACCGCTTCTG` | `|||||||||||||X||||||` | 6 | -7 | 0.2701 |

基因组窗口视图（示例）：

```text
window      CTCGTCACCAGGACCCCTTCTGCGGC
target      ..GGTCACCAGGACCCCTTCTG....
sgRNA       ..GGTCACCAGGATCCCTTCTG....
```

**ATP5E**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| ATP5E_ATP5E_-_57607036.23-P1P2_00 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCAGGGGCACTCTGT` | `||||||||||||||||||||` | - | - | - |
| ATP5E_ATP5E_-_57607036.23-P1P2_01 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCTGGGGCACTCTGT` | `|||||||X||||||||||||` | 12 | -13 | 0.8194 |
| ATP5E_ATP5E_-_57607036.23-P1P2_04 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCAGGAGCACTCTGT` | `||||||||||X|||||||||` | 9 | -10 | 0.4505 |
| ATP5E_ATP5E_-_57607036.23-P1P2_14 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCAGGGGCACTGTGT` | `||||||||||||||||X|||` | 3 | -4 | 0.1606 |
| ATP5E_ATP5E_-_57607036.23-P1P2_16 | `GGTGTCCAGGGGCACTCTGT` | `GGTGTCCAGGGGCGCTCTGT` | `|||||||||||||X||||||` | 6 | -7 | 0.6788 |

基因组窗口视图（示例）：

```text
window      AAGGTGTCCAGGGGCACTCTGTGGGT
target      GGTGTCCAGGGGCACTCTGT......
sgRNA       GGTGTCCTGGGGCACTCTGT......
```

**BCR**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| BCR_BCR_+_23523092.23-P1P2_00 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGGGCCCGTCTCAG` | `||||||||||||||||||||` | - | - | - |
| BCR_BCR_+_23523092.23-P1P2_04 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGAGCCCGTCTCAG` | `||||||||X|||||||||||` | 11 | -12 | 0.7508 |
| BCR_BCR_+_23523092.23-P1P2_05 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGCGCCCGTCTCAG` | `||||||||X|||||||||||` | 11 | -12 | 0.5088 |
| BCR_BCR_+_23523092.23-P1P2_07 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGGGCTCGTCTCAG` | `|||||||||||X||||||||` | 8 | -9 | 2.0979 |
| BCR_BCR_+_23523092.23-P1P2_13 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGGGCCCATCTCAG` | `|||||||||||||X||||||` | 6 | -7 | 0.1661 |
| BCR_BCR_+_23523092.23-P1P2_15 | `GCGCGCGGGGCCCGTCTCAG` | `GCGCGCGGGGCCCGTCGCAG` | `||||||||||||||||X|||` | 3 | -4 | 0.3345 |

基因组窗口视图（示例）：

```text
window      CTGCGCGCGGGGCCCGTCTCAGCGGC
target      GCGCGCGGGGCCCGTCTCAG......
sgRNA       GCGCGCGGAGCCCGTCTCAG......
```

**CAD**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| CAD_CAD_+_27440280.23-P1P2_00 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGAGAGAAGCCGGGCG` | `||||||||||||||||||||` | - | - | - |
| CAD_CAD_+_27440280.23-P1P2_03 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGTGAGAAGCCGGGCG` | `||||||X|||||||||||||` | 13 | -14 | 1.1189 |
| CAD_CAD_+_27440280.23-P1P2_06 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGAGAGTAGCCGGGCG` | `||||||||||X|||||||||` | 9 | -10 | 0.2255 |
| CAD_CAD_+_27440280.23-P1P2_07 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGAGCGAAGCCGGGCG` | `||||||||X|||||||||||` | 11 | -12 | 0.5050 |
| CAD_CAD_+_27440280.23-P1P2_13 | `GGCTGGAGAGAAGCCGGGCG` | `GGCTGGAGAGAAGCCTGGCG` | `|||||||||||||||X||||` | 4 | -5 | 0.1001 |

基因组窗口视图（示例）：

```text
window      GGCGCTGGAGAGAAGCCGGGCGCGGC
target      ..GGCTGGAGAGAAGCCGGGCG....
sgRNA       ..GGCTGGTGAGAAGCCGGGCG....
```

**CDC23**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| CDC23_CDC23_-_137548987.23-P1P2_00 | `GACAGCCACCGGGACCATGG` | `GACAGCCACCGGGACCATGG` | `||||||||||||||||||||` | - | - | - |
| CDC23_CDC23_-_137548987.23-P1P2_02 | `GACAGCCACCGGGACCATGG` | `GACAGCTACCGGGACCATGG` | `||||||X|||||||||||||` | 13 | -14 | 0.9317 |
| CDC23_CDC23_-_137548987.23-P1P2_04 | `GACAGCCACCGGGACCATGG` | `GACAGCCAACGGGACCATGG` | `||||||||X|||||||||||` | 11 | -12 | 0.7840 |
| CDC23_CDC23_-_137548987.23-P1P2_08 | `GACAGCCACCGGGACCATGG` | `GACAGCCATCGGGACCATGG` | `||||||||X|||||||||||` | 11 | -12 | 0.8096 |
| CDC23_CDC23_-_137548987.23-P1P2_11 | `GACAGCCACCGGGACCATGG` | `GACAGCCACCGGGACCACGG` | `|||||||||||||||||X||` | 2 | -3 | 0.4707 |

基因组窗口视图（示例）：

```text
window      GTCACAGCCACCGGGACCATGGAGGT
target      ..GACAGCCACCGGGACCATGG....
sgRNA       ..GACAGCTACCGGGACCATGG....
```

**COX11**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| COX11_COX11_+_53045977.23-P1P2_00 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGGCGTCCTGGATGG` | `||||||||||||||||||||` | - | - | - |
| COX11_COX11_+_53045977.23-P1P2_03 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGTCGTCCTGGATGG` | `|||||||X||||||||||||` | 12 | -13 | 0.8949 |
| COX11_COX11_+_53045977.23-P1P2_04 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGGCGCCCTGGATGG` | `||||||||||X|||||||||` | 9 | -10 | 0.7976 |
| COX11_COX11_+_53045977.23-P1P2_05 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGGCGTCTTGGATGG` | `||||||||||||X|||||||` | 7 | -8 | 0.3201 |
| COX11_COX11_+_53045977.23-P1P2_10 | `GGCTCTGGCGTCCTGGATGG` | `GGCTCTGGCGTCCCGGATGG` | `|||||||||||||X||||||` | 6 | -7 | 0.2203 |

基因组窗口视图（示例）：

```text
window      AGGGCTCTGGCGTCCTGGATGGAGGT
target      GGCTCTGGCGTCCTGGATGG......
sgRNA       GGCTCTGTCGTCCTGGATGG......
```

**DBR1**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| DBR1_DBR1_+_137893744.23-P1P2_00 | `GTTTGCAGGAGTCTACACCC` | `GTTTGCAGGAGTCTACACCC` | `||||||||||||||||||||` | - | - | - |
| DBR1_DBR1_+_137893744.23-P1P2_01 | `GTTTGCAGGAGTCTACACCC` | `GATTGCAGGAGTCTACACCC` | `|X||||||||||||||||||` | 18 | -19 | 1.1673 |
| DBR1_DBR1_+_137893744.23-P1P2_05 | `GTTTGCAGGAGTCTACACCC` | `GTTTGCAGGAGTGTACACCC` | `||||||||||||X|||||||` | 7 | -8 | 0.6855 |
| DBR1_DBR1_+_137893744.23-P1P2_07 | `GTTTGCAGGAGTCTACACCC` | `GTTTGCAGGGGTCTACACCC` | `|||||||||X||||||||||` | 10 | -11 | 0.9483 |
| DBR1_DBR1_+_137893744.23-P1P2_08 | `GTTTGCAGGAGTCTACACCC` | `GTTTGCAGTAGTCTACACCC` | `||||||||X|||||||||||` | 11 | -12 | 0.1529 |

基因组窗口视图（示例）：

```text
window      TATTTTGCAGGAGTCTACACCCAGGC
target      ..GTTTGCAGGAGTCTACACCC....
sgRNA       ..GATTGCAGGAGTCTACACCC....
```

**DUT**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| DUT_DUT_+_48624411.23-P1P2_00 | `GCGAGCGAGGAGACCACCGG` | `GCGAGCGAGGAGACCACCGG` | `||||||||||||||||||||` | - | - | - |
| DUT_DUT_+_48624411.23-P1P2_01 | `GCGAGCGAGGAGACCACCGG` | `GCCAGCGAGGAGACCACCGG` | `||X|||||||||||||||||` | 17 | -18 | 0.8345 |
| DUT_DUT_+_48624411.23-P1P2_07 | `GCGAGCGAGGAGACCACCGG` | `GCGAGCGAGGAGCCCACCGG` | `||||||||||||X|||||||` | 7 | -8 | 0.5071 |
| DUT_DUT_+_48624411.23-P1P2_08 | `GCGAGCGAGGAGACCACCGG` | `GCGAGCGAGGAGGCCACCGG` | `||||||||||||X|||||||` | 7 | -8 | 0.6386 |
| DUT_DUT_+_48624411.23-P1P2_10 | `GCGAGCGAGGAGACCACCGG` | `GCGAGCGAGGAGACCAACGG` | `||||||||||||||||X|||` | 3 | -4 | 0.2741 |

基因组窗口视图（示例）：

```text
window      AGGCGAGCGAGGAGACCACCGGGGGA
target      ..GCGAGCGAGGAGACCACCGG....
sgRNA       ..GCCAGCGAGGAGACCACCGG....
```

**EIF2S1**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| EIF2S1_EIF2S1_-_67827080.23-P1P2_00 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGAAGCGCACGCTGAGG` | `||||||||||||||||||||` | - | - | - |
| EIF2S1_EIF2S1_-_67827080.23-P1P2_01 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGAAACGCACGCTGAGG` | `|||||||X||||||||||||` | 12 | -13 | 0.2661 |
| EIF2S1_EIF2S1_-_67827080.23-P1P2_02 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGCAGCGCACGCTGAGG` | `|||||X||||||||||||||` | 14 | -15 | 0.5807 |
| EIF2S1_EIF2S1_-_67827080.23-P1P2_06 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGAAGCGCGCGCTGAGG` | `|||||||||||X||||||||` | 8 | -9 | 0.4984 |
| EIF2S1_EIF2S1_-_67827080.23-P1P2_07 | `GAGCGAAGCGCACGCTGAGG` | `GAGCGAAGCGCTCGCTGAGG` | `|||||||||||X||||||||` | 8 | -9 | 0.0347 |

基因组窗口视图（示例）：

```text
window      GTGAGCGAAGCGCACGCTGAGGAGGA
target      GAGCGAAGCGCACGCTGAGG......
sgRNA       GAGCGAAACGCACGCTGAGG......
```

**GATA1**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| GATA1_GATA1_-_48645022.23-P1P2_00 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTGCCACATCCCCA` | `||||||||||||||||||||` | - | - | - |
| GATA1_GATA1_-_48645022.23-P1P2_03 | `GTGAGCTTGCCACATCCCCA` | `GTGCGCTTGCCACATCCCCA` | `|||X||||||||||||||||` | 16 | -17 | 0.8463 |
| GATA1_GATA1_-_48645022.23-P1P2_04 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTACCACATCCCCA` | `||||||||X|||||||||||` | 11 | -12 | 0.6188 |
| GATA1_GATA1_-_48645022.23-P1P2_06 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTGCGACATCCCCA` | `||||||||||X|||||||||` | 9 | -10 | 0.2881 |
| GATA1_GATA1_-_48645022.23-P1P2_08 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTTCCACATCCCCA` | `||||||||X|||||||||||` | 11 | -12 | 0.4760 |
| GATA1_GATA1_-_48645022.23-P1P2_12 | `GTGAGCTTGCCACATCCCCA` | `GTGAGCTTGCCACATCCGCA` | `|||||||||||||||||X||` | 2 | -3 | 0.0784 |

基因组窗口视图（示例）：

```text
window      CACTGAGCTTGCCACATCCCCAAGGC
target      ..GTGAGCTTGCCACATCCCCA....
sgRNA       ..GTGCGCTTGCCACATCCCCA....
```

**GINS1**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| GINS1_GINS1_-_25388381.23-P1P2_00 | `GGACTAGAACGAAAGGAGTG` | `GGACTAGAACGAAAGGAGTG` | `||||||||||||||||||||` | - | - | - |
| GINS1_GINS1_-_25388381.23-P1P2_03 | `GGACTAGAACGAAAGGAGTG` | `GGACTATAACGAAAGGAGTG` | `||||||X|||||||||||||` | 13 | -14 | 0.5906 |
| GINS1_GINS1_-_25388381.23-P1P2_06 | `GGACTAGAACGAAAGGAGTG` | `GGACTAGAACGGAAGGAGTG` | `|||||||||||X||||||||` | 8 | -9 | 0.7740 |
| GINS1_GINS1_-_25388381.23-P1P2_08 | `GGACTAGAACGAAAGGAGTG` | `GGACTAGAGCGAAAGGAGTG` | `||||||||X|||||||||||` | 11 | -12 | 1.0996 |
| GINS1_GINS1_-_25388381.23-P1P2_14 | `GGACTAGAACGAAAGGAGTG` | `GGACTAGAACGAAAGGAGCG` | `||||||||||||||||||X|` | 1 | -2 | 0.3095 |

基因组窗口视图（示例）：

```text
window      TAGGACTAGAACGAAAGGAGTGAGGC
target      GGACTAGAACGAAAGGAGTG......
sgRNA       GGACTATAACGAAAGGAGTG......
```

**GNB2L1**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| GNB2L1_GNB2L1_+_180670873.23-P1P2_02 | `GTGCAAGGCGGCGGCAGGAG` | `GTGCAAGACGGCGGCAGGAG` | `|||||||X||||||||||||` | 12 | -13 | 0.2049 |
| GNB2L1_GNB2L1_+_180670873.23-P1P2_07 | `GTGCAAGGCGGCGGCAGGAG` | `GTGCAAGGCGGGGGCAGGAG` | `|||||||||||X||||||||` | 8 | -9 | 0.3337 |
| GNB2L1_GNB2L1_+_180670873.23-P1P2_08 | `GTGCAAGGCGGCGGCAGGAG` | `GTGCAAGGTGGCGGCAGGAG` | `||||||||X|||||||||||` | 11 | -12 | 1.2641 |
| GNB2L1_GNB2L1_+_180670873.23-P1P2_13 | `GTGCAAGGCGGCGGCAGGAG` | `GTGCAAGGCGGCGGCGGGAG` | `|||||||||||||||X||||` | 4 | -5 | 0.4584 |

基因组窗口视图（示例）：

```text
window      CACTGCAAGGCGGCGGCAGGAGAGGT
target      ..GTGCAAGGCGGCGGCAGGAG....
sgRNA       ..GTGCAAGACGGCGGCAGGAG....
```

**HSPA5**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| HSPA5_HSPA5_+_128003624.23-P1P2_00 | `GAGCCGAGTAGGCGACGGTG` | `GAGCCGAGTAGGCGACGGTG` | `||||||||||||||||||||` | - | - | - |
| HSPA5_HSPA5_+_128003624.23-P1P2_01 | `GAGCCGAGTAGGCGACGGTG` | `GAACCGAGTAGGCGACGGTG` | `||X|||||||||||||||||` | 17 | -18 | 0.9105 |
| HSPA5_HSPA5_+_128003624.23-P1P2_04 | `GAGCCGAGTAGGCGACGGTG` | `GAGCCGAGAAGGCGACGGTG` | `||||||||X|||||||||||` | 11 | -12 | 1.1825 |
| HSPA5_HSPA5_+_128003624.23-P1P2_06 | `GAGCCGAGTAGGCGACGGTG` | `GAGCCGAGTAGACGACGGTG` | `|||||||||||X||||||||` | 8 | -9 | 0.2975 |
| HSPA5_HSPA5_+_128003624.23-P1P2_08 | `GAGCCGAGTAGGCGACGGTG` | `GAGCCGAGTGGGCGACGGTG` | `|||||||||X||||||||||` | 10 | -11 | 0.9807 |

基因组窗口视图（示例）：

```text
window      ATAAGCCGAGTAGGCGACGGTGAGGT
target      ..GAGCCGAGTAGGCGACGGTG....
sgRNA       ..GAACCGAGTAGGCGACGGTG....
```

**HSPA9**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| HSPA9_HSPA9_-_137911079.23-P1P2_00 | `GGAGCTGCGCGATGCGGTGG` | `GGAGCTGCGCGATGCGGTGG` | `||||||||||||||||||||` | - | - | - |
| HSPA9_HSPA9_-_137911079.23-P1P2_02 | `GGAGCTGCGCGATGCGGTGG` | `GGAGTTGCGCGATGCGGTGG` | `||||X|||||||||||||||` | 15 | -16 | 0.6057 |
| HSPA9_HSPA9_-_137911079.23-P1P2_04 | `GGAGCTGCGCGATGCGGTGG` | `GGAGCTGCGCAATGCGGTGG` | `||||||||||X|||||||||` | 9 | -10 | 0.2255 |
| HSPA9_HSPA9_-_137911079.23-P1P2_07 | `GGAGCTGCGCGATGCGGTGG` | `GGAGCTGCGGGATGCGGTGG` | `|||||||||X||||||||||` | 10 | -11 | 0.9056 |
| HSPA9_HSPA9_-_137911079.23-P1P2_08 | `GGAGCTGCGCGATGCGGTGG` | `GGAGCTGCTCGATGCGGTGG` | `||||||||X|||||||||||` | 11 | -12 | 0.4609 |

基因组窗口视图（示例）：

```text
window      AAAGAGCTGCGCGATGCGGTGGCGGC
target      ..GGAGCTGCGCGATGCGGTGG....
sgRNA       ..GGAGTTGCGCGATGCGGTGG....
```

**HSPE1**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| HSPE1_HSPE1_+_198365089.23-P1P2_00 | `GGAGACTCGCAGTCCGGCCC` | `GGAGACTCGCAGTCCGGCCC` | `||||||||||||||||||||` | - | - | - |
| HSPE1_HSPE1_+_198365089.23-P1P2_01 | `GGAGACTCGCAGTCCGGCCC` | `GGAGACACGCAGTCCGGCCC` | `||||||X|||||||||||||` | 13 | -14 | 0.9382 |
| HSPE1_HSPE1_+_198365089.23-P1P2_02 | `GGAGACTCGCAGTCCGGCCC` | `GGAGACTGGCAGTCCGGCCC` | `|||||||X||||||||||||` | 12 | -13 | 0.5637 |
| HSPE1_HSPE1_+_198365089.23-P1P2_03 | `GGAGACTCGCAGTCCGGCCC` | `GGTGACTCGCAGTCCGGCCC` | `||X|||||||||||||||||` | 17 | -18 | 0.8658 |
| HSPE1_HSPE1_+_198365089.23-P1P2_14 | `GGAGACTCGCAGTCCGGCCC` | `GGAGACTCGCAGTCCTGCCC` | `|||||||||||||||X||||` | 4 | -5 | 0.3024 |

基因组窗口视图（示例）：

```text
window      AAAGAGACTCGCAGTCCGGCCCTGGC
target      ..GGAGACTCGCAGTCCGGCCC....
sgRNA       ..GGAGACACGCAGTCCGGCCC....
```

**MTOR**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| MTOR_MTOR_+_11322547.23-P1P2_00 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGCCTGAAGCGG` | `||||||||||||||||||||` | - | - | - |
| MTOR_MTOR_+_11322547.23-P1P2_05 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGCTTGAAGCGG` | `|||||||||||X||||||||` | 8 | -9 | 0.4677 |
| MTOR_MTOR_+_11322547.23-P1P2_06 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGGCTGAAGCGG` | `||||||||||X|||||||||` | 9 | -10 | 0.3030 |
| MTOR_MTOR_+_11322547.23-P1P2_07 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGTCTGAAGCGG` | `||||||||||X|||||||||` | 9 | -10 | 0.8185 |
| MTOR_MTOR_+_11322547.23-P1P2_10 | `GGGCAGGGGGCCTGAAGCGG` | `GGGCAGGGGGCCTGAAGCAG` | `||||||||||||||||||X|` | 1 | -2 | 0.5089 |

基因组窗口视图（示例）：

```text
window      TGGGGCAGGGGGCCTGAAGCGGCGGT
target      GGGCAGGGGGCCTGAAGCGG......
sgRNA       GGGCAGGGGGCTTGAAGCGG......
```

**POLR1D**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| POLR1D_POLR1D_+_28196016.23-P1_00 | `GGGAAGCAAGGACCGACCGA` | `GGGAAGCAAGGACCGACCGA` | `||||||||||||||||||||` | - | - | - |
| POLR1D_POLR1D_+_28196016.23-P1_01 | `GGGAAGCAAGGACCGACCGA` | `GGGAAGCCAGGACCGACCGA` | `|||||||X||||||||||||` | 12 | -13 | 0.8852 |
| POLR1D_POLR1D_+_28196016.23-P1_03 | `GGGAAGCAAGGACCGACCGA` | `GGTAAGCAAGGACCGACCGA` | `||X|||||||||||||||||` | 17 | -18 | 1.0944 |
| POLR1D_POLR1D_+_28196016.23-P1_07 | `GGGAAGCAAGGACCGACCGA` | `GGGAAGCAAGGAGCGACCGA` | `||||||||||||X|||||||` | 7 | -8 | 0.2701 |
| POLR1D_POLR1D_+_28196016.23-P1_08 | `GGGAAGCAAGGACCGACCGA` | `GGGAAGCAGGGACCGACCGA` | `||||||||X|||||||||||` | 11 | -12 | 1.1592 |

基因组窗口视图（示例）：

```text
window      GCAGGAAGCAAGGACCGACCGACGGA
target      ..GGGAAGCAAGGACCGACCGA....
sgRNA       ..GGGAAGCCAGGACCGACCGA....
```

**POLR2H**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| POLR2H_POLR2H_+_184081251.23-P1P2_00 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGAGAGCAGCAGA` | `||||||||||||||||||||` | - | - | - |
| POLR2H_POLR2H_+_184081251.23-P1P2_07 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGAGTGCAGCAGA` | `|||||||||||X||||||||` | 8 | -9 | 0.0472 |
| POLR2H_POLR2H_+_184081251.23-P1P2_08 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGCGAGCAGCAGA` | `|||||||||X||||||||||` | 10 | -11 | 0.2492 |
| POLR2H_POLR2H_+_184081251.23-P1P2_11 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGAGAGCAGCGGA` | `|||||||||||||||||X||` | 2 | -3 | 0.7822 |
| POLR2H_POLR2H_+_184081251.23-P1P2_12 | `GGGGCCACGAGAGCAGCAGA` | `GGGGCCACGAGAGCAGGAGA` | `||||||||||||||||X|||` | 3 | -4 | 0.1553 |

基因组窗口视图（示例）：

```text
window      AGGGGGCCACGAGAGCAGCAGAAGGG
target      GGGGCCACGAGAGCAGCAGA......
sgRNA       GGGGCCACGAGTGCAGCAGA......
```

**RAN**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| RAN_RAN_+_131356438.23-P1P2_00 | `GGCGGTCGCTGCGCTTAGGG` | `GGCGGTCGCTGCGCTTAGGG` | `||||||||||||||||||||` | - | - | - |
| RAN_RAN_+_131356438.23-P1P2_02 | `GGCGGTCGCTGCGCTTAGGG` | `GGCGGCCGCTGCGCTTAGGG` | `|||||X||||||||||||||` | 14 | -15 | - |
| RAN_RAN_+_131356438.23-P1P2_03 | `GGCGGTCGCTGCGCTTAGGG` | `GGGGGTCGCTGCGCTTAGGG` | `||X|||||||||||||||||` | 17 | -18 | - |
| RAN_RAN_+_131356438.23-P1P2_04 | `GGCGGTCGCTGCGCTTAGGG` | `GGCGGTCGCGGCGCTTAGGG` | `|||||||||X||||||||||` | 10 | -11 | - |
| RAN_RAN_+_131356438.23-P1P2_12 | `GGCGGTCGCTGCGCTTAGGG` | `GGCGGTCGCTGCGCTTAGGT` | `|||||||||||||||||||X` | 0 | -1 | - |

**RPL9**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| RPL9_RPL9_+_39460483.23-P1P2_00 | `GGATGTTTCTGTGCTCGTGG` | `GGATGTTTCTGTGCTCGTGG` | `||||||||||||||||||||` | - | - | - |
| RPL9_RPL9_+_39460483.23-P1P2_01 | `GGATGTTTCTGTGCTCGTGG` | `GGATGATTCTGTGCTCGTGG` | `|||||X||||||||||||||` | 14 | -15 | 0.3911 |
| RPL9_RPL9_+_39460483.23-P1P2_04 | `GGATGTTTCTGTGCTCGTGG` | `GGATGTTTCAGTGCTCGTGG` | `|||||||||X||||||||||` | 10 | -11 | 0.2013 |
| RPL9_RPL9_+_39460483.23-P1P2_05 | `GGATGTTTCTGTGCTCGTGG` | `GGATGTTTCGGTGCTCGTGG` | `|||||||||X||||||||||` | 10 | -11 | 0.2814 |
| RPL9_RPL9_+_39460483.23-P1P2_07 | `GGATGTTTCTGTGCTCGTGG` | `GGATGTTTCTGCGCTCGTGG` | `|||||||||||X||||||||` | 8 | -9 | 0.0636 |

基因组窗口视图（示例）：

```text
window      AAGGATGTTTCTGTGCTCGTGGGGGT
target      ..GGATGTTTCTGTGCTCGTGG....
sgRNA       ..GGATGATTCTGTGCTCGTGG....
```

**RPS14**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| RPS14_RPS14_+_149829238.23-P1P2_00 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCGGGCGCGACAATC` | `||||||||||||||||||||` | - | - | - |
| RPS14_RPS14_+_149829238.23-P1P2_01 | `GAGGCCCGGGCGCGACAATC` | `GAGACCCGGGCGCGACAATC` | `|||X||||||||||||||||` | 16 | -17 | 0.9361 |
| RPS14_RPS14_+_149829238.23-P1P2_02 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCTGGCGCGACAATC` | `|||||||X||||||||||||` | 12 | -13 | 0.5687 |
| RPS14_RPS14_+_149829238.23-P1P2_04 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCGCGCGCGACAATC` | `||||||||X|||||||||||` | 11 | -12 | 0.2249 |
| RPS14_RPS14_+_149829238.23-P1P2_08 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCGGGCTCGACAATC` | `|||||||||||X||||||||` | 8 | -9 | 0.0274 |
| RPS14_RPS14_+_149829238.23-P1P2_13 | `GAGGCCCGGGCGCGACAATC` | `GAGGCCCGGGCGCGACAGTC` | `|||||||||||||||||X||` | 2 | -3 | 0.0910 |

基因组窗口视图（示例）：

```text
window      AGGAGGCCCGGGCGCGACAATCGGGG
target      ..GAGGCCCGGGCGCGACAATC....
sgRNA       ..GAGACCCGGGCGCGACAATC....
```

**RPS15**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| RPS15_RPS15_-_1438413.23-P1P2_00 | `GACCAAAGCGATCTCTTCTG` | `GACCAAAGCGATCTCTTCTG` | `||||||||||||||||||||` | - | - | - |
| RPS15_RPS15_-_1438413.23-P1P2_01 | `GACCAAAGCGATCTCTTCTG` | `GACCAAACCGATCTCTTCTG` | `|||||||X||||||||||||` | 12 | -13 | - |
| RPS15_RPS15_-_1438413.23-P1P2_02 | `GACCAAAGCGATCTCTTCTG` | `GACCAAGGCGATCTCTTCTG` | `||||||X|||||||||||||` | 13 | -14 | - |
| RPS15_RPS15_-_1438413.23-P1P2_07 | `GACCAAAGCGATCTCTTCTG` | `GACCAAAGCGGTCTCTTCTG` | `||||||||||X|||||||||` | 9 | -10 | - |
| RPS15_RPS15_-_1438413.23-P1P2_12 | `GACCAAAGCGATCTCTTCTG` | `GACCAAAGCGATCTCTTGTG` | `|||||||||||||||||X||` | 2 | -3 | - |

**RPS18**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| RPS18_RPS18_+_33239917.23-P1P2_00 | `GCTGCGATGCCGCTGGATCA` | `GCTGCGATGCCGCTGGATCA` | `||||||||||||||||||||` | - | - | - |
| RPS18_RPS18_+_33239917.23-P1P2_01 | `GCTGCGATGCCGCTGGATCA` | `GCTGCAATGCCGCTGGATCA` | `|||||X||||||||||||||` | 14 | -15 | 0.6302 |
| RPS18_RPS18_+_33239917.23-P1P2_02 | `GCTGCGATGCCGCTGGATCA` | `GCTGGGATGCCGCTGGATCA` | `||||X|||||||||||||||` | 15 | -16 | 0.4740 |
| RPS18_RPS18_+_33239917.23-P1P2_04 | `GCTGCGATGCCGCTGGATCA` | `GCTGCGATCCCGCTGGATCA` | `||||||||X|||||||||||` | 11 | -12 | 0.0219 |
| RPS18_RPS18_+_33239917.23-P1P2_08 | `GCTGCGATGCCGCTGGATCA` | `GCTGCGATTCCGCTGGATCA` | `||||||||X|||||||||||` | 11 | -12 | 0.1609 |

基因组窗口视图（示例）：

```text
window      GAGCTGCGATGCCGCTGGATCACGGC
target      ..GCTGCGATGCCGCTGGATCA....
sgRNA       ..GCTGCAATGCCGCTGGATCA....
```

**SEC61A1**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| SEC61A1_SEC61A1_-_127771295.23-P1_00 | `GGCACTGACGTGTCTCTCGG` | `GGCACTGACGTGTCTCTCGG` | `||||||||||||||||||||` | - | - | - |
| SEC61A1_SEC61A1_-_127771295.23-P1_01 | `GGCACTGACGTGTCTCTCGG` | `GGCACTGTCGTGTCTCTCGG` | `|||||||X||||||||||||` | 12 | -13 | 0.5814 |
| SEC61A1_SEC61A1_-_127771295.23-P1_02 | `GGCACTGACGTGTCTCTCGG` | `GGCGCTGACGTGTCTCTCGG` | `|||X||||||||||||||||` | 16 | -17 | 0.9131 |
| SEC61A1_SEC61A1_-_127771295.23-P1_03 | `GGCACTGACGTGTCTCTCGG` | `GGTACTGACGTGTCTCTCGG` | `||X|||||||||||||||||` | 17 | -18 | 0.4840 |
| SEC61A1_SEC61A1_-_127771295.23-P1_04 | `GGCACTGACGTGTCTCTCGG` | `GGCACTGAAGTGTCTCTCGG` | `||||||||X|||||||||||` | 11 | -12 | 0.1890 |

基因组窗口视图（示例）：

```text
window      CTAGCACTGACGTGTCTCTCGGCGGA
target      ..GGCACTGACGTGTCTCTCGG....
sgRNA       ..GGCACTGTCGTGTCTCTCGG....
```

**TUBB**

| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |
| --- | --- | --- | --- | --- | ---: | ---: |
| TUBB_TUBB_+_30688126.23-P1_00 | `GCGGCAGGAAGGTTCTGAGA` | `GCGGCAGGAAGGTTCTGAGA` | `||||||||||||||||||||` | - | - | - |
| TUBB_TUBB_+_30688126.23-P1_01 | `GCGGCAGGAAGGTTCTGAGA` | `GCAGCAGGAAGGTTCTGAGA` | `||X|||||||||||||||||` | 17 | -18 | 0.8663 |
| TUBB_TUBB_+_30688126.23-P1_03 | `GCGGCAGGAAGGTTCTGAGA` | `GCGGCAGCAAGGTTCTGAGA` | `|||||||X||||||||||||` | 12 | -13 | 0.5085 |
| TUBB_TUBB_+_30688126.23-P1_06 | `GCGGCAGGAAGGTTCTGAGA` | `GCGGCAGGACGGTTCTGAGA` | `|||||||||X||||||||||` | 10 | -11 | 0.8256 |
| TUBB_TUBB_+_30688126.23-P1_10 | `GCGGCAGGAAGGTTCTGAGA` | `GCGGCAGGAAGGTTCAGAGA` | `|||||||||||||||X||||` | 4 | -5 | 0.1138 |

基因组窗口视图（示例）：

```text
window      CGACGGCAGGAAGGTTCTGAGAGGGA
target      ..GCGGCAGGAAGGTTCTGAGA....
sgRNA       ..GCAGCAGGAAGGTTCTGAGA....
```

## 3. 模型 latent 原始值

### 3.1 delta_g（模型为每个扰动学到的 sgRNA 向量，128 维原始值）

完整 128 维见 `runs\gse132080_target_200ep\preliminary\sgrna_delta_vectors.csv`；下方按条件列出前 16 维。

| perturbation_id | target gene | delta_g[0:16] |
| ---: | --- | --- |
| 14 | ALDOA | 0.0002, -0.0003, 0.0050, 0.0005, 0.0015, -0.0018, 0.0001, -0.0004, 0.0001, 0.0000, 0.0027, -0.0013, 0.0021, -0.0012, -0.0010, 0.0010, … |
| 8 | ATP5E | 0.0014, 0.0029, 0.0035, 0.0072, 0.0043, -0.0051, 0.0053, 0.0015, 0.0017, 0.0007, -0.0053, 0.0001, 0.0019, 0.0056, -0.0039, -0.0009, … |
| 23 | BCR | 0.0074, -0.0060, 0.0043, 0.0057, 0.0039, -0.0091, -0.0051, 0.0038, -0.0088, -0.0010, 0.0095, 0.0054, 0.0110, -0.0129, 0.0044, 0.0011, … |
| 7 | CAD | 0.0093, 0.0030, -0.0026, 0.0015, -0.0042, -0.0011, -0.0112, 0.0143, -0.0025, 0.0027, 0.0057, 0.0040, 0.0072, -0.0026, -0.0028, -0.0042, … |
| 12 | CDC23 | 0.0061, -0.0012, -0.0029, -0.0022, 0.0057, -0.0011, -0.0013, 0.0000, -0.0027, -0.0042, 0.0004, 0.0030, -0.0003, 0.0001, 0.0009, 0.0005, … |
| 1 | COX11 | -0.0027, 0.0007, -0.0052, 0.0006, 0.0023, 0.0010, 0.0049, -0.0026, 0.0013, -0.0002, -0.0007, -0.0008, 0.0013, 0.0007, -0.0023, 0.0019, … |
| 17 | DBR1 | 0.0056, 0.0042, -0.0018, -0.0040, -0.0098, -0.0013, -0.0009, -0.0108, 0.0016, 0.0007, -0.0012, 0.0064, 0.0004, -0.0060, 0.0043, -0.0009, … |
| 5 | DUT | 0.0043, 0.0087, 0.0055, 0.0115, 0.0024, -0.0034, 0.0013, 0.0115, 0.0040, 0.0041, -0.0001, -0.0022, 0.0051, -0.0042, -0.0035, -0.0031, … |
| 22 | EIF2S1 | 0.0025, -0.0017, -0.0021, -0.0013, 0.0070, 0.0004, 0.0036, -0.0073, 0.0056, -0.0222, 0.0032, 0.0089, 0.0002, 0.0030, 0.0102, -0.0078, … |
| 2 | GATA1 | 0.0054, 0.0086, 0.0187, 0.0022, 0.0081, -0.0015, -0.0089, -0.0060, 0.0075, -0.0159, -0.0050, -0.0017, -0.0049, -0.0117, -0.0356, -0.0011, … |
| 3 | GINS1 | 0.0008, 0.0021, 0.0021, -0.0040, 0.0008, -0.0014, -0.0002, 0.0085, -0.0024, -0.0057, 0.0030, 0.0049, -0.0003, -0.0027, -0.0057, -0.0029, … |
| 9 | GNB2L1 | 0.0003, -0.0046, 0.0061, -0.0087, 0.0017, -0.0024, -0.0010, -0.0024, -0.0003, -0.0017, 0.0030, -0.0014, 0.0001, -0.0033, 0.0003, 0.0048, … |
| 19 | HSPA5 | 0.0138, 0.0116, 0.0240, 0.0015, 0.0378, -0.0106, 0.0157, -0.0077, -0.0048, -0.0305, 0.0129, -0.0191, -0.0083, -0.0129, 0.0097, -0.0036, … |
| 25 | HSPA9 | 0.0014, 0.0041, -0.0026, -0.0041, 0.0047, -0.0016, 0.0026, -0.0043, 0.0108, -0.0224, -0.0005, 0.0120, -0.0025, 0.0018, 0.0045, -0.0122, … |
| 20 | HSPE1 | -0.0014, 0.0061, 0.0021, -0.0038, 0.0027, -0.0024, 0.0021, -0.0112, -0.0001, -0.0054, -0.0018, -0.0014, 0.0039, 0.0035, -0.0024, -0.0014, … |
| 21 | MTOR | -0.0000, 0.0000, -0.0000, 0.0000, -0.0072, -0.0000, -0.0000, -0.0000, -0.0002, -0.0000, -0.0000, -0.0000, 0.0000, -0.0000, 0.0000, 0.0000, … |
| 4 | POLR1D | 0.0077, -0.0009, -0.0056, -0.0026, -0.0012, -0.0050, -0.0048, -0.0031, -0.0002, -0.0032, 0.0087, -0.0019, 0.0101, 0.0006, 0.0011, 0.0044, … |
| 13 | POLR2H | 0.0018, -0.0012, 0.0049, -0.0105, 0.0056, 0.0012, 0.0007, -0.0119, 0.0019, -0.0072, 0.0007, -0.0036, 0.0043, -0.0015, -0.0057, 0.0003, … |
| 11 | RAN | 0.0009, -0.0037, 0.0007, -0.0036, -0.0031, 0.0007, -0.0011, 0.0069, -0.0010, 0.0002, 0.0015, 0.0063, 0.0008, -0.0028, 0.0032, -0.0028, … |
| 18 | RPL9 | 0.0004, 0.0049, -0.0074, 0.0009, -0.0010, -0.0055, -0.0002, -0.0065, -0.0058, -0.0050, -0.0030, -0.0088, -0.0008, -0.0021, 0.0016, 0.0020, … |
| 10 | RPS14 | 0.0037, 0.0016, -0.0014, -0.0013, -0.0015, -0.0085, -0.0025, -0.0070, -0.0022, -0.0001, 0.0017, -0.0081, -0.0012, -0.0010, 0.0034, 0.0102, … |
| 24 | RPS15 | 0.0009, 0.0003, -0.0000, -0.0001, -0.0009, -0.0057, -0.0002, -0.0026, -0.0006, -0.0011, 0.0005, -0.0027, -0.0016, -0.0019, 0.0003, 0.0033, … |
| 15 | RPS18 | 0.0035, 0.0024, -0.0016, -0.0031, -0.0040, -0.0057, -0.0017, -0.0092, 0.0027, -0.0026, 0.0028, -0.0034, 0.0013, -0.0022, 0.0018, 0.0035, … |
| 16 | SEC61A1 | -0.0008, 0.0029, 0.0009, 0.0018, 0.0068, 0.0000, 0.0035, -0.0016, -0.0009, -0.0038, 0.0004, -0.0029, -0.0032, -0.0006, -0.0018, -0.0004, … |
| 6 | TUBB | 0.0084, -0.0019, 0.0001, -0.0003, 0.0056, -0.0005, -0.0039, -0.0001, -0.0017, -0.0029, 0.0043, -0.0005, 0.0032, -0.0013, -0.0015, -0.0039, … |

### 3.2 Z-sem（SemanticEncoder 输出的 128 维 latent 原始值）

| 条件 | z_sem[0:16] |
| --- | --- |
| control 池均值 | -0.0915, -0.0416, 0.3218, 0.0316, 0.0423, 0.0507, 0.0625, 0.3651, 0.1674, -0.0794, 0.2645, -0.2967, 0.0782, 0.0435, 0.1042, 0.2893, … |
| control 平均表达编码 | -0.0998, 0.0485, 0.0218, 0.1129, 0.2283, -0.0227, 0.1560, -0.0696, 0.1287, 0.1040, 0.0228, -0.0950, 0.0620, -0.1407, 0.0882, 0.1164, … |
| ALDOA_ALDOA_+_30077139.23-P1P2_00 观测值 | -0.1685, -0.0538, 0.0836, -0.0294, 0.1442, 0.2379, -0.1582, 0.1678, 0.5803, -0.3644, -0.3137, 0.4072, 0.3466, -0.2442, -0.1938, 0.3878, … |
| ALDOA_ALDOA_+_30077139.23-P1P2_06 观测值 | -0.2773, -0.4802, 0.7206, -0.4209, -0.1973, 0.0247, -0.4900, 0.0612, 0.0134, -0.1291, -0.1471, -0.1147, 0.5726, -0.3781, -0.1164, -0.1211, … |
| ALDOA_ALDOA_+_30077139.23-P1P2_07 观测值 | 0.1435, 0.1497, 0.2345, -0.5877, -0.0287, 0.0850, -0.1744, 0.6934, 0.4753, -0.3058, -0.3124, -0.0294, 1.0054, -0.3238, 0.2359, 0.9183, … |
| ALDOA_ALDOA_+_30077139.23-P1P2_13 观测值 | -0.0605, -0.1748, 0.2824, -0.0839, -0.3244, 0.1301, 0.0013, -0.0504, 0.0180, -0.0619, -0.3431, 0.2020, 0.3099, -0.3071, 0.1729, 0.3756, … |
| ALDOA_ALDOA_+_30077139.23-P1P2_14 观测值 | -0.2488, -0.7054, -0.3371, -0.0193, 0.3978, -0.1174, 0.3277, 0.0806, -0.0947, -0.3431, -0.3607, 0.0742, -0.3147, -0.0623, -0.3348, -0.3196, … |
| ATP5E_ATP5E_-_57607036.23-P1P2_00 观测值 | 0.2083, -0.3660, 0.0482, 0.0864, 0.3929, -0.1890, -0.5903, -0.0343, -0.5274, -0.1280, 0.3813, 0.0828, 0.0847, -0.5212, 0.1925, 0.1325, … |
| ATP5E_ATP5E_-_57607036.23-P1P2_01 观测值 | -0.3290, -0.4307, 0.0626, 0.0863, 0.6539, -0.3185, -0.3315, 0.4387, -0.2937, 0.2639, 0.1135, 0.0904, 0.2729, -0.2362, 0.2146, 0.3175, … |
| ATP5E_ATP5E_-_57607036.23-P1P2_04 观测值 | -0.1615, -0.3394, 0.3936, -0.1383, 0.3180, -0.1885, -0.1285, -0.3402, -0.2374, -0.0400, 0.4119, 0.0686, 0.2608, 0.0153, -0.0627, -0.1072, … |
| ATP5E_ATP5E_-_57607036.23-P1P2_14 观测值 | 0.3356, 0.5586, 0.4230, 0.5182, 0.6961, -0.2518, 0.0625, 0.2322, -0.5827, -0.4139, 0.2948, 0.1324, 0.0852, -0.6748, 0.4692, 0.1847, … |
| ATP5E_ATP5E_-_57607036.23-P1P2_16 观测值 | -0.1962, -0.1023, 0.0898, -0.0740, 0.2215, -0.0959, -0.0146, -0.3212, -0.3240, 0.3412, 0.2714, -0.0096, 0.2490, -0.6451, 0.1356, -0.1561, … |
| BCR_BCR_+_23523092.23-P1P2_00 观测值 | 0.3355, 0.8495, -0.4992, 0.6906, 0.1947, -0.4425, -0.7696, 1.2341, -0.4499, -1.0046, -0.0835, 0.0075, 1.3773, 0.9926, -0.2072, 1.1178, … |
| BCR_BCR_+_23523092.23-P1P2_04 观测值 | 0.2594, 0.8111, -1.5801, -0.1713, 0.9632, -0.5379, -0.5971, 1.7385, -0.6029, -1.0373, -0.1264, -0.3010, 0.5906, 0.5792, 0.1765, 1.0048, … |

全部 127 条 sgRNA 的观测 Z-sem 与反事实 `z_cf = z_sem(control) + delta_g` 见 `runs\gse132080_target_200ep\preliminary\z_sem_conditions.csv`，对照细胞逐个 latent 见 `runs\gse132080_target_200ep\preliminary\z_sem_control_cells.csv`。

## 4. 原始表达值

| 文件 | 内容 |
| --- | --- |
| `runs\gse132080_target_200ep\preliminary\expression_control_mean.csv` | 211 个 control 细胞的平均表达（1019 基因） |
| `runs\gse132080_target_200ep\preliminary\sgrna_expression_matrix.csv` | 每条 sgRNA 的观测平均表达（1019 × 127） |
| `runs\gse132080_target_200ep\preliminary\sgrna_predicted_expression_matrix.csv` | DDIM 生成表达（1019 × 127） |
| `runs\gse132080_target_200ep\preliminary\sgrna_top_gene_expression_long.csv` | 每条 sgRNA 的 top 差异基因长表 |

下方按 target gene 列出代表 sgRNA 的 top 6 差异基因的原始表达值。

**ALDOA**

`ALDOA_ALDOA_+_30077139.23-P1P2_00`（sgRNA `GGTCACCAGGACCCCTTCTG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ALDOA | 2.6565 | 1.1875 | 2.6862 |
| CHI3L2 | 0.8015 | 1.3865 | 0.7727 |
| GAL | 1.8857 | 1.5022 | 2.0522 |
| KRT18 | 1.5131 | 1.8669 | 1.5594 |
| FAM178B | 0.7202 | 1.0694 | 0.8307 |
| MYLIP | 0.3438 | 0.6705 | 0.2501 |

`ALDOA_ALDOA_+_30077139.23-P1P2_06`（sgRNA `GGTCACCAGGATCCCTTCTG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ALDOA | 2.6565 | 1.3005 | 2.6817 |
| NUSAP1 | 0.5682 | 1.0686 | 0.5109 |
| HMGB2 | 1.8561 | 2.3419 | 1.8689 |
| MYL4 | 1.3474 | 1.7474 | 1.4158 |
| HIST1H2BJ | 1.1310 | 1.5192 | 1.1874 |
| CDC20 | 0.7090 | 1.0684 | 0.6923 |

`ALDOA_ALDOA_+_30077139.23-P1P2_07`（sgRNA `GGTCACCAGGCCCCCTTCTG`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ALDOA | 2.6565 | 0.8659 | 2.6841 |
| HIST1H4C | 2.8420 | 2.2401 | 3.0274 |
| HBZ | 2.7562 | 3.3426 | 2.8217 |
| CDC45 | 0.9935 | 0.4222 | 1.1029 |
| SOCS2 | 1.0925 | 0.5756 | 1.1261 |
| DUT | 1.6702 | 1.2655 | 1.7606 |

`ALDOA_ALDOA_+_30077139.23-P1P2_13`（sgRNA `GGTCACCAGGACCCCTTTTG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HNRNPH1 | 1.2575 | 0.9389 | 1.3225 |
| PRSS57 | 0.9886 | 0.6755 | 1.0476 |
| PIM1 | 0.5387 | 0.8467 | 0.4948 |
| SLC3A2 | 0.7141 | 0.4130 | 0.6540 |
| HMGB2 | 1.8561 | 2.1549 | 1.8659 |
| HSP90B1 | 1.4657 | 1.1699 | 1.4812 |

`ALDOA_ALDOA_+_30077139.23-P1P2_14`（sgRNA `GGTCACCAGGACCGCTTCTG`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ALDOA | 2.6565 | 1.3457 | 2.6812 |
| CENPF | 0.9886 | 1.4870 | 1.0397 |
| TUBB4B | 1.3472 | 1.7869 | 1.4954 |
| SH3BGRL3 | 0.8037 | 1.2225 | 0.7932 |
| HIST1H2AC | 1.0096 | 1.4211 | 1.0078 |
| PIF1 | 0.2967 | 0.6833 | 0.2627 |

**ATP5E**

`ATP5E_ATP5E_-_57607036.23-P1P2_00`（sgRNA `GGTGTCCAGGGGCACTCTGT`，cells=14）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ATP5E | 3.1685 | 1.6075 | 3.1043 |
| TUBB2B | 0.3264 | 0.6881 | 0.3660 |
| KPNA2 | 0.8283 | 1.1084 | 0.9105 |
| FAM83A | 0.6063 | 0.8806 | 0.4977 |
| HIST1H4C | 2.8420 | 2.5878 | 3.0315 |
| HIST1H2AC | 1.0096 | 0.7570 | 1.0063 |

`ATP5E_ATP5E_-_57607036.23-P1P2_01`（sgRNA `GGTGTCCTGGGGCACTCTGT`，cells=14）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ATP5E | 3.1685 | 1.5340 | 3.0988 |
| HBG1 | 0.9801 | 1.3648 | 1.0033 |
| APOE | 3.0168 | 3.3379 | 3.0770 |
| TMSB4X | 3.9916 | 3.7039 | 3.9882 |
| SAMSN1 | 0.4144 | 0.6936 | 0.3321 |
| NANS | 0.9681 | 1.2439 | 1.1799 |

`ATP5E_ATP5E_-_57607036.23-P1P2_04`（sgRNA `GGTGTCCAGGAGCACTCTGT`，cells=20）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ATP5E | 3.1685 | 2.0048 | 3.1025 |
| HBG1 | 0.9801 | 0.6778 | 1.0008 |
| TRIB3 | 0.4538 | 0.7385 | 0.4845 |
| HNRNPH1 | 1.2575 | 0.9931 | 1.3207 |
| ACTG1 | 2.2437 | 2.4985 | 2.3135 |
| ATF4 | 1.5476 | 1.7968 | 1.5348 |

`ATP5E_ATP5E_-_57607036.23-P1P2_14`（sgRNA `GGTGTCCAGGGGCACTGTGT`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ATP5E | 3.1685 | 1.7476 | 3.0873 |
| POLR2H | 1.2807 | 0.9207 | 1.5263 |
| HSPA5 | 0.7632 | 0.4428 | 0.7767 |
| CCNB1 | 0.7544 | 0.4393 | 0.6661 |
| HMGCS1 | 0.4739 | 0.1822 | 0.4266 |
| CRELD2 | 0.4823 | 0.7540 | 0.4639 |

`ATP5E_ATP5E_-_57607036.23-P1P2_16`（sgRNA `GGTGTCCAGGGGCGCTCTGT`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ATP5E | 3.1685 | 1.5329 | 3.1057 |
| HBZ | 2.7562 | 2.2244 | 2.8249 |
| RP11-301G19.1 | 1.7747 | 1.3421 | 1.8649 |
| SLC25A37 | 0.8188 | 0.4515 | 0.7868 |
| HBA1 | 1.0624 | 0.7022 | 1.0805 |
| KPNA2 | 0.8283 | 1.1759 | 0.9068 |

**BCR**

`BCR_BCR_+_23523092.23-P1P2_00`（sgRNA `GCGCGCGGGGCCCGTCTCAG`，cells=13）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 5.5147 | 2.8235 |
| ALAS2 | 0.2129 | 2.1752 | 0.1659 |
| BLVRB | 1.1025 | 2.8101 | 1.2033 |
| SLC25A37 | 0.8188 | 2.2822 | 0.7798 |
| HBG2 | 3.1952 | 4.5309 | 3.2701 |
| CHI3L2 | 0.8015 | 2.0356 | 0.7728 |

`BCR_BCR_+_23523092.23-P1P2_04`（sgRNA `GCGCGCGGAGCCCGTCTCAG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 5.2668 | 2.8232 |
| SLC25A37 | 0.8188 | 2.8845 | 0.7696 |
| HBG2 | 3.1952 | 5.1209 | 3.2586 |
| ALAS2 | 0.2129 | 2.1056 | 0.1683 |
| BLVRB | 1.1025 | 2.5064 | 1.2105 |
| HBA1 | 1.0624 | 2.2196 | 1.0844 |

`BCR_BCR_+_23523092.23-P1P2_05`（sgRNA `GCGCGCGGCGCCCGTCTCAG`，cells=21）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 3.8013 | 2.8285 |
| HBG2 | 3.1952 | 3.9710 | 3.2727 |
| BLVRB | 1.1025 | 1.6749 | 1.2130 |
| ALAS2 | 0.2129 | 0.7383 | 0.1726 |
| MYL4 | 1.3474 | 1.8060 | 1.4226 |
| HBG1 | 0.9801 | 1.4146 | 1.0023 |

`BCR_BCR_+_23523092.23-P1P2_07`（sgRNA `GCGCGCGGGGCTCGTCTCAG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 5.5255 | 2.8341 |
| ALAS2 | 0.2129 | 2.2596 | 0.1714 |
| SLC25A37 | 0.8188 | 2.7237 | 0.7882 |
| HBG2 | 3.1952 | 4.8717 | 3.2719 |
| BLVRB | 1.1025 | 2.5693 | 1.2257 |
| GYPB | 1.5517 | 2.8936 | 1.6353 |

`BCR_BCR_+_23523092.23-P1P2_13`（sgRNA `GCGCGCGGGGCCCATCTCAG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBG2 | 3.1952 | 4.1952 | 3.2600 |
| HBZ | 2.7562 | 3.6551 | 2.8213 |
| BLVRB | 1.1025 | 1.7456 | 1.2271 |
| CHI3L2 | 0.8015 | 1.2893 | 0.7740 |
| HBG1 | 0.9801 | 1.4521 | 0.9941 |
| COL18A1 | 0.5658 | 1.0056 | 0.5729 |

`BCR_BCR_+_23523092.23-P1P2_15`（sgRNA `GCGCGCGGGGCCCGTCGCAG`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 3.5668 | 2.8243 |
| HBG2 | 3.1952 | 3.8961 | 3.2587 |
| BLVRB | 1.1025 | 1.5871 | 1.2179 |
| HBG1 | 0.9801 | 1.4007 | 0.9965 |
| GYPB | 1.5517 | 1.9642 | 1.6286 |
| CENPF | 0.9886 | 1.3689 | 1.0523 |

**CAD**

`CAD_CAD_+_27440280.23-P1P2_00`（sgRNA `GGCTGGAGAGAAGCCGGGCG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 4.1445 | 2.8167 |
| BLVRB | 1.1025 | 2.0865 | 1.2282 |
| HBG2 | 3.1952 | 4.1114 | 3.2582 |
| AC016629.3 | 0.0803 | 0.9166 | 0.0654 |
| HIST1H1C | 2.5286 | 3.3547 | 2.5560 |
| KLF1 | 0.6525 | 1.4670 | 0.7385 |

`CAD_CAD_+_27440280.23-P1P2_03`（sgRNA `GGCTGGTGAGAAGCCGGGCG`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 4.9791 | 2.8299 |
| BLVRB | 1.1025 | 2.5652 | 1.2167 |
| HBG2 | 3.1952 | 4.3796 | 3.2648 |
| ALAS2 | 0.2129 | 1.3365 | 0.1603 |
| HBA1 | 1.0624 | 2.1183 | 1.0867 |
| FAM178B | 0.7202 | 1.7307 | 0.8389 |

`CAD_CAD_+_27440280.23-P1P2_06`（sgRNA `GGCTGGAGAGTAGCCGGGCG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| BLVRB | 1.1025 | 1.6476 | 1.2344 |
| HBZ | 2.7562 | 3.2715 | 2.8180 |
| TXNIP | 0.4501 | 0.8962 | 0.4117 |
| HSPA8 | 2.3720 | 1.9433 | 2.4627 |
| CARHSP1 | 0.8657 | 1.2938 | 0.7957 |
| KLF1 | 0.6525 | 1.0692 | 0.7388 |

`CAD_CAD_+_27440280.23-P1P2_07`（sgRNA `GGCTGGAGCGAAGCCGGGCG`，cells=28）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 4.0148 | 2.8283 |
| BLVRB | 1.1025 | 1.9061 | 1.2161 |
| TXNIP | 0.4501 | 1.0452 | 0.4257 |
| AC016629.3 | 0.0803 | 0.6563 | 0.0707 |
| NUCKS1 | 2.1284 | 1.5574 | 2.0518 |
| CARHSP1 | 0.8657 | 1.4236 | 0.8136 |

`CAD_CAD_+_27440280.23-P1P2_13`（sgRNA `GGCTGGAGAGAAGCCTGGCG`，cells=16）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| PVT1 | 0.2119 | 0.5140 | 0.2651 |
| PRSS57 | 0.9886 | 0.6946 | 1.0485 |
| GDF15 | 0.2748 | 0.5451 | 0.2750 |
| TIMP1 | 0.8194 | 1.0852 | 0.7866 |
| CENPA | 0.5214 | 0.2611 | 0.4658 |
| FCER1G | 0.4132 | 0.6682 | 0.3822 |

**CDC23**

`CDC23_CDC23_-_137548987.23-P1P2_00`（sgRNA `GACAGCCACCGGGACCATGG`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H4C | 2.8420 | 2.1525 | 3.0310 |
| GAL | 1.8857 | 1.2715 | 2.0622 |
| LGALS1 | 0.7898 | 1.3790 | 0.7823 |
| LAPTM5 | 1.0762 | 1.6173 | 1.0339 |
| KRT18 | 1.5131 | 1.9655 | 1.5684 |
| HIST1H2BJ | 1.1310 | 1.5388 | 1.1939 |

`CDC23_CDC23_-_137548987.23-P1P2_02`（sgRNA `GACAGCTACCGGGACCATGG`，cells=7）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| IER2 | 0.6795 | 0.1422 | 0.6444 |
| SQLE | 1.1485 | 1.5752 | 1.1419 |
| BTG1 | 1.3322 | 1.7502 | 1.2942 |
| TUBB | 2.5379 | 2.1324 | 2.6580 |
| CENPF | 0.9886 | 1.3819 | 1.0472 |
| RP11-717F1.1 | 1.3860 | 1.7646 | 1.4830 |

`CDC23_CDC23_-_137548987.23-P1P2_04`（sgRNA `GACAGCCAACGGGACCATGG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H4C | 2.8420 | 1.2149 | 3.0357 |
| CCNB2 | 0.5547 | 1.2574 | 0.5503 |
| RP11-301G19.1 | 1.7747 | 1.2309 | 1.8643 |
| CITED2 | 0.8732 | 0.3989 | 0.9659 |
| AURKA | 0.5848 | 1.0463 | 0.6145 |
| MKI67 | 0.7211 | 1.1810 | 0.6557 |

`CDC23_CDC23_-_137548987.23-P1P2_08`（sgRNA `GACAGCCATCGGGACCATGG`，cells=17）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H1C | 2.5286 | 3.0942 | 2.5530 |
| HIST1H2BJ | 1.1310 | 1.5769 | 1.1804 |
| HBG2 | 3.1952 | 3.6219 | 3.2617 |
| HIST1H4C | 2.8420 | 2.4337 | 3.0206 |
| HBG1 | 0.9801 | 1.3543 | 1.0009 |
| HBZ | 2.7562 | 3.1219 | 2.8182 |

`CDC23_CDC23_-_137548987.23-P1P2_11`（sgRNA `GACAGCCACCGGGACCACGG`，cells=24）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H1C | 2.5286 | 2.1965 | 2.5480 |
| HIST1H2BJ | 1.1310 | 0.8653 | 1.1711 |
| CYBA | 1.7294 | 1.4666 | 1.7486 |
| CDC23 | 0.3030 | 0.0406 | 0.3097 |
| UBC | 1.3914 | 1.1315 | 1.4523 |
| FAM178B | 0.7202 | 0.4643 | 0.8306 |

**COX11**

`COX11_COX11_+_53045977.23-P1P2_00`（sgRNA `GGCTCTGGCGTCCTGGATGG`，cells=19）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 2.3245 | 2.8110 |
| TSC22D1 | 0.4523 | 0.8525 | 0.5017 |
| HIST1H2BJ | 1.1310 | 0.7850 | 1.1839 |
| VIM | 1.9881 | 2.2940 | 2.0203 |
| COX11 | 0.4480 | 0.1478 | 0.5326 |
| UBC | 1.3914 | 1.6863 | 1.4361 |

`COX11_COX11_+_53045977.23-P1P2_03`（sgRNA `GGCTCTGTCGTCCTGGATGG`，cells=30）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| COX11 | 0.4480 | 0.0646 | 0.5555 |
| HIST1H2BJ | 1.1310 | 0.8661 | 1.1698 |
| HSPA1A | 0.4833 | 0.2436 | 0.5305 |
| TXNIP | 0.4501 | 0.2278 | 0.4266 |
| CYBA | 1.7294 | 1.9445 | 1.7393 |
| PDIA6 | 1.3895 | 1.5799 | 1.5434 |

`COX11_COX11_+_53045977.23-P1P2_04`（sgRNA `GGCTCTGGCGCCCTGGATGG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBG2 | 3.1952 | 2.6515 | 3.2634 |
| HBZ | 2.7562 | 2.2773 | 2.8172 |
| KLF1 | 0.6525 | 0.1855 | 0.7304 |
| HIST1H4C | 2.8420 | 2.4466 | 3.0270 |
| MS4A3 | 0.9113 | 0.5385 | 0.8791 |
| HSP90B1 | 1.4657 | 1.1137 | 1.4631 |

`COX11_COX11_+_53045977.23-P1P2_05`（sgRNA `GGCTCTGGCGTCTTGGATGG`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| MYL4 | 1.3474 | 1.8423 | 1.4176 |
| HIST3H2A | 0.6953 | 1.0587 | 0.6843 |
| KLF1 | 0.6525 | 0.2915 | 0.7334 |
| HIST1H2BJ | 1.1310 | 1.4824 | 1.1756 |
| MYC | 1.1698 | 1.5208 | 1.1915 |
| HIST1H2AC | 1.0096 | 1.3434 | 1.0219 |

`COX11_COX11_+_53045977.23-P1P2_10`（sgRNA `GGCTCTGGCGTCCCGGATGG`，cells=4）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 1.1426 | 2.8029 |
| HIST1H1C | 2.5286 | 1.6839 | 2.5452 |
| HBG2 | 3.1952 | 2.4129 | 3.2601 |
| PHF19 | 1.8021 | 1.1941 | 1.7678 |
| HBD | 0.5985 | 0.0000 | 0.6315 |
| AURKA | 0.5848 | 1.1543 | 0.5993 |

**DBR1**

`DBR1_DBR1_+_137893744.23-P1P2_00`（sgRNA `GTTTGCAGGAGTCTACACCC`，cells=30）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| ATF7IP2 | 1.1485 | 1.4979 | 1.2396 |
| HBA1 | 1.0624 | 0.8032 | 1.0885 |
| MARCKSL1 | 2.1143 | 1.8878 | 2.1989 |
| SLC20A1 | 0.4134 | 0.6360 | 0.3880 |
| XBP1 | 0.4939 | 0.7043 | 0.3889 |
| HIST3H2A | 0.6953 | 0.9057 | 0.6990 |

`DBR1_DBR1_+_137893744.23-P1P2_01`（sgRNA `GATTGCAGGAGTCTACACCC`，cells=21）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBA1 | 1.0624 | 0.4295 | 1.0851 |
| ATF7IP2 | 1.1485 | 1.4907 | 1.2380 |
| RP11-717F1.1 | 1.3860 | 1.0607 | 1.4905 |
| SQLE | 1.1485 | 0.8246 | 1.1434 |
| MARCKSL1 | 2.1143 | 1.7968 | 2.1880 |
| HBZ | 2.7562 | 2.4578 | 2.8195 |

`DBR1_DBR1_+_137893744.23-P1P2_05`（sgRNA `GTTTGCAGGAGTGTACACCC`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H4C | 2.8420 | 2.3381 | 3.0226 |
| HBA1 | 1.0624 | 0.5653 | 1.0810 |
| HBZ | 2.7562 | 2.3722 | 2.8231 |
| TIMP1 | 0.8194 | 0.4460 | 0.7714 |
| HBG2 | 3.1952 | 2.8337 | 3.2558 |
| CHI3L2 | 0.8015 | 0.4516 | 0.7657 |

`DBR1_DBR1_+_137893744.23-P1P2_07`（sgRNA `GTTTGCAGGGGTCTACACCC`，cells=18）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBG2 | 3.1952 | 2.5809 | 3.2538 |
| ATF7IP2 | 1.1485 | 1.6982 | 1.2349 |
| HBZ | 2.7562 | 2.2680 | 2.8140 |
| GYPB | 1.5517 | 1.2046 | 1.6210 |
| SLC25A37 | 0.8188 | 0.4718 | 0.7821 |
| MS4A3 | 0.9113 | 0.5673 | 0.8899 |

`DBR1_DBR1_+_137893744.23-P1P2_08`（sgRNA `GTTTGCAGTAGTCTACACCC`，cells=14）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| GYPB | 1.5517 | 0.9800 | 1.6221 |
| AURKB | 0.5029 | 0.9253 | 0.4985 |
| AURKA | 0.5848 | 1.0057 | 0.5999 |
| CHI3L2 | 0.8015 | 0.4001 | 0.7739 |
| PIF1 | 0.2967 | 0.6897 | 0.2631 |
| MDK | 1.6126 | 1.2299 | 1.5968 |

**DUT**

`DUT_DUT_+_48624411.23-P1P2_00`（sgRNA `GCGAGCGAGGAGACCACCGG`，cells=22）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DUT | 1.6702 | 0.5942 | 1.7551 |
| HBG1 | 0.9801 | 1.4004 | 1.0001 |
| NEAT1 | 1.4726 | 1.8628 | 1.4672 |
| ACTG1 | 2.2437 | 2.6303 | 2.2971 |
| MYC | 1.1698 | 0.8228 | 1.1856 |
| HBZ | 2.7562 | 3.0846 | 2.8128 |

`DUT_DUT_+_48624411.23-P1P2_01`（sgRNA `GCCAGCGAGGAGACCACCGG`，cells=32）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DUT | 1.6702 | 0.4793 | 1.7562 |
| HIST3H2A | 0.6953 | 1.0751 | 0.6985 |
| ACTG1 | 2.2437 | 2.6010 | 2.3004 |
| HBZ | 2.7562 | 3.0701 | 2.8193 |
| RP11-717F1.1 | 1.3860 | 1.6941 | 1.4850 |
| HIST1H4C | 2.8420 | 3.1413 | 3.0359 |

`DUT_DUT_+_48624411.23-P1P2_07`（sgRNA `GCGAGCGAGGAGCCCACCGG`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DUT | 1.6702 | 0.4142 | 1.7629 |
| HBZ | 2.7562 | 3.3692 | 2.8286 |
| HIST3H2A | 0.6953 | 1.2425 | 0.6962 |
| BLVRB | 1.1025 | 1.6258 | 1.2047 |
| NCL | 2.4893 | 1.9795 | 2.7509 |
| HNRNPH1 | 1.2575 | 0.7546 | 1.3179 |

`DUT_DUT_+_48624411.23-P1P2_08`（sgRNA `GCGAGCGAGGAGGCCACCGG`，cells=30）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DUT | 1.6702 | 0.4050 | 1.7505 |
| ACTG1 | 2.2437 | 2.6519 | 2.3061 |
| DNAJA1 | 1.4641 | 1.1138 | 1.5942 |
| NEAT1 | 1.4726 | 1.8224 | 1.4671 |
| KIAA0101 | 1.8159 | 2.1478 | 1.8443 |
| UBC | 1.3914 | 1.0709 | 1.4263 |

`DUT_DUT_+_48624411.23-P1P2_10`（sgRNA `GCGAGCGAGGAGACCAACGG`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DUT | 1.6702 | 0.5887 | 1.7528 |
| CCNB1 | 0.7544 | 0.4045 | 0.6891 |
| TUBB2A | 0.2089 | 0.5106 | 0.1988 |
| TUBB | 2.5379 | 2.2422 | 2.6567 |
| HBD | 0.5985 | 0.8831 | 0.6346 |
| TMSB10 | 1.8482 | 2.1318 | 1.7645 |

**EIF2S1**

`EIF2S1_EIF2S1_-_67827080.23-P1P2_00`（sgRNA `GAGCGAAGCGCACGCTGAGG`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DDIT3 | 0.2305 | 1.6044 | 0.2670 |
| DDIT4 | 0.1407 | 1.4276 | 0.1190 |
| TRIB3 | 0.4538 | 1.6008 | 0.4603 |
| HBG2 | 3.1952 | 2.0903 | 3.2655 |
| GAL | 1.8857 | 0.7818 | 2.0645 |
| CCPG1 | 0.2294 | 1.3186 | 0.2711 |

`EIF2S1_EIF2S1_-_67827080.23-P1P2_01`（sgRNA `GAGCGAAACGCACGCTGAGG`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| IGFBP2 | 0.2020 | 0.9351 | 0.0778 |
| HBZ | 2.7562 | 2.0348 | 2.8189 |
| DDIT4 | 0.1407 | 0.8345 | 0.1197 |
| HBG2 | 3.1952 | 2.5693 | 3.2587 |
| PRSS57 | 0.9886 | 1.5211 | 1.0488 |
| TRIB3 | 0.4538 | 0.9789 | 0.4909 |

`EIF2S1_EIF2S1_-_67827080.23-P1P2_02`（sgRNA `GAGCGCAGCGCACGCTGAGG`，cells=14）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DDIT4 | 0.1407 | 1.4114 | 0.1237 |
| IGFBP2 | 0.2020 | 1.4530 | 0.0869 |
| TRIB3 | 0.4538 | 1.5996 | 0.4681 |
| HBZ | 2.7562 | 1.6616 | 2.8187 |
| DDIT3 | 0.2305 | 1.2515 | 0.2581 |
| PHGDH | 0.2620 | 1.1880 | 0.2033 |

`EIF2S1_EIF2S1_-_67827080.23-P1P2_06`（sgRNA `GAGCGAAGCGCGCGCTGAGG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 1.5838 | 2.8183 |
| HBG2 | 3.1952 | 2.0403 | 3.2606 |
| DDIT4 | 0.1407 | 1.2328 | 0.1141 |
| HIST1H1C | 2.5286 | 1.6258 | 2.5618 |
| CCPG1 | 0.2294 | 1.0748 | 0.2680 |
| MAP1B | 0.3669 | 1.1862 | 0.4550 |

`EIF2S1_EIF2S1_-_67827080.23-P1P2_07`（sgRNA `GAGCGAAGCGCTCGCTGAGG`，cells=6）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H1C | 2.5286 | 3.2050 | 2.5616 |
| HSP90AA1 | 2.6816 | 3.0939 | 2.7940 |
| SLC35B1 | 0.5832 | 0.9688 | 0.5406 |
| EIF2S1 | 0.7331 | 1.1120 | 0.7660 |
| PLD3 | 0.6038 | 0.9789 | 0.5523 |
| FCER1G | 0.4132 | 0.0572 | 0.3655 |

**GATA1**

`GATA1_GATA1_-_48645022.23-P1P2_00`（sgRNA `GTGAGCTTGCCACATCCCCA`，cells=3）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBG2 | 3.1952 | 0.9245 | 3.2729 |
| TYROBP | 0.0678 | 1.8314 | 0.1145 |
| ATF7IP2 | 1.1485 | 2.8289 | 1.2372 |
| AIF1 | 0.2036 | 1.8746 | 0.2048 |
| LGALS1 | 0.7898 | 2.3887 | 0.7870 |
| SH3BGRL3 | 0.8037 | 2.3859 | 0.8089 |

`GATA1_GATA1_-_48645022.23-P1P2_03`（sgRNA `GTGCGCTTGCCACATCCCCA`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| LGALS1 | 0.7898 | 2.4084 | 0.7898 |
| CTSC | 0.9578 | 2.5630 | 1.0056 |
| S100A4 | 0.2834 | 1.8156 | 0.3457 |
| LTB | 0.0642 | 1.5406 | 0.1267 |
| JCHAIN | 0.0051 | 1.4590 | 0.0414 |
| GAL | 1.8857 | 0.4781 | 2.0639 |

`GATA1_GATA1_-_48645022.23-P1P2_04`（sgRNA `GTGAGCTTACCACATCCCCA`，cells=6）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TYROBP | 0.0678 | 2.1891 | 0.1195 |
| LST1 | 0.1161 | 2.0937 | 0.0883 |
| LGALS1 | 0.7898 | 2.7654 | 0.7830 |
| AIF1 | 0.2036 | 2.1379 | 0.2136 |
| LTB | 0.0642 | 1.8052 | 0.1314 |
| S100A4 | 0.2834 | 2.0162 | 0.3510 |

`GATA1_GATA1_-_48645022.23-P1P2_06`（sgRNA `GTGAGCTTGCGACATCCCCA`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| LTB | 0.0642 | 1.6363 | 0.1304 |
| LGALS1 | 0.7898 | 2.2392 | 0.7900 |
| HBZ | 2.7562 | 1.4852 | 2.8239 |
| FCER1G | 0.4132 | 1.5872 | 0.3686 |
| LST1 | 0.1161 | 1.2794 | 0.0857 |
| S100A4 | 0.2834 | 1.4237 | 0.3419 |

`GATA1_GATA1_-_48645022.23-P1P2_08`（sgRNA `GTGAGCTTTCCACATCCCCA`，cells=5）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| LGALS1 | 0.7898 | 2.5329 | 0.7922 |
| CTSC | 0.9578 | 2.6280 | 1.0131 |
| S100A4 | 0.2834 | 1.7166 | 0.3452 |
| LST1 | 0.1161 | 1.5446 | 0.0854 |
| ID3 | 0.0904 | 1.4924 | 0.1303 |
| SOCS1 | 0.1827 | 1.4194 | 0.1773 |

`GATA1_GATA1_-_48645022.23-P1P2_12`（sgRNA `GTGAGCTTGCCACATCCGCA`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| MAP1B | 0.3669 | 0.8598 | 0.4549 |
| CHI3L2 | 0.8015 | 1.2455 | 0.7760 |
| HBG1 | 0.9801 | 1.4061 | 1.0227 |
| CDC42SE1 | 0.4550 | 0.8791 | 0.4477 |
| PTTG1 | 1.5473 | 1.1446 | 1.6006 |
| KRT8 | 0.6739 | 1.0576 | 0.6493 |

**GINS1**

`GINS1_GINS1_-_25388381.23-P1P2_00`（sgRNA `GGACTAGAACGAAAGGAGTG`，cells=3）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| CENPF | 0.9886 | 1.8273 | 1.0522 |
| HBG2 | 3.1952 | 4.0125 | 3.2653 |
| HBG1 | 0.9801 | 1.6916 | 1.0027 |
| TOP2A | 0.7916 | 1.4950 | 0.7559 |
| NPW | 0.7693 | 1.4578 | 0.7631 |
| CD7 | 0.3776 | 1.0545 | 0.4761 |

`GINS1_GINS1_-_25388381.23-P1P2_03`（sgRNA `GGACTATAACGAAAGGAGTG`，cells=15）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| NEAT1 | 1.4726 | 2.1952 | 1.4679 |
| VIM | 1.9881 | 2.5159 | 1.9952 |
| BTG1 | 1.3322 | 1.8538 | 1.2918 |
| LGALS1 | 0.7898 | 1.2739 | 0.7798 |
| FAM178B | 0.7202 | 1.2004 | 0.8342 |
| ACTG1 | 2.2437 | 2.7188 | 2.3116 |

`GINS1_GINS1_-_25388381.23-P1P2_06`（sgRNA `GGACTAGAACGGAAGGAGTG`，cells=16）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TOP2A | 0.7916 | 1.3240 | 0.7478 |
| CHI3L2 | 0.8015 | 0.3894 | 0.7732 |
| HBG2 | 3.1952 | 3.5955 | 3.2706 |
| UBE2S | 2.0784 | 2.4771 | 2.1317 |
| HIST1H2BJ | 1.1310 | 1.5165 | 1.1851 |
| PLK1 | 0.5970 | 0.9818 | 0.5674 |

`GINS1_GINS1_-_25388381.23-P1P2_08`（sgRNA `GGACTAGAGCGAAAGGAGTG`，cells=25）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 3.2590 | 2.8169 |
| MYL4 | 1.3474 | 1.8312 | 1.4334 |
| TOP2A | 0.7916 | 1.2614 | 0.7583 |
| PIF1 | 0.2967 | 0.7298 | 0.2600 |
| IDH2 | 0.8453 | 1.2749 | 0.8667 |
| SLC25A37 | 0.8188 | 1.2313 | 0.7794 |

`GINS1_GINS1_-_25388381.23-P1P2_14`（sgRNA `GGACTAGAACGAAAGGAGCG`，cells=16）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBG2 | 3.1952 | 3.7451 | 3.2646 |
| HIST1H2BJ | 1.1310 | 1.5773 | 1.1811 |
| CHI3L2 | 0.8015 | 0.3993 | 0.7741 |
| MYL4 | 1.3474 | 1.7338 | 1.4262 |
| RP11-717F1.1 | 1.3860 | 1.7432 | 1.4902 |
| HIST1H2BD | 0.3383 | 0.6693 | 0.2918 |

**GNB2L1**

`GNB2L1_GNB2L1_+_180670873.23-P1P2_02`（sgRNA `GTGCAAGACGGCGGCAGGAG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| GNB2L1 | 3.8848 | 2.8213 | 3.9477 |
| BST2 | 0.2795 | 0.7451 | 0.1962 |
| TPX2 | 0.8295 | 0.3855 | 0.8184 |
| HBG2 | 3.1952 | 2.7591 | 3.2572 |
| HIST2H2AC | 0.4454 | 0.8279 | 0.3940 |
| VIMP | 0.6712 | 1.0517 | 0.7374 |

`GNB2L1_GNB2L1_+_180670873.23-P1P2_07`（sgRNA `GTGCAAGGCGGGGGCAGGAG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| GNB2L1 | 3.8848 | 2.5296 | 3.9552 |
| RP11-717F1.1 | 1.3860 | 0.9657 | 1.4877 |
| CLCA1 | 0.0984 | 0.4715 | 0.0860 |
| EIF3L | 1.8948 | 1.5325 | 1.9719 |
| HNRNPH1 | 1.2575 | 0.9027 | 1.3218 |
| MS4A4A | 0.6556 | 1.0099 | 0.7082 |

`GNB2L1_GNB2L1_+_180670873.23-P1P2_08`（sgRNA `GTGCAAGGTGGCGGCAGGAG`，cells=58）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| GNB2L1 | 3.8848 | 2.0026 | 3.9527 |
| HBZ | 2.7562 | 2.0662 | 2.8139 |
| EIF3L | 1.8948 | 1.4047 | 1.9758 |
| SOCS2 | 1.0925 | 1.4648 | 1.1066 |
| RP11-717F1.1 | 1.3860 | 1.0244 | 1.4864 |
| C6orf48 | 1.9819 | 1.6373 | 2.0910 |

`GNB2L1_GNB2L1_+_180670873.23-P1P2_13`（sgRNA `GTGCAAGGCGGCGGCGGGAG`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| GNB2L1 | 3.8848 | 3.1997 | 3.9539 |
| IER2 | 0.6795 | 1.1030 | 0.6544 |
| UBC | 1.3914 | 1.7664 | 1.4341 |
| DDIT3 | 0.2305 | 0.5899 | 0.2549 |
| CXCL8 | 0.0058 | 0.3641 | 0.0028 |
| CENPF | 0.9886 | 1.3274 | 1.0519 |

**HSPA5**

`HSPA5_HSPA5_+_128003624.23-P1P2_00`（sgRNA `GAGCCGAGTAGGCGACGGTG`，cells=6）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| SDF2L1 | 2.7222 | 4.7418 | 2.8121 |
| MANF | 0.6586 | 2.6574 | 0.7388 |
| HSP90B1 | 1.4657 | 3.3069 | 1.4894 |
| HBG2 | 3.1952 | 1.5187 | 3.2756 |
| DDIT3 | 0.2305 | 1.8082 | 0.2581 |
| HERPUD1 | 0.2392 | 1.6700 | 0.1825 |

`HSPA5_HSPA5_+_128003624.23-P1P2_01`（sgRNA `GAACCGAGTAGGCGACGGTG`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HSP90B1 | 1.4657 | 3.6667 | 1.4746 |
| SDF2L1 | 2.7222 | 4.6395 | 2.8136 |
| MANF | 0.6586 | 2.5058 | 0.7349 |
| PDIA4 | 0.5018 | 2.2204 | 0.5256 |
| DDIT3 | 0.2305 | 1.8000 | 0.2590 |
| PDIA3 | 0.6189 | 2.0919 | 0.5821 |

`HSPA5_HSPA5_+_128003624.23-P1P2_04`（sgRNA `GAGCCGAGAAGGCGACGGTG`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| SDF2L1 | 2.7222 | 4.6532 | 2.8236 |
| MANF | 0.6586 | 2.5095 | 0.7421 |
| PDIA4 | 0.5018 | 2.1616 | 0.5169 |
| HSP90B1 | 1.4657 | 3.0927 | 1.4727 |
| HERPUD1 | 0.2392 | 1.5304 | 0.1835 |
| DDIT3 | 0.2305 | 1.4623 | 0.2528 |

`HSPA5_HSPA5_+_128003624.23-P1P2_06`（sgRNA `GAGCCGAGTAGACGACGGTG`，cells=22）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| MANF | 0.6586 | 1.6528 | 0.7425 |
| SDF2L1 | 2.7222 | 3.7086 | 2.8231 |
| HSP90B1 | 1.4657 | 2.4491 | 1.4750 |
| PDIA6 | 1.3895 | 2.1620 | 1.5511 |
| PDIA4 | 0.5018 | 1.1635 | 0.5145 |
| DNAJB11 | 0.5149 | 1.1412 | 0.6013 |

`HSPA5_HSPA5_+_128003624.23-P1P2_08`（sgRNA `GAGCCGAGTGGGCGACGGTG`，cells=16）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| SDF2L1 | 2.7222 | 4.6846 | 2.8191 |
| MANF | 0.6586 | 2.5781 | 0.7330 |
| HSP90B1 | 1.4657 | 3.3266 | 1.4898 |
| PDIA4 | 0.5018 | 1.9517 | 0.5096 |
| CRELD2 | 0.4823 | 1.8768 | 0.4527 |
| DDIT3 | 0.2305 | 1.6096 | 0.2660 |

**HSPA9**

`HSPA9_HSPA9_-_137911079.23-P1P2_00`（sgRNA `GGAGCTGCGCGATGCGGTGG`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HSPA9 | 1.3504 | 0.4088 | 1.5227 |
| LGALS1 | 0.7898 | 1.3442 | 0.7726 |
| SH3BGRL3 | 0.8037 | 1.3091 | 0.7844 |
| DDIT4 | 0.1407 | 0.6392 | 0.1214 |
| HBA1 | 1.0624 | 0.5961 | 1.0956 |
| TSC22D1 | 0.4523 | 0.8946 | 0.5058 |

`HSPA9_HSPA9_-_137911079.23-P1P2_02`（sgRNA `GGAGTTGCGCGATGCGGTGG`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TRIB3 | 0.4538 | 1.4385 | 0.4665 |
| HSPA9 | 1.3504 | 0.4740 | 1.5184 |
| DDIT4 | 0.1407 | 0.9224 | 0.1150 |
| PHGDH | 0.2620 | 1.0355 | 0.2082 |
| SERPINF1 | 0.3973 | 1.0604 | 0.4505 |
| DDIT3 | 0.2305 | 0.8801 | 0.2466 |

`HSPA9_HSPA9_-_137911079.23-P1P2_04`（sgRNA `GGAGCTGCGCAATGCGGTGG`，cells=13）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| CTSC | 0.9578 | 1.3536 | 0.9995 |
| HIST1H1C | 2.5286 | 2.2296 | 2.5646 |
| CARHSP1 | 0.8657 | 0.5782 | 0.8137 |
| HSPA9 | 1.3504 | 1.0665 | 1.5132 |
| CFD | 1.0335 | 1.3126 | 1.1674 |
| ALB | 0.0180 | 0.2884 | 0.0388 |

`HSPA9_HSPA9_-_137911079.23-P1P2_07`（sgRNA `GGAGCTGCGGGATGCGGTGG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBG2 | 3.1952 | 1.7219 | 3.2621 |
| HBZ | 2.7562 | 1.6447 | 2.8194 |
| DDIT4 | 0.1407 | 1.1334 | 0.1221 |
| HIST1H1C | 2.5286 | 1.6199 | 2.5681 |
| TRIB3 | 0.4538 | 1.2781 | 0.4479 |
| PHGDH | 0.2620 | 1.0631 | 0.2006 |

`HSPA9_HSPA9_-_137911079.23-P1P2_08`（sgRNA `GGAGCTGCTCGATGCGGTGG`，cells=5）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DDIT4 | 0.1407 | 0.8560 | 0.1219 |
| APOE | 3.0168 | 2.4014 | 3.0724 |
| MYL4 | 1.3474 | 0.7668 | 1.4394 |
| IGFBP2 | 0.2020 | 0.7692 | 0.1011 |
| PHGDH | 0.2620 | 0.8099 | 0.2113 |
| POLR2H | 1.2807 | 0.7330 | 1.5230 |

**HSPE1**

`HSPE1_HSPE1_+_198365089.23-P1P2_00`（sgRNA `GGAGACTCGCAGTCCGGCCC`，cells=19）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HSPE1 | 3.3758 | 1.6609 | 3.4305 |
| DDIT4 | 0.1407 | 0.9637 | 0.1184 |
| HBG2 | 3.1952 | 2.5368 | 3.2609 |
| HNRNPH1 | 1.2575 | 0.7872 | 1.3213 |
| IGFBP2 | 0.2020 | 0.6685 | 0.0892 |
| PHGDH | 0.2620 | 0.6401 | 0.2077 |

`HSPE1_HSPE1_+_198365089.23-P1P2_01`（sgRNA `GGAGACACGCAGTCCGGCCC`，cells=22）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HSPE1 | 3.3758 | 1.5030 | 3.4295 |
| DDIT4 | 0.1407 | 0.6558 | 0.1198 |
| PIM1 | 0.5387 | 0.9422 | 0.4719 |
| HIST1H4C | 2.8420 | 3.2270 | 3.0303 |
| WARS | 0.3991 | 0.7512 | 0.4282 |
| SDF2L1 | 2.7222 | 2.3783 | 2.8064 |

`HSPE1_HSPE1_+_198365089.23-P1P2_02`（sgRNA `GGAGACTGGCAGTCCGGCCC`，cells=17）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HSPE1 | 3.3758 | 1.7651 | 3.4303 |
| HSPA9 | 1.3504 | 1.7378 | 1.5179 |
| HIST1H1C | 2.5286 | 2.1751 | 2.5513 |
| SQLE | 1.1485 | 0.8507 | 1.1483 |
| CTNNB1 | 0.5801 | 0.2969 | 0.4522 |
| HNRNPH1 | 1.2575 | 0.9790 | 1.3248 |

`HSPE1_HSPE1_+_198365089.23-P1P2_03`（sgRNA `GGTGACTCGCAGTCCGGCCC`，cells=15）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HSPE1 | 3.3758 | 1.7640 | 3.4294 |
| HBA1 | 1.0624 | 0.4592 | 1.0885 |
| DDIT4 | 0.1407 | 0.6977 | 0.1191 |
| TRIB3 | 0.4538 | 0.9912 | 0.4706 |
| C6orf48 | 1.9819 | 2.4226 | 2.0965 |
| CHI3L2 | 0.8015 | 0.3694 | 0.7814 |

`HSPE1_HSPE1_+_198365089.23-P1P2_14`（sgRNA `GGAGACTCGCAGTCCTGCCC`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HSPE1 | 3.3758 | 2.5448 | 3.4326 |
| ATF7IP2 | 1.1485 | 0.7114 | 1.2356 |
| CKS2 | 1.7367 | 1.4035 | 1.8221 |
| HIST1H1C | 2.5286 | 2.8358 | 2.5599 |
| GAL | 1.8857 | 2.1799 | 2.0423 |
| AURKA | 0.5848 | 0.3007 | 0.5892 |

**MTOR**

`MTOR_MTOR_+_11322547.23-P1P2_00`（sgRNA `GGGCAGGGGGCCTGAAGCGG`，cells=39）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 3.2234 | 2.8243 |
| PRSS57 | 0.9886 | 1.3920 | 1.0500 |
| EIF3L | 1.8948 | 2.2836 | 1.9735 |
| CHI3L2 | 0.8015 | 1.1834 | 0.7730 |
| TRAPPC6A | 0.5362 | 0.9175 | 0.4827 |
| TMSB4X | 3.9916 | 3.6802 | 3.9790 |

`MTOR_MTOR_+_11322547.23-P1P2_05`（sgRNA `GGGCAGGGGGCTTGAAGCGG`，cells=16）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| EIF2S3 | 1.0909 | 1.4157 | 1.0493 |
| LGALS1 | 0.7898 | 0.4660 | 0.7737 |
| UBE2S | 2.0784 | 1.7695 | 2.1283 |
| CCNB1 | 0.7544 | 0.4474 | 0.6706 |
| HIST3H2A | 0.6953 | 0.9994 | 0.6922 |
| VIM | 1.9881 | 2.2830 | 2.0079 |

`MTOR_MTOR_+_11322547.23-P1P2_06`（sgRNA `GGGCAGGGGGGCTGAAGCGG`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 3.2444 | 2.8135 |
| NUSAP1 | 0.5682 | 0.9567 | 0.4877 |
| CENPF | 0.9886 | 1.3748 | 1.0685 |
| HIST1H4C | 2.8420 | 2.4573 | 3.0291 |
| PDIA6 | 1.3895 | 1.0237 | 1.5531 |
| H1FX | 0.9767 | 1.3177 | 0.9031 |

`MTOR_MTOR_+_11322547.23-P1P2_07`（sgRNA `GGGCAGGGGGTCTGAAGCGG`，cells=6）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| MS4A3 | 0.9113 | 1.5732 | 0.8864 |
| MKI67 | 0.7211 | 0.1763 | 0.6648 |
| UBE2L6 | 0.2068 | 0.6934 | 0.1817 |
| MYL4 | 1.3474 | 1.7989 | 1.4351 |
| DUT | 1.6702 | 1.2742 | 1.7588 |
| HBZ | 2.7562 | 3.1391 | 2.8121 |

`MTOR_MTOR_+_11322547.23-P1P2_10`（sgRNA `GGGCAGGGGGCCTGAAGCAG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HSP90B1 | 1.4657 | 0.9612 | 1.4672 |
| HIST1H2BJ | 1.1310 | 1.6050 | 1.1806 |
| CHI3L2 | 0.8015 | 0.3414 | 0.7722 |
| LGALS1 | 0.7898 | 1.2201 | 0.7840 |
| HIST1H1C | 2.5286 | 2.9557 | 2.5603 |
| PIF1 | 0.2967 | 0.7017 | 0.2565 |

**POLR1D**

`POLR1D_POLR1D_+_28196016.23-P1_00`（sgRNA `GGGAAGCAAGGACCGACCGA`，cells=21）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR1D | 1.6896 | 0.6115 | 1.6556 |
| HBZ | 2.7562 | 3.7235 | 2.8255 |
| GYPA | 1.4214 | 1.9838 | 1.4748 |
| EIF3L | 1.8948 | 1.3579 | 1.9791 |
| VIM | 1.9881 | 1.4880 | 1.9966 |
| GYPB | 1.5517 | 2.0505 | 1.6270 |

`POLR1D_POLR1D_+_28196016.23-P1_01`（sgRNA `GGGAAGCCAGGACCGACCGA`，cells=24）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR1D | 1.6896 | 0.4344 | 1.6575 |
| EIF3L | 1.8948 | 1.2837 | 1.9635 |
| BLVRB | 1.1025 | 1.6997 | 1.2237 |
| HBZ | 2.7562 | 3.3283 | 2.8160 |
| HSP90AA1 | 2.6816 | 2.3122 | 2.7781 |
| GYPA | 1.4214 | 1.7858 | 1.4683 |

`POLR1D_POLR1D_+_28196016.23-P1_03`（sgRNA `GGTAAGCAAGGACCGACCGA`，cells=13）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR1D | 1.6896 | 0.3657 | 1.6631 |
| HBZ | 2.7562 | 3.6058 | 2.8162 |
| HIST1H4C | 2.8420 | 3.5316 | 3.0322 |
| MS4A4A | 0.6556 | 0.0808 | 0.7064 |
| EIF3L | 1.8948 | 1.3362 | 1.9775 |
| HSPA8 | 2.3720 | 1.8175 | 2.4553 |

`POLR1D_POLR1D_+_28196016.23-P1_07`（sgRNA `GGGAAGCAAGGAGCGACCGA`，cells=33）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR1D | 1.6896 | 0.8264 | 1.6569 |
| HBZ | 2.7562 | 3.1403 | 2.8193 |
| BLVRB | 1.1025 | 1.3369 | 1.2168 |
| AURKA | 0.5848 | 0.8116 | 0.6067 |
| HIST1H1C | 2.5286 | 2.7551 | 2.5518 |
| CDC20 | 0.7090 | 0.9303 | 0.6891 |

`POLR1D_POLR1D_+_28196016.23-P1_08`（sgRNA `GGGAAGCAGGGACCGACCGA`，cells=26）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR1D | 1.6896 | 0.6971 | 1.6673 |
| HBZ | 2.7562 | 3.5603 | 2.8277 |
| VIM | 1.9881 | 1.4060 | 2.0081 |
| GYPB | 1.5517 | 2.0592 | 1.6349 |
| EIF3L | 1.8948 | 1.3950 | 1.9653 |
| TSC22D1 | 0.4523 | 0.9392 | 0.4938 |

**POLR2H**

`POLR2H_POLR2H_+_184081251.23-P1P2_00`（sgRNA `GGGGCCACGAGAGCAGCAGA`，cells=6）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR2H | 1.2807 | 0.1240 | 1.5279 |
| TUBB | 2.5379 | 1.6727 | 2.6604 |
| FABP5 | 1.4353 | 0.6369 | 1.6229 |
| MS4A3 | 0.9113 | 0.1240 | 0.8870 |
| DUT | 1.6702 | 0.8982 | 1.7701 |
| SERPINF1 | 0.3973 | 1.1310 | 0.4478 |

`POLR2H_POLR2H_+_184081251.23-P1P2_07`（sgRNA `GGGGCCACGAGTGCAGCAGA`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR2H | 1.2807 | 0.4909 | 1.5246 |
| PPP1R18 | 0.8326 | 0.3639 | 0.8255 |
| HBG2 | 3.1952 | 2.7278 | 3.2552 |
| HIST1H2BJ | 1.1310 | 1.5767 | 1.1851 |
| CHI3L2 | 0.8015 | 0.4535 | 0.7707 |
| ATF4 | 1.5476 | 1.1998 | 1.5341 |

`POLR2H_POLR2H_+_184081251.23-P1P2_08`（sgRNA `GGGGCCACGCGAGCAGCAGA`，cells=6）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR2H | 1.2807 | 0.2426 | 1.5297 |
| RP11-301G19.1 | 1.7747 | 0.7893 | 1.8521 |
| CHI3L2 | 0.8015 | 0.1213 | 0.7735 |
| HIST1H2BJ | 1.1310 | 1.7928 | 1.1729 |
| ATF7IP2 | 1.1485 | 0.5198 | 1.2324 |
| NPW | 0.7693 | 0.1443 | 0.7620 |

`POLR2H_POLR2H_+_184081251.23-P1P2_11`（sgRNA `GGGGCCACGAGAGCAGCGGA`，cells=7）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR2H | 1.2807 | 0.2107 | 1.5208 |
| RP11-301G19.1 | 1.7747 | 0.9353 | 1.8629 |
| HSPA8 | 2.3720 | 1.5932 | 2.4665 |
| HIST1H2BD | 0.3383 | 1.0602 | 0.2786 |
| HIST1H2BJ | 1.1310 | 1.7773 | 1.1842 |
| VIM | 1.9881 | 1.3492 | 2.0038 |

`POLR2H_POLR2H_+_184081251.23-P1P2_12`（sgRNA `GGGGCCACGAGAGCAGGAGA`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| POLR2H | 1.2807 | 0.6411 | 1.5158 |
| HBG2 | 3.1952 | 2.7115 | 3.2590 |
| HBZ | 2.7562 | 2.3096 | 2.8136 |
| MS4A3 | 0.9113 | 0.4980 | 0.8840 |
| IFITM2 | 0.7608 | 1.1442 | 0.8074 |
| APOBEC3C | 0.7639 | 0.3862 | 0.6625 |

**RAN**

`RAN_RAN_+_131356438.23-P1P2_00`（sgRNA `GGCGGTCGCTGCGCTTAGGG`，cells=11）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| RAN | 3.2471 | 2.8421 | 3.3275 |
| HBZ | 2.7562 | 3.1548 | 2.8199 |
| CDC20 | 0.7090 | 1.1068 | 0.6816 |
| MDK | 1.6126 | 1.9751 | 1.6047 |
| GDF15 | 0.2748 | 0.6264 | 0.2714 |
| NEAT1 | 1.4726 | 1.8224 | 1.4739 |

`RAN_RAN_+_131356438.23-P1P2_02`（sgRNA `GGCGGCCGCTGCGCTTAGGG`，cells=24）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H4C | 2.8420 | 2.2676 | 3.0319 |
| RAN | 3.2471 | 2.8000 | 3.3285 |
| HIST1H2BJ | 1.1310 | 1.5112 | 1.1980 |
| HBZ | 2.7562 | 3.1018 | 2.8181 |
| BLVRB | 1.1025 | 1.4445 | 1.2131 |
| HBG1 | 0.9801 | 1.3132 | 1.0033 |

`RAN_RAN_+_131356438.23-P1P2_03`（sgRNA `GGGGGTCGCTGCGCTTAGGG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 3.2813 | 2.8181 |
| CHI3L2 | 0.8015 | 0.3497 | 0.7722 |
| RAN | 3.2471 | 2.8480 | 3.3261 |
| UBC | 1.3914 | 1.7138 | 1.4394 |
| GYPA | 1.4214 | 1.7426 | 1.4732 |
| APOE | 3.0168 | 3.3362 | 3.0772 |

`RAN_RAN_+_131356438.23-P1P2_04`（sgRNA `GGCGGTCGCGGCGCTTAGGG`，cells=14）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TOP2A | 0.7916 | 1.2211 | 0.7631 |
| CENPF | 0.9886 | 1.3158 | 1.0383 |
| MKI67 | 0.7211 | 1.0436 | 0.6560 |
| CHI3L2 | 0.8015 | 0.4837 | 0.7751 |
| GMFG | 0.4990 | 0.1901 | 0.5802 |
| RAN | 3.2471 | 2.9385 | 3.3224 |

`RAN_RAN_+_131356438.23-P1P2_12`（sgRNA `GGCGGTCGCTGCGCTTAGGT`，cells=14）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| CDK1 | 0.5468 | 0.8948 | 0.5592 |
| HBG1 | 0.9801 | 0.6344 | 1.0072 |
| NPW | 0.7693 | 0.4555 | 0.7725 |
| HIST1H4C | 2.8420 | 3.1511 | 3.0359 |
| HIST1H2BJ | 1.1310 | 1.4254 | 1.1884 |
| MKI67 | 0.7211 | 0.4347 | 0.6547 |

**RPL9**

`RPL9_RPL9_+_39460483.23-P1P2_00`（sgRNA `GGATGTTTCTGTGCTCGTGG`，cells=12）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| KIAA0101 | 1.8159 | 0.8954 | 1.8159 |
| DDIT3 | 0.2305 | 0.9315 | 0.2602 |
| HIST1H1C | 2.5286 | 1.8929 | 2.5489 |
| TXNIP | 0.4501 | 1.0033 | 0.4156 |
| SDF2L1 | 2.7222 | 2.1719 | 2.8093 |
| SNHG7 | 0.6105 | 1.1551 | 0.7345 |

`RPL9_RPL9_+_39460483.23-P1P2_01`（sgRNA `GGATGATTCTGTGCTCGTGG`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H2BJ | 1.1310 | 0.6083 | 1.1821 |
| DDIT3 | 0.2305 | 0.7373 | 0.2485 |
| SNHG7 | 0.6105 | 1.0704 | 0.7195 |
| TUBB4B | 1.3472 | 0.9007 | 1.4915 |
| SQSTM1 | 0.6113 | 0.1714 | 0.6152 |
| PRSS57 | 0.9886 | 0.5565 | 1.0504 |

`RPL9_RPL9_+_39460483.23-P1P2_04`（sgRNA `GGATGTTTCAGTGCTCGTGG`，cells=7）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| MS4A4A | 0.6556 | 1.2713 | 0.7072 |
| HBG1 | 0.9801 | 1.4996 | 1.0063 |
| HIST1H1C | 2.5286 | 3.0370 | 2.5471 |
| MS4A3 | 0.9113 | 1.3686 | 0.8957 |
| BLVRB | 1.1025 | 1.5524 | 1.1941 |
| HBZ | 2.7562 | 3.1985 | 2.8196 |

`RPL9_RPL9_+_39460483.23-P1P2_05`（sgRNA `GGATGTTTCGGTGCTCGTGG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| DDIT3 | 0.2305 | 0.9655 | 0.2454 |
| HIST1H1C | 2.5286 | 1.8053 | 2.5396 |
| KIAA0101 | 1.8159 | 1.1522 | 1.8230 |
| HSP90B1 | 1.4657 | 0.8072 | 1.4663 |
| DUT | 1.6702 | 1.0893 | 1.7640 |
| APOE | 3.0168 | 3.5557 | 3.0734 |

`RPL9_RPL9_+_39460483.23-P1P2_07`（sgRNA `GGATGTTTCTGCGCTCGTGG`，cells=17）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| SAMSN1 | 0.4144 | 0.7980 | 0.3269 |
| TSC22D1 | 0.4523 | 0.8023 | 0.5040 |
| HNRNPH1 | 1.2575 | 0.9074 | 1.3390 |
| C6orf48 | 1.9819 | 2.3253 | 2.0941 |
| BLVRB | 1.1025 | 1.4377 | 1.2228 |
| FAM178B | 0.7202 | 0.4055 | 0.8374 |

**RPS14**

`RPS14_RPS14_+_149829238.23-P1P2_00`（sgRNA `GAGGCCCGGGCGCGACAATC`，cells=14）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| EIF3L | 1.8948 | 1.0575 | 1.9693 |
| SNHG7 | 0.6105 | 1.3654 | 0.7225 |
| HBG2 | 3.1952 | 2.5823 | 3.2555 |
| DDIT3 | 0.2305 | 0.8102 | 0.2546 |
| HBZ | 2.7562 | 3.2612 | 2.8204 |
| HBD | 0.5985 | 1.0927 | 0.6324 |

`RPS14_RPS14_+_149829238.23-P1P2_01`（sgRNA `GAGACCCGGGCGCGACAATC`，cells=15）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBG2 | 3.1952 | 2.3371 | 3.2617 |
| SNHG7 | 0.6105 | 1.4283 | 0.7298 |
| HIST1H1C | 2.5286 | 1.7994 | 2.5527 |
| EIF3L | 1.8948 | 1.2581 | 1.9772 |
| KIAA0101 | 1.8159 | 1.2622 | 1.8350 |
| UBC | 1.3914 | 1.8479 | 1.4423 |

`RPS14_RPS14_+_149829238.23-P1P2_02`（sgRNA `GAGGCCCTGGCGCGACAATC`，cells=13）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 3.5284 | 2.8139 |
| EIF3L | 1.8948 | 1.2365 | 1.9672 |
| DDIT3 | 0.2305 | 0.8178 | 0.2449 |
| FABP5 | 1.4353 | 0.9629 | 1.6324 |
| SNHG7 | 0.6105 | 1.0817 | 0.7174 |
| HSPA8 | 2.3720 | 1.9898 | 2.4615 |

`RPS14_RPS14_+_149829238.23-P1P2_04`（sgRNA `GAGGCCCGCGCGCGACAATC`，cells=22）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| EIF3L | 1.8948 | 1.5351 | 1.9697 |
| HBZ | 2.7562 | 3.1073 | 2.8157 |
| MYC | 1.1698 | 0.8302 | 1.1903 |
| MDK | 1.6126 | 1.3107 | 1.5966 |
| NEAT1 | 1.4726 | 1.7576 | 1.4731 |
| HIST1H4C | 2.8420 | 2.5762 | 3.0259 |

`RPS14_RPS14_+_149829238.23-P1P2_08`（sgRNA `GAGGCCCGGGCTCGACAATC`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| CHI3L2 | 0.8015 | 0.2688 | 0.7770 |
| HBG2 | 3.1952 | 2.6764 | 3.2507 |
| HBG1 | 0.9801 | 0.5928 | 1.0019 |
| TIMP1 | 0.8194 | 0.4450 | 0.7764 |
| PNRC1 | 0.1829 | 0.5409 | 0.1295 |
| IFITM2 | 0.7608 | 0.4241 | 0.8028 |

`RPS14_RPS14_+_149829238.23-P1P2_13`（sgRNA `GAGGCCCGGGCGCGACAGTC`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBZ | 2.7562 | 3.3151 | 2.8219 |
| BLVRB | 1.1025 | 1.6381 | 1.2071 |
| EIF3L | 1.8948 | 1.3755 | 1.9651 |
| ATF7IP2 | 1.1485 | 1.6056 | 1.2344 |
| MS4A3 | 0.9113 | 1.3296 | 0.8872 |
| MS4A4A | 0.6556 | 1.0530 | 0.6951 |

**RPS15**

`RPS15_RPS15_-_1438413.23-P1P2_00`（sgRNA `GACCAAAGCGATCTCTTCTG`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| EIF3L | 1.8948 | 1.2305 | 1.9720 |
| SNHG7 | 0.6105 | 1.2466 | 0.7240 |
| PPIB | 1.4270 | 0.8433 | 1.4729 |
| DDIT3 | 0.2305 | 0.7804 | 0.2570 |
| PTTG1 | 1.5473 | 0.9977 | 1.6145 |
| TUBB | 2.5379 | 2.0045 | 2.6525 |

`RPS15_RPS15_-_1438413.23-P1P2_01`（sgRNA `GACCAAACCGATCTCTTCTG`，cells=23）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HNRNPH1 | 1.2575 | 0.9702 | 1.3366 |
| PPP1CB | 0.6669 | 0.3855 | 0.6260 |
| SERPINF1 | 0.3973 | 0.6483 | 0.4446 |
| IER2 | 0.6795 | 0.4331 | 0.6468 |
| APOBEC3C | 0.7639 | 0.5220 | 0.6605 |
| TOP2A | 0.7916 | 1.0332 | 0.7433 |

`RPS15_RPS15_-_1438413.23-P1P2_02`（sgRNA `GACCAAGGCGATCTCTTCTG`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| EIF3L | 1.8948 | 1.3835 | 1.9788 |
| HIST1H4C | 2.8420 | 3.3335 | 3.0389 |
| TMSB4X | 3.9916 | 3.5169 | 3.9977 |
| EIF2S3 | 1.0909 | 0.6211 | 1.0402 |
| NEAT1 | 1.4726 | 1.9348 | 1.4646 |
| HSPA8 | 2.3720 | 1.9497 | 2.4560 |

`RPS15_RPS15_-_1438413.23-P1P2_07`（sgRNA `GACCAAAGCGGTCTCTTCTG`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| EIF3L | 1.8948 | 1.3483 | 1.9693 |
| SERPINF1 | 0.3973 | 0.9086 | 0.4565 |
| SNHG7 | 0.6105 | 1.1176 | 0.7235 |
| DDIT3 | 0.2305 | 0.7092 | 0.2577 |
| BLVRB | 1.1025 | 1.5627 | 1.2125 |
| SSR3 | 1.2539 | 0.8147 | 1.3948 |

`RPS15_RPS15_-_1438413.23-P1P2_12`（sgRNA `GACCAAAGCGATCTCTTGTG`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBG2 | 3.1952 | 2.6476 | 3.2541 |
| MS4A3 | 0.9113 | 0.4100 | 0.8798 |
| IFITM2 | 0.7608 | 0.3852 | 0.8109 |
| HBZ | 2.7562 | 2.4158 | 2.8109 |
| CCNB1 | 0.7544 | 0.4164 | 0.6823 |
| HIST1H1E | 0.3595 | 0.6962 | 0.3552 |

**RPS18**

`RPS18_RPS18_+_33239917.23-P1P2_00`（sgRNA `GCTGCGATGCCGCTGGATCA`，cells=4）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H2BJ | 1.1310 | 0.2283 | 1.1905 |
| KIAA0101 | 1.8159 | 0.9149 | 1.8365 |
| DDIT3 | 0.2305 | 1.0789 | 0.2545 |
| PTTG1 | 1.5473 | 0.7149 | 1.6187 |
| SNHG7 | 0.6105 | 1.4424 | 0.7205 |
| HMGB2 | 1.8561 | 1.0693 | 1.8703 |

`RPS18_RPS18_+_33239917.23-P1P2_01`（sgRNA `GCTGCAATGCCGCTGGATCA`，cells=8）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H1C | 2.5286 | 1.5492 | 2.5470 |
| SNHG7 | 0.6105 | 1.3851 | 0.7261 |
| EIF3L | 1.8948 | 1.1989 | 1.9726 |
| HMGB2 | 1.8561 | 1.1720 | 1.8693 |
| HBA1 | 1.0624 | 0.3899 | 1.0740 |
| HBG2 | 3.1952 | 2.5238 | 3.2532 |

`RPS18_RPS18_+_33239917.23-P1P2_02`（sgRNA `GCTGGGATGCCGCTGGATCA`，cells=5）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HBA1 | 1.0624 | 0.3023 | 1.0853 |
| HIST1H4C | 2.8420 | 3.5956 | 3.0298 |
| EIF3L | 1.8948 | 1.2013 | 1.9756 |
| CENPF | 0.9886 | 1.6023 | 1.0398 |
| CDC45 | 0.9935 | 1.5868 | 1.1110 |
| HIST1H1C | 2.5286 | 1.9413 | 2.5505 |

`RPS18_RPS18_+_33239917.23-P1P2_04`（sgRNA `GCTGCGATCCCGCTGGATCA`，cells=13）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| HIST1H4C | 2.8420 | 2.3910 | 3.0256 |
| HBZ | 2.7562 | 3.1485 | 2.8203 |
| EIF2S3 | 1.0909 | 0.7117 | 1.0579 |
| EIF3L | 1.8948 | 1.5159 | 1.9693 |
| RP11-301G19.1 | 1.7747 | 1.4371 | 1.8645 |
| PLK1 | 0.5970 | 0.9335 | 0.5577 |

`RPS18_RPS18_+_33239917.23-P1P2_08`（sgRNA `GCTGCGATTCCGCTGGATCA`，cells=23）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| EIF3L | 1.8948 | 1.3245 | 1.9763 |
| KIAA0101 | 1.8159 | 1.3309 | 1.8405 |
| VIM | 1.9881 | 1.6082 | 2.0053 |
| SNHG7 | 0.6105 | 0.9873 | 0.7226 |
| GNB2L1 | 3.8848 | 3.5273 | 3.9532 |
| HIST1H1C | 2.5286 | 2.2357 | 2.5553 |

**SEC61A1**

`SEC61A1_SEC61A1_-_127771295.23-P1_00`（sgRNA `GGCACTGACGTGTCTCTCGG`，cells=20）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| SLC3A2 | 0.7141 | 1.8781 | 0.6485 |
| HBG2 | 3.1952 | 2.5948 | 3.2628 |
| LAPTM5 | 1.0762 | 1.5611 | 1.0128 |
| KRT8 | 0.6739 | 1.1510 | 0.6586 |
| PIM1 | 0.5387 | 0.9964 | 0.4904 |
| FABP5 | 1.4353 | 0.9952 | 1.6289 |

`SEC61A1_SEC61A1_-_127771295.23-P1_01`（sgRNA `GGCACTGTCGTGTCTCTCGG`，cells=7）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| SLC3A2 | 0.7141 | 1.7538 | 0.6607 |
| HBZ | 2.7562 | 1.9326 | 2.8191 |
| HBG2 | 3.1952 | 2.4562 | 3.2643 |
| HBA1 | 1.0624 | 0.3272 | 1.0854 |
| ACSM3 | 0.3153 | 1.0029 | 0.1871 |
| EIF2S3 | 1.0909 | 1.7772 | 1.0426 |

`SEC61A1_SEC61A1_-_127771295.23-P1_02`（sgRNA `GGCGCTGACGTGTCTCTCGG`，cells=19）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| SLC3A2 | 0.7141 | 1.7325 | 0.6556 |
| SEC61A1 | 0.5475 | 0.0923 | 0.5887 |
| HBA1 | 1.0624 | 0.6105 | 1.0808 |
| APOE | 3.0168 | 3.4091 | 3.0816 |
| VIM | 1.9881 | 2.3685 | 2.0122 |
| FAM83A | 0.6063 | 0.9772 | 0.5072 |

`SEC61A1_SEC61A1_-_127771295.23-P1_03`（sgRNA `GGTACTGACGTGTCTCTCGG`，cells=16）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| SLC3A2 | 0.7141 | 1.7536 | 0.6556 |
| HBZ | 2.7562 | 2.1438 | 2.8180 |
| PPIB | 1.4270 | 1.9636 | 1.4674 |
| DNAJA1 | 1.4641 | 0.9679 | 1.5922 |
| HBA1 | 1.0624 | 0.5724 | 1.0835 |
| SSR3 | 1.2539 | 1.7169 | 1.4047 |

`SEC61A1_SEC61A1_-_127771295.23-P1_04`（sgRNA `GGCACTGAAGTGTCTCTCGG`，cells=10）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| RP11-301G19.1 | 1.7747 | 1.2171 | 1.8692 |
| HBG1 | 0.9801 | 1.5157 | 1.0046 |
| HBZ | 2.7562 | 2.2438 | 2.8198 |
| HBA1 | 1.0624 | 0.5829 | 1.0828 |
| PDIA3 | 0.6189 | 1.0787 | 0.6115 |
| CDC20 | 0.7090 | 1.1025 | 0.6898 |

**TUBB**

`TUBB_TUBB_+_30688126.23-P1_00`（sgRNA `GCGGCAGGAAGGTTCTGAGA`，cells=16）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TUBB | 2.5379 | 1.4649 | 2.6607 |
| HBZ | 2.7562 | 3.6098 | 2.8193 |
| HBG2 | 3.1952 | 3.9138 | 3.2673 |
| ALAS2 | 0.2129 | 0.6801 | 0.1607 |
| TUBB4B | 1.3472 | 1.8077 | 1.4951 |
| BLVRB | 1.1025 | 1.5475 | 1.2208 |

`TUBB_TUBB_+_30688126.23-P1_01`（sgRNA `GCAGCAGGAAGGTTCTGAGA`，cells=7）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TUBB | 2.5379 | 1.6401 | 2.6573 |
| UBC | 1.3914 | 2.0703 | 1.4518 |
| DDIT3 | 0.2305 | 0.8973 | 0.2588 |
| HIST1H4C | 2.8420 | 2.1898 | 3.0231 |
| CTSL | 0.3472 | 0.9842 | 0.2994 |
| CCNB2 | 0.5547 | 1.1688 | 0.5152 |

`TUBB_TUBB_+_30688126.23-P1_03`（sgRNA `GCGGCAGCAAGGTTCTGAGA`，cells=5）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TUBB | 2.5379 | 0.7851 | 2.6584 |
| HIST1H4C | 2.8420 | 2.2742 | 3.0279 |
| HBZ | 2.7562 | 3.3203 | 2.8183 |
| ATF7IP2 | 1.1485 | 0.6309 | 1.2421 |
| MARCKSL1 | 2.1143 | 1.6178 | 2.1823 |
| HIST1H2AC | 1.0096 | 1.4769 | 1.0141 |

`TUBB_TUBB_+_30688126.23-P1_06`（sgRNA `GCGGCAGGACGGTTCTGAGA`，cells=9）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TUBB | 2.5379 | 0.8422 | 2.6475 |
| HBZ | 2.7562 | 3.8911 | 2.8222 |
| HBG1 | 0.9801 | 1.6213 | 0.9925 |
| HBG2 | 3.1952 | 3.6738 | 3.2572 |
| HIST3H2A | 0.6953 | 1.1583 | 0.6993 |
| HIST1H1C | 2.5286 | 2.9832 | 2.5527 |

`TUBB_TUBB_+_30688126.23-P1_10`（sgRNA `GCGGCAGGAAGGTTCAGAGA`，cells=18）

| gene | control mean | observed mean | predicted mean |
| --- | ---: | ---: | ---: |
| TUBB | 2.5379 | 1.7578 | 2.6575 |
| TUBB4B | 1.3472 | 1.8231 | 1.4933 |
| HBZ | 2.7562 | 3.1155 | 2.8191 |
| EIF2S3 | 1.0909 | 0.7426 | 1.0546 |
| HIST1H2BJ | 1.1310 | 1.4784 | 1.1858 |
| PPP1R18 | 0.8326 | 0.5049 | 0.8323 |

## 5. 产出文件

- `runs\gse132080_target_200ep\preliminary\expression_control_mean.csv`
- `runs\gse132080_target_200ep\preliminary\sgrna_delta_vectors.csv`
- `runs\gse132080_target_200ep\preliminary\sgrna_expression_matrix.csv`
- `runs\gse132080_target_200ep\preliminary\sgrna_predicted_expression_matrix.csv`
- `runs\gse132080_target_200ep\preliminary\sgrna_summary.csv`
- `runs\gse132080_target_200ep\preliminary\sgrna_top_gene_expression_long.csv`
- `runs\gse132080_target_200ep\preliminary\z_sem_conditions.csv`
- `runs\gse132080_target_200ep\preliminary\z_sem_control.csv`
- `runs\gse132080_target_200ep\preliminary\z_sem_control_cells.csv`

按“条件级”粒度导出的对照版本：

- `runs\gse132080_target_200ep\preliminary_condition\condition_delta_vectors.csv`
- `runs\gse132080_target_200ep\preliminary_condition\condition_expression_matrix.csv`
- `runs\gse132080_target_200ep\preliminary_condition\condition_predicted_expression_matrix.csv`
- `runs\gse132080_target_200ep\preliminary_condition\condition_summary.csv`
- `runs\gse132080_target_200ep\preliminary_condition\condition_top_gene_expression_long.csv`
- `runs\gse132080_target_200ep\preliminary_condition\expression_control_mean.csv`
- `runs\gse132080_target_200ep\preliminary_condition\manifest.json`
- `runs\gse132080_target_200ep\preliminary_condition\z_sem_conditions.csv`
- `runs\gse132080_target_200ep\preliminary_condition\z_sem_control.csv`
- `runs\gse132080_target_200ep\preliminary_condition\z_sem_control_cells.csv`

