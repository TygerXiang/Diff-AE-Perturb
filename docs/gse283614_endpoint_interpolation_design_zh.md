# GSE283614 端点训练、中间剂量盲测设计

## 研究问题

检验模型仅观察较轻和较强的 `NANOG` / `OCT4` CRISPRi 剂量后，能否预测未见过的中间剂量表达分布，以及 DE 谱系程序的非单调转折。该任务是剂量插值，不应与随机留出细胞或普通 guide-CV 混为一谈。

剂量采用论文 Table S4 的 TF expression percentage，并定义：

```text
tf_remaining = TF expression percentage / 100
knockdown_fraction = clip(1 - tf_remaining, 0, 1)
```

分割在 `NANOG` 和 `OCT4` 内分别完成。不能用 guide capture UMI 数代替生物学剂量。

## 主分割

每个靶点按 `knockdown_fraction` 排序，将两端各 30% 定义为端点池：

- 训练：两端池中除边界 guide 外的全部 guide。
- 验证：轻、重两端最靠近中间区的各 1 条 guide；仅用于 early stopping 和超参数选择。
- 测试：中间区全部 guide；在模型、超参数和评价方案冻结前不可查看其表达结果。
- 对照：每个细胞状态内按固定随机种子分为 60%/20%/20%。

所有细胞按完整 `guide_identity` 整组划分。同一 guide 的细胞不得跨 train/validation/test。ESC 和 DE 分开生成，以免把状态泛化与剂量插值混在同一个结论中。

当前数据的实际分割为：

| split | NANOG guides | OCT4 guides | ESC cells | DE cells |
| --- | ---: | ---: | ---: | ---: |
| train | 10 | 8 | 3,099 | 5,035 |
| validation | 2 | 2 | 850 | 1,417 |
| middle-dose test | 5 | 6 | 1,605 | 2,508 |

完整 guide、剂量和细胞数记录在 `reproducibility/prepared/gse283614/endpoint_interpolation/endpoint_split_manifest.json`。

## 模型输入与对照方法

主模型的条件变量只使用 `cell_state`、靶点身份和连续 `knockdown_fraction`。guide 名称、序列、基因组位置或该 guide 的任何中间剂量表达统计不能作为输入，否则任务会变成 guide 识别而不是剂量预测。

至少比较以下预注册基线：

1. 对照细胞不变基线。
2. 最近端点基线。
3. 两端 pseudobulk 的线性插值。
4. 只知道 `NANOG` / `OCT4`、不知道剂量的 target-only 模型。
5. 同架构的 dose-conditioned 模型。

现有 rank-stratified 5-fold guide-CV 作为辅助实验保留，用于回答“未知 guide 的总体泛化”；它不能替代本主实验，因为其训练集覆盖中间剂量。

## 盲测终点

评价单位必须是 guide，而不是把数千个细胞当成独立重复。先对每条 guide 做 pseudobulk 或均值分布摘要，再在 guide 层面 bootstrap。

主终点（DE）：

- differentiation gene-set score 的预测相关性和 RMSE；
- 预测曲线是否为 inverted-U；
- 预测峰值位置与真实峰值位置的绝对误差；
- 中间剂量相对轻、重端点的方向是否同时正确。

次终点：

- 全基因 perturbation delta 的 Pearson/Spearman、RMSE；
- top DE genes 的方向一致率；
- 生成分布与真实分布的 energy distance 或 MMD；
- ESC pluripotency score，以及 DE off-target/neuroectoderm score。

成功标准应在解盲前固定。推荐要求 dose-conditioned 模型同时满足：中间 guide 的主终点显著优于 target-only 和线性插值；两个靶点的 inverted-U 方向正确；95% guide-bootstrap 区间不跨越零改进。不要仅用单细胞级显著性作为成功证据。

## 稳健性分析

- 将端点比例改为 25% 和 35%，结论应保持一致。
- 分别报告 ESC/DE、NANOG/OCT4，不以合并均值掩盖失败组合。
- 用 NT control 与 SafeHarbor control 分开复核基线。
- 现有五折 guide-CV 用于检查结果是否由少数 guide 驱动。
- 在完全冻结方案后，才可绘制中间剂量真实曲线。

## 独立实验验证

数据内盲测证明的是 held-out-guide 预测，不是独立生物实验验证。后续建议在每个靶点围绕模型预测峰值选择至少 5 个剂量点（两个端点、峰前、峰值附近、峰后），每个剂量使用独立 guide 或可调 CRISPRi，并完成至少 3 次独立分化批次。预先固定 DE marker panel、off-target lineage panel 和峰值位置容差；实验人员在定量完成前不查看模型排序。只有新批次中的峰值位置、方向和表达分布均达到预注册标准，才能声称模型具有前瞻预测能力。

## 生成分割

```powershell
C:\Users\Admin\myprojects\diff-project\.conda\envs\perturbdiff-gse132080\python.exe `
  scripts/create_gse283614_endpoint_split.py `
  --adata-path reproducibility/prepared/gse283614/gse283614_combined_assigned.h5ad `
  --dosage-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-4.xlsx `
  --guide-library-workbook C:/Users/Admin/myprojects/diff-project/datasets/GSE283614/media-1.xlsx `
  --output-dir reproducibility/prepared/gse283614/endpoint_interpolation
```

加入 `--manifest-only` 可只审计分割而不写出大型 `.h5ad` 文件。
