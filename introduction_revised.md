# Introduction

CRISPR pooled screen 与单细胞 RNA 测序的结合，使研究者能够在单细胞分辨率下系统测量遗传扰动对转录组的影响。Perturb-seq 将 guide RNA 身份与单细胞表达谱对应起来，把传统 pooled screen 中较低维的生长、存活或 reporter readout 扩展为全转录组表型 [1,2]。CROP-seq 进一步建立了将 CRISPR 扰动与液滴式单细胞转录组读出相结合的可扩展实验框架 [3]。近年来，基因组尺度的 CRISPRi Perturb-seq 已经能够在数百万细胞中测量大量基因扰动的转录效应，并利用这些高维表型推断基因功能、细胞状态和复杂的 genotype-phenotype landscape [4]。这些实验进展为建立可预测、可解释的扰动响应模型提供了数据基础。

然而，CRISPRi 扰动响应的建模仍然面临多个挑战。首先，单细胞表达数据具有高噪声、稀疏性和强烈的细胞状态依赖性，观测到的表达差异不仅来自目标基因扰动，也受到细胞周期、基础转录状态、批次效应和测序捕获效率等因素影响。scVI 等深度生成模型表明，低维潜变量可以在保留生物学结构的同时吸收部分技术和背景变异 [5]，因此在潜空间中建模扰动效应是一个自然选择。其次，CRISPRi 并不是严格的二元干预。同一个靶基因的不同 sgRNA 可能具有不同 repression efficiency，其活性受 guide 序列、靶点位置、转录起始位点附近的相对距离和局部染色质状态等因素影响；Horlbeck 等人的 CRISPRi/CRISPRa guide 设计研究已经系统证明了这些因素会影响 guide activity [6]。第三，即便在相同扰动标签下，单细胞也可能表现出不同的响应幅度和响应状态。例如 Adamson 等人在 Perturb-seq 研究中观察到，相同扰动下的细胞可出现 unfolded protein response 分支激活的分化模式 [2]。因此，仅使用“靶基因身份”作为条件变量可能不足以刻画真实的 CRISPRi 响应；guide-level activity 和单细胞状态依赖性都需要在模型中被区分。

已有计算方法已经显著推动了单细胞扰动预测的发展，但对遗传扰动标签、guide activity 和细胞状态之间关系的处理仍然不充分。scGen 使用变分自编码器和潜空间向量运算预测 out-of-sample perturbation response，说明扰动可以近似表示为低维表达流形中的方向性偏移 [7]。Compositional perturbation autoencoder（CPA）进一步将扰动、协变量和剂量组合纳入模型，可以预测未见过的剂量、细胞类型和扰动组合 [8]。GEARS 利用基因关系图和 perturbation embedding 预测未见过的单基因或多基因扰动转录结果 [9]。CellOT 将单细胞扰动响应建模为 control 与 perturbed 分布之间的神经最优传输映射，强调了未配对单细胞分布之间的异质性响应 [10]。scGPT 等 foundation model 则表明，大规模预训练模型能够学习基因和细胞表示，并用于扰动响应预测和基因网络推断 [11]。这些方法分别从潜空间方向、剂量组合、基因图先验、分布映射和大规模预训练角度推进了扰动建模，但多数方法仍将遗传扰动主要视为离散标签，或在剂量建模时依赖外部已知剂量信息。对于 CRISPRi screen 中常见的 sgRNA activity 不同且难以直接由单细胞表达矩阵观测的问题，仍然需要一个能够同时保留 guide-level 条件和反事实生成能力的紧凑模型。

基于这一需求，当前模型采用一个紧凑的条件扩散框架，将 scGen 式潜空间扰动表示与条件 DDIM 解码器结合起来。模型首先通过语义编码器将表达谱映射为低维语义向量 $z_{\mathrm{sem}}=E_{\phi}(x)$；随后为每个非对照扰动条件学习一个潜空间扰动向量 $\boldsymbol{\delta}_g$。该向量表在训练开始时可以由扰动细胞与对照细胞的潜空间均值差进行初始化：

$$
\boldsymbol{\delta}_g
= \mathbb{E}[E_{\phi}(x)\mid g]
- \mathbb{E}[E_{\phi}(x)\mid \mathrm{control}] .
$$

在观测细胞的训练路径中，模型使用该细胞自身的语义表示作为 denoiser 条件，并将 $z_{\mathrm{basal}}=z_{\mathrm{sem}}-\boldsymbol{\delta}_g$ 保留为扰动移除后的潜变量解释。对于反事实生成，模型从对照细胞出发执行“移除源扰动、加入目标扰动”的潜空间运算：

$$
z_{\mathrm{cf}}
= E_{\phi}(x_{\mathrm{control}})
- \boldsymbol{\delta}_{g_{\mathrm{source}}}
+ \boldsymbol{\delta}_{g_{\mathrm{target}}} .
$$

在默认的对照到扰动生成中，源扰动为零，因此退化为 $z_{\mathrm{cf}}=E_{\phi}(x_{\mathrm{control}})+\boldsymbol{\delta}_{g}$。条件 denoiser 接收带噪表达、扩散时间步和扰动条件潜变量，并预测干净表达谱或噪声；Gaussian diffusion 提供前向加噪过程，DDIM 提供确定性的反向采样过程。

与早期设想的“从每个细胞表达中直接推断一个隐藏 knockdown percentage”不同，当前实现将 activity conditioning 设计为一个可选分支，而不是默认的细胞级强度推断头。对于 predicted activity 模式，模型使用 guide 的 genome input 与 sgRNA input 序列构成的 Table S8 风格特征，通过 CNN GuideStrengthEncoder 预测非负的 guide activity；对于 direct activity 模式，模型可以直接读取外部实验或参考 screen 提供的 guide-level activity。随后，target-specific GeneralizedSigmoid 将 activity 映射为目标相关的连续条件值。默认设置下，该连续值经过一个小型投影网络得到 activity embedding，并作为额外条件输入 denoiser；只有在显式开启兼容选项时，activity 才会进一步乘到 $\boldsymbol{\delta}_g$ 上。这样可以把 activity 对生成器的作用与扰动方向本身分开，也保留了 folded conditioning 作为消融和旧模型兼容路径。

这一设计将当前模型中的条件因素分为三部分：第一，来自表达谱的细胞语义状态；第二，由 target 或 sgRNA 条件决定的潜空间扰动方向；第三，由 guide 序列或外部测量决定的 guide-level activity 条件。模型可以使用 target-level 标签，也可以通过 raw strategy 保留完整 sgRNA 标签；在 target-level 扰动与 guide-level activity 同时存在时，perturbation label 和 activity label 可以使用不同的字段，从而分别控制扰动方向的粒度和 activity 查找的粒度。需要强调的是，当前 activity 分支主要表达 guide-level 的可观测或可预测代理，并不等同于已经识别出每个细胞的物理 knockdown percentage；细胞间响应异质性仍主要由语义表示和扩散生成分布承担。

训练目标也相应从原先的强度可辨识性约束转向条件生成质量与表示稳定性。基础训练使用带有 signature-gene weighting 的扩散重建损失，可以选择普通 MSE 或 GEARS 风格的 autofocus 与方向感知损失；此外还支持语义对齐损失和同一 target 下 guide 方向对齐损失。在 activity conditioning 模式下，可以使用外部 guide activity 对 GuideStrengthEncoder 进行可选的监督。当前默认路径不依赖早期设想的强度排序损失、强度熵正则或强度锚定损失，因此模型输出的 activity 应被解释为用于条件控制和 guide 比较的相对变量，而不是绝对物理意义上的 repression percentage。

在生成模块上，条件扩散模型仍然是当前实现的核心。去噪扩散概率模型通过学习逐步加噪过程的逆过程，已经成为高质量生成建模的重要框架 [12]；DDIM 在保持相同训练目标的基础上提供了更高效的确定性采样方式 [13]。在单细胞领域，scDiffusion 等工作已经显示，条件扩散模型可以用于生成高质量的单细胞表达数据 [14]。对于 CRISPRi 响应建模，扩散解码器的优势在于它不只输出一个平均表达状态，而是能够在给定细胞状态、扰动方向和 activity 条件后生成表达分布，从而保留单细胞转录响应中的残余异质性。当前 denoiser 仍采用面向向量表达谱的 MLP，是一个有意保持紧凑的基线，尚未引入基因图、注意力或更大规模的 foundation model 结构。

当前实验结果支持 activity conditioning 作为一个有价值但并非无条件优越的扩展。在 GSE132080 上，200 epoch 的 target-level 与 sgRNA-level 对比显示，保留 raw sgRNA 标签并不会自动带来稳定的生成精度提升，两者总体接近且 target-level 略有优势。因此，guide-level 表示本身不能替代对 guide activity 的显式建模。在同一数据集的 127 个可评估 guide 上，200 epoch 的 explicit activity channel 相比无 activity 基线改善了 mean-profile MSE、mean-shift correlation、energy distance 和 common-DEGs，但 Wasserstein 与 KL divergence 略有变差。在 GSE283614 的 33 个 targeting guide 对比中，使用参考 screen log2FC 构造的 direct activity 条件将 mean-profile MSE 从 $0.005275$ 降至 $0.004504$，并将 pcc_delta 从 $0.057729$ 提升至 $0.295953$；与此同时，energy distance 和 KL divergence 变差。由此可见，activity 条件更明显地改善了平均响应方向和表达轮廓，但对完整单细胞分布的改善并不一致。

综上，本研究当前的模型框架不再把“细胞级隐藏扰动强度推断”作为已经完成的核心假设，而是采用一个可复现的条件生成基线，并在其上加入可开关的 guide activity conditioning。通过将细胞语义状态、target/sgRNA 扰动方向和 guide-level activity 条件显式分离，模型同时支持 target-only 基线、sgRNA-level 条件建模、外部 activity 条件输入以及从对照细胞出发的反事实 DDIM 生成。该框架为研究 CRISPRi guide heterogeneity、比较不同 activity 代理和进行连续条件模拟提供了统一接口，但 activity 的生物学解释和跨 guide 泛化仍需要严格的 guide-held-out 验证。

## 可行性与文献依据说明

当前模型的基础结构具有清晰的文献依据。scGen 证明了潜空间扰动方向可以用于预测未见过的单细胞扰动响应 [7]；CPA 说明了连续剂量或强度变量能够补充离散扰动标签 [8]；Horlbeck 等人的研究则为 sgRNA activity 存在 guide-to-guide 和 context-dependent variation 提供了实验基础 [6]。因此，本模型将 guide-level activity 作为条件变量是合理的，但当前实现把它作为可选输入和相对代理，而不是直接宣称恢复真实的 repression level。

当前结果也明确了模型的边界。首先，GSE132080 上 target-level 与 sgRNA-level 的结果接近，说明简单增加 sgRNA embedding 不能保证生成质量提升；需要进一步区分 guide identity、guide sequence 和 activity 信息各自贡献。其次，GSE283614 上 explicit activity channel 对平均响应指标有明显帮助，却在 energy distance 和 KL divergence 上出现退化，说明 activity 条件可能提高方向性拟合，同时改变生成分布的校准。后续比较应同时报告均值误差、响应方向、分布距离和差异基因恢复，而不能只依赖单一指标。

GSE283614 的生物学剂量验证已经建立了较严格的分析流程，包括 48 个 guide 和 33 个 targeting guide 的覆盖检查、Table S4 中六类响应关系的复拟合、guide-level bootstrap、essentiality buffering 检验，以及按 guide 划分的五折交叉验证设计。当前观察阶段的覆盖和 essentiality buffering 结果基本满足预期，但独立复拟合尚未通过预先设定的完整 gate：NANOG-DE differentiation 的二次模型顶点略低于观测剂量范围，OCT4-DE off-target 的 score sheet 重新拟合结果也与 Table S4 报告的 Hill 模型不一致。因此，正式的 200 epoch dose-response OOF 训练应在这些 score/model discrepancy 解决前保持阻断；当前已完成的 full-data refit 和 virtual dose scan 只能作为探索性分析。

对于后续实验，建议至少保留以下消融：去掉 activity 条件、使用 direct activity 与 predicted activity 的比较、关闭或开启 activity 对 $\boldsymbol{\delta}_g$ 的额外缩放、target-level 与 raw sgRNA 条件的比较，以及使用普通 MLP/VAE decoder 替代 diffusion decoder。对于 GSE283614 等 guide 数量有限的数据集，应优先使用 guide-held-out split，避免同一 guide 的细胞同时出现在训练和测试中；activity 表中的缺失值也不应在正式剂量实验中静默填充为零。对于 virtual guide scale，当前实现可以用 learned embedding norm 相对于 fully matched sgRNA embedding norm 构造相对 $s$ 值，但该量没有在训练中直接监督，只能作为可解释性和反事实扫描的探索性指标。

扩散解码器有文献基础，但仍需要用实验支撑它对当前任务的必要性。DDPM/DDIM 是经典生成模型基础 [12,13]，scDiffusion 说明 diffusion 可以适配单细胞表达生成 [14]。但审稿人可能会质疑其复杂度，因此应比较 diffusion decoder 与较简单 decoder 在差异表达基因恢复、表达分布相似性、反事实插值平滑性以及高 activity/低 activity 生成稳定性上的表现，并将当前 activity conditioning 在均值和分布指标上的 trade-off 一并报告。

## References

[1] Dixit, A. et al. Perturb-Seq: Dissecting molecular circuits with scalable single-cell RNA profiling of pooled genetic screens. *Cell* 167, 1853-1866.e17 (2016). https://doi.org/10.1016/j.cell.2016.11.038

[2] Adamson, B. et al. A multiplexed single-cell CRISPR screening platform enables systematic dissection of the unfolded protein response. *Cell* 167, 1867-1882.e21 (2016). https://doi.org/10.1016/j.cell.2016.11.048

[3] Datlinger, P. et al. Pooled CRISPR screening with single-cell transcriptome readout. *Nature Methods* 14, 297-301 (2017). https://doi.org/10.1038/nmeth.4177

[4] Replogle, J. M. et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559-2575.e28 (2022). https://doi.org/10.1016/j.cell.2022.05.013

[5] Lopez, R. et al. Deep generative modeling for single-cell transcriptomics. *Nature Methods* 15, 1053-1058 (2018). https://doi.org/10.1038/s41592-018-0229-2

[6] Horlbeck, M. A. et al. Compact and highly active next-generation libraries for CRISPR-mediated gene repression and activation. *eLife* 5, e19760 (2016). https://doi.org/10.7554/eLife.19760

[7] Lotfollahi, M., Wolf, F. A. & Theis, F. J. scGen predicts single-cell perturbation responses. *Nature Methods* 16, 715-721 (2019). https://doi.org/10.1038/s41592-019-0494-8

[8] Lotfollahi, M. et al. Predicting cellular responses to complex perturbations in high-throughput screens. *Molecular Systems Biology* 19, e11517 (2023). https://doi.org/10.15252/msb.202211517

[9] Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology* 42, 927-935 (2024). https://doi.org/10.1038/s41587-023-01905-6

[10] Bunne, C. et al. Learning single-cell perturbation responses using neural optimal transport. *Nature Methods* 20, 1759-1768 (2023). https://doi.org/10.1038/s41592-023-01969-x

[11] Cui, H. et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods* 21, 1470-1480 (2024). https://doi.org/10.1038/s41592-024-02201-0

[12] Ho, J., Jain, A. & Abbeel, P. Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems* 33, 6840-6851 (2020). https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html

[13] Song, J., Meng, C. & Ermon, S. Denoising diffusion implicit models. *International Conference on Learning Representations* (2021). https://openreview.net/forum?id=St1giarCHLP

[14] Luo, E., Hao, M., Wei, L. & Zhang, X. scDiffusion: conditional generation of high-quality single-cell data using diffusion model. *Bioinformatics* 40, btae518 (2024). https://doi.org/10.1093/bioinformatics/btae518
