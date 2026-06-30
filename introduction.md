# Introduction

CRISPR pooled screen 与单细胞 RNA 测序的结合，使研究者能够在单细胞分辨率下系统测量遗传扰动对转录组的影响。Perturb-seq 将 guide RNA 身份与单细胞表达谱对应起来，把传统 pooled screen 中较低维的生长、存活或 reporter readout 扩展为全转录组表型 [1,2]。CROP-seq 进一步建立了将 CRISPR 扰动与液滴式单细胞转录组读出相结合的可扩展实验框架 [3]。近年来，基因组尺度的 CRISPRi Perturb-seq 已经能够在数百万细胞中测量大量基因扰动的转录效应，并利用这些高维表型推断基因功能、细胞状态和复杂的 genotype-phenotype landscape [4]。这些实验进展为建立可预测、可解释的扰动响应模型提供了数据基础。

然而，CRISPRi 扰动响应的建模仍然面临多个挑战。首先，单细胞表达数据具有高噪声、稀疏性和强烈的细胞状态依赖性，观测到的表达差异不仅来自目标基因扰动，也受到细胞周期、基础转录状态、批次效应和测序捕获效率等因素影响。scVI 等深度生成模型表明，低维潜变量可以在保留生物学结构的同时吸收部分技术和背景变异 [5]，因此在潜空间中建模扰动效应是一个自然选择。其次，CRISPRi 并不是严格的二元干预。同一个靶基因的不同 sgRNA 可能具有不同 repression efficiency，其活性受 guide 序列、靶点位置、转录起始位点附近的相对距离和局部染色质状态等因素影响；Horlbeck 等人的 CRISPRi/CRISPRa guide 设计研究已经系统证明了这些因素会影响 guide activity [6]。第三，即便在相同扰动标签下，单细胞也可能表现出不同的响应幅度和响应状态。例如 Adamson 等人在 Perturb-seq 研究中观察到，相同扰动下的细胞可出现 unfolded protein response 分支激活的分化模式 [2]。因此，仅使用“靶基因身份”作为条件变量可能不足以刻画真实的 CRISPRi 响应；有效扰动强度本身也应被显式建模。

已有计算方法已经显著推动了单细胞扰动预测的发展，但对“未观测的细胞级扰动强度”处理仍不充分。scGen 使用变分自编码器和潜空间向量运算预测 out-of-sample perturbation response，说明扰动可以近似表示为低维表达流形中的方向性偏移 [7]。Compositional perturbation autoencoder（CPA）进一步将扰动、协变量和剂量组合纳入模型，可以预测未见过的剂量、细胞类型和扰动组合 [8]。GEARS 利用基因关系图和 perturbation embedding 预测未见过的单基因或多基因扰动转录结果 [9]。CellOT 将单细胞扰动响应建模为 control 与 perturbed 分布之间的神经最优传输映射，强调了未配对单细胞分布之间的异质性响应 [10]。scGPT 等 foundation model 则表明，大规模预训练模型能够学习基因和细胞表示，并用于扰动响应预测和基因网络推断 [11]。这些方法分别从潜空间方向、剂量组合、基因图先验、分布映射和大规模预训练角度推进了扰动建模，但多数方法仍将遗传扰动主要视为离散标签，或在剂量建模时依赖外部已知剂量信息。对于 CRISPRi screen 中常见的细胞级 repression efficiency 不可直接观测的问题，现有模型通常没有显式推断一个可解释的连续强度变量。

基于这一缺口，我们提出一个面向 CRISPRi 扰动转录组预测的条件生成模型，其核心创新是引入扰动强度推断头。模型首先通过语义编码器将单细胞表达谱映射为表示背景细胞状态的低维语义向量；随后为每个靶基因学习一个基础扰动向量，用于表示该基因扰动在潜空间中的主要响应方向；最后，强度推断头根据细胞语义状态和靶基因扰动向量估计细胞级扰动强度，并将该强度与基础扰动向量相乘，得到最终扰动偏移。通过这一设计，模型将三个因素显式分离：基础细胞状态、靶基因特异的响应方向，以及该细胞中实际生效的扰动强度。

扰动强度推断头在生物学和建模上都是可行的。生物学上，CRISPRi 的有效 repression level 本就不是固定常数，而是由 sgRNA 活性、细胞状态和实验噪声共同决定 [6]。建模上，该设计与 scGen 的潜空间扰动向量思想 [7] 和 CPA 的剂量响应建模思想 [8] 有直接联系，但进一步面向 CRISPRi 中“真实有效剂量未被直接观测”的场景，将强度作为一个从表达背景和扰动嵌入中推断出的隐变量。与简单地为每个 sgRNA 指定固定强度相比，细胞级强度推断能够更好地解释同一靶基因、同一 guide 下仍然存在的响应异质性；与仅使用扰动标签的模型相比，它也为反事实模拟提供了连续控制轴。

不过，强度推断头也带来可辨识性风险。如果不加约束，模型可以通过同时缩放基础扰动向量和强度标量来产生相似的生成结果，使 $s_i$ 失去可解释性。因此，我们在训练目标中加入三类约束：第一，将对照细胞锚定到零扰动强度，使强度变量具有明确零点；第二，使用基于观测响应幅度代理的排序损失，使推断强度与表达响应强弱保持单调关系；第三，使用强度熵正则，避免强度值过早坍缩到全 0 或全 1。这些约束的目标不是得到绝对物理意义上的 knockdown percentage，而是学习一个可比较、可控制的相对有效扰动强度。

在生成模块上，我们采用条件扩散模型作为解码器。去噪扩散概率模型通过学习逐步加噪过程的逆过程，已经成为高质量生成建模的重要框架 [12]；DDIM 在保持相同训练目标的基础上提供了更高效的确定性采样方式 [13]。在单细胞领域，scDiffusion 等工作已经显示，条件扩散模型可以用于生成高质量的单细胞表达数据 [14]。对于 CRISPRi 响应建模，扩散解码器的优势在于它不只输出一个平均表达状态，而是能够在给定扰动后潜变量的条件下生成表达分布，从而保留单细胞转录响应中的残余异质性。结合扰动强度推断头后，模型可以同时支持两类任务：对观测到的扰动细胞自动估计有效扰动强度，以及在给定对照细胞和目标基因时，通过手动调节强度变量生成不同 knockdown level 下的反事实表达谱。

综上，本研究试图弥合 Perturb-seq 实验数据与现有扰动预测模型之间的一个关键缺口：CRISPRi 扰动标签并不等同于真实生效强度。通过将细胞背景、基因扰动方向和细胞级扰动强度显式分解，并使用条件扩散模型生成完整转录组，本模型为分析 CRISPRi efficiency heterogeneity 和进行可控反事实扰动模拟提供了一个统一框架。

## 可行性与文献依据说明

扰动强度推断头可以作为本模型的核心创新。它的合理性来自两方面：实验上，CRISPRi sgRNA 活性确实存在 guide-to-guide 和 context-dependent variation [6]；计算上，scGen 证明了潜空间扰动方向的可用性 [7]，CPA 证明了剂量或强度变量对扰动响应预测有价值 [8]。本模型的区别在于：它不假设有效强度已知，而是将其作为细胞级隐变量推断。

需要在论文中主动承认的主要风险是 identifiability。强度 $s_i$ 和基础扰动向量 $\boldsymbol{\delta}^{\mathrm{base}}_g$ 的尺度可能互相补偿，因此 methods 中的 control anchoring、response ranking 和 entropy regularization 是必要设计。建议在实验中加入 ablation：去掉强度头、去掉排序损失、去掉对照锚定、将强度改为 guide-level 固定参数，以及用普通 MLP/VAE decoder 替代 diffusion decoder。

扩散解码器有文献基础，但需要用实验支撑它对本任务的必要性。DDPM/DDIM 是经典生成模型基础 [12,13]，scDiffusion 说明 diffusion 可以适配单细胞表达生成 [14]。但审稿人可能会质疑其复杂度，因此建议比较 diffusion decoder 与较简单 decoder 在差异表达基因恢复、表达分布相似性、反事实插值平滑性和高强度/低强度生成稳定性上的表现。

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
