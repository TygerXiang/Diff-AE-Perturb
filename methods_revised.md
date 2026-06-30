# 方法（修订版）

本文档描述当前 `PerturbDiff` 实现对应的方法设计，并在最后讨论与当前结构自然兼容的扰动强度建模方案。相比早期版本，当前模型不再包含单独的细胞级 strength head，也不使用 strength regression、ranking、control anchor 或 entropy 正则。当前设计的核心是：在保留 sgRNA 级扰动标签的条件下，学习一个条件扩散模型；同一 target 的不同 sgRNA 可以共享相近的扰动方向，而 sgRNA embedding 的幅度可作为扰动强弱的候选表征。

## 1 符号约定

设第 \(i\) 个细胞的表达向量为 \(\mathbf{x}_{i,0}\in\mathbb{R}^{G}\)，其中 \(G\) 为输入表达矩阵中的基因数。扰动条件记为 \(g_i\in\{0,1,\ldots,K\}\)，其中 \(g_i=0\) 表示 control 或 non-targeting 条件，\(g_i>0\) 表示一个非对照扰动条件。当前实现中，扰动条件可以是 target gene，也可以是完整 sgRNA label；为了分析同一 target 下不同 sgRNA 的强弱差异，训练时使用 `target_strategy="raw"` 保留 sgRNA 级标签。

当需要表达“同一 target 的多个 sgRNA”时，另记 \(c_i\) 为 target-level 标签。若 \(g_i\) 是完整 sgRNA label，则多个不同的 \(g\) 可以对应同一个 target \(c\)。模型潜空间维度记为 \(d\)，语义编码器参数记为 \(\phi\)，条件去噪网络参数记为 \(\theta\)，扰动 embedding 表记为 \(W_{\mathrm{pert}}\)。

## 2 模型结构

当前模型由三部分组成：语义编码器、扰动 embedding 模块和条件扩散去噪器。语义编码器把表达谱压缩为低维 latent 表示；扰动模块为每个非对照条件维护一个可学习的 latent 扰动向量；去噪器在扩散时间步、带噪表达谱和条件 latent 的共同条件下预测 clean expression 或扩散噪声。

### 2.1 语义编码器

语义编码器 \(E_\phi\) 是一个多层感知器，当前实现包含线性层、batch normalization、ReLU 和可选 dropout。它把输入表达谱映射为 latent 表示：

$$
\mathbf{z}_i = E_\phi(\mathbf{x}_{i,0}), \qquad \mathbf{z}_i\in\mathbb{R}^{d}.
$$

这里的 \(\mathbf{z}_i\) 不是独立训练的 VAE latent，也没有显式 KL 先验；它是服务于条件扩散重构和反事实生成的确定性表示。

### 2.2 sgRNA 级扰动 embedding

扰动模块为每个非对照条件学习一个 latent 扰动向量：

$$
\boldsymbol{\delta}_g = W_{\mathrm{pert}}[g], \qquad g\in\{1,\ldots,K\}.
$$

control row 固定为零：

$$
\boldsymbol{\delta}_0=\mathbf{0}.
$$

在 sgRNA 级训练中，\(\boldsymbol{\delta}_g\) 表示某一条 guide 对 latent 状态的条件偏移。它同时包含方向和幅度两个部分：方向主要表示 target identity 相关的响应方向，幅度则自然成为 guide-level perturbation strength 的候选表征。当前模型没有额外引入 \(s_i\boldsymbol{\delta}_g\) 形式的细胞级强度头，因此不会把强度解释强行绑定到一个尚未验证的标量输出上。

### 2.3 scGen-style latent shift 初始化

训练开始前，模型使用当前初始化的编码器对训练集细胞计算 latent 表示，并用 condition mean shift 初始化扰动 embedding。设 \(\mathcal{I}_g=\{i:g_i=g\}\)，\(\mathcal{I}_0=\{i:g_i=0\}\)，则：

$$
\boldsymbol{\delta}_g^{(0)}
=
\frac{1}{|\mathcal{I}_g|}\sum_{i\in\mathcal{I}_g} E_\phi(\mathbf{x}_{i,0})
-
\frac{1}{|\mathcal{I}_0|}\sum_{i\in\mathcal{I}_0} E_\phi(\mathbf{x}_{i,0}).
$$

若某个 condition 在训练集中没有细胞，则该 condition 的初始化偏移为零。该初始化借鉴 scGen 的 latent vector arithmetic 思路，用平均 latent 差异提供一个合理的扰动方向起点，随后 embedding 在扩散重构目标下继续端到端更新。

### 2.4 训练时条件表示

对观测到的训练细胞，去噪器直接使用编码器输出 \(\mathbf{z}_i\) 作为条件 latent：

$$
\mathbf{z}^{\mathrm{cond}}_i=\mathbf{z}_i.
$$

同时，为了让 sgRNA 扰动向量具有“从 basal state 到 perturbed state 的偏移”含义，模型定义一个去扰动后的 basal latent：

$$
\mathbf{z}^{\mathrm{basal}}_i=\mathbf{z}_i-\boldsymbol{\delta}_{g_i}.
$$

这个 basal latent 不直接作为去噪器输入，而是用于可选的语义对齐损失。该设计避免在训练时把已观测 perturbed cell 的 latent 重新加一次 perturbation，同时仍然让 \(\boldsymbol{\delta}_g\) 在反事实生成时有明确的加性偏移解释。

### 2.5 条件扩散去噪器

前向扩散过程对表达向量加高斯噪声。令 \(t\in\{1,\ldots,T\}\)，\(\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s\)，则：

$$
\mathbf{x}_{i,t}
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_{i,0}
+
\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

条件去噪器 \(D_\theta\) 是一个 MLP。它接收带噪表达 \(\mathbf{x}_{i,t}\)、正弦时间步 embedding 和条件 latent \(\mathbf{z}^{\mathrm{cond}}_i\)，输出维度与表达向量相同。当前实现支持两种预测目标：

$$
D_\theta(\mathbf{x}_{i,t},t,\mathbf{z}^{\mathrm{cond}}_i)
\approx
\mathbf{x}_{i,0}
$$

或

$$
D_\theta(\mathbf{x}_{i,t},t,\mathbf{z}^{\mathrm{cond}}_i)
\approx
\boldsymbol{\epsilon}.
$$

在当前 GSE132080 实验中，`prediction_type="x0"` 更符合实现中的主要设置，即直接预测 clean expression。

## 3 训练目标

当前训练目标由扩散重构损失和两个可选正则项组成：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{diff}}
+
\lambda_{\mathrm{sem}}\mathcal{L}_{\mathrm{sem}}
+
\lambda_{\mathrm{dir}}\mathcal{L}_{\mathrm{dir}}.
$$

其中 \(\lambda_{\mathrm{sem}}\) 和 \(\lambda_{\mathrm{dir}}\) 分别对应 `lambda_semantic_align` 和 `lambda_direction_align`。当前实现不包含 strength regression loss、response-proxy ranking loss、control anchor loss 或 entropy regularization。

### 3.1 扩散重构损失

设模型输出为 \(\widehat{\mathbf{y}}_i\)，训练目标 \(\mathbf{y}_i\) 由 `prediction_type` 决定：

$$
\mathbf{y}_i =
\begin{cases}
\mathbf{x}_{i,0}, & \text{if prediction\_type}=\texttt{x0},\\
\boldsymbol{\epsilon}, & \text{if prediction\_type}=\texttt{eps}.
\end{cases}
$$

默认的扩散重构损失为加权均方误差：

$$
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{B}\sum_{i=1}^{B}
\frac{
\sum_{j=1}^{G} w_{g_i,j}(\widehat{y}_{i,j}-y_{i,j})^2
}{
\sum_{j=1}^{G} w_{g_i,j}
}.
$$

默认 \(w_{g,j}=1\)。如果提供 signature gene summary，则可对 condition 对应的 signature genes 使用更高权重；这只是重构层面的 gene weighting，不等价于扰动强度监督。

当前实现还支持一个可选的 GEARS-style Autofocus direction-aware loss，用于替代默认 MSE。该损失定义在 clean-expression 空间中；若模型使用 `prediction_type="eps"`，则先根据扩散公式将预测噪声转换为 \(\widehat{\mathbf{x}}_{i,0}\)。设训练集 control 细胞的表达均值为 \(\bar{\mathbf{x}}_{\mathrm{ctrl}}\)，Autofocus 项为：

$$
\mathcal{L}_{\mathrm{AF}}
=
\frac{1}{B}\sum_{i=1}^{B}
\frac{
\sum_{j=1}^{G}
w_{g_i,j}
\left|
\widehat{x}_{i,0,j}-x_{i,0,j}
\right|^{2+\gamma}
}{
\sum_{j=1}^{G} w_{g_i,j}
}.
$$

其中 \(\gamma\ge 0\) 控制对大误差基因的聚焦强度；当 \(\gamma=0\) 时，该项退化为表达空间的加权 MSE。Direction-aware 项比较预测响应和真实响应相对 control mean 的方向：

$$
\mathcal{L}_{\mathrm{DA}}
=
\frac{1}{B}\sum_{i=1}^{B}
\frac{
\sum_{j=1}^{G}
w_{g_i,j}
\left[
\mathrm{sign}(x_{i,0,j}-\bar{x}_{\mathrm{ctrl},j})
-
\mathrm{sign}(\widehat{x}_{i,0,j}-\bar{x}_{\mathrm{ctrl},j})
\right]^2
}{
\sum_{j=1}^{G} w_{g_i,j}
}.
$$

因此 GEARS-style 重构损失为：

$$
\mathcal{L}_{\mathrm{GEARS}}
=
\mathcal{L}_{\mathrm{AF}}
+
\lambda_{\mathrm{DA}}\mathcal{L}_{\mathrm{DA}}.
$$

最终的 \(\mathcal{L}_{\mathrm{diff}}\) 由训练参数 `reconstruction_loss` 决定：

$$
\mathcal{L}_{\mathrm{diff}}
=
\begin{cases}
\mathcal{L}_{\mathrm{MSE}}, & \text{if reconstruction\_loss}=\texttt{mse},\\
\mathcal{L}_{\mathrm{GEARS}}, & \text{if reconstruction\_loss}=\texttt{gears}.
\end{cases}
$$

GEARS-style loss 更强调强响应基因和相对 control 的变化方向，但其优化目标与整体表达 MSE 不完全一致。在当前实验中它作为可选替代项用于对比，而不是默认训练目标。

### 3.2 basal latent 语义对齐

当启用 \(\lambda_{\mathrm{sem}}>0\) 时，模型约束 perturbed cells 的 basal latent 接近 batch 内 control cells 的 basal latent 均值。设当前 batch 中 control 集合为 \(\mathcal{B}_0\)，非对照集合为 \(\mathcal{B}_p\)，则：

$$
\bar{\mathbf{z}}^{\mathrm{basal}}_0
=
\frac{1}{|\mathcal{B}_0|}
\sum_{i\in\mathcal{B}_0}
\mathbf{z}^{\mathrm{basal}}_i,
$$

$$
\mathcal{L}_{\mathrm{sem}}
=
\frac{1}{|\mathcal{B}_p|}
\sum_{i\in\mathcal{B}_p}
\left\|
\mathbf{z}^{\mathrm{basal}}_i-\bar{\mathbf{z}}^{\mathrm{basal}}_0
\right\|_2^2.
$$

该项的作用是让 \(\mathbf{z}_i-\boldsymbol{\delta}_{g_i}\) 更接近未扰动背景状态，从而提高扰动向量作为 latent shift 的可解释性。它只在 batch 中同时存在 control 和 perturbed cells 时计算。

### 3.3 同 target 的 sgRNA 方向对齐

当使用 sgRNA 级 label 时，同一个 target gene 通常对应多条 sgRNA。为了让这些 guide 的 embedding 共享相近的 target response 方向，同时保留不同幅度，模型可以启用方向对齐损失。对 batch 内属于同一 target 的 guide pair \((a,b)\)，计算其扰动向量余弦相似度：

$$
\cos(\boldsymbol{\delta}_a,\boldsymbol{\delta}_b)
=
\frac{
\boldsymbol{\delta}_a^\top\boldsymbol{\delta}_b
}{
\|\boldsymbol{\delta}_a\|_2\|\boldsymbol{\delta}_b\|_2
}.
$$

给定 margin \(m\)，方向对齐损失为：

$$
\mathcal{L}_{\mathrm{dir}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(a,b)\in\mathcal{P}}
\max\left(0,\,m-\cos(\boldsymbol{\delta}_a,\boldsymbol{\delta}_b)\right),
$$

其中 \(\mathcal{P}\) 是 batch 内同 target、非对照的 guide pair 集合。该项只约束方向，不直接约束 \(\|\boldsymbol{\delta}_g\|_2\)。因此，在方向被拉近后，embedding norm 可以更自然地承载同 target 内不同 sgRNA 的扰动强弱差异。

## 4 反事实生成

训练完成后，给定一个 control cell \(\mathbf{x}_{\mathrm{ctrl}}\) 和目标扰动 condition \(g\)，模型先计算 control latent：

$$
\mathbf{z}_{\mathrm{ctrl}}=E_\phi(\mathbf{x}_{\mathrm{ctrl}}).
$$

随后使用目标扰动 embedding 构造反事实条件 latent：

$$
\mathbf{z}_{\mathrm{cf}}
=
\mathbf{z}_{\mathrm{ctrl}}+\boldsymbol{\delta}_g.
$$

DDIM 采样从标准高斯噪声开始：

$$
\widehat{\mathbf{x}}_{\tau_S}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
$$

并沿时间序列

$$
\tau_S>\tau_{S-1}>\cdots>\tau_1>\tau_0=0
$$

进行确定性更新。在第 \(k\) 步，去噪器输出先被转换为 \(\widehat{\mathbf{x}}_0^{(k)}\) 和 \(\widehat{\boldsymbol{\epsilon}}^{(k)}\)。若模型直接预测 \(\mathbf{x}_0\)，则：

$$
\widehat{\mathbf{x}}_0^{(k)}
=
D_\theta(\widehat{\mathbf{x}}_{\tau_k},\tau_k,\mathbf{z}_{\mathrm{cf}}),
$$

$$
\widehat{\boldsymbol{\epsilon}}^{(k)}
=
\frac{
\widehat{\mathbf{x}}_{\tau_k}
-
\sqrt{\bar{\alpha}_{\tau_k}}\widehat{\mathbf{x}}_0^{(k)}
}{
\sqrt{1-\bar{\alpha}_{\tau_k}}
}.
$$

若模型预测噪声，则先按标准 DDPM 公式从预测噪声恢复 \(\widehat{\mathbf{x}}_0^{(k)}\)。确定性 DDIM 更新为：

$$
\widehat{\mathbf{x}}_{\tau_{k-1}}
=
\sqrt{\bar{\alpha}_{\tau_{k-1}}}\widehat{\mathbf{x}}_0^{(k)}
+
\sqrt{1-\bar{\alpha}_{\tau_{k-1}}}\widehat{\boldsymbol{\epsilon}}^{(k)}.
$$

最终输出为：

$$
\widehat{\mathbf{x}}_{\mathrm{cf}}
=
\widehat{\mathbf{x}}_{\tau_0}.
$$

## 5 扰动强度的当前解释与后续设计

当前实现没有显式细胞级强度变量 \(s_i\)。在现有结构下，更合理的强度解释是 guide-level 的：同一 target 下，不同 sgRNA 的扰动 embedding 被方向对齐项约束到相似方向，而 embedding norm \(\|\boldsymbol{\delta}_g\|_2\) 保留为可学习的幅度。因此，一个自然的强度候选为：

$$
A_g=\|\boldsymbol{\delta}_g\|_2.
$$

这一设计比早期的 strength head 更保守：它不假设每个细胞都能从表达背景中可靠推断一个独立强度，也不引入额外 ranking loss。当前实验中，\(A_g\) 与基于表达响应投影得到的 guide-level `response_proxy` 均值呈明显正相关，说明 embedding norm 已经可能捕捉到 sgRNA 有效扰动强弱。由于 `response_proxy` 是按 target gene 拟合并缩放的，它更适合验证同一 target 内的 guide 强弱排序，而不应被直接解释为跨 target 的统一绝对剂量。

后续如果要把扰动强度显式纳入模型，可以在不破坏当前结构的基础上加入一个可选标量调制项：

$$
\mathbf{z}_{\mathrm{cf}}(s)
=
E_\phi(\mathbf{x}_{\mathrm{ctrl}})
+
s\,\boldsymbol{\delta}_g,
\qquad s\in[0,1].
$$

其中 \(s\) 可以先作为用户指定的反事实控制变量，用于模拟从弱扰动到强扰动的连续响应；如果需要自动推断细胞级有效强度，再单独引入 strength head，并用外部或独立构造的 response proxy 进行监督或校准。这个扩展应作为后续模型版本，而不是当前方法的已实现组成部分。

## 6 当前实现对应关系

当前 `PerturbDiff` 实现与本文方法部分的对应关系如下：

| 方法模块 | 当前实现 |
| --- | --- |
| 语义编码器 | `src/perturbdiff/modules.py::SemanticEncoder` |
| sgRNA/condition embedding | `src/perturbdiff/modules.py::PerturbationModule` |
| latent shift 初始化 | `PerturbDiffusionModel.initialize_perturbation_from_latent_shift` |
| 前向扩散 | `src/perturbdiff/diffusion.py::GaussianDiffusion.q_sample` |
| 条件去噪器 | `src/perturbdiff/modules.py::ConditionalDenoiserMLP` |
| 加权重构损失 | `src/perturbdiff/losses.py::weighted_mse_loss` |
| GEARS-style Autofocus direction-aware loss | `src/perturbdiff/losses.py::gears_autofocus_direction_loss` |
| 语义对齐与方向对齐 | `src/perturbdiff/model.py::training_losses` |
| DDIM 反事实生成 | `PerturbDiffusionModel.sample_counterfactual` |

## 参考文献

1. Ho, J., Jain, A., and Abbeel, P. Denoising Diffusion Probabilistic Models. NeurIPS, 2020.
2. Song, J., Meng, C., and Ermon, S. Denoising Diffusion Implicit Models. ICLR, 2021.
3. Lotfollahi, M., Wolf, F. A., and Theis, F. J. scGen predicts single-cell perturbation responses. Nature Methods, 2019.
4. Roohani, Y., Huang, K., and Leskovec, J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. Nature Biotechnology, 2023. DOI: 10.1038/s41587-023-01905-6.
5. Li et al. Decoding heterogeneous single-cell perturbation responses. Nature Cell Biology, 2025. DOI: 10.1038/s41556-025-01626-9.
