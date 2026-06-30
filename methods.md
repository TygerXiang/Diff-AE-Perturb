# PerturbDiff 当前简化方法

当前实现采用一个接近 SquiDiff/scGen 思路的 conditional DDIM。模型不再包含扰动强度头，也不再使用 strength regression、ranking、control anchor、entropy 或 latent target-direction 约束。

## 1. 数据表示

每个细胞有表达向量：

```text
x_i in R^G
```

以及条件标签：

```text
g_i in {0, 1, ..., K}
```

其中 `g_i = 0` 表示 control/non-targeting 条件，`g_i > 0` 表示一个 perturbation condition。该 condition 可以是 target gene，也可以是具体 sgRNA。为了保留不同 sgRNA 信息，预处理和训练时推荐使用：

```text
--target-strategy raw
```

这样 `TP53_sg1` 和 `TP53_sg2` 会作为两个不同条件保留。以后如果要重新加入扰动强度，可以在 sgRNA 层面再建模。

## 2. 模型结构

模型包含四个组件。

1. Semantic encoder：

```text
z = E_phi(x)
```

2. Perturbation embedding table：

```text
delta_g = W[g]
```

control row 固定为零：

```text
delta_0 = 0
```

3. Conditional denoiser：

```text
D_theta(x_t, t, z_cond)
```

4. Gaussian diffusion / DDIM sampler。

## 3. scGen-style Delta 估计

训练开始前，先用当前 encoder 得到所有训练细胞的 latent representation：

```text
z_i = E_phi(x_i)
```

然后对每个 condition 估计 latent mean difference：

```text
delta_g = mean(z | g) - mean(z | control)
```

这就是当前版本唯一的 perturbation direction 初始化方式。它不除以 strength，不使用 response proxy，也不加入额外 target-direction loss。

在更复杂的数据集中，如果存在多个 cell type，可以按 scGen/SquiDiff 的做法先在每个 condition 内对 cell type 上采样到相同大小，再对样本量较大的 condition 下采样，使两个 condition 样本量匹配，最后计算 mean difference。当前 GSE132080/K562 设置基本是单一细胞背景，因此实现中使用 target/control 的整体 latent 均值差。

## 4. 条件 DDIM 训练

训练时对真实细胞表达加噪：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
```

条件 latent 为：

```text
z_cond = E_phi(x_0) + delta_g
```

denoiser 预测 clean expression 或 noise，取决于 `prediction_type`：

```text
D_theta(x_t, t, z_cond) -> x_0
```

或：

```text
D_theta(x_t, t, z_cond) -> eps
```

当前推荐使用 `prediction_type = x0`，因为单细胞表达谱的 no-change 成分很强，直接预测 clean expression 更稳定。

训练损失只有 diffusion reconstruction loss：

```text
L = MSE(prediction, target)
```

可选地，signature genes 可以在 gene 维度上加权，但这只是 reconstruction weighting，不是 strength supervision。

## 5. Counterfactual 生成

给定 control cell 和目标 condition：

```text
z_ctrl = E_phi(x_ctrl)
z_cf = z_ctrl + delta_g
```

然后用 DDIM 采样：

```text
x_pred = DDIM(D_theta, z_cf)
```

当前版本没有强度输入：

```text
x_pred = f(x_ctrl, g)
```

而不是：

```text
x_pred = f(x_ctrl, g, s)
```

这使模型更简单，也更适合先验证 conditional DDIM 是否能稳定超过 no-change/scGen baselines。扰动强度建模可以在 sgRNA 信息保留后作为下一阶段扩展。

## 6. 当前设计取舍

删除的内容：

- strength head
- strength regression loss
- response proxy ranking loss
- control anchor loss
- strength entropy
- latent mean-shift target constraint
- expression-space `s * delta_x_g`

保留的内容：

- encoder
- scGen-style latent delta estimation
- conditional denoiser
- DDIM sampling
- sgRNA-level label preservation

当前模型定位是：

```text
scGen-style latent delta + conditional DDIM decoder
```
