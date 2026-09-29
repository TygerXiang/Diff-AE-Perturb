# Diff-AE-Perturb 为什么没有超过简单基线：架构与条件接口诊断

日期：2026-09-28 · 被诊断 checkpoint：`Diff-AE-Perturb/runs/gse132080_target_200ep/checkpoint_best.pt`（epoch 198）
诊断脚本：`scperturbench_diffae_architecture.py` · 原始输出：`.scperturbench/results/gse132080_6metrics/gse132080/diffae/seed7/architecture_diagnostic.json`

---

## 0. 先更正我上一轮的一个错误结论

我上一轮说过「Squidiff 自己的预处理 = benchmark 已经在用的预处理，两者 1019/1019 完全相同」。**这个结论站不住**，你的质疑是对的。原因是我的移植并不独立：

`scperturbench_squidiff_prep.py` 里的细胞 QC 阈值（`total_counts≥1000`、`200≤n_genes≤10000`、`pct_mt<20`）、基因过滤（`min_cells≥3` + 去 MT-/RPS/RPL）、以及**训练/测试细胞集合**，我全部照抄的是**你项目里的** `preprocess_gse132080.py` 与 `benchmark split 文件`。所以它当然和你的矩阵一致——那是循环论证，不是独立证据。

把两套配方各自**按自己的文档**跑一遍（`tools/compare_squidiff_vs_project_prep.py`，从 raw 10x 开始）：

| | 你的项目 `preprocess_gse132080.py` | Squidiff `prep_drug_screen_data.ipynb` 族 |
|---|---|---|
| 细胞 QC | `total_counts≥1000`、`200≤n_genes≤10000`、`pct_mt<20` | 只做 `filter_cells(min_genes=300)`（无 counts/pct_mt 门槛） |
| 基因过滤 | `min_cells≥3` 且 **去 MT-/RPS/RPL** | `filter_genes(min_cells=3)` 且 **去 MT-/RP-** |
| 剩下细胞 | **19 579** | **23 608** |
| HVG 前基因数 | 19 780 | 20 261 |
| panel（HVG1000 ∪ 强留靶基因） | **1 019**（= benchmark） | **1 022** |
| HVG 集合 | 1000 | 1000 |
| **两者 panel 相同？** | **否** | |
| **panel 交集** | **780 / 1019（77%）** | |
| **HVG∩HVG** | **761 / 1000** | |

也就是说：**两套预处理并不一样**，细胞集差约 20%，基因 panel 只重合 77%。

还有一层我上一轮没查清的：Squidiff 基因扰动那条链的第一步（GEARS 的 `perturb_processed.h5ad`）在**镜像里根本不存在**。我把 `gears` 包里 `pertdata.py` / `data_utils.py` 全 grep 了一遍，**没有任何 `normalize_total` / `log1p` / `highly_variable_genes` / `filter_genes` 调用**：`new_data_process()` 只算 DE 基因，要求调用方传入已经处理好的 `adata`。所以 Squidiff 基因扰动输入的**确切归一化尺度我无法从镜像或 `Squidiff_reproducibility` 仓库验证**——我之前说的 "log1p CP10K" 是从 GEARS 论文惯例推的，不是从代码读出来的。

**修正后的正确表述**：我在 §1–§4 里做的「按 Squidiff 配方重跑」只忠实覆盖了 `prep_*` 家族里**有代码**的那部分（QC/HVG/强留基因），并没有复刻 GEARS 侧的尺度；而 `_repro` 那一版实际用的是你项目的细胞集与基因集。这两点都写进了 `reports/squidiff_repro_preprocessing.md`，我下面也会把它同步过去。

好消息是：**这不改变主结论**，因为 Squidiff 的失败点已定位到上游采样代码（`pred_xstart.clamp(0,)`），与基因选择无关；而你的模型失败点（本报告）更是完全在模型侧。

---

## 1. 结论

**你的模型不是「训得不好」，而是「条件通路在设计上被短路了」。**

一句话：`z_pert = z_sem`（`z_sem = encoder(x0)`，x0 就是去噪目标本身），所以训练时 denoiser 拿到的是**答案的编码**；而每个扰动唯一的条件信息 `delta`（那个 `nn.Embedding`）**从来没有进入过 denoiser**。到了推理，条件接口换成 `encoder(对照细胞) + delta`，于是 `delta` 必须独自承担全部条件信息——但它①尺度只有真实潜移位的 **1/50**，②从未被训练过。结果就是输出 ≈ 对照均值，六指标退化成 `control_mean`。

| 现象（benchmark 实测） | 本诊断给出的机制 | 证据 |
|---|---|---|
| `pcc_delta@100 = 0.028`，接近 0 | 生产条件接口下 delta 相关只有 **0.021** | §3.4 实验 E |
| `mse`/`e_distance`/`wasserstein`/`kl` ≈ `control_mean` | 条件信号（预测的跨条件 sd = 0.00066）比「换一批对照细胞」的噪声（0.0111）还小 **17×** | §3.5 实验 C |
| 训练/验证 loss 只有 0.0231（数据方差是 0.388） | 条件里带着答案：**纯噪声步**上也能以 corr 0.969 复原 x0 | §3.1 实验 A |
| 模型「看起来能用」的指标（重建类）都不差 | 它本质是个 autoencoder，重建本来就该好 | §3.6 |

---

## 2. 两个项目的架构对照

| 维度 | Squidiff（`Squidiff/MLPModel.py`） | Diff-AE-Perturb（checkpoint = commit `a8c355f`） |
|---|---|---|
| 语义编码器 | `EncoderMLPModel`：1019→2048→2048→60，BatchNorm | `SemanticEncoder`：1019→512→256→**128**，BatchNorm |
| 去噪主干 | `MLPModel`：`input_layer(1019→2048)` + **3 × `MLPBlock(2048)`**；每块 `h = SiLU(LN(fc1(x))) + time_dense(emb) + zsem_dense(z_sem)`；`output_layer(2048→1019)` | `ConditionalDenoiserMLP`：`Linear([x_t ; t_emb ; z_pert] = 1275 → 1024) → ReLU → Linear(1024→1024) → ReLU → Linear(1024→1019)` |
| 条件如何注入 | `z_sem` **相加**进每一层（残差式），时间嵌入同样相加 | `z_pert` 与 `x_t`、`t_emb` **拼接**后进第一层 |
| 参数量（主干） | ~57.2 M | ~4.1 M（checkpoint 16.5 MB） |
| 预测目标 | **eps** | **x0** |
| 训练损失 | `MSE(epŝ, eps)` | `MSE(x̂0, x0)` + `0.1 · (1 − cos)` 同靶 delta 方向对齐；`λ_semantic_align = 0.0`（关） |
| **训练时 denoiser 的条件** | `z_mod = z_sem = enc(x0)` | `z_pert = z_sem = enc(x0)`（**同一个 leak**） |
| **推理时 denoiser 的条件** | `enc(目标条件的真实细胞)` —— **与训练同分布** | `enc(对照细胞) + delta_target` —— **与训练不同分布** |
| 扰动身份条件 | 不使用 | 设计上要用 `delta`，但训练时没接进 denoiser |
| 优化器 / 日程 | AdamW 1e-4，**线性退火到 0**，EMA 0.9999，batch 64 | AdamW 1e-4，**无日程**（常数），**无 EMA**，batch 256 |
| 训练规模 | ~1e5 步 | 200 epochs（best epoch 198） |

**关键差异只有一个，但它是致命的**：Squidiff 的 `z` 在训练和推理时来自**同一类对象**（某个真实细胞的编码），所以那个 "encoder 泄露答案" 的结构在推理时仍然成立；你的模型想把它变成**反事实预测器**——推理时 `z` 要由「对照编码 + 一个学习到的偏移」**构造**出来。这要求 `delta` 恰好把对照的潜表示搬到目标扰动的潜表示邻域，而当前训练没有任何信号去逼它做到这一点。

---

## 3. 逐条证据

### 3.1 训练时 denoiser 拿到的是「答案的编码」

**代码**（`git show a8c355f:src/perturbdiff/modules.py`，第 118–125 行）：

```python
def forward(self, z_sem, perturbation_ids):
    delta_base = self.embedding(perturbation_ids)
    delta = delta_base
    z_basal = z_sem - delta
    z_pert = z_sem          # ← 条件里没有 perturbation embedding
    return {...}
```

**代码**（`git show a8c355f:src/perturbdiff/model.py`，第 146–149 行）：

```python
condition  = self.encode_condition(x0, perturbation_ids)   # 内部就是上面的 forward
x_t, noise = self.diffusion.q_sample(x0, timesteps, noise=noise)
prediction = self.denoiser(x_t, timesteps, condition["z_pert"])   # z_pert = encoder(x0)
target     = x0 if self.diffusion.prediction_type == "x0" else noise
```

`encode_condition` 在 `a8c355f` 里就是 `return {**pert}`——**连反事实那条分支都不存在**。HEAD 版本更明确了这一点（`model.py:147-152`）：

```python
if self.guide_doser is None:
    ...
    return {**pert}        # ← 没有 guide encoder 时，整条 counterfactual 重参数化被跳过
```

而这个 checkpoint **正是** `guide_doser is None`（state_dict 只有 `encoder`/`perturbation`/`denoiser`/`diffusion` 四组，没有 `guide_strength.*`、`guide_doser.*`、`activity_condition.*`；`compatible_config_from_state` 会把它补成 `enabled=False`）。

**实测**（实验 A）：取 512 个真实细胞 x0，在**纯噪声步** t∈[800,1000) 送进 denoiser（此时 x_t 里没有任何 x0 信息），比较四种条件：

| 条件 z | x0 MSE | x0 corr |
|---|---|---|
| **`self` = encoder(x0)**（训练时的条件） | **0.0202** | **0.969** |
| `permuted` = encoder(另一个细胞) | 0.0980 | 0.850 |
| `zero` | 0.0622 | 0.904 |
| `control_mean` = 对照编码均值 | 0.0605 | 0.906 |
| 参照：直接预测对照均值 | 0.0597 | — |
| 参照：直接预测 0 | 0.3883 | — |

**在纯噪声上能把 x0 复原到 corr 0.969、MSE 0.020（比「预测对照均值」的 0.060 还好 3 倍）**，只可能是条件里带着答案。而且 `permuted`（0.098）比 `zero`/`control_mean`（0.060）**更差**——说明 denoiser 已经学会「信任并读出 z」，而不是学着保留退路。训练 loss 0.0231 与 `self` 的 0.020 量级一致，即**模型正是沿着这条泄露通路被优化的**。

### 3.2 条件偏移在训练时从未进入 denoiser

由 §3.1 的代码可见：`delta` 在训练里只出现在 `z_basal = z_sem − delta`，而 `z_basal` 只被 `semantic_align_loss` 使用——而该系数 **`λ_semantic_align = 0.0`**（checkpoint 里读出来就是 0）。唯一真正作用于 `delta` 的是 `λ_direction_align = 0.1` 的 `(1 − cos)`，它在测试集上的值是 **2.0e-8**（≈0，且只在同一 batch 里出现 ≥2 个同靶 guide 时才激活）。

所以 `delta` 在 200 个 epoch 里几乎收不到梯度。

### 3.3 `delta` 的尺度停留在初始化（实测）

| 量 | 数值 |
|---|---|
| 训练后编码器的真实潜移位 `‖Δ_real(c)‖ = ‖mean(z_sem\|c) − mean(z_sem\|对照)‖` | **1.36 – 5.98** |
| checkpoint 里学习到的 `‖delta(c)‖` | **0.023 – 0.180** |
| 比值 `‖delta‖ / ‖Δ_real‖` | **0.013 – 0.039（中位 0.021，即小 ~48×）** |
| `cos(Δ_real, delta)` | 0.003 – 0.456（多数 0.1–0.35） |
| 对照细胞编码的**细胞间**散布 `‖z_sem − z̄‖` | **9.06 – 12.42**（对照池 9.18） |

**为什么停在初始化尺度？** 因为初始化的时机不对。`a8c355f` 的 `initialize_from_latent_shift`（`modules.py:95-113`）就是 `delta = target_means − control_mean`，**用当时那个编码器**算的：

```python
delta = target_means.clone() - control_mean.unsqueeze(0)
delta[0].zero_()
self.set_base_embeddings(delta)      # 直接 copy，没有任何缩放
```

`train_with_split.py` 在**训练开始前**调用它，此时编码器还是随机初始化的。我复现了这一步（实验 F，同 seed 重建一个未训练模型再算潜移位）：

| | 中位数 `‖delta‖` |
|---|---|
| 用**未训练**编码器算出的初始化值 | **0.0689**（0.058–0.104） |
| checkpoint 里的值 | **0.0533**（0.030–0.180） |
| 用**训练后**编码器算出的真实值 | **≈2.9**（1.36–5.98） |

**checkpoint 的 delta 与「未训练编码器的初始化」同一量级，而比「训练后编码器的真实潜移位」小约 50 倍。** 训练把编码器的输出尺度从 ~0.07 级放大到 ~9.2 级，而 delta 表既没有梯度（§3.2）也没有任何机制跟着放大；`(1 − cos)` 是**尺度不变**的，AdamW 的 `weight_decay=0.0` 也不会缩小它——所以它就悬在初始化尺度上不动。

### 3.4 推理接口与训练接口不是同一个分布（决定性实验）

固定同一个 denoiser，只换条件**接口**，用 benchmark 自己的口径（delta 相关 = `pcc_delta` 的分子）度量：

| 条件 | 训练接口：`z = enc(该扰动的真实细胞)` | 生产接口：`z = enc(对照细胞) + delta` |
|---|---|---|
| ALDOA | 0.622 | 0.276 |
| CAD | 0.886 | 0.225 |
| DBR1 | 0.613 | 0.061 |
| DUT | 0.752 | 0.146 |
| EIF2S1 | 0.962 | −0.097 |
| GATA1 | 0.986 | −0.101 |
| GINS1 | 0.892 | 0.001 |
| GNB2L1 | 0.914 | −0.139 |
| HSPA5 | 0.986 | −0.179 |
| HSPA9 | 0.975 | −0.104 |
| HSPE1 | 0.754 | 0.068 |
| POLR1D | 0.940 | 0.120 |
| POLR2H | 0.788 | −0.184 |
| RPS14 | 0.511 | 0.073 |
| TUBB | 0.762 | 0.145 |
| **均值** | **0.823** | **0.021** |

**生产接口的 0.021 与 benchmark 报出的 `pcc_delta = 0.028` 完全对上。** 同时，训练接口能到 **0.823**——说明**网络本身完全有能力表达扰动响应**，被毁掉的是条件接口，不是表征能力。

### 3.5 条件信号被对照细胞的噪声淹没

固定 32 个对照细胞、15 个条件，用生产路径（50 步 DDIM）采样：

| 量 | 数值 |
|---|---|
| 预测的**跨条件** sd（每基因） | **0.00066** |
| 真值的跨条件 sd | **0.0556**（是预测的 **84×**） |
| 只换一批对照细胞引起的 sd | **0.0111** |
| ⇒ 条件效应 / 对照噪声 | **0.06（条件效应比噪声小 17×）** |

把 delta 人为放大做扫描（同一条件、只乘倍数）：

| delta 倍数 | 0× | 1× | 10× | 100× | 1000× |
|---|---|---|---|---|---|
| 输出相对 0× 的 L2 位移 | 0 | **0.027** | 0.169 | 1.635 | 13.94 |

结论：denoiser **能**响应 delta（1000× 时位移 13.9），但在训练出来的尺度上只移动 0.027——**比换一批对照细胞的影响还小**。这与 §3.3 的「小 50×」一致，也给出了修复方向：把 delta 放到正确量级应当能恢复大部分跨条件差异。

### 3.6 六个指标因此退化成 `control_mean`

| metric | diffae | control_mean |
|---|---|---|
| mse@100 | 0.143308 | 0.140728 |
| pcc_delta@100 | 0.028314 | −0.012392 |
| e_distance@100 | 2.016981 | 1.802712 |
| wasserstein@100 | 43.882389 | 49.622519 |
| kl_divergence@100 | 13.821154 | 13.348100 |
| common_degs | 0.036309 | 0.000000 |

六个指标里五个与 `control_mean` 同级——这正是「输出 ≈ 对照均值」应有的样子。**所以你的模型没有输给 GEARS/scGen/linear，它是输给了 `control_mean` 这条恒等基线。**

---

## 4. 为什么 Squidiff 在同样的骨架下有 0.34

同一个 DiffAE 骨架，Squidiff 的 `pcc_delta` 是 0.337（虽然也远低于基线，但显著非零），差别不在骨架而在**条件的来源**：

- **Squidiff**：训练 `z_mod = enc(x0)`；推理 `z_mod = enc(目标条件的真实细胞)`。两侧都是「某个真实细胞的编码」，是同一分布，denoiser 不需要外推 → 输出至少能落在目标条件附近。
- **Diff-AE-Perturb**：训练 `z_pert = enc(x0)`；推理 `z_pert = enc(对照细胞) + delta`。**要求 delta 独自把对照潜表示搬 2.9 个单位的距离（§3.3），而它只有 0.053 且没有梯度。** 于是输出 ≈ 对照。

换个说法：Squidiff 是「把编码器当条件用的自编码器」；你想要的「骨架 + 一个条件项」是一种**反事实**模型，而当前实现只把反事实接在了推理侧——训练侧根本没有对应的监督信号。

---

## 5. 顺带发现的数据口径问题

`preprocess_gse132080.py` 先做基因过滤（**去掉所有 `RPS*`/`RPL*`**），再做靶基因强留（`adata.var_names.isin(target_genes)`）。因为强留只能「保住已经存在的基因」，所以任何在过滤阶段被删掉的扰动靶基因**再也回不来**：

```
perturbation targets ABSENT from the 1019-gene panel: ['RPS14','RPS15','RPS18','RPL9']
evaluated conditions whose own target gene is absent: ['RPS14']
```

25 个扰动靶基因里有 4 个（全部是核糖体基因）不在基因空间里，其中 **RPS14 是 15 个被评测条件之一**——也就是说 RPS14 这个条件的评测里，模型连被扰动的那个基因都看不到。修法很简单：把靶基因强留放到核糖体过滤**之前**，或在核糖体过滤里豁免靶基因（`preprocess_gse132080.py:170-198`）。

---

## 6. 建议的修复（按预期收益排序）

### ① 让 `delta` 进入训练时的 denoiser，并且**在编码器训练完之后**再定标（最高优先）

两处最小改动，效果互补：

**(a) 打通条件通路** —— `encode_condition` / `PerturbationModule.forward` 里 `z_pert = z_sem` 改成：

```python
z_control = z_sem[perturbation_ids == 0].mean(0, keepdim=True)   # 或用对照细胞编码
z_pert = torch.where(perturbation_ids.unsqueeze(-1) == 0,
                     z_sem,
                     z_control + delta)
```

并且**去掉 `guide_doser is None` 的早退短路**（HEAD `model.py:147-152`），否则在没有 guide encoder 的配置下这条路永远走不到。这样 `delta` 会通过 denoiser 的损失直接收到梯度。

**(b) 修正初始化时机** —— 现在 `initialize_from_latent_shift` 用的是随机编码器（§3.3）。两个选择：训练前不初始化（`--skip-delta-init`，让梯度自己学），或者在**训练结束后**用训练好的编码器重算一次。后者我已经实现并跑了对照实验（见 §7）。

### ② 训练/推理接口对齐

训练时也用 `z_control_batch_mean + delta`（而不是 `enc(x0)`），推理时也用**对照细胞的编码均值**（而不是逐细胞编码）——两边完全一致后，模型学到的就是「这个条件相对对照的潜移位」，这正是 benchmark 要评的量。代价是放弃逐细胞重建，但这本来就是反事实预测该做的事。

### ③ 训练质量（次要但便宜）

- 加 **LR 日程**（现在是常数 1e-4、200 epochs、无 warmup/decay）；
- 加 **EMA**（Squidiff 用 0.9999，你没有）；
- `prediction_type="x0"` 时高噪声步几乎不可学，建议改成预测 **eps**（或 x0 与 eps 混采，Squidiff 用的是 eps）并给不同 t 加权；
- 主干只有 4.1 M 参数、latent 只有 128 维、batch 256 无梯度裁剪——与 Squidiff 的 57 M / latent 60 / batch 64 相比不一定更差，但 **denoiser 的条件通路修好之前，这些都看不到效果**。

### ④ 数据口径

把靶基因强留挪到核糖体过滤之前（§5）。

---

## 7. 因果验证（把诊断变成可测的预测）

为了不让上面停在「解释」层面，我直接做了对照实验：**冻结 checkpoint，只重写 `delta` 表**，其余（编码器、denoiser、DDIM 50 步、15 条件、对照细胞预算、seed）完全不变，然后走原有打分流程。

| 方法 | `delta` 表的来源 |
|---|---|
| `diffae` | checkpoint 原值（= 未训练编码器的初始化尺度） |
| `diffae_delta_recompute` | 用**训练好的**编码器重算 `mean(z_sem\|c) − mean(z_sem\|对照)`（scGen 式，但时机正确） |
| `diffae_delta_x50` | 原值 × 50（补偿 §3.3 的尺度差） |

复现：`tools/run_diffae_delta_repair.sh`；适配器开关 `--diffae-delta-mode {checkpoint,rescale,recompute}`。

**结果**（15 条件均值，seed 7；括号是该 metric×budget 单元在 14 个方法中的排名）：

| metric | `diffae`（原样） | `diffae_delta_x50` | **`diffae_delta_recompute`** | control_mean | 最好基线 |
|---|---|---|---|---|---|
| mse@100 ↓ | 0.143308 (10) | 0.136338 (8) | **0.040743 (4)** | 0.140728 | 0.020826 gears |
| **pcc_delta@100 ↑** | 0.028314 (11) | 0.240854 (8) | **0.884925 (2)** | −0.012392 | 0.895589 gears |
| e_distance@100 ↓ | 2.016981 (9) | 1.965989 (8) | **0.775817 (3)** | 1.802712 | 0.458947 gears |
| wasserstein@100 ↓ | 43.882389 (8) | 42.980988 (7) | **32.707175 (2)** | 49.622519 | 27.299535 scgen |
| kl_divergence@100 ↓ | 13.821154 (8) | 13.498325 (7) | **12.745897 (2)** | 13.348100 | 12.526211 scgen |
| common_degs ↑ | 0.036309 (12) | 0.056217 (8) | **0.118034 (5)** | 0.000000 | 0.267264 linear_ridge |
| mse@all ↓ | 0.016483 | 0.018585 | **0.005458 (4)** | 0.016079 | 0.002984 scgen |
| pcc_delta@all ↑ | 0.023067 | 0.192955 | **0.794822 (3)** | −0.011666 | 0.803045 linear_ridge |
| e_distance@all ↓ | 1.800163 | 2.014221 | **0.854295 (3)** | 1.440864 | 0.477923 linear_ridge |
| kl_divergence@all ↓ | 37.431762 | 37.725979 | **37.744988 (4)** | 39.784120 | 35.495660 scgen |

**怎么读这三行：**

1. **只改 `delta` 表、其余全部冻结，模型从「不如 control_mean」变成「10 个单元全部排 2–5 名」。**
   `pcc_delta@100 = 0.885`，仅次于 GEARS（0.896），**高于 scGen（0.864）与 linear_ridge（0.867）**；
   `wasserstein` 与 `kl_divergence` 都升到第 2，`e_distance` 第 3，`mse` 第 4。
   也就是说 §1 的诊断不是解释性说明，而是**可证伪并已被证实**的：表征能力一直都在，缺的只是条件偏移的正确取值。
2. **只放大 50 倍（方向不变）只能恢复一小部分**：`pcc_delta` 0.028 → 0.241，而重算 0.885。
   说明**尺度与方向都要对**——denoiser 期望的是**训练好的编码器**所定义的潜流形上的那个偏移，而不是随机初始化编码器留下的方向。
3. **公平性**：`recompute` 的偏移只用**这些条件的训练细胞**计算（`unseen-guide / seen-target` 协议下允许；scGen、Squidiff 用的是同一批信息），**没有触碰任何测试细胞**，不是泄漏。
4. **这不是「训练出来的」修复**：它只是把条件变量的值给对了。按 §6 ① ② 把条件通路接进训练损失之后，应当至少不差于这个数，而且大概率更好——因为它还能顺带修正方向的高频成分（当前 `recompute` 只对齐了条件均值）。
5. Top-1 计数（10 个单元）仍然是 scgen 4 / gears 3 / linear_ridge 3，**修复后的模型一次第一也没拿到**，但已经稳定在第一梯队。

---

## 8. 产物

```
scperturbench_diffae_architecture.py            本报告的诊断脚本（实验 A–F）
tools/compare_squidiff_vs_project_prep.py       两套预处理配方的独立对照（§0）
tools/run_diffae_delta_repair.sh                delta 修复的因果对照实验
tools/inspect_diffae_checkpoint.py              checkpoint 条件分支/尺度检查
logs/a8c355f_model.py, logs/a8c355f_modules.py  训练所用 commit 的关键源码（用于引用行号）
.scperturbench/inputs/gse132080_squidiff_repro/recipe_comparison.json   §0 的原始数字
.scperturbench/results/gse132080_6metrics/gse132080/diffae/seed7/architecture_diagnostic.json  实验 A–F 原始输出
```
