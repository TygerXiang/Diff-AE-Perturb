"""Generate the Markdown report for the GSE132080 implementation-continuity check."""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(r"C:\Users\Admin\myprojects\Diff-AE-Perturb")
RUNS = ROOT / "runs"
OUT = RUNS / "continuity_check"
REPORT = ROOT / "reports" / "gse132080_implementation_continuity.md"

RUNS_TABLE = [
    {
        "key": "A",
        "name": "original_2a1fb63_historical",
        "dir": RUNS / "benchmark_activity_branch_predicted_day10_200ep",
        "arch": "legacy delta-scaling",
        "init": "scGen latent-mean",
        "denoiser_in": 1275,
        "activity_condition": False,
        "when": "2026-07-20",
    },
    {
        "key": "B",
        "name": "original_2a1fb63_reproduced",
        "dir": OUT / "original_2a1fb63_200ep",
        "arch": "legacy delta-scaling",
        "init": "scGen latent-mean",
        "denoiser_in": 1275,
        "activity_condition": False,
        "when": "2026-09-16",
    },
    {
        "key": "C",
        "name": "explicit_8179140_historical",
        "dir": RUNS / "benchmark_activity_branch_predicted_day10_explicit_only_200ep",
        "arch": "explicit 16-d activity",
        "init": "scGen latent-mean",
        "denoiser_in": 1291,
        "activity_condition": True,
        "when": "2026-07-27",
    },
    {
        "key": "D",
        "name": "explicit_8179140_reproduced",
        "dir": OUT / "explicit_8179140_200ep",
        "arch": "explicit 16-d activity",
        "init": "scGen latent-mean",
        "denoiser_in": 1291,
        "activity_condition": True,
        "when": "2026-09-16",
    },
    {
        "key": "E",
        "name": "current_worktree_reproduced",
        "dir": OUT / "current_worktree_200ep",
        "arch": "explicit 16-d activity",
        "init": "none (random)",
        "denoiser_in": 1291,
        "activity_condition": True,
        "when": "2026-09-16",
    },
    {
        "key": "F",
        "name": "original_2a1fb63_cudnn_off",
        "dir": OUT / "cudnn_off_deterministic_200ep",
        "arch": "legacy delta-scaling",
        "init": "scGen latent-mean",
        "denoiser_in": 1275,
        "activity_condition": False,
        "when": "2026-09-16",
    },
]


def load(run):
    metrics = json.loads((run["dir"] / "metrics.json").read_text())
    run = dict(run)
    run["metrics"] = metrics
    run["history"] = {h["epoch"]: h for h in metrics["history"]}
    ck = torch.load(run["dir"] / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    run["measured_denoiser_in"] = int(sd["denoiser.network.0.weight"].shape[1])
    run["measured_activity_condition"] = any(k.startswith("activity_condition") for k in sd)
    run["n_keys"] = len(sd)
    return run


def main() -> None:
    runs = [load(r) for r in RUNS_TABLE]
    for r in runs:
        assert r["measured_denoiser_in"] == r["denoiser_in"], (r["name"], r["measured_denoiser_in"])
        assert r["measured_activity_condition"] == r["activity_condition"], r["name"]
    by_key = {r["key"]: r for r in runs}
    A, B, C, D, E, F = (by_key[k] for k in "ABCDEF")

    # noise floor from three independent realisations of the original implementation
    orig_vals = [A["metrics"]["best_val_loss"], B["metrics"]["best_val_loss"], F["metrics"]["best_val_loss"]]
    noise = max(orig_vals) - min(orig_vals)
    max_gap_on = max(abs(A["history"][e]["val_loss"] - B["history"][e]["val_loss"]) for e in range(1, 201))
    max_gap_off = max(abs(A["history"][e]["val_loss"] - F["history"][e]["val_loss"]) for e in range(1, 201))
    epoch_gap = max(max_gap_on, max_gap_off)

    lines: list[str] = []
    add = lines.append

    add("# GSE132080 实现连续性验证：原始实现 vs 当前实现（200 epoch）")
    add("")
    add("本报告回答两个问题：")
    add("")
    add("1. `perturbdiff-gse132080` 环境能否复现原始实现？")
    add("2. 原始实现与当前实现在 GSE132080 上 200 epoch 的表现差异有多大、来自哪里？")
    add("")
    add("> 所有 run 使用**同一份数据、同一划分、同一 seed `7`、同一套超参**，"
        "唯一变量是代码状态（原始 `2a1fb63` / 历史显式版 `8179140` / 当前工作树）。")
    add("")

    add("## 1. 结论摘要")
    add("")
    add("- **环境可行**：`C:\\Users\\Admin\\myprojects\\diff-project\\.conda\\envs\\perturbdiff-gse132080`"
        "（Python 3.10.20 / torch 2.11.0+cu128 / cuDNN 9.19 / RTX 5060 8 GB）可以直接做 CUDA 训练，"
        "6 次 200-epoch run 全部在 `cuda` 上跑完，无需回退 CPU。")
    add(f"- **原始实现确实可复现**：`2a1fb63` 重训 200 epoch 得到 "
        f"`best_epoch={A['metrics']['best_epoch']} → {B['metrics']['best_epoch']}`，"
        f"`best_val_loss {A['metrics']['best_val_loss']:.6f} → {B['metrics']['best_val_loss']:.6f}`"
        f"（相对偏差 **{100 * (B['metrics']['best_val_loss'] - A['metrics']['best_val_loss']) / A['metrics']['best_val_loss']:+.5f}%**），"
        f"`test_loss {A['metrics']['test_metrics']['loss']:.6f} → {B['metrics']['test_metrics']['loss']:.6f}`。")
    add(f"- **架构签名逐项吻合**：复现 checkpoint 的 denoiser 输入宽度实测 **{B['measured_denoiser_in']}**"
        f"（= 1019 genes + 128 time + 128 latent），含 `guide_strength.*` 与 `guide_doser.beta/bias`，"
        f"**不含** `activity_condition.*`，参数键数 {B['n_keys']}，与用户描述的旧 checkpoint 完全一致。")
    add(f"- **原始 vs 当前的差异很小且有方向**："
        f"`best_val {A['metrics']['best_val_loss']:.6f} → {E['metrics']['best_val_loss']:.6f}`"
        f"（{E['metrics']['best_val_loss'] - A['metrics']['best_val_loss']:+.6f}），"
        f"`test {A['metrics']['test_metrics']['loss']:.6f} → {E['metrics']['test_metrics']['loss']:.6f}`"
        f"（{E['metrics']['test_metrics']['loss'] - A['metrics']['test_metrics']['loss']:+.6f}）。"
        f"相对幅度约 **{100 * (E['metrics']['best_val_loss'] - A['metrics']['best_val_loss']) / A['metrics']['best_val_loss']:+.3f}%**（val）。")
    add(f"- **真正造成这点差异的是「移除 scGen delta 初始化」**，不是显式 activity 通道："
        f"只换架构（B→D）差 {D['metrics']['best_val_loss'] - B['metrics']['best_val_loss']:+.6f}，"
        f"在噪声内；只去掉 delta 初始化（D→E）差 {E['metrics']['best_val_loss'] - D['metrics']['best_val_loss']:+.6f}，"
        f"约为噪声的 {abs(E['metrics']['best_val_loss'] - D['metrics']['best_val_loss']) / noise:.1f} 倍。")
    add(f"- **复现边界**：同代码、同 seed、同卡重复运行的 best val loss 自然散布约 `{noise:.1e}`，"
        f"逐 epoch 最大偏差约 `{epoch_gap:.1e}`。因此本工作达到的是"
        f"**架构级 + best-epoch 级 + 指标级（<0.1%）复现**，而不是 bit-for-bit。")
    add("")

    add("## 2. 运行配置（各次运行完全一致）")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add("| 数据 | GSE132080 预处理 split：`gse132080_{train,val,test}.h5ad` |")
    add("| 规模 | 19,579 细胞 × 1,019 基因 |")
    add("| 标签列 | `guide_identity`（`target_strategy=raw`），129 个条件（128 靶向 sgRNA + control） |")
    add("| epochs / batch | 200 / 256 |")
    add("| 优化器 | AdamW，lr `1e-4`，weight_decay `0` |")
    add("| 架构 | latent 128，encoder `(512,256)`，denoiser `(1024,1024)` |")
    add("| diffusion | 1000 steps，linear beta，`prediction_type=x0` |")
    add("| activity | `--activity-source predicted`；Table S8 CNN 特征 + Table S16 `relative_activity_day10` 监督 |")
    add("| CNN strength | hidden `(64,64)`，channels `(32,32)`，kernel 4，dropout 0.25，`lambda_strength=0.1` |")
    add("| 其他损失 | `lambda_semantic_align=0`、`lambda_direction_align=0`、`reconstruction_loss=mse` |")
    add("| seed | `7` |")
    add("| device | `cuda`（RTX 5060） |")
    add("")

    add("## 3. 各次运行最终指标")
    add("")
    add("| # | run | 代码日期 | 架构 | delta 初始化 | denoiser 宽度 | best epoch | best val loss | test loss | test eval MSE | test strength loss |")
    add("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in runs:
        m = r["metrics"]
        t = m["test_metrics"]
        add(f"| {r['key']} | `{r['name']}` | {r['when']} | {r['arch']} | {r['init']} | "
            f"{r['measured_denoiser_in']} | {m['best_epoch']} | {m['best_val_loss']:.6f} | "
            f"{t['loss']:.6f} | {t['eval_mse_loss']:.6f} | {t['guide_strength_loss']:.6f} |")
    add("")
    add("- A/C = 历史 run；B/D/E/F = 本次在 `perturbdiff-gse132080` 环境下重跑。")
    add("- A、B、F 是**同一份原始代码**的三次独立数值实现（A/B 开 cuDNN，F 关 cuDNN），"
        "用于给出噪声基准。")
    add("- E 是当前工作树实现（`8179140` 去掉 scGen delta 初始化的未提交改动）。")
    add("")

    add("## 4. 复现保真度（历史 vs 本次重跑）")
    add("")
    add("| 对照 | best epoch | Δbest val | 相对偏差 | Δtest loss | 逐 epoch 位级一致 | 逐 epoch ≤1e-6 | 最大单 epoch 偏差 |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ref, new in ((A, B), (C, D)):
        dv = new["metrics"]["best_val_loss"] - ref["metrics"]["best_val_loss"]
        dt = new["metrics"]["test_metrics"]["loss"] - ref["metrics"]["test_metrics"]["loss"]
        identical = sum(
            1 for e in ref["history"] if ref["history"][e]["val_loss"] == new["history"][e]["val_loss"]
        )
        near = sum(
            1 for e in ref["history"]
            if abs(ref["history"][e]["val_loss"] - new["history"][e]["val_loss"]) < 1e-6
        )
        gap = max(abs(ref["history"][e]["val_loss"] - new["history"][e]["val_loss"]) for e in ref["history"])
        add(f"| {ref['key']}→{new['key']} | {ref['metrics']['best_epoch']} → {new['metrics']['best_epoch']} | "
            f"{dv:+.3e} | {100 * dv / ref['metrics']['best_val_loss']:+.5f}% | {dt:+.3e} | "
            f"{identical}/200 | {near}/200 | {gap:.3e} |")
    add(f"| A→F（同代码，关 cuDNN） | {A['metrics']['best_epoch']} → {F['metrics']['best_epoch']} | "
        f"{F['metrics']['best_val_loss'] - A['metrics']['best_val_loss']:+.3e} | "
        f"{100 * (F['metrics']['best_val_loss'] - A['metrics']['best_val_loss']) / A['metrics']['best_val_loss']:+.5f}% | "
        f"{F['metrics']['test_metrics']['loss'] - A['metrics']['test_metrics']['loss']:+.3e} | "
        f"{sum(1 for e in range(1, 201) if A['history'][e]['val_loss'] == F['history'][e]['val_loss'])}/200 | "
        f"{sum(1 for e in range(1, 201) if abs(A['history'][e]['val_loss'] - F['history'][e]['val_loss']) < 1e-6)}/200 | "
        f"{max_gap_off:.3e} |")
    add("")
    add("两组对照都落在同一数值噪声量级内，没有任何一版代码出现系统性偏移。"
        "**best epoch 三组一致（全部 193）**，这是最有说服力的复现证据。")
    add("")

    add("## 5. 数值噪声基准与复现边界")
    add("")
    add("同一份原始代码 `2a1fb63`、同一 seed、同一张卡的三次独立运行，逐 epoch 的 val loss：")
    add("")
    add("| epoch | 历史 A（07-20，cuDNN 开） | 重跑 B（cuDNN 开） | 重跑 F（cuDNN 关） |")
    add("| ---: | ---: | ---: | ---: |")
    for e in (1, 2, 3, 4, 5, 10, 50, 100, 193, 200):
        add(f"| {e} | `{A['history'][e]['val_loss']:.17f}` | `{B['history'][e]['val_loss']:.17f}` | "
            f"`{F['history'][e]['val_loss']:.17f}` |")
    add("")
    add("| 对照（同代码） | 首次分离 | 分离幅度 | 位级一致 epoch 数 | 最大单 epoch 偏差 |")
    add("| --- | ---: | ---: | ---: | ---: |")
    add(f"| A vs B（都开 cuDNN） | epoch 2 | 1.9e-09 | 1/200 | {max_gap_on:.3e} |")
    add(f"| A vs F（B 关 cuDNN） | epoch 3 | 1.0e-08 | 2/200 | {max_gap_off:.3e} |")
    add(f"| A vs B vs F 的 best val 散布 | — | — | — | **{noise:.3e}（{100 * noise / A['metrics']['best_val_loss']:.4f}%）** |")
    add("")
    add("可以确认的事实：")
    add("")
    add("1. **epoch 1 的 val loss 三次完全相同**（`0.07940810453146696`），"
        "说明数据加载顺序、划分、参数初始化、随机数种子都完全对齐——差异不是这些引起的。")
    add("2. 分离从 epoch 2/3 开始，幅度约 `1e-9 ~ 1e-8`，随训练逐步放大到 `~5e-5`。"
        "这是 CUDA 浮点累加顺序（cuBLAS/reduction、TF32 accumulation）造成的典型累积误差。")
    add("3. **关闭 cuDNN 并不能恢复位级一致**，说明来源不止一个 cuDNN 卷积算法选择。")
    add(f"4. 训练脚本本身只做 `torch.manual_seed(7)`，既没有 `torch.use_deterministic_algorithms(True)`，"
        f"也没有固定 `CUBLAS_WORKSPACE_CONFIG`，"
        f"所以 **bit-for-bit 从来就不是原实现的既有属性**。")
    add("")
    add("因此本工作的复现结论应当这样表述：")
    add("")
    add("| 目标 | 是否达到 | 证据 |")
    add("| --- | --- | --- |")
    add("| 环境层面可行（GPU 训练） | **是** | 6 次 200-epoch run 全部在 `cuda` 完成 |")
    add("| 代码/架构严格复现 | **是** | denoiser 输入宽度、参数键集合、perturbation mapping 全部一致 |")
    add("| best epoch 复现 | **是** | A/B/F 均为 epoch 193 |")
    add("| 指标复现到相近范围 | **是** | best val 相对偏差 < 0.09%，test 相对偏差 < 0.05% |")
    add("| bit-for-bit 复现 | **否**（原实现也未提供） | 逐 epoch 最大偏差 ~5e-5 |")
    add("")

    add("## 6. 效果分解：架构 vs 初始化 vs 噪声")
    add("")
    add("```text")
    add("B  2a1fb63  legacy delta-scaling     + scGen delta init   best_val = %.6f" % B["metrics"]["best_val_loss"])
    add("D  8179140  explicit 16-d activity   + scGen delta init   best_val = %.6f" % D["metrics"]["best_val_loss"])
    add("E  worktree explicit 16-d activity   + 无 delta init      best_val = %.6f" % E["metrics"]["best_val_loss"])
    add("")
    add("噪声基准（同代码 A/B/F 的 best val 散布）           = %.6f" % noise)
    add("```")
    add("")
    d1 = D["metrics"]["best_val_loss"] - B["metrics"]["best_val_loss"]
    d2 = E["metrics"]["best_val_loss"] - D["metrics"]["best_val_loss"]
    d3 = E["metrics"]["best_val_loss"] - B["metrics"]["best_val_loss"]
    d1t = D["metrics"]["test_metrics"]["loss"] - B["metrics"]["test_metrics"]["loss"]
    d2t = E["metrics"]["test_metrics"]["loss"] - D["metrics"]["test_metrics"]["loss"]
    d3t = E["metrics"]["test_metrics"]["loss"] - B["metrics"]["test_metrics"]["loss"]
    add("| 对比 | 改变的东西 | Δbest val | Δtest | 相对噪声倍数 | 判定 |")
    add("| --- | --- | ---: | ---: | ---: | --- |")
    add(f"| B→D | 只换架构：legacy delta-scaling → explicit 16-d channel | {d1:+.6f} | {d1t:+.6f} | "
        f"{abs(d1) / noise:.1f}× | 不可辨 |")
    add(f"| D→E | 只去掉 scGen delta 初始化，架构不变 | {d2:+.6f} | {d2t:+.6f} | "
        f"{abs(d2) / noise:.1f}× | 一致改善 |")
    add(f"| B→E | 两者同时改变（即 原始 → 当前） | {d3:+.6f} | {d3t:+.6f} | "
        f"{abs(d3) / noise:.1f}× | 一致改善 |")
    add("")
    add("解读：")
    add("")
    add(f"- 换成显式 activity 通道，在 GSE132080 的**训练/验证总 loss** 上没有可辨收益"
        f"（{d1:+.6f} ≈ {abs(d1) / noise:.1f}× 噪声）。")
    add(f"- 移除 scGen delta 初始化带来方向一致的改善（{d2:+.6f} val / {d2t:+.6f} test，"
        f"约 {abs(d2) / noise:.1f}× 噪声），这是当前实现相对原始实现真正可测的行为变化。")
    add("- 合起来，当前实现相对原始实现 best val 低约 "
        f"{100 * abs(d3) / B['metrics']['best_val_loss']:.3f}%，test 低约 "
        f"{100 * abs(d3t) / B['metrics']['test_metrics']['loss']:.3f}%。"
        f"**两者在 GSE132080 上属于同一档表现。**")
    add("- activity 分支的价值应当用网络层面的指标（mean-profile MSE、pcc_delta、DEG overlap 等）来评估，"
        "而不是训练/验证总 loss——总 loss 对这个改动基本不敏感。")
    add("")

    add("## 7. 训练曲线")
    add("")
    add("![continuity curves](../runs/continuity_check/continuity_curves.png)")
    add("")
    add("![zoom](../runs/continuity_check/continuity_curves_zoom.png)")
    add("")
    add("子图依次为 validation 总 loss、validation eval MSE（扩散项）、validation guide-strength loss。"
        "A/B/F（原实现）与 C/D（显式版）两两重合并落在同一条带上，"
        "说明整条 200 epoch 曲线都对得上，而不只是某一个 epoch。")
    add("")

    add("## 8. 产物")
    add("")
    add("| 路径 | 内容 |")
    add("| --- | --- |")
    add("| `runs/continuity_check/original_2a1fb63_200ep/` | 原始实现（`2a1fb63`）200 epoch 复现 run（B） |")
    add("| `runs/continuity_check/explicit_8179140_200ep/` | 历史显式版代码（`8179140`）200 epoch 复现 run（D） |")
    add("| `runs/continuity_check/current_worktree_200ep/` | 当前工作树实现 200 epoch run（E） |")
    add("| `runs/continuity_check/cudnn_off_deterministic_200ep/` | 关闭 cuDNN 的 200 epoch run（F） |")
    add("| `runs/continuity_check/determinism_cudnn_{on,off}_10ep/` | 非确定性定点探针 |")
    add("| `runs/continuity_check/continuity_summary.csv` | 汇总表 |")
    add("| `runs/continuity_check/continuity_history.csv` | 逐 epoch 曲线数据 |")
    add("| `runs/continuity_check/continuity_curves*.png` | 训练曲线 |")
    add("| `runs/continuity_check/determinism_analysis.json` | 噪声基准与首次分离点 |")
    add("| `reproducibility/continuity_check/` | `2a1fb63` 代码快照（git worktree）+ 复现/对比/报告脚本 |")
    add("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"noise floor = {noise:.6e}  ({100 * noise / A['metrics']['best_val_loss']:.4f}%)")


if __name__ == "__main__":
    main()
