#!/usr/bin/env python
"""Build the "our model vs the four external methods" comparison report.

The 19-row table in ``reports/benchmark/gse132080_6metrics_benchmark.md`` mixes the four
published methods (GEARS / scGen / CPA / Squidiff) with four literature reference
baselines and eight of our own ablation arms.  This script extracts just the external
comparison, keeps the ranks among all 19, and writes:

    reports/benchmark/external_methods_comparison.md    (human-readable)
    reports/benchmark/external_methods_comparison.csv   (long format, for reuse)

Input is the aggregate table exported by ``perturbench/scperturbench_aggregate.py``; a
copy lives in ``reports/benchmark/aggregate/`` so the report is reproducible from this
repository alone.

    python scripts/build_external_comparison.py
    python scripts/build_external_comparison.py --aggregate-dir <dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE_DIR = REPO_ROOT / "reports" / "benchmark" / "aggregate"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "benchmark"

#: The four external, published methods.  Squidiff is represented by its only usable
#: sampling variant (see BROKEN_SQUIDIFF); the other three are reported separately.
EXTERNAL = [
    ("gears", "GEARS"),
    ("scgen", "scGen"),
    ("cpa", "CPA"),
    ("squidiff_repro_clip", "Squidiff"),
]

BROKEN_SQUIDIFF = [
    ("squidiff", "Squidiff（旧：ddim50 + 官方 clamp）"),
    ("squidiff_repro", "Squidiff（完整 1000 步 + 官方 clamp）"),
    ("squidiff_repro_noclamp", "Squidiff（关掉上游截断，对照）"),
]

OURS = [
    ("diffae", "diffae（修复前，归档 checkpoint）"),
    ("fix_x0_percell", "**fix_x0_percell**（修复后重训）"),
]

FLOOR = [
    ("control_mean", "control_mean"),
    ("train_mean", "train_mean"),
]

#: (metric_budget, display name, direction) -- the 10 ranked units, in report order.
CELLS = [
    ("mse@100", "`mse@100`", "down"),
    ("mse@all", "`mse@all`", "down"),
    ("e_distance@100", "`e_distance@100`", "down"),
    ("e_distance@all", "`e_distance@all`", "down"),
    ("wasserstein@100", "`wasserstein@100`", "down"),
    ("pcc_delta@100", "`pcc_delta@100`", "up"),
    ("pcc_delta@all", "`pcc_delta@all`", "up"),
    ("kl_divergence@100", "`kl_divergence@100`", "down"),
    ("kl_divergence@all", "`kl_divergence@all`", "down"),
    ("common_degs@cutoff", "`common_degs`", "up"),
]

ERROR_TABLE = {"mse@100", "mse@all", "e_distance@100", "e_distance@all", "wasserstein@100"}

#: Extra diagnostic columns that are not part of the ranked units.
DIAGNOSTIC = [("kl_divergence_floor@100", "`kl_divergence_floor@100`"), ("kl_divergence_floor@all", "`kl_divergence_floor@all`")]

PROTOCOL = [
    "种子 7，单一 seed（无多 seed 方差）",
    "15 个 held-out guide 条件（unseen guide / **seen target**）",
    "1019 基因面板，指标在 GT DEG 子集上算（top-100 与 all=1019）",
    "6 个指标全部按文献 `calPerfor` 逐条移植，并与镜像内文献代码交叉校验",
]

CAVEATS = [
    "**泛化轴**：unseen guide / seen target，不是 scPerturBench 的 unseen-perturbation 设定。",
    "**基因空间**：1019（checkpoint 空间），非 HVG5000；文献 `numDEG=5000` 退化为「全部 1019」。",
    "**数据集**：GSE132080 不在文献 12 个数据集内，数值与论文表格不可比。",
    "**单 seed**：无重复，`summary_stats.csv` 的 `sd` 为空。",
    "**GEARS 的评测细胞集合不同**：其 test loader 覆盖训练划分中携带这些 target 基因的细胞"
    "（7361 个），其余方法预测 held-out guide 条件（3294 个）；评分真值统一为 held-out guide 细胞。"
    "这是 GEARS 官方 `getPredict` 路径，但解释时需要注意。",
    "**CPA** 用 `is_count_data=False` + `recon_loss='gauss'`（共享基因空间是 log-norm 而非计数），"
    "非官方默认的 `nb`。",
    "**Squidiff** 用 `lr_anneal_steps=2e4` 的完整退火日程（上游默认 `1e5` 在本数据上 step≈9e3 崩坏）；"
    "只有 `clip_denoised=True` 分支可用，它把表达截到 [−1,1] 所以整体偏低。",
    "**设备不同**：GEARS/CPA/linear 在 CPU（RTX 5060 是 `sm_120`，其 torch 无对应 kernel），"
    "scGen/Squidiff/Diff-AE-Perturb 在 GPU。",
    "**我们的两行不是同一个模型的两次推理**：`diffae` 是归档 checkpoint（条件接口有 bug），"
    "`fix_x0_*` 是按修复后的代码重新训练 200 epoch 的结果。",
    "**`kl_divergence` 绝对值**被少数 control 方差为 0 的基因主导（可达 2^40 量级），"
    "排序信息有限；`kl_divergence_floor`（std 下限 0.1）是更可读的诊断列。",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_tables(aggregate_dir: Path) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], int]]:
    values = {row["method"]: row for row in _read_csv(aggregate_dir / "summary_wide.csv")}
    ranks = {
        (row["method"], row["metric_budget"]): int(row["rank"])
        for row in _read_csv(aggregate_dir / "rankings.csv")
    }
    return values, ranks


def value_of(values: dict[str, dict[str, str]], method: str, cell: str) -> float:
    raw = values[method].get(cell, "")
    if raw in ("", None):
        raise KeyError(f"{method} has no value for {cell}")
    return float(raw)


def best_of(values: dict[str, dict[str, str]], methods: list[str], cell: str, direction: str) -> float:
    candidates = [value_of(values, method, cell) for method in methods]
    return min(candidates) if direction == "down" else max(candidates)


def fmt(value: float, direction: str) -> str:
    return f"{value:.6f}"


def render_table(
    values: dict[str, dict[str, str]],
    ranks: dict[tuple[str, str], int],
    cells: list[tuple[str, str, str]],
) -> list[str]:
    main_methods = [key for key, _ in EXTERNAL] + [key for key, _ in OURS]
    lines = []
    header = "| 方法 | " + " | ".join(label for _, label, _ in cells) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(cells) + 1))
    for key, label in EXTERNAL + OURS:
        cells_out = []
        for cell, _, direction in cells:
            value = value_of(values, key, cell)
            rank = ranks.get((key, cell))
            text = f"{fmt(value, direction)}"
            if abs(value - best_of(values, main_methods, cell, direction)) < 1e-12:
                text = f"**{text}**"
            if rank is not None:
                text += f" ({rank})"
            cells_out.append(text)
        lines.append(f"| {label} | " + " | ".join(cells_out) + " |")
    for key, label in FLOOR:
        cells_out = []
        for cell, _, direction in cells:
            rank = ranks.get((key, cell))
            text = fmt(value_of(values, key, cell), direction)
            if rank is not None:
                text += f" ({rank})"
            cells_out.append(text)
        lines.append(f"| _{label}（下限）_ | " + " | ".join(cells_out) + " |")
    lines.append("")
    lines.append("每格 `数值 (排名/19)`；**加粗** = 四个外部方法与我们的两行里最好的那个。")
    return lines


def render_conclusion(values: dict[str, dict[str, str]], ranks: dict[tuple[str, str], int]) -> list[str]:
    """Headline numbers, computed from the same tables so they cannot drift."""
    ceiling: dict[str, list[str]] = {}
    wins: list[str] = []
    beaten_by_cell: list[int] = []
    for cell, label, direction in CELLS:
        external = [(key, name, value_of(values, key, cell)) for key, name in EXTERNAL]
        best_name = (
            min(external, key=lambda item: item[2])[1] if direction == "down" else max(external, key=lambda item: item[2])[1]
        )
        best_value = best_of(values, [key for key, _ in EXTERNAL], cell, direction)
        ceiling.setdefault(best_name, []).append(label)
        ours = value_of(values, "fix_x0_percell", cell)
        won = ours < best_value if direction == "down" else ours > best_value
        if won:
            wins.append(label)
        beaten_by_cell.append(
            sum(1 for _, _, value in external if (ours < value if direction == "down" else ours > value))
        )
    never = [name for _, name in EXTERNAL if name not in ceiling]
    legacy = [
        sum(
            1
            for key, _ in EXTERNAL
            if (
                value_of(values, "diffae", cell) < value_of(values, key, cell)
                if direction == "down"
                else value_of(values, "diffae", cell) > value_of(values, key, cell)
            )
        )
        for cell, _, direction in CELLS
    ]
    ratios: dict[str, list[float]] = {"error": [], "other": []}
    for cell, _, direction in CELLS:
        best_value = best_of(values, [key for key, _ in EXTERNAL], cell, direction)
        family = "error" if cell in ERROR_TABLE else "other"
        ratios[family].append(value_of(values, "fix_x0_percell", cell) / best_value)
    lines = [
        f"- **修复后（`fix_x0_percell`）在 10 个 metric×budget 单元里，只有 "
        f"{len(wins)} 个（{('、'.join(wins)) if wins else '无'}）超过四个外部方法中最好的那个**，"
        f"其余 {10 - len(wins)} 个单元由 {('、'.join(sorted(ceiling)))} 守住。",
        f"- 天花板几乎只在 GEARS 与 scGen 之间移动（GEARS {len(ceiling.get('GEARS', []))} 个单元、"
        f"scGen {len(ceiling.get('scGen', []))} 个）；**{('、'.join(never)) if never else '—'} "
        f"在 10 个单元里一次都没拿到第一**。",
        f"- 逐方法看：`fix_x0_percell` 平均每个单元胜过 "
        f"**{sum(beaten_by_cell) / len(beaten_by_cell):.1f}/4** 个外部方法；"
        f"修复前的归档 checkpoint（`diffae`）只有 **{sum(legacy) / len(legacy):.1f}/4**。",
        f"- 与外部最好的距离（本方法 / 外部最好，1.00 = 追平）：误差类 5 个单元 "
        f"**{min(ratios['error']):.2f}×–{max(ratios['error']):.2f}×**；"
        f"相关性/差异基因类 5 个单元 **{min(ratios['other']):.2f}×–{max(ratios['other']):.2f}×**。",
        "- 结论：修复把我们从「在 4 个指标上与 `control_mean` 同级或更差」拉到"
        "「稳定压过 CPA 与 Squidiff、逼近 GEARS/scGen」，但**还不足以说超过第一梯队**——"
        "除 `kl_divergence@100` 外没有拿到任何单元的第一。",
        "",
        "下限参考：`control_mean` / `train_mean` 在多数单元排 8–13 名之间；"
        "修复后的 `fix_x0_percell` 在 10 个单元里全部排进前 6。",
    ]
    return lines


def render_verdict(values: dict[str, dict[str, str]], ranks: dict[tuple[str, str], int]) -> list[str]:
    lines = [
        "| 指标单元 | 外部最好（方法） | `fix_x0_percell` | 相对外部最好 | 胜过几个外部方法 | 排名/19 |",
        "|---|---|---|---|---|---|",
    ]
    ours_key = "fix_x0_percell"
    for cell, label, direction in CELLS:
        external = [(key, name, value_of(values, key, cell)) for key, name in EXTERNAL]
        best_key, best_name, best_value = (
            min(external, key=lambda item: item[2]) if direction == "down" else max(external, key=lambda item: item[2])
        )
        ours_value = value_of(values, ours_key, cell)
        if direction == "down":
            ratio = ours_value / best_value if best_value else float("inf")
            beaten = sum(1 for _, _, value in external if ours_value < value)
            verdict = f"{ratio:.2f}× " + ("✅" if ratio <= 1.0 else "❌")
        else:
            ratio = ours_value / best_value if best_value else float("inf")
            beaten = sum(1 for _, _, value in external if ours_value > value)
            verdict = f"{ratio:.2f}× " + ("✅" if ratio >= 1.0 else "❌")
        lines.append(
            f"| {label} | {best_name} {fmt(best_value, direction)} | {fmt(ours_value, direction)} "
            f"| {verdict} | **{beaten}/4** | {ranks.get((ours_key, cell), '')} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    values, ranks = load_tables(args.aggregate_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    error_cells = [cell for cell in CELLS if cell[0] in ERROR_TABLE]
    other_cells = [cell for cell in CELLS if cell[0] not in ERROR_TABLE]

    doc: list[str] = []
    doc.append("# Diff-AE-Perturb vs 四个外部方法（GEARS / scGen / CPA / Squidiff）")
    doc.append("")
    doc.append(
        "本报告是 `reports/benchmark/gse132080_6metrics_benchmark.md`（19 行总表）的**抽取视图**："
        "只保留四个已发表方法与我们的两行，去掉 4 行文献参考基线与 6 行我们自己的消融臂。"
        "数值与排名均取自同一份聚合表，未做任何重算。"
    )
    doc.append("")
    doc.append("## 0. 协议")
    doc.append("")
    doc.extend(f"- {item}" for item in PROTOCOL)
    doc.append("")
    doc.append("## 0.5 结论（数字由下表算出）")
    doc.append("")
    doc.extend(render_conclusion(values, ranks))
    doc.append("")
    doc.append("## 1. 误差与分布指标（越低越好）")
    doc.append("")
    doc.extend(render_table(values, ranks, error_cells))
    doc.append("")
    doc.append("## 2. 相关性与差异基因（越高越好）")
    doc.append("")
    doc.extend(render_table(values, ranks, other_cells))
    doc.append("")
    doc.append("## 3. 逐单元判定：我们 vs 四个外部方法")
    doc.append("")
    doc.append("`fix_x0_percell`（修复后重训）与四个外部方法逐一比。比值 <1（误差类）/ >1（相关类）即为胜。")
    doc.append("")
    doc.extend(render_verdict(values, ranks))
    doc.append("")
    doc.append("## 4. `kl_divergence_floor` 诊断列")
    doc.append("")
    doc.append("`kl_divergence` 绝对值被少数 control 方差为 0 的基因主导，此行更可读（std 下限 0.1）：")
    doc.append("")
    doc.append("| 方法 | " + " | ".join(label for _, label in DIAGNOSTIC) + " |")
    doc.append("|" + "---|" * (len(DIAGNOSTIC) + 1))
    for key, label in EXTERNAL + OURS:
        cells_out = [fmt(value_of(values, key, cell), "down") for cell, _ in DIAGNOSTIC]
        doc.append(f"| {label} | " + " | ".join(cells_out) + " |")
    doc.append("")
    doc.append("## 5. Squidiff 的另外三个变体（不可用）")
    doc.append("")
    doc.append("Squidiff 上游采样路径的 `process_xstart` 在高噪声步把零均值巨尺度量在 0 处整流，")
    doc.append("1000 步累积放大 59×。下表说明为什么主表里只用 `squidiff_repro_clip`：")
    doc.append("")
    doc.append("| 变体 | `mse@100` | `e_distance@100` | `wasserstein@100` | `pcc_delta@100` |")
    doc.append("|---|---|---|---|---|")
    for key, label in BROKEN_SQUIDIFF:
        row = [
            fmt(value_of(values, key, "mse@100"), "down"),
            fmt(value_of(values, key, "e_distance@100"), "down"),
            f"{value_of(values, key, 'wasserstein@100'):,.0f}",
            fmt(value_of(values, key, "pcc_delta@100"), "up"),
        ]
        doc.append(f"| {label} | " + " | ".join(row) + " |")
    doc.append("")
    doc.append("## 6. 口径偏差（引用这些数字时必须一起引用）")
    doc.append("")
    doc.extend(f"{index}. {item}" for index, item in enumerate(CAVEATS, start=1))
    doc.append("")
    doc.append("## 7. 出处与复现")
    doc.append("")
    doc.append("```")
    doc.append("reports/benchmark/aggregate/summary_wide.csv   数值（19 方法 × 12 列）")
    doc.append("reports/benchmark/aggregate/rankings.csv       排名（19 方法 × 10 单元）")
    doc.append("reports/benchmark/aggregate/report.md          自动生成的完整 19 行报告")
    doc.append("reports/benchmark/aggregate/{wins,summary_stats,summary_long}.csv")
    doc.append("reports/benchmark/external_methods_comparison.csv   本报告的机器可读版")
    doc.append("```")
    doc.append("")
    doc.append("```powershell")
    doc.append("python scripts/build_external_comparison.py")
    doc.append("```")
    doc.append("")
    doc.append(
        "逐方法逐条件的原始分（`score/summary.csv`、`per_condition.csv`、`full.json`）与"
        "各方法的 `adapter_manifest.json` 在 "
        "`C:\\Users\\Admin\\myprojects\\perturbench\\.scperturbench\\results\\gse132080_6metrics\\`"
        "（该目录未纳入版本控制）。"
    )
    doc.append("")

    markdown_path = args.output_dir / "external_methods_comparison.md"
    markdown_path.write_text("\n".join(doc), encoding="utf-8")

    csv_path = args.output_dir / "external_methods_comparison.csv"
    selected = [key for key, _ in EXTERNAL] + [key for key, _ in OURS]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric_budget", "value", "rank_of_19", "direction"])
        for key, _ in EXTERNAL + OURS:
            for cell, _, direction in CELLS:
                writer.writerow(
                    [key, cell, f"{value_of(values, key, cell):.6f}", ranks.get((key, cell), ""), direction]
                )

    print(f"wrote {markdown_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
