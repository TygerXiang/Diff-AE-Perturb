"""Build a Chinese Markdown report of the preliminary diffusion-model results.

Reads the CSV outputs of ``scripts/export_diffusion_prelim.py`` and writes
``reports/diffusion_preliminary_results.md`` with:

* run / model / data provenance,
* the sgRNA sequence alignment against the genomic target window, including the
  mismatch distance to the PAM,
* raw learned sgRNA perturbation vectors ``delta_g``,
* raw ``z_sem`` latents (control pool, control mean expression, per guide),
* raw gene expression values (control mean / observed sgRNA mean / model
  prediction).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        genes = []
        values = []
        for row in reader:
            genes.append(row[0])
            values.append([float(v) for v in row[1:]])
    return header[1:], genes, np.asarray(values, dtype=np.float64)


def fmt(value: str | float, digits: int = 4) -> str:
    if value in ("", None):
        return "-"
    return f"{float(value):.{digits}f}"


def raw(values: np.ndarray, limit: int = 8, digits: int = 4) -> str:
    head = ", ".join(fmt(v, digits) for v in values[:limit])
    return f"{head}, …" if values.size > limit else head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preliminary-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="扩散模型初步结果")
    args = parser.parse_args()

    base = args.preliminary_dir
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    guides = read_rows(base / "sgrna_summary.csv")
    control_rows = read_rows(base / "expression_control_mean.csv")
    control_mean = {row["gene"]: float(row["control"]) for row in control_rows}
    control_columns, genes, observed_matrix = read_matrix(base / "sgrna_expression_matrix.csv")
    _, _, predicted_matrix = read_matrix(base / "sgrna_predicted_expression_matrix.csv")
    observed = observed_matrix
    predicted = predicted_matrix
    column_index = {name: index for index, name in enumerate(control_columns)}
    conditions = read_rows(base / "z_sem_conditions.csv")
    control_zsem = read_rows(base / "z_sem_control.csv")
    delta_rows = read_rows(base / "sgrna_delta_vectors.csv")
    delta_by_guide = {row["guide_identity"]: row for row in delta_rows}

    zsem_keys = [key for key in control_zsem[0] if key.startswith("zsem_")]
    pool = {key: float(control_zsem[0][key]) for key in zsem_keys}
    mean_expr = {key: float(control_zsem[1][key]) for key in zsem_keys}
    delta_keys = [key for key in delta_rows[0] if key.startswith("delta_") and key[6:].isdigit()]

    lines: list[str] = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append("本报告由 `scripts/export_diffusion_prelim.py` 的导出结果自动生成，")
    lines.append("全部数值都是原始值（表达值、Z-sem latent、delta_g latent），不含 L2 等汇总指标。")
    lines.append("")
    lines.append("## 1. 结果来源")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| checkpoint | `{manifest['checkpoint']}` |")
    lines.append(f"| 数据 | `{manifest['data_path']}` |")
    lines.append(f"| 扰动标签列 | `{manifest['guide_col']}` |")
    lines.append(f"| sgRNA 身份列 | `{manifest['guide_identity_col']}` |")
    lines.append(f"| 导出粒度 | `{manifest['granularity']}` |")
    lines.append(f"| 对照细胞数 | {manifest['n_control_cells']} |")
    lines.append(f"| 基因数 | {manifest['n_genes']} |")
    lines.append(f"| 行数 / 条件数 | {manifest['n_rows']} / {manifest['n_conditions']} |")
    lines.append(f"| latent 维度 | {manifest['latent_dim']} |")
    lines.append(f"| DDIM 步数 | {manifest['ddim_steps']} |")
    lines.append(f"| 带 target 序列的行数 | {manifest['n_rows_with_target_segment']} / {manifest['n_rows']} |")
    lines.append(f"| 含错配的行数 | {manifest['n_rows_with_mismatch']} / {manifest['n_rows']} |")
    lines.append(f"| 单错配且与文库 PAM 距离一致的行数 | {manifest['n_single_mismatch_rows_matching_library_distance']} / {manifest['n_rows_with_library_mismatch_label']} |")
    lines.append("")
    lines.append("模型：`z_condition = E(x_control) + delta_g`，`x_pred = DDIM(z_condition)`。")
    lines.append("")

    lines.append("## 2. sgRNA 序列、target 序列与错配位置")
    lines.append("")
    lines.append("`target_segment_20nt` 是该 sgRNA 设计要识别的 20 nt 目标序列，取自文库中同一 family 的 fully matched guide")
    lines.append("（即 Table S10/S2 的 `_00` 行），因此每个 target 的多条变体共享同一段 target，可直接横向比较错配位置。")
    lines.append("`alignment_target_vs_sgrna` 用 `|` 表示匹配、`X` 表示错配。")
    lines.append("`mismatch_distance_to_pam` 从 PAM 端计数：紧邻 PAM 的碱基为 0，20 nt 中最远端为 19；")
    lines.append("`mismatch_label_from_library` 是论文文库原始编号（-1 ~ -19），两者关系为 `距离 = |编号| - 1`。")
    lines.append("")
    lines.append(f"校验：{manifest['n_single_mismatch_rows_matching_library_distance']} 条单错配 guide 的计算距离与文库编号完全一致，")
    lines.append(f"{manifest['n_rows_with_mismatch']} / {manifest['n_rows']} 条带错配，其余为 fully matched guide。")
    lines.append("")
    lines.append("`target_window_26nt` 与 `target_window_target_row` / `target_window_sgrna_row` 是论文 Table S8 的基因组窗口及对齐视图；")
    lines.append("该窗口的 protospacer 相对 sgRNA 有一条固定的 2 nt 5' 偏移，只作为 PAM 侧上下文参考，不参与上面的错配判定。")
    lines.append("")
    lines.append("| id | target | sgRNA identity | target 20 nt (5'->3') | sgRNA (5'->3') | alignment | 错配数 | 错配位置(1-based) | dist to PAM | 文库编号 | cells |")
    lines.append("| ---: | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |")
    for row in guides:
        lines.append(
            "| {pid} | {target} | {identity} | `{seg}` | `{seq}` | `{aln}` | {n} | {pos} | {dist} | {lib} | {cells} |".format(
                pid=row["perturbation_id"],
                target=row["target_gene"],
                identity=row["guide_identity"],
                seg=row["target_segment_20nt"] or "-",
                seq=row["sgrna_sequence"],
                aln=row["alignment_target_vs_sgrna"] or "-",
                n=row["n_mismatches_vs_target"] if row["n_mismatches_vs_target"] != "" else "-",
                pos=row["mismatch_positions_1based"] or "-",
                dist=row["mismatch_distance_to_pam"] or "-",
                lib=row["mismatch_label_from_library"] or "-",
                cells=row["n_cells"],
            )
        )
    lines.append("")

    lines.append("### 按 target gene 分组对比")
    lines.append("")
    lines.append("同一 target gene 下的多条 sgRNA 共享同一段 target 20 nt，下表把每条 sgRNA 与 target 的差异并排显示。")
    lines.append("")
    per_target: dict[str, list[dict[str, str]]] = {}
    for row in guides:
        per_target.setdefault(row["target_gene"], []).append(row)
    for target in sorted(per_target):
        lines.append(f"**{target}**")
        lines.append("")
        lines.append("| sgRNA identity | target 20 nt | sgRNA | alignment | dist to PAM | 文库编号 | paper relative gamma |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: |")
        for row in per_target[target]:
            lines.append(
                "| {identity} | `{seg}` | `{seq}` | `{aln}` | {dist} | {lib} | {gamma} |".format(
                    identity=row["guide_identity"],
                    seg=row["target_segment_20nt"] or "-",
                    seq=row["sgrna_sequence"],
                    aln=row["alignment_target_vs_sgrna"] or "-",
                    dist=row["mismatch_distance_to_pam"] or "-",
                    lib=row["mismatch_label_from_library"] or "-",
                    gamma=fmt(row["paper_mean_relative_gamma"]) if row["paper_mean_relative_gamma"] else "-",
                )
            )
        lines.append("")
        sample_window = next((row for row in per_target[target] if row["target_window_26nt"]), None)
        if sample_window is not None:
            lines.append("基因组窗口视图（示例）：")
            lines.append("")
            lines.append("```text")
            lines.append(f"window      {sample_window['target_window_26nt']}")
            lines.append(f"target      {sample_window['target_window_target_row']}")
            lines.append(f"sgRNA       {sample_window['target_window_sgrna_row']}")
            lines.append("```")
            lines.append("")

    lines.append("## 3. 模型 latent 原始值")
    lines.append("")
    lines.append("### 3.1 delta_g（模型为每个扰动学到的 sgRNA 向量，128 维原始值）")
    lines.append("")
    lines.append("完整 128 维见 `" + str(base / "sgrna_delta_vectors.csv") + "`；下方按条件列出前 16 维。")
    lines.append("")
    lines.append("| perturbation_id | target gene | delta_g[0:16] |")
    lines.append("| ---: | --- | --- |")
    seen: set[int] = set()
    for target in sorted(per_target):
        row = delta_by_guide[per_target[target][0]["guide_identity"]]
        perturb_id = int(row["perturbation_id"])
        if perturb_id in seen:
            continue
        seen.add(perturb_id)
        vector = np.asarray([float(row[key]) for key in delta_keys])
        lines.append(f"| {perturb_id} | {target} | {raw(vector, 16)} |")
    lines.append("")

    lines.append("### 3.2 Z-sem（SemanticEncoder 输出的 128 维 latent 原始值）")
    lines.append("")
    lines.append("| 条件 | z_sem[0:16] |")
    lines.append("| --- | --- |")
    lines.append("| control 池均值 | " + raw(np.asarray([pool[key] for key in zsem_keys]), 16) + " |")
    lines.append("| control 平均表达编码 | " + raw(np.asarray([mean_expr[key] for key in zsem_keys]), 16) + " |")
    for row in guides[:12]:
        entry = next(
            item
            for item in conditions
            if item["guide_identity"] == row["guide_identity"] and item["z_sem_source"] == "observed_mean_expression"
        )
        vector = np.asarray([float(entry[key]) for key in zsem_keys])
        lines.append(f"| {row['guide_identity']} 观测值 | {raw(vector, 16)} |")
    lines.append("")
    lines.append("全部 127 条 sgRNA 的观测 Z-sem 与反事实 `z_cf = z_sem(control) + delta_g` 见 "
                 f"`{base / 'z_sem_conditions.csv'}`，对照细胞逐个 latent 见 `{base / 'z_sem_control_cells.csv'}`。")
    lines.append("")

    lines.append("## 4. 原始表达值")
    lines.append("")
    lines.append("| 文件 | 内容 |")
    lines.append("| --- | --- |")
    lines.append(f"| `{base / 'expression_control_mean.csv'}` | 211 个 control 细胞的平均表达（1019 基因） |")
    lines.append(f"| `{base / 'sgrna_expression_matrix.csv'}` | 每条 sgRNA 的观测平均表达（1019 × 127） |")
    lines.append(f"| `{base / 'sgrna_predicted_expression_matrix.csv'}` | DDIM 生成表达（1019 × 127） |")
    lines.append(f"| `{base / 'sgrna_top_gene_expression_long.csv'}` | 每条 sgRNA 的 top 差异基因长表 |")
    lines.append("")
    lines.append("下方按 target gene 列出代表 sgRNA 的 top 6 差异基因的原始表达值。")
    lines.append("")
    for target in sorted(per_target):
        rows = per_target[target]
        lines.append(f"**{target}**")
        lines.append("")
        for row in rows:
            identity = row["guide_identity"]
            index = column_index[identity]
            observed_delta = observed[:, index] - np.asarray([control_mean[gene] for gene in genes])
            order = np.argsort(-np.abs(observed_delta))[:6]
            lines.append(f"`{identity}`（sgRNA `{row['sgrna_sequence']}`，cells={row['n_cells']}）")
            lines.append("")
            lines.append("| gene | control mean | observed mean | predicted mean |")
            lines.append("| --- | ---: | ---: | ---: |")
            for gene_index in order:
                gene = genes[gene_index]
                lines.append(
                    f"| {gene} | {fmt(control_mean[gene])} | {fmt(observed[gene_index, index])} | "
                    f"{fmt(predicted[gene_index, index])} |"
                )
            lines.append("")

    lines.append("## 5. 产出文件")
    lines.append("")
    for name in manifest["outputs"]:
        lines.append(f"- `{base / name}`")
    lines.append("")
    if args.condition_dir is not None and args.condition_dir.exists():
        lines.append("按“条件级”粒度导出的对照版本：")
        lines.append("")
        for name in sorted(p.name for p in args.condition_dir.iterdir() if p.is_file()):
            lines.append(f"- `{args.condition_dir / name}`")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
