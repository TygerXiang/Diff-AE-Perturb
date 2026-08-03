from __future__ import annotations

import csv
import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


BENCH_FULL = REPO_ROOT / "runs/benchmarks/explicit_only_vs_no_activity_200ep_6metrics_guide_all_full.json"
EMB_ACTIVITY = REPO_ROOT / "runs/benchmarks/embedding_activity_correlation_200ep_no_diralign/summary.json"
SEQ_SUMMARY = REPO_ROOT / "runs/benchmarks/embedding_sequence_relationship_200ep_no_diralign/summary.json"
DIRECTION_SUMMARY = REPO_ROOT / "runs/benchmarks/embedding_direction_clustering_200ep_no_diralign/summary.json"
NO_ACTIVITY_TARGETS = REPO_ROOT / "runs/benchmarks/embedding_direction_clustering_200ep_no_diralign/no_activity_200ep/per_target_direction_summary.csv"
EXPLICIT_TARGETS = REPO_ROOT / "runs/benchmarks/embedding_direction_clustering_200ep_no_diralign/explicit_only_200ep/per_target_direction_summary.csv"
NO_ACTIVITY_METRICS = REPO_ROOT / "runs/benchmark_activity_branch_none_200ep/metrics.json"
EXPLICIT_METRICS = REPO_ROOT / "runs/benchmark_activity_branch_predicted_day10_explicit_only_200ep/metrics.json"
OUTPUT = REPO_ROOT / "reports/embedding_activity_visual_report.html"


METRICS = ["mse", "e_distance", "pcc_delta", "wasserstein", "kl_divergence", "common_degs"]
METRIC_LABELS = {
    "mse": "MSE",
    "e_distance": "E-distance",
    "pcc_delta": "PCC delta",
    "wasserstein": "Wasserstein",
    "kl_divergence": "KL divergence",
    "common_degs": "Common DEGs",
}
LOWER_BETTER = {"mse", "e_distance", "wasserstein", "kl_divergence"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    if number == 0:
        return "0"
    if abs(number) < 0.001:
        return f"{number:.1e}"
    return f"{number:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def svg_bar_chart(
    rows: list[dict[str, object]],
    *,
    label_key: str,
    value_keys: list[str],
    colors: list[str],
    width: int = 760,
    height: int = 300,
    value_format=fmt,
    title: str = "",
    higher_is_better: bool | None = None,
) -> str:
    margin = {"top": 26, "right": 24, "bottom": 58, "left": 78}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]
    labels = [str(row[label_key]) for row in rows]
    values = [float(row[key]) for row in rows for key in value_keys]
    lo = min(0.0, min(values))
    hi = max(values)
    if hi == lo:
        hi = lo + 1.0
    group_w = inner_w / max(len(rows), 1)
    bar_w = group_w / (len(value_keys) + 1.2)

    def x_pos(i: int, j: int) -> float:
        return margin["left"] + i * group_w + (j + 0.6) * bar_w

    def y_pos(v: float) -> float:
        return margin["top"] + (hi - v) / (hi - lo) * inner_h

    y0 = y_pos(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<line x1="{margin["left"]}" y1="{y0:.1f}" x2="{width - margin["right"]}" y2="{y0:.1f}" class="axis"/>',
    ]
    for tick in range(5):
        value = lo + (hi - lo) * tick / 4
        y = y_pos(value)
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt(value, 2)}</text>')
    for i, row in enumerate(rows):
        for j, key in enumerate(value_keys):
            value = float(row[key])
            x = x_pos(i, j)
            y = y_pos(value)
            h = abs(y0 - y)
            y_rect = min(y, y0)
            label = f'{row[label_key]} {key}: {value_format(value)}'
            parts.append(
                f'<rect x="{x:.1f}" y="{y_rect:.1f}" width="{bar_w * 0.86:.1f}" height="{max(h, 1):.1f}" '
                f'class="{colors[j]}" aria-label="{html.escape(label)}"><title>{html.escape(label)}</title></rect>'
            )
            parts.append(
                f'<text x="{x + bar_w * 0.43:.1f}" y="{max(y_rect - 5, 12):.1f}" text-anchor="middle" class="bar-label">{value_format(value)}</text>'
            )
        parts.append(
            f'<text x="{margin["left"] + i * group_w + group_w / 2:.1f}" y="{height - 28}" text-anchor="middle" class="x-label">{html.escape(labels[i])}</text>'
        )
    if higher_is_better is not None:
        marker = "higher is better" if higher_is_better else "lower is better"
        parts.append(f'<text x="{width - margin["right"]}" y="18" text-anchor="end" class="caption">{marker}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_horizontal_bars(
    rows: list[dict[str, object]],
    *,
    label_key: str,
    value_key: str,
    width: int = 760,
    height: int = 360,
    max_value: float | None = None,
    color_class: str = "fill-a",
    value_format=fmt,
    title: str = "",
) -> str:
    margin = {"top": 22, "right": 90, "bottom": 28, "left": 136}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]
    max_v = max_value if max_value is not None else max(float(row[value_key]) for row in rows)
    row_h = inner_h / max(len(rows), 1)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for i, row in enumerate(rows):
        y = margin["top"] + i * row_h + row_h * 0.18
        value = float(row[value_key])
        w = max(1.0, value / max_v * inner_w)
        parts.append(f'<text x="{margin["left"] - 10}" y="{y + row_h * 0.34:.1f}" text-anchor="end" class="x-label">{html.escape(str(row[label_key]))}</text>')
        parts.append(
            f'<rect x="{margin["left"]}" y="{y:.1f}" width="{w:.1f}" height="{row_h * 0.56:.1f}" class="{color_class}">'
            f'<title>{html.escape(str(row[label_key]))}: {value_format(value)}</title></rect>'
        )
        parts.append(f'<text x="{margin["left"] + w + 8:.1f}" y="{y + row_h * 0.36:.1f}" class="bar-label">{value_format(value)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def table(headers: list[str], rows: list[list[object]], *, class_name: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    return f'<div class="table-wrap"><table class="{class_name}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def metric_row_data(bench: dict) -> list[dict[str, object]]:
    avgs = bench["summary"]["method_averages"]
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "metric": METRIC_LABELS[metric],
                "no_activity": avgs["no_activity_200ep"][metric],
                "explicit": avgs["explicit_only_200ep"][metric],
                "higher_better": metric not in LOWER_BETTER,
            }
        )
    return rows


def compact_metric_rows(bench: dict) -> list[list[str]]:
    avgs = bench["summary"]["method_averages"]
    wins = bench["summary"]["method_metric_win_counts"]
    rows = []
    for metric in METRICS:
        better = "lower" if metric in LOWER_BETTER else "higher"
        rows.append(
            [
                METRIC_LABELS[metric],
                better,
                fmt(avgs["no_activity_200ep"][metric]),
                fmt(avgs["explicit_only_200ep"][metric]),
                f'{wins["no_activity_200ep"][metric]} / {wins["explicit_only_200ep"][metric]}',
            ]
        )
    return rows


def get_nested(obj: dict, keys: list[str]) -> object:
    current = obj
    for key in keys:
        current = current[key]
    return current


def method_card(title: str, value: str, note: str) -> str:
    return f"""
      <div class="stat">
        <div class="stat-title">{html.escape(title)}</div>
        <div class="stat-value">{html.escape(value)}</div>
        <div class="stat-note">{html.escape(note)}</div>
      </div>
    """


def main() -> None:
    bench = load_json(BENCH_FULL)
    emb = load_json(EMB_ACTIVITY)
    seq = load_json(SEQ_SUMMARY)
    direction = load_json(DIRECTION_SUMMARY)
    no_metrics = load_json(NO_ACTIVITY_METRICS)
    explicit_metrics = load_json(EXPLICIT_METRICS)
    no_targets = read_csv_dicts(NO_ACTIVITY_TARGETS)
    explicit_targets = read_csv_dicts(EXPLICIT_TARGETS)

    benchmark_rows = metric_row_data(bench)
    rank_rows = [
        {"model": "no_activity_200ep", "overall rank": bench["summary"]["method_overall_rank"]["no_activity_200ep"]},
        {"model": "explicit_only_200ep", "overall rank": bench["summary"]["method_overall_rank"]["explicit_only_200ep"]},
    ]

    emb_corr_rows = []
    for method in ["no_activity_200ep", "explicit_only_200ep"]:
        info = emb[method]
        emb_corr_rows.append(
            {
                "method": method.replace("_200ep", ""),
                "pearson": info["embedding_norm_vs_activity"]["pearson"]["stat"],
                "spearman": info["embedding_norm_vs_activity"]["spearman"]["stat"],
                "rank": info["within_target_rank_agreement"]["embedding_norm_vs_activity"]["agreement_fraction"],
            }
        )

    seq_rows = []
    first_g = seq["methods"]["no_activity_200ep"]["distal_first_g_match"]["embedding_s"]
    first_g_counts = f'{int(first_g["true_n"])} true / {int(first_g["false_n"])} false'
    for method in ["no_activity_200ep", "explicit_only_200ep"]:
        s = seq["methods"][method]
        seq_rows.append(
            [
                method,
                fmt(s["pam_distance_correlations"]["embedding_s"]["spearman"]["stat"]),
                fmt(s["pam_distance_family_centered_correlations"]["embedding_s"]["stat"]),
                f'{fmt(s["rg_dt_vs_other"]["embedding_s"]["true_mean"])} vs {fmt(s["rg_dt_vs_other"]["embedding_s"]["false_mean"])}',
                fmt(s["distal_gc_content_correlations"]["embedding_s"]["spearman"]["stat"]),
                f'{fmt(s["distal_first_g_match"]["embedding_s"]["true_mean"])} vs {fmt(s["distal_first_g_match"]["embedding_s"]["false_mean"])}',
                fmt(s["surrounding_g_count_correlations"]["embedding_s"]["all_mismatches"]["stat"]),
            ]
        )

    direction_rows = []
    for method in ["no_activity_200ep", "explicit_only_200ep"]:
        d = direction["methods"][method]
        direction_rows.append(
            {
                "method": method.replace("_200ep", ""),
                "same": d["cosine_same_vs_cross"]["same_target"]["mean"],
                "cross": d["cosine_same_vs_cross"]["cross_target"]["mean"],
                "top1": d["nearest_neighbor"]["top1_same_target_fraction"]["mean"],
            }
        )

    no_tight = sorted(no_targets, key=lambda r: float(r["within_cosine_mean"]), reverse=True)[:5]
    no_loose = sorted(no_targets, key=lambda r: float(r["within_cosine_mean"]))[:5]
    ex_tight = sorted(explicit_targets, key=lambda r: float(r["within_cosine_mean"]), reverse=True)[:5]
    ex_loose = sorted(explicit_targets, key=lambda r: float(r["within_cosine_mean"]))[:5]

    train_table = table(
        ["Parameter", "no_activity_200ep", "explicit_only_200ep"],
        [
            ["epochs", "200", "200"],
            ["best epoch", no_metrics["best_epoch"], explicit_metrics["best_epoch"]],
            ["best val loss", fmt(no_metrics["best_val_loss"]), fmt(explicit_metrics["best_val_loss"])],
            ["test loss", fmt(no_metrics["test_metrics"]["loss"]), fmt(explicit_metrics["test_metrics"]["loss"])],
            ["activity source", no_metrics["activity_source"], explicit_metrics["activity_source"]],
            ["feature dim", no_metrics["conditioning_summary"]["feature_dim"], explicit_metrics["conditioning_summary"]["feature_dim"]],
            ["activity supervision", no_metrics["conditioning_summary"].get("activity_supervision", False), explicit_metrics["conditioning_summary"].get("activity_supervision", False)],
            ["device", no_metrics["device"], explicit_metrics["device"]],
        ],
    )

    css = """
    <style>
      :root {
        --bg: #f7f7f3;
        --ink: #1e2621;
        --muted: #607066;
        --panel: #ffffff;
        --line: #d8ded6;
        --soft: #eef2ec;
        --a: #316b5b;
        --b: #9b5d36;
        --c: #4a6f9d;
        --good: #2f7d55;
        --bad: #a34b42;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font-family: "Segoe UI", Arial, sans-serif;
        line-height: 1.5;
      }
      main { max-width: 1180px; margin: 0 auto; padding: 42px 28px 64px; }
      header { margin-bottom: 28px; border-bottom: 1px solid var(--line); padding-bottom: 24px; }
      h1 { margin: 0 0 10px; font-size: 34px; letter-spacing: 0; }
      h2 { margin: 36px 0 14px; font-size: 24px; }
      h3 { margin: 22px 0 10px; font-size: 18px; }
      p { margin: 8px 0 12px; }
      code { background: var(--soft); padding: 1px 5px; border-radius: 4px; }
      .subtitle { color: var(--muted); max-width: 900px; }
      .grid { stroke: var(--line); stroke-width: 1; }
      .axis { stroke: var(--ink); stroke-width: 1.2; }
      .tick, .caption, .x-label, .bar-label { fill: var(--muted); font-size: 12px; }
      .bar-label { fill: var(--ink); }
      .fill-a { fill: var(--a); }
      .fill-b { fill: var(--b); }
      .fill-c { fill: var(--c); }
      .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; align-items: start; }
      .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px;
        margin: 16px 0;
      }
      .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 18px 0; }
      .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
      .stat-title { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
      .stat-value { font-size: 25px; font-weight: 600; margin-top: 4px; }
      .stat-note { color: var(--muted); font-size: 13px; margin-top: 2px; }
      .legend { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 13px; margin: 8px 0; }
      .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
      .swatch-a { background: var(--a); }
      .swatch-b { background: var(--b); }
      .swatch-c { background: var(--c); }
      .table-wrap { overflow-x: auto; margin: 12px 0 20px; }
      table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
      th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
      th { background: var(--soft); font-weight: 600; }
      tr:last-child td { border-bottom: none; }
      .note { color: var(--muted); font-size: 13px; }
      .callout { border-left: 4px solid var(--a); padding: 10px 14px; background: var(--panel); margin: 14px 0; }
      .two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
      svg { width: 100%; height: auto; display: block; }
      @media (max-width: 640px) {
        main { padding: 28px 16px 48px; }
        h1 { font-size: 28px; }
        .chart-grid { grid-template-columns: 1fr; }
      }
    </style>
    """

    metric_panels = []
    for metric in METRICS:
        rows = [
            {
                "model": "no_activity",
                "value": bench["summary"]["method_averages"]["no_activity_200ep"][metric],
            },
            {
                "model": "explicit_only",
                "value": bench["summary"]["method_averages"]["explicit_only_200ep"][metric],
            },
        ]
        metric_panels.append(
            f"""
            <div class="panel">
              <h3>{html.escape(METRIC_LABELS[metric])}</h3>
              <div class="legend"><span><span class="swatch swatch-a"></span>no_activity</span><span><span class="swatch swatch-b"></span>explicit_only</span></div>
              {svg_bar_chart(rows, label_key="model", value_keys=["value"], colors=["fill-a"], height=230, title=METRIC_LABELS[metric], higher_is_better=metric not in LOWER_BETTER)}
            </div>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Embedding, Activity, and Sequence Analysis Report</title>
  {css}
</head>
<body>
<main>
  <header>
    <h1>PerturbDiff sgRNA Activity 与 Embedding 分析报告</h1>
    <p class="subtitle">汇总 activity conditioning 模块、127-guide 六指标评估、Embedding/activity 相关性、sgRNA sequence 关系以及 embedding 方向聚集性。报告由本地结果文件自动生成。</p>
  </header>

  <section>
    <h2>1. 模型与 Activity 模块</h2>
    <div class="callout">
      <p><strong>PerturbDiff</strong> 使用表达向量编码器得到语义 latent <code>z_sem</code>，为每个 perturbation/sgRNA 学习 latent delta embedding，再用条件 DDIM denoiser 从控制细胞生成目标扰动表达。</p>
    </div>
    <div class="two-col">
      <div class="panel">
        <h3>基础条件路径</h3>
        <p><code>SemanticEncoder</code> 将 <code>x0</code> 编码为 <code>z_sem</code>；<code>PerturbationModule</code> 为每个 sgRNA 提供 <code>delta_base</code>。推理时从控制细胞 latent 出发，构造 <code>z_cf = z_sem - source_delta + target_delta</code>，再传入 DDIM denoiser。</p>
        <p>因此 perturbation embedding 本身既有长度（强度）也有方向（target identity / response direction）。</p>
      </div>
      <div class="panel">
        <h3>Activity conditioning 实现</h3>
        <p>activity 模块打开后，模型先把 sgRNA 特征转换为非负 <code>strength</code>。在 <code>predicted</code> 模式中，<code>GuideStrengthEncoder</code> 用 Table S8 的 <code>genome input</code> 与 <code>sgRNA input</code> 两通道 one-hot 序列，经 CNN 回归 strength；在 <code>direct</code> 模式中直接读取 activity 标量。</p>
        <p><code>GeneralizedSigmoid</code> 为每个 target 学习 beta/bias，将 strength 映射为 target-specific <code>activity_value</code>。本次 <code>explicit_only_200ep</code> 中，activity_value 被线性投影成 16 维 <code>activity_embedding</code> 作为 denoiser 额外输入，而没有启用 delta scaling。</p>
      </div>
    </div>
  </section>

  <section>
    <h2>2. 127 guides 六指标评估：explicit_only vs no_activity</h2>
    <div class="stats">
      {method_card("Conditions", str(bench["summary"]["n_conditions"]), "guide-level test conditions")}
      {method_card("Overall rank", f'no_activity {fmt(bench["summary"]["method_overall_rank"]["no_activity_200ep"])} / explicit {fmt(bench["summary"]["method_overall_rank"]["explicit_only_200ep"])}', "lower rank is better")}
      {method_card("MSE improvement", f'{(1 - bench["summary"]["method_averages"]["explicit_only_200ep"]["mse"] / bench["summary"]["method_averages"]["no_activity_200ep"]["mse"]) * 100:.1f}%', "explicit_only lower than no_activity")}
      {method_card("PCC delta gain", f'{bench["summary"]["method_averages"]["explicit_only_200ep"]["pcc_delta"] - bench["summary"]["method_averages"]["no_activity_200ep"]["pcc_delta"]:.3f}', "explicit_only minus no_activity")}
    </div>
    <div class="panel">
      <h3>Metric averages</h3>
      <div class="legend"><span><span class="swatch swatch-a"></span>no_activity_200ep</span><span><span class="swatch swatch-b"></span>explicit_only_200ep</span></div>
      {svg_bar_chart(benchmark_rows, label_key="metric", value_keys=["no_activity", "explicit"], colors=["fill-a", "fill-b"], height=330, title="Six benchmark metrics")}
    </div>
    {table(["Metric", "Better", "no_activity_200ep", "explicit_only_200ep", "win counts no/ex"], compact_metric_rows(bench))}
    <p class="note">主要结论：按平均值看，explicit_only 在 MSE、E-distance、PCC_delta 和 Common DEGs 上更好；no_activity 在 Wasserstein 与 KL divergence 上更好。按逐 guide win counts 看，E-distance/Wasserstein/KL 更偏向 no_activity，说明均值与逐条件稳定性存在差异。overall rank explicit_only 略优。</p>
  </section>

  <section>
    <h2>3. Embedding/activity 相关性</h2>
    <div class="panel">
      <h3>Embedding norm vs activity</h3>
      <div class="legend"><span><span class="swatch swatch-a"></span>Pearson</span><span><span class="swatch swatch-b"></span>Spearman</span><span><span class="swatch swatch-c"></span>within-target rank agreement</span></div>
      {svg_bar_chart(emb_corr_rows, label_key="method", value_keys=["pearson", "spearman", "rank"], colors=["fill-a", "fill-b", "fill-c"], height=300, title="Embedding/activity correlations")}
    </div>
    {table(["Method", "Pearson", "Spearman", "within-target rank agreement"], [[r["method"], fmt(r["pearson"]), fmt(r["spearman"]), pct(r["rank"])] for r in emb_corr_rows])}
    <p class="note">embedding norm 与实测 activity 呈中等正相关；activity-conditioned explicit_only 的相关性更高。</p>
  </section>

  <section>
    <h2>4. sgRNA sequence 与 embedding 的关系</h2>
    {table(
        ["Method", "PAM distance Spearman vs embedding_s", "family-centered r", "rG:dT embedding_s mean vs other", "distal GC rho", "first G match embedding_s true vs false", "neighbor G rho"],
        seq_rows
    )}
    <div class="callout">
      <p>强信号被 embedding 部分捕获：PAM distance 越远，activity 与 <code>embedding_s</code> 越高；rG:dT mismatch 的 <code>embedding_s</code> 也高于其它 mismatch。GC、邻位 G 等弱/context-dependent 特征未稳定复现。修正后的 first invariant G 判断来自 Table S8 中 <code>genome input</code> 与 <code>sgRNA input</code> 的同位点比较，不是直接比较整段输入首字符；当前 distal 子集为 {first_g_counts}，因此不是全 true，且方向与文献描述不一致。</p>
    </div>
  </section>

  <section>
    <h2>5. Embedding 方向聚集性</h2>
    <div class="stats">
      {method_card("no_activity same-gene cosine", fmt(direction["methods"]["no_activity_200ep"]["cosine_same_vs_cross"]["same_target"]["mean"]), f'cross-gene {fmt(direction["methods"]["no_activity_200ep"]["cosine_same_vs_cross"]["cross_target"]["mean"])}')}
      {method_card("explicit same-gene cosine", fmt(direction["methods"]["explicit_only_200ep"]["cosine_same_vs_cross"]["same_target"]["mean"]), f'cross-gene {fmt(direction["methods"]["explicit_only_200ep"]["cosine_same_vs_cross"]["cross_target"]["mean"])}')}
      {method_card("no_activity top1 same gene", pct(direction["methods"]["no_activity_200ep"]["nearest_neighbor"]["top1_same_target_fraction"]["mean"]), "random baseline 3.26%")}
      {method_card("explicit top1 same gene", pct(direction["methods"]["explicit_only_200ep"]["nearest_neighbor"]["top1_same_target_fraction"]["mean"]), "random baseline 3.26%")}
    </div>
    <div class="panel">
      <h3>Same-gene vs cross-gene cosine</h3>
      <div class="legend"><span><span class="swatch swatch-a"></span>same gene</span><span><span class="swatch swatch-b"></span>cross gene</span><span><span class="swatch swatch-c"></span>top1 same-gene fraction</span></div>
      {svg_bar_chart(direction_rows, label_key="method", value_keys=["same", "cross", "top1"], colors=["fill-a", "fill-b", "fill-c"], height=300, title="Direction clustering")}
    </div>
    <div class="two-col">
      <div>
        <h3>no_activity: tightest / loosest genes</h3>
        {table(["Tightest target", "n", "within cosine"], [[r["target"], r["n_guides"], fmt(r["within_cosine_mean"])] for r in no_tight])}
        {table(["Loosest target", "n", "within cosine"], [[r["target"], r["n_guides"], fmt(r["within_cosine_mean"])] for r in no_loose])}
      </div>
      <div>
        <h3>explicit_only: tightest / loosest genes</h3>
        {table(["Tightest target", "n", "within cosine"], [[r["target"], r["n_guides"], fmt(r["within_cosine_mean"])] for r in ex_tight])}
        {table(["Loosest target", "n", "within cosine"], [[r["target"], r["n_guides"], fmt(r["within_cosine_mean"])] for r in ex_loose])}
      </div>
    </div>
    <p class="note">置换检验 5000 次：cosine gap 与 top1 same-gene hit 均达到经验 p 值下限 0.0002，说明同 gene 聚集不是随机 group size 造成的。</p>
  </section>

  <section>
    <h2>6. 综合解读</h2>
    <div class="panel">
      <p>activity 模块更明显地改善了均值型/方向型表达预测指标：MSE 降低，PCC_delta 大幅提高；但分布距离指标并未同步改善，说明 explicit activity condition 可能使平均 perturbation direction 更准确，却未完全改善细胞分布层面的生成匹配。</p>
      <p>embedding 既编码 activity 强度，也编码 target identity。norm 与 activity 相关，而方向空间中同 gene sgRNA 明显聚集。sequence 层面，PAM distance 与 rG:dT 两个强 activity determinant 在 embedding_s 中能看到同向模式；较弱的 GC、neighbor G 和 first G 特征在当前训练子集中不稳定。</p>
    </div>
  </section>

  <section>
    <h2>附录：实验设计与参数</h2>
    <h3>模型训练</h3>
    {train_table}
    <h3>Benchmark 设置</h3>
    {table(["Parameter", "Value"], [[k, v] for k, v in bench["config"].items() if k not in {"checkpoints", "names"}])}
    <h3>输入结果文件</h3>
    {table(["Analysis", "File"], [
        ["6 metrics full", BENCH_FULL.relative_to(REPO_ROOT)],
        ["Embedding/activity correlation", EMB_ACTIVITY.relative_to(REPO_ROOT)],
        ["Sequence relationship", SEQ_SUMMARY.relative_to(REPO_ROOT)],
        ["Direction clustering", DIRECTION_SUMMARY.relative_to(REPO_ROOT)],
    ])}
  </section>
</main>
</body>
</html>
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html_text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
