"""Tables and the quality-vs-cost curve for probe-moe-adapter-placement.md.

Usage: moe_adapter_report.py <runs-dir> <out-dir> [--edge ARM ...]
Writes summary.json (moe_adapter_rule.summarise), tables.md and one SVG per model
plotting evaluation NLL and summed leg-2 suite score against trainable parameters
and adapter AdamW bytes. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import moe_adapter_rule as rule  # noqa: E402

ARM_ORDER = ("A", "AS", "AT", "AE")


def _fmt_bytes(value: float) -> str:
    return f"{value / 2**30:.2f} GiB"


def tables(summary: dict) -> str:
    lines = []
    for model, result in summary.items():
        lines.append(f"### {model}: {result['verdict']}")
        lines.append("")
        if "validity" in result:
            lines.append("Validity: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}"
                                                  for k, v in result["validity"].items()))
            lines.append(f"Base: eval NLL {result['base_nll']:.4f}, task accuracy "
                         f"{result['base_accuracy']:.3f}. Fullest arm: `{result['fullest']}`.")
        if result.get("not_applicable"):
            lines.append("Not applicable: " + ", ".join(f"`{a}`" for a in result["not_applicable"]))
        if result.get("void"):
            lines.append("Void runs: " + json.dumps(result["void"]))
        lines.append("")
        arms = result.get("arms", {})
        if not arms:
            continue
        lines.append("| Arm | Seeds | Eval NLL mean ± SD | Gap vs fullest | Material "
                     "| Leg 2 failed | Accuracy (per seed) | Pooled 95% CI | Trainable "
                     "| AdamW fp32 state | Peak CUDA | s/step | Verdict |")
        lines.append("|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---|")
        for arm in ARM_ORDER:
            if arm not in arms:
                continue
            row = arms[arm]
            gap = row["task_gap"]
            lines.append(
                f"| `{arm}` | {row['seeds']} | {row['nll_mean']:.4f} ± {row['nll_sd']:.4f} "
                f"| {gap['gap']:+.4f} | {'yes' if gap['material'] else 'no'} "
                f"| {', '.join(row['leg2_failed']) or '-'} "
                f"| {', '.join(f'{a:.3f}' for a in row['accuracy'])} "
                f"| [{row['accuracy_pooled_ci95'][0]:.3f}, {row['accuracy_pooled_ci95'][1]:.3f}] "
                f"| {row['trainable_parameters'] / 1e6:.1f}M "
                f"| {_fmt_bytes(row['adapter_bytes_adamw_fp32'])} "
                f"| {_fmt_bytes(row['peak_cuda_bytes'])} | {row['seconds_per_step']:.2f} "
                f"| **{row['verdict']}** |")
        lines.append("")
        suites = list(result["base_gate"])
        lines.append("| Suite | Base | " + " | ".join(f"`{a}`" for a in ARM_ORDER if a in arms)
                     + " |")
        lines.append("|---|---:|" + "---:|" * sum(a in arms for a in ARM_ORDER))
        for suite in suites:
            cells = [", ".join(f"{v:.3f}" for v in arms[a]["gate"][suite])
                     for a in ARM_ORDER if a in arms]
            lines.append(f"| {suite} | {result['base_gate'][suite]:.3f} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def svg(model: str, result: dict) -> str:
    """Two panels: eval NLL and mean leg-2 suite score against log10 trainable params."""
    arms = [(a, result["arms"][a]) for a in ARM_ORDER if a in result.get("arms", {})]
    width, height, pad = 720, 300, 60
    xs = [math.log10(row["trainable_parameters"]) for _a, row in arms]
    x_lo, x_hi = min(xs) - 0.2, max(xs) + 0.2
    parts = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              'font-family="sans-serif" font-size="11">'),
             (f'<text x="{width / 2}" y="16" text-anchor="middle" font-size="13">{model}: '
              'quality against trainable LoRA parameters (log scale)</text>')]
    panels = (("eval NLL (lower is better)", _nll_band, [result["base_nll"]]),
              ("mean leg-2 suite score", _gate_band,
               [sum(result["base_gate"].values()) / len(result["base_gate"])]))
    panel_w = (width - 3 * pad) / 2
    for index, (title, values, base) in enumerate(panels):
        left = pad + index * (panel_w + pad)
        top, bottom = 40, height - 40
        all_y = [v for _a, row in arms for v in values(row)] + base
        y_lo, y_hi = min(all_y), max(all_y)
        span = (y_hi - y_lo) or 1.0
        y_lo, y_hi = y_lo - 0.1 * span, y_hi + 0.1 * span

        def px(x, left=left):
            return left + (x - x_lo) / (x_hi - x_lo) * panel_w

        def py(y, y_lo=y_lo, y_hi=y_hi, top=top, bottom=bottom):
            return bottom - (y - y_lo) / (y_hi - y_lo) * (bottom - top)

        parts.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{bottom - top}" '
                     'fill="none" stroke="#999"/>')
        parts.append(f'<text x="{left + panel_w / 2}" y="{top - 6}" text-anchor="middle">'
                     f'{title}</text>')
        parts.append(f'<text x="{left}" y="{py(y_lo) + 14}">{y_lo:.3f}</text>')
        parts.append(f'<text x="{left}" y="{py(y_hi) - 4}">{y_hi:.3f}</text>')
        for b in base:
            parts.append(f'<line x1="{left}" x2="{left + panel_w}" y1="{py(b):.1f}" '
                         f'y2="{py(b):.1f}" stroke="#c33" stroke-dasharray="4 3"/>')
            parts.append(f'<text x="{left + panel_w - 4}" y="{py(b) - 3:.1f}" '
                         'text-anchor="end" fill="#c33">base</text>')
        prev = None
        for (arm, row), x in zip(arms, xs):
            low, mid, high = values(row)
            cx = px(x)
            parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{py(low):.1f}" '
                         f'y2="{py(high):.1f}" stroke="#36c"/>')
            parts.append(f'<circle cx="{cx:.1f}" cy="{py(mid):.1f}" r="4" fill="#36c"/>')
            parts.append(f'<text x="{cx + 6:.1f}" y="{py(mid) - 6:.1f}">{arm} '
                         f'({row["trainable_parameters"] / 1e6:.1f}M)</text>')
            if prev:
                parts.append(f'<line x1="{prev[0]:.1f}" x2="{cx:.1f}" y1="{prev[1]:.1f}" '
                             f'y2="{py(mid):.1f}" stroke="#36c" stroke-opacity="0.4"/>')
            prev = (cx, py(mid))
    parts.append("</svg>")
    return "\n".join(parts)


def _nll_band(row: dict) -> list:
    return [row["nll_mean"] - row["nll_sd"], row["nll_mean"], row["nll_mean"] + row["nll_sd"]]


def _gate_band(row: dict) -> list:
    per_seed = [sum(v[i] for v in row["gate"].values()) / len(row["gate"])
                for i in range(len(row["seeds"]))]
    return [min(per_seed), sum(per_seed) / len(per_seed), max(per_seed)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs")
    parser.add_argument("out")
    parser.add_argument("--edge", action="append", default=[],
                        help="model:arm whose LR winner sat at the edge of the grid (V3)")
    args = parser.parse_args(argv)
    records = rule.load_runs(args.runs)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for model in sorted({r["model"] for r in records}):
        edges = {e.split(":", 1)[1]: True for e in args.edge if e.split(":", 1)[0] == model}
        summary.update(rule.summarise([r for r in records if r["model"] == model],
                                      edge_lrs=edges))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "tables.md").write_text(tables(summary), encoding="utf-8")
    for model, result in summary.items():
        if result.get("arms"):
            (out / f"curve-{model}.svg").write_text(svg(model, result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
