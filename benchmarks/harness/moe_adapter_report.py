"""Tables and the quality-vs-cost curve for probe-moe-adapter-placement.md.

Usage: moe_adapter_report.py <runs-dir> <out-dir> --harness-sha256 H --split-sha256 S
       --hot MODEL=SHA256 [--hot ...]
Reads every run-*.json (counted candidates) and check-*.json (cross-box base checks)
under <runs-dir>, laid out as <box>/<results-dir>/<name>.json. Writes summary.json
(moe_adapter_rule.analyse plus per-item leg-2 flips), tables.md and one SVG per
model plotting evaluation NLL and mean leg-2 suite score against trainable
parameters. The flips re-score the recorded gate outputs with Soup's own scorer
from this checkout's src/, which must be at the records' scorer revision.
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
REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def box_of(path: str) -> str:
    return path.split("/", 1)[0]


def _scorer():
    if str(REPO_SRC) not in sys.path:
        sys.path.insert(0, str(REPO_SRC))
    from soup_cli.eval.gate_suites import (
        BUNDLED_SCORER_REVISION,
        DEFAULT_GENERAL_SUITE,
        score_bundled_suite,
    )

    return BUNDLED_SCORER_REVISION, DEFAULT_GENERAL_SUITE, score_bundled_suite


def flips(base: dict, other: dict) -> dict:
    """Per suite ``[gained, lost]``: items whose pass flips from ``base`` to ``other``.

    Every bundled suite scores each item from that item's output alone, so putting
    one of ``other``'s outputs among ``base``'s moves the suite score by 0 or
    exactly 1/n. Both records must re-score to their recorded values, and base +
    (gained - lost) / n must equal ``other``'s score, or this raises.
    """
    revision, suites, score = _scorer()
    for record in (base, other):
        if record["gate"]["scorer_revision"] != revision:
            raise ValueError(f"{record.get('_path')}: scorer revision "
                             f"{record['gate']['scorer_revision']}, this checkout {revision}")
    base_out, out = base["gate"]["outputs"], other["gate"]["outputs"]
    result = {}
    for suite in suites:
        prompts: list = []
        score(suite, lambda prompt: prompts.append(prompt) or "")
        base_score = score(suite, base_out.__getitem__)
        recorded = other["gate"]["scores"][suite]
        for record, value in ((base, base_score), (other, score(suite, out.__getitem__))):
            if abs(value - record["gate"]["scores"][suite]) > 1e-12:
                raise ValueError(f"{record.get('_path')}: {suite} re-scores to {value}, "
                                 f"recorded {record['gate']['scores'][suite]}")
        gained = lost = 0
        for prompt in prompts:
            if out[prompt] == base_out[prompt]:
                continue
            swapped = score(suite, lambda q, p=prompt: out[q] if q == p else base_out[q])
            step = round((swapped - base_score) * len(prompts))
            if abs(step) > 1:
                raise ValueError(f"{suite}: one output moved the score by {step} items")
            gained += step == 1
            lost += step == -1
        if abs(base_score + (gained - lost) / len(prompts) - recorded) > 1e-9:
            raise ValueError(f"{other.get('_path')}: {suite} items do not add up")
        result[suite] = [gained, lost]
    return result


def leg2_detail(analysis: dict, records: list, checks: list) -> dict:
    """Per model: item flips of every counted run and every base check vs the base."""
    by_path = {r["_path"]: r for r in [*records, *checks]}
    detail = {}
    for model, result in analysis["models"].items():
        if "base_source" not in result:
            continue
        base = by_path[result["base_source"]]
        entry: dict = {"base_box": box_of(base["_path"]), "arms": {}, "checks": {}}
        for arm, row in result.get("arms", {}).items():
            entry["arms"][arm] = [flips(base, by_path[path]) for path in row["sources"]]
        for check in checks:
            if check["model"] != model:
                continue
            box = box_of(check["_path"])
            if box in entry["checks"]:
                raise ValueError(f"two base checks for {model} on {box}")
            entry["checks"][box] = {
                "source": check["_path"], "flips": flips(base, check),
                "diff": {s: check["gate"]["scores"][s] - v
                         for s, v in base["gate"]["scores"].items()},
                "nll_diff": check["eval"]["nll"] - base["eval"]["nll"],
                "accuracy_diff": check["task"]["accuracy"] - base["task"]["accuracy"],
            }
        detail[model] = entry
    return detail


def _fmt_bytes(value: float) -> str:
    return f"{value / 2**30:.2f} GiB"


def sweep_table(sweep: dict) -> list:
    rates = sorted({float(lr) for row in sweep.values() for lr in row["nll"]})
    lines = [f"### Learning-rate sweep ({rule.SWEEP_MODEL}, seed 0, evaluation NLL)", "",
             "| Arm | " + " | ".join(f"{lr:g}" for lr in rates) + " | Winner | Status |",
             "|---|" + "---:|" * len(rates) + "---:|---|"]
    for arm in ARM_ORDER:
        if arm not in sweep:
            continue
        row = sweep[arm]
        cells = [f"{row['nll'][f'{lr:g}']:.4f}" if f"{lr:g}" in row["nll"] else "-"
                 for lr in rates]
        winner = "-" if row["winner"] is None else f"{row['winner']:g}"
        lines.append(f"| `{arm}` | " + " | ".join(cells) + f" | {winner} | {row['status']} |")
    return lines + [""]


def _mark(source: str, suite: str, entry: dict) -> str:
    """``†``: made on a box whose base check differs on ``suite``; ``‡``: box has no check."""
    box = box_of(source)
    if box == entry["base_box"]:
        return ""
    if box not in entry["checks"]:
        return "‡"
    return "†" if entry["checks"][box]["diff"][suite] != 0 else ""


def tables(analysis: dict, detail: dict) -> str:
    lines = sweep_table(analysis["sweep"])
    for model, result in analysis["models"].items():
        lines.append(f"### {model}: {result['verdict']}")
        lines.append("")
        if result.get("reason"):
            lines.append(f"Reason: {result['reason']}.")
        if result.get("unsettled_rates"):
            lines.append("Rate not settled for: "
                         + ", ".join(f"`{a}`" for a in result["unsettled_rates"]))
        if "validity" in result:
            lines.append("Validity: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}"
                                                  for k, v in result["validity"].items()))
            lines.append(f"Base: eval NLL {result['base_nll']:.4f}, task accuracy "
                         f"{result['base_accuracy']:.3f} (`{result['base_source']}`). "
                         f"Fullest arm: `{result['fullest']}`.")
        if result.get("not_applicable"):
            lines.append("Not applicable: " + ", ".join(f"`{a}`" for a in result["not_applicable"]))
        if result.get("void"):
            lines.append("Void runs: " + json.dumps(result["void"]))
        lines.append("")
        entry = detail.get(model, {"checks": {}})
        if entry["checks"]:
            lines.append("Cross-box base checks against the counted base (never in a verdict):")
            lines.append("")
            lines.append("| Box | Eval NLL diff | Accuracy diff | Suites that differ |")
            lines.append("|---|---:|---:|---|")
            for box, check in sorted(entry["checks"].items()):
                differ = [f"{s} {d:+.3f} (+{check['flips'][s][0]}/-{check['flips'][s][1]})"
                          for s, d in check["diff"].items() if d != 0]
                lines.append(f"| `{box}` | {check['nll_diff']:+.6f} "
                             f"| {check['accuracy_diff']:+.3f} | {'; '.join(differ) or 'none'} |")
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
        lines.append("Leg-2 suite score per seed, with items gained/lost against the base:")
        lines.append("")
        shown = [a for a in ARM_ORDER if a in arms]
        lines.append("| Suite | Base | " + " | ".join(f"`{a}`" for a in shown) + " |")
        lines.append("|---|---:|" + "---:|" * len(shown))
        for suite in result["base_gate"]:
            cells = []
            for arm in shown:
                row = arms[arm]
                cells.append(", ".join(
                    f"{score:.3f} +{moved[suite][0]}/-{moved[suite][1]}"
                    f"{_mark(source, suite, entry)}"
                    for score, moved, source in zip(row["gate"][suite], entry["arms"][arm],
                                                    row["sources"])))
            lines.append(f"| {suite} | {result['base_gate'][suite]:.3f} | "
                         + " | ".join(cells) + " |")
        lines.append("")
        lines.append("† made on a box whose base check scores that suite differently from "
                     "the counted base; ‡ made on a box with no base check.")
        lines.append("")
    for title, key, fmt in (
            ("Not counted (not at the arm's winning rate)", "not_counted",
             lambda r: f"`{r['model']}` `{r['arm']}` lr {r['lr']:g} seed {r['seed']}: "
                       f"`{r['source']}`"),
            ("Rejected (not the bundled harness, split or hot set)", "rejected",
             lambda r: f"`{r['source']}`: {r['problem']}")):
        if analysis.get(key):
            lines += [f"### {title}", ""] + [f"- {fmt(r)}" for r in analysis[key]] + [""]
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
             'quality against trainable LoRA parameters (log scale; label: AdamW fp32 '
             'state)</text>')]
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
                         f'({row["trainable_parameters"] / 1e6:.1f}M, '
                         f'{_fmt_bytes(row["adapter_bytes_adamw_fp32"])})</text>')
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
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--hot", action="append", required=True, metavar="MODEL=SHA256",
                        help="SHA-256 of the hot-set file every AT run of MODEL must record")
    args = parser.parse_args(argv)
    expect = {"harness_sha256": args.harness_sha256, "split_sha256": args.split_sha256,
              "hot_sha256": dict(item.split("=", 1) for item in args.hot)}
    records = rule.load_runs(args.runs)
    analysis = rule.analyse(records, expect=expect)
    checks = []
    for check in rule.load_runs(args.runs, "check-*.json"):
        problem = rule.provenance_problem(check, **expect)
        if problem is None:
            checks.append(check)
        else:
            analysis["rejected"].append({"source": check["_path"], "problem": problem})
    detail = leg2_detail(analysis, records, checks)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"analysis": analysis, "leg2": detail},
                                                 indent=1), encoding="utf-8")
    (out / "tables.md").write_text(tables(analysis, detail), encoding="utf-8")
    for model, result in analysis["models"].items():
        if result.get("arms"):
            (out / f"curve-{model}.svg").write_text(svg(model, result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
