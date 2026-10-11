"""Tables and the quality-against-cost curves for probe-moe-adapter-placement.md.

Usage: moe_adapter_report.py <runs-dir> <out-dir> --harness-sha256 H --split-sha256 S
       --hot MODEL=SHA256 [--hot ...]
Reads every run-*.json (counted candidates) and check-*.json (cross-box base checks)
under <runs-dir>, laid out as <box>/<results-dir>/<name>.json. Writes, with LF line
ends on every OS, summary.json (moe_adapter_rule.analyse plus per-item leg-2 flips),
tables.md (the gate's status and scope at the top, each model's rate scope with its
verdict) and one SVG per model: 2 x 2 panels, the NLL gap to the fullest arm (seed-
matched mean +- 1 SD) and task accuracy (mean over seeds, 95% CI with questions
resampled) against trainable parameters (log scale) and peak CUDA memory. The flips
re-score the recorded gate outputs with Soup's own scorer from this checkout's src/,
which must be at the records' scorer revision.
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


THINK_CLOSE = "</think>"


def think_closure(base: dict, other: dict) -> dict:
    """Gate answers that never close ``<think>`` (the 256-token budget ended inside the
    reasoning), and the base->other item flips split by whether the base answer closed it.

    Reporting only, for models whose default chat template opens a think block.
    """
    _revision, suites, score = _scorer()
    base_out, out = base["gate"]["outputs"], other["gate"]["outputs"]
    split = {"gained_base_unclosed": 0, "gained_base_closed": 0,
             "lost_base_unclosed": 0, "lost_base_closed": 0}
    for suite in suites:
        prompts: list = []
        score(suite, lambda prompt: prompts.append(prompt) or "")
        base_score = score(suite, base_out.__getitem__)
        for prompt in prompts:
            if out[prompt] == base_out[prompt]:
                continue
            swapped = score(suite, lambda q, p=prompt: out[q] if q == p else base_out[q])
            step = round((swapped - base_score) * len(prompts))
            if step:
                kind = "gained" if step > 0 else "lost"
                closed = "closed" if THINK_CLOSE in base_out[prompt] else "unclosed"
                split[f"{kind}_base_{closed}"] += 1
    return {"base_unclosed": sum(THINK_CLOSE not in o for o in base_out.values()),
            "unclosed": sum(THINK_CLOSE not in o for o in out.values()),
            "answers": len(out), **split}


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
        if any(THINK_CLOSE in o for o in base["gate"]["outputs"].values()):
            entry["think"] = {arm: [think_closure(base, by_path[path]) for path in row["sources"]]
                              for arm, row in result.get("arms", {}).items()}
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
    gate = analysis.get("gate")
    lines = [] if gate is None else [
        "**Quality gate.** "
        + ", ".join(f"{name} {value:g}" for name, value in gate["thresholds"].items())
        + f". Status: {gate['status']}. Scope: {gate['scope']}.", ""]
    lines += sweep_table(analysis["sweep"])
    for model, result in analysis["models"].items():
        lines.append(f"### {model}: {result['verdict']}")
        lines.append("")
        if result.get("scope"):
            lines.append(f"Scope: {result['scope']}.")
        if result.get("reason"):
            lines.append(f"Reason: {result['reason']}.")
        if result.get("unsettled_rates"):
            lines.append("Rate not settled for: "
                         + ", ".join(f"`{a}`" for a in result["unsettled_rates"]))
        if result.get("missing"):
            lines.append("Planned runs without an ok record: "
                         + ", ".join(f"`{m}`" for m in result["missing"]))
        if result.get("ae_did_not_fit"):
            lines.append(f"`AE` did not fit; evidence: `{result['ae_did_not_fit']}`")
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
        if result.get("not_counted"):
            lines.append("Seeds outside the plan, not counted: " + ", ".join(
                f"`{r['arm']}` seed {r['seed']}" for r in result["not_counted"]))
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
                     "| Leg 2 failed | Accuracy (per seed) | Accuracy 95% CI (questions resampled) "
                     "| Paired vs fullest: X-only/F-only, McNemar p (per seed) "
                     "| Paired vs fullest over seeds, clustered by question: difference, "
                     "questions X/F better, sign-flip p | Trainable "
                     "| AdamW fp32 state | Peak CUDA | s/step | Verdict |")
        lines.append("|---|---|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|---|")
        for arm in ARM_ORDER:
            if arm not in arms:
                continue
            row = arms[arm]
            gap = row["task_gap"]
            paired = "-" if row["paired_vs_fullest"] is None else ", ".join(
                f"{p['x_only']}/{p['f_only']} p={p['p']:.3g}" for p in row["paired_vs_fullest"])
            c = row["paired_clustered"]
            clustered = "-" if c is None else (
                f"{c['difference'] * 100:+.1f} pp, {c['questions_x_better']}/"
                f"{c['questions_f_better']}, p={c['p']:.3g}")
            lines.append(
                f"| `{arm}` | {row['seeds']} | {row['nll_mean']:.4f} ± {row['nll_sd']:.4f} "
                f"| {gap['gap']:+.4f} | {'yes' if gap['material'] else 'no'} "
                f"| {', '.join(row['leg2_failed']) or '-'} "
                f"| {', '.join(f'{a:.3f}' for a in row['accuracy'])} "
                f"| [{row['accuracy_ci95'][0]:.3f}, {row['accuracy_ci95'][1]:.3f}] "
                f"| {paired} | {clustered} "
                f"| {row['trainable_parameters'] / 1e6:.1f}M "
                f"| {_fmt_bytes(row['adapter_bytes_adamw_fp32'])} "
                f"| {_fmt_bytes(row['peak_cuda_bytes'])} | {row['seconds_per_step']:.2f} "
                f"| **{row['verdict']}** |")
        lines.append("")
        if result["leg2_failed_by_every_arm"]:
            moved = "; ".join(
                f"`{arm}` " + ", ".join(
                    f"{s} (+{sum(m[s][0] for m in entry['arms'][arm])}"
                    f"/-{sum(m[s][1] for m in entry['arms'][arm])})"
                    for s in arms[arm]["leg2_failed"])
                for arm in ARM_ORDER if arm in arms)
            lines.append(
                "Leg 2 does not discriminate placements on this model: every arm fails it, "
                f"the fullest arm `{result['fullest']}` included. Items gained/lost against "
                f"the base over each arm's seeds, per failing suite: {moved}.")
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
        if entry.get("think"):
            sample = next(iter(entry["think"].values()))[0]
            lines.append(
                f"Gate answers that never close `<think>` within the 256-token budget: base "
                f"{sample['base_unclosed']} of {sample['answers']}. Per arm and seed: answers "
                "left unclosed; items gained and lost, split by whether the base answer had "
                "closed `<think>` (reporting only):")
            lines.append("")
            lines.append("| Arm | Seed | Unclosed | Gained (base unclosed / closed) "
                         "| Lost (base unclosed / closed) |")
            lines.append("|---|---:|---:|---|---|")
            for arm in shown:
                for seed, t in zip(arms[arm]["seeds"], entry["think"][arm]):
                    lines.append(f"| `{arm}` | {seed} | {t['unclosed']} "
                                 f"| {t['gained_base_unclosed']} / {t['gained_base_closed']} "
                                 f"| {t['lost_base_unclosed']} / {t['lost_base_closed']} |")
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


COLORS = {"A": "#1b6ca8", "AS": "#2a9d8f", "AT": "#e76f51", "AE": "#6a4c93"}


def _gap_band(row: dict, full: dict) -> list:
    """NLL minus the fullest arm's, seed-matched: mean - SD, mean, mean + SD over seeds."""
    gaps = [x - y for x, y in zip(row["nll"], full["nll"])]
    mean, sd = sum(gaps) / len(gaps), rule.sample_sd(gaps)
    return [mean - sd, mean, mean + sd]


def _accuracy_band(row: dict) -> list:
    """Low end of the question-resampled 95% CI, mean accuracy over seeds, high end."""
    low, high = row["accuracy_ci95"]
    return [low, sum(row["accuracy"]) / len(row["accuracy"]), high]


def _spread(ys: dict, gap: float) -> dict:
    """Label heights at least ``gap`` apart, each as close to its own point as allowed."""
    placed: dict = {}
    for arm in sorted(ys, key=ys.get):
        floor = max(placed.values(), default=None)
        placed[arm] = ys[arm] if floor is None else max(ys[arm], floor + gap)
    return placed


def _label_positions(points: dict, gap: float, near: float = 30) -> dict:
    """Label ``(x span, y)`` per arm. Points closer than ``near`` px in x form one group
    whose labels stand in one column beside the group, stacked at least ``gap`` px apart,
    so no label crosses a neighbour's interval bar."""
    groups: list = []
    for arm in sorted(points, key=lambda a: points[a][0]):
        if groups and points[arm][0] - points[groups[-1][-1]][0] < near:
            groups[-1].append(arm)
        else:
            groups.append([arm])
    out: dict = {}
    for group in groups:
        heights = _spread({arm: points[arm][1] for arm in group}, gap)
        xs = [points[arm][0] for arm in group]
        out.update({arm: ((min(xs), max(xs)), heights[arm]) for arm in group})
    return out


def svg(model: str, result: dict) -> str:
    """Quality against cost, 2 x 2 panels.

    Rows: NLL gap to the fullest arm (mean +- SD over seeds) and task accuracy (mean over
    seeds, 95% CI resampled by question). Columns: trainable parameters (log scale) and
    measured peak CUDA memory. The base is a caption, not a point: it sits far from every
    arm on both measures and would flatten the scale.
    """
    arms = [(a, result["arms"][a]) for a in ARM_ORDER if a in result.get("arms", {})]
    full = result["arms"][result["fullest"]]
    rows = (("NLL gap to the fullest arm, nat/token (lower is better)",
             lambda row: _gap_band(row, full), 0.0, "{:+.4f}"),
            ("task accuracy, 95% CI by question (higher is better)", _accuracy_band, None,
             "{:.3f}"))
    columns = (("trainable parameters (log scale)",
                lambda row: math.log10(row["trainable_parameters"]),
                lambda v: f"{10 ** v / 1e6:.0f}M"),
               ("peak CUDA memory allocated, GiB",
                lambda row: row["peak_cuda_bytes"] / 2 ** 30, lambda v: f"{v:.1f}"))
    width, height = 860, 700
    left_pad, right_pad, top_pad, gap_x, gap_y, panel_h = 70, 40, 70, 110, 100, 200
    panel_w = (width - left_pad - right_pad - gap_x) / 2
    seeds = len(full["seeds"])
    parts = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              'font-family="sans-serif" font-size="11">'),
             f'<rect width="{width}" height="{height}" fill="#fff"/>',
             (f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14">{model}: '
              f'quality against cost ({seeds} seeds; fullest arm {result["fullest"]})</text>'),
             (f'<text x="{width / 2}" y="38" text-anchor="middle" fill="#555">base, not '
              f'plotted: eval NLL {result["base_nll"]:.4f}, task accuracy '
              f'{result["base_accuracy"]:.3f}</text>')]
    for r, (y_title, band, ref, y_fmt) in enumerate(rows):
        values = {arm: band(row) for arm, row in arms}
        y_lo = min(v[0] for v in values.values())
        y_hi = max(v[2] for v in values.values())
        if ref is not None:
            y_lo, y_hi = min(y_lo, ref), max(y_hi, ref)
        span = (y_hi - y_lo) or 1.0
        y_lo, y_hi = y_lo - 0.12 * span, y_hi + 0.12 * span
        top = top_pad + r * (panel_h + gap_y)
        bottom = top + panel_h

        def py(y, y_lo=y_lo, y_hi=y_hi, top=top, bottom=bottom):
            return bottom - (y - y_lo) / (y_hi - y_lo) * (bottom - top)

        for c, (x_title, x_of, x_label) in enumerate(columns):
            left = left_pad + c * (panel_w + gap_x)
            xs = {arm: x_of(row) for arm, row in arms}
            x_lo, x_hi = min(xs.values()), max(xs.values())
            x_span = (x_hi - x_lo) or 1.0
            x_lo, x_hi = x_lo - 0.08 * x_span, x_hi + 0.08 * x_span

            def px(x, left=left, x_lo=x_lo, x_hi=x_hi):
                return left + (x - x_lo) / (x_hi - x_lo) * panel_w

            parts.append(f'<rect x="{left:.1f}" y="{top}" width="{panel_w:.1f}" '
                         f'height="{panel_h}" fill="none" stroke="#999"/>')
            parts.append(f'<text x="{left + panel_w / 2:.1f}" y="{top - 8}" '
                         f'text-anchor="middle">{y_title}</text>')
            parts.append(f'<text x="{left + panel_w / 2:.1f}" y="{bottom + 30}" '
                         f'text-anchor="middle" fill="#555">{x_title}</text>')
            for tick in (y_lo, (y_lo + y_hi) / 2, y_hi):
                parts.append(f'<text x="{left - 4:.1f}" y="{py(tick) + 4:.1f}" '
                             f'text-anchor="end" fill="#555">{y_fmt.format(tick)}</text>')
            for tick in (x_lo, (x_lo + x_hi) / 2, x_hi):
                parts.append(f'<text x="{px(tick):.1f}" y="{bottom + 14}" text-anchor="middle" '
                             f'fill="#555">{x_label(tick)}</text>')
            if ref is not None:
                parts.append(f'<line x1="{left:.1f}" x2="{left + panel_w:.1f}" '
                             f'y1="{py(ref):.1f}" y2="{py(ref):.1f}" stroke="#999" '
                             'stroke-dasharray="4 3"/>')
            points = {arm: (px(xs[arm]), py(values[arm][1])) for arm, _row in arms}
            labels = _label_positions(points, 12)
            for arm, _row in arms:
                low, mid, high = values[arm]
                (cx, cy), color = points[arm], COLORS[arm]
                parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{py(low):.1f}" '
                             f'y2="{py(high):.1f}" stroke="{color}" stroke-width="2"/>')
                parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>')
                (group_lo, group_hi), label_y = labels[arm]
                anchor = "end" if group_hi > left + panel_w - 40 else "start"
                label_x = group_lo - 8 if anchor == "end" else group_hi + 8
                parts.append(f'<text x="{label_x:.1f}" y="{label_y + 4:.1f}" '
                             f'text-anchor="{anchor}" fill="{color}">{arm}</text>')
    legend_top = top_pad + 2 * panel_h + gap_y + 56
    for i, (arm, row) in enumerate(arms):
        parts.append(f'<text x="{left_pad}" y="{legend_top + 14 * i}" fill="{COLORS[arm]}">'
                     f'{arm}: {row["trainable_parameters"] / 1e6:.1f}M trainable, '
                     f'{_fmt_bytes(row["adapter_bytes_adamw_fp32"])} AdamW fp32 state, peak '
                     f'{_fmt_bytes(row["peak_cuda_bytes"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs")
    parser.add_argument("out")
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--hot", action="append", required=True, metavar="MODEL=SHA256",
                        help="SHA-256 of the hot-set file every AT run of MODEL must record")
    parser.add_argument("--ae-oom-log", action="append", default=[], metavar="MODEL=PATH",
                        help="log of an AE run of MODEL that died; used only if it holds a "
                             "CUDA out-of-memory error and no AE run of MODEL completed")
    args = parser.parse_args(argv)
    expect = {"harness_sha256": args.harness_sha256, "split_sha256": args.split_sha256,
              "hot_sha256": dict(item.split("=", 1) for item in args.hot)}
    ae_oom_logs = {model: pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
                   for model, path in (item.split("=", 1) for item in args.ae_oom_log)}
    records = rule.load_runs(args.runs)
    analysis = rule.analyse(records, expect=expect, ae_oom_logs=ae_oom_logs)
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
                                                 indent=1) + "\n", encoding="utf-8",
                                      newline="\n")
    (out / "tables.md").write_text(tables(analysis, detail) + "\n", encoding="utf-8",
                                   newline="\n")
    for model, result in analysis["models"].items():
        if result.get("arms"):
            (out / f"curve-{model}.svg").write_text(svg(model, result) + "\n",
                                                    encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
