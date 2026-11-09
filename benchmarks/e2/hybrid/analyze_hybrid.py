"""Offline analysis of the E2 hybrid check (hybrid/PROTOCOL.txt sections 4-5). CPU only.

  python benchmarks/e2/hybrid/analyze_hybrid.py RESULTS_DIR [--out NEW.json]

RESULTS_DIR is the unzipped session folder holding H24, H16 and W32.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recipe_check"))
import analyze_recipe as recipe  # noqa: E402

PRIMARY, SECONDARY, CONTROL = "H24", "H16", "W32"
KEEP_TOP = {"H24": 24, "H16": 16}
LAYERS = 32
BORDER = 1.0
TOPK_SAME_SESSION_REFERENCE = {"H24": 1.5449222934611186}  # recipe_check W24 vs W32


def load(arm_dir: Path) -> dict:
    """recipe_check checks against the W32 template, plus the hybrid mechanics (V3)."""
    arm = recipe.load_arm(arm_dir, template=CONTROL)
    if arm_dir.name in KEEP_TOP:
        arm["problems"].extend(_hybrid_problems(arm_dir, KEEP_TOP[arm_dir.name]))
    return arm


def _hybrid_problems(arm_dir: Path, keep_top: int, layers: int | None = None) -> list[str]:
    try:
        hybrid = json.loads((arm_dir / "quality.json").read_text(encoding="utf-8"))["hybrid"]
    except (OSError, ValueError, KeyError) as exc:
        return [f"V3: no hybrid record ({exc})"]
    cutoff = (layers or LAYERS) - keep_top
    checks = {
        "freeze reached": hybrid.get("frozen") is True,
        "freeze at step 2000": hybrid.get("freeze_at_step") == 2000,
        "frozen layers 0..31-K": hybrid.get("frozen_layers") == list(range(cutoff)),
        "two LoRA tensors per q/v module": len(hybrid.get("frozen_tensors", []))
        == cutoff * 2 * 2,
        "bottom changed before freeze": hybrid.get("bottom_changed_before_freeze") is True,
        "bottom bit-identical after freeze": hybrid.get("bottom_unchanged_after_freeze") is True,
        "top changed after freeze": hybrid.get("top_changed_after_freeze") is True,
        "bottom requires_grad False": hybrid.get("requires_grad_after") == ["False"],
    }
    return [f"V3: {name}" for name, ok in checks.items() if not ok]


def label(row: dict) -> str:
    suites_ok = all(row["suite_deltas"][n] >= -recipe.SUITE_MARGIN for n in recipe.GENERAL)
    refusal_ok = row["over_refusal_top"] >= recipe.OVER_REFUSAL_FLOOR
    if suites_ok and refusal_ok and row["ce_gap_percent"] <= recipe.CLOSED_GAP:
        return "WITHIN_MARGIN"
    if suites_ok and refusal_ok and row["ce_gap_percent"] <= BORDER:
        return "BORDERLINE"
    return "FAILS"


def analyze(root: Path) -> dict:
    arms = {
        p.name: load(p) for p in sorted(root.iterdir())
        if p.is_dir() and p.name in (PRIMARY, SECONDARY, CONTROL)
    }
    result = {
        "schema": "e2-hybrid-analysis-v1",
        "protocol": "benchmarks/e2/hybrid/PROTOCOL.txt",
        "scorer_revision": recipe.suites.BUNDLED_SCORER_REVISION,
        "arms": arms,
        "not_claimed": ["speed", "cache parity across the freeze", "other seeds or recipes"],
    }
    reasons = [f"{a}: missing" for a in (PRIMARY, SECONDARY, CONTROL) if a not in arms]
    for name in (PRIMARY, CONTROL):
        if name in arms:
            reasons += [f"{name}: {p}" for p in arms[name]["problems"]]
    if CONTROL in arms and not arms[CONTROL]["problems"]:
        anchor = abs(arms[CONTROL]["ce"] - recipe.RECORD_CE_W32) / recipe.RECORD_CE_W32
        result["anchor_relative_difference"] = anchor
        if anchor > recipe.ANCHOR_TOLERANCE:
            reasons.append(f"V2: CE(W32) differs from the record by {anchor:.4%}")
    if any(r.startswith((PRIMARY, CONTROL, "V2")) for r in reasons):
        return {**result, "label": "NO_VERDICT", "reasons": reasons}
    for name in (PRIMARY, SECONDARY):
        if name not in arms:
            continue
        if arms[name]["problems"]:
            result[name] = {"label": "NO_VERDICT", "reasons": arms[name]["problems"]}
            continue
        row = recipe.pair(arms[name], arms[CONTROL])
        row["label"] = label(row)
        if name in TOPK_SAME_SESSION_REFERENCE:
            row["top_k_gap_reference"] = TOPK_SAME_SESSION_REFERENCE[name]
        result[name] = row
    result["label"] = result[PRIMARY]["label"]
    if reasons:
        result["reasons"] = reasons
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = json.dumps(analyze(args.results), indent=2, default=str) + "\n"
    if args.out:
        with open(args.out, "x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
