"""Offline analysis of the E2 recipe check (PROTOCOL.txt sections 4-6). CPU only.

  python benchmarks/e2/recipe_check/analyze_recipe.py RESULTS_DIR [MORE_DIRS] [--out NEW.json]

Each RESULTS_DIR is an unzipped results folder from the notebook (one per Kaggle
session); arm folders (W32, W24, S32, S24, S16) are collected from all of them.
Gate scores are recomputed from the raw outputs with the scorer of this checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from soup_cli.eval import gate_suites as suites

HERE = Path(__file__).resolve().parent
REQUIRED = ("W32", "W24", "S32", "S24")
SECONDARY = ("S16",)
RECORD_CE_W32 = 0.876791179180
RECORD_WEAK_K16_GAP = 8.0491
ANCHOR_TOLERANCE = 0.005
WEAK_GAP_MIN = 1.0
CLOSED_GAP = 0.5
SUITE_MARGIN = 0.05
OVER_REFUSAL_FLOOR = 0.95
PERSIST_FRACTION = 0.75
STEPS, TRAIN_ROWS, VAL_ROWS = 4000, 2000, 200
GENERAL = (
    "mini_mmlu", "mini_common_sense", "mini_instruction",
    "mini_arithmetic", "mini_tool_call", "mini_format_json",
)
REPORTED_ONLY = ("mini_safety",)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _items(name: str):
    return suites.MINI_BENCHMARKS[name] if name in suites.MINI_BENCHMARKS \
        else suites.load_suite_items(name)


def rescore(gate: Path, name: str) -> float:
    """Score suite `name` by replaying the recorded outputs in order."""
    trace = iter(_json(gate / f"{name}.generation_audit.json"))

    def replay(prompt: str) -> str:
        row = next(trace)
        if row["prompt_sha256"] != _sha(prompt.encode()):
            raise ValueError(f"{name}: recorded prompt order differs from the suite")
        if "error_type" in row:
            raise ValueError(f"{name}: a generation failed")
        return row["scored_output_text"]

    score = float(suites.score_bundled_suite(name, replay))
    if next(trace, None) is not None:
        raise ValueError(f"{name}: more recorded outputs than suite items")
    return score


def _config_matches_template(arm_dir: Path, arm: str) -> bool:
    text = (arm_dir / "config.yaml").read_text(encoding="utf-8")
    text = re.sub(r"(?m)^base: .*$", "base: __BASE__", text)
    text = re.sub(r"(?m)^output: .*$", "output: __OUTPUT__", text)
    return text == (HERE / "configs" / f"{arm}.yaml").read_text(encoding="utf-8")


def load_arm(arm_dir: Path) -> dict:
    """Read one arm folder; `problems` lists every V1/V4 failure found."""
    arm = arm_dir.name
    out = {"arm": arm, "problems": []}
    problems = out["problems"]
    try:
        record = _json(arm_dir / "arm.json")
        quality = _json(arm_dir / "quality.json")
        report = _json(arm_dir / "gate" / "report.json")
    except (OSError, ValueError) as exc:
        problems.append(f"missing or unreadable file: {exc}")
        return out
    if not record.get("ok"):
        problems.append("arm.json is not ok")
    if not _config_matches_template(arm_dir, arm):
        problems.append("config differs from its template beyond base/output")
    ce = quality.get("validation_loss")
    checks = {
        "steps": quality.get("steps") == STEPS,
        "finite CE": isinstance(ce, (int, float)) and math.isfinite(ce),
        "train rows": quality.get("train_examples") == TRAIN_ROWS,
        "validation rows": quality.get("validation_examples") == VAL_ROWS,
        "parameters changed": quality.get("parameters_changed") is True,
        "deterministic": quality.get("deterministic_algorithms") is True,
        "scorer revision 5": report.get("scorer_revision") == 5,
        "adapter verified": report.get("saved_adapter_weights_verified") is True,
        "fingerprint": report.get("loaded_lora_fingerprint")
        == quality.get("trainable_fingerprint"),
    }
    problems.extend(f"check failed: {name}" for name, ok in checks.items() if not ok)
    out["ce"] = ce
    scores = {}
    for name in suites.DEFAULT_GENERAL_SUITE:
        recorded = (report.get("scores") or {}).get(name)
        if not recorded:
            problems.append(f"{name}: no score")
            continue
        try:
            again = rescore(arm_dir / "gate", name)
        except (OSError, ValueError, StopIteration) as exc:
            problems.append(f"{name}: replay failed ({exc})")
            continue
        if recorded["items"] != len(_items(name)) or recorded["generation_errors"]:
            problems.append(f"{name}: incomplete generation")
        if abs(again - recorded["score"]) > 1e-12:
            problems.append(f"{name}: re-scored {again} != notebook {recorded['score']}")
        scores[name] = again
    out["scores"] = scores
    return out


def collect(dirs: list[Path]) -> dict:
    arms = {}
    for root in dirs:
        for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if arm_dir.name not in REQUIRED + SECONDARY:
                continue
            if arm_dir.name in arms:
                raise ValueError(f"arm {arm_dir.name} appears twice")
            arms[arm_dir.name] = load_arm(arm_dir)
    return arms


def gap(top: dict, full: dict) -> float:
    return 100.0 * (top["ce"] - full["ce"]) / full["ce"]


def pair(top: dict, full: dict) -> dict:
    deltas = {n: top["scores"][n] - full["scores"][n] for n in GENERAL + REPORTED_ONLY}
    return {
        "ce_gap_percent": gap(top, full),
        "suite_deltas": deltas,
        "over_refusal_top": top["scores"]["mini_over_refusal"],
    }


def label(strong: dict, weak_gap: float) -> str:
    closed = (
        strong["ce_gap_percent"] <= CLOSED_GAP
        and all(strong["suite_deltas"][n] >= -SUITE_MARGIN for n in GENERAL)
        and strong["over_refusal_top"] >= OVER_REFUSAL_FLOOR
    )
    if closed:
        return "GAP_CLOSED"
    if strong["ce_gap_percent"] >= PERSIST_FRACTION * weak_gap:
        return "GAP_PERSISTS"
    return "GAP_REDUCED"


def analyze(dirs: list[Path]) -> dict:
    arms = collect(dirs)
    result = {
        "schema": "e2-recipe-analysis-v1",
        "protocol": "benchmarks/e2/recipe_check/PROTOCOL.txt",
        "scorer_revision": suites.BUNDLED_SCORER_REVISION,
        "arms": arms,
        "not_claimed": ["other seeds", "other models", "more data", "speed", "the cache"],
    }
    reasons = [f"{a}: missing" for a in REQUIRED if a not in arms]
    reasons += [f"{a}: {p}" for a in REQUIRED if a in arms for p in arms[a]["problems"]]
    if reasons:
        return {**result, "label": "NO_VERDICT", "reasons": reasons}
    anchor = abs(arms["W32"]["ce"] - RECORD_CE_W32) / RECORD_CE_W32
    weak = pair(arms["W24"], arms["W32"])
    strong = pair(arms["S24"], arms["S32"])
    result.update({
        "anchor_relative_difference": anchor,
        "weak": weak,
        "strong": strong,
        "strong_minus_weak_full_ce": arms["S32"]["ce"] - arms["W32"]["ce"],
    })
    if anchor > ANCHOR_TOLERANCE:
        reasons.append(f"V2: CE(W32) differs from the record by {anchor:.4%}")
    if weak["ce_gap_percent"] < WEAK_GAP_MIN:
        reasons.append(f"V3: weak gap {weak['ce_gap_percent']:.3f} < {WEAK_GAP_MIN}")
    if reasons:
        return {**result, "label": "NO_VERDICT", "reasons": reasons}
    result["label"] = label(strong, weak["ce_gap_percent"])
    if "S16" in arms:
        if arms["S16"]["problems"]:
            result["secondary_s16"] = {"label": "NO_VERDICT", "reasons": arms["S16"]["problems"]}
        else:
            s16 = pair(arms["S16"], arms["S32"])
            result["secondary_s16"] = {**s16, "label": label(s16, RECORD_WEAK_K16_GAP)}
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = json.dumps(analyze(args.results), indent=2) + "\n"
    if args.out:
        with open(args.out, "x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
