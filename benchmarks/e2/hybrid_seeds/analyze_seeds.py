"""Offline analysis of the E2 hybrid seed series (hybrid_seeds/PROTOCOL.txt 4-5). CPU only.

  python benchmarks/e2/hybrid_seeds/analyze_seeds.py SESSION_DIR [SESSION_DIR ...] \\
      [--reuse benchmarks/e2/hybrid/reports/20261105/evidence/20261105_082550] [--out NEW.json]

SESSION_DIR: unzipped e2hs_results_<stamp> folders. --reuse: the 20261105 hybrid session,
whose W32, H24 and H16 are the seed-42 arms.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recipe_check"))
sys.path.insert(0, str(HERE.parent / "hybrid"))
import analyze_hybrid as hybrid  # noqa: E402
import analyze_recipe as recipe  # noqa: E402

REUSE_DEFAULT = (
    HERE.parent / "hybrid" / "reports" / "20261105" / "evidence" / "20261105_082550"
)
SEEDS = (42, 901, 1234)
KS = (24, 16, 8)
ANCHOR_CE = 0.8757693767547607
ANCHOR_TOLERANCE = 1e-9
VERSIONS = {"torch": "2.11.0+cu130", "transformers": "5.20.2", "peft": "0.21.2"}
CONFIG_PROBLEM = "config differs from its template beyond base/output"


def _config_problems(arm_dir: Path, seed: int) -> list[str]:
    text = (arm_dir / "config.yaml").read_text(encoding="utf-8")
    seeds = re.findall(r"(?m)^  (seed|data_seed): (\d+)$", text)
    if sorted(seeds) != [("data_seed", str(seed)), ("seed", str(seed))]:
        return [f"V4: seed/data_seed are not {seed}"]
    text = re.sub(r"(?m)^  (seed|data_seed): \d+$", r"  \1: 42", text)
    text = re.sub(r"(?m)^base: .*$", "base: __BASE__", text)
    text = re.sub(r"(?m)^output: .*$", "output: __OUTPUT__", text)
    template = (recipe.HERE / "configs" / "W32.yaml").read_text(encoding="utf-8")
    return [] if text == template else ["V4: config differs from the W32 template"]


def _anchor(arm_dir: Path) -> dict:
    out = {"arm": "ANCHOR_s42", "problems": []}
    try:
        record = json.loads((arm_dir / "arm.json").read_text(encoding="utf-8"))
        quality = json.loads((arm_dir / "quality.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        out["problems"].append(f"missing or unreadable file: {exc}")
        return out
    ce = quality.get("validation_loss")
    out["ce"] = ce
    if not record.get("ok"):
        out["problems"].append("arm.json is not ok")
    if quality.get("steps") != recipe.STEPS or quality.get("deterministic_algorithms") is not True:
        out["problems"].append("V1: anchor run incomplete")
    if not isinstance(ce, (int, float)) or not math.isfinite(ce) \
            or abs(ce - ANCHOR_CE) / ANCHOR_CE > ANCHOR_TOLERANCE:
        out["problems"].append(f"V2: anchor CE {ce} != {ANCHOR_CE}")
    out["problems"] += _config_problems(arm_dir, 42)
    return out


def load(arm_dir: Path, name: str) -> dict:
    kind, seed = name.split("_s")
    seed = int(seed)
    if kind == "ANCHOR":
        return _anchor(arm_dir)
    arm = recipe.load_arm(arm_dir, template="W32")
    arm["arm"] = name
    arm["problems"] = [p for p in arm["problems"] if p != CONFIG_PROBLEM]
    arm["problems"] += _config_problems(arm_dir, seed)
    if kind != "W32":
        arm["problems"] += hybrid._hybrid_problems(arm_dir, int(kind[1:]))
    return arm


def _versions(session: Path) -> list[str]:
    try:
        freeze = json.loads((session / "session.json").read_text())["pip_freeze"]
    except (OSError, ValueError, KeyError):
        return [f"V5: no session.json in {session.name}"]
    found = dict(line.split("==", 1) for line in freeze if "==" in line)
    return [
        f"V5: {session.name} has {pkg} {found.get(pkg)}, expected {version}"
        for pkg, version in VERSIONS.items() if found.get(pkg) != version
    ]


def collect(sessions: list[Path], reuse: Path) -> tuple[dict, list[str]]:
    arms, problems = {}, []
    for session in [reuse, *sessions]:
        problems += _versions(session)
        for arm_dir in sorted(p for p in session.iterdir() if p.is_dir()):
            name = f"{arm_dir.name}_s42" if session == reuse else arm_dir.name
            if session == reuse and arm_dir.name not in {"W32", "H24", "H16"}:
                continue
            if not re.fullmatch(r"(W32|ANCHOR|H24|H16|H8)_s\d+", name):
                continue
            if name in arms:
                raise ValueError(f"arm {name} appears twice")
            arms[name] = load(arm_dir, name)
    return arms, problems


def decide(rows: list[dict]) -> str:
    gaps = [r["ce_gap_percent"] for r in rows]
    mean_gap = statistics.fmean(gaps)
    mean_delta = {
        n: statistics.fmean(r["suite_deltas"][n] for r in rows) for n in recipe.GENERAL
    }
    suites_ok = all(v >= -recipe.SUITE_MARGIN for v in mean_delta.values())
    refusal_ok = all(r["over_refusal_top"] >= recipe.OVER_REFUSAL_FLOOR for r in rows)
    if mean_gap <= recipe.CLOSED_GAP and max(gaps) <= hybrid.BORDER and suites_ok and refusal_ok:
        return "WITHIN_MARGIN"
    if mean_gap <= hybrid.BORDER and suites_ok and refusal_ok:
        return "BORDERLINE"
    return "FAILS"


def analyze(sessions: list[Path], reuse: Path = REUSE_DEFAULT) -> dict:
    arms, problems = collect(sessions, reuse)
    expected = ["ANCHOR_s42"] + [
        f"{kind}_s{seed}" for seed in SEEDS for kind in ("W32", "H24", "H16", "H8")
    ]
    problems += [f"{name}: missing" for name in expected if name not in arms]
    problems += [f"{name}: {p}" for name, arm in arms.items() for p in arm["problems"]]
    result = {
        "schema": "e2-hybrid-seeds-analysis-v1",
        "protocol": "benchmarks/e2/hybrid_seeds/PROTOCOL.txt",
        "scorer_revision": recipe.suites.BUNDLED_SCORER_REVISION,
        "arms": arms,
        "not_claimed": ["speed", "cache parity across the freeze", "other models or data"],
    }
    if problems:
        return {**result, "label": "NO_VERDICT", "reasons": problems}
    per_k = {}
    for k in KS:
        rows = []
        for seed in SEEDS:
            row = recipe.pair(arms[f"H{k}_s{seed}"], arms[f"W32_s{seed}"])
            rows.append({"seed": seed, **row})
        per_k[f"K{k}"] = {
            "per_seed": rows,
            "mean_ce_gap_percent": statistics.fmean(r["ce_gap_percent"] for r in rows),
            "mean_suite_deltas": {
                n: statistics.fmean(r["suite_deltas"][n] for r in rows)
                for n in recipe.GENERAL + recipe.REPORTED_ONLY
            },
            "label": decide(rows),
        }
    passing = [k for k in KS if per_k[f"K{k}"]["label"] == "WITHIN_MARGIN"]
    result.update({
        "per_k": per_k,
        "recommended_k_for_speed_check": min(passing) if passing else None,
        "label": "DECIDED",
    })
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--reuse", type=Path, default=REUSE_DEFAULT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = json.dumps(analyze(args.sessions, args.reuse), indent=2) + "\n"
    if args.out:
        with open(args.out, "x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
