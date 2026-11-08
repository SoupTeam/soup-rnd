"""Offline analysis of the E2 hybrid-cache series (hybrid_cache/PROTOCOL.txt 4-5). CPU only.

  python benchmarks/e2/hybrid_cache/analyze_speed.py SESSION_DIR [--out NEW.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recipe_check"))
sys.path.insert(0, str(HERE.parent / "hybrid"))
import analyze_hybrid as hybrid  # noqa: E402
import analyze_recipe as recipe  # noqa: E402

STEPS, STEPS_PER_EPOCH, EPOCHS, KEEP_TOP = 6000, 2000, 3, 8
MIN_MEDIAN, MIN_EACH = 1.10, 1.00
ORDER = ("C",) + tuple(
    f"{kind}_c{cycle}_{index}"
    for cycle in (1, 2, 3)
    for kind, index in (("A", 1), ("B", 1), ("B", 2), ("A", 2))
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(arm_dir: Path) -> dict:
    name = arm_dir.name
    out = {"arm": name, "problems": []}
    problems = out["problems"]
    try:
        record, quality = _json(arm_dir / "arm.json"), _json(arm_dir / "quality.json")
    except (OSError, ValueError) as exc:
        problems.append(f"missing or unreadable file: {exc}")
        return out
    ce = quality.get("validation_loss")
    epochs = quality.get("epoch_timing") or []
    out.update({
        "ce": ce,
        "fingerprint": quality.get("trainable_fingerprint"),
        "batches": quality.get("training_batches_sha256"),
        "epoch_seconds": [e["duration_seconds"] for e in epochs],
        "epoch_cache": [(e.get("cache_hits"), e.get("cache_misses")) for e in epochs],
        "process_wall_seconds": record.get("process_wall_seconds"),
        "gpu_state_before": record.get("gpu_state_before"),
        "gpu_state_after": record.get("gpu_state_after"),
        "cache": (quality.get("hybrid") or {}).get("cache"),
    })
    checks = {
        "run ok": record.get("ok") is True,
        "6000 steps": quality.get("steps") == STEPS,
        "finite CE": isinstance(ce, (int, float)) and math.isfinite(ce),
        "rows": quality.get("train_examples") == recipe.TRAIN_ROWS
        and quality.get("validation_examples") == recipe.VAL_ROWS,
        "deterministic": quality.get("deterministic_algorithms") is True,
        "three epochs of 2000 steps": [e["end_step"] - e["start_step"] for e in epochs]
        == [STEPS_PER_EPOCH] * EPOCHS,
    }
    problems += [f"V1: {n}" for n, ok in checks.items() if not ok]
    if not recipe._config_matches_template(arm_dir, "W32"):
        problems.append("V5: config differs from the W32 template")
    hybrid_record = quality.get("hybrid")
    if name == "C" or name.startswith("B"):
        problems += hybrid._hybrid_problems(arm_dir, KEEP_TOP)
    elif hybrid_record is not None:
        problems.append("V1: an A run froze layers")
    if name.startswith("B"):
        problems += _cache_problems(out, record)
    elif out["cache"] is not None:
        problems.append("V3: a cache was installed outside B")
    return out


def _cache_problems(run: dict, record: dict) -> list[str]:
    cache = run["cache"] or {}
    if cache.get("installed_at_step") != STEPS_PER_EPOCH or cache.get("cutoff") != 32 - KEEP_TOP:
        return ["V3: cache not installed at step 2000 over layers 0-23"]
    if len(run["epoch_cache"]) != EPOCHS:
        return ["V3: no per-epoch cache counters"]
    (_, _), (h2, m2), (h3, m3) = run["epoch_cache"]
    problems = []
    if h2 is None or m2 is None or h2 + m2 != STEPS_PER_EPOCH:
        problems.append(f"V3: epoch 2 hits+misses {h2}+{m2} != 2000")
    if (h3, m3) != (STEPS_PER_EPOCH, 0):
        problems.append(f"V3: epoch 3 hits/misses {h3}/{m3}, expected 2000/0")
    if cache.get("files") != m2:
        problems.append(f"V3: {cache.get('files')} cache files for {m2} misses")
    return problems


def _parity_problems(runs: dict) -> list[str]:
    problems = []
    keys = ("ce", "fingerprint", "batches")
    reference = runs["C"]
    for name, run in runs.items():
        if name.startswith("B") and any(run[k] != reference[k] for k in keys):
            problems.append(f"V2: {name} differs from C (cache parity)")
    a_runs = [run for name, run in runs.items() if name.startswith("A")]
    if any(any(run[k] != a_runs[0][k] for k in keys) for run in a_runs[1:]):
        problems.append("V2: A runs differ from each other")
    return problems


def label(ratios: list[float], good: str) -> str:
    ok = statistics.median(ratios) >= MIN_MEDIAN and all(r >= MIN_EACH for r in ratios)
    return good if ok else "NOT_SHOWN"


def analyze(session: Path) -> dict:
    runs = {p.name: load(p) for p in sorted(session.iterdir()) if p.name in ORDER}
    reasons = [f"{name}: missing" for name in ORDER if name not in runs]
    reasons += [f"{n}: {p}" for n, r in runs.items() for p in r["problems"]]
    result = {
        "schema": "e2-hybrid-cache-analysis-v1",
        "protocol": "benchmarks/e2/hybrid_cache/PROTOCOL.txt",
        "runs": runs,
        "not_claimed": ["other GPUs, models, K, data or epochs", "quality"],
    }
    if not reasons:
        reasons += _parity_problems(runs)
    if reasons:
        return {**result, "label": "NO_VERDICT", "reasons": reasons}

    def mean(names, metric):
        return statistics.fmean(metric(runs[n]) for n in names)

    cycles = []
    for c in (1, 2, 3):
        a, b = [f"A_c{c}_1", f"A_c{c}_2"], [f"B_c{c}_1", f"B_c{c}_2"]
        cycles.append({
            "cycle": c,
            "warm_epoch3": mean(a, lambda r: r["epoch_seconds"][2])
            / mean(b, lambda r: r["epoch_seconds"][2]),
            "whole_training": mean(a, lambda r: sum(r["epoch_seconds"]))
            / mean(b, lambda r: sum(r["epoch_seconds"])),
            "epoch2": mean(a, lambda r: r["epoch_seconds"][1])
            / mean(b, lambda r: r["epoch_seconds"][1]),
            "process_wall": mean(a, lambda r: r["process_wall_seconds"])
            / mean(b, lambda r: r["process_wall_seconds"]),
        })
    warm = [c["warm_epoch3"] for c in cycles]
    whole = [c["whole_training"] for c in cycles]
    result.update({
        "cycles": cycles,
        "warm_median": statistics.median(warm),
        "whole_training_median": statistics.median(whole),
        "label": label(warm, "WARM_SPEEDUP"),
        "secondary_label": label(whole, "TRAINING_SPEEDUP"),
        "descriptive": {
            "epoch2_median": statistics.median(c["epoch2"] for c in cycles),
            "process_wall_median": statistics.median(c["process_wall"] for c in cycles),
            "install_seconds": [runs[n]["cache"]["install_seconds"] for n in runs
                                if n.startswith("B")],
            "cache_bytes": [runs[n]["cache"]["bytes"] for n in runs if n.startswith("B")],
            "c_epoch_seconds": runs["C"]["epoch_seconds"],
        },
    })
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = json.dumps(analyze(args.session), indent=2) + "\n"
    if args.out:
        with open(args.out, "x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
