"""Offline analysis of one G3 Windows DirectStorage series (PROTOCOL.txt sections 7 and 8).

  python benchmarks/g3/directstorage/analyze_ds.py RESULTS_DIR [--out NEW.json]

RESULTS_DIR is the unzipped results_<timestamp> folder from run_g3.ps1, or several of them
(pilot plus one top-up) passed one after another. Nothing here touches a GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
import arm_validity  # noqa: E402

ORDER = ("A", "B", "B", "A")
VOID_BLOCK_LIMIT = 2
BLOCK_CAP = 12
# Two-sided 95% Student t quantiles for df = 1..11 (at most 12 blocks).
T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
}
ENVIRONMENT_ERRORS = ("LoadLibrary", "missing", "no DXGI adapter", "CreateFileW")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_runs(dirs: list[Path]) -> list[dict]:
    runs = []
    for root in dirs:
        for entry in _load(root / "series.json"):
            name = f"b{entry['block']:02d}_{entry['position']}_{entry['arm']}"
            path = root / "runs" / f"{name}.json"
            report = _load(path) if path.exists() else None
            runs.append({**entry, "name": name, "report": report})
    runs.sort(key=lambda r: (r["block"], r["position"]))
    seen = {(r["block"], r["position"]) for r in runs}
    if len(seen) != len(runs):
        raise ValueError("Two runs share a block and position")
    return runs


def _outcome(run: dict) -> str:
    report = run["report"]
    return report["validity"]["outcome"] if report else arm_validity.UNKNOWN


def _ok(run: dict) -> bool:
    return bool(run["report"] and run["report"]["ok"] and run["exit"] == 0)


def capability(runs: list[dict]) -> dict:
    used = [r for r in runs if _outcome(r) != arm_validity.VOID]
    a = [r for r in used if r["arm"] == "A"]
    b = [r for r in used if r["arm"] == "B"]
    errors = {}
    for r in used:
        if not _ok(r):
            error = (r["report"] or {}).get("error") or f"exit {r['exit']} without a report"
            errors.setdefault(r["arm"], []).append(error)
    a_err, b_err = errors.get("A", []), errors.get("B", [])

    def environment(error: str) -> bool:
        return any(word in error for word in ENVIRONMENT_ERRORS)

    if a and b and not errors:
        outcome = "DS_TO_CUDA_VERIFIED"
    elif a_err:  # the ordinary control did not pass
        outcome = "ENVIRONMENT_BLOCKED" if all(map(environment, a_err)) else "NO_VERDICT"
    elif b_err and all(map(environment, b_err)):
        outcome = "ENVIRONMENT_BLOCKED"
    elif len(b_err) >= 2 and len(set(b_err)) == 1 and len(b_err) == len(b) and a:
        outcome = "DS_UNAVAILABLE_ON_TESTED_STACK"
    else:
        outcome = "NO_VERDICT"
    return {
        "outcome": outcome,
        "errors": errors,
        "valid_runs": len(used),
        "scope": "this laptop, driver and DirectStorage 1.3.0; route not attributed",
    }


def performance(runs: list[dict]) -> dict:
    blocks = {}
    for r in runs:
        blocks.setdefault(r["block"], []).append(r)
    ordered = []
    for number in sorted(blocks):
        block = sorted(blocks[number], key=lambda r: r["position"])
        if [r["arm"] for r in block] != list(ORDER):
            raise ValueError(f"block {number} is not A B B A")
        ordered.append(block)
    rule_blocks = [
        [
            arm_validity.Run((r["report"] or {}).get("useful_gb_per_s") or 0.0, _outcome(r))
            for r in b
        ]
        for b in ordered
    ]
    # arm_validity.valid_blocks: a block with a void run is replaced, not counted.
    valid = [b for b in ordered if all(_outcome(r) != arm_validity.VOID for r in b)]
    void = len(ordered) - len(valid)
    result = {"blocks_run": len(ordered), "valid_blocks": len(valid), "void_blocks": void}
    if void > VOID_BLOCK_LIMIT:
        return {**result, "label": "NO_VERDICT_HOST_NOISE"}
    if len(valid) < arm_validity.MIN_BLOCKS:
        return {**result, "label": "NO_VERDICT", "reason": "fewer than 2 valid blocks"}
    to_add = arm_validity.blocks_to_add(rule_blocks)
    needed = len(valid) + to_add
    result["blocks_needed"] = needed
    if to_add:
        cap_hit = needed > BLOCK_CAP
        return {
            **result,
            "label": "NO_VERDICT",
            "reason": "blocks needed exceed the cap" if cap_hit else f"run {to_add} more block(s)",
            "top_up_blocks": 0 if cap_hit else to_add,
        }
    if any(_outcome(r) != arm_validity.OK for b in valid for r in b):
        return {**result, "label": "NO_VERDICT", "reason": "a required check is unknown"}
    values = {arm: [] for arm in "AB"}
    diffs = []
    for block in valid:
        rates = {
            arm: [r["report"]["useful_gb_per_s"] for r in block if r["arm"] == arm] for arm in "AB"
        }
        for arm in "AB":
            values[arm].extend(rates[arm])
        diffs.append(statistics.fmean(rates["B"]) - statistics.fmean(rates["A"]))
    mean = statistics.fmean(diffs)
    half = T975[len(diffs) - 1] * statistics.stdev(diffs) / len(diffs) ** 0.5
    low, high = mean - half, mean + half
    label = "B_FASTER" if low > 0 else ("B_SLOWER" if high < 0 else "NO_DETECTABLE_DIFFERENCE")
    mean_a, mean_b = statistics.fmean(values["A"]), statistics.fmean(values["B"])
    return {
        **result,
        "label": label,
        "mean_a_gb_per_s": mean_a,
        "mean_b_gb_per_s": mean_b,
        "mean_difference_gb_per_s": mean,
        "difference_95": [low, high],
        "block_differences": diffs,
        "ratio_b_over_a_descriptive": mean_b / mean_a,
    }


def descriptive(runs: list[dict]) -> dict:
    out = {}
    for arm in "AB":
        good = [
            r["report"]
            for r in runs
            if r["arm"] == arm and _ok(r) and _outcome(r) != arm_validity.VOID
        ]
        if not good:
            out[arm] = None
            continue
        out[arm] = {
            "runs": len(good),
            "median_gb_per_s": statistics.median(g["useful_gb_per_s"] for g in good),
            "median_cpu_seconds_per_read": statistics.median(
                g["cpu_seconds_per_read"] for g in good
            ),
            "median_private_bytes_after_reads": statistics.median(
                g["memory"]["after_reads"]["private_bytes"] for g in good
            ),
            "median_peak_working_set": statistics.median(
                g["memory"]["after_reads"]["peak_working_set"] for g in good
            ),
            "explicit_pinned_host_bytes": good[0]["explicit_pinned_host_bytes"],
        }
        if arm == "B":
            dtod = [g["dtod_copy_seconds"] for g in good if g["dtod_copy_seconds"] > 0]
            out[arm]["median_dtod_copy_seconds"] = statistics.median(dtod) if dtod else None
    return out


def analyze(dirs: list[Path]) -> dict:
    runs = load_runs(dirs)
    cap = capability(runs)
    result = {
        "schema": "g3-ds-analysis-v1",
        "protocol": "benchmarks/g3/directstorage/PROTOCOL.txt",
        "capability": cap,
        "descriptive": descriptive(runs),
        "machine": [_load(d / "machine.json") for d in dirs if (d / "machine.json").exists()],
        "not_claimed": ["peer-to-peer NVMe-to-GPU route", "training speed-up", "other machines"],
    }
    if cap["outcome"] == "DS_TO_CUDA_VERIFIED":
        result["performance"] = performance(runs)
    else:
        result["performance"] = {"label": "NO_VERDICT", "reason": "capability not verified"}
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
