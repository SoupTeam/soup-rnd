"""Re-audit retained H2-close evidence for the review of 2026-11-04.

  python benchmarks/harness/h2_close_reaudit.py updates --series EXTRACTED_U_SERIES \
      [--archive ARCHIVE.tar.gz --archive-sha256 HEX] --out NEW.json
  python benchmarks/harness/h2_close_reaudit.py runs --series-dir STAGE_P_SERIES_DIR

``updates``: EXTRACTED_U_SERIES is the `updates` series root inside the preserved stage U archive
(contract.json, baseline_training/, ce_only_training/). For each arm this reruns the unchanged
training-evidence reader (file pins, tensor digests, ordered work, optimizer events and moments)
and then the U2 rule of H2_CLOSE_PROTOCOL.txt: every declared trainable tensor changed its bytes.
It does not rerun the GPU, the F1/kernel checks or the source check: the measured source is the
contract's and this reviewer code is newer by design, so both are recorded.

``runs``: rechecks every stage P run of series.json against its own run.json with the frozen
runner's per-run rules (``h2_perf_abba.run_problems`` and the frozen step-token receipt), then
recomputes the pair labels with the frozen ``h2_perf_abba.analyze``. The frozen runner itself is
not edited: its SHA256 is pinned by perf_workload_receipt.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from h2_close_updates import ARMS, unchanged_adapters  # noqa: E402

REPO = HERE.parents[1]
SPEC = REPO / "benchmarks" / "h2"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reaudit(series: Path, work: list | None = None) -> dict:
    from h2_ce_training import read_training

    if work is None:
        work = json.loads((SPEC / "ce_confirmation_workset.json").read_text())["training_batches"]
    contract = json.loads((series / "contract.json").read_text())
    arms = {}
    for arm, phase in ARMS.items():
        training = series / phase / "runner" / "training"
        try:
            record, _ = read_training(training, work)
            unchanged = unchanged_adapters(record)
        except (OSError, ValueError, KeyError) as error:
            arms[arm] = {"outcome": "U2_REAUDIT_FAIL", "reason": f"{type(error).__name__}: {error}"}
            continue
        declared = len(record["before"]["trainable"])
        arms[arm] = {
            "outcome": "U2_REAUDIT_FAIL" if unchanged else "U2_REAUDIT_PASS",
            "declared_adapter_tensors": declared,
            "changed_adapter_tensors": declared - len(unchanged),
            "unchanged": unchanged,
            "training_files_sha256": json.loads((training / "files_sha256.json").read_text()),
        }
    return {
        "schema": "h2-close-u2-reaudit-v1",
        "rule": "H2_CLOSE_PROTOCOL.txt U2: every declared trainable tensor changed its bytes",
        "arms": arms,
        "measured_source_commit": contract.get("source_commit"),
        "measured_artifacts_sha256": {
            k: v
            for k, v in contract.get("artifacts_sha256", {}).items()
            if k.endswith(("h2_close_updates.py", "h2_ce_training.py"))
        },
        "reviewer_sha256": {
            "benchmarks/harness/h2_close_updates.py": sha_file(HERE / "h2_close_updates.py"),
            "benchmarks/harness/h2_ce_training.py": sha_file(HERE / "h2_ce_training.py"),
            "benchmarks/harness/h2_close_reaudit.py": sha_file(Path(__file__).resolve()),
        },
        "not_rechecked": [
            "F1 and GPU-exclusivity controls",
            "kernel attribution (U5)",
            "measured-source check (reviewer code is newer by design)",
        ],
    }


def review_runs(series_dir: Path, expected_steps: Optional[Sequence[dict]] = None) -> dict:
    """Recheck each logged stage P run against its run.json, then recompute the pair labels."""
    import arm_validity
    from h2_perf_abba import RunRecord, analyze, run_problems

    series_dir = Path(series_dir)
    if expected_steps is None:
        receipt = json.loads((SPEC / "perf_workload_receipt.json").read_text())
        expected_steps = receipt["steps"]["step_tokens"]
    saved = json.loads((series_dir / "series.json").read_text())
    mismatches, records = [], []
    for entry in saved["runs"]:
        name = f"b{entry['block']:02d}_{entry['position']}_{entry['arm']}"
        path = series_dir / name / "run.json"
        report = json.loads(path.read_text()) if path.exists() else None
        if report is None:
            problems = [f"child exit {entry['exit']}"]
        else:
            problems = run_problems(report, expected_steps)
        exclusive = entry["gpu_processes_before"] == [] and entry["gpu_processes_after"] == []
        outcome = entry["watch"]["outcome"] if exclusive else arm_validity.VOID
        value = None
        if report is not None and not problems:
            value = report["throughput"]["useful_tokens_per_second"]
            if entry.get("median_step_seconds") != report["timing"]["median_seconds"]:
                mismatches.append(name + ": median step seconds")
        if problems != entry["problems"]:
            mismatches.append(name + ": problems")
        if outcome != entry["outcome"]:
            mismatches.append(name + ": outcome")
        if value != entry.get("useful_tokens_per_second"):
            mismatches.append(name + ": useful tokens/s")
        memory = dict(report["memory"]) if report else {}
        if memory != (entry.get("memory") or {}):
            mismatches.append(name + ": memory")
        records.append(
            RunRecord(entry["arm"], entry["block"], entry["position"], outcome, value, memory)
        )
    fresh = analyze(records, tuple(saved["order"])) if records else None
    return {
        "status": saved["status"],
        "runs_checked": len(records),
        "run_mismatches": mismatches,
        "analysis": fresh,
        "matches_box": not mismatches
        and json.dumps(fresh, sort_keys=True, default=str)
        == json.dumps(saved["analysis"], sort_keys=True, default=str),
        "taken_as_logged": "F1 watch outcome per run (run.json has no raw sensor data)",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    u = sub.add_parser("updates")
    u.add_argument("--series", type=Path, required=True)
    u.add_argument("--archive", type=Path)
    u.add_argument("--archive-sha256")
    u.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("runs")
    r.add_argument("--series-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "runs":
        print(json.dumps(review_runs(args.series_dir), indent=2, default=str))
        return
    result = reaudit(args.series)
    if args.archive:
        actual = sha_file(args.archive)
        result["archive"] = {
            "name": args.archive.name,
            "sha256": actual,
            "matches_expected": actual == args.archive_sha256,
        }
        if actual != args.archive_sha256:
            result["arms"] = {
                a: {"outcome": "U2_REAUDIT_FAIL", "reason": "archive SHA mismatch"} for a in ARMS
            }
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print({arm: state["outcome"] for arm, state in result["arms"].items()})


if __name__ == "__main__":
    main()
