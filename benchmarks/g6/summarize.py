"""Audit four full/pilot arm receipts and export comparison plus blind review forms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint_eval import sha256
from eval_suite import blind_exports
from run import write
from run_support import contract


def summarize(spec: dict, roots: list[Path], mode: str, contract_sha: str, out: Path) -> dict:
    arms = {}
    inventories = set()
    for root in roots:
        prospective = json.loads((root / "prospective_run.json").read_text())
        if prospective["contract_sha"] != contract_sha or prospective["mode"] != mode:
            raise ValueError("Run contract/stage mismatch")
        arm = prospective["arm"]
        if arm not in spec["arm_order"] or arm in arms:
            raise ValueError("Unknown or duplicate arm")
        receipt = json.loads((root / arm / "receipt.json").read_text())
        if receipt["status"] != "ARM_COMPLETED_REVIEW_PENDING":
            raise ValueError("Incomplete/failed/synthetic run cannot become scientific evidence")
        expected = 65536 if mode == "pilot" else 5_000_000
        if (
            receipt["training"]["actual_useful_tokens"] != expected
            or not receipt["training"]["AMP_exercised"]
            or not receipt["reload_exact"]
        ):
            raise ValueError("Invalid actual exposure or reload proof")
        identity = json.loads((root / "inventory.json").read_text())
        inventories.add(json.dumps(identity, sort_keys=True))
        # Verify all retained outputs against per-run completion manifest.
        files = json.loads((root / "files_sha256.json").read_text())
        for name, digest in files.items():
            from run_support import contained

            if sha256(contained(root, name)) != digest:
                raise ValueError("Retained output changed")
        arms[arm] = {"receipt": receipt, "root": root}
    if set(arms) != set(spec["arm_order"]) or len(inventories) != 1:
        raise ValueError("Four arms on one exact GPU/runtime inventory required")
    initials = {value["receipt"]["initial_state_sha256"] for value in arms.values()}
    if len(initials) != 1:
        raise ValueError("Arms did not start from identical states")
    baseline = arms["baseline"]["receipt"]["final_eval"]["mean_ce"]
    initial_ces = {value["receipt"]["initial_eval"]["mean_ce"] for value in arms.values()}
    if len(initial_ces) != 1:
        raise ValueError("Initial eval differs across runs")
    rows = {}
    for arm in spec["arm_order"]:
        value = arms[arm]["receipt"]
        rows[arm] = {
            "final_ce": value["final_eval"]["mean_ce"],
            "relative_ce_vs_baseline": value["final_eval"]["mean_ce"] / baseline - 1,
            "actual_train_s": value["training"]["elapsed_s"],
            "train_useful_tokens_s": value["training"]["actual_useful_tokens"]
            / value["training"]["elapsed_s"],
            "peak_allocated_bytes": value["training"]["cuda_peak_allocated_bytes"],
            "peak_reserved_bytes": value["training"]["cuda_peak_reserved_bytes"],
            "task": value["diagnostics"],
            "F1_timing_accepted": value["F1_timing_accepted"],
            "checkpoint_reload_s": value["checkpoint_and_reload_s"],
        }
    result = {
        "mode": mode,
        "contract_sha256": contract_sha,
        "rows": rows,
        "status": "PILOT_REVIEW_PENDING" if mode == "pilot" else "MANUAL_REVIEW_PENDING",
        "quality_verdict": "NO_VERDICT",
        "single_seed_exploratory": True,
        "initial_state_sha256": next(iter(initials)),
        "F1_all_accepted": all(row["F1_timing_accepted"] for row in rows.values()),
    }
    if mode == "full":
        files = {
            arm: value["root"] / arm / "final_diagnostics.jsonl" for arm, value in arms.items()
        }
        files["initial"] = arms["baseline"]["root"] / "baseline/initial_diagnostics.jsonl"
        suite = json.loads((Path(__file__).parent / "spec/eval_suite_v1.json").read_text())
        blind_exports(files, suite, out / "blind_review")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--runs", type=Path, nargs=4, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = contract(args.contract, args.contract_sha)
    args.out.mkdir(parents=True, exist_ok=False)
    result = summarize(spec, args.runs, args.mode, args.contract_sha, args.out)
    write(args.out / "comparison.json", result)


if __name__ == "__main__":
    main()
