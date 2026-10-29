"""Replay the corrected F2 allocation cohort from its original small records."""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--harness", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.harness))
    import f2_capacity_reference as capacity
    import f2_reference as f2
    from arm_validity import arm_outcome

    from soup_cli.bench.train_report import ParamSnapshot, StepRecord, build_train_report
    from soup_cli.bench.train_run import config_hash

    root = args.directory
    frozen = f2.read_json(root / "frozen.json")
    original_root = next(
        p.rsplit("/", 1)[0] for p in frozen["inputs"] if p.endswith("/schedule.json")
    )
    snapshots = f2.read_json(root / "frozen-inputs-index.json")
    for original, expected in frozen["inputs"].items():
        assert f2.digest(root / snapshots[original]["path"]) == expected, original
    schedule = f2.read_json(root / "schedule.json")
    outcomes = f2.read_json(root / "outcomes.json")
    if outcomes["scheduled"] > 150:
        schedule += f2.read_json(root / "supplement-schedule.json")
    assert len(outcomes["outcomes"]) == len(schedule), "Scheduled commands missing"
    assert len({r["index"] for r in outcomes["outcomes"]}) == len(schedule)
    assert outcomes["pooled_with_prior"] is False
    rows = []
    native_rebuilt = foreign_rebuilt = 0
    for expected, outcome in zip(schedule, outcomes["outcomes"]):
        assert all(outcome[key] == value for key, value in expected.items())
        group, length, kind = outcome["group"], outcome["length"], outcome["kind"]
        assert outcome["raw_result"].startswith(original_root + "/")
        record_path = root / outcome["raw_result"][len(original_root) + 1 :]
        parent = record_path.parent
        record = f2.read_json(record_path)
        audit = f2.read_json(parent / "audit.json")
        validity = f2.read_json(parent / "validity.json")
        command = f2.read_json(parent / "command.json")
        log = (parent / "console.log").read_text(encoding="utf-8")
        assert record["audit"] == audit and record["validity"] == validity
        assert f2.digest(parent / "console.log") == record["log_sha256"]
        assert command["environment_overrides"]["CUDA_VISIBLE_DEVICES"] == frozen["gpu"]["uuid"]
        assert command["native_cli_argv"] == [
            "soup",
            "--verbose",
            *command["argv"][command["argv"].index("--") + 1 :],
        ]
        assert validity["outcome"] == arm_outcome(validity["checks"], validity["required"])
        check = validity["checks"]["foreign_reads"]
        evidence = check["evidence"]
        if evidence.get("foreign_read_bytes_min") is not None:
            disk = evidence["disk_read_bytes_end"] - evidence["disk_read_bytes_start"]
            elsewhere = (
                evidence["other_disk_read_bytes_end"] - evidence["other_disk_read_bytes_start"]
            )
            own = evidence["own_read_bytes"]
            least, most = max(disk - own, 0), max(disk - max(own - elsewhere, 0), 0)
            assert evidence["foreign_read_bytes_min"] == least
            assert evidence["foreign_read_bytes_max"] == most
            verdict = "void" if least >= 1e9 else "ok" if most < 1e9 else "unknown"
            assert check["outcome"] == verdict
            foreign_rebuilt += 1
        trace = [json.loads(line) for line in (parent / "trace.jsonl").read_text().splitlines()]
        telemetry = f2.trace_summary(trace, audit)
        assert telemetry == record["telemetry"]
        assert 7.5 * 1024**3 <= audit["physical_vram_bytes"] <= frozen["gpu"]["memory_bytes"]
        mode = capacity.GROUPS[group][0]
        assert record["cell"] == f"{mode}-{length}" and record["kind"] == kind
        argv = command["native_cli_argv"]
        assert argv[argv.index("--config") + 1] == (
            f"{original_root}/{group}/configs/{mode}-{length}.json"
        )
        full = dict(
            record,
            audit=audit,
            validity=validity,
            log=log,
            telemetry_complete=telemetry["complete"],
            tier_confirmed=f2.tier_confirmed(mode, log),
        )
        generic, _ = f2.judge(full, kind, mode)
        assert generic == record["status"]
        if group == "boundary":
            status, reason = capacity.capacity_verdict(full, log)
            classification = f2.read_json(parent / "capacity-classification.json")
            assert classification == dict(
                status=status, reason=reason, raw_generic_f2_status=generic
            )
        else:
            status = generic
        assert status == outcome["status"]
        prepared = f2.read_json(root / group / "configs" / f"{mode}-{length}.json")
        assert prepared["training"]["stream_source"] == ("auto" if group == "overflow" else "ram")
        assert prepared["training"]["quantization"] == ("4bit" if group == "ram" else "none")
        before = f2.read_json(parent / "allocation-before.json")
        assert before["effective_total"] == frozen["memory"]["effective_total"]
        assert int(before["cgroup"]["memory.low"]) == before["effective_total"]
        bench = record["bench"]
        if (parent / "bench.json").exists():
            assert bench == f2.read_json(parent / "bench.json")
            resolved = dict(prepared)
            resolved.pop("output")
            assert bench["resolved_config"] == resolved
            assert bench["config_hash"] == config_hash(resolved)
            rebuilt = build_train_report(
                steps=[StepRecord(**step) for step in audit["steps"]],
                warmup_steps=4,
                trainable=ParamSnapshot(
                    audit["count"],
                    audit["fingerprint_counted_first"],
                    audit["fingerprint_last"],
                    audit["meta_excluded"],
                    audit["notes"],
                ),
                provenance=bench["provenance"],
                memory={
                    "max_memory_allocated_bytes": audit["peak_allocated_bytes"],
                    "max_memory_reserved_bytes": audit["peak_reserved_bytes"],
                },
                config_hash=config_hash(resolved),
                steps_requested=12,
                extra={"resolved_config": resolved},
            )
            assert bench == rebuilt, f"Native report rebuild differs: {parent}"
            native_rebuilt += 1
        if status == "PASS":
            assert {s["total_tokens"] for s in audit["steps"]} == {length}
            assert audit["count"] == 192 and audit["meta_excluded"] == 0
        rows.append(
            dict(
                group=group,
                length=length,
                kind=kind,
                status=status,
                id=parent.name,
                index=outcome["index"],
                block=outcome["block"],
                phase="primary" if outcome["index"] <= 150 else "supplement",
                record=record,
                validity=validity,
            )
        )

    cells = []
    for group in capacity.GROUPS:
        for length in f2.LENGTHS:
            selected = [r for r in rows if r["group"] == group and r["length"] == length]
            working = [r["record"] for r in selected if r["status"] == "PASS"]
            benches = [
                r["record"] for r in selected if r["status"] == "PASS" and r["kind"] == "bench"
            ]
            cell = dict(
                group=group,
                length=length,
                counts=dict(Counter(f"{r['kind']}:{r['status']}" for r in selected)),
            )
            if working:
                for name in ("allocated", "reserved"):
                    cell[f"peak_{name}_all_phases_bytes"] = max(
                        max(
                            r["audit"][f"peak_{name}_bytes"],
                            r["audit"].get(f"setup_peak_{name}_bytes", 0),
                        )
                        for r in working
                    )
                cell["sampled_peak_process_tree_rss_bytes"] = max(
                    r["telemetry"]["sampled_peak_process_tree_rss_bytes"] for r in working
                )
            if benches:
                med = statistics.median
                rates = [r["bench"]["throughput"]["useful_tokens_per_second"] for r in benches]
                cell.update(
                    valid_bench_runs=len(benches),
                    step_seconds_median=med(
                        r["bench"]["timing"]["median_seconds"] for r in benches
                    ),
                    useful_tokens_per_second_median=med(
                        r["bench"]["throughput"]["useful_tokens_per_second"] for r in benches
                    ),
                    total_tokens_per_second_median=med(
                        r["bench"]["throughput"]["total_tokens_per_second"] for r in benches
                    ),
                    mean_gpu_util_percent=statistics.mean(
                        r["telemetry"]["mean_gpu_util_percent"] for r in benches
                    ),
                    median_sm_clock_mhz=med(r["telemetry"]["median_sm_clock_mhz"] for r in benches),
                    median_power_watts=med(r["telemetry"]["median_power_watts"] for r in benches),
                    sampled_read_MBps_median=med(
                        r["telemetry"]["sampled_read_bytes_delta"]
                        / r["telemetry"]["sampled_span_seconds"]
                        / 1e6
                        for r in benches
                    ),
                    useful_tokens_per_second_min=min(rates),
                    useful_tokens_per_second_max=max(rates),
                    useful_tokens_per_second_cv=(
                        statistics.stdev(rates) / statistics.mean(rates) if len(rates) > 1 else None
                    ),
                )
            cells.append(cell)
    summary = dict(
        source_commit=frozen["commit"],
        records_rejudged=len(rows),
        native_reports_rebuilt=native_rebuilt,
        performance_bench_admitted=sum(
            r["kind"] == "bench" and r["status"] == "PASS" for r in rows
        ),
        working_commands_completed_12_steps_with_updates=sum(
            r["group"] != "boundary"
            and r["record"]["audit"].get("complete", False)
            and len(r["record"]["audit"].get("steps", [])) == 12
            and bool(r["record"]["audit"].get("fingerprint_first"))
            and r["record"]["audit"].get("fingerprint_first")
            != r["record"]["audit"].get("fingerprint_last")
            and bool(r["record"]["audit"].get("fingerprint_counted_first"))
            and r["record"]["audit"].get("fingerprint_counted_first")
            != r["record"]["audit"].get("fingerprint_last")
            for r in rows
        ),
        foreign_bounds_rebuilt=foreign_rebuilt,
        statuses=dict(Counter(r["status"] for r in rows)),
        cells=cells,
        acceptance_counts_satisfied=not capacity.needs_supplement(outcomes["outcomes"]),
        baseline_allocation_bytes=frozen["memory"]["effective_total"],
        initial_baseline_free_bytes=frozen["initial_baseline_free_bytes"],
        burst_limit_bytes=int(frozen["memory"]["cgroup"]["memory.max"]),
        pooled_with_prior=False,
        phases={
            phase: {
                "records": sum(r["phase"] == phase for r in rows),
                "statuses": dict(Counter(r["status"] for r in rows if r["phase"] == phase)),
                "cells": [
                    {
                        "group": group,
                        "length": length,
                        "kind": kind,
                        "statuses": dict(
                            Counter(
                                r["status"]
                                for r in rows
                                if r["phase"] == phase
                                and r["group"] == group
                                and r["length"] == length
                                and r["kind"] == kind
                            )
                        ),
                    }
                    for group in capacity.GROUPS
                    for length in f2.LENGTHS
                    for kind in ("train", "bench")
                ],
            }
            for phase in ("primary", "supplement")
        },
        initial_fingerprint_variants={
            f"{g}-{length}": len(
                {
                    r["record"]["audit"]["fingerprint_first"]
                    for r in rows
                    if r["group"] == g and r["length"] == length and r["status"] == "PASS"
                }
            )
            for g in ("ram", "overflow")
            for length in f2.LENGTHS
        },
        whole_execution_clock_events=dict(
            sum(
                (Counter(r["validity"]["gpu_clock"].get("reasons_seen", {})) for r in rows),
                Counter(),
            )
        ),
        unknown_clock_records=[
            r["id"] for r in rows if r["validity"]["gpu_clock"]["outcome"] != "ok"
        ],
        tensor_hash_recheck="Raw replay excludes checkpoint/cache tensor blobs; "
        "full SHA hashes and per-arm metadata checks were recorded on the rental.",
        invalid=[
            {
                **{k: r[k] for k in ("id", "group", "length", "kind", "status", "phase")},
                "reason": r["record"]["reason"],
                "f1": r["validity"]["outcome"],
                "f1_foreign_reads": r["validity"]["checks"]["foreign_reads"],
            }
            for r in rows
            if r["status"] not in ("PASS", "EXPECTED_CAPACITY_REFUSAL")
        ],
    )
    (root / "analysis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8", newline="\n"
    )
    with (root / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "index",
                "block",
                "phase",
                "group",
                "length",
                "kind",
                "status",
                "useful_tok_s",
                "step_seconds",
                "f1",
            ],
        )
        writer.writeheader()
        for row in rows:
            bench = row["record"]["bench"] or {}
            writer.writerow(
                {
                    **{
                        k: row[k]
                        for k in (
                            "id",
                            "index",
                            "block",
                            "phase",
                            "group",
                            "length",
                            "kind",
                            "status",
                        )
                    },
                    "useful_tok_s": bench.get("throughput", {}).get("useful_tokens_per_second"),
                    "step_seconds": bench.get("timing", {}).get("median_seconds"),
                    "f1": row["validity"]["outcome"],
                }
            )
    Console().print(
        {
            k: summary[k]
            for k in (
                "records_rejudged",
                "native_reports_rebuilt",
                "statuses",
                "acceptance_counts_satisfied",
            )
        }
    )


if __name__ == "__main__":
    main()
