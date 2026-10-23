"""Source-pinned complete trained-OLMoE optimizer-step cells.

No formal cell runs before clean-Git export of a fully prepared protocol.
Development tests use the assembly helper directly with clearly synthetic data.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import time
import traceback
from pathlib import Path

from d3_full_model import file_hash, tensor_hash, validate_bundle
from d3_training_model import MODES, open_training_model
from d3_training_protocol import (
    CODECS,
    ROOT,
    export_packet,
    frozen_source,
    native_oracle_path,
    read_rule,
)


class NumericalFailureError(RuntimeError):
    """Numerical/control mismatch, never a replaceable invalid measurement."""


def save(path: Path, data: dict) -> None:
    path.write_bytes((json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def same_frame(want: dict, actual: dict, *, partial: bool = False) -> None:
    import torch

    expected = set(actual) if partial else set(want)
    if set(actual) != expected or not expected <= set(want):
        raise NumericalFailureError("complete-step tensor set differs from the native oracle")
    for name in sorted(expected):
        if (want[name].dtype != actual[name].dtype or want[name].shape != actual[name].shape
                or not torch.equal(want[name], actual[name])):
            raise NumericalFailureError(f"complete-step native exactness failed: {name}")


def runtime_identity(rule: dict, library: Path) -> dict:
    import platform

    import torch
    from arm_validity import _query_gpu, gpu_processes

    if sys.platform != "linux" or gpu_processes() != []:
        raise RuntimeError("one idle Linux CUDA workload with readable peer telemetry required")
    software = {"python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "driver": _query_gpu("driver_version")[0]}
    software.update({name: importlib.metadata.version(name) for name in
                     ("transformers", "triton", "bitsandbytes", "safetensors")})
    if software != rule["software"] or _query_gpu("uuid") != [rule["gpu_uuid"]]:
        raise ValueError("actual execution software/GPU differs from the frozen protocol")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("native BF16 CUDA required")
    if torch.cuda.get_device_properties(0).total_memory < 23 * 2**30:
        raise RuntimeError("the declared 24 GB GPU class is required for this training gate")
    if file_hash(library) != rule["cublas_library_sha256"]:
        raise ValueError("cuBLAS library differs from the frozen protocol")
    return software


def prepared_identity(prepared: Path, codec: str, rule: dict) -> tuple[dict, object]:
    from d3_f1_controls import verified_selector

    for name in ("mapping", "controls"):
        if file_hash(prepared / (name + ".json")) != rule[name + "_sha256"]:
            raise ValueError("prepared instrument identity differs from the frozen protocol")
    controls = json.loads((prepared / "controls.json").read_text())
    if (controls.get("status") != "controls_passed"
            or [row["case"] for row in controls["controls"]] != ["baseline", "own", "foreign"]
            or [row["observed"] for row in controls["controls"]] != ["ok", "ok", "void"]):
        raise ValueError("effective baseline/own/independent-foreign F1 controls required")
    mapping = json.loads((prepared / "mapping.json").read_text())
    if (mapping.get("counter_source", "cgroup-v2") != rule.get("disk_counter_source", "cgroup-v2")
            or controls.get("counter_source", "cgroup-v2")
            != rule.get("disk_counter_source", "cgroup-v2")):
        raise ValueError("F1 discovery/controls do not match the frozen counter family")
    selector = verified_selector(mapping)
    folder = prepared / codec
    bundle = validate_bundle(folder)
    if file_hash(folder / "bundle.json") != rule["bundle_sha256"][codec]:
        raise ValueError("prepared bundle manifest differs from the frozen protocol")
    if (bundle["provenance"]["model_revision"] != rule["model_revision"]
            or bundle["provenance"]["checkpoint_sha256"] != rule["checkpoint_files"]):
        raise ValueError("prepared weights do not descend from the pinned checkpoint")
    config = json.loads((folder / "config.json").read_text())
    if tuple(config[name] for name in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                       "num_experts", "num_experts_per_tok")) != (
        2048, 1024, 16, 64, 8,
    ):
        raise ValueError("synthetic/tiny model cannot enter the trained-model gate")
    if len(bundle["layers"]) != 16 or len({selector(str(folder / row["file"]))
                                           for row in bundle["layers"]}) != 1:
        raise ValueError("all sixteen expert shards need the verified storage mapping")
    return bundle, selector


def oracle_identity(session: Path, codec: str, source: dict, rule: dict,
                    session_id: str, input_hash: str) -> tuple[Path, dict]:
    folder = native_oracle_path(session, codec)
    record = json.loads((folder / "result.json").read_text())
    if (record.get("status") != "ok" or record.get("mode") != "native"
            or record.get("source") != source or record.get("session_id") != session_id
            or record.get("protocol_sha256") != file_hash(ROOT / (
                "benchmarks/protocols/d3-trained-step/protocol.json"))
            or record.get("input_sha256") != input_hash
            or record.get("bundle_sha256") != rule["bundle_sha256"][codec]
            or record.get("F1", {}).get("outcome") != "ok"):
        raise ValueError("native oracle is not an accepted cell from this exact session")
    for name, expected in record["oracle_files"].items():
        if Path(name).name != name or file_hash(folder / name) != expected:
            raise ValueError("native oracle tensor file identity differs")
    return folder, record


def checked_step(run, ids, *, partial: bool = False, expected: dict | None = None):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = run.step(ids)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    row = {"seconds": seconds, "peak_allocated": torch.cuda.max_memory_allocated(),
           "peak_reserved": torch.cuda.max_memory_reserved()}
    frame = run.frame(outputs)
    del outputs
    if expected is not None:
        same_frame(expected, frame, partial=partial)
    if any(frame["gradient/" + name].abs().sum().item() == 0 for name in ("a", "b")):
        raise NumericalFailureError("adapter gradient is identically zero")
    row["tensor_sha256"] = {name: tensor_hash(value) for name, value in frame.items()}
    run.check_dispatch()
    if run.owner.gemm is not None and run.owner.gemm.stats()["grouped_matrices"] < 1:
        raise RuntimeError("the trained grouped path did not exercise grouped GEMM")
    return frame, row


def correctness(run, ids, folder: Path, bundle: dict, rule: dict, oracle: Path | None) -> dict:
    import torch
    from d3_full_model_gate import zero_active_down
    from safetensors.torch import load_file, save_file

    positive, _row = checked_step(run, ids)
    run.reset()
    aa, _row = checked_step(run, ids, expected=positive)
    del aa
    target = min(run.routes[0].unique().tolist())
    run.reset()
    with zero_active_down(run.owner, target, bundle):
        negative, _row = checked_step(run, ids)
    if torch.equal(positive["logits"], negative["logits"]) or all(
        torch.equal(positive["gradient/" + name], negative["gradient/" + name])
        for name in ("a", "b")
    ):
        raise NumericalFailureError("changed active expert did not change output and derivatives")
    negative_hashes = {name: tensor_hash(value) for name, value in negative.items()}
    del negative
    run.reset()
    restored, _row = checked_step(run, ids, expected=positive)
    del restored, positive
    run.reset()
    rows, files = [], {}
    for step in range(rule["steps"]):
        want = load_file(str(oracle / f"step-{step}.safetensors")) if oracle else None
        frame, row = checked_step(run, ids, expected=want)
        if run.mode == "native":
            name = f"step-{step}.safetensors"
            save_file(frame, str(folder / name))
            files[name] = file_hash(folder / name)
        rows.append(row)
        del frame, want
    if all(torch.equal(run.parameters[name].detach().cpu(), run.initial[name])
           for name in ("a", "b")):
        raise NumericalFailureError("optimizer steps did not change adapter parameters")
    return {"steps": rows, "oracle_files": files, "control_target": target,
            "negative_tensor_sha256": negative_hashes,
            "A_A_exact": True, "restored_exact": True}


def timed_or_profile(run, ids, folder: Path, rule: dict, oracle: Path, kind: str) -> dict:
    import torch
    from safetensors.torch import load_file

    rows = []
    for step in range(rule["warmup_steps"]):
        want = load_file(str(oracle / f"step-{step}.safetensors"))
        frame, _row = checked_step(run, ids, partial=True, expected=want)
        del frame, want
    for offset in range(rule["timed_steps"] if kind == "timing" else 1):
        step = rule["warmup_steps"] + offset
        want = load_file(str(oracle / f"step-{step}.safetensors"))
        if kind == "profile":
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as trace:
                outputs = run.step(ids)
                torch.cuda.synchronize()
            trace.export_chrome_trace(str(folder / "cuda-profile.json"))
            frame = run.frame(outputs)
            same_frame(want, frame, partial=True)
            events = json.loads((folder / "cuda-profile.json").read_text())["traceEvents"]
            kernels = [event for event in events if event.get("cat") == "kernel"
                       and event.get("ph") == "X"]
            if not kernels:
                raise RuntimeError("profiler did not retain actual CUDA kernel events")
            hashes = {name: tensor_hash(value) for name, value in frame.items()}
            rows.append({"cuda_kernel_events": len(kernels),
                         "kernel_names": [event["name"] for event in kernels],
                         "tensor_sha256": hashes})
            del outputs
        else:
            frame, row = checked_step(run, ids, partial=True, expected=want)
            rows.append(row)
        del frame, want
    run.check_dispatch()
    return {"steps": rows}


def child(args, rule: dict) -> int:
    source = frozen_source()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "benchmarks/protocols/d3-trained-step/protocol.json"
    record = {"status": "no_verdict", "source": source, "mode": args.mode, "codec": args.codec,
              "kind": args.kind, "formal_execution": True, "d3_accepted": False,
              "protocol_sha256": file_hash(protocol_path)}
    run = monitor = None
    try:
        import resource

        import torch
        from arm_validity import (
            ArmWatch,
            gpu_processes,
        )
        from d3_cublas_grouped_probe import reference_policy
        from d3_f1_controls import verified_counters

        session = json.loads((args.session / "session.json").read_text())
        if session["source"] != source or session["status"] != "running":
            raise ValueError("formal cell needs its live one-attempt parent session")
        record["session_id"] = session["session_id"]
        record["software"] = runtime_identity(rule, args.library)
        bundle, selector = prepared_identity(args.prepared, args.codec, rule)
        mapping = json.loads((args.prepared / "mapping.json").read_text())
        disk_counter, other_counter = verified_counters(mapping)
        folder = args.prepared / args.codec
        watched = [folder / row["file"] for row in bundle["layers"]]
        watched += [folder / name for name in ("config.json", "nonexpert.safetensors",
                                               "bundle.json")]

        def fingerprints():
            return {str(path): tuple(getattr(path.stat(), name) for name in
                                     ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
                    for path in watched}

        before_files = fingerprints()
        config = json.loads((folder / "config.json").read_text())
        ids = torch.randint(2, config["vocab_size"], (1, rule["tokens"]),
                            generator=torch.Generator().manual_seed(rule["input_seed"])).cuda()
        record.update(input_sha256=tensor_hash(ids),
                      bundle_sha256=rule["bundle_sha256"][args.codec])
        oracle = None
        if args.mode != "native":
            oracle, _reference = oracle_identity(args.session, args.codec, source, rule,
                                                 session["session_id"], record["input_sha256"])
        torch.backends.cuda.matmul.allow_tf32 = False
        with reference_policy("cublaslt-no-splitk"):
            gc.collect()
            torch.cuda.empty_cache()
            run = open_training_model(folder, mode=args.mode, device="cuda", library=args.library,
                                      retain_derivatives=args.kind == "correctness",
                                      seed=rule["adapter_seed"], rank=rule["adapter_rank"],
                                      lr=rule["learning_rate"],
                                      native_rows_max=rule["native_rows_max"],
                                      native_rows_from=rule.get("native_rows_from"),
                                      weight_budget_bytes=rule["weight_budget_bytes"],
                                      gemm_call_budget_bytes=rule["gemm_call_budget_bytes"],
                                      activation_budget_bytes=rule["activation_budget_bytes"])
            peers = gpu_processes()
            if peers is None or len(peers) != 1:
                raise RuntimeError("the trained cell must be the sole exposed GPU workload")
            path = folder / bundle["layers"][0]["file"]
            with ArmWatch(arm=args.mode, round=0, run=0, model_path=str(path),
                          disk_device=selector, disk_reads=disk_counter,
                          other_disk_reads=other_counter) as monitor:
                if args.kind == "correctness":
                    record.update(correctness(run, ids, args.out, bundle, rule, oracle))
                else:
                    if args.mode not in rule["timing_pair"] or oracle is None:
                        raise ValueError("only the two declared streamed timing arms are allowed")
                    record.update(timed_or_profile(run, ids, args.out, rule, oracle, args.kind))
                record["dispatch"] = run.check_dispatch()
            monitor.record["counter_scope"] = mapping["scope"]
            monitor.record["counter_source"] = mapping.get("counter_source", "cgroup-v2")
            if monitor.record["outcome"] != "ok":
                raise RuntimeError("F1 unknown/void: stop without a replacement")
            after = gpu_processes()
            if after is None or {row["pid"] for row in after} != {row["pid"] for row in peers}:
                raise RuntimeError("GPU peer set changed")
            if fingerprints() != before_files or frozen_source() != source:
                raise ValueError("source/data identity changed during the training cell")
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            record["peak_process_rss_bytes"] = peak_rss
            record["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001 - every failed cell must be retained
        if isinstance(exc, NumericalFailureError):
            record["status"] = "fail"
        record["traceback"] = traceback.format_exc()
    finally:
        if monitor is not None and monitor.record is not None:
            record["F1"] = monitor.record
        if run is not None:
            try:
                run.close()
            except BaseException:  # noqa: BLE001 - record cleanup failure with the cell
                record["status"] = "no_verdict"
                record["cleanup_traceback"] = traceback.format_exc()
        save(args.out / "result.json", record)
    return 0 if record["status"] == "ok" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--out", type=Path, required=True)
    for name in ("series", "child"):
        entry = sub.add_parser(name)
        for flag in ("prepared", "out", "library"):
            entry.add_argument("--" + flag, type=Path, required=True)
        if name == "child":
            entry.add_argument("--session", type=Path, required=True)
            entry.add_argument("--oracle", type=Path)
            entry.add_argument("--mode", choices=MODES, required=True)
            entry.add_argument("--codec", choices=CODECS, required=True)
            entry.add_argument("--kind", choices=("correctness", "timing", "profile"),
                               required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        print(json.dumps(export_packet(args.out), indent=2))
        return 0
    rule = read_rule()
    if args.command == "child":
        return child(args, rule)
    from d3_training_series import parent

    return parent(args, rule)


if __name__ == "__main__":
    raise SystemExit(main())
