"""Fresh-process public-trainer series; requires prospective committed sources/evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from benchmarks.harness.d3_trainer_gate import (
    NumericalFailureError,
    assemble,
    correctness,
    public_workspace,
    require_exact,
    save,
    software,
    tensor_record,
)
from benchmarks.harness.d3_trainer_protocol import (
    CODECS,
    ENGINE_FILES,
    PROTOCOL,
    ROOT,
    TRAINING_MODES,
    cell_plan,
    contained,
    file_hash,
    launch_blockers,
    read_rule,
    repeat_target,
    validate_rule,
)


def oracle_path(session: Path, codec: str) -> Path:
    if codec not in CODECS:
        raise ValueError("unsupported oracle codec")
    return session / f"correctness-{codec}-ordinary-native-resident" / "result.json"


def frozen_source() -> dict:
    rule = read_rule()
    validate_rule(rule)
    if rule["execution_enabled"] is not True:
        raise ValueError("formal execution disabled; prepare and commit the new trainer rule")
    blockers = launch_blockers(rule, ROOT, check_git=False)
    if blockers:
        raise ValueError("unready frozen preparation: " + "; ".join(blockers))
    receipt = json.loads((ROOT / "FROZEN-D3-TRAINER-SOURCE.json").read_text())
    files = receipt.get("git_blob_sha256", {})
    required = {
        PROTOCOL,
        *ENGINE_FILES,
        "benchmarks/harness/d3_trainer_protocol.py",
        "src/soup_cli/utils/expert_training.py",
        *(row["path"] for row in rule["preparation"].values()),
    }
    if not required <= set(files):
        raise ValueError("incomplete actual committed-source receipt")
    for name, expected in files.items():
        if file_hash(contained(ROOT, name)) != expected:
            raise ValueError(f"executed source differs from committed blob: {name}")
    return receipt


def verify_prepared(path: Path, rule: dict) -> dict:
    pin = rule["preparation"]["shards"]
    if file_hash(path) != pin["sha256"]:
        raise ValueError("prepared manifest differs from frozen receipt")
    data = json.loads(path.read_text())
    if data["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise ValueError("guest restarted; rediscovery and a new frozen rule are required")
    if data["boot_id"] != rule["hardware"]["boot_id"]:
        raise ValueError("preparation boot differs from the frozen hardware rule")
    seen = {}
    for name, expected in data["files"].items():
        file = Path(name)
        info = file.stat()
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            expected["st_dev"],
            expected["inode"],
            expected["bytes"],
            expected["mtime_ns"],
        ):
            raise ValueError(f"prepared data identity changed: {name}")
        key = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if key not in seen:
            seen[key] = file_hash(file)
        if seen[key] != expected["sha256"]:
            raise ValueError(f"prepared data hash changed: {name}")
    return data


def gpu_peers() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def verify_host(rule: dict, prepared: dict) -> None:
    import torch

    versions = software()
    for key, value in versions.items():
        if value != rule["software"][key]:
            raise ValueError(f"runtime software mismatch: {key}")
    gpu = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True
        )
        .strip()
        .split(",")
    )
    if [item.strip() for item in gpu] != [
        rule["hardware"]["gpu_uuid"],
        rule["software"]["driver"],
    ] or file_hash(Path(prepared["library"])) != rule["software"]["cublas_sha256"]:
        raise ValueError("GPU/driver/cuBLAS identity mismatch")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("native BF16 is required")
    if torch.cuda.get_device_properties(0).total_memory < rule["hardware"]["minimum_vram_bytes"]:
        raise ValueError("insufficient device capacity for the frozen recipe")
    if set(gpu_peers()) - {os.getpid()}:
        raise ValueError("another exposed CUDA workload is running")


def performance(rule: dict, prepared: dict, out: Path, cell: dict, oracle: dict) -> dict:
    import torch
    from transformers import TrainerCallback

    wrapper = assemble(
        rule, prepared, out / "adapter", cell["codec"], cell["mode"], cell["training"]
    )
    # Compile every possible routed-row activation shape before timing. The
    # activation kernel specializes on element count, so one arbitrary training
    # warmup does not cover all later shapes. No pretrained forward is needed.
    with torch.no_grad():
        resources = wrapper._stream_runtime.expert_resources
        reader = resources.readers[0]
        for width in range(1, resources.pool.group_width + 1):
            with resources.pool.lease(reader, list(range(width))):
                pass
        if cell["mode"] == "triton":
            from soup_cli.utils.expert_group_activation import fused_swiglu

            for rows in range(1, rule["tokens"] + 1):
                matrix = torch.zeros((rows, 2048), device="cuda", dtype=torch.bfloat16)
                value = fused_swiglu(matrix, output_budget_bytes=rule["activation_budget_bytes"])
                del value, matrix
        torch.cuda.synchronize()
    expected = oracle["numerical"]["steps"]
    windows = []
    original_batches = wrapper.trainer.get_batch_samples

    def batches(*args, **kwargs):
        samples, count = original_batches(*args, **kwargs)
        windows.append({
            "samples": [
                {key: tensor_record(value) for key, value in row.items()
                 if isinstance(value, torch.Tensor)} for row in samples
            ],
            "num_items": int(count) if count is not None else None,
        })
        return samples, count

    wrapper.trainer.get_batch_samples = batches

    class Window(TrainerCallback):
        def __init__(self):
            self.rows = []
            self.profile = None
            self.frames = []

        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == rule["warmup_steps"]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self.stats_before = wrapper._stream_runtime.stats()
                if cell["kind"] == "profile":
                    self.profile = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    )
                    self.profile.__enter__()
            torch.cuda.synchronize()
            self.begin = time.perf_counter()
            self.event = torch.cuda.Event(enable_timing=True)
            self.event.record()
            self.index = state.global_step

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            stop = torch.cuda.Event(enable_timing=True)
            stop.record()
            torch.cuda.synchronize()
            self.before_seconds = time.perf_counter() - self.begin
            self.before_ms = self.event.elapsed_time(stop)
            self.gradients = {
                name: tensor_record(param.grad)
                for name, param in model.named_parameters()
                if param.requires_grad and param.grad is not None
            }
            require_exact(
                self.gradients,
                expected[self.index]["parameter_gradients"],
                "performance step gradients",
            )
            self.resume = time.perf_counter()
            self.resume_event = torch.cuda.Event(enable_timing=True)
            self.resume_event.record()

        def on_step_end(self, args, state, control, model=None, **kwargs):
            stop = torch.cuda.Event(enable_timing=True)
            stop.record()
            torch.cuda.synchronize()
            wall = self.before_seconds + time.perf_counter() - self.resume
            milliseconds = self.before_ms + self.resume_event.elapsed_time(stop)
            if self.index >= rule["warmup_steps"]:
                rows = rule["optimizer"]["gradient_windows"][self.index]
                self.rows.append(
                    {
                        "step": self.index,
                        "wall_seconds": wall,
                        "cuda_ms": milliseconds,
                        "input_tokens": rows * rule["tokens"],
                    }
                )
            if self.profile is not None:
                self.profile.__exit__(None, None, None)
                self.profile.export_chrome_trace(str(out / "cuda-profile.json"))
                self.profile = None
            parameters = {
                name: tensor_record(param)
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            require_exact(parameters, expected[self.index]["parameters"], "performance updates")
            self.frames.append({"parameter_gradients": self.gradients, "parameters": parameters})
            self.peaks = {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
            self.stats_after = wrapper._stream_runtime.stats()

        def on_train_end(self, args, state, control, model=None, **kwargs):
            model.eval()
            ids = torch.tensor([wrapper.trainer.train_dataset[0]["input_ids"]], device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                require_exact(
                    tensor_record(model(input_ids=ids).logits.float()),
                    oracle["numerical"]["evaluation"],
                    "performance final logits",
                )

    callback = Window()
    profile_routes = []
    profile_hooks = []
    if cell["kind"] == "profile":
        from soup_cli.utils.layer_stream_runtime import decoder_owner

        def route_hook(index):
            def record(_module, args):
                if callback.profile is not None:
                    ids = args[1].detach().cpu()
                    counts = torch.bincount(
                        ids.flatten(), minlength=wrapper.model.config.num_experts
                    )
                    profile_routes.append({"layer": index, "rows": ids.shape[0],
                                           "top_k": ids.shape[1], "counts": counts.tolist(),
                                           "grad_enabled": torch.is_grad_enabled()})
            return record

        for index, layer in enumerate(decoder_owner(wrapper.model).layers):
            profile_hooks.append(layer.mlp.experts.register_forward_pre_hook(route_hook(index)))
    wrapper.trainer.add_callback(callback)
    try:
        with public_workspace(prepared, out):
            result = wrapper.train()
        if result["total_steps"] != rule["optimizer_steps"] or len(callback.rows) != 3:
            raise ValueError("incomplete timed optimizer windows")
        require_exact(windows, oracle["numerical"]["windows"], "timed delivered inputs/labels")
        tokens = sum(row["input_tokens"] for row in callback.rows)
        if tokens != rule["timed_input_tokens"]:
            raise ValueError("timed token count differs from frozen partial-window rule")
        stats = callback.stats_after
        if cell["mode"] != "sequential" and (
            stats["expert_gemm"]["grouped_matrices"]
            - callback.stats_before["expert_gemm"]["grouped_matrices"] <= 0
        ):
            raise ValueError("no positive grouped dispatch")
        record = {
            "samples": callback.rows,
            "frames": callback.frames,
            "peaks": callback.peaks,
            "seconds_per_input_token": (sum(row["wall_seconds"] for row in callback.rows) / tokens),
            "runtime": stats,
            "runtime_before_timing": callback.stats_before,
            "windows": windows,
        }
        if cell["kind"] == "profile":
            trace = json.loads((out / "cuda-profile.json").read_text())
            kernels = [
                row
                for row in trace["traceEvents"]
                if row.get("cat") == "kernel" and row.get("ph") == "X"
            ]
            if not kernels:
                raise ValueError("no actual CUPTI CUDA kernel events")
            record["cuda_kernel_events"] = len(kernels)
            record["profile_sha256"] = file_hash(out / "cuda-profile.json")
            record["profile_routes"] = profile_routes
            record["kernel_names"] = [row["name"] for row in kernels]
            if cell["mode"] == "triton" and not any(
                "activate" in name for name in record["kernel_names"]
            ):
                raise ValueError("profile records no executed D3 Triton activation kernel")
        return record
    finally:
        if callback.profile is not None:
            callback.profile.__exit__(None, None, None)
        for hook in profile_hooks:
            hook.remove()
        wrapper._close_stream_runtime()


def child(args: Any) -> int:
    rule = read_rule()
    source = frozen_source()
    import resource
    session = json.loads((args.session / "session.json").read_text())
    names = {cell["name"]: cell for cell in cell_plan(rule, blocks=rule["max_blocks"])}
    if args.cell not in names or session["source"] != source:
        raise ValueError("unknown prospective cell or source")
    cell = names[args.cell]
    args.out.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "no_verdict",
        "cell": cell,
        "source": source,
        "protocol_sha256": file_hash(ROOT / PROTOCOL),
        "session_id": session["session_id"],
        "formal_execution": True,
        "pid": os.getpid(),
    }
    monitor = None
    try:
        prepared = verify_prepared(args.prepared, rule)
        verify_host(rule, prepared)
        import torch

        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        record["device_before"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.used,temperature.gpu,power.draw,clocks.sm",
             "--format=csv"], text=True, timeout=20,
        )
        # The unchanged historical helpers use flat imports internally.
        sys.path.insert(0, str(ROOT / "benchmarks/harness"))
        from arm_validity import ArmWatch
        from d3_f1_controls import verified_counters, verified_selector

        mapping_pin = rule["preparation"]["mapping"]
        mapping = json.loads(contained(ROOT, mapping_pin["path"]).read_text())
        selector = verified_selector(mapping)
        disk_counter, other_counter = verified_counters(mapping)
        for row in mapping["files"]:
            selector(row["path"])
        reference = None if args.oracle is None else json.loads(args.oracle.read_text())
        if reference is not None:
            if (
                reference["status"] != "ok"
                or reference["source"] != source
                or reference["F1"]["outcome"] != "ok"
                or reference["session_id"] != session["session_id"]
                or reference["protocol_sha256"] != record["protocol_sha256"]
                or reference["formal_execution"] is not True
                or reference["cell"]["name"] != oracle_path(args.session, cell["codec"]).parent.name
            ):
                raise ValueError("invalid ordinary native oracle")
            reference = reference["result"]
        with ArmWatch(
            arm=cell["mode"],
            round=0,
            run=0,
            model_path=mapping["files"][0]["path"],
            disk_device=selector,
            disk_reads=disk_counter,
            other_disk_reads=other_counter,
        ) as monitor:
            if cell["kind"] == "correctness":
                native = None if reference is None else reference["numerical"]
                record["result"] = correctness(
                    rule,
                    prepared,
                    args.out,
                    cell["codec"],
                    cell["mode"],
                    cell["training"],
                    native,
                )
            else:
                if reference is None:
                    raise ValueError("performance/profile requires ordinary native oracle")
                record["result"] = performance(rule, prepared, args.out, cell, reference)
        record["F1"] = monitor.record
        if monitor.record["outcome"] != "ok":
            raise ValueError("F1 unknown/void: no replacement")
        if set(gpu_peers()) - {os.getpid()}:
            raise ValueError("GPU peer set changed")
        verify_prepared(args.prepared, rule)
        require_exact(frozen_source(), source, "post-cell committed source")
        record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        record["process_peak_rss_scope"] = "whole child, including setup/correctness/reload"
        record["device_after"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.used,temperature.gpu,power.draw,clocks.sm",
             "--format=csv"], text=True, timeout=20,
        )
        record["status"] = "ok"
    except BaseException as exc:
        if isinstance(exc, NumericalFailureError):
            record["status"] = "fail"
        record["traceback"] = traceback.format_exc()
    finally:
        if monitor is not None and monitor.record is not None:
            record["F1"] = monitor.record
        save(args.out / "result.json", record)
    return 0 if record["status"] == "ok" else 1


def child_command(prepared: Path, session: Path, cell: dict) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "benchmarks.harness.d3_trainer_gate",
        "child",
        "--prepared",
        str(prepared),
        "--out",
        str(session / cell["name"]),
        "--session",
        str(session),
        "--cell",
        cell["name"],
    ]
    if cell["name"] != f"correctness-{cell['codec']}-ordinary-native-resident":
        command += ["--oracle", str(oracle_path(session, cell["codec"]))]
    return command


def parent(args: Any) -> int:
    import uuid

    rule = read_rule()
    source = frozen_source()
    prepared = verify_prepared(args.prepared, rule)
    verify_host(rule, prepared)
    marker_dir = args.prepared.parent / ".d3-public-trainer-sessions"
    marker_dir.mkdir(exist_ok=True)
    with (marker_dir / f"{source['git_head']}.json").open("x") as marker:
        json.dump({"output": str(args.out), "source": source}, marker)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    session = {
        "session_id": uuid.uuid4().hex,
        "status": "running",
        "source": source,
        "protocol_sha256": file_hash(ROOT / PROTOCOL),
        "cells": [],
        "d3_accepted": False,
        "formal_execution": True,
    }
    save(args.out / "protocol.json", rule)
    save(args.out / "session.json", session)

    def launch(cell: dict) -> dict:
        remaining = rule["session_timeout_seconds"] - (time.monotonic() - started)
        if remaining <= 0 or len(session["cells"]) >= rule["attempt_budget"]["maximum_total_cells"]:
            raise ValueError("finite session time/cell budget exhausted")
        command = child_command(args.prepared, args.out, cell)
        row = {**cell, "status": "started", "command": command}
        session["cells"].append(row)
        save(args.out / "session.json", session)
        try:
            with (args.out / (cell["name"] + ".log")).open("x") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    timeout=min(rule["cell_timeout_seconds"], remaining),
                    check=False,
                )
            row["returncode"] = process.returncode
            result_path = args.out / cell["name"] / "result.json"
            result = json.loads(result_path.read_text())
            row.update(status=result["status"], result_sha256=file_hash(result_path))
            if (
                process.returncode
                or result["status"] != "ok"
                or result["source"] != source
                or result["F1"]["outcome"] != "ok"
                or result["session_id"] != session["session_id"]
                or result["protocol_sha256"] != session["protocol_sha256"]
                or result["formal_execution"] is not True
                or result["cell"] != cell
            ):
                raise ValueError(f"cell failed/no verdict: {cell['name']}")
            return result
        except BaseException:
            row.update(status="failed_or_no_verdict", traceback=traceback.format_exc())
            raise
        finally:
            save(args.out / "session.json", session)

    try:
        pilot = cell_plan(rule)
        for cell in pilot:
            if cell["kind"] == "correctness":
                launch(cell)
        full = cell_plan(rule, blocks=rule["max_blocks"])
        session["repeat_calculations"] = {}
        for codec in CODECS:
            for training in TRAINING_MODES:
                blocks = []
                target = rule["pilot_blocks"]
                while len(blocks) < target:
                    index = len(blocks) + 1
                    block = []
                    for cell in full:
                        if (
                            cell["kind"] == "timing"
                            and cell["codec"] == codec
                            and cell["training"] == training
                            and cell["block"] == index
                        ):
                            result = launch(cell)
                            block.append(
                                {
                                    "mode": cell["mode"],
                                    "outcome": result["F1"]["outcome"],
                                    "seconds_per_input_token": result["result"][
                                        "seconds_per_input_token"
                                    ],
                                }
                            )
                    blocks.append(block)
                    if len(blocks) == target:
                        estimate = repeat_target(blocks)
                        session["repeat_calculations"][f"{codec}/{training}"] = estimate
                        save(args.out / "session.json", session)
                        if estimate["status"] == "no_verdict_precision_budget":
                            raise ValueError("fixed precision budget exhausted")
                        target = estimate["target"]
                for cell in pilot:
                    if (
                        cell["kind"] == "profile"
                        and cell["codec"] == codec
                        and cell["training"] == training
                    ):
                        launch(cell)
        session["status"] = "complete_pending_offline_analysis_and_review"
    except BaseException:
        session.update(status="incomplete", traceback=traceback.format_exc())
    finally:
        save(args.out / "session.json", session)
    return 0 if session["status"] == "complete_pending_offline_analysis_and_review" else 1
