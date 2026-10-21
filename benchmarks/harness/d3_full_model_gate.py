"""Prospective D3 whole-model forward correctness gate; never a training claim.

Prepare immutable bundles, then freeze host/input/source pins in a clean commit.
Each formal cell is a fresh process; one session and zero replacement attempts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

from d3_full_model import (
    contained,
    file_hash,
    open_resident_model,
    open_streamed_model,
    prepare_bundle,
    tensor_hash,
    validate_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmarks/protocols/d3-full-model/protocol.json"
CODECS = ("dense", "nf4", "nf4-dq")
MODES = ("native", "sequential", "grouped-resident", "grouped")
REQUIRED_SOURCES = (
    "benchmarks/harness/d3_full_model.py",
    "benchmarks/harness/d3_full_model_gate.py",
    "benchmarks/harness/d3_grouped_restore_check.py",
    "benchmarks/harness/d3_cublas_grouped_probe.py",
    "benchmarks/harness/d3_f1_mapping_probe.py",
    "benchmarks/harness/d3_f1_controls.py",
    "benchmarks/harness/arm_validity.py",
    "benchmarks/harness/b1_expert_forward_gate.py",
    "benchmarks/protocols/b1.2-expert-forward/protocol.json",
    "benchmarks/protocols/d3-full-model/setup-evidence.zip",
    "src/soup_cli/utils/expert_group.py",
    "src/soup_cli/utils/expert_group_decode.py",
    "src/soup_cli/utils/expert_group_gemm.py",
    "src/soup_cli/utils/expert_group_compute.py",
    "src/soup_cli/utils/expert_shard.py",
    "src/soup_cli/utils/expert_stream.py",
    "src/soup_cli/utils/layer_shard.py",
    "src/soup_cli/utils/layer_stream_runtime.py",
    "src/soup_cli/utils/safetensors_reader.py",
)


class NumericalGateFailureError(RuntimeError):
    """A failed scientific control, distinct from unavailable execution evidence."""


def save(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def frozen(rule: dict) -> dict:
    if rule.get("execution_enabled") is not True:
        raise ValueError("whole-model execution disabled until new-host preparation and commit")
    pins = [rule.get("mapping_sha256"), rule.get("controls_sha256"),
            rule.get("cublas_library_sha256"), rule.get("setup_archive_sha256")]
    inputs = rule.get("bundle_sha256", {})
    if set(inputs) != set(CODECS):
        raise ValueError("freeze all three full-model bundles before formal cells")
    pins.extend(inputs.values())
    if any(not isinstance(value, str) or len(value) != 64
           or any(char not in "0123456789abcdef" for char in value) for value in pins):
        raise ValueError("whole-model data/instrument/library pins are incomplete")
    if (rule.get("attempts") != 1 or rule.get("replacements") != 0
            or not rule.get("gpu_uuid") or not rule.get("software")
            or set(rule.get("sources", [])) != set(REQUIRED_SOURCES)
            or rule.get("tokens") != 512 or rule.get("input_seed") != 1234
            or rule.get("batch") != 1 or tuple(rule.get("formats", ())) != CODECS
            or tuple(rule.get("modes", ())) != MODES
            or rule.get("weight_budget_bytes") != 64 * 2**20
            or rule.get("workspace_bytes") != 4 * 2**20
            or rule.get("activation_budget_bytes") != 256 * 2**20
            or rule.get("negative_control") != "zero_entire_active_expert_down_projection"
            or rule.get("native_olmoe_rows_max") != 128
            or rule.get("minimum_reported_vram_bytes", 0) < 15 * 2**30):
        raise ValueError("whole-model execution policy is incomplete")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)

    if git("status", "--porcelain", "--untracked-files=normal").strip():
        raise ValueError("commit whole-model sources/protocol in a clean checkout first")
    hashes = {}
    for name in [*rule["sources"], str(PROTOCOL.relative_to(ROOT)).replace("\\", "/")]:
        data = git("show", "HEAD:" + name)
        working = contained(ROOT, name).read_bytes()
        if Path(name).suffix in (".py", ".md", ".json"):
            working = working.replace(b"\r\n", b"\n")
        if working != data:
            raise ValueError("whole-model working source differs from its committed bytes")
        hashes[name] = hashlib.sha256(data).hexdigest()
    if hashes["benchmarks/protocols/d3-full-model/setup-evidence.zip"] != rule[
        "setup_archive_sha256"
    ]:
        raise ValueError("committed full-model setup evidence differs from its pinned hash")
    return {"git_head": git("rev-parse", "HEAD").decode().strip(), "git_blob_sha256": hashes}


def oracle_record(args, source: dict, rule: dict, input_hash: str) -> dict:
    """A later cell cannot substitute an unrelated tensor for its native oracle."""
    expected = args.out.parent / (args.codec + "-native")
    if os.path.realpath(args.oracle) != os.path.realpath(expected):
        raise ValueError("native oracle must come from the same declared session")
    record = json.loads((args.oracle / "result.json").read_text(encoding="utf-8"))
    if (record.get("status") != "ok" or record.get("mode") != "native"
            or record.get("codec") != args.codec or record.get("source") != source
            or record.get("F1", {}).get("outcome") != "ok"
            or record.get("protocol_sha256") != file_hash(PROTOCOL)
            or record.get("bundle_sha256") != rule["bundle_sha256"][args.codec]
            or record.get("input_sha256") != input_hash):
        raise ValueError("native oracle provenance/input/validity does not match this cell")
    return record


def check_traces(traces: list, *, mode: str, first_fetches: int | None) -> None:
    if len(traces) != 16:
        raise ValueError("every trained decoder layer must execute its experts")
    hits = 0
    for layer, rows in enumerate(traces):
        experts = [row["expert"] for row in rows]
        if not experts or experts != sorted(set(experts)):
            raise ValueError("each full-model layer needs unique ascending expert dispatch")
        for row in rows:
            if row["layer"] != layer or not 0 <= row["expert"] < 64 or row["token_routes"] < 1:
                raise ValueError("incorrect full-model layer/expert routing trace")
            if row["status"] != ("routed" if mode == "native" else "ok"):
                raise ValueError("full-model expert dispatch did not complete")
        hits += len(rows)
    if mode in ("sequential", "grouped") and first_fetches != hits:
        raise ValueError("streamed model must load every routed expert exactly once per forward")


def check_compute_coverage(mode: str, stats: dict | None) -> None:
    if mode in ("grouped", "grouped-resident") and (
        not stats or stats.get("grouped_matrices", 0) < 1
        or stats.get("native_olmoe_rows_max") != 128
    ):
        raise ValueError("positive full forward must exercise grouped GEMM under the frozen policy")


@contextmanager
def zero_active_down(run, target: int, bundle: dict):
    """Temporarily zero one actually routed down projection, then restore ownership.

    NF4 uses its exact zero code in both nibbles, including nested NF4. This
    changes the same whole matrix as the resident/dense control, without changing
    immutable files or allocating an additional CUDA weight tensor.
    """
    if run.pool is None:
        down = run.resident_experts[0].down_proj[target]
        saved = down.detach().cpu().clone()
        try:
            down.zero_()
            yield
        finally:
            down.copy_(saved)
        return

    from soup_cli.utils.expert_shard import expert_key
    from soup_cli.utils.layer_shard import NF4_CODE_KEY
    from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

    reader = run.readers[0]
    key = expert_key(target, "down")
    packed_zero = None
    if reader.quant_specs:
        codes = bundle["layers"][0]["codes"][NF4_CODE_KEY]
        zero_indices = [i for i, value in enumerate(codes) if value == 0.0]
        if len(codes) != 16 or len(zero_indices) != 1:
            raise ValueError("negative control requires a unique exact NF4 zero code")
        packed_zero = zero_indices[0] * 17
        key = quant_sidecar_keys(key, reader.quant_specs["down"])[0]
    original = reader.read_host

    def corrupt(expert, staging):
        data = original(expert, staging)
        if expert == target:
            if packed_zero is None:
                data[key].zero_()
            else:
                data[key].fill_(packed_zero)
        return data

    reader.read_host = corrupt
    try:
        yield
    finally:
        reader.read_host = original


def prepare(args, rule: dict) -> None:
    """Development preparation only; reuse pinned B1 source/codec, no downloads."""
    import torch
    from b1_expert_forward_gate import _checkpoint_identity, _layer_tensors
    from safetensors import safe_open
    from transformers import OlmoeConfig

    baseline = json.loads((ROOT / "benchmarks/protocols/b1.2-expert-forward/protocol.json")
                          .read_text(encoding="utf-8"))
    identity = _checkpoint_identity(args.model_dir, baseline)
    config = OlmoeConfig.from_pretrained(args.model_dir, local_files_only=True)
    if (config.hidden_size, config.intermediate_size, config.num_hidden_layers,
            config.num_experts, config.num_experts_per_tok) != (2048, 1024, 16, 64, 8):
        raise ValueError("the full-model gate is limited to the pinned trained OLMoE recipe")
    nonexpert = {}
    for path in sorted(args.model_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if ".mlp.experts." not in key:
                    if key in nonexpert:
                        raise ValueError("duplicate nonexpert checkpoint tensor")
                    nonexpert[key] = handle.get_tensor(key).to(torch.bfloat16).clone()

    def weights():
        for layer in range(config.num_hidden_layers):
            tensors = _layer_tensors(args.model_dir, layer)

            def pairs(tensors=tensors):
                for expert in range(config.num_experts):
                    stem = f"mlp.experts.{expert}."
                    yield (torch.cat((tensors[stem + "gate_proj.weight"],
                                      tensors[stem + "up_proj.weight"])).to(torch.bfloat16),
                           tensors[stem + "down_proj.weight"].to(torch.bfloat16))

            yield layer, pairs()

    args.prepared.mkdir(parents=True, exist_ok=True)
    prepare_bundle(config, nonexpert, weights(), args.prepared / args.codec,
                   codec=args.codec, quant_device=args.device,
                   provenance={"checkpoint_sha256": identity,
                               "model_revision": baseline["model_revision"],
                               "status": "development preparation; no formal verdict"})


def prepared_identity(prepared: Path, codec: str, rule: dict) -> dict:
    folder = contained(prepared, codec)
    record = validate_bundle(folder)
    if (record["codec"] != codec
            or file_hash(folder / "bundle.json") != rule["bundle_sha256"][codec]):
        raise ValueError("prepared bundle differs from the committed full-model protocol")
    return record


def child(args, rule: dict) -> int:
    source = frozen(rule)
    if sys.platform != "linux":
        raise RuntimeError("the trained full-model gate requires Linux CUDA")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"status": "no_verdict", "source": source, "mode": args.mode,
              "codec": args.codec, "scope": "whole-model forward only", "d3_accepted": False,
              "protocol_sha256": file_hash(PROTOCOL), "formal_execution": True}
    run = monitor = None
    try:
        import resource

        import torch
        from arm_validity import ArmWatch, _query_gpu, gpu_processes
        from d3_cublas_grouped_probe import reference_policy
        from d3_f1_controls import verified_selector
        from safetensors.torch import load_file, save_file

        if gpu_processes() != []:
            raise RuntimeError("visible GPU peers exist or telemetry is unavailable before setup")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("native BF16 CUDA is required")
        capacity = torch.cuda.get_device_properties(0).total_memory
        if capacity < rule["minimum_reported_vram_bytes"]:
            raise RuntimeError("full resident oracle requires the declared 16 GB+ GPU class")
        software = {"torch": torch.__version__, "driver": _query_gpu("driver_version")[0]}
        software.update({name: importlib.metadata.version(name) for name in (
            "transformers", "bitsandbytes", "safetensors", "triton",
        )})
        if _query_gpu("uuid")[0] != rule["gpu_uuid"] or software != rule["software"]:
            raise ValueError("GPU/software differs from the committed whole-model protocol")
        if file_hash(args.library) != rule["cublas_library_sha256"]:
            raise ValueError("cuBLAS binary differs from the committed protocol")
        for name in ("mapping", "controls"):
            if file_hash(args.prepared / (name + ".json")) != rule[name + "_sha256"]:
                raise ValueError("whole-model instrument identity differs from the protocol")
        controls = json.loads((args.prepared / "controls.json").read_text())
        if controls.get("status") != "controls_passed" or [r["observed"] for r in controls[
            "controls"
        ]] != ["ok", "ok", "void"]:
            raise ValueError("real baseline/own/foreign instrument controls required")
        selector = verified_selector(json.loads((args.prepared / "mapping.json").read_text()))
        bundle = prepared_identity(args.prepared, args.codec, rule)
        paths = [args.prepared / args.codec / row["file"] for row in bundle["layers"]]
        if len({selector(str(path)) for path in paths}) != 1:
            raise ValueError("all prepared expert shards need one verified counter scope")
        config = json.loads((args.prepared / args.codec / "config.json").read_text())
        if (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"],
                config["num_experts"], config["num_experts_per_tok"]) != (2048, 1024, 16, 64, 8):
            raise ValueError("a tiny fixture cannot be reported as the trained full model")
        ids = torch.randint(2, config["vocab_size"], (1, rule["tokens"]),
                            generator=torch.Generator().manual_seed(rule["input_seed"])).cuda()
        result.update(software=software, gpu_uuid=rule["gpu_uuid"], vram_bytes=capacity,
                      input_sha256=tensor_hash(ids),
                      bundle_sha256=rule["bundle_sha256"][args.codec])
        oracle = (oracle_record(args, source, rule, result["input_sha256"])
                  if args.mode != "native" else None)
        torch.backends.cuda.matmul.allow_tf32 = False
        with reference_policy("cublaslt-no-splitk"), torch.no_grad():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if args.mode in ("native", "grouped-resident"):
                run = open_resident_model(
                    args.prepared / args.codec, device="cuda", library=args.library,
                    backend="native" if args.mode == "native" else "grouped-compute",
                    native_olmoe_rows_max=rule["native_olmoe_rows_max"],
                )
            else:
                run = open_streamed_model(
                    args.prepared / args.codec, device="cuda", library=args.library,
                    backend="sequential" if args.mode == "sequential" else "grouped-compute",
                    native_olmoe_rows_max=rule["native_olmoe_rows_max"],
                )
            peers = gpu_processes()
            if peers is None or len(peers) != 1:
                raise RuntimeError("the full-model process is not the only visible GPU workload")
            result["visible_workload"] = peers
            with ArmWatch(arm=args.mode, round=0, run=0, model_path=str(paths[0]),
                          disk_device=selector) as monitor:
                observed = run.forward(ids).cpu()
                result["first_forward_gemm"] = run.gemm.stats() if run.gemm is not None else None
                check_compute_coverage(args.mode, result["first_forward_gemm"])
                traces = run.traces()
                check_traces(traces, mode=args.mode,
                             first_fetches=run.pool.fetches if run.pool is not None else None)
                if not torch.isfinite(observed).all():
                    raise NumericalGateFailureError("full-model logits are nonfinite")
                oracle_file = args.oracle / "logits.safetensors" if args.oracle else None
                expected = observed if args.mode == "native" else load_file(str(oracle_file))[
                    "logits"
                ]
                if oracle is not None and tensor_hash(expected) != oracle["expected_sha256"]:
                    raise ValueError("saved native oracle tensor differs from its retained result")
                if not torch.equal(observed, expected) or not torch.equal(
                    run.forward(ids).cpu(), expected,
                ):
                    raise NumericalGateFailureError("full-model exact equality / A-A gate failed")
                target = traces[0][0]["expert"]
                result["negative_control"] = rule["negative_control"]
                result["control_target"] = {"layer": 0, "expert": target}
                with zero_active_down(run, target, bundle):
                    negative = run.forward(ids).cpu()
                if torch.equal(negative, expected):
                    raise NumericalGateFailureError("full-model active-weight negative ineffective")
                if not torch.equal(run.forward(ids).cpu(), expected):
                    raise NumericalGateFailureError("full-model restored positive control failed")
            if monitor.record["outcome"] != "ok":
                raise RuntimeError("F1 unknown/void: preserve the cell without replacement")
            after = gpu_processes()
            if after is None or {r["pid"] for r in after} != {r["pid"] for r in peers}:
                raise RuntimeError("visible GPU workload changed during the full-model cell")
            prepared_identity(args.prepared, args.codec, rule)
            result.update(expected_sha256=tensor_hash(expected),
                          negative_sha256=tensor_hash(negative), traces=traces,
                          peak_allocated=torch.cuda.max_memory_allocated(),
                          peak_reserved=torch.cuda.max_memory_reserved(),
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          pool=run.pool.stats() if run.pool is not None else None)
            if run.gemm is not None:
                result["gemm"] = run.gemm.stats()
            if args.mode == "native":
                save_file({"logits": expected.contiguous()}, str(args.out / "logits.safetensors"))
            result["status"] = "ok"
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, NumericalGateFailureError):
            result["status"] = "fail"
        result["traceback"] = traceback.format_exc()
    finally:
        if monitor is not None and monitor.record is not None:
            result["F1"] = monitor.record
        if run is not None:
            try:
                run.close()
            except BaseException:
                result["status"] = "no_verdict"
                result["cleanup_traceback"] = traceback.format_exc()
        save(args.out / "result.json", result)
    return 0 if result["status"] == "ok" else 1


def parent(args, rule: dict) -> int:
    source = frozen(rule)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.prepared / f"full-model-attempt-{source['git_head']}.json").open("x") as handle:
        json.dump({"source": source, "session": str(args.out)}, handle)
    session = {"status": "running", "source": source, "cells": [], "d3_accepted": False}
    try:
        for codec in CODECS:
            oracle = args.out / (codec + "-native")
            for mode in MODES:
                folder = args.out / (codec + "-" + mode)
                command = [sys.executable, str(Path(__file__)), mode, "--prepared",
                           str(args.prepared), "--out", str(folder), "--library", str(args.library),
                           "--codec", codec]
                if mode != "native":
                    command += ["--oracle", str(oracle)]
                with folder.with_suffix(".log").open("x") as log:
                    proc = subprocess.run(command, cwd=ROOT, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=1800)
                if proc.returncode:
                    raise RuntimeError(f"full-model cell failed: {folder}; zero replacements")
                row = json.loads((folder / "result.json").read_text())
                session["cells"].append(row)
                if (row["status"] != "ok" or row["source"] != source
                        or row.get("F1", {}).get("outcome") != "ok"
                        or row.get("mode") != mode or row.get("codec") != codec
                        or row.get("protocol_sha256") != file_hash(PROTOCOL)
                        or row.get("formal_execution") is not True
                        or row.get("bundle_sha256") != rule["bundle_sha256"][codec]
                        or row.get("negative_sha256") == row.get("expected_sha256")):
                    raise RuntimeError("whole-model cell/source invalid")
                if mode != "native":
                    check_compute_coverage(mode, row.get("first_forward_gemm"))
                    native = json.loads((oracle / "result.json").read_text())
                    if any(row[key] != native[key] for key in (
                        "input_sha256", "expected_sha256",
                    )):
                        raise RuntimeError("whole-model cell differs from native oracle identity")
                save(args.out / "session.json", session)
        session["status"] = "full_forward_gate_complete_review_required"
    except (Exception, KeyboardInterrupt):
        session["status"] = "incomplete"
        session["traceback"] = traceback.format_exc()
    finally:
        save(args.out / "session.json", session)
    return 0 if session["status"] == "full_forward_gate_complete_review_required" else 1


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("prepare", "run", *MODES))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codec", choices=CODECS, default="dense")
    args = parser.parse_args(argv)
    rule = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    try:
        if args.mode == "prepare":
            if args.model_dir is None:
                parser.error("prepare requires --model-dir; downloads are not performed")
            prepare(args, rule)
            return 0
        if args.out is None or args.library is None:
            parser.error("formal modes require --out and --library")
        if args.mode not in ("run", "native") and args.oracle is None:
            parser.error("non-native cells require --oracle")
        return parent(args, rule) if args.mode == "run" else child(args, rule)
    except (Exception, KeyboardInterrupt):
        Console().print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
