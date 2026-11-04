"""Prospective D3 layer-performance runner. Formal modes require a clean commit.

Smoke does correctness/negative/F1 controls ONLY and records no timings.
Timed/profile children use the committed policy and immutable prepared inputs.
No downloads, source commits, trainer integration or end-to-end claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmarks/protocols/d3-grouped-experts/performance-protocol.json"
CODECS = ("dense", "nf4", "nf4-dq")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    target = os.path.realpath(os.path.join(root, relative))
    if os.path.isabs(relative) or os.path.commonpath([os.path.realpath(root), target]) != (
        os.path.realpath(root)
    ):
        raise ValueError("prepared input escapes its named directory")
    return Path(target)


def frozen_source(protocol: dict) -> dict:
    if protocol.get("execution_enabled") is not True:
        raise ValueError("formal execution is disabled until protocol preparation is complete")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)

    if git("status", "--porcelain", "--untracked-files=normal").strip():
        raise ValueError("commit reviewed source/protocol in a clean checkout before formal arms")
    pins = [protocol.get("mapping_sha256"), protocol.get("controls_sha256"),
            protocol.get("cublas_library_sha256"), protocol.get("setup_archive_sha256")]
    recipes = protocol.get("recipe_sha256", {})
    if set(recipes) != set(CODECS):
        raise ValueError("all three input recipes must be pinned before formal arms")
    pins += list(recipes.values())
    if any(not isinstance(value, str) or len(value) != 64
           or any(char not in "0123456789abcdef" for char in value) for value in pins):
        raise ValueError("incomplete input/instrument/library pins; do not start formal arms")
    if (protocol.get("warmup") != 5 or protocol.get("samples") != 20
            or protocol.get("pilot_blocks") != 5 or protocol.get("replacements") != 0
            or not 5 <= protocol.get("max_blocks", 0) <= 10
            or not isinstance(protocol.get("gpu_uuid"), str)
            or not protocol["gpu_uuid"].startswith("GPU-")
            or not protocol.get("driver_version") or not protocol.get("sources")):
        raise ValueError("incomplete or unsupported prospective execution policy")
    names = protocol["sources"] + [str(PROTOCOL.relative_to(ROOT)).replace("\\", "/")]
    values = {}
    for name in names:
        blob = git("show", f"HEAD:{name}")
        path = contained(ROOT, name)
        working = path.read_bytes()
        if path.suffix in (".py", ".json", ".md"):
            working = working.replace(b"\r\n", b"\n")
        if working != blob:
            raise ValueError(f"working source differs from its committed LF bytes: {name}")
        values[name] = hashlib.sha256(blob).hexdigest()
    if values.get(protocol.get("setup_archive")) != protocol["setup_archive_sha256"]:
        raise ValueError("committed setup evidence does not match the protocol's archive hash")
    return {"git_head": git("rev-parse", "HEAD").decode().strip(), "git_blob_sha256": values}


def validate_prepared(prepared: Path, codec: str, protocol: dict, *, formal: bool) -> tuple:
    if codec not in CODECS:
        raise ValueError("unknown codec")
    folder = contained(prepared, f"layer-07-{codec}")
    recipe_path = folder / "recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if (recipe["codec"] != codec or recipe["layer"] != 7 or recipe["dtype"] != "bfloat16"
            or recipe["reference_math"] != "cublaslt-no-splitk"):
        raise ValueError("prepared recipe is outside the declared paired comparison")
    for name, key in (("experts.safetensors", "weights_sha256"),
                      ("inputs.safetensors", "inputs_sha256")):
        if sha256(folder / name) != recipe[key]:
            raise ValueError(f"prepared file hash differs: {name}")
    if formal and sha256(recipe_path) != protocol["recipe_sha256"][codec]:
        raise ValueError("prepared recipe differs from the prospectively committed protocol")
    return folder, recipe


def _tensor_hash(value):
    import torch

    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def child(args, protocol: dict) -> int:
    formal = args.mode != "smoke"
    source = frozen_source(protocol) if formal else {"scope": "uncommitted development smoke"}
    if not formal:
        source["working_file_sha256"] = {name: sha256(contained(ROOT, name))
                                         for name in protocol["sources"]}
    if sys.platform != "linux":
        raise RuntimeError("the performance runner requires Linux CUDA")
    import resource

    import torch
    from arm_validity import ArmWatch, _query_gpu, gpu_processes
    from d3_cublas_grouped_probe import reference_policy
    from d3_f1_controls import verified_selector
    from safetensors.torch import load_file
    from transformers.activations import ACT2FN

    from soup_cli.utils.expert_group import GroupedExpertPool
    from soup_cli.utils.expert_group_compute import grouped_compute_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from soup_cli.utils.expert_shard import ExpertShardReader, expert_key
    from soup_cli.utils.expert_stream import ExpertBufferPool, streamed_experts_class
    from soup_cli.utils.layer_shard import NF4WeightSpec
    from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

    args.out.mkdir(parents=True, exist_ok=False)
    result = {"status": "no_verdict", "formal_measurement": formal, "d3_accepted": False,
              "arm": args.arm, "codec": args.codec, "mode": args.mode,
              "source": source, "samples": [], "protocol_sha256": sha256(PROTOCOL)}
    reader = pool = gemm = monitor = None
    try:
        folder, recipe = validate_prepared(args.prepared, args.codec, protocol, formal=formal)
        mapping = json.loads((args.prepared / "mapping.json").read_text())
        controls = json.loads((args.prepared / "controls.json").read_text())
        if controls["status"] != "controls_passed" or [row["observed"] for row in controls[
            "controls"
        ]] != ["ok", "ok", "void"]:
            raise ValueError("real F1 baseline/own/foreign controls are required")
        if formal:
            if (sha256(args.prepared / "mapping.json") != protocol["mapping_sha256"]
                    or sha256(args.prepared / "controls.json") != protocol["controls_sha256"]):
                raise ValueError("instrument receipts differ from the committed protocol")
        selector = verified_selector(mapping)
        selector(str(folder / "experts.safetensors"))
        peers = gpu_processes()
        if peers is None or peers:
            raise RuntimeError("GPU peer telemetry unavailable or a peer exists before CUDA setup")
        result["visible_gpu_peers_before_setup"] = peers
        if (not torch.cuda.is_available()
                or not torch.cuda.is_bf16_supported(including_emulation=False)):
            raise RuntimeError("native BF16 CUDA required")
        gpu_uuid = _query_gpu("uuid")[0]
        if formal and (gpu_uuid != protocol["gpu_uuid"] or torch.__version__ != protocol["torch"]):
            raise ValueError("GPU/software differs from the committed protocol")
        packages = {name: importlib.metadata.version(name) for name in protocol["packages"]}
        result["packages"] = packages
        result["driver_version"] = _query_gpu("driver_version")[0]
        if formal and (packages != protocol["packages"]
                       or result["driver_version"] != protocol["driver_version"]):
            raise ValueError("package/driver versions differ from the committed protocol")
        library = args.library
        if formal and sha256(library) != protocol["cublas_library_sha256"]:
            raise ValueError("cuBLAS binary differs from the committed protocol")
        result.update(gpu_uuid=gpu_uuid, torch=torch.__version__,
                      recipe_sha256=sha256(folder / "recipe.json"))
        cpu = load_file(str(folder / "inputs.safetensors"))
        expected = cpu["expected"]
        inputs = [cpu[name].to("cuda") for name in ("hidden", "index", "weights")]
        if inputs[0].shape != (512, 2048) or inputs[1].shape != (512, 8):
            raise ValueError("the formal recipe requires 512 tokens and the pinned OLMoE geometry")
        specs = {key: NF4WeightSpec.from_json(value)
                 for key, value in recipe["quant_specs"].items()}
        reader = ExpertShardReader(str(folder / "experts.safetensors"), count=recipe["count"],
                                   shapes={k: tuple(v) for k, v in recipe["shapes"].items()},
                                   quant_specs=specs, dtype=recipe["dtype"], direct=False)
        codes = {key: torch.tensor(value, dtype=torch.float32)
                 for key, value in recipe["codes"].items()}
        torch.backends.cuda.matmul.allow_tf32 = False
        with reference_policy("cublaslt-no-splitk") as policy, torch.no_grad():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result["allocated_before_pool"] = torch.cuda.memory_allocated()
            if args.arm == "A":
                pool = ExpertBufferPool(reader, device="cuda", dtype="bfloat16",
                                        weight_budget_bytes=64 * 2**20, codes=codes)
                module = streamed_experts_class()(reader, pool, ACT2FN[recipe["activation"]], 7)
            else:
                pool = GroupedExpertPool(reader, device="cuda", dtype="bfloat16", group_width=4,
                                         weight_budget_bytes=64 * 2**20, codes=codes)
                gemm = GroupedGemm(library, call_budget_bytes=64 * 2**20)
                module = grouped_compute_experts_class()(
                    reader, pool, ACT2FN[recipe["activation"]], 7, gemm=gemm,
                    activation_budget_bytes=256 * 2**20,
                )

            def exact():
                value = module(*inputs)
                if not torch.equal(value.cpu(), expected) or not torch.isfinite(value).all():
                    raise RuntimeError("exact output gate failed")
                return value

            exact()
            target = int(inputs[1].flatten()[0])
            original = reader.read_host

            def corrupt(expert, staging):
                data = original(expert, staging)
                if expert == target:
                    key = expert_key(expert, "down")
                    if "down" in specs:
                        key = quant_sidecar_keys(key, specs["down"])[0]
                        data[key].view(-1)[:1].bitwise_xor_(16)
                    else:
                        data[key].view(-1)[:1].add_(0.125)
                return data

            reader.read_host = corrupt
            try:
                negative = module(*inputs).cpu()
                if torch.equal(negative, expected):
                    raise RuntimeError("active-weight negative control was ineffective")
                result["negative_sha256"] = _tensor_hash(negative)
            finally:
                reader.read_host = original
            exact()
            result["reference_policy"] = policy
            for _ in range(protocol["warmup"] if formal else 1):
                exact()
            own_peers = gpu_processes()
            if own_peers is None or len(own_peers) != 1:
                raise RuntimeError("cannot isolate the single visible CUDA workload")
            result["visible_workload"] = own_peers
            with ArmWatch(arm=args.arm, round=0, run=0, model_path=reader.path,
                          disk_device=selector) as monitor:
                if args.mode == "timed":
                    for _ in range(protocol["samples"]):
                        torch.cuda.synchronize()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                            enable_timing=True,
                        )
                        start.record()
                        began = time.perf_counter()
                        output = module(*inputs)
                        end.record()
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - began
                        if not torch.equal(output.cpu(), expected):
                            raise RuntimeError("timed output differs from the exact oracle")
                        result["samples"].append({"wall_seconds": wall,
                                                  "cuda_dispatch_span_ms": start.elapsed_time(end),
                                                  "output_sha256": _tensor_hash(output)})
                        (args.out / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
                elif args.mode == "profile":
                    with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
                    ]) as p:
                        exact()
                        torch.cuda.synchronize()
                    p.export_chrome_trace(str(args.out / "trace.json"))
                    events = json.loads((args.out / "trace.json").read_text())["traceEvents"]
                    kernels = [row for row in events if row.get("cat") == "kernel"
                               and row.get("ph") == "X"]
                    if not kernels:
                        raise RuntimeError("no actual CUDA kernels in profiling trace")
                    result["cuda_kernel_events"] = len(kernels)
                else:
                    exact()
                    exact()
            result["F1"] = monitor.record
            if result["F1"]["outcome"] != "ok":
                raise RuntimeError("F1 is unknown/void; no formal timing verdict")
            after_peers = gpu_processes()
            if after_peers is None or {row["pid"] for row in after_peers} != {
                row["pid"] for row in own_peers
            }:
                raise RuntimeError("visible GPU workload set changed during the arm")
            validate_prepared(args.prepared, args.codec, protocol, formal=formal)
            result.update(expected_sha256=_tensor_hash(expected), pool=pool.stats(),
                          compute=gemm.stats() if gemm else None,
                          peak_allocated=torch.cuda.max_memory_allocated(),
                          peak_reserved=torch.cuda.max_memory_reserved(),
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            result["median_seconds"] = statistics.median(
                row["wall_seconds"] for row in result["samples"]
            ) if result["samples"] else None
            result["status"] = "ok"
    except (Exception, KeyboardInterrupt):
        result["traceback"] = traceback.format_exc()
        if monitor is not None and monitor.record is not None:
            result["F1"] = monitor.record
    finally:
        try:
            if reader is not None:
                reader.close()
            if gemm is not None:
                gemm.close()
            if pool is not None:
                pool.close()
        except BaseException:
            result["status"] = "no_verdict"
            result["cleanup_traceback"] = traceback.format_exc()
        (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "ok" else 1


def analyze(blocks: list, sample_count: int) -> dict:
    """Reject incomplete/invalid blocks; all valid samples and ratios count."""
    if not blocks:
        raise ValueError("no ABBA blocks to analyze")
    ratios = []
    identity = None
    for block in blocks:
        if len(block) != 4 or [row["arm"] for row in block] != ["A", "B", "B", "A"]:
            raise ValueError("incomplete or reordered ABBA block")
        for row in block:
            if (row["status"] != "ok" or row["F1"]["outcome"] != "ok"
                    or len(row["samples"]) != sample_count or row["mode"] != "timed"
                    or row["formal_measurement"] is not True):
                raise ValueError("invalid/incomplete arm cannot enter the comparison")
            current = (row["source"], row["protocol_sha256"], row["recipe_sha256"], row["codec"],
                       row["expected_sha256"])
            if identity is None:
                identity = current
            elif current != identity:
                raise ValueError("arms mix source/protocol/data/reference identities")
            values = [sample["wall_seconds"] for sample in row["samples"]]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("nonpositive/nonfinite timing")
            if statistics.median(values) != row["median_seconds"]:
                raise ValueError("arm median does not reproduce")
            if any(sample["output_sha256"] != row["expected_sha256"] for sample in row["samples"]):
                raise ValueError("sample output differs from the exact reference hash")
        a = (block[0]["median_seconds"] + block[3]["median_seconds"]) / 2
        b = (block[1]["median_seconds"] + block[2]["median_seconds"]) / 2
        ratios.append(b / a)
    return {"blocks": len(blocks), "block_ratios_B_over_A": ratios,
            "mean_block_ratio": statistics.fmean(ratios), "d3_accepted": False}


def parent(args, protocol: dict) -> int:
    from arm_validity import Run, blocks_to_add, repeats

    source = frozen_source(protocol)
    args.out.mkdir(parents=True, exist_ok=False)
    # One attempt per actual source commit, across all output folder names.
    attempt = args.prepared / f"attempt-{source['git_head']}.json"
    with attempt.open("x") as handle:
        json.dump({"source": source, "session": os.path.realpath(args.out)}, handle)
    session = {"status": "running", "source": source, "configurations": {},
               "protocol_sha256": sha256(PROTOCOL), "d3_accepted": False}
    try:
        for codec in CODECS:
            blocks = []
            session["configurations"][codec] = {"blocks": blocks}
            while True:
                runs = [[Run(row["median_seconds"], row["F1"]["outcome"]) for row in block]
                        for block in blocks]
                additions = blocks_to_add(runs)
                if additions == 0:
                    break
                if len(blocks) + additions > protocol["max_blocks"]:
                    raise RuntimeError("ADR 0003 requests more than the fixed attempt budget")
                for _ in range(additions):
                    block = []
                    blocks.append(block)
                    for position, arm in enumerate(("A", "B", "B", "A")):
                        folder = args.out / f"{codec}-block-{len(blocks):02d}-{position}-{arm}"
                        command = [sys.executable, str(Path(__file__)), "timed", "--prepared",
                                   str(args.prepared), "--library", str(args.library),
                                   "--codec", codec,
                                   "--arm", arm, "--out", str(folder)]
                        with folder.with_suffix(".log").open("x") as log:
                            proc = subprocess.run(command, cwd=ROOT, stdout=log,
                                                  stderr=subprocess.STDOUT, timeout=600)
                        if proc.returncode:
                            raise RuntimeError(f"arm failed; retain and inspect {folder}")
                        row = json.loads((folder / "result.json").read_text())
                        block.append(row)
                        if row["status"] != "ok" or row["F1"]["outcome"] != "ok":
                            raise RuntimeError("invalid arm: zero replacements under this protocol")
                        if row["source"] != source:
                            raise RuntimeError("child source identity differs from its parent")
                        (args.out / "session.json").write_text(json.dumps(session, indent=2) + "\n")
            record = session["configurations"][codec]
            estimate = repeats(runs)
            record["repeat_calculation"] = {"sigma": estimate.sigma, "mean_a": estimate.mean_a,
                                            "blocks_needed": estimate.blocks_needed}
            record["analysis"] = analyze(blocks, protocol["samples"])
            record["profiles"] = {}
            for arm in ("A", "B"):
                folder = args.out / f"{codec}-profile-{arm}"
                command = [sys.executable, str(Path(__file__)), "profile", "--prepared",
                           str(args.prepared), "--library", str(args.library), "--codec", codec,
                           "--arm", arm, "--out", str(folder)]
                with folder.with_suffix(".log").open("x") as log:
                    proc = subprocess.run(command, cwd=ROOT, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=600)
                if proc.returncode:
                    raise RuntimeError("profile failed; preserve it without replacement")
                record["profiles"][arm] = json.loads((folder / "result.json").read_text())
                if (record["profiles"][arm]["status"] != "ok"
                        or record["profiles"][arm]["source"] != source):
                    raise RuntimeError("profile status/source does not match the frozen session")
        session["status"] = "layer_performance_complete"
    except (Exception, KeyboardInterrupt):
        session["status"] = "incomplete"
        session["traceback"] = traceback.format_exc()
    finally:
        (args.out / "session.json").write_text(json.dumps(session, indent=2) + "\n")
    return 0 if session["status"] == "layer_performance_complete" else 1


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("smoke", "timed", "profile", "run"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--codec", choices=CODECS, default="dense")
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    args = parser.parse_args(argv)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    try:
        code = parent(args, protocol) if args.mode == "run" else child(args, protocol)
    except (Exception, KeyboardInterrupt):
        Console().print(traceback.format_exc())
        return 1
    Console().print(f"exit={code}: {args.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
