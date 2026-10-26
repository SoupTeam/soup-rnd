"""Qualify opt-in grouped expert routes against the unchanged legacy constructor.

Both timing variants preserve the native ``two_read_cache`` arm and all cache
ceilings. Three full numerical captures precede eight fresh AB/BA workers.
This harness never rents hardware or downloads weights.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

GIB = 2**30
VARIANTS = ("legacy", "grouped")
ORDERS = (("legacy", "grouped"), ("grouped", "legacy")) * 2
ACCURACY_VARIANTS = ("ordinary_legacy", "legacy", "grouped")
NATIVE_ARMS = {"ordinary_legacy": "ordinary", "legacy": "two_read_cache",
               "grouped": "two_read_cache"}
FIXED = {"dtype": "bfloat16", "quant": "nf4", "seq_len": 512, "batch_size": 1,
         "accumulation": 2, "accuracy_updates": 2, "warmup_updates": 2,
         "timed_updates": 6, "rounds": 4, "lora_r": 4, "lora_alpha": 8,
         "host_cache_policy": "static", "host_cache_bytes": 8 * GIB,
         "device_cache_bytes": 2 * GIB, "device_cap_bytes": 8 * GIB}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".incoming")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_protocol(spec: dict[str, Any]) -> None:
    require(all(type(spec.get(key)) is type(value) and spec[key] == value
                for key, value in FIXED.items()), "Fixed two-variant recipe/ceilings changed")
    require(type(spec.get("max_host_bytes")) is int and
            28 * GIB <= spec["max_host_bytes"] <= 32 * GIB, "Native RAM bound changed")
    require(set(spec.get("lora_targets", [])) == {"q_proj", "kv_b_proj", "o_proj"},
            "Kanana attention-LoRA targets changed")


def _default_constructor_control(module: Any) -> None:
    constructor = getattr(module, "build_streamed_model", None)
    require(callable(constructor), "Required model constructor is absent; model must not start")
    flag = inspect.signature(constructor).parameters.get("grouped_expert_routes")
    require(flag is not None and flag.default is False,
            "Constructor must expose grouped_expert_routes=False without changing the default")


def project(root: Path, expected_commit: str) -> tuple[Any, Any, Any, Any, dict[str, str]]:
    root = Path(os.path.realpath(root))
    require(re.fullmatch("[0-9a-f]{40}", expected_commit), "Exact source commit is required")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(actual == expected_commit, "Executed Git HEAD changed")
    require(not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
    ).strip(), "Tracked source is dirty")
    sys.path[:0] = [str(root / "src"), str(root)]
    from benchmarks.harness import two_read_cache as native
    from benchmarks.harness import two_read_cache_common as common
    from benchmarks.harness import two_read_cache_worker as worker
    from soup_cli.utils import expert_stream, layer_stream_runtime

    require(os.path.normcase(os.path.realpath(native.ROOT)) == os.path.normcase(str(root)),
            "Imported a different benchmark source")
    paths = [Path(os.path.realpath(module.__file__))
             for module in (expert_stream, layer_stream_runtime)]
    require(all(os.path.normcase(os.path.commonpath((str(root), str(path))))
                == os.path.normcase(str(root)) for path in paths),
            "Imported a different expert/runtime implementation")
    _default_constructor_control(layer_stream_runtime)
    require(common.RTOL == .01 and common.ATOL == 1e-5,
            "Original numerical tolerances changed")
    identity = {"source_commit": actual, "expert_module_sha256": digest(paths[0]),
                "runtime_module_sha256": digest(paths[1]),
                "constructor_source_sha256": hashlib.sha256(inspect.getsource(
                    layer_stream_runtime.build_streamed_model).encode("utf-8")).hexdigest(),
                "wrapper_sha256": digest(Path(__file__))}
    return native, worker, common, layer_stream_runtime, identity


def expected_routes(enabled: bool) -> dict[str, bool]:
    return {str(index): enabled for index in range(1, 48)}


def built_route_evidence(model: Any, runtime: Any, enabled: bool) -> dict[str, Any]:
    from soup_cli.utils.expert_stream import streamed_experts_class

    cls = streamed_experts_class()
    modules = [module for module in model.modules() if isinstance(module, cls)]
    flags = {str(module.layer_idx): module.grouped_expert_routes for module in modules}
    require(len(modules) == 47 and len(flags) == 47
            and all(type(value) is bool for value in flags.values())
            and flags == expected_routes(enabled),
            "Actual expert modules do not match the requested grouped route flag")
    require(runtime.stats().get("expert_grouped_routes") == flags,
            "Runtime route evidence differs from actual expert modules")
    return {"checked_before_cache_admission": True, "expert_modules": 47,
            "actual_expert_grouped_routes": flags, "runtime_expert_grouped_routes": flags}


@contextlib.contextmanager
def dispatch(module: Any, variant: str) -> Iterator[dict[str, Any]]:
    """Inject only the opt-in constructor kwarg, before native cache admission."""
    require(variant in ACCURACY_VARIANTS, "Unknown dispatch variant")
    _default_constructor_control(module)
    original = module.build_streamed_model
    enabled = variant == "grouped"
    evidence = {"mode": "grouped_constructor" if enabled else "legacy_constructor",
                "grouped_expert_routes": enabled, "factory_calls": 0,
                "factory_restored": False, "build_proofs": []}

    @functools.wraps(original)
    def build(*args: Any, **kwargs: Any) -> Any:
        require("grouped_expert_routes" not in kwargs
                or kwargs["grouped_expert_routes"] is enabled,
                "Native worker requested a conflicting grouped route flag")
        kwargs["grouped_expert_routes"] = enabled
        model, runtime = original(*args, **kwargs)
        try:
            proof = built_route_evidence(model, runtime, enabled)
        except BaseException:
            runtime.close()
            raise
        evidence["factory_calls"] += 1
        evidence["build_proofs"].append(proof)
        return model, runtime

    module.build_streamed_model = build
    try:
        yield evidence
    finally:
        module.build_streamed_model = original
        evidence["factory_restored"] = True


def validate_dispatch_evidence(evidence: dict[str, Any], variant: str) -> None:
    enabled = variant == "grouped"
    require(evidence.get("mode") == ("grouped_constructor" if enabled else "legacy_constructor")
            and evidence.get("grouped_expert_routes") is enabled
            and type(evidence.get("factory_calls")) is int and evidence["factory_calls"] == 1
            and evidence.get("factory_restored") is True
            and len(evidence.get("build_proofs", [])) == 1,
            "Constructor variant was relabelled or its factory was not exercised/restored")
    proof = evidence["build_proofs"][0]
    require(proof.get("checked_before_cache_admission") is True
            and proof.get("expert_modules") == 47
            and proof.get("actual_expert_grouped_routes") == expected_routes(enabled)
            and proof.get("runtime_expert_grouped_routes") == expected_routes(enabled)
            and all(type(value) is bool for name in (
                "actual_expert_grouped_routes", "runtime_expert_grouped_routes",
            ) for value in proof[name].values()),
            "Actual expert construction did not prove the selected variant")


def validate_native_routes(raw: dict[str, Any], variant: str) -> None:
    flags = expected_routes(variant == "grouped")
    for name in ("setup", "accuracy", "warmup", "timing"):
        if name not in raw:
            continue
        for point in ("runtime_before", "runtime_after"):
            if point in raw[name]:
                actual = raw[name][point].get("expert_grouped_routes")
                require(isinstance(actual, dict) and actual == flags
                        and all(type(value) is bool for value in actual.values()),
                        "Native runtime route flags differ from the constructor variant")


def worker_main(spec_path: Path, variant: str, phase: str, out: Path,
                source_root: Path, source_commit: str) -> int:
    require(variant in ACCURACY_VARIANTS and phase in {"accuracy", "timing"},
            "Unknown worker role")
    require(phase == "accuracy" or variant in VARIANTS, "Ordinary is an accuracy control only")
    out.mkdir(parents=True, exist_ok=True)
    require(not any(path.name != "worker.log" for path in out.iterdir()),
            "Refuse replacing any existing worker evidence")
    envelope: dict[str, Any] = {"schema_version": 1, "status": "running", "variant": variant,
                                "native_arm": NATIVE_ARMS[variant], "phase": phase,
                                "spec_sha256": digest(spec_path), "pid": os.getpid()}
    envelope_path = out / "dispatch-envelope.json"
    save(envelope_path, envelope)
    try:
        native, worker, _common, module, identity = project(source_root, source_commit)
        envelope.update(identity)
        spec = read(spec_path)
        validate_protocol(spec)
        require(spec["git_head"] == source_commit, "Prepared source commit changed")
        require(native.source_fingerprint() == spec["source_fingerprint"],
                "Executed source changed after preparation")
        require(digest(Path(spec["training_data_path"])) == spec["training_data_sha256"],
                "Frozen input changed")
        _default_constructor_control(module)
        save(envelope_path, envelope)
        with dispatch(module, variant) as control:
            status = worker.run_worker(spec_path, NATIVE_ARMS[variant], phase, out)
        envelope["dispatch"] = control
        require(status == 0, "Native worker failed; its raw result remains retained")
        validate_dispatch_evidence(control, variant)
        raw_path = out / "result.json"
        raw = read(raw_path)
        require(raw["arm"] == NATIVE_ARMS[variant] and raw["phase"] == phase and
                raw["status"] == "passed" and raw["spec_sha256"] == digest(spec_path) and
                raw["pid"] == envelope["pid"], "Native worker identity mismatch")
        validate_native_routes(raw, variant)
        envelope.update(status="passed", native_result_sha256=digest(raw_path))
    except BaseException as error:
        envelope.update(status="failed", error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc())
    finally:
        save(envelope_path, envelope)
    return 0 if envelope["status"] == "passed" else 1


def launch(spec_path: Path, spec: dict[str, Any], variant: str, phase: str, out: Path,
           source_root: Path, source_commit: str) -> dict[str, Any]:
    native, _worker, _common, _module, identity = project(source_root, source_commit)
    native.verify_spec(spec, content=False)
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).absolute()), "worker", "--spec", str(spec_path),
               "--variant", variant, "--phase", phase, "--out", str(out),
               "--source-root", str(source_root), "--source-commit", source_commit]
    env = dict(os.environ, PYTHONPATH=str(source_root / "src") + os.pathsep +
               str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.perf_counter()
    with (out / "worker.log").open("wb") as log:
        process = subprocess.run(command, cwd=source_root, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=spec["timeout_seconds"])
    elapsed = time.perf_counter() - started
    require(process.returncode == 0, f"Native {phase}/{variant} failed; raw evidence preserved")
    envelope, raw = read(out / "dispatch-envelope.json"), read(out / "result.json")
    require(envelope["status"] == raw["status"] == "passed" and
            envelope["variant"] == variant and envelope["native_arm"] == raw["arm"] ==
            NATIVE_ARMS[variant] and envelope["phase"] == raw["phase"] == phase and
            envelope["spec_sha256"] == raw["spec_sha256"] == digest(spec_path) and
            envelope["pid"] == raw["pid"] and
            envelope["native_result_sha256"] == digest(out / "result.json") and
            all(envelope[key] == value for key, value in identity.items()),
            "Worker envelope differs from native result/source/spec")
    validate_dispatch_evidence(envelope["dispatch"], variant)
    validate_native_routes(raw, variant)
    result = {"variant": variant, "native_arm": raw["arm"], "worker": raw,
              "envelope": envelope, "envelope_sha256": digest(out / "dispatch-envelope.json"),
              "fresh_process_wall_seconds": elapsed}
    save(out / "controller-result.json", result)
    return result


def cache_delta(raw: dict[str, Any], phase: str, *, cached: bool) -> dict[str, int]:
    stage = raw[phase]
    before, after = (stage[key]["expert_cache"] for key in ("runtime_before", "runtime_after"))
    row = {key: after[key] - before[key] for key in
           ("logical_fetches", "storage_reads", "storage_bytes_read", "materializations")}
    for tier, ceiling in (("host", 8 * GIB), ("device", 2 * GIB)):
        for snapshot in (before, after):
            require(snapshot[tier]["budget_bytes"] == (ceiling if cached else 0) and
                    0 <= snapshot[tier]["payload_bytes"] <= (ceiling if cached else 0),
                    "Expert payload/ceiling differs from the fixed arm")
        row[tier + "_hits"] = after[tier]["hits"] - before[tier]["hits"]
    require(all(type(value) is int and value >= 0 for value in row.values()) and
            row["logical_fetches"] > 0 and row["storage_reads"] > 0 and
            row["storage_bytes_read"] > 0, "Invalid or missing expert counters")
    require(row["storage_reads"] + row["host_hits"] + row["device_hits"] ==
            row["logical_fetches"] and row["materializations"] ==
            row["logical_fetches"] - row["device_hits"], "Expert counter accounting changed")
    if cached:
        require(before.get("host_cache_policy") == after.get("host_cache_policy") == "static"
                and min(row["host_hits"], row["device_hits"]) > 0,
                "Both admitted cache tiers must actually be exercised")
        plan = raw["setup"]["cache_plan"]
        require(plan["host_cache_bytes"] == 8 * GIB and plan["device_cache_bytes"] == 2 * GIB
                and plan["host_cache_policy"] == "static", "Full cache admission changed")
    else:
        require(row["host_hits"] == row["device_hits"] == 0, "Ordinary unexpectedly used caches")
    return row


def full_capture_controls(capture: Any, inputs: Any, common: Any) -> dict[str, Any]:
    import torch

    gate = common.validate_capture(capture)
    require(gate["passed"], "Native full-capture structural/finite gate failed")
    require(len(capture["updates"]) == 2 and len(capture["microbatches"]) == 4 and
            len(capture["initial_adapters"]) == 288, "Full Kanana trajectory is incomplete")
    initial, final = capture["initial_adapters"], capture["updates"][-1]["adapters"]
    require(all("lora_" in name and value.dtype == torch.float32 and
                final[name].dtype == torch.float32 and torch.count_nonzero(value).item() > 0
                and torch.count_nonzero(final[name]).item() > 0
                and not torch.equal(value, final[name]) for name, value in initial.items()),
            "Every adapter must be nonzero and actually changed")
    require(all(any(torch.count_nonzero(batch["gradients"][name]).item() > 0
                    for batch in capture["microbatches"]) for name in initial),
            "Every adapter must receive a nonzero gradient")
    require(all(torch.equal(batch["input_ids"], inputs[index])
                for index, batch in enumerate(capture["microbatches"])),
            "Complete capture input IDs differ from the frozen fixture")
    return {"passed": True, "adapters_nonzero_changed_and_nonzero_gradients": 288,
            "updates": 2, "microbatches": 4, "native_gate": gate,
            "full_microbatch_losses": [float(batch["loss"]) for batch in capture["microbatches"]]}


def loss_gate(reference: list[float], candidate: list[float], common: Any) -> dict[str, Any]:
    require(reference and len(reference) == len(candidate), "Loss sequence is empty/incomplete")
    failures = []
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        finite = all(type(value) in (int, float) and math.isfinite(value)
                     for value in (expected, actual))
        if (not finite or abs(actual - expected) > common.ATOL + common.RTOL * abs(expected)
                or f"{actual:.3f}" != f"{expected:.3f}"):
            failures.append({"index": index, "reference": expected, "candidate": actual})
    return {"passed": not failures, "checks": len(reference), "failures": failures,
            "bitwise_scalar_values": reference == candidate}


def timing_summary(blocks: list[dict[str, Any]], common: Any) -> dict[str, Any]:
    require(len(blocks) == 8, "Exactly eight complete timing blocks are required")
    seen, pids, by_round = set(), set(), {}
    identities = set()
    loss_checks, counter_checks = [], []
    for block in blocks:
        number, position, variant = block["round"], block["position"], block["variant"]
        require(type(number) is int and 0 <= number < 4 and type(position) is int and
                0 <= position < 2 and ORDERS[number][position] == variant and
                (number, variant) not in seen, "Unbalanced/duplicate timing schedule")
        seen.add((number, variant))
        result = block["result"]
        raw, envelope = result["worker"], result["envelope"]
        require(raw["arm"] == result["native_arm"] == envelope["native_arm"] ==
                "two_read_cache" and result["variant"] == envelope["variant"] == variant
                and raw["phase"] == envelope["phase"] == "timing" and
                raw["status"] == envelope["status"] == "passed" and
                type(raw["pid"]) is int and raw["pid"] > 0 and
                raw["pid"] == envelope["pid"] and raw["pid"] not in pids,
                "Native arm/envelope/fresh-process identity changed")
        validate_dispatch_evidence(envelope["dispatch"], variant)
        validate_native_routes(raw, variant)
        require(raw["spec_sha256"] == envelope["spec_sha256"], "Native/envelope spec differs")
        identities.add(tuple(envelope[key] for key in (
            "spec_sha256", "source_commit", "expert_module_sha256", "constructor_source_sha256",
            "runtime_module_sha256", "wrapper_sha256")))
        pids.add(raw["pid"])
        for stage, count in (("warmup", 2), ("timing", 6)):
            data = raw[stage]
            require(data.get("complete") is True and data.get("updates_complete") == count and
                    data.get("finite_state_checks_per_update") is True and
                    len(data.get("update_seconds", [])) == len(data.get("losses", [])) == count
                    and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                            for value in data["update_seconds"]), "Incomplete finite timing block")
        by_round.setdefault(number, {})[variant] = block
    require(len(identities) == 1, "Timing workers did not use one frozen source/spec")
    logs = []
    for number in range(4):
        base, candidate = (by_round[number][variant]["result"]["worker"] for variant in VARIANTS)
        logs.append(math.log(statistics.median(candidate["timing"]["update_seconds"]) /
                             statistics.median(base["timing"]["update_seconds"])))
        for stage in ("warmup", "timing"):
            gate = loss_gate(base[stage]["losses"], candidate[stage]["losses"], common)
            require(gate["passed"], "Recorded warmup/timed numerical loss gate failed")
            loss_checks.append({"round": number, "stage": stage, **gate})
        left = cache_delta(base, "timing", cached=True)
        right = cache_delta(candidate, "timing", cached=True)
        require(left == right, "Variants performed different timed cache/expert work")
        counter_checks.append({"round": number, "equal": True, "reference": left,
                               "candidate": right})
    rng = random.Random(29)
    draws = sorted(100 * (1 - math.exp(statistics.mean(rng.choices(logs, k=4))))
                   for _ in range(5000))
    lower, upper = draws[124], draws[4874]
    absolute = {}
    for variant in VARIANTS:
        values = [block["result"] for block in blocks if block["variant"] == variant]
        absolute[variant] = {
            "median_step_seconds": statistics.median(
                value for row in values for value in row["worker"]["timing"]["update_seconds"]),
            "median_setup_seconds": statistics.median(row["worker"]["setup"]["seconds"]
                                                       for row in values),
            "median_whole_entry_seconds": statistics.median(
                row["fresh_process_wall_seconds"] for row in values),
            "max_step_allocated_bytes": max(row["worker"]["timing"]["cuda"][
                "peak_allocated_bytes"] for row in values),
            "max_step_reserved_bytes": max(row["worker"]["timing"]["cuda"][
                "peak_reserved_bytes"] for row in values),
            "max_sampled_timing_rss_bytes": max(row["worker"]["timing"]["resources"][
                "sampled_peak_rss_bytes"] for row in values),
        }
    return {
        "reference_variant": "legacy", "candidate_variant": "grouped",
        "native_arm_for_both": "two_read_cache", "paired_rounds": 4,
        "step_time_reduction_percent": 100 * (1 - math.exp(statistics.mean(logs))),
        "paired_bootstrap_95_percent": [lower, upper],
        "passes_practical_2_percent_gate": lower >= 2., "bootstrap_samples": 5000,
        "bootstrap_seed": 29, "absolute": absolute,
        "recorded_loss_gates": loss_checks, "equal_cache_work": counter_checks,
        "recording_limits": ["NO VERDICT for unrecorded first timed microbatch loss.",
                             "Full gradients/AdamW states cover the separate accuracy phase.",
                             "Frozen random tokens; downstream quality/convergence unmeasured."],
    }


def run(spec_path: Path, out: Path, source_root: Path, source_commit: str) -> dict[str, Any]:
    import torch

    require(not out.exists() or not any(out.iterdir()), "Output must be new/empty; never overwrite")
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": 1, "status": "running", "accuracy": {},
                              "timing_blocks": [], "gpu_rented_by_runner": False,
                              "spec_sha256": digest(spec_path), "source_commit": source_commit,
                              "hypothesis": "opt-in grouped expert routes after baseline profile"}
    try:
        native, _worker, common, module, identity = project(source_root, source_commit)
        _default_constructor_control(module)
        report["identity"] = identity
        spec = read(spec_path)
        validate_protocol(spec)
        native.verify_spec(spec, content=True)
        report["original_gates"] = {"reference_relative_rtol": common.RTOL, "atol": common.ATOL,
                                    "loss_three_decimal": True, "finite": True}
        for variant in ACCURACY_VARIANTS:
            report["accuracy"][variant] = {"result": launch(
                spec_path, spec, variant, "accuracy", out / "accuracy" / variant,
                source_root, source_commit)}
            save(out / "run.json", report)
        inputs = torch.load(spec["training_data_path"], map_location="cpu", weights_only=True)
        for reference in ("ordinary_legacy", "legacy"):
            expected_path = out / "accuracy" / reference / "capture.pt"
            expected = torch.load(expected_path, map_location="cpu", weights_only=True)
            report["accuracy"][reference]["capture_controls"] = full_capture_controls(
                expected, inputs, common)
            candidates = (("legacy", "grouped") if reference == "ordinary_legacy" else ("grouped",))
            for variant in candidates:
                path = out / "accuracy" / variant / "capture.pt"
                actual = torch.load(path, map_location="cpu", weights_only=True)
                report["accuracy"][variant]["capture_controls"] = full_capture_controls(
                    actual, inputs, common)
                gate = common.compare_captures(expected, actual)
                report["accuracy"][variant]["versus_" + reference] = gate
                require(gate["passed"], "Unchanged full trajectory accuracy gate failed")
                del actual
            del expected
        counters, accuracy_pids = {}, set()
        for variant in ACCURACY_VARIANTS:
            cell = report["accuracy"][variant]
            path = out / "accuracy" / variant / "capture.pt"
            raw = cell["result"]["worker"]
            require(digest(path) == raw["accuracy"]["capture_sha256"], "Capture SHA changed")
            require(raw["pid"] not in accuracy_pids, "Accuracy workers were not fresh")
            accuracy_pids.add(raw["pid"])
            cell["capture"] = {"bytes": path.stat().st_size, "sha256": digest(path)}
            counters[variant] = cache_delta(raw, "accuracy", cached=variant != "ordinary_legacy")
        require(counters["legacy"] == counters["grouped"] and
                counters["ordinary_legacy"]["logical_fetches"] * 2 ==
                counters["legacy"]["logical_fetches"] * 3,
                "Three/two-read or equal cached-work invariant changed")
        report["accuracy_cache_counters"] = counters
        native.verify_spec(spec, content=True)
        report["accuracy_passed"] = True
        save(out / "run.json", report)
        for number, order in enumerate(ORDERS):
            for position, variant in enumerate(order):
                result = launch(spec_path, spec, variant, "timing",
                                out / "timing" / f"round-{number:02d}" / variant,
                                source_root, source_commit)
                require(result["worker"]["pid"] not in accuracy_pids,
                        "Timing reused an accuracy process")
                report["timing_blocks"].append({"round": number, "position": position,
                                                "variant": variant, "result": result})
                save(out / "run.json", report)
        report["timing_summary"] = timing_summary(report["timing_blocks"], common)
        report["warmup_versus_accuracy_loss_gates"] = []
        for block in report["timing_blocks"]:
            expected = report["accuracy"][block["variant"]]["capture_controls"][
                "full_microbatch_losses"][1::2]
            gate = loss_gate(expected, block["result"]["worker"]["warmup"]["losses"], common)
            report["warmup_versus_accuracy_loss_gates"].append({
                "round": block["round"], "variant": block["variant"], **gate,
            })
            require(gate["passed"], "Fresh warmup differs from the full accuracy trajectory")
        native.verify_spec(spec, content=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        save(out / "run.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--source-commit", required=True)
        if name == "worker":
            command.add_argument("--variant", choices=ACCURACY_VARIANTS, required=True)
            command.add_argument("--phase", choices=("accuracy", "timing"), required=True)
    args = parser.parse_args(argv)
    args.spec = Path(os.path.realpath(args.spec))
    args.out = Path(os.path.realpath(args.out))
    args.source_root = Path(os.path.realpath(args.source_root))
    if args.command == "worker":
        return worker_main(args.spec, args.variant, args.phase, args.out,
                           args.source_root, args.source_commit)
    result = run(args.spec, args.out, args.source_root, args.source_commit)
    Console().print({"status": result["status"], "out": str(args.out),
                     "accuracy_passed": result["accuracy_passed"],
                     "step_gate": result["timing_summary"]["passes_practical_2_percent_gate"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
