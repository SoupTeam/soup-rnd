"""D3 development checks on pinned OLMoE layers, not a formal timing protocol.

Groups NF4 restoration only; GEMMs remain the unchanged eager operations.
Reuses local B1.2 checkpoint files, never downloads weights or trains a model.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _baseline() -> Any:
    path = ROOT / "benchmarks/harness/b1_expert_forward_gate.py"
    spec = importlib.util.spec_from_file_location("d3_restore_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResidentControlPool:
    """Oracle only: existing resident weights, the SAME compute class/backend.

    This is not a bounded streaming pool. It deliberately retains the resident
    reference outside the streamed-memory measurements and performs no reads.
    """

    def __init__(self, module: Any, reader: Any, width: int) -> None:
        self.module = module
        self.device = module.gate_up_proj.device
        self.dtype = str(module.gate_up_proj.dtype).removeprefix("torch.")
        self._shapes = dict(reader.shapes)
        self.group_width = width

    def validate_reader(self, reader: Any) -> None:
        if dict(reader.shapes) != self._shapes or reader.count != len(self.module.down_proj):
            raise ValueError("resident control geometry does not match reader")

    @contextmanager
    def lease(self, reader: Any, experts: list[int], *, trace: list | None = None) -> Any:
        import torch

        self.validate_reader(reader)
        if trace is not None:
            trace.extend({"expert": expert, "logical_bytes_read": 0} for expert in experts)
        try:
            yield tuple((self.module.gate_up_proj[expert], self.module.down_proj[expert])
                        for expert in experts)
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)


def profile_control(folder: Path, operations: dict, expected: Any) -> dict:
    """Separate development tracing; never derive formal throughput from durations."""
    import torch

    result = {}
    for name, operation in operations.items():
        with torch.no_grad():
            operation()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
            ]) as profile:
                observed = operation().cpu()
                torch.cuda.synchronize()
        if not torch.equal(observed, expected):
            raise RuntimeError("profiled operation does not preserve the exact output")
        path = folder / f"profile-{name}.json"
        profile.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        kernels = [row for row in events if row.get("cat") == "kernel" and row.get("ph") == "X"]
        if not kernels:
            raise RuntimeError("no actual CUDA kernel events in development trace")
        result[name] = {"cuda_kernel_events": len(kernels),
                        "kernel_names": [row["name"] for row in kernels]}
    return result


def restore_control(
    folder: Path, *, config: Any, tensors: dict, device: str, codec: str,
    widths: tuple[int, ...] = (1, 2, 4), layer: int = 0,
    compute_library: Path | None = None, tokens: int = 17,
    profile: bool = False,
) -> dict:
    """One reproducible development cell; no timings or task-level verdict."""
    import torch
    from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter

    from soup_cli.utils.expert_group import GroupedExpertPool, grouped_experts_class
    from soup_cli.utils.expert_group_compute import grouped_compute_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.expert_stream import ExpertBufferPool, streamed_experts_class
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    if codec not in ("dense", "nf4", "nf4-dq") or device not in ("cpu", "cuda"):
        raise ValueError("unsupported development codec/device")
    if type(tokens) is not int or not 1 <= tokens <= 512:
        raise ValueError("development token count must be from 1 through 512")
    if compute_library is not None and device != "cuda":
        raise ValueError("grouped GEMM integration requires CUDA")
    if profile and compute_library is None:
        raise ValueError("full compute profiling requires an explicit compute library")
    if (not widths or any(type(width) is not int or width not in (1, 2, 4) for width in widths)
            or len(set(widths)) != len(widths)):
        raise ValueError("widths must be unique values from 1, 2, 4")
    folder.mkdir(parents=True, exist_ok=False)
    baseline = _baseline()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    module = baseline._experts_module(config, tensors, device, dtype)
    path = folder / "experts.safetensors"
    info = write_expert_shard(
        str(path), ((gate.detach(), down.detach()) for gate, down in zip(
            module.gate_up_proj, module.down_proj
        )), quantise=codec != "dense", double_quant=codec == "nf4-dq", device=device,
    )
    codes = {NF4_CODE_KEY: info.codes[0], NF4_NESTED_CODE_KEY: info.codes[1]}
    reader = ExpertShardReader(
        str(path), count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype=str(dtype).removeprefix("torch."), direct=False,
    )
    pool = None
    gemm = None
    result = {"codec": codec, "layer": layer, "groups": [], "passed": False}
    try:
        if codec != "dense":
            baseline._nf4_reference(module, reader, codes, device)
        gen = torch.Generator().manual_seed(1234)
        states = torch.randn(tokens, config.hidden_size, generator=gen)
        states = (states * tensors["post_attention_layernorm.weight"].float()).to(device, dtype)
        router = OlmoeTopKRouter(config).requires_grad_(False).to(device=device, dtype=dtype)
        with torch.no_grad():
            router.weight.copy_(tensors["mlp.gate.weight"])
            _, weights, index = router(states)
            hit = sorted(set(index.cpu().flatten().tolist()))
            expected = module(states, index, weights).cpu()
            repeat = module(states, index, weights).cpu()
            module.down_proj[hit[0]].add_(0.125)
            negative = module(states, index, weights).cpu()
        result.update({
            "reference_sha256": baseline._tensor_hash(expected),
            "negative_sha256": baseline._tensor_hash(negative),
            "repeat_equal": torch.equal(expected, repeat),
            "negative_differs": not torch.equal(expected, negative),
            "nonzero_finite": bool(torch.isfinite(expected).all() and expected.count_nonzero()),
            "routing_indices": index.cpu().tolist(), "routing_weights": weights.cpu().tolist(),
            "tokens": tokens, "reader_direct_io": reader.direct,
            "input_scope": "synthetic hidden states scaled by layer norm; actual router weights",
        })
        # Immutable setup artifacts for a FUTURE committed performance protocol.
        # No timings are collected here; source/data identity precedes formal arms.
        from safetensors.torch import save_file

        inputs_path = folder / "inputs.safetensors"
        save_file({"hidden": states.cpu().contiguous(), "index": index.cpu().contiguous(),
                   "weights": weights.cpu().contiguous(), "expected": expected.contiguous()},
                  str(inputs_path))
        recipe = {
            "count": info.count, "shapes": {key: list(value) for key, value in info.shapes.items()},
            "quant_specs": {key: spec.to_json() for key, spec in info.quant_specs.items()},
            "codes": {key: value.cpu().tolist() for key, value in codes.items()
                      if value is not None},
            "dtype": reader.dtype, "activation": config.hidden_act,
            "weights_sha256": baseline._sha256(path),
            "inputs_sha256": baseline._sha256(inputs_path), "codec": codec, "layer": layer,
            "reference_math": "cublaslt-no-splitk" if compute_library else "current",
        }
        recipe_path = folder / "recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        result["recipe_sha256"] = baseline._sha256(recipe_path)
        if compute_library is not None:
            # The negative mutation above must not enter this resident oracle.
            module.down_proj[hit[0]].copy_(tensors[
                f"mlp.experts.{hit[0]}.down_proj.weight"
            ])
            if codec != "dense":
                baseline._nf4_reference(module, reader, codes, device)
            resident_gemm = GroupedGemm(compute_library, call_budget_bytes=64 * 2**20)
            result["resident_same_kernel"] = []
            try:
                for width in widths:
                    control_pool = ResidentControlPool(module, reader, width)
                    control = grouped_compute_experts_class()(
                        reader, control_pool, module.act_fn, layer, gemm=resident_gemm,
                        activation_budget_bytes=256 * 2**20,
                    )
                    with torch.no_grad():
                        resident = control(states, index, weights).cpu()
                    result["resident_same_kernel"].append({
                        "width": width, "equal": torch.equal(resident, expected),
                        "sha256": baseline._tensor_hash(resident),
                        "compute": resident_gemm.stats(),
                    })
                del control, control_pool
            finally:
                resident_gemm.close()
        act_fn = module.act_fn
        del module, router
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        # Check the unchanged sequential reader before the grouped pool, using
        # identical bytes/routes, after releasing the resident expert stack.
        pool = ExpertBufferPool(reader, device=device, dtype=reader.dtype,
                                weight_budget_bytes=64 * 2**20, codes=codes)
        with torch.no_grad():
            sequential = streamed_experts_class()(reader, pool, act_fn, layer)(
                states, index, weights,
            ).cpu()
        result["b12_equal"] = torch.equal(expected, sequential)
        pool.close()
        pool = None
        if compute_library is not None:
            gemm = GroupedGemm(compute_library, call_budget_bytes=64 * 2**20)
        for width in widths:
            baseline._reset_peak(device)
            before = baseline._memory(device)
            pool = GroupedExpertPool(reader, device=device, dtype=reader.dtype, group_width=width,
                                     weight_budget_bytes=64 * 2**20, codes=codes)
            if gemm is None:
                candidate = grouped_experts_class()(reader, pool, act_fn, layer)
            else:
                candidate = grouped_compute_experts_class()(
                    reader, pool, act_fn, layer, gemm=gemm, activation_budget_bytes=256 * 2**20,
                )
            compute_before = gemm.stats() if gemm is not None else None
            fetches = reader.fetches
            with torch.no_grad():
                observed = candidate(states, index, weights).cpu()
                again = candidate(states, index, weights).cpu()
            row = {
                "width": width, "equal": torch.equal(expected, observed),
                "repeat_equal": torch.equal(observed, again),
                "observed_sha256": baseline._tensor_hash(observed),
                "finite": bool(torch.isfinite(observed).all()),
                "loads_ok": reader.fetches - fetches == 2 * len(hit)
                and [entry["expert"] for entry in candidate.last_trace] == hit,
                "max_abs_diff": float((expected.float() - observed.float()).abs().max()),
                "trace": candidate.last_trace, "pool": pool.stats(),
                "compute_before": compute_before, "compute_after": gemm.stats() if gemm else None,
                "activation_plan": candidate.last_workspace_plan if gemm else None,
                "memory": {"before_pool": before, "after_forwards": baseline._memory(device)},
            }
            row["passed"] = all(row[key] for key in ("equal", "repeat_equal", "finite", "loads_ok"))
            if gemm is not None:
                row["grouped_matrices_executed"] = (
                    row["compute_after"]["grouped_matrices"] - compute_before["grouped_matrices"]
                )
                if tokens == 512:
                    row["passed"] &= row["grouped_matrices_executed"] > 0
            result["groups"].append(row)
            pool.close()
            pool = None
            del candidate
            if not row["passed"]:
                raise RuntimeError(f"grouped restore development check failed at width {width}")
        if compute_library is not None:
            result["same_kernel_resident_equal"] = all(
                row["equal"] for row in result["resident_same_kernel"]
            )
            if not result["same_kernel_resident_equal"]:
                raise RuntimeError("same-kernel resident control differs from native reference")
        if profile:
            pool = GroupedExpertPool(reader, device=device, dtype=reader.dtype, group_width=4,
                                     weight_budget_bytes=64 * 2**20, codes=codes)
            candidate = grouped_compute_experts_class()(
                reader, pool, act_fn, layer, gemm=gemm, activation_budget_bytes=256 * 2**20,
            )
            sequential_pool = ExpertBufferPool(reader, device=device, dtype=reader.dtype,
                                              weight_budget_bytes=64 * 2**20, codes=codes)
            try:
                sequential = streamed_experts_class()(reader, sequential_pool, act_fn, layer)
                result["development_profile"] = profile_control(folder, {
                    "A-native": lambda: sequential(states, index, weights),
                    "B-grouped": lambda: candidate(states, index, weights),
                }, expected)
            finally:
                sequential_pool.close()
                pool.close()
                pool = None
                del candidate
        result["passed"] = all(result[key] for key in (
            "repeat_equal", "negative_differs", "nonzero_finite", "b12_equal",
        )) and all(row["passed"] for row in result["groups"])
        result["experts_file_sha256"] = baseline._sha256(path)
        return result
    finally:
        reader.close()
        if pool is not None:
            pool.close()
        if gemm is not None:
            gemm.close()
        (folder / "progress.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compute-library", type=Path)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 7, 15])
    args = parser.parse_args(argv)
    if args.profile and args.compute_library is None:
        parser.error("--profile requires --compute-library")
    if (not args.layers or len(set(args.layers)) != len(args.layers)
            or set(args.layers) - {0, 7, 15}):
        parser.error("--layers must be unique values from 0, 7, 15")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "formal_measurement": False, "d3_accepted": False, "cells": []}
    try:
        import torch
        from transformers import OlmoeConfig

        if sys.platform != "linux" or not torch.cuda.is_available():
            raise RuntimeError("Linux CUDA required")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("native BF16 required")
        baseline = _baseline()
        names = (*baseline.SOURCES, "benchmarks/harness/d3_grouped_restore_check.py",
                 "src/soup_cli/utils/expert_group.py", "src/soup_cli/utils/expert_group_decode.py",
                 "src/soup_cli/utils/expert_group_compute.py",
                 "src/soup_cli/utils/expert_group_gemm.py",
                 "benchmarks/harness/d3_cublas_grouped_probe.py")
        report["source_sha256"] = {name: baseline._sha256(ROOT / name) for name in names}
        for name in names:
            target = args.out / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        report["git_head"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip()
        report["source_scope"] = "working files, not a frozen formal-run commit"
        rule = json.loads(baseline.PROTOCOL.read_text(encoding="utf-8"))
        report["checkpoint_sha256"] = baseline._checkpoint_identity(args.model_dir, rule)
        report["environment"] = baseline._box()
        # Read only the pinned local config; never use from_pretrained or the Hub.
        config = OlmoeConfig(**json.loads((args.model_dir / "config.json").read_text()))
        torch.backends.cuda.matmul.allow_tf32 = False
        context = nullcontext({"mode": "current"})
        if args.compute_library is not None:
            path = ROOT / "benchmarks/harness/d3_cublas_grouped_probe.py"
            spec = importlib.util.spec_from_file_location("d3_compute_policy", path)
            policy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(policy_module)
            context = policy_module.reference_policy("cublaslt-no-splitk")
            report["compute_library_sha256"] = baseline._sha256(args.compute_library)
        with context as policy:
            report["reference_policy"] = policy
            for layer in args.layers:
                tensors = baseline._layer_tensors(args.model_dir, layer)
                for codec in ("dense", "nf4", "nf4-dq"):
                    result = restore_control(
                        args.out / f"layer-{layer:02d}-{codec}", config=config,
                        tensors=tensors, device="cuda", codec=codec, layer=layer,
                        compute_library=args.compute_library, tokens=args.tokens,
                        profile=args.profile and layer == 7,
                    )
                    report["cells"].append(result)
                    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                    if not result["passed"]:
                        raise RuntimeError(f"development control failed: layer {layer}, {codec}")
                del tensors
        report["source_unchanged"] = all(
            baseline._sha256(ROOT / name) == value
            for name, value in report["source_sha256"].items()
        )
        if not report["source_unchanged"]:
            raise RuntimeError("working source changed during the checks")
        report["status"] = "trained_layer_development_checks_passed"
    except (Exception, KeyboardInterrupt):
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Console().print(f"{report['status']}: {os.fspath(args.out)}")
    return 0 if report["status"] == "trained_layer_development_checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
