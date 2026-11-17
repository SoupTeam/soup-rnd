#!/usr/bin/env python3
"""One L2L optimizer step on tiny synthetic DeepSeek-V3: BF16 vs FP8 vs FP8 + Oracle.

Run from the checkout root with src on PYTHONPATH:
    python benchmarks/harness/p3_fp8_deepseek_step.py --device cuda --output out/

Every arm builds the same NF4 per-expert checkpoint (fixed seed), the same initial
LoRA adapters and the same inputs, and computes in BF16. Only the boundary storage
differs: ``none`` (BF16 boundaries), ``fp8_e4m3``, and ``fp8_e4m3`` with the #66
Oracle, which replays the original forward's routes in the recompute. Stores come
from the production ``_store_for`` against a resolved ``plan_moe_l2l``; the routing
guard is never relaxed. Each (arm, k) runs in its own process so the RSS peak
belongs to that arm. The BF16 arm is the reference for E1-E3.

This is synthetic tiny-model evidence, not DeepSeek-R1 671B training.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console

ARMS = ("bf16", "fp8", "fp8_oracle")
CODEC = {"bf16": "none", "fp8": "fp8_e4m3", "fp8_oracle": "fp8_e4m3"}
SEED = 23
SHAPE = (1, 64, 128)
HOST_BUDGET_BYTES = 256 * 1024**2
GUARD = re.compile(
    r"layer (\d+) micro-batch (\d+) was routed differently.*?\((\d+) of (\d+) tokens"
)


def _harness(name: str) -> Any:
    prefix = f"{__package__}." if __package__ else ""
    return importlib.import_module(prefix + name)


def _rss() -> int:
    with open("/proc/self/statm", encoding="ascii") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _rss_high_water() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _PeakRss:
    """Sampled RSS peak of the step (lower bound) beside the process high-water mark."""

    def __enter__(self) -> "_PeakRss":
        self.peak, self.stop = _rss(), threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self) -> None:
        while not self.stop.wait(0.005):
            self.peak = max(self.peak, _rss())

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, _rss())


class _Boundaries:
    """Record each layer's forward and recompute input/output for E1 and E2."""

    def __init__(self, layers: Sequence[Any]) -> None:
        import torch

        self.seen: dict[tuple[str, int, int], tuple[Any, Any]] = {}
        self.visits: dict[tuple[str, int], int] = {}
        self.handles = []
        for index, layer in enumerate(layers):
            def hook(_module: Any, args: Any, kwargs: Any, output: Any, *, at: int = index):
                phase = "recompute" if torch.is_grad_enabled() else "forward"
                mb = self.visits.get((phase, at), 0)
                self.visits[(phase, at)] = mb + 1
                hidden = args[0] if args else kwargs["hidden_states"]
                out = output[0] if isinstance(output, tuple) else output
                self.seen[(phase, at, mb)] = (
                    hidden.detach().float().cpu(), out.detach().float().cpu(),
                )
            self.handles.append(layer.register_forward_hook(hook, with_kwargs=True))

    def errors(self) -> dict[str, Any]:
        metrics = _harness("g4_activation_codec")
        restore: dict[int, list[Any]] = {}
        branch: dict[int, list[Any]] = {}
        for (phase, layer, mb), (hidden, out) in self.seen.items():
            if phase != "forward" or ("recompute", layer, mb) not in self.seen:
                continue
            r_hidden, r_out = self.seen[("recompute", layer, mb)]
            restore.setdefault(layer, []).append(metrics.restore_error(hidden, r_hidden))
            branch.setdefault(layer, []).append(
                metrics.restore_error(out - hidden, r_out - r_hidden)
            )
        return {
            "e1_restore": {str(k): metrics.merge_errors(v) for k, v in sorted(restore.items())},
            "e2_branch": {str(k): metrics.merge_errors(v) for k, v in sorted(branch.items())},
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def run_child(arm: str, k: int, device: str, output: Path) -> dict[str, Any]:
    """One arm, one optimizer step; writes gradients beside the JSON record."""
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    smoke = _harness("p3_deepseek_fp8_smoke")
    fixture = _harness("b13_moe_fixture")
    codec = CODEC[arm]
    record: dict[str, Any] = {"arm": arm, "k": k, "codec": codec, "device": device,
                              "status": "FAIL", "rss_before_build_bytes": _rss()}
    budget = HostMemoryBudget(HOST_BUDGET_BYTES)
    model = runtime = state = probe = None
    temporary = tempfile.TemporaryDirectory(prefix=f"p3-step-{arm}-k{k}-")
    try:
        meta: dict[str, Any] = {}
        model, runtime, config = smoke._build_candidate(Path(temporary.name), device, meta)
        record["model"] = {"config": meta["model_config"], "quantization": {
            key: meta["quantization"][key]
            for key in ("weight_format", "double_quant", "expert_layout", "shard_dtype",
                        "synthetic_checkpoint_sha256")
        }, "precision": {key: meta["precision"][key] for key in (
            "compute_dtype", "trainable_tensors", "trainable_elements", "targets",
        )}}
        parameters = fixture.trainables(model)
        generator = torch.Generator().manual_seed(SEED)
        ids = [torch.randint(0, config.vocab_size, SHAPE[:2], generator=generator).to(device)
               for _ in range(k)]
        labels = [value.clone() for value in ids]
        for index, label in enumerate(labels):
            label[0, : 1 + index % 3] = -100
        total = sum(int((label[:, 1:] != -100).sum()) for label in labels)
        n_moe = config.num_hidden_layers - config.first_k_dense_replace
        plan = plan_moe_l2l(
            budget, n_layers=config.num_hidden_layers, n_moe_layers=n_moe, max_shape=SHAPE,
            experts_per_token=config.num_experts_per_tok, requested_k=k, dtype="bfloat16",
            pin=device == "cuda", expert_input_dtype="bfloat16", codec=codec,
            device_type=device,
        )
        if plan.effective_k != k:
            raise MemoryError(f"plan reduced k={k} to {plan.effective_k}")
        record["memory_plan"] = asdict(plan)
        state = L2LState(runtime=runtime, dtype=torch.bfloat16, activation_codec=codec,
                         host_budget=budget, memory_plan=plan, has_dropout=False)
        calls = [capture_layer_call(model, value, draws=False) for value in ids]
        if arm == "fp8_oracle":
            oracle = _harness("p3_deepseek_oracle")
            probe = oracle.DeepseekOracleProbe(model, [SHAPE] * k, budget)
            stepper = smoke._stepper(model, runtime, calls, probe)
        else:
            stepper = L2LStep(model, runtime, calls[0])
        store = _store_for(state, stepper, ids, ids[0].device)
        boundaries = _Boundaries(stepper.layers)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        record["rss_before_step_bytes"] = _rss()
        started = time.perf_counter()
        try:
            with _PeakRss() as peak:
                losses = stepper.run(
                    ids, store, labels=labels, n_items=total, calls=calls,
                    routing_budget=(plan.routing_record_bytes, plan.routing_comparison_bytes),
                )
                if device == "cuda":
                    torch.cuda.synchronize()
        except RuntimeError as error:
            match = GUARD.search(str(error))
            if match is None:
                raise
            layer, mb, changed, tokens = (int(value) for value in match.groups())
            record.update(status="ROUTING_REFUSED", routing={
                "complete": False, "layer": layer, "microbatch": mb,
                "changed_tokens": changed, "tokens": tokens, "message": str(error),
            })
            record["step_rss_sampled_peak_bytes"] = peak.peak
            return record
        finally:
            boundaries.close()
            record["step_seconds"] = time.perf_counter() - started
            record["process_rss_high_water_bytes"] = _rss_high_water()
            if device == "cuda":
                record["cuda_allocated_peak_bytes"] = torch.cuda.max_memory_allocated()
                record["cuda_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
            record["store"] = store.host_bytes() if hasattr(store, "host_bytes") else {
                "host_bytes": plan.activation_bytes,
            }
            record["store_class"] = type(store).__name__
            if probe is not None:
                record["oracle"] = {key: value for key, value in probe.report().items()
                                    if key in ("complete", "counts", "host_reserved_bytes",
                                               "device_scratch_bound_bytes")}
            record["budget_used_during_step_bytes"] = budget.used_bytes
        record["step_rss_sampled_peak_bytes"] = peak.peak
        record["routing"] = {"complete": True}
        record["losses"] = [float(loss.detach()) for loss in losses]
        record["window_loss"] = sum(record["losses"])
        record.update(boundaries.errors())
        if probe is not None:
            probe.require_complete()
        grads = {name: value.grad.detach().float().cpu() for name, value in parameters.items()}
        if any(grad is None or not bool(torch.isfinite(grad).all()) for grad in grads.values()):
            raise ValueError("missing or nonfinite adapter gradient")
        torch.save(grads, output.with_suffix(".grads.pt"))
        optimizer = torch.optim.SGD(parameters.values(), lr=0.05)
        optimizer.step()
        record["optimizer_applied"] = True
        record["status"] = "COMPLETE"
        return record
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        record["traceback"] = traceback.format_exc()
        return record
    finally:
        if probe is not None:
            probe.close()
        if state is not None:
            state.close()
        record["budget_used_after_close_bytes"] = budget.used_bytes
        if runtime is not None:
            runtime.close()
        temporary.cleanup()


def _gradient_gate(reference: Path, candidate: Path) -> dict[str, Any]:
    import torch

    g4 = _harness("g4_rule")
    metrics = _harness("g4_activation_codec").grad_agreement(
        torch.load(reference), torch.load(candidate),
    )
    return {
        "global_rel_l2": metrics["global_rel_l2"], "min_cosine": metrics["min_cosine"],
        "tensors": metrics["tensors"],
        "pass": metrics["global_rel_l2"] <= g4.GRAD_GLOBAL_BOUND
        and metrics["min_cosine"] >= g4.GRAD_MIN_COSINE,
    }


def _boundary_gate(record: dict[str, Any]) -> dict[str, Any]:
    g4 = _harness("g4_rule")
    e1 = record.get("e1_restore", {})
    e2 = record.get("e2_branch", {})
    e1_rms = max((row["rms_row_rel"] for row in e1.values()), default=math.nan)
    e1_worst = max((row["worst_row_rel"] for row in e1.values()), default=math.nan)
    e2_rms = max((row["rms_row_rel"] for row in e2.values()), default=math.nan)
    return {
        "e1_rms_row_rel_max": e1_rms, "e1_worst_row_rel_max": e1_worst,
        "e1_pass": e1_rms <= g4.RESTORE_RMS_BOUND and e1_worst <= g4.RESTORE_WORST_ROW_BOUND,
        "e2_rms_row_rel_max": e2_rms, "e2_pass": e2_rms <= g4.BRANCH_RMS_BOUND,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--k", type=int)
    args = parser.parse_args(argv)
    if args.arm is not None:
        record = run_child(args.arm, args.k, args.device, args.output)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return 0

    console = Console(width=110)
    args.output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {"device": args.device, "shape": list(SHAPE), "seed": SEED,
                               "ks": args.ks, "cases": []}
    for k in args.ks:
        records = {}
        for arm in ARMS:
            path = args.output / f"k{k}-{arm}.json"
            subprocess.run(
                [sys.executable, __file__, "--device", args.device, "--output", str(path),
                 "--arm", arm, "--k", str(k)],
                check=True, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            )
            records[arm] = json.loads(path.read_text(encoding="utf-8"))
        reference = records["bf16"]
        case: dict[str, Any] = {"k": k, "arms": {}}
        for arm, record in records.items():
            row: dict[str, Any] = {
                "status": record["status"], "routing": record.get("routing"),
                "window_loss": record.get("window_loss"),
                "store_host_bytes": record.get("store", {}).get("host_bytes"),
                "oracle_host_bytes": record.get("oracle", {}).get("host_reserved_bytes", 0),
                "step_rss_sampled_peak_bytes": record.get("step_rss_sampled_peak_bytes"),
                "process_rss_high_water_bytes": record.get("process_rss_high_water_bytes"),
                "cuda_allocated_peak_bytes": record.get("cuda_allocated_peak_bytes"),
                "error": record.get("error"),
            }
            if record["status"] == "COMPLETE":
                row.update(_boundary_gate(record))
                if arm != "bf16" and reference["status"] == "COMPLETE":
                    row["e3"] = _gradient_gate(
                        (args.output / f"k{k}-bf16.json").with_suffix(".grads.pt"),
                        (args.output / f"k{k}-{arm}.json").with_suffix(".grads.pt"),
                    )
                    row["loss_abs_diff"] = abs(record["window_loss"] - reference["window_loss"])
            case["arms"][arm] = row
            console.print(f"k={k} {arm}: {json.dumps(row, default=str)}")
        summary["cases"].append(case)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
