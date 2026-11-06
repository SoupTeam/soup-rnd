#!/usr/bin/env python3
"""Bounded P3 caller contract through HF Trainer and the public builder seam.

Run on the admitted experiment host, not during local development:
    python benchmarks/harness/p3_bf16_trainer_contract.py --device cuda --output caller.json

The native DeepSeek arm is internal only. The public DeepSeek refusal is tested, not
removed. A supported tiny Llama separately exercises the builder. Passing these caller
checks does not implement B1.4 automatic policy or establish BF16 numerical acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import tempfile
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from rich.console import Console

KS = (1, 4, 16, 64)
HOST_LIMIT = 128 * 1024 * 1024
FIXTURE_WORKSPACE = 32 * 1024 * 1024
MAX_LENGTH = 9


def _modules() -> tuple[Any, Any, Any]:
    try:
        from . import b13_moe_fixture, l2l_mistral_stand, p3_bf16_fixture
    except ImportError:
        import b13_moe_fixture
        import l2l_mistral_stand
        import p3_bf16_fixture
    return b13_moe_fixture, l2l_mistral_stand, p3_bf16_fixture


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _refusal(action: Callable[[], Any], *, text: Sequence[str]) -> dict[str, Any]:
    try:
        action()
    except (ValueError, MemoryError) as error:
        _require(any(fragment in str(error) for fragment in text), str(error))
        return {"status": "PASS", "expected_refusal": _error(error)}
    raise AssertionError("the negative contract was accepted")


def _case(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"status": "PASS", **action()}
    except Exception as error:
        return {"status": "FAIL", "error": _error(error), "traceback": traceback.format_exc(),
                "partial_evidence": getattr(error, "contract_evidence", None)}


def _clean_json(value: Any, nonfinite: list[str], path: str = "$") -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(path)
        return None
    if isinstance(value, dict):
        return {str(key): _clean_json(item, nonfinite, f"{path}.{key}")
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item, nonfinite, f"{path}[{index}]")
                for index, item in enumerate(value)]
    return value


def _plan(config: Any, budget: Any, k: int, device: str, *, dtype: str = "bfloat16",
          codec: str = "none", batch_size: int = 1) -> Any:
    from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

    return plan_moe_l2l(
        budget, n_layers=int(config.num_hidden_layers),
        n_moe_layers=int(config.num_hidden_layers - config.first_k_dense_replace),
        max_shape=(batch_size, MAX_LENGTH, int(config.hidden_size)),
        experts_per_token=int(config.num_experts_per_tok), requested_k=k,
        dtype=dtype, pin=device.startswith("cuda"), codec=codec,
        device_type=device.split(":")[0],
    )


def _admit(config: Any, device: str, ks: Sequence[int]) -> dict[str, Any]:
    """Plan before constructing weights or dataset tensors. Bound the tiny fixture only."""
    import psutil
    import torch

    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    _require(max(ks) <= 64, "this contract admits at most k=64")
    _require(config.hidden_size == 32 and config.num_hidden_layers == 3,
             "fixture dimensions changed; revise bounded admission before allocating")
    budget = HostMemoryBudget(HOST_LIMIT)
    plans = [_plan(config, budget, k, device) for k in ks]
    required = max(plan.host_bytes for plan in plans) + FIXTURE_WORKSPACE
    free = int(psutil.virtual_memory().available)
    _require(required <= HOST_LIMIT and required <= free // 2,
             "bounded fixture plus workspace exceeds admitted free host RAM")
    evidence: dict[str, Any] = {
        "status": "PASS", "host_limit_bytes": HOST_LIMIT, "available_host_bytes": free,
        "fixture_workspace_bound_bytes": FIXTURE_WORKSPACE,
        "required_host_bytes": required, "plans": [asdict(plan) for plan in plans],
        "scope": "tiny fixture tensors, not an RSS guarantee or full-model admission",
    }
    if device.startswith("cuda"):
        _require(torch.cuda.is_available(), "CUDA unavailable")
        _require(torch.cuda.is_bf16_supported(), "BF16 CUDA unavailable; no FP16 substitute")
        _require(torch.cuda.device_count() == 1,
                 "expose one CUDA device; the caller contract does not exercise DataParallel")
        free_device, total = torch.cuda.mem_get_info()
        _require(FIXTURE_WORKSPACE <= free_device // 2, "insufficient bounded device workspace")
        evidence["device"] = {
            "name": torch.cuda.get_device_name(), "free_bytes": free_device,
            "total_bytes": total, "workspace_bound_bytes": FIXTURE_WORKSPACE,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "native_bf16_supported": torch.cuda.is_bf16_supported(including_emulation=False),
            "bf16_including_emulation": torch.cuda.is_bf16_supported(),
        }
    return evidence


def _preweight_contract(config: Any, device: str) -> dict[str, Any]:
    """Run insufficient-budget refusals before constructing a fixture checkpoint."""
    from soup_cli.utils.c3_l2l_budget import plan_c3_l2l
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.partial_ram_source import PartialRamBudget

    tiny = HostMemoryBudget(1)
    refusal = _refusal(lambda: _plan(config, tiny, 4, device), text=("k=1 needs",))
    _require(tiny.used_bytes == 0, "failed preweight planner acquired leases")
    plan = _plan(config, HostMemoryBudget(HOST_LIMIT), 4, device)
    routing = plan.routing_record_bytes + plan.routing_comparison_bytes
    c3_refusal = _refusal(
        lambda: plan_c3_l2l(
            PartialRamBudget(4097, 4096, 32768), n_layers=plan.n_layers,
            batch_size=1, max_length=MAX_LENGTH, hidden_size=int(config.hidden_size),
            microbatches=plan.effective_k, dtype=plan.dtype, pin=plan.pin,
            shape_round=1, device_type=plan.device_type, routing_bytes=routing,
        ), text=("reservations",),
    )
    return {"status": "PASS", "insufficient_moe": refusal, "insufficient_c3": c3_refusal,
            "weights_constructed_before_check": False, "host_used_bytes": tiny.used_bytes}


def _rows(config: Any, k: int, batch_size: int = 1) -> list[dict[str, Any]]:
    import torch

    generator = torch.Generator().manual_seed(61)
    rows = []
    # First window is small, second grows each position, final one-row window reuses it.
    window_rows = k * batch_size
    for index in range(2 * window_rows + 1):
        length = (3, 4, 5)[index % 3] if index < window_rows else (9, 7, 8)[index % 3]
        if index == 2 * window_rows:
            length = 3
        ids = torch.randint(0, config.vocab_size, (length,), generator=generator)
        labels = ids.clone()
        labels[:1 + index % 2] = -100
        mask = torch.ones_like(ids)
        if length >= 7 and index % 2:
            ids[-1] = 0
            labels[-1] = -100
            mask[-1] = 0
        rows.append({"input_ids": ids, "labels": labels, "attention_mask": mask})
    return rows


def _collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad mixed lengths; the last micro-batch may also have fewer rows."""
    import torch

    width = max(row["input_ids"].numel() for row in rows)
    batch = {}
    for name, fill in (("input_ids", 0), ("labels", -100), ("attention_mask", 0)):
        value = torch.full((len(rows), width), fill, dtype=torch.long)
        for index, row in enumerate(rows):
            value[index, :row[name].numel()] = row[name]
        batch[name] = value
    return batch


def _restore(model: Any, values: dict[str, Any]) -> None:
    import torch

    fixture, _, _ = _modules()
    params = fixture.trainables(model)
    _require(params.keys() == values.keys(), "adapter names changed")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(values[name])
    model.zero_grad(set_to_none=True)


@contextlib.contextmanager
def _sdpa_observer() -> Any:
    import torch.nn.functional as functional

    original = functional.scaled_dot_product_attention
    seen: dict[str, Any] = {"calls": 0, "query_dtypes": []}

    def observed(query: Any, *args: Any, **kwargs: Any) -> Any:
        seen["calls"] += 1
        dtype = str(query.dtype)
        if dtype not in seen["query_dtypes"]:
            seen["query_dtypes"].append(dtype)
        return original(query, *args, **kwargs)

    functional.scaled_dot_product_attention = observed
    try:
        yield seen
    finally:
        functional.scaled_dot_product_attention = original


def _train(model: Any, runtime: Any, rows: Any, directory: Path, *, plan: Any,
           budget: Any, l2l: bool, builder: Any = None,
           batch_size: int = 1) -> tuple[Any, dict[str, Any], Any]:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    fixture, stand, _ = _modules()
    k = plan.effective_k
    before = fixture.snapshot(model)
    grads, rates, windows, stores, updated = [], [], [], [], []
    holders: dict[str, Any] = {}
    cleanup: dict[str, Any] = {}
    kept_stores: list[Any] = []
    observed_dtypes: set[str] = set()
    autocast_flags: set[bool] = set()
    scheduler_steps: list[int] = []
    ctor: list[dict[str, Any]] = []
    step_seconds: list[float] = []
    hooks = []

    def record_input(_module: Any, args: Any) -> None:
        if args and hasattr(args[0], "dtype"):
            observed_dtypes.add(str(args[0].dtype))
            autocast_flags.add(bool(torch.is_autocast_enabled(plan.device_type)))

    for module in model.modules():
        if type(module).__name__ in ("DeepseekV3DecoderLayer", "LlamaDecoderLayer"):
            hooks.append(module.register_forward_pre_hook(record_input))

    class OrderedTrainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            value = kwargs["args"].gradient_accumulation_steps
            ctor.append({"gradient_accumulation_steps": value, "effective_k": k})
            _require(value == k, "effective_k was not applied before trainer construction")
            super().__init__(*args, **kwargs)

        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            return torch.utils.data.SequentialSampler(
                self.train_dataset if train_dataset is None else train_dataset
            )

        def get_batch_samples(self, epoch_iterator: Any, num_batches: int, device: Any) -> Any:
            samples, n_items = super().get_batch_samples(epoch_iterator, num_batches, device)
            windows.append({"microbatches": len(samples), "label_tokens": int(n_items),
                            "shapes": [list(item["input_ids"].shape) for item in samples]})
            return samples, n_items

        def create_scheduler(self, num_training_steps: int, optimizer: Any = None) -> Any:
            scheduler_steps.append(num_training_steps)
            _require(self.args.gradient_accumulation_steps == k,
                     "scheduler saw unresolved accumulation")
            return super().create_scheduler(num_training_steps, optimizer)

    class Record(TrainerCallback):
        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if plan.device_type == "cuda":
                torch.cuda.synchronize()
            holders["step_start"] = time.perf_counter()

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any,
                                  model: Any = None, **kwargs: Any) -> None:
            values = fixture.snapshot(model, gradients=True)
            _require(values and all(value is not None for value in values.values()),
                     "missing adapter gradients")
            _require(all(bool(torch.isfinite(value).all()) for value in values.values()),
                     "nonfinite adapter gradient")
            _require(all(bool(value.ne(0).any()) for value in values.values()),
                     "an adapter gradient is entirely zero")
            grads.append(values)
            state_l2l = holders.get("state")
            if state_l2l is not None:
                store = state_l2l.store
                _require(store is not None, "no activation store during optimizer step")
                if not any(store is old for old in kept_stores):
                    kept_stores.append(store)
                stores.append({
                    "identity": next(i for i, old in enumerate(kept_stores) if old is store),
                    "k_capacity": store.k,
                    "position_capacities": [math.prod(shape) for shape in store.mb_shapes],
                    "dtype": str(store.dtype), "device": str(store.device),
                    "used_host_bytes": budget.used_bytes,
                    "store_class": type(store).__name__,
                    "allocation": (
                        store.allocation.as_dict() if hasattr(store, "allocation") else None
                    ),
                })

        def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None,
                        optimizer: Any = None, **kwargs: Any) -> None:
            if plan.device_type == "cuda":
                torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - holders["step_start"])
            rates.append(optimizer.param_groups[0]["lr"])
            values = fixture.snapshot(model)
            previous = before if not updated else holders["previous"]
            changed = [name for name in values if not torch.equal(previous[name], values[name])]
            _require(changed, "optimizer step changed no adapter")
            updated.append({"step": int(state.global_step), "changed_adapters": changed})
            holders["previous"] = values

    args = TrainingArguments(
        output_dir=str(directory), per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=k, learning_rate=0.01, optim="sgd",
        lr_scheduler_type="constant", num_train_epochs=1, logging_steps=1,
        save_strategy="no", report_to=[], use_cpu=plan.device_type == "cpu",
        seed=61, data_seed=61, max_grad_norm=0.0, disable_tqdm=True,
        bf16=False, fp16=False, dataloader_pin_memory=False,
    )
    kwargs = dict(model=model, args=args, train_dataset=rows, callbacks=[Record()],
                  data_collator=_collate)
    trainer, state_l2l = None, None
    start = time.perf_counter()
    try:
        if builder is not None:
            trainer = builder(OrderedTrainer, kwargs, args)
            state_l2l = trainer._soup_l2l
        else:
            cls = make_l2l_trainer_class(OrderedTrainer) if l2l else OrderedTrainer
            trainer = cls(**kwargs)
            if l2l:
                state_l2l = attach_l2l_state(
                    trainer, runtime, dtype=getattr(torch, plan.dtype), memory_plan=plan,
                    host_budget=budget, activation_codec=plan.codec,
                )
        holders["state"] = state_l2l
        if plan.device_type == "cuda":
            torch.cuda.synchronize()
        with _sdpa_observer() as sdpa:
            trainer.train()
        if plan.device_type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        expected = math.ceil(len(rows) / (k * batch_size))
        _require(trainer.state.global_step == len(grads) == len(updated) == expected,
                 "optimizer/window counts differ")
        _require(scheduler_steps == [expected], "scheduler was sized with the wrong k")
        expected_windows = []
        for offset in range(0, len(rows), k * batch_size):
            group = rows[offset:offset + k * batch_size]
            expected_windows.append((math.ceil(len(group) / batch_size), sum(
                int(row["labels"][1:].ne(-100).sum()) for row in group
            )))
        _require([(row["microbatches"], row["label_tokens"]) for row in windows]
                 == expected_windows, "HF label-token normalization/window sizes differ")
        if l2l:
            _require(state_l2l.steps == expected and state_l2l.window is None,
                     "L2L did not consume every window")
            _require(state_l2l.store is None and budget.used_bytes == 0,
                     "train exit retained an activation or routing lease")
            _require(stores[-1]["identity"] == stores[-2]["identity"],
                     "short final window replaced the larger store")
            _require(stores[1]["identity"] != stores[0]["identity"],
                     "larger second window failed to grow the store")
        _require(autocast_flags <= {False}, "unexpected autocast in native dtype arm")
        _require(observed_dtypes == {f"torch.{plan.dtype}"},
                 "decoder input dtype was unobserved or differs from the admitted native dtype")
        logs = [row for row in trainer.state.log_history if "loss" in row]
        record = stand.RunRecord(logs=logs, rates=rates, grads=grads,
                                 steps=int(trainer.state.global_step), l2l=l2l)
        evidence = {
            "status": "PASS", "constructor": ctor, "scheduler_training_steps": scheduler_steps,
            "windows": windows, "optimizer_updates": updated, "stores": stores,
            "steps": expected, "l2l_steps": state_l2l.steps if l2l else None,
            "gradient_finite_nonzero_every_adapter": True, "whole_train_seconds": seconds,
            "whole_optimizer_window_seconds": step_seconds,
            "timing_boundary": "HF on_step_begin through on_step_end, CUDA synchronized; "
                               "includes forward, backward and optimizer, not batch fetching",
            "precision": {"training_args_bf16": False, "training_args_fp16": False,
                          "decoder_input_dtypes": sorted(observed_dtypes),
                          "autocast_enabled_observed": sorted(autocast_flags),
                          "trainable_dtypes": sorted({str(p.dtype) for p in
                                                      fixture.trainables(model).values()})},
            "sdpa_observed": sdpa, "loss_logs": logs, "lr_per_step": rates,
            "cleanup": cleanup,
        }
        return record, evidence, fixture.snapshot(model)
    except Exception as error:
        error.contract_evidence = {
            "constructor": ctor, "scheduler_training_steps": scheduler_steps,
            "windows": windows, "stores": stores, "optimizer_updates": updated,
            "cleanup": cleanup,
        }
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if state_l2l is not None:
            state_l2l.close()
        cleanup.update({
            "host_used_bytes": budget.used_bytes,
            "store_closed": state_l2l.store is None if state_l2l is not None else True,
        })
        _require(budget.used_bytes == 0, "caller cleanup retained host leases")


def _gate_and_storage(fixture: Any, device: str) -> dict[str, Any]:
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for, attach_l2l_state
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils.c3_l2l_budget import plan_c3_l2l
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.partial_ram_source import PartialRamBudget

    budget = HostMemoryBudget(HOST_LIMIT)
    plan = _plan(fixture.config, budget, 4, device)
    refusals: dict[str, Any] = {}
    refusals["fp8_unsupported_compute_dtype"] = _refusal(
        lambda: L2LState(runtime=fixture.runtime, dtype=torch.float16,
                         activation_codec="fp8_e4m3"), text=("compute dtype",),
    )
    kwargs = {"runtime": fixture.runtime, "dtype": torch.bfloat16,
              "memory_plan": plan, "host_budget": budget}
    trainer = SimpleNamespace(args=SimpleNamespace(gradient_accumulation_steps=5))
    refusals["attach_hf_k"] = _refusal(
        lambda: attach_l2l_state(trainer, **kwargs), text=("effective_k",)
    )
    _require(not hasattr(trainer, "_soup_l2l"), "mismatched attachment mutated trainer")
    wrapper = _wrapper(fixture.streamed, fixture.runtime, device)
    # Uses the unchanged native model_type. The builder must refuse before construction.
    constructed = []

    class ConstructorWitness:
        def __init__(self, **options: Any) -> None:
            constructed.append(options)
            raise AssertionError("public DeepSeek reached trainer construction")

    refusals["public_deepseek"] = _refusal(
        lambda: StreamingSetupMixin._build_l2l_trainer(
            wrapper, ConstructorWitness, {},
            SimpleNamespace(n_gpu=1, fp16=False, gradient_accumulation_steps=4),
            memory_plan=plan, host_budget=budget,
        ), text=("deepseek_v3", "enabled for"),
    )
    _require(not constructed and budget.used_bytes == 0,
             "public gate ran constructor or acquired leases")
    routing = plan.routing_record_bytes + plan.routing_comparison_bytes
    original_weights = PartialRamBudget(HOST_LIMIT, 4096, 32768)
    c3_options = dict(
        n_layers=plan.n_layers, batch_size=1, max_length=MAX_LENGTH,
        hidden_size=fixture.config.hidden_size, microbatches=plan.effective_k,
        dtype=plan.dtype, pin=plan.pin, shape_round=1, device_type=plan.device_type,
    )
    missing = plan_c3_l2l(original_weights, **c3_options)
    refusals["combined_missing_routing"] = _refusal(
        lambda: L2LState(**{**kwargs, "host_budget": missing.activations}, c3_plan=missing),
        text=("routing_bytes",),
    )
    c3 = plan_c3_l2l(original_weights, routing_bytes=routing, **c3_options)
    _require(c3.weights.reserved_bytes == original_weights.reserved_bytes + plan.host_bytes,
             "C3 did not reserve activation and routing before optional weights")
    state = L2LState(runtime=fixture.runtime, dtype=torch.bfloat16, memory_plan=plan,
                     host_budget=c3.activations, c3_plan=c3)
    stepper = SimpleNamespace(causal=SimpleNamespace(config=fixture.config),
                              layers=[None] * plan.n_layers)
    ids = [torch.zeros((1, 3), dtype=torch.long)]
    allocation_budget = HostMemoryBudget(routing)
    allocation_state = L2LState(runtime=fixture.runtime, dtype=torch.bfloat16,
                                memory_plan=plan, host_budget=allocation_budget)
    try:
        refusals["constructor_budget_exhaustion"] = _refusal(
            lambda: _store_for(allocation_state, stepper, ids, device),
            text=("budget", "reserve", "available"),
        )
    finally:
        allocation_state.close()
    _require(allocation_budget.used_bytes == 0,
             "failed activation construction retained a routing lease")
    growth = []
    try:
        first = _store_for(state, stepper, ids, device)
        value = torch.arange(3 * fixture.config.hidden_size, dtype=torch.float32).reshape(
            1, 3, fixture.config.hidden_size
        ).to(device=device, dtype=torch.bfloat16)
        first.put(0, 0, value)
        _require(torch.equal(first.get(0, 0).reshape_as(value), value),
                 "BF16 store roundtrip changed data")
        usage = c3.activations.used_bytes
        # These requests run the allocator entry point, retaining an existing live store.
        actions = {
            "shape": lambda: _store_for(
                state, stepper, [torch.zeros((1, MAX_LENGTH + 1), dtype=torch.long)], device
            ),
            "window_k": lambda: _store_for(state, stepper, ids * 5, device),
        }
        for name, action in actions.items():
            refusals[name] = _refusal(action, text=("plan", "shape"))
            _require(state.store is first and c3.activations.used_bytes == usage,
                     f"{name} rejection replaced live storage")
        state.dtype = torch.float32
        refusals["dtype"] = _refusal(
            lambda: _store_for(state, stepper, ids, device), text=("plan",)
        )
        state.dtype = torch.bfloat16
        state.activation_codec = "fp8_e4m3"
        refusals["codec"] = _refusal(
            lambda: _store_for(state, stepper, ids, device), text=("plan",)
        )
        state.activation_codec = "none"
        # Change the bound plan, not the requested device, so no foreign device is touched.
        state.memory_plan = replace(plan, device_type="cpu" if plan.device_type == "cuda"
                                    else "cuda")
        refusals["device"] = _refusal(
            lambda: _store_for(state, stepper, ids, device), text=("device",)
        )
        state.memory_plan = plan
        _require(state.store is first and c3.activations.used_bytes == usage,
                 "metadata rejection changed store identity or lease")
        growth.append({"k": first.k, "used_bytes": usage})
        state.close_store()
        _require(c3.activations.used_bytes == routing,
                 "growth released protected routing headroom")
        larger = _store_for(state, stepper,
                            [torch.zeros((1, MAX_LENGTH), dtype=torch.long)] * 4, device)
        growth.append({"k": larger.k, "used_bytes": c3.activations.used_bytes})
        _require(c3.activations.used_bytes == c3.activations.limit_bytes,
                 "maximum BF16 store plus routing does not match C3 partition")
        _require(_store_for(state, stepper, ids, device) is larger,
                 "short window did not reuse maximum store")
    finally:
        state.close()
    _require(c3.activations.used_bytes == budget.used_bytes == 0,
             "combined storage cleanup leaked a lease")
    return {"refusals": refusals, "combined_c3": {
        "ordering": ["fixed consumers", "resolve MoE k against remaining envelope",
                     "reserve C3 activations plus routes", "optional weights may allocate"],
        "routing_bytes": routing, "activation_partition_bytes": c3.activations.limit_bytes,
        "weights_before": asdict(original_weights), "weights_after": asdict(c3.weights),
        "storage_growth": growth, "roundtrip_equal": True, "short_window_reused": True,
        "cleanup_used_bytes": c3.activations.used_bytes,
        "scope": "allocator behavioral contract; no MoE weight cache was built here",
    }}


def _wrapper(model: Any, runtime: Any, device: str, c3: Any = None,
             codec: str = "none") -> Any:
    return SimpleNamespace(
        model=model, device=device, _stream_runtime=runtime, _stream_weight_dirs=(),
        _c3_l2l_plan=c3,
        config=SimpleNamespace(task="sft", backend="transformers", modality="text",
                               training=SimpleNamespace(l2l=True, stream_layers=True,
                                                        l2l_activation_codec=codec,
                                                        l2l_spill_dir=None, multipack=False)),
    )


def _builder(root: Path, device: str, *, codec: str = "none",
             batch_size: int = 1) -> dict[str, Any]:
    """Supported dense Llama, separate from native internal MoE and its launch gate."""
    import torch
    from peft import LoraConfig, TaskType
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer

    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils.c3_l2l_budget import plan_c3_l2l
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import resolve_stream_dtype
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.partial_ram_source import PartialRamBudget

    fixture, _, _ = _modules()
    dtype = resolve_stream_dtype(device)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, attention_dropout=0,
        tie_word_embeddings=False, use_cache=False,
    )
    # plan_moe_l2l requires >=1 MoE layer. Reserve one as a conservative synthetic route
    # allowance for this dense builder check; this is not a Llama routing measurement.
    bounds = SimpleNamespace(num_hidden_layers=3, first_k_dense_replace=2, hidden_size=32,
                             num_experts_per_tok=2)
    fixed, limit = 4096, 8 * 1024 * 1024
    probe = _plan(bounds, HostMemoryBudget(limit, fixed), 2, device, dtype=dtype,
                  codec=codec, batch_size=batch_size)
    # A predeclared activation/route quota forces k4 -> k2. It is independent of the
    # larger C3 total envelope, which must also fit weights and loading transients.
    # This caller test does not claim automatic global-k optimization.
    envelope = fixed + probe.host_bytes
    resolved = _plan(bounds, HostMemoryBudget(envelope, fixed), 4, device, dtype=dtype,
                     codec=codec, batch_size=batch_size)
    _require(resolved.effective_k == 2, "bounded builder fixture did not resolve k4 to k2")
    routes = resolved.routing_record_bytes + resolved.routing_comparison_bytes
    c3 = plan_c3_l2l(
        PartialRamBudget(limit, fixed, 32768), n_layers=3, batch_size=batch_size,
        max_length=MAX_LENGTH, hidden_size=32, microbatches=resolved.effective_k,
        dtype=dtype, pin=device.startswith("cuda"), shape_round=1,
        device_type=device.split(":")[0], routing_bytes=routes, codec=codec,
    )
    root.mkdir(parents=True, exist_ok=True)
    source, shards = root / "weights", root / "shards"
    torch.manual_seed(61)
    base = LlamaForCausalLM(config).to(dtype=getattr(torch, dtype))
    base.save_pretrained(source, safe_serialization=True)
    del base
    index = shard_checkpoint(str(source), str(shards), dtype=dtype, arch="llama")
    model, runtime = build_streamed_model(
        model_id=str(source), shard_dir=str(shards), index=index,
        lora_config=LoraConfig(r=2, lora_alpha=4, lora_dropout=0, bias="none",
                               target_modules=["q_proj", "v_proj"],
                               task_type=TaskType.CAUSAL_LM),
        device=device, dtype=dtype, pin=device.startswith("cuda"), seed=67,
        quant="none", tier="disk", read_ahead=1, partial_ram_budget=c3.weights,
    )
    source_evidence: dict[str, Any] = {}
    try:
        fixture.randomize_adapters(model, seed=67)
        wrapper = _wrapper(model, runtime, device, c3, codec=codec)
        constructed: list[int] = []

        class WitnessTrainer(Trainer):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                constructed.append(kwargs["args"].gradient_accumulation_steps)
                super().__init__(*args, **kwargs)

        bad_args = SimpleNamespace(n_gpu=1, fp16=False, gradient_accumulation_steps=4)
        refusal = _refusal(
            lambda: StreamingSetupMixin._build_l2l_trainer(
                wrapper, WitnessTrainer, {}, bad_args,
                memory_plan=resolved, host_budget=c3.activations,
            ), text=("effective_k",),
        )
        _require(not constructed and bad_args.gradient_accumulation_steps == 4,
                 "builder constructed or silently rescheduled mismatched HF k")
        _require(c3.activations.used_bytes == 0, "builder mismatch acquired a lease")

        def builder(cls: Any, kwargs: Any, args: Any) -> Any:
            return StreamingSetupMixin._build_l2l_trainer(
                wrapper, cls, kwargs, args, memory_plan=resolved, host_budget=c3.activations,
            )

        _, evidence, _ = _train(
            model, runtime, _rows(config, 2, batch_size), root / "train",
            plan=resolved, budget=c3.activations, l2l=True, builder=builder,
            batch_size=batch_size,
        )
        if codec == "fp8_e4m3":
            _require(all(row["store_class"] == "FP8ActivationStore"
                         for row in evidence["stores"]), "builder did not use FP8 storage")
        _require(all(row["used_host_bytes"] <= resolved.host_bytes
                     for row in evidence["stores"]), "trainer exceeded resolved host envelope")
        source_evidence.update({
            "type": type(runtime.source).__name__,
            "resident_indices": list(runtime.source.resident_indices),
            "accounted_peak_bytes": runtime.source.accounted_peak_bytes,
            "weights_limit_bytes": c3.weights.limit_bytes,
        })
        _require(runtime.source.budget == c3.weights, "weight source ignored C3 partition")
        _require(runtime.source.accounted_peak_bytes <= c3.weights.limit_bytes,
                 "weight source exceeded C3 envelope")
        return {
            "scope": "supported tiny Llama builder, not public DeepSeek enablement",
            "route_allowance": "one synthetic MoE layer reserved conservatively for dense Llama",
            "activation_codec": codec, "maximum_batch_size": batch_size,
            "requested_k": 4, "effective_k": resolved.effective_k,
            "activation_route_quota_bytes": probe.host_bytes,
            "total_c3_host_envelope_bytes": limit,
            "quota_policy": "predeclared test quota; not a public automatic-k policy",
            "plan": asdict(resolved), "mismatched_hf_k": refusal,
            "training": evidence, "weight_source": source_evidence,
        }
    finally:
        runtime.close()
        _require(c3.activations.used_bytes == 0, "builder exit retained activation leases")
        source_budget = runtime.source.host_budget
        source_evidence["cleanup"] = {
            "used_bytes": source_budget.used_bytes,
            "fixed_reserved_bytes": source_budget.reserved_bytes,
            "active_lease_bytes": source_budget.used_bytes - source_budget.reserved_bytes,
            "prefetch_hook_removed": runtime.hook is None,
        }
        _require(source_budget.used_bytes == source_budget.reserved_bytes,
                 "builder exit retained a weight-source lease")


def _internal(fixture: Any, k: int, device: str, root: Path) -> dict[str, Any]:
    import torch

    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    helpers, stand, _ = _modules()
    initial = helpers.snapshot(fixture.streamed)
    rows = _rows(fixture.config, k)
    records, evidence, final = {}, {}, {}
    floor_error = None
    budget = HostMemoryBudget(HOST_LIMIT)
    plan = _plan(fixture.config, budget, k, device)
    _require(plan.effective_k == k, "internal contract unexpectedly reduced requested k")
    try:
        for name, model, runtime, l2l, backend in (
            ("ga1", fixture.resident, None, False, "MATH"),
            ("ga2", fixture.resident, None, False, "MATH"),
            ("l2l", fixture.streamed, fixture.runtime, True, "MATH"),
            ("ga_floor", fixture.resident, None, False, "EFFICIENT_ATTENTION"),
        ):
            _restore(model, initial)
            torch.manual_seed(61)
            try:
                with stand.pinned_sdpa(backend):
                    records[name], evidence[name], final[name] = _train(
                        model, runtime, rows, root / name, plan=plan, budget=budget, l2l=l2l,
                    )
                if name == "ga_floor" and evidence[name]["sdpa_observed"]["calls"] == 0:
                    floor_error = {"type": "NoExecutedSDPA", "message":
                                   "alternate SDPA backend did not execute SDPA calls"}
                    records.pop(name)
            except Exception as error:
                if name != "ga_floor":
                    raise
                floor_error = _error(error)
                evidence[name] = {"status": "BLOCKED", "error": floor_error}
        verdict = stand.decide(records["ga1"], records["ga2"], records["l2l"], mode="bf16",
                               floor=records.get("ga_floor"))
        # Keep update comparisons as measurements. A3, not finite gradients, decides numerics.
        update_errors = {
            name: float((final["ga1"][name].float() - final["l2l"][name].float()).abs().max())
            for name in initial
        }
        return {
            "scope": "internal native DeepSeek dense BF16 none, exact rank2 attention targets",
            "k": k, "plan": asdict(plan), "arms": evidence, "a3": verdict,
            "numeric_status": "PASS" if verdict["pass"] else (
                "BLOCKED" if verdict["status"] in ("no_floor", "invalid_floor") else "FAIL"
            ),
            "floor_error": floor_error,
            "update_max_abs_vs_resident": update_errors,
            "cleanup_used_bytes": budget.used_bytes,
        }
    finally:
        _restore(fixture.streamed, initial)
        _restore(fixture.resident, initial)
        _require(budget.used_bytes == 0, "internal run exit retained leases")


def _static_findings() -> dict[str, Any]:
    """Snapshot context, not a runtime assertion that public automatic-k integration exists."""
    import soup_cli

    package = Path(soup_cli.__file__).parent
    names = ("trainer/stream_setup.py", "trainer/l2l_trainer.py", "trainer/sft.py",
             "utils/moe_l2l_budget.py", "utils/c3_l2l_budget.py")
    return {
        "status": "BLOCKED", "public_automatic_policy_implemented_by_this_cli": False,
        "findings": [
            {"file": "src/soup_cli/trainer/sft.py", "symbol": "SFTTrainerWrapper.setup",
             "lines_at_review": "944-1085,1210",
             "finding": "training arguments use configured k; _build_l2l_trainer call passes "
                        "neither memory_plan nor host_budget"},
            {"file": "src/soup_cli/trainer/stream_setup.py",
             "symbol": "StreamingSetupMixin._setup_streaming_transformers",
             "lines_at_review": "1173-1181",
             "finding": "C3 reserves configured k and omits MoE routing_bytes before weights"},
            {"file": "src/soup_cli/trainer/stream_setup.py",
             "symbol": "StreamingSetupMixin._build_l2l_trainer",
             "lines_at_review": "1280-1339",
             "finding": "seam accepts a resolved plan, checks HF k, never selects k; "
                        "public launch gate still excludes DeepSeek"},
            {"file": "src/soup_cli/trainer/l2l_trainer.py", "symbol": "attach_l2l_state",
             "lines_at_review": "776-820",
             "finding": "attachment enforces pre-resolved effective_k, cannot resize scheduler"},
        ],
        "required_order": ["bound fixed consumers and maximum shape",
                           "resolve MoE k once from remaining host envelope",
                           "reserve C3 activation plus resolved routing before optional weights",
                           "construct HF args/trainer/scheduler with effective_k"],
        "circular_budget_warning": "Do not select k against a C3 partition already sized with "
                                   "requested k, or subtract resolved activations twice.",
        "pr_snapshot_observed_2026_11_05": {
            "52": {"url": "https://github.com/SoupTeam/soup-rnd/pull/52",
                   "state": "OPEN", "draft": True, "merge_state": "BLOCKED",
                   "scope": "documentation contract; says B1.4 wiring remains"},
            "53": {"url": "https://github.com/SoupTeam/soup-rnd/pull/53",
                   "state": "OPEN", "draft": False, "merge_state": "BLOCKED",
                   "updated_at": "2026-11-05T17:37:44Z",
                   "scope": "planner only; B1.4 consumes effective_k separately"},
            "open_pr_listing": "No B1.4 implementation PR appeared in the open PRs "
                               "returned at review time. Absence elsewhere was not established.",
        },
        "source_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                          for name in names},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    args = parser.parse_args(argv)
    try:
        ks = tuple(int(value) for value in args.ks.split(","))
    except ValueError:
        parser.error("--ks must be a comma-separated subset of 1,4,16,64")
    if not ks or len(set(ks)) != len(ks) or any(k not in KS for k in ks):
        parser.error("--ks must be a nonempty unique subset of 1,4,16,64")
    console = Console(stderr=True)
    report: dict[str, Any] = {
        "schema": "p3_bf16_trainer_contract_v1", "device": args.device, "ks": list(ks),
        "status": "BLOCKED", "blockers": [], "cases": {},
        "no_full_model_training": True, "weight_format": "dense",
        "fp8_training_acceptance": "not claimed; expected refusals are negative contracts only",
    }
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        helpers, stand, fixture_module = _modules()
        stand.set_deterministic_env()
        stand.enable_deterministic()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(2)
        report["static_b14_callsite"] = _static_findings()
        report["admission"] = _admit(helpers.moe_config(), args.device, ks)
        report["cases"]["preweight_admission"] = _preweight_contract(
            helpers.moe_config(), args.device
        )
        with tempfile.TemporaryDirectory(prefix="p3-bf16-caller-") as work:
            root = Path(work)
            fixture = fixture_module.build_bf16_fixture(
                root / "native", device=args.device, targets=helpers.TARGETS,
                weight_format="dense",
            )
            try:
                report["fixture_precision"] = fixture.precision_evidence
                _require(fixture.precision_evidence["rank"] == 2
                         and fixture.precision_evidence["targets"] == list(helpers.TARGETS),
                         "fixture did not preserve exact rank2 P3 attention targets")
                _require(len(helpers.trainables(fixture.streamed)) == 30,
                         "P3 fixture must train A/B for five projections in three layers")
                report["cases"]["gate_and_storage"] = _case(
                    lambda: _gate_and_storage(fixture, args.device)
                )
                for k in ks:
                    console.print(f"[cyan]Internal BF16 caller contract k={k}[/]")
                    report["cases"][f"native_k{k}"] = _case(
                        lambda k=k: _internal(fixture, k, args.device, root / f"k{k}")
                    )
            finally:
                fixture.close()
                report["fixture_cleanup"] = {
                    "close_returned": True,
                    "prefetch_hook_removed": fixture.runtime.hook is None,
                    "scope": "runtime closed; activation/routing leases recorded per arm",
                }
            report["cases"]["supported_builder"] = _case(
                lambda: _builder(root / "llama", args.device)
            )
        numeric = [row for name, row in report["cases"].items() if name.startswith("native_k")]
        behavior_ok = all(row["status"] == "PASS" for row in report["cases"].values())
        report["caller_behavior_status"] = "PASS" if behavior_ok else "FAIL"
        for row in numeric:
            if row.get("numeric_status") == "BLOCKED":
                report["blockers"].append({"k": row["k"], "reason": row["a3"]["status"]})
        report["blockers"].append({"scope": "public B1.4 automatic policy",
                                   "reason": "missing product caller, separate ownership"})
        numerical_failure = any(row.get("numeric_status") == "FAIL" for row in numeric)
        report["status"] = "FAIL" if not behavior_ok or numerical_failure else "BLOCKED"
    except Exception as error:
        report["fatal"] = {"error": _error(error), "traceback": traceback.format_exc()}
        report["blockers"].append(_error(error))
    nonfinite: list[str] = []
    report = _clean_json(report, nonfinite)
    if nonfinite:
        report["nonfinite_json_fields"] = nonfinite
        report["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    console.print(f"[{ 'green' if report['status'] == 'PASS' else 'yellow'}]"
                  f"Caller contract: {report['status']}[/] {args.output}")
    return 0 if report.get("caller_behavior_status") == "PASS" and report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
