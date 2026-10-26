"""CPU controls for constructor-isolated grouped-route qualification."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from benchmarks.harness import expert_cache_dispatch_qualification as qualify


def spec():
    return {
        **qualify.FIXED,
        "max_host_bytes": 32 * qualify.GIB,
        "lora_targets": ["q_proj", "kv_b_proj", "o_proj"],
    }


def common():
    return SimpleNamespace(RTOL=0.01, ATOL=1e-5)


def test_strict_fixed_recipe_accepts_only_original_ceiling():
    qualify.validate_protocol(spec())


@pytest.mark.parametrize(
    "key,value",
    [
        ("host_cache_bytes", 7 * 2**30),
        ("device_cache_bytes", 1 * 2**30),
        ("rounds", 2),
        ("warmup_updates", 1),
        ("timed_updates", 5),
        ("accuracy_updates", 1),
        ("accumulation", 1),
        ("host_cache_policy", "lru"),
        ("device_cap_bytes", 9 * 2**30),
        ("max_host_bytes", 33 * 2**30),
        ("dtype", "float32"),
        ("seq_len", 256),
        ("batch_size", True),
        ("lora_targets", ["q_proj", "expert_proj"]),
    ],
)
def test_recipe_changes_are_not_silent_fallbacks(key, value):
    changed = spec()
    changed[key] = value
    with pytest.raises(ValueError):
        qualify.validate_protocol(changed)


def test_constructor_missing_fails_before_model_and_retains_envelope(tmp_path, monkeypatch):
    import torch

    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec()))
    touched = []

    def refuse(*_args):
        with qualify.dispatch(SimpleNamespace(), "legacy"):
            touched.append("model")

    monkeypatch.setattr(qualify, "project", refuse)
    assert (
        qualify.worker_main(path, "grouped", "accuracy", tmp_path / "out", tmp_path, "a" * 40) == 1
    )
    envelope = json.loads((tmp_path / "out/dispatch-envelope.json").read_text())
    assert envelope["status"] == "failed" and "constructor is absent" in envelope["error"]
    assert not touched and not torch.cuda.is_initialized()


def constructor_module():
    def build(*, grouped_expert_routes=False):
        return SimpleNamespace(grouped=grouped_expert_routes), SimpleNamespace(close=lambda: None)

    return SimpleNamespace(build_streamed_model=build)


def dispatch_proof(enabled):
    flags = qualify.expected_routes(enabled)
    return {
        "checked_before_cache_admission": True,
        "expert_modules": 47,
        "actual_expert_grouped_routes": flags,
        "runtime_expert_grouped_routes": flags,
    }


def dispatch_evidence(variant):
    enabled = variant == "grouped"
    return {
        "mode": "grouped_constructor" if enabled else "legacy_constructor",
        "grouped_expert_routes": enabled,
        "factory_calls": 1,
        "factory_restored": True,
        "build_proofs": [dispatch_proof(enabled)],
    }


@pytest.mark.parametrize("variant", qualify.ACCURACY_VARIANTS)
def test_constructor_flag_is_set_before_admission_and_factory_is_restored(variant, monkeypatch):
    module = constructor_module()
    original = module.build_streamed_model
    seen = []

    def proof(model, runtime, enabled):
        assert model.grouped is enabled
        seen.append(enabled)
        return dispatch_proof(enabled)

    monkeypatch.setattr(qualify, "built_route_evidence", proof)
    with qualify.dispatch(module, variant) as evidence:
        model, _runtime = module.build_streamed_model()
        assert model.grouped is (variant == "grouped")
        assert seen == [variant == "grouped"]
    assert module.build_streamed_model is original
    qualify.validate_dispatch_evidence(evidence, variant)
    with pytest.raises(RuntimeError), qualify.dispatch(module, variant):
        raise RuntimeError("interrupted")
    assert module.build_streamed_model is original


def test_constructor_conflicts_and_changed_defaults_fail_closed(monkeypatch):
    module = constructor_module()
    with qualify.dispatch(module, "legacy"):
        with pytest.raises(ValueError, match="conflicting"):
            module.build_streamed_model(grouped_expert_routes=True)
    module = SimpleNamespace(build_streamed_model=lambda *, grouped_expert_routes=True: None)
    with pytest.raises(ValueError, match="without changing the default"):
        qualify._default_constructor_control(module)


def test_constructor_proof_failure_closes_runtime_and_restores_factory(monkeypatch):
    closed = []

    def build(*, grouped_expert_routes=False):
        return object(), SimpleNamespace(close=lambda: closed.append(True))

    module = SimpleNamespace(build_streamed_model=build)

    def refuse(*_args):
        raise ValueError("wrong actual modules")

    monkeypatch.setattr(qualify, "built_route_evidence", refuse)
    with pytest.raises(ValueError, match="actual modules"), qualify.dispatch(module, "grouped"):
        module.build_streamed_model()
    assert closed == [True] and module.build_streamed_model is build


@pytest.mark.parametrize("failure", ("invalid_commit", "head", "dirty"))
def test_source_mismatch_is_refused_before_project_imports(tmp_path, monkeypatch, failure):
    calls = []

    def output(args, **_kwargs):
        calls.append(args)
        if args[1] == "rev-parse":
            return ("b" if failure == "head" else "a") * 40
        return b" M src/changed.py" if failure == "dirty" else b""

    monkeypatch.setattr(qualify.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="commit|HEAD|dirty"):
        qualify.project(tmp_path, "short" if failure == "invalid_commit" else "a" * 40)
    assert len(calls) == {"invalid_commit": 0, "head": 1, "dirty": 2}[failure]


@pytest.mark.parametrize("enabled", (False, True))
def test_proof_inspects_actual_modules_and_runtime_instead_of_only_requested_flag(enabled):
    import torch

    from soup_cli.utils.expert_stream import streamed_experts_class

    cls = streamed_experts_class()
    model = torch.nn.ModuleList([
        cls(SimpleNamespace(count=128), object(), lambda value: value, index,
            two_reads=True, grouped_expert_routes=enabled)
        for index in range(1, 48)
    ])
    runtime = SimpleNamespace(stats=lambda: {
        "expert_grouped_routes": {
            str(module.layer_idx): module.grouped_expert_routes for module in model
        },
    })
    assert qualify.built_route_evidence(model, runtime, enabled) == dispatch_proof(enabled)
    model[0].grouped_expert_routes = not enabled
    with pytest.raises(ValueError, match="Actual expert modules"):
        qualify.built_route_evidence(model, runtime, enabled)


def cache_snapshot(cached, after):
    return {
        "expert_cache": {
            "logical_fetches": 100 if after else 0,
            "storage_reads": (40 if cached else 100) if after else 0,
            "storage_bytes_read": (4000 if cached else 10000) if after else 0,
            "materializations": (80 if cached else 100) if after else 0,
            "host_cache_policy": "static",
            "host": {
                "budget_bytes": 8 * 2**30 if cached else 0,
                "payload_bytes": 1000 if cached else 0,
                "hits": 40 if cached and after else 0,
            },
            "device": {
                "budget_bytes": 2 * 2**30 if cached else 0,
                "payload_bytes": 1000 if cached else 0,
                "hits": 20 if cached and after else 0,
            },
        }
    }


def timing_blocks():
    blocks = []
    for number, order in enumerate(qualify.ORDERS):
        for position, variant in enumerate(order):
            timing = {
                "complete": True,
                "updates_complete": 6,
                "finite_state_checks_per_update": True,
                "update_seconds": [1.0 if variant == "legacy" else 0.8] * 6,
                "losses": [1.2341] * 6,
                "runtime_before": cache_snapshot(True, False),
                "runtime_after": cache_snapshot(True, True),
                "cuda": {"peak_allocated_bytes": 100, "peak_reserved_bytes": 200},
                "resources": {"sampled_peak_rss_bytes": 300},
            }
            warmup = {
                **timing,
                "updates_complete": 2,
                "update_seconds": [1.0, 1.0],
                "losses": [1.2341] * 2,
            }
            for stage in (timing, warmup):
                for point in ("runtime_before", "runtime_after"):
                    stage[point]["expert_grouped_routes"] = qualify.expected_routes(
                        variant == "grouped"
                    )
            pid = 100 + number * 2 + position
            raw = {
                "arm": "two_read_cache",
                "status": "passed",
                "phase": "timing",
                "pid": pid,
                "spec_sha256": "a" * 64,
                "warmup": warmup,
                "timing": timing,
                "setup": {
                    "seconds": 1.0,
                    "runtime_after": {
                        "expert_grouped_routes": qualify.expected_routes(variant == "grouped")
                    },
                    "cache_plan": {
                        "host_cache_bytes": 8 * 2**30,
                        "device_cache_bytes": 2 * 2**30,
                        "host_cache_policy": "static",
                    },
                },
            }
            envelope = {
                "variant": variant,
                "native_arm": "two_read_cache",
                "phase": "timing",
                "status": "passed",
                "pid": pid,
                "spec_sha256": "a" * 64,
                "source_commit": "b" * 40,
                "expert_module_sha256": "c" * 64,
                "constructor_source_sha256": "d" * 64,
                "runtime_module_sha256": "f" * 64,
                "wrapper_sha256": "e" * 64,
                "dispatch": dispatch_evidence(variant),
            }
            blocks.append(
                {
                    "round": number,
                    "position": position,
                    "variant": variant,
                    "result": {
                        "variant": variant,
                        "native_arm": "two_read_cache",
                        "worker": raw,
                        "envelope": envelope,
                        "fresh_process_wall_seconds": 8.0,
                    },
                }
            )
    return blocks


def test_fixed_ab_ba_rounds_and_unrelabeled_native_arm():
    blocks = timing_blocks()
    result = qualify.timing_summary(blocks, common())
    assert result["step_time_reduction_percent"] == pytest.approx(20.0)
    assert result["paired_bootstrap_95_percent"] == pytest.approx([20.0, 20.0])
    assert result["passes_practical_2_percent_gate"] is True
    assert all(block["result"]["worker"]["arm"] == "two_read_cache" for block in blocks)
    assert len(result["recorded_loss_gates"]) == 8


@pytest.mark.parametrize(
    "corrupt",
    [
        "short",
        "order",
        "pid",
        "raw_arm",
        "mode",
        "source",
        "loss",
        "finite",
        "cache",
        "different_work",
        "runtime_flag",
        "factory_unused",
        "factory_not_restored",
        "module_proof",
    ],
)
def test_incomplete_or_mismatched_rounds_never_grant_speed(corrupt):
    blocks = timing_blocks()
    if corrupt == "short":
        blocks.pop()
    elif corrupt == "order":
        blocks[0]["position"] = 1
    elif corrupt == "pid":
        blocks[1]["result"]["worker"]["pid"] = blocks[0]["result"]["worker"]["pid"]
        blocks[1]["result"]["envelope"]["pid"] = blocks[0]["result"]["worker"]["pid"]
    elif corrupt == "raw_arm":
        blocks[0]["result"]["worker"]["arm"] = "legacy"
    elif corrupt == "mode":
        blocks[0]["result"]["envelope"]["dispatch"]["mode"] = "grouped_constructor"
    elif corrupt == "source":
        blocks[0]["result"]["envelope"]["source_commit"] = "f" * 40
    elif corrupt == "loss":
        blocks[0]["result"]["worker"]["timing"]["losses"][0] = 1.2351
    elif corrupt == "finite":
        blocks[0]["result"]["worker"]["timing"]["finite_state_checks_per_update"] = False
    elif corrupt == "cache":
        blocks[0]["result"]["worker"]["setup"]["cache_plan"]["host_cache_bytes"] -= 1
    elif corrupt == "runtime_flag":
        blocks[0]["result"]["worker"]["timing"]["runtime_after"]["expert_grouped_routes"] = (
            qualify.expected_routes(True)
        )
    elif corrupt == "factory_unused":
        blocks[0]["result"]["envelope"]["dispatch"]["factory_calls"] = 0
    elif corrupt == "factory_not_restored":
        blocks[0]["result"]["envelope"]["dispatch"]["factory_restored"] = False
    elif corrupt == "module_proof":
        proof = blocks[0]["result"]["envelope"]["dispatch"]["build_proofs"][0]
        proof["actual_expert_grouped_routes"] = qualify.expected_routes(True)
    else:
        cache = blocks[0]["result"]["worker"]["timing"]["runtime_after"]["expert_cache"]
        cache["storage_bytes_read"] += 10
    with pytest.raises(ValueError):
        qualify.timing_summary(blocks, common())


def test_negative_speed_is_retained_and_original_loss_gate_stays_reference_relative():
    blocks = timing_blocks()
    for block in blocks:
        if block["variant"] == "grouped":
            block["result"]["worker"]["timing"]["update_seconds"] = [1.1] * 6
    result = qualify.timing_summary(blocks, common())
    assert result["step_time_reduction_percent"] == pytest.approx(-10.0)
    assert result["passes_practical_2_percent_gate"] is False
    assert qualify.loss_gate([100.0001], [100.0002], common())["passed"]
    assert not qualify.loss_gate([1.2341], [1.2351], common())["passed"]
    assert not qualify.loss_gate([float("nan")], [1.0], common())["passed"]


def test_one_fast_round_cannot_pass_the_fixed_confidence_lower_bound():
    blocks = timing_blocks()
    for block in blocks:
        if block["variant"] == "grouped":
            seconds = 0.5 if block["round"] == 0 else 1.0
            block["result"]["worker"]["timing"]["update_seconds"] = [seconds] * 6
    result = qualify.timing_summary(blocks, common())
    assert result["step_time_reduction_percent"] > 0
    assert result["paired_bootstrap_95_percent"][0] == 0.0
    assert result["passes_practical_2_percent_gate"] is False


def test_missing_timed_first_microbatch_remains_explicitly_unmeasured():
    result = qualify.timing_summary(timing_blocks(), common())
    assert "NO VERDICT for unrecorded first timed microbatch loss." in result["recording_limits"]


def full_capture():
    import torch

    initial = {f"layer{index}.lora_A.weight": torch.tensor([1.0, 2.0]) for index in range(288)}
    batches = [
        {
            "input_ids": torch.ones((1, 512), dtype=torch.int64),
            "loss": torch.tensor(1.2341),
            "logits": torch.tensor([0.1, -0.2]),
            "gradients": {name: torch.ones_like(value) for name, value in initial.items()},
            "layer_inputs": {"layer0": torch.ones(1, 3, 2)},
            "router_logits": {
                "layer0": {
                    "values": torch.tensor([0.2, 0.3]),
                    "gradients": torch.tensor([0.01, 0.02]),
                }
            },
            "routing_weights": {
                "layer0": {
                    "values": torch.tensor([0.4, 0.6]),
                    "gradients": torch.tensor([0.03, 0.04]),
                    "indices": torch.tensor([0, 1]),
                }
            },
        }
        for _ in range(4)
    ]
    updates = [
        {
            "adapters": {name: value + step * 0.01 for name, value in initial.items()},
            "optimizer": {
                name: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.ones_like(value) * step * 0.1,
                    "exp_avg_sq": torch.ones_like(value) * step * 0.001,
                }
                for name, value in initial.items()
            },
        }
        for step in (1, 2)
    ]
    return {"initial_adapters": initial, "microbatches": batches, "updates": updates}


def native_common():
    from benchmarks.harness import two_read_cache_common

    return two_read_cache_common


def test_full_capture_requires_all_288_changed_and_differentiated_adapters():
    import torch

    capture = full_capture()
    inputs = torch.stack([row["input_ids"] for row in capture["microbatches"]])
    control = qualify.full_capture_controls(capture, inputs, native_common())
    assert control["adapters_nonzero_changed_and_nonzero_gradients"] == 288
    assert len(control["full_microbatch_losses"]) == 4
    different = copy.deepcopy(capture)
    different["microbatches"][0]["input_ids"][0, 0] += 1
    with pytest.raises(ValueError, match="input IDs"):
        qualify.full_capture_controls(different, inputs, native_common())
    for row in capture["microbatches"]:
        row["gradients"]["layer0.lora_A.weight"].zero_()
    with pytest.raises(ValueError, match="nonzero gradient"):
        qualify.full_capture_controls(capture, inputs, native_common())


def test_native_comparator_rejects_full_gradient_drift_without_gate_changes():
    reference = full_capture()
    candidate = copy.deepcopy(reference)
    candidate["microbatches"][0]["gradients"]["layer0.lora_A.weight"][0] += 0.1
    assert not native_common().compare_captures(reference, candidate)["passed"]


def mocked_run(tmp_path, monkeypatch, corruption=None):
    import torch

    inputs = torch.ones((4, 1, 512), dtype=torch.int64)
    data = tmp_path / "inputs.pt"
    torch.save(inputs, data)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({**spec(), "training_data_path": str(data)}))
    events, verifies = [], []
    native = SimpleNamespace(verify_spec=lambda _spec, content: verifies.append(content))
    module = constructor_module()
    monkeypatch.setattr(
        qualify,
        "project",
        lambda *_args: (native, None, native_common(), module, {"source_commit": "a" * 40}),
    )

    def fake_launch(spec_path, _spec, variant, phase, out, _source, _commit):
        events.append((phase, variant))
        out.mkdir(parents=True)
        if phase == "accuracy":
            capture = full_capture()
            if variant == "grouped":
                if corruption == "worker":
                    raise RuntimeError("native worker failed")
                if corruption == "gradient":
                    capture["microbatches"][0]["gradients"]["layer0.lora_A.weight"][0] += 0.1
                if corruption == "input":
                    capture["microbatches"][0]["input_ids"][0, 0] += 1
                if corruption == "short":
                    capture["updates"].pop()
                    capture["microbatches"] = capture["microbatches"][:2]
            torch.save(capture, out / "capture.pt")
            cached = variant != "ordinary_legacy"
            after = cache_snapshot(cached, True)
            if not cached:
                after["expert_cache"].update(
                    logical_fetches=150, storage_reads=150, materializations=150
                )
            raw = {
                "pid": 200 + qualify.ACCURACY_VARIANTS.index(variant),
                "arm": qualify.NATIVE_ARMS[variant],
                "setup": {
                    "cache_plan": {
                        "host_cache_bytes": 8 * 2**30,
                        "device_cache_bytes": 2 * 2**30,
                        "host_cache_policy": "static",
                    }
                },
                "accuracy": {
                    "capture_sha256": qualify.digest(out / "capture.pt"),
                    "runtime_before": cache_snapshot(cached, False),
                    "runtime_after": after,
                },
            }
            if variant == "grouped" and corruption == "capture_hash":
                raw["accuracy"]["capture_sha256"] = "0" * 64
            return {"worker": raw}
        assert events[:3] == [("accuracy", variant) for variant in qualify.ACCURACY_VARIANTS]
        assert corruption is None, "Timing started after an injected accuracy failure"
        number = int(out.parent.name.rsplit("-", 1)[-1])
        return copy.deepcopy(
            next(
                block["result"]
                for block in timing_blocks()
                if block["round"] == number and block["variant"] == variant
            )
        )

    monkeypatch.setattr(qualify, "launch", fake_launch)
    return path, events, verifies


def test_full_orchestration_finishes_all_accuracy_before_ab_ba_timing(tmp_path, monkeypatch):
    path, events, verifies = mocked_run(tmp_path, monkeypatch)
    report = qualify.run(path, tmp_path / "run", tmp_path, "a" * 40)
    assert report["status"] == "completed" and report["accuracy_passed"]
    assert events[:3] == [("accuracy", variant) for variant in qualify.ACCURACY_VARIANTS]
    assert events[3:] == [("timing", variant) for order in qualify.ORDERS for variant in order]
    assert len(report["warmup_versus_accuracy_loss_gates"]) == 8
    assert all(row["passed"] for row in report["warmup_versus_accuracy_loss_gates"])
    assert verifies == [True, True, True]


@pytest.mark.parametrize("corruption", ["gradient", "input", "short", "worker", "capture_hash"])
def test_accuracy_failure_preserves_report_and_never_launches_timing(
    tmp_path,
    monkeypatch,
    corruption,
):
    path, events, _verifies = mocked_run(tmp_path, monkeypatch, corruption)
    out = tmp_path / "run"
    with pytest.raises((ValueError, RuntimeError)):
        qualify.run(path, out, tmp_path, "a" * 40)
    report = json.loads((out / "run.json").read_text())
    assert report["status"] == "failed" and "error" in report
    assert report["timing_blocks"] == []
    assert all(phase == "accuracy" for phase, _variant in events)
    assert (out / "accuracy/ordinary_legacy/capture.pt").is_file()
