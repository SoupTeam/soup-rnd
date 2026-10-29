"""Scoped reference arithmetic; real PEFT, SYNTHETIC weights, CPU evidence only."""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _scale_fixture(kind: str, dtype: Any, scale: float, storage: str = "dense") -> Any:
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_act="silu")
            self.act_fn = nn.SiLU()
            shapes = {
                "single": {"o_proj": (10, 7)},
                "qkv": {"q_proj": (10, 7), "k_proj": (10, 6), "v_proj": (10, 4)},
                "mlp": {"gate_proj": (10, 14), "up_proj": (10, 14), "down_proj": (14, 7)},
            }[kind]
            self.targets = ("o_proj",) if kind == "single" else (
                ("q_proj",) if kind == "qkv" else ("down_proj",)
            )
            for name, (width, output) in shapes.items():
                if storage == "dense":
                    layer = nn.Linear(width, output, bias=False, dtype=dtype)
                else:
                    bnb = pytest.importorskip("bitsandbytes")
                    self.is_loaded_in_4bit = True
                    layer = bnb.nn.Linear4bit(
                        width, output, bias=False, compute_dtype=dtype,
                        compress_statistics=storage == "nf4-nested", quant_type="nf4",
                    )
                setattr(self, name, layer)

        def forward(self, inputs: Any) -> Any:
            if kind == "single":
                return self.o_proj(inputs)
            if kind == "qkv":
                return self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)
            return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

    with torch.random.fork_rng():
        torch.manual_seed(41)
        model = Block().to("cpu")
        inject_adapter_in_model(
            LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0, target_modules=list(model.targets)),
            model,
        )
        for name in model.targets:
            proj = getattr(model, name)
            proj.lora_A["default"].float()
            proj.lora_B["default"].float()
            proj.scaling["default"] = scale
            with torch.no_grad():
                proj.lora_B["default"].weight.normal_(std=0.2)
    return model


def _patch(model: Any, kind: str) -> None:
    from soup_cli.utils.fast_lora import patch_fast_lora_single_projection
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    patcher = {"single": patch_fast_lora_single_projection,
               "qkv": patch_fast_lora_qkv, "mlp": patch_fast_lora_mlp}[kind]
    assert patcher(model) == 1


@pytest.mark.parametrize("kind", ["single", "qkv", "mlp"])
@pytest.mark.parametrize("scale", [0.75, 1.3])
def test_scaling_precedes_adapter_backward_gemms(kind: str, scale: float) -> None:
    torch = pytest.importorskip("torch")
    reference = _scale_fixture(kind, torch.bfloat16, scale)
    fast = copy.deepcopy(reference)
    _patch(fast, kind)
    generator = torch.Generator().manual_seed(173)
    inputs = torch.randn(2, 3, 10, generator=generator).bfloat16()
    upstream = torch.randn(2, 3, 7, generator=generator).bfloat16()
    outputs = [model(inputs.clone()) for model in (reference, fast)]
    if kind == "qkv":
        outputs = [values[0] for values in outputs]
    assert type(outputs[1].grad_fn).__name__ == {
        "single": "_FastLoraSingleProjectionBackward", "qkv": "_FastLoraQKVBackward",
        "mlp": "_FastLoraSwiGLUBackward",
    }[kind]
    for value in outputs:
        value.backward(upstream)
    for name in reference.targets:
        for letter in ("A", "B"):
            ref = getattr(getattr(reference, name), f"lora_{letter}")["default"].weight.grad
            got = getattr(getattr(fast, name), f"lora_{letter}")["default"].weight.grad
            assert ref is not None and got is not None
            torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["single", "qkv", "mlp"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("storage", ["nf4", "nf4-nested"])
def test_nf4_scaled_update_casts_before_addition(
    kind: str, dtype_name: str, storage: str,
) -> None:
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    reference = _scale_fixture(kind, dtype, 1.3, storage)
    fast = copy.deepcopy(reference)
    _patch(fast, kind)
    for name in reference.targets:
        rw = getattr(reference, name).get_base_layer().weight
        fw = getattr(fast, name).get_base_layer().weight
        assert torch.equal(rw, fw)
        assert torch.equal(rw.quant_state.absmax, fw.quant_state.absmax)
        assert (rw.quant_state.state2 is not None) == (storage == "nf4-nested")
    generator = torch.Generator().manual_seed(173)
    inputs = torch.randn(2, 3, 10, generator=generator).to(dtype)
    expected, actual = reference(inputs), fast(inputs)
    if kind == "qkv":
        expected, actual = expected[0], actual[0]
    assert type(actual.grad_fn).__name__ == {
        "single": "_FastLoraSingleProjectionBackward", "qkv": "_FastLoraQKVBackward",
        "mlp": "_FastLoraSwiGLUBackward",
    }[kind]
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _llama_fixture(
    kind: str, dtype: Any, *, partial: bool = False, storage: str = "dense",
    parent_type: Any = None,
) -> Any:
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

    config = LlamaConfig(hidden_size=12, intermediate_size=18, num_attention_heads=3,
                         num_key_value_heads=1, hidden_act="silu", attention_dropout=0.0)
    config._attn_implementation = "eager"
    names = ("q_proj", "k_proj", "v_proj") if kind == "qkv" else (
        "gate_proj", "up_proj", "down_proj"
    )
    targets = (names[0], names[2]) if partial else names
    with torch.random.fork_rng():
        torch.manual_seed(41)
        cls = parent_type or (LlamaAttention if kind == "qkv" else LlamaMLP)
        model = cls(config, layer_idx=0) if kind == "qkv" else cls(config)
        model.to(dtype=dtype)
        if storage != "dense":
            bnb = pytest.importorskip("bitsandbytes")
            model.is_loaded_in_4bit = True
            for name in names:
                dense = getattr(model, name)
                layer = bnb.nn.Linear4bit(
                    dense.in_features, dense.out_features, bias=False, compute_dtype=dtype,
                    compress_statistics=storage == "nf4-nested", quant_type="nf4",
                )
                with torch.no_grad():
                    layer.weight.copy_(dense.weight)
                setattr(model, name, layer)
            model.to("cpu")
        inject_adapter_in_model(
            LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, target_modules=list(targets),
                       rank_pattern={names[0]: 1, names[1]: 3, names[2]: 2}), model,
        )
        for name, scale in zip(names, (0.75, 1.3, 2.0)):
            proj = getattr(model, name)
            if not hasattr(proj, "lora_A"):
                continue
            proj.lora_A["default"].float()
            proj.lora_B["default"].float()
            proj.scaling["default"] = scale
            with torch.no_grad():
                proj.lora_B["default"].weight.normal_(std=0.2)
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
        model.targets = targets
    return model


def _call_llama(model: Any, inputs: Any, kind: str) -> Any:
    if kind == "mlp":
        return model(inputs)
    cos = inputs.new_ones((*inputs.shape[:-1], model.head_dim))
    sin = inputs.new_zeros(cos.shape)
    return model(inputs, position_embeddings=(cos, sin), attention_mask=None)[0]


def _rounded_float64_llama(model: Any, kind: str, dtype: Any) -> Any:
    """Use precisely the reference's dequantized, compute-rounded NF4 weights."""
    import torch
    from torch import nn

    oracle = copy.deepcopy(model)
    names = ("q_proj", "k_proj", "v_proj") if kind == "qkv" else (
        "gate_proj", "up_proj", "down_proj"
    )
    for name in names:
        projection = getattr(model, name)
        base = projection.get_base_layer() if hasattr(projection, "lora_A") else projection
        quant = getattr(base.weight, "quant_state", None)
        if quant is None:
            continue
        from bitsandbytes.functional import dequantize_4bit

        weight = dequantize_4bit(base.weight, quant).to(dtype)
        replacement = nn.Linear(base.in_features, base.out_features, bias=False, dtype=dtype)
        with torch.no_grad():
            replacement.weight.copy_(weight)
        replacement.requires_grad_(False)
        destination = getattr(oracle, name)
        if hasattr(destination, "lora_A"):
            destination.base_layer = replacement
        else:
            setattr(oracle, name, replacement)
    return oracle.double()


def _assert_same_quantization(left: Any, right: Any) -> None:
    import torch

    assert torch.equal(left.weight, right.weight)
    def compare(a: Any, b: Any) -> None:
        assert (a is None) == (b is None)
        if a is None:
            return
        for field in ("shape", "dtype", "blocksize", "quant_type"):
            assert getattr(a, field) == getattr(b, field)
        for field in ("absmax", "code", "offset"):
            av, bv = getattr(a, field), getattr(b, field)
            assert (av is None) == (bv is None)
            if av is not None:
                assert torch.equal(av, bv)
        compare(a.state2, b.state2)
    compare(getattr(left.weight, "quant_state", None), getattr(right.weight, "quant_state", None))


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("needs_dx", [False, True])
@pytest.mark.parametrize("kind", ["qkv", "mlp"])
@pytest.mark.parametrize("storage", ["dense", "nf4", "nf4-nested"])
def test_standard_llama_uses_scoped_separate_reference_arithmetic(
    dtype_name: str, partial: bool, needs_dx: bool, kind: str, storage: str,
    record_property: Any,
) -> None:
    import json

    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    reference = _llama_fixture(kind, dtype, partial=partial, storage=storage)
    fast = copy.deepcopy(reference)
    oracle = _rounded_float64_llama(reference, kind, dtype)
    names = ("q_proj", "k_proj", "v_proj") if kind == "qkv" else (
        "gate_proj", "up_proj", "down_proj"
    )
    for name in names:
        left, right = getattr(reference, name), getattr(fast, name)
        left = left.get_base_layer() if hasattr(left, "lora_A") else left
        right = right.get_base_layer() if hasattr(right, "lora_A") else right
        _assert_same_quantization(left, right)
    _patch(fast, kind)
    generator = torch.Generator().manual_seed(173)
    rounded = torch.randn(2, 3, 12, generator=generator).to(dtype)
    xs = [rounded.clone().requires_grad_(needs_dx) for _ in range(2)]
    seen = {}
    handles = []
    observed_names = ("q_proj", "k_proj", "v_proj") if kind == "qkv" else ("Y",)
    for name in observed_names:
        def observe(_module: Any, _inputs: Any, value: Any, key: str = name) -> None:
            seen[key] = value
        module = getattr(fast, name) if kind == "qkv" else fast
        handles.append(module.register_forward_hook(observe))
    try:
        expected = _call_llama(reference, xs[0], kind)
        actual = _call_llama(fast, xs[1], kind)
    finally:
        for handle in handles:
            handle.remove()
    assert set(seen) == set(observed_names)
    for value in seen.values():
        node = "_FastLoraQKVBackward" if kind == "qkv" else "_FastLoraSwiGLUBackward"
        assert type(value.grad_fn).__name__ == node
        assert getattr(value.grad_fn, "reference_order", False), "reference scope not selected"
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn(actual.shape, generator=generator).to(dtype)
    expected.backward(upstream)
    actual.backward(upstream)
    high_x = rounded.double().requires_grad_(needs_dx)
    high_y = _call_llama(oracle, high_x, kind)
    high_y.backward(upstream.double())
    if needs_dx:
        torch.testing.assert_close(xs[1].grad, xs[0].grad, atol=0, rtol=0)
    else:
        assert xs[0].grad is None and xs[1].grad is None
    ref_params, fast_params, high_params = (
        dict(model.named_parameters()) for model in (reference, fast, oracle)
    )
    assert ref_params.keys() == fast_params.keys() == high_params.keys()
    required = {f"{name}.lora_{letter}.default.weight"
                for name in reference.targets for letter in ("A", "B")}
    assert {name for name in ref_params if "lora_" in name} == required
    quantities = {"Y": (actual, expected, high_y), "dX": (xs[1].grad, xs[0].grad, high_x.grad)}
    for (name, ref), (got_name, got) in zip(reference.named_parameters(), fast.named_parameters()):
        assert got_name == name
        if "lora_" in name:
            assert ref.grad is not None and got.grad is not None
            torch.testing.assert_close(got.grad, ref.grad, atol=0, rtol=0)
            quantities[name] = (got.grad, ref.grad, high_params[name].grad)
        else:
            assert ref.grad is None and got.grad is None
    evidence = []
    for name, (got, ref, high) in quantities.items():
        if ref is None:
            assert got is None and high is None and name == "dX" and not needs_dx
            evidence.append({"quantity": name, "contract": "None"})
            continue
        assert got.dtype == ref.dtype and got.shape == ref.shape == high.shape
        assert got.device == ref.device == high.device
        assert all(torch.isfinite(value).all() for value in (got, ref, high))
        fast_error = (got.double() - high).abs().max().item()
        peft_error = (ref.double() - high).abs().max().item()
        bound = 2 * peft_error + 1e-8
        evidence.append({"quantity": name, "bit_exact": torch.equal(got, ref),
                         "fast_float64_max_abs_error": fast_error,
                         "peft_float64_max_abs_error": peft_error, "bound": bound})
        assert fast_error <= bound, name
    record_property("rounded_oracle_quantities", json.dumps(evidence))


def _harness() -> Any:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "benchmarks/harness/fast_lora_probe.py"
    spec = importlib.util.spec_from_file_location("d2_reference_precision_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_mlp_remains_reference_after_normal_single_patcher_install() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

    model = _llama_fixture("mlp", torch.float16)
    _patch(model, "mlp")
    assert patch_fast_lora_single_projection(model) == 3
    result = model(torch.ones(2, 3, 12, dtype=torch.float16, requires_grad=True))
    assert type(result.grad_fn).__name__ == "_FastLoraSwiGLUBackward"
    assert getattr(result.grad_fn, "reference_order", False)


def test_unchanged_seed792_fp16_strict_50_step_cpu_gate(record_property: Any) -> None:
    import json

    report = _harness().run_loss(seed=792, steps=50, device="cpu", dtype="fp16")
    record_property("raw_loss_report", json.dumps(report, default=str))
    assert len(report["rows"]) == 50
    assert report["negative_control"]["unpatched_rejected"]
    assert report["passed"], report["failing_steps"]
    assert all(row["rounded_3_equal"] for row in report["rows"])


@pytest.mark.parametrize("kind", ["single", "qkv", "mlp"])
@pytest.mark.parametrize("later_replacement", [False, True])
def test_unpatch_restores_descriptor_without_clobbering_later_user_forward(
    kind: str, later_replacement: bool,
) -> None:
    import types

    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora import unpatch_fast_lora_single_projection
    from soup_cli.utils.fast_lora_mlp import unpatch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import unpatch_fast_lora_qkv

    model = (_scale_fixture("single", torch.float16, 1.3) if kind == "single"
             else _llama_fixture(kind, torch.float16))
    target = model.o_proj if kind == "single" else model
    before = target.forward
    had_instance = "forward" in vars(target)
    _patch(model, kind)
    if later_replacement:
        def replacement(self: Any, *args: Any, **kwargs: Any) -> Any:
            return before(*args, **kwargs)
        target.forward = types.MethodType(replacement, target)
        wanted = target.forward
    else:
        wanted = before
    unpatcher = {"single": unpatch_fast_lora_single_projection,
                 "qkv": unpatch_fast_lora_qkv, "mlp": unpatch_fast_lora_mlp}[kind]
    assert unpatcher(model) == 1
    assert target.forward == wanted
    if not later_replacement:
        assert ("forward" in vars(target)) == had_instance
    assert unpatcher(model) == 0


def _observed_scope(model: Any, inputs: Any, kind: str) -> bool:
    observed = []
    module = model.q_proj if kind == "qkv" else model
    def hook(_module: Any, _inputs: Any, value: Any) -> None:
        node = "_FastLoraQKVBackward" if kind == "qkv" else "_FastLoraSwiGLUBackward"
        assert type(value.grad_fn).__name__ == node, "unsupported fallback is not scope evidence"
        observed.append(bool(getattr(value.grad_fn, "reference_order", False)))
    handle = module.register_forward_hook(hook)
    try:
        _call_llama(model, inputs, kind)
    finally:
        handle.remove()
    assert len(observed) == 1
    return observed[0]


def test_concurrent_cold_initialization_publishes_complete_reference_metadata(
    monkeypatch: Any,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    torch = pytest.importorskip("torch")
    from peft.tuners.tuners_utils import BaseTunerLayer

    from soup_cli.utils import fast_lora

    models = [_llama_fixture("qkv", torch.float16) for _ in range(2)]
    for model in models:
        _patch(model, "qkv")
    inputs = torch.ones(2, 3, 12, dtype=torch.float16)
    paused = threading.Event()
    release = threading.Event()
    initializer_thread = None
    verify = fast_lora._verified_class_forward

    def pause_cast_verification(cls: type, method_name: str = "forward") -> Any:
        if (cls is BaseTunerLayer and method_name == "_cast_input_dtype"
                and threading.get_ident() == initializer_thread):
            paused.set()
            assert release.wait(timeout=30), "cold initializer was not released"
        return verify(cls, method_name)

    def initialize_through_canonical_attention() -> bool:
        nonlocal initializer_thread
        initializer_thread = threading.get_ident()
        return _observed_scope(models[0], inputs, "qkv")

    monkeypatch.setattr(fast_lora, "_REFERENCE_DEFINITIONS", None)
    monkeypatch.setattr(fast_lora, "_REFERENCE_CAST_METHOD", None)
    monkeypatch.setattr(fast_lora, "_verified_class_forward", pause_cast_verification)
    assert fast_lora._REFERENCE_DEFINITIONS is None
    assert fast_lora._REFERENCE_CAST_METHOD is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        initializer = pool.submit(initialize_through_canonical_attention)
        try:
            assert paused.wait(timeout=30), "did not reach cold cast verification"
            peer = pool.submit(_observed_scope, models[1], inputs, "qkv")
            # The peer must execute real HF/PEFT while the first call is paused,
            # not a warmed getter or a timing-dependent concurrency stress test.
            assert peer.result(timeout=30)
            assert not release.is_set()
        finally:
            release.set()
        assert initializer.result(timeout=30)
    assert fast_lora._REFERENCE_DEFINITIONS is not None
    assert fast_lora._REFERENCE_CAST_METHOD[0] is BaseTunerLayer._cast_input_dtype


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
def test_scope_rejects_in_place_rewrite_of_trusted_forward_code(
    kind: str, monkeypatch: Any,
) -> None:
    import types

    torch = pytest.importorskip("torch")
    model = _llama_fixture(kind, torch.float16)
    original = type(model).forward
    pristine = types.FunctionType(original.__code__, original.__globals__,
                                  argdefs=original.__defaults__)
    pristine.__kwdefaults__ = original.__kwdefaults__
    model._test_pristine_forward = types.MethodType(pristine, model)
    _patch(model, kind)
    def rewritten(self: Any, *args: Any, **kwargs: Any) -> Any:
        return self._test_pristine_forward(*args, **kwargs)
    monkeypatch.setattr(original, "__code__", rewritten.__code__)
    assert not _observed_scope(model, torch.ones(2, 3, 12, dtype=torch.float16), kind)


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
def test_scope_rejects_global_inner_execution_hooks(kind: str) -> None:
    torch = pytest.importorskip("torch")
    from torch.nn.modules.module import register_module_forward_pre_hook

    model = _llama_fixture(kind, torch.float16)
    _patch(model, kind)
    handle = register_module_forward_pre_hook(lambda _module, _inputs: None)
    try:
        assert not _observed_scope(model, torch.ones(2, 3, 12, dtype=torch.float16), kind)
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
def test_scope_rejects_custom_peft_input_cast(kind: str, monkeypatch: Any) -> None:
    import types

    torch = pytest.importorskip("torch")
    model = _llama_fixture(kind, torch.float16)
    _patch(model, kind)
    proj = model.q_proj if kind == "qkv" else model.gate_proj
    def custom_cast(self: Any, value: Any, dtype: Any) -> Any:
        return value.to(dtype) * 2
    monkeypatch.setattr(proj, "_cast_input_dtype", types.MethodType(custom_cast, proj))
    assert not _observed_scope(model, torch.ones(2, 3, 12, dtype=torch.float16), kind)


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
def test_foreign_bound_peft_cast_method_is_not_reference_eligible(
    kind: str, monkeypatch: Any,
) -> None:
    torch = pytest.importorskip("torch")
    reference = _llama_fixture(kind, torch.float16)
    fast = copy.deepcopy(reference)
    donor = _llama_fixture(kind, torch.float16)
    name = "q_proj" if kind == "qkv" else "gate_proj"
    other = getattr(donor, name)
    other.cast_input_dtype_enabled = False
    for model in (reference, fast):
        proj = getattr(model, name)
        monkeypatch.setattr(proj, "cast_input_dtype_enabled", True)
        monkeypatch.setattr(proj, "_cast_input_dtype", other._cast_input_dtype)
        assert proj._cast_input_dtype.__func__ is type(proj)._cast_input_dtype
        assert proj._cast_input_dtype.__self__ is other and other is not proj
        assert proj.cast_input_dtype_enabled and not other.cast_input_dtype_enabled
    inputs = torch.ones(2, 3, 12, dtype=torch.float16)
    # Installed PEFT reads the foreign receiver's disabled-cast flag, leaving
    # fp16 inputs incompatible with this projection's fp32 adapter masters.
    with pytest.raises(RuntimeError, match="same dtype"):
        _call_llama(reference, inputs, kind)
    _patch(fast, kind)
    # Unsupported casts retain the legacy API contract; a genuine grouped
    # Function may execute, but must never claim scoped reference arithmetic.
    assert not _observed_scope(fast, inputs, kind)
    assert torch.isfinite(_call_llama(fast, inputs, kind)).all()


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
@pytest.mark.parametrize("change", ["subclass", "parent-before", "parent-after",
                                    "class-before", "class-after", "adapter-forward",
                                    "projection-forward", "inner-hook", "fp32", "low-adapters"])
def test_reference_scope_is_not_selected_for_nonstandard_execution(
    kind: str, change: str, monkeypatch: Any,
) -> None:
    import functools
    import types

    torch = pytest.importorskip("torch")
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

    from soup_cli.utils import fast_lora

    cls = LlamaAttention if kind == "qkv" else LlamaMLP
    if change == "subclass":
        class Subclass(cls):
            pass
        cls = Subclass
    dtype = torch.float32 if change == "fp32" else torch.float16
    model = _llama_fixture(kind, dtype, parent_type=cls)
    original = model.forward
    @functools.wraps(original.__func__)
    def alternate(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)
    if change == "parent-before":
        model.forward = types.MethodType(alternate, model)
    if change == "class-before":
        monkeypatch.setattr(cls, "forward", alternate)
        # A first-use guard must not bless a replacement with copied metadata.
        monkeypatch.setattr(fast_lora, "_REFERENCE_DEFINITIONS", None)
    _patch(model, kind)
    if change == "parent-after":
        # Use a saved original rather than calling the installed scope wrapper.
        model.forward = types.MethodType(alternate, model)
    if change == "class-after":
        monkeypatch.setattr(cls, "forward", alternate)
    projection = model.k_proj if kind == "qkv" else model.up_proj
    if change in ("adapter-forward", "projection-forward"):
        target = projection.lora_A["default"] if change == "adapter-forward" else projection
        old = target.forward
        def changed_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            return old(*args, **kwargs)
        target.forward = types.MethodType(changed_forward, target)
    handle = None
    if change == "inner-hook":
        handle = projection.lora_A["default"].register_forward_hook(lambda *_args: None)
    if change == "low-adapters":
        for module in model.modules():
            if hasattr(module, "lora_A"):
                module.lora_A["default"].to(dtype=dtype)
                module.lora_B["default"].to(dtype=dtype)
    try:
        if change == "parent-after" and kind == "mlp":
            # User deliberately replaced the installed parent wrapper. This is
            # an explicit unsupported PEFT path, NOT positive custom evidence.
            value = _call_llama(model, torch.ones(2, 3, 12, dtype=dtype), kind)
            assert type(value.grad_fn).__name__ != "_FastLoraSwiGLUBackward"
            assert not getattr(value.grad_fn, "reference_order", False)
        else:
            assert not _observed_scope(model, torch.ones(2, 3, 12, dtype=dtype), kind)
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("order", [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
                                   (2, 0, 1), (2, 1, 0)])
def test_standalone_qkv_evaluation_orders_do_not_claim_reference_scope(order: tuple[int, ...]):
    torch = pytest.importorskip("torch")
    model = _llama_fixture("qkv", torch.float16)
    _patch(model, "qkv")
    inputs = torch.ones(2, 3, 12, dtype=torch.float16, requires_grad=True)
    projections = [model.q_proj, model.k_proj, model.v_proj]
    values = [projections[index](inputs) for index in order]
    assert any(type(value.grad_fn).__name__ == "_FastLoraQKVBackward" for value in values)
    assert all(not getattr(value.grad_fn, "reference_order", False) for value in values)


@pytest.mark.parametrize("roots", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2),
                                   (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
                                   (2, 0, 1), (2, 1, 0)])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("partial", [False, True])
def test_scoped_qkv_root_order_unused_grads_and_adamw(
    roots: tuple[int, ...], dtype_name: str, partial: bool,
) -> None:
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    reference = _llama_fixture("qkv", dtype, partial=partial)
    fast = copy.deepcopy(reference)
    _patch(fast, "qkv")
    generator = torch.Generator().manual_seed(173)
    rounded = torch.randn(2, 3, 12, generator=generator).to(dtype)
    xs = [rounded.clone().requires_grad_(True) for _ in range(2)]
    branches = []
    names = ("q_proj", "k_proj", "v_proj")
    for model, inputs in zip((reference, fast), xs):
        seen = {}
        handles = []
        for name in names:
            def hook(_module: Any, _inputs: Any, value: Any, key: str = name) -> None:
                seen[key] = value
            handles.append(getattr(model, name).register_forward_hook(hook))
        try:
            _call_llama(model, inputs, "qkv")
        finally:
            for handle in handles:
                handle.remove()
        branches.append(tuple(seen[name] for name in names))
    assert all(value.grad_fn.reference_order for value in branches[1])
    upstream = [torch.randn(value.shape, generator=generator).to(dtype) for value in branches[0]]
    initial = {name: p.detach().clone() for name, p in fast.named_parameters()}
    optimizers = [torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=0.003, weight_decay=0.1, foreach=False)
                  for model in (reference, fast)]
    for values in branches:
        torch.autograd.backward(tuple(values[i] for i in roots), tuple(upstream[i] for i in roots))
    torch.testing.assert_close(xs[1].grad, xs[0].grad, atol=0, rtol=0)
    ref_params = dict(reference.named_parameters())
    for name, parameter in fast.named_parameters():
        ref_grad = ref_params[name].grad
        assert (parameter.grad is None) == (ref_grad is None), name
        if ref_grad is not None:
            torch.testing.assert_close(parameter.grad, ref_grad, atol=0, rtol=0)
    for optimizer in optimizers:
        optimizer.step()
    for name, parameter in fast.named_parameters():
        torch.testing.assert_close(parameter, ref_params[name], atol=0, rtol=0)
        if "lora_" in name and not any(name.startswith(names[i]) for i in roots):
            assert parameter.grad is None and parameter not in optimizers[1].state
            torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)


def test_scope_cache_does_not_escape_parent_or_survive_exception() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import _CALL

    model = _llama_fixture("qkv", torch.float16)
    _patch(model, "qkv")
    inputs = torch.ones(2, 3, 12, dtype=torch.float16, requires_grad=True)
    assert _observed_scope(model, inputs, "qkv")
    outside = model.k_proj(inputs)
    assert type(outside.grad_fn).__name__ != "_FastLoraQKVBackward"
    assert _CALL.get() is None
    # The real HF forward fails only after all projection calls without rotary data.
    with pytest.raises(TypeError):
        model(inputs, position_embeddings=None, attention_mask=None)
    assert _CALL.get() is None
    assert getattr(model, "_soup_fast_lora_qkv_cache") is None
    outside = model.q_proj(inputs)
    assert not outside.grad_fn.reference_order


def test_scope_and_sibling_cache_mode_changes_cannot_reuse_outputs() -> None:
    torch = pytest.importorskip("torch")
    model = _llama_fixture("qkv", torch.float16)
    _patch(model, "qkv")
    inputs = torch.ones(2, 3, 12, dtype=torch.float16)
    assert not model.q_proj(inputs).grad_fn.reference_order
    assert _observed_scope(model, inputs, "qkv")
    assert type(model.k_proj(inputs).grad_fn).__name__ != "_FastLoraQKVBackward"
    seen = []
    def change_mode(_module: Any, _inputs: Any, value: Any) -> None:
        seen.append(value.grad_fn.reference_order)
        model.k_proj.lora_A["default"].register_forward_pre_hook(lambda *_args: None)
    handle = model.q_proj.register_forward_hook(change_mode)
    try:
        _call_llama(model, inputs, "qkv")
        assert seen == [True]
        assert getattr(model, "_soup_fast_lora_qkv_last_cache_hits")[0] == 0
    finally:
        handle.remove()
        model.k_proj.lora_A["default"]._forward_pre_hooks.clear()


def test_reference_scope_is_thread_local_and_reentrant() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import _CALL

    model = _llama_fixture("qkv", torch.float16)
    _patch(model, "qkv")
    barrier = threading.Barrier(2, timeout=30)
    observed = []
    def wait_for_peer(_module: Any, _inputs: Any, value: Any) -> None:
        observed.append(value.grad_fn.reference_order)
        barrier.wait()
    handle = model.q_proj.register_forward_hook(wait_for_peer)
    inputs = [torch.ones(2, 3, 12, dtype=torch.float16) * value for value in (1, 2)]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(lambda x: _call_llama(model, x, "qkv"), inputs))
    finally:
        handle.remove()
    assert observed == [True, True] and all(value.isfinite().all() for value in values)
    assert _CALL.get() is None
    nested = []
    active = False
    def recurse(_module: Any, _inputs: Any, value: Any) -> None:
        nonlocal active
        assert value.grad_fn.reference_order
        if not active:
            active = True
            nested.append(_call_llama(model, inputs[1], "qkv"))
            active = False
    handle = model.q_proj.register_forward_hook(recurse)
    try:
        result = _call_llama(model, inputs[0], "qkv")
    finally:
        handle.remove()
    assert len(nested) == 1 and _CALL.get() is None
    torch.testing.assert_close(result, values[0], atol=0, rtol=0)
    torch.testing.assert_close(nested[0], values[1], atol=0, rtol=0)


def test_reference_autograd_hook_does_not_retain_parent_module() -> None:
    import gc
    import weakref

    torch = pytest.importorskip("torch")
    model = _llama_fixture("qkv", torch.float16)
    _patch(model, "qkv")
    inputs = torch.ones(2, 3, 12, dtype=torch.float16, requires_grad=True)
    value = _call_llama(model, inputs, "qkv")
    ref = weakref.ref(model)
    del model
    gc.collect()
    assert ref() is None
    value.sum().backward()
    assert inputs.grad is not None


def test_qkv_unpatch_keeps_user_child_wrapper_even_with_copied_owner_metadata() -> None:
    import functools
    import types

    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import unpatch_fast_lora_qkv

    model = _llama_fixture("qkv", torch.float16)
    _patch(model, "qkv")
    installed = model.q_proj.forward
    @functools.wraps(installed.__func__)
    def user_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        return installed(*args, **kwargs)
    model.q_proj.forward = types.MethodType(user_forward, model.q_proj)
    wanted = model.q_proj.forward
    assert unpatch_fast_lora_qkv(model) == 1
    assert model.q_proj.forward == wanted


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_reference_nf4_frozen_input_skips_prefix_base_backward_and_unpacks_once(
    kind: str, checkpointed: bool, monkeypatch: Any,
) -> None:
    import sys

    torch = pytest.importorskip("torch")
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils import fast_lora_mlp, fast_lora_qkv

    model = _llama_fixture(kind, torch.float16, storage="nf4-nested")
    _patch(model, kind)
    module = fast_lora_qkv if kind == "qkv" else fast_lora_mlp
    original_dense = module._dense_weight
    calls = []
    def dense(*args: Any, **kwargs: Any) -> Any:
        calls.append(sys._getframe(1).f_code.co_name)
        return original_dense(*args, **kwargs)
    monkeypatch.setattr(module, "_dense_weight", dense)
    tokens = []
    def pack(value: Any) -> Any:
        token = [value, 0]
        tokens.append(token)
        return token
    def unpack(token: Any) -> Any:
        token[1] += 1
        assert token[1] == 1, "second saved tensor unpack"
        return token[0]
    inputs = torch.ones(2, 3, 12, dtype=torch.float16)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        if checkpointed:
            value = checkpoint(lambda x: _call_llama(model, x, kind), inputs, use_reentrant=False)
        else:
            value = _call_llama(model, inputs, kind)
        value.sum().backward()
    assert inputs.grad is None
    assert calls.count("backward") == (0 if kind == "qkv" else 1)
    assert all(token[1] <= 1 for token in tokens)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
def test_reference_scope_requires_present_fp32_adapters(kind: str) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora import _projection_state, _reference_projections

    model = _llama_fixture(kind, torch.float16)
    names = ("q_proj", "k_proj", "v_proj") if kind == "qkv" else (
        "gate_proj", "up_proj", "down_proj"
    )
    projections = [getattr(model, name).get_base_layer() for name in names]
    inputs = torch.ones(2, 3, 12, dtype=torch.float16)
    states = [_projection_state(proj, inputs, allow_unadapted=True) for proj in projections]
    assert all(not state.lora_a.numel() for state in states)
    assert not _reference_projections(projections, [proj.forward for proj in projections],
                                      states, inputs)


@pytest.mark.parametrize("layout", ["down-only", "gate-up-only"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("storage", ["dense", "nf4-nested"])
def test_standard_mlp_partial_adapter_layouts_keep_reference_arithmetic(
    layout: str, dtype_name: str, storage: str,
) -> None:
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    reference = _llama_fixture("mlp", dtype, storage=storage)
    absent = ("gate_proj", "up_proj") if layout == "down-only" else ("down_proj",)
    for name in absent:
        setattr(reference, name, getattr(reference, name).get_base_layer())
    fast = copy.deepcopy(reference)
    _patch(fast, "mlp")
    generator = torch.Generator().manual_seed(173)
    rounded = torch.randn(2, 3, 12, generator=generator).to(dtype)
    xs = [rounded.clone().requires_grad_(True) for _ in range(2)]
    expected, actual = reference(xs[0]), fast(xs[1])
    assert type(actual.grad_fn).__name__ == "_FastLoraSwiGLUBackward"
    assert actual.grad_fn.reference_order
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn(actual.shape, generator=generator).to(dtype)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(xs[1].grad, xs[0].grad, atol=0, rtol=0)
    ref_params, fast_params = dict(reference.named_parameters()), dict(fast.named_parameters())
    assert ref_params.keys() == fast_params.keys()
    for name in ref_params:
        rg, fg = ref_params[name].grad, fast_params[name].grad
        if "lora_" in name:
            assert rg is not None and fg is not None
            torch.testing.assert_close(fg, rg, atol=0, rtol=0)
        else:
            assert rg is None and fg is None


@pytest.mark.parametrize("kind", ["qkv", "mlp"])
def test_first_use_rejects_forward_clone_with_copied_code_and_metadata(
    kind: str, monkeypatch: Any,
) -> None:
    import types

    torch = pytest.importorskip("torch")
    from soup_cli.utils import fast_lora

    model = _llama_fixture(kind, torch.float16)
    original = type(model).forward
    clone = types.FunctionType(original.__code__, dict(original.__globals__),
                               name=original.__name__, argdefs=original.__defaults__)
    clone.__module__, clone.__qualname__ = original.__module__, original.__qualname__
    clone.__kwdefaults__ = original.__kwdefaults__
    monkeypatch.setattr(type(model), "forward", clone)
    monkeypatch.setattr(fast_lora, "_REFERENCE_DEFINITIONS", None)
    _patch(model, kind)
    assert not _observed_scope(model, torch.ones(2, 3, 12, dtype=torch.float16), kind)
