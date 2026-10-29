"""D2 mixed base/adapter backward cast boundaries; SYNTHETIC CPU fixtures."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _model(kind: str, dtype: Any, *, quantized: bool = False, compressed: bool = False) -> Any:
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_act="silu")
            self.act_fn = nn.SiLU()
            shapes = {
                "single": {"o_proj": (8, 8)},
                "qkv": {name: (8, 8) for name in ("q_proj", "k_proj", "v_proj")},
                "mlp": {"gate_proj": (8, 12), "up_proj": (8, 12), "down_proj": (12, 8)},
            }[kind]
            self.targets = tuple(shapes)
            for name, (input_width, output_width) in shapes.items():
                if quantized:
                    bnb = pytest.importorskip("bitsandbytes")
                    self.is_loaded_in_4bit = True
                    projection = bnb.nn.Linear4bit(
                        input_width, output_width, bias=False, compute_dtype=dtype,
                        compress_statistics=compressed, quant_type="nf4",
                    )
                else:
                    projection = nn.Linear(input_width, output_width, bias=False, dtype=dtype)
                setattr(self, name, projection)

        def forward(self, inputs: Any) -> Any:
            if kind == "single":
                return self.o_proj(inputs)
            if kind == "qkv":
                return tuple(getattr(self, name)(inputs) for name in self.targets)
            return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

    with torch.random.fork_rng():
        torch.manual_seed(41)
        model = Block().to("cpu")
        inject_adapter_in_model(
            LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=list(model.targets)),
            model,
        )
        for name in model.targets:
            projection = getattr(model, name)
            projection.lora_A["default"].to(dtype=torch.float32)
            projection.lora_B["default"].to(dtype=torch.float32)
            with torch.no_grad():
                projection.lora_B["default"].weight.normal_(std=0.2)
            assert projection.lora_A["default"].weight.dtype == torch.float32
            assert projection.lora_B["default"].weight.count_nonzero() > 0
            assert not projection.get_base_layer().weight.requires_grad
    return model


def _patch(model: Any, kind: str) -> None:
    from soup_cli.utils.fast_lora import patch_fast_lora_single_projection
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    patcher = {
        "single": patch_fast_lora_single_projection,
        "qkv": patch_fast_lora_qkv,
        "mlp": patch_fast_lora_mlp,
    }[kind]
    assert patcher(model) == 1


def _adapter_gradients(model: Any) -> dict[str, Any]:
    expected = {
        f"{name}.lora_{letter}.default.weight"
        for name in model.targets for letter in ("A", "B")
    }
    parameters = dict(model.named_parameters())
    assert {name for name in parameters if "lora_" in name} == expected
    for name, parameter in parameters.items():
        if name not in expected:
            assert not parameter.requires_grad and parameter.grad is None
    return {name: parameters[name].grad for name in sorted(expected)}


def _assert_tensor(actual: Any, expected: Any, *, exact: bool = True) -> None:
    import torch

    assert actual is not None and expected is not None, "requested gradient missing"
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape and actual.device == expected.device
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    if exact:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("needs_input_grad", [False, True])
def test_single_adapter_dx_casts_completed_product(
    dtype_name: str, needs_input_grad: bool,
) -> None:
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    reference = _model("single", dtype)
    fast = copy.deepcopy(reference)
    _patch(fast, "single")
    generator = torch.Generator().manual_seed(792)
    inputs = torch.randn(2, 3, 8, generator=generator).to(dtype)
    upstream = torch.randn(2, 3, 8, generator=generator).to(dtype)
    xs = [inputs.clone().requires_grad_(needs_input_grad) for _ in range(2)]
    expected, actual = reference(xs[0]), fast(xs[1])
    assert type(actual.grad_fn).__name__ == "_FastLoraSingleProjectionBackward"
    _assert_tensor(actual, expected)
    expected.backward(upstream)
    actual.backward(upstream)
    if needs_input_grad:
        _assert_tensor(xs[1].grad, xs[0].grad)
    else:
        assert xs[0].grad is None and xs[1].grad is None
    ref_grads, fast_grads = _adapter_gradients(reference), _adapter_gradients(fast)
    assert fast_grads.keys() == ref_grads.keys()
    for name in ref_grads:
        _assert_tensor(fast_grads[name], ref_grads[name])


def _scalar_cast_model(path: str, dtype: Any) -> Any:
    """Isolate each sibling cast without multi-edge reduction or SiLU-chain error."""
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_act="silu")
            self.act_fn = nn.SiLU()
            self.targets = (
                ("q_proj", "k_proj", "v_proj") if path == "qkv" else (f"{path}_proj",)
            )
            names = self.targets if path == "qkv" else ("gate_proj", "up_proj", "down_proj")
            for name in names:
                projection = nn.Linear(1, 1, dtype=dtype)
                with torch.no_grad():
                    projection.weight.fill_(1.0)
                    projection.bias.zero_()
                    if name == "gate_proj":
                        projection.weight.zero_()
                        projection.bias.fill_(16.0)
                setattr(self, name, projection)

        def forward(self, inputs: Any) -> Any:
            if path == "qkv":
                return tuple(getattr(self, name)(inputs) for name in self.targets)
            return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

    with torch.random.fork_rng():
        torch.manual_seed(792)
        model = Block()
        inject_adapter_in_model(
            LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=list(model.targets)),
            model,
        )
        with torch.no_grad():
            for name in model.targets:
                projection = getattr(model, name)
                projection.lora_A["default"].to(dtype=torch.float32)
                projection.lora_B["default"].to(dtype=torch.float32)
                projection.lora_A["default"].weight.copy_(torch.tensor(
                    [[0.0653076171875], [-0.125244140625], [-0.074951171875],
                     [-0.1280517578125]], dtype=torch.float32,
                ))
                projection.lora_B["default"].weight.copy_(torch.tensor(
                    [[0.012041543610394001, 0.0010000348556786776,
                      -0.0005367482081055641, -0.02039286307990551]], dtype=torch.float32,
                ))
                projection.get_base_layer().weight.fill_(0.006153106689453125)
    return model


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("path", ["qkv", "up", "down"])
def test_sibling_adapter_dx_cast_seam_is_exact(path: str, dtype_name: str) -> None:
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    kind = "qkv" if path == "qkv" else "mlp"
    reference = _scalar_cast_model(path, dtype)
    fast = copy.deepcopy(reference)
    _patch(fast, kind)
    xs = [torch.tensor([[0.5]], dtype=dtype, requires_grad=True) for _ in range(2)]
    expected, actual = reference(xs[0]), fast(xs[1])
    expected_outputs = expected if kind == "qkv" else (expected,)
    actual_outputs = actual if kind == "qkv" else (actual,)
    node = "_FastLoraQKVBackward" if kind == "qkv" else "_FastLoraSwiGLUBackward"
    assert all(type(value.grad_fn).__name__ == node for value in actual_outputs)
    for actual_value, expected_value in zip(actual_outputs, expected_outputs):
        _assert_tensor(actual_value, expected_value)
    # Only q is consumed: k/v adapter grads must remain None, not materialized zeros.
    actual_outputs[0].backward(torch.ones_like(actual_outputs[0]))
    expected_outputs[0].backward(torch.ones_like(expected_outputs[0]))
    ref_grads, fast_grads = _adapter_gradients(reference), _adapter_gradients(fast)
    for name in ref_grads:
        if kind == "qkv" and not name.startswith("q_proj."):
            assert fast_grads[name] is None and ref_grads[name] is None
        else:
            _assert_tensor(fast_grads[name], ref_grads[name])
    _assert_tensor(xs[1].grad, xs[0].grad)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("needs_input_grad", [False, True])
def test_mlp_mul_backward_rounds_before_silu_opmath(
    dtype_name: str, needs_input_grad: bool,
) -> None:
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    reference = _scalar_cast_model("gate", dtype)
    with torch.no_grad():
        gate = reference.gate_proj
        gate.lora_A["default"].weight.fill_(0.25)
        gate.lora_B["default"].weight.fill_(0.25)
        adapter_value = gate.lora_B["default"](
            gate.lora_A["default"](torch.tensor([[0.5]], dtype=torch.float32))
        ).squeeze()
        gate.get_base_layer().bias.copy_(
            (torch.tensor(-0.10595703125) - adapter_value).reshape(1)
        )
        reference.up_proj.weight.zero_()
        reference.up_proj.bias.fill_(0.142333984375)
    fast = copy.deepcopy(reference)
    _patch(fast, "mlp")
    xs = [torch.tensor([[0.5]], dtype=dtype, requires_grad=needs_input_grad) for _ in range(2)]
    expected, actual = reference(xs[0]), fast(xs[1])
    assert type(actual.grad_fn).__name__ == "_FastLoraSwiGLUBackward"
    _assert_tensor(actual, expected)
    upstream = torch.tensor([[0.017120361328125]], dtype=dtype)
    expected.backward(upstream)
    actual.backward(upstream)
    ref_grads, fast_grads = _adapter_gradients(reference), _adapter_gradients(fast)
    for name in ref_grads:
        _assert_tensor(fast_grads[name], ref_grads[name])
    if needs_input_grad:
        _assert_tensor(xs[1].grad, xs[0].grad)
    else:
        assert xs[0].grad is None and xs[1].grad is None


def _float64_oracle(model: Any, dtype: Any) -> Any:
    """Use precisely the rounded dense weights or the same dequantized NF4 state."""
    import torch
    from torch import nn

    oracle = copy.deepcopy(model)
    for name in model.targets:
        source = getattr(model, name).get_base_layer()
        if getattr(source.weight, "quant_state", None) is None:
            continue
        from bitsandbytes.functional import dequantize_4bit

        dense = dequantize_4bit(source.weight, source.weight.quant_state).to(dtype)
        replacement = nn.Linear(source.in_features, source.out_features, bias=False, dtype=dtype)
        with torch.no_grad():
            replacement.weight.copy_(dense)
        replacement.requires_grad_(False)
        getattr(oracle, name).base_layer = replacement
    return oracle.double()


def _evaluate(model: Any, inputs: Any, upstreams: tuple[Any, ...]) -> dict[str, Any]:
    import torch

    values = model(inputs)
    outputs = values if isinstance(values, tuple) else (values,)
    torch.autograd.backward(outputs, upstreams)
    result = {f"Y/{index}": value.detach() for index, value in enumerate(outputs)}
    if inputs.requires_grad:
        result["dX"] = inputs.grad
    else:
        assert inputs.grad is None
    result.update(_adapter_gradients(model))
    return result


@pytest.mark.parametrize("kind", ["single", "qkv", "mlp"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("needs_input_grad", [False, True])
@pytest.mark.parametrize("storage", ["dense", "nf4", "nf4-nested"])
def test_mixed_precision_all_requested_gradients_match_declared_float64_bound(
    kind: str, dtype_name: str, needs_input_grad: bool, storage: str,
    record_property: Any,
) -> None:
    """Full grouped reductions use D2's named bound, NOT a bit-equality claim."""
    import json

    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    reference = _model(
        kind, dtype, quantized=storage != "dense", compressed=storage == "nf4-nested",
    )
    fast = copy.deepcopy(reference)
    oracle = _float64_oracle(reference, dtype)
    _patch(fast, kind)
    generator = torch.Generator().manual_seed(792)
    rounded_inputs = torch.randn(2, 3, 8, generator=generator).to(dtype)
    xs = [rounded_inputs.clone().requires_grad_(needs_input_grad) for _ in range(2)]
    oracle_inputs = rounded_inputs.double().requires_grad_(needs_input_grad)
    upstreams = tuple(
        torch.randn(2, 3, 8, generator=generator).to(dtype)
        for _ in range(3 if kind == "qkv" else 1)
    )
    # Observe the actual graph before backward; no PEFT fallback may satisfy this test.
    probe = fast(xs[1])
    outputs = probe if isinstance(probe, tuple) else (probe,)
    node = {
        "single": "_FastLoraSingleProjectionBackward",
        "qkv": "_FastLoraQKVBackward", "mlp": "_FastLoraSwiGLUBackward",
    }[kind]
    assert all(type(value.grad_fn).__name__ == node for value in outputs)
    torch.autograd.backward(outputs, upstreams)
    actual = {f"Y/{index}": value.detach() for index, value in enumerate(outputs)}
    if needs_input_grad:
        actual["dX"] = xs[1].grad
    else:
        assert xs[1].grad is None
    actual.update(_adapter_gradients(fast))
    expected = _evaluate(reference, xs[0], upstreams)
    high_precision = _evaluate(oracle, oracle_inputs, tuple(value.double() for value in upstreams))
    assert actual.keys() == expected.keys() == high_precision.keys()
    for name in expected:
        got, ref, high = actual[name], expected[name], high_precision[name]
        assert got is not None and ref is not None and high is not None, name
        assert got.dtype == ref.dtype, name
        assert got.shape == ref.shape == high.shape, name
        assert got.device == ref.device == high.device, name
        assert all(torch.isfinite(value).all() for value in (got, ref, high)), name
        fast_error = (got.double() - high).abs().max().item()
        peft_error = (ref.double() - high).abs().max().item()
        bound = 2 * peft_error + 1e-8
        record_property(name, json.dumps({
            "bit_exact": torch.equal(got, ref), "fast_float64_max_abs_error": fast_error,
            "peft_float64_max_abs_error": peft_error, "proposed_max_error_bound": bound,
        }))
        assert fast_error <= bound, f"{name}: fast error {fast_error} > declared bound {bound}"
