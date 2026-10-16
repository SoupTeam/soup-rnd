"""D2 correctness regressions: NF4 checkpointing and heterogeneous MLP adapters."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest


def _adapter_grads(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("needs_input_grad", [False, True])
@pytest.mark.parametrize(
    ("device", "dtype_name", "rtol", "atol"),
    [
        pytest.param("cpu", "float32", 1e-5, 1e-6, id="cpu-fp32"),
        pytest.param("cuda", "float16", 1e-2, 1e-2, marks=pytest.mark.gpu, id="cuda-fp16"),
    ],
)
def test_nf4_single_non_reentrant_checkpoint_matches_peft(
    compressed: bool, needs_input_grad: bool,
    device: str, dtype_name: str, rtol: float, atol: float,
) -> None:
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")
    from peft import LoraConfig, inject_adapter_in_model
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

    dtype = getattr(torch, dtype_name)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_loaded_in_4bit = True
            self.o_proj = bnb.nn.Linear4bit(
                8, 8, bias=False, compute_dtype=dtype,
                compress_statistics=compressed, quant_type="nf4",
            )

        def forward(self, inputs):
            return self.o_proj(inputs)

    torch.manual_seed(331)
    model = Model().to(device)
    inject_adapter_in_model(
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["o_proj"]), model
    )
    with torch.no_grad():
        model.o_proj.lora_B["default"].weight.normal_(std=0.2)
    assert model.o_proj.get_base_layer().weight.quant_state is not None
    reference = copy.deepcopy(model)
    inputs = torch.randn(
        2, 3, 8, device=device, dtype=dtype, requires_grad=needs_input_grad
    )
    expected = checkpoint(reference, inputs, use_reentrant=False)
    expected.square().mean().backward()
    reference_grads = _adapter_grads(reference)
    expected_dx = inputs.grad.clone() if needs_input_grad else None

    actual_inputs = inputs.detach().clone().requires_grad_(needs_input_grad)
    assert patch_fast_lora_single_projection(model) == 1
    actual = checkpoint(model, actual_inputs, use_reentrant=False)
    assert type(actual.grad_fn).__name__ == "_FastLoraSingleProjectionBackward"
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    if needs_input_grad:
        torch.testing.assert_close(actual_inputs.grad, expected_dx, rtol=rtol, atol=atol)
    assert _adapter_grads(model).keys() == reference_grads.keys()
    for name, gradient in _adapter_grads(model).items():
        torch.testing.assert_close(gradient, reference_grads[name], rtol=rtol, atol=atol)


@pytest.mark.parametrize("fp32_projection", ["gate_proj", "up_proj", "down_proj"])
def test_mlp_heterogeneous_adapter_dtypes_delegate_safely(fp32_projection: str) -> None:
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, inject_adapter_in_model
    from torch import nn

    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 12, bias=False)
            self.up_proj = nn.Linear(8, 12, bias=False)
            self.down_proj = nn.Linear(12, 8, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, inputs):
            return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_act="silu")
            self.mlp = MLP()

        def forward(self, inputs):
            return self.mlp(inputs)

    torch.manual_seed(837)
    model = Model().to(dtype=torch.bfloat16)
    inject_adapter_in_model(
        LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        ), model,
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(model.mlp, name)
        dtype = torch.float32 if name == fp32_projection else torch.bfloat16
        projection.lora_A["default"].to(dtype=dtype)
        projection.lora_B["default"].to(dtype=dtype)
        with torch.no_grad():
            projection.lora_B["default"].weight.normal_(std=0.2)
    reference = copy.deepcopy(model)
    inputs = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    expected = reference(inputs)
    expected.float().square().mean().backward()
    reference_grads = _adapter_grads(reference)
    actual_inputs = inputs.detach().clone().requires_grad_(True)
    assert patch_fast_lora_mlp(model) == 1
    actual = model(actual_inputs)
    assert type(actual.grad_fn).__name__ != "_FastLoraSwiGLUBackward"
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_inputs.grad, inputs.grad, rtol=0, atol=0)
    for name, gradient in _adapter_grads(model).items():
        torch.testing.assert_close(gradient, reference_grads[name], rtol=0, atol=0)
