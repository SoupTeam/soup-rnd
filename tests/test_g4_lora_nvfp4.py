"""Tests for the G4 LoRA-on-frozen-NVFP4 harness (benchmarks/harness/g4_lora_nvfp4.py, ticket 03).

No test touches a GPU or TransformerEngine: the frozen base is a plain ``nn.Linear`` here, which
exercises the adapter, the training loop and the verdict. Numbers are synthetic.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def g4l():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_g4_lora_nvfp4_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "g4_lora_nvfp4.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _plain_linear(in_features, out_features):
    return torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)


def test_adapter_starts_as_the_identity_on_the_base(g4l):
    base = _plain_linear(32, 16)
    layer = g4l.LoRALinear(base, rank=4, alpha=8, dtype=torch.float32)
    x = torch.randn(5, 32)
    assert torch.equal(layer(x), base(x))


def test_only_the_adapter_trains(g4l):
    layer = g4l.LoRALinear(_plain_linear(32, 16), rank=4, alpha=8, dtype=torch.float32)
    trainable = {name for name, p in layer.named_parameters() if p.requires_grad}
    assert trainable == {"lora_a", "lora_b"}
    assert all(not p.requires_grad for p in layer.base.parameters())


def test_adapter_is_bf16_by_default(g4l):
    layer = g4l.LoRALinear(_plain_linear(32, 16), rank=4, alpha=8)
    assert layer.lora_a.dtype == torch.bfloat16 and layer.lora_b.dtype == torch.bfloat16


def test_adapter_forward_adds_the_scaled_low_rank_product(g4l):
    base = _plain_linear(32, 16)
    layer = g4l.LoRALinear(base, rank=4, alpha=8, dtype=torch.float32)
    with torch.no_grad():
        layer.lora_b.normal_()
    x = torch.randn(5, 32)
    expected = base(x) + 2.0 * (x @ layer.lora_a.T @ layer.lora_b.T)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def _tiny_model(g4l, seed=0):
    torch.manual_seed(seed)
    return g4l.TinyLoRALM(
        vocab=32, hidden=64, ffn=128, layers=2, rank=4, alpha=8,
        linear_factory=_plain_linear, dtype=torch.float32,
    )  # fmt: skip


def test_every_linear_gets_an_adapter_and_nothing_else_trains(g4l):
    model = _tiny_model(g4l)
    adapters = dict(model.adapters())
    # per layer: up, down; plus the head
    assert len(adapters) == 2 * 2 + 1
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith(("lora_a", "lora_b")) for n in trainable)


def test_first_step_gradient_reaches_every_adapter_b_through_the_base(g4l):
    model = _tiny_model(g4l)
    tokens, targets = g4l.synthetic_batch(vocab=32, batch=4, seq=16, seed=1)
    loss = g4l.loss_fn(model, tokens, targets)
    loss.backward()
    norms = g4l.adapter_grad_norms(model)
    # lora_b has a gradient at step one even though lora_b starts at zero; lora_a does not yet.
    assert all(n["lora_b"] > 0 for n in norms.values())
    assert all(n["lora_a"] == 0 for n in norms.values())


def test_training_lowers_the_loss_and_moves_the_adapters(g4l):
    model = _tiny_model(g4l)
    report = g4l.train(
        model, vocab=32, batch=4, seq=16, steps=40, lr=5e-3, device="cpu", seed=3
    )  # fmt: skip
    losses = report["losses"]
    assert len(losses) == 40
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    assert report["first_step_grad_norm"] > 0
    assert report["verdict"]["ok"], report["verdict"]
    assert report["data_kind"] == "synthetic"


def _record(**overrides):
    record = {
        "first_step_grad_norm": 0.5,
        "losses": [4.0, 3.9, 3.8, 3.7, 3.6, 3.2, 3.0, 2.9, 2.8, 2.7],
        "adapters_changed": {"a": True, "b": True},
        "trainable_outside_adapter": 0,
    }
    record.update(overrides)
    return record


def test_verdict_passes_a_healthy_run(g4l):
    verdict = g4l.verdict(_record())
    assert verdict["ok"] is True
    assert set(verdict["checks"]) == {
        "grad_norm_positive", "adapters_changed", "loss_decreased", "finite", "base_frozen",
    }  # fmt: skip
    assert all(verdict["checks"].values())


@pytest.mark.parametrize(
    "overrides, failing",
    [
        ({"first_step_grad_norm": 0.0}, "grad_norm_positive"),
        ({"first_step_grad_norm": float("nan")}, "grad_norm_positive"),
        ({"adapters_changed": {"a": True, "b": False}}, "adapters_changed"),
        ({"losses": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]}, "loss_decreased"),
        ({"losses": [4.0, 3.9, 3.8, 3.7, 3.6, float("nan"), 3.0, 2.9, 2.8, 2.7]}, "finite"),
        ({"losses": [4.0, 3.9, 3.8, 3.7, 3.6, 3.2, 3.0, 2.9, 2.8, math.inf]}, "finite"),
        ({"trainable_outside_adapter": 3}, "base_frozen"),
    ],
)
def test_verdict_names_the_failing_check(g4l, overrides, failing):
    verdict = g4l.verdict(_record(**overrides))
    assert verdict["ok"] is False
    assert verdict["checks"][failing] is False


def test_shape_check_refuses_a_misaligned_model_without_falling_back(g4l):
    # hidden 520 is not a multiple of 32: the FP4 GEMM cannot run it (ticket 02).
    with pytest.raises(g4l.UnsupportedShapeError):
        g4l.check_model_shape(tokens=256, hidden=520, ffn=1024, vocab=64)
    with pytest.raises(g4l.UnsupportedShapeError):
        g4l.check_model_shape(tokens=250, hidden=512, ffn=1024, vocab=64)
    g4l.check_model_shape(tokens=256, hidden=512, ffn=1024, vocab=64)


def test_report_label_says_synthetic(g4l):
    assert "synthetic" in g4l.REPORT_LABEL
