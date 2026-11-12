"""training.fp16_init_scale: the fp16 GradScaler's starting scale.

torch starts the scaler at 2**16 and halves it on every step whose gradients overflow, and
the step is skipped. On Qwen3.5 / Qwen3.8 in fp16 (T4) the first Gated DeltaNet layer
overflows from a scale of 1024 (ZERO part A2), so a run skips its first ~7 optimizer steps
before it trains at all; a 6-step run on Kaggle updated nothing. The setting starts lower.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string

_BASE = """
base: some/model
task: {task}
data:
  train: ./data.jsonl
  format: alpaca
training:
  epochs: 1
{extra}
"""


def _cfg(extra: str = "", task: str = "sft"):
    return load_config_from_string(_BASE.format(task=task, extra=extra))


def test_default_is_unset():
    assert _cfg().training.fp16_init_scale is None


def test_accepts_a_power_of_two():
    assert _cfg("  fp16_init_scale: 256").training.fp16_init_scale == 256


@pytest.mark.parametrize("value", ["0", "-4", "300", "131072"])
def test_refuses_non_powers_of_two_and_out_of_range(value):
    with pytest.raises(Exception, match="fp16_init_scale"):
        _cfg(f"  fp16_init_scale: {value}")


def test_refused_outside_transformers_sft():
    with pytest.raises(Exception, match="fp16_init_scale"):
        _cfg("  fp16_init_scale: 256", task="dpo")


class _Scaler:
    """The attributes torch.amp.GradScaler keeps for its growth policy."""

    def __init__(self):
        self._growth_factor = 3.0
        self._backoff_factor = 0.25
        self._growth_interval = 7
        self._enabled = True


def _trainer(scaler):
    return SimpleNamespace(accelerator=SimpleNamespace(scaler=scaler))


def test_replaces_the_scaler_with_the_start_scale_and_keeps_its_policy(monkeypatch):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.mixed_precision import apply_fp16_init_scale

    made = {}

    class _New:
        def __init__(self, device, **kwargs):
            made.update(device=device, **kwargs)

    monkeypatch.setattr(torch.amp, "GradScaler", _New)
    trainer = _trainer(_Scaler())
    assert apply_fp16_init_scale(trainer, 256) is True
    assert isinstance(trainer.accelerator.scaler, _New)
    assert made == {"device": "cuda", "init_scale": 256.0, "growth_factor": 3.0,
                    "backoff_factor": 0.25, "growth_interval": 7, "enabled": True}


def test_real_scaler_starts_at_the_given_scale():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GradScaler.get_scale needs CUDA")
    from soup_cli.utils.mixed_precision import apply_fp16_init_scale

    trainer = _trainer(torch.amp.GradScaler("cuda"))
    apply_fp16_init_scale(trainer, 256)
    assert trainer.accelerator.scaler.get_scale() == 256.0


def test_without_a_scaler_it_changes_nothing():
    from soup_cli.utils.mixed_precision import apply_fp16_init_scale

    trainer = _trainer(None)
    assert apply_fp16_init_scale(trainer, 256) is False
    assert trainer.accelerator.scaler is None
    assert apply_fp16_init_scale(SimpleNamespace(), 256) is False
