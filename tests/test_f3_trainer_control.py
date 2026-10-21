"""Seam 2 of the F3 spec: the zero-gradient control through a real HF Trainer on CPU.

A tiny random Llama, every label masked (``labels == -100``), ``weight_decay`` zero.
The collector is attached as a callback by the adapter in
``benchmarks/harness/f3_trainer_adapter.py``. ``build_train_report`` must return
``valid: false`` and refuse both ``grad_norm`` and ``parameters_changed``. No network,
no GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
STEPS = 12


@pytest.fixture(scope="module")
def adapter():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    path = HARNESS / "f3_trainer_adapter.py"
    assert path.exists(), "adapter benchmarks/harness/f3_trainer_adapter.py is missing"
    name = "_f3_trainer_adapter_test_module"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(config)


class _MaskedRows(torch.utils.data.Dataset):
    """Rows whose labels are all -100, so the loss sees no supervised token."""

    def __len__(self) -> int:
        return 64

    def __getitem__(self, index: int):
        ids = torch.randint(0, 64, (8,))
        return {"input_ids": ids, "labels": torch.full((8,), -100, dtype=torch.long)}


@pytest.fixture(scope="module")
def trained(adapter, tmp_path_factory):
    from transformers import Trainer, TrainingArguments

    out = tmp_path_factory.mktemp("f3-control")
    args = TrainingArguments(
        output_dir=str(out),
        max_steps=STEPS,
        per_device_train_batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        use_cpu=True,
        disable_tqdm=True,
    )
    trainer = Trainer(model=_tiny_model(), args=args, train_dataset=_MaskedRows())
    collector = adapter.attach_collector(trainer)
    trainer.train()
    return adapter.build_report(collector, steps_requested=STEPS)


def test_control_is_rejected_as_invalid(trained):
    assert trained["valid"] is False


def test_both_checks_refuse(trained):
    refused = {item["check"] for item in trained["failures"]}
    assert {"grad_norm", "parameters_changed"} <= refused


def test_all_steps_were_measured(trained):
    assert trained["steps_measured"] == STEPS


def test_no_supervised_token_was_seen(trained):
    assert trained["tokens"]["useful"] == 0


def test_grad_norm_refusal_is_for_zero_not_nan(trained):
    """The derived claim: fully masked labels log grad_norm 0.0 (not NaN) in this stack."""
    message = next(f["message"] for f in trained["failures"] if f["check"] == "grad_norm")
    assert "grad_norm == 0.0" in message
