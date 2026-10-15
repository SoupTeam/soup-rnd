"""B2 ticket 18: every command that loads an adapter reloads a shared-expert adapter correctly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

from tests.test_b2_shared_expert_lora import (
    _SHARED_TARGETS,
    _adapter_modules,
    _config,
    _expected_shared_modules,
    _logits,
    _randomise_adapter,
    _tiny_model,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR


class _Tok:
    pad_token = "<pad>"
    eos_token = "<eos>"


def _patch_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """The base model and tokenizer come from the tiny in-memory fixture."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", classmethod(lambda cls, *a, **k: _tiny_model())
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok())


def _save(tmp_path: Any, targets: Any, *, plain: bool = False) -> torch.Tensor:
    """Save a nonzero adapter and return the logits of the model that trained it."""
    from peft import get_peft_model

    from soup_cli.utils.peft_wiring import attach_adapter

    build = get_peft_model if plain else attach_adapter
    trained = build(_tiny_model(), _config(targets))
    _randomise_adapter(trained)
    trained.save_pretrained(tmp_path / "adapter")
    return _logits(trained)


def _chat(path: str) -> Any:
    from soup_cli.commands.chat import _load_model

    return _load_model(path, "tiny", True, "cpu")[0]


def _infer(path: str) -> Any:
    from soup_cli.commands.infer import _load_model

    return _load_model(path, "tiny", "cpu")[0]


def _diff(path: str) -> Any:
    from soup_cli.commands.diff import _load_model

    return _load_model(path, "tiny", "cpu")[0]


def _live_eval(path: str) -> Any:
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    return load_model_and_tokenizer("tiny", adapter=path, device="cpu")[0]


def _iterative_dpo(path: str) -> Any:
    from soup_cli.utils.iterative_dpo import _load_sampling_model

    return _load_sampling_model("tiny", path, "cpu")


def _mole(path: str) -> Any:
    from soup_cli.trainer.mole_routing import MoleRoutingTrainerWrapper

    wrapper = object.__new__(MoleRoutingTrainerWrapper)
    tcfg = SimpleNamespace(
        mole_task_adapters=[path, path], mole_top_k=None, mole_temperature=None, seed=1
    )
    wrapper.config = SimpleNamespace(base="tiny", training=tcfg)
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    wrapper.setup({"train": []})
    return wrapper.model


_CONSUMERS: list[Callable[[str], Any]] = [
    _chat,
    _infer,
    _diff,
    _live_eval,
    _iterative_dpo,
    _mole,
]
_IDS = ["chat", "infer", "diff", "live_eval", "iterative_dpo", "mole_routing"]


def _saved_adapter_dir(tmp_path: Any) -> str:
    # an adapter dir the consumers treat as a local adapter
    return str(tmp_path / "adapter")


@pytest.mark.parametrize("load", _CONSUMERS, ids=_IDS)
def test_consumer_loads_a_shared_expert_adapter_onto_the_shared_experts(
    load: Callable[[str], Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from soup_cli.utils.peft_wiring import load_adapter

    _patch_hub(monkeypatch)
    _save(tmp_path, _SHARED_TARGETS)
    path = _saved_adapter_dir(tmp_path)
    expected = _logits(load_adapter(_tiny_model(), path))

    model = load(path)

    assert _adapter_modules(model) == _expected_shared_modules(model.config)
    assert torch.equal(_logits(model), expected)


@pytest.mark.parametrize("load", _CONSUMERS, ids=_IDS)
def test_consumer_loads_a_routed_expert_adapter_the_way_peft_does(
    load: Callable[[str], Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from peft import PeftModel

    _patch_hub(monkeypatch)
    _save(tmp_path, ["gate_proj", "up_proj", "down_proj"], plain=True)
    path = _saved_adapter_dir(tmp_path)
    theirs = PeftModel.from_pretrained(_tiny_model(), path)

    model = load(path)

    assert _adapter_modules(model) == _adapter_modules(theirs)
    assert torch.equal(_logits(model), _logits(theirs))


def test_control_plain_peft_differs_from_the_helper_reload(tmp_path: Any) -> None:
    from peft import PeftModel

    from soup_cli.utils.peft_wiring import load_adapter

    _save(tmp_path, _SHARED_TARGETS)
    path = _saved_adapter_dir(tmp_path)

    plain = _logits(PeftModel.from_pretrained(_tiny_model(), path))
    helper = _logits(load_adapter(_tiny_model(), path))

    assert not torch.equal(plain, helper)


def test_mole_serve_and_training_load_every_task_adapter_onto_the_shared_experts(
    tmp_path: Any,
) -> None:
    """``load_task_adapters`` serves both ``setup`` and ``load_mole_for_serve``."""
    from soup_cli.utils.mole_routing import load_task_adapters
    from soup_cli.utils.peft_wiring import attach_adapter

    for name in ("a", "b"):
        trained = attach_adapter(_tiny_model(), _config(_SHARED_TARGETS))
        _randomise_adapter(trained)
        trained.save_pretrained(tmp_path / name)

    model, names = load_task_adapters(
        _tiny_model(), [str(tmp_path / "a"), str(tmp_path / "b")]
    )

    assert names == ["task_0", "task_1"]
    for name in names:
        owners = {
            n.rsplit(".lora_", 1)[0].removeprefix("base_model.model.")
            for n, _ in model.named_parameters()
            if f".lora_A.{name}." in n or f".lora_B.{name}." in n
        }
        assert owners == _expected_shared_modules(model.config), name
