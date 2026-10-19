"""A ``kimi_k2`` folder outside layer streaming is refused by name.

Five paths load a base model without layer streaming: ``export``, ``merge``, ``serve``,
the resident trainers (``soup train`` without ``training.stream_layers``) and
``soup recipes verify`` (``attach_preflight.load_hf_config``). Each must stop before any
``from_pretrained`` or ``AutoConfig`` call on a ``kimi_k2`` folder, and leave a
``deepseek_v3`` folder alone.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.kimi_k2 import KimiK2OutsideStreamingError, refuse_kimi_k2_outside_streaming

runner = CliRunner()


class _LoadedError(Exception):
    """Raised by a stub loader, so a test sees that the path reached a load."""


def _folder(root: Path, name: str, model_type: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return path


def _adapter(root: Path, base: Path) -> Path:
    path = root / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base), "r": 8})
    )
    return path


@pytest.fixture
def kimi(tmp_path: Path) -> Path:
    return _folder(tmp_path, "kimi", "kimi_k2")


@pytest.fixture
def v3(tmp_path: Path) -> Path:
    return _folder(tmp_path, "v3", "deepseek_v3")


@pytest.fixture
def no_loads(monkeypatch):
    """Every base-model load raises ``_LoadedError``, so a path that gets there is visible."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    def boom(*_args, **_kwargs):
        raise _LoadedError

    for target in (AutoConfig, AutoModelForCausalLM, AutoTokenizer):
        monkeypatch.setattr(target, "from_pretrained", boom)


def _named(text: str, folder: Path) -> bool:
    flat = " ".join(text.split())
    return "kimi_k2" in flat and "stream_layers" in flat and "layer streaming" in flat


# --- the guard ---


def test_guard_names_cause_and_fix(kimi: Path):
    with pytest.raises(KimiK2OutsideStreamingError) as info:
        refuse_kimi_k2_outside_streaming(str(kimi))
    assert _named(str(info.value), kimi)


def test_guard_passes_other_folders(v3: Path, tmp_path: Path):
    refuse_kimi_k2_outside_streaming(str(v3))
    refuse_kimi_k2_outside_streaming(str(tmp_path / "does-not-exist"))
    refuse_kimi_k2_outside_streaming(str(tmp_path))  # a folder without config.json


# --- merge ---


def test_merge_refuses_kimi_base(tmp_path: Path, kimi: Path, no_loads):
    adapter = _adapter(tmp_path, kimi)
    result = runner.invoke(app, ["merge", "--adapter", str(adapter), "--output", "out"])
    assert result.exit_code == 1
    assert not isinstance(result.exception, _LoadedError)
    assert _named(result.output, kimi)


def test_merge_leaves_deepseek_v3_alone(tmp_path: Path, v3: Path, no_loads):
    adapter = _adapter(tmp_path, v3)
    result = runner.invoke(app, ["merge", "--adapter", str(adapter), "--output", "out"])
    assert "kimi_k2" not in result.output


# --- export ---


@pytest.mark.parametrize("fmt", ["gguf", "awq", "gptq", "onnx"])
def test_export_refuses_kimi_model(kimi: Path, no_loads, fmt: str):
    result = runner.invoke(app, ["export", "--model", str(kimi), "--format", fmt])
    assert result.exit_code == 1
    assert not isinstance(result.exception, _LoadedError)
    assert _named(result.output, kimi)


def test_export_refuses_kimi_base_of_an_adapter(tmp_path: Path, kimi: Path, no_loads):
    adapter = _adapter(tmp_path, kimi)
    result = runner.invoke(app, ["export", "--model", str(adapter)])
    assert result.exit_code == 1
    assert not isinstance(result.exception, _LoadedError)
    assert _named(result.output, kimi)


def test_export_leaves_deepseek_v3_alone(tmp_path: Path, v3: Path, no_loads):
    adapter = _adapter(tmp_path, v3)
    result = runner.invoke(app, ["export", "--model", str(adapter)])
    assert "kimi_k2" not in result.output
    assert isinstance(result.exception, _LoadedError)


# --- serve ---


def test_serve_load_refuses_kimi(kimi: Path, no_loads):
    from soup_cli.commands.serve import _load_model

    with pytest.raises(KimiK2OutsideStreamingError):
        _load_model(str(kimi), None, False, "cpu")


def test_serve_load_refuses_kimi_base_of_an_adapter(tmp_path: Path, kimi: Path, no_loads):
    from soup_cli.commands.serve import _load_model

    adapter = _adapter(tmp_path, kimi)
    with pytest.raises(KimiK2OutsideStreamingError):
        _load_model(str(adapter), str(kimi), True, "cpu")


def test_serve_command_refuses_kimi(kimi: Path, no_loads):
    result = runner.invoke(app, ["serve", "--model", str(kimi), "--backend", "transformers"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, _LoadedError)
    assert _named(result.output, kimi)


def test_serve_load_leaves_deepseek_v3_alone(v3: Path, no_loads):
    from soup_cli.commands.serve import _load_model

    with pytest.raises(_LoadedError):
        _load_model(str(v3), None, False, "cpu")


# --- resident trainers ---


def _train_yaml(tmp_path: Path, base: Path, stream: bool) -> Path:
    path = tmp_path / "soup.yaml"
    path.write_text(
        f"base: {base}\n"
        "task: sft\n"
        "data:\n  train: ./data.jsonl\n  format: alpaca\n"
        f"training:\n  epochs: 1\n  stream_layers: {'true' if stream else 'false'}\n"
    )
    return path


def test_train_refuses_kimi_without_streaming(tmp_path: Path, kimi: Path, no_loads):
    config = _train_yaml(tmp_path, kimi, stream=False)
    result = runner.invoke(app, ["train", "--config", str(config), "--yes"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, _LoadedError)
    assert _named(result.output, kimi)


def test_train_does_not_refuse_kimi_with_streaming(tmp_path: Path, kimi: Path, no_loads):
    config = _train_yaml(tmp_path, kimi, stream=True)
    result = runner.invoke(app, ["train", "--config", str(config), "--yes"])
    assert "not supported outside layer streaming" not in " ".join(result.output.split())


def test_train_leaves_deepseek_v3_alone(tmp_path: Path, v3: Path, no_loads):
    config = _train_yaml(tmp_path, v3, stream=False)
    result = runner.invoke(app, ["train", "--config", str(config), "--yes"])
    assert "kimi_k2" not in result.output


# --- soup recipes (attach_preflight) ---


def test_load_hf_config_refuses_kimi(kimi: Path, no_loads):
    from soup_cli.utils.attach_preflight import load_hf_config

    with pytest.raises(KimiK2OutsideStreamingError):
        load_hf_config(str(kimi))


def test_load_hf_config_leaves_deepseek_v3_alone(v3: Path, no_loads):
    from soup_cli.utils.attach_preflight import load_hf_config

    with pytest.raises(_LoadedError):
        load_hf_config(str(v3))


# --- no kimi_k2 model with random routed experts ---


def test_guarded_loaders_are_never_reached(kimi: Path, tmp_path: Path, monkeypatch):
    """Every loader is a stub that would return a model; none is reached."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.commands.serve import _load_model
    from soup_cli.utils.attach_preflight import load_hf_config

    reached = []
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", lambda *a, **k: reached.append("model")
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **k: reached.append("config"))
    with pytest.raises(KimiK2OutsideStreamingError):
        _load_model(str(kimi), None, False, "cpu")
    with pytest.raises(KimiK2OutsideStreamingError):
        load_hf_config(str(kimi))
    assert reached == []


def _recipe(base: Path, stream: bool):
    from types import SimpleNamespace

    return SimpleNamespace(
        base=str(base),
        task="sft",
        training=SimpleNamespace(stream_layers=stream),
    )


def test_recipes_verify_reports_kimi_refusal_by_name(kimi: Path, no_loads):
    from soup_cli.commands.recipes import _config_loader_for
    from soup_cli.utils.attach_preflight import load_hf_config

    loader = _config_loader_for(_recipe(kimi, False), load_hf_config)
    with pytest.raises(KimiK2OutsideStreamingError):
        loader(str(kimi))


def test_recipes_verify_streaming_recipe_skips_the_refusal(kimi: Path, no_loads):
    from soup_cli.commands.recipes import _config_loader_for
    from soup_cli.utils.attach_preflight import load_hf_config

    loader = _config_loader_for(_recipe(kimi, True), load_hf_config)
    assert loader is not load_hf_config
