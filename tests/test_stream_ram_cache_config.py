"""An opt-in RAM budget must reach the SSD source and be visible to users."""

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig


def payload(value=0):
    return {
        "base": "local-model",
        "task": "sft",
        "data": {"train": "train.jsonl", "format": "alpaca"},
        "training": {
            "stream_layers": True,
            "batch_size": 1,
            "quantization": "none",
            "stream_ram_cache_mb": value,
        },
    }


@pytest.mark.parametrize("value", [0, 1, 14336])
def test_cache_budget_round_trips(value):
    config = SoupConfig.model_validate(payload(value))
    assert config.training.stream_ram_cache_mb == value
    assert (
        SoupConfig.model_validate_json(config.model_dump_json()).training.stream_ram_cache_mb
        == value
    )


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "8", "auto", None, []])
def test_cache_budget_is_a_strict_nonnegative_integer(value):
    with pytest.raises(ValidationError, match="stream_ram_cache_mb"):
        SoupConfig.model_validate(payload(value))


def test_cache_disabled_by_default():
    data = payload()
    del data["training"]["stream_ram_cache_mb"]
    assert SoupConfig.model_validate(data).training.stream_ram_cache_mb == 0


def test_cache_requires_streaming():
    data = payload(1024)
    data["training"]["stream_layers"] = False
    with pytest.raises(ValidationError, match="stream_ram_cache_mb"):
        SoupConfig.model_validate(data)


def test_explicit_full_ram_tier_rejects_redundant_partial_cache():
    data = payload(1024)
    data["training"]["stream_source"] = "ram"
    with pytest.raises(ValidationError, match="stream_ram_cache_mb"):
        SoupConfig.model_validate(data)


def test_cache_budget_reaches_shared_streaming_setup(tmp_path, monkeypatch):
    from tests.test_issue623_stream_pin_allocation import _drive_wiring

    kwargs, _text = _drive_wiring(
        tmp_path,
        monkeypatch,
        stream_pin_yaml="",
        extra_training_yaml="  stream_source: disk\n  stream_ram_cache_mb: 1024\n",
    )
    assert kwargs["ram_cache_bytes"] == 1024 * 2**20


def test_factory_passes_budget_and_preserves_io_failures(monkeypatch):
    import soup_cli.utils.async_disk_source as disk
    from soup_cli.utils.layer_stream_runtime import _build_source

    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("cache read sentinel")

    monkeypatch.setattr(disk, "AsyncDiskSource", fake)
    with pytest.raises(OSError, match="sentinel"):
        _build_source("unused", 8, {}, True, None, tier="disk", ram_cache_bytes=1024)
    assert len(calls) == 1
    assert calls[0]["ram_cache_bytes"] == 1024


def test_operator_line_distinguishes_cache_from_staging():
    from soup_cli.trainer.stream_setup import _stream_source_line

    text = _stream_source_line(
        {
            "tier": "disk",
            "pinned": True,
            "pinned_bytes": 6 * 2**30,
            "disk_bytes": 20 * 2**30,
            "store_bytes": 2 * 2**30,
            "read_ahead": 2,
            "ram_cache_allocated_bytes": 4 * 2**30,
            "ram_cache_indices": [3, 4],
        }
    )
    assert "host staging" in text and "RAM cache" in text and "2 layers" in text
