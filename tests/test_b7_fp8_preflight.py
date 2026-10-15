"""B7: FP8 source bytes must expand before both pre-sharding budget checks."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from soup_cli.utils.layer_stream import estimate_stream_store_bytes

FP8_QUANT_CONFIG = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}


@pytest.mark.parametrize("dtype, expected", [("bfloat16", 2_000), ("float32", 4_000)])
def test_unquantised_fp8_expands_to_target_dtype(dtype: str, expected: int) -> None:
    assert estimate_stream_store_bytes(1_000, dtype=dtype, fp8_source=True) == expected


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_nf4_fp8_estimate_does_not_depend_on_target_dtype(dtype: str) -> None:
    # 64 elements: 32 packed bytes + one fp32 absmax. Source weights occupy 64 bytes.
    assert estimate_stream_store_bytes(
        64, dtype=dtype, quant="nf4", double_quant=False, fp8_source=True,
    ) == 36


@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_ordinary_source_estimate_is_unchanged(quant: str) -> None:
    old_default = estimate_stream_store_bytes(2_000, dtype="bfloat16", quant=quant)
    assert estimate_stream_store_bytes(
        2_000, dtype="bfloat16", quant=quant, fp8_source=False,
    ) == old_default
    if quant == "none":
        assert old_default == 2_000


@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_negative_source_size_still_refused(quant: str) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_stream_store_bytes(-1, dtype="bfloat16", quant=quant, fp8_source=True)


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_estimate_against_actual_cpu_shards(
    tmp_path: Path, fp8: bool, quant: str,
) -> None:
    """Fixture-specific 2% bound, not a promised bound for arbitrary mixtures."""
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import shard_checkpoint, source_weight_bytes

    src = tmp_path / "source"
    src.mkdir()
    config = {"model_type": "llama", "num_hidden_layers": 1}
    if fp8:
        config["quantization_config"] = FP8_QUANT_CONFIG
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    names = ("self_attn.q_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
    tensors = {}
    generator = torch.Generator().manual_seed(7)
    for name in names:
        key = "model.layers.0." + name
        weight = torch.randn(1024, 1024, generator=generator)
        tensors[key] = weight.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
        if fp8:
            tensors[key + "_scale_inv"] = torch.ones(8, 8, dtype=torch.float32)
    tensors["model.embed_tokens.weight"] = torch.ones(16, 64, dtype=torch.bfloat16)
    tensors["model.norm.weight"] = torch.ones(64, dtype=torch.bfloat16)
    save_file(tensors, str(src / "model.safetensors"))
    out = tmp_path / "cache"
    shard_checkpoint(
        str(src), str(out), dtype="bfloat16", arch="llama", quant=quant,
        quant_suffixes=names if quant == "nf4" else (), double_quant=False,
        quant_device="cpu" if quant == "nf4" else None,
    )
    actual = sum(file.stat().st_size for file in out.glob("*.safetensors"))
    estimate = estimate_stream_store_bytes(
        source_weight_bytes(str(src)), dtype="bfloat16", quant=quant,
        double_quant=False, fp8_source=fp8,
    )
    assert abs(estimate - actual) / actual < 0.02
    if fp8:
        # The previous estimate misses nearly half the cache; this control makes
        # the test detect forgetting to thread the new source format flag.
        previous = estimate_stream_store_bytes(
            source_weight_bytes(str(src)), dtype="bfloat16", quant=quant, double_quant=False,
        )
        assert previous < actual * 0.55


class StopPlanningError(Exception):
    """Test boundary: no weights should be sharded in a preflight test."""


def _setup_before_sharding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fp8: bool, quantization: str, disk: bool,
) -> tuple[Any, Any, list[int]]:
    from transformers import AutoConfig, AutoTokenizer

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_shard, layer_stream, spectrum_scan, stream_adapter_preflight

    model_config = SimpleNamespace(model_type="llama")
    if fp8:
        model_config.quantization_config = FP8_QUANT_CONFIG
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *_a, **_k: model_config)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained",
        lambda *_a, **_k: SimpleNamespace(pad_token="pad", eos_token="eos"),
    )
    monkeypatch.setattr(layer_stream, "resolve_stream_dtype", lambda *_a: "bfloat16")
    monkeypatch.setattr(
        layer_stream, "free_ram_bytes",
        lambda: 20_000_000 if quantization == "none" else 6_000_000,
    )
    monkeypatch.setattr(layer_shard, "source_weight_bytes", lambda *_a: 10_000_000)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda *_a: str(tmp_path / "cache"))
    monkeypatch.setattr(
        stream_adapter_preflight, "build_stream_adapter_plan",
        lambda *_a, **_k: SimpleNamespace(quant_suffixes=(), lora_config=None),
    )
    monkeypatch.setattr(
        stream_adapter_preflight, "check_stream_adapter_budget", lambda *_a, **_k: None,
    )
    monkeypatch.setattr("soup_cli.utils.stripe_roots.resolve_stripe_roots", lambda *_a, **_k: ())
    monkeypatch.setattr(layer_shard, "estimate_oq_stream_cache_bytes", lambda *_a, **_k: None)

    def stop_sharding(*_args: Any, **_kwargs: Any) -> None:
        raise StopPlanningError("reached sharding")

    monkeypatch.setattr(layer_shard, "shard_checkpoint", stop_sharding)
    footprints = []

    def capture_disk(**kwargs: Any) -> None:
        footprints.append(kwargs["shard_bytes"])
        raise StopPlanningError("disk footprint captured")

    monkeypatch.setattr("soup_cli.trainer.stream_setup._render_stream_disk_preflight", capture_disk)

    def resolve(_base: str, **kwargs: Any) -> str:
        if disk:
            kwargs["before_materialize"](SimpleNamespace(
                source_bytes=10_000_000, needs_materialization=True,
                weights_dir=str(tmp_path), materialized_copy_bytes=0,
                materialize_bytes=0,
            ))
        return str(tmp_path)

    monkeypatch.setattr(spectrum_scan, "resolve_model_weights", resolve)
    cfg = load_config_from_string(f"""
base: fake/local
data:
  train: data.jsonl
training:
  batch_size: 1
  gradient_accumulation_steps: 1
  stream_layers: true
  stream_source: ram
  quantization: {quantization}
""")
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.device = "cpu"
    return wrapper, cfg, footprints


@pytest.mark.parametrize("fp8, quantization, expected", [
    (True, "none", 20_000_000), (True, "4bit", 5_625_000),
    (False, "none", 10_000_000), (False, "4bit", 2_579_345),
])
def test_disk_preflight_threads_fp8_config_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fp8: bool,
    quantization: str, expected: int,
) -> None:
    wrapper, cfg, footprints = _setup_before_sharding(
        tmp_path, monkeypatch, fp8=fp8, quantization=quantization, disk=True,
    )
    with pytest.raises(StopPlanningError, match="disk footprint captured"):
        wrapper._setup_streaming_transformers(cfg, cfg.training)
    assert footprints == [expected]


@pytest.mark.parametrize("quantization", ["none", "4bit"])
def test_ram_preflight_refuses_fp8_before_sharding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quantization: str,
) -> None:
    wrapper, cfg, _ = _setup_before_sharding(
        tmp_path, monkeypatch, fp8=True, quantization=quantization, disk=False,
    )
    with pytest.raises(ValueError, match="stream_source='ram'"):
        wrapper._setup_streaming_transformers(cfg, cfg.training)


def test_same_source_bytes_bf16_nf4_is_allowed_to_reach_sharding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, cfg, _ = _setup_before_sharding(
        tmp_path, monkeypatch, fp8=False, quantization="4bit", disk=False,
    )
    with pytest.raises(StopPlanningError, match="reached sharding"):
        wrapper._setup_streaming_transformers(cfg, cfg.training)
