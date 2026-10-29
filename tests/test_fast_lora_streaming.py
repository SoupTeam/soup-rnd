"""Fast-LoRA must execute through Soup's streamed SFT path, including slot reuse."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig
from tests.test_fast_lora_training import config_payload


@pytest.mark.parametrize("quantization", ["none", "4bit"])
def test_fast_lora_streaming_config_round_trip(quantization):
    payload = config_payload()
    payload["training"].update(stream_layers=True, quantization=quantization, batch_size=1)
    config = SoupConfig.model_validate(payload)
    restored = SoupConfig.model_validate_json(config.model_dump_json())
    assert restored.training.stream_layers and restored.training.fast_lora
    assert restored.training.quantization == quantization


@pytest.mark.parametrize("quantization", ["8bit", "fp8"])
def test_fast_lora_streaming_refuses_other_quantization(quantization):
    payload = config_payload()
    payload["training"].update(stream_layers=True, quantization=quantization, batch_size=1)
    with pytest.raises(ValidationError):
        SoupConfig.model_validate(payload)


def _tiny_streaming_base(tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests.test_issue691_sft_packing import _write_tiny_tokenizer

    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
            max_position_embeddings=128,
            use_cache=False,
        )
    )
    base = tmp_path / "model"
    model.save_pretrained(base)
    _write_tiny_tokenizer(str(base))
    return str(base)


def _stream_config(base, output, *, enabled, quantization, source="ram"):
    payload = config_payload()
    payload.update(base=base, output=str(output))
    payload["data"]["max_length"] = 64
    payload["training"].update(
        fast_lora=enabled,
        stream_layers=True,
        stream_buffers=2,
        stream_source=source,
        stream_pin=False,
        stream_disk_kind="nvme" if source == "disk" else None,
        quantization=quantization,
        batch_size=1,
        epochs=1,
        logging_steps=1,
        save_steps=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        warmup_ratio=0,
    )
    payload["training"]["lora"]["target_modules"] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return SoupConfig.model_validate(payload)


@pytest.mark.parametrize("quantization", ["none", "4bit"])
@pytest.mark.parametrize("source", ["ram", "disk"])
@pytest.mark.parametrize(
    "device,precision",
    [
        ("cpu", None),
        pytest.param("cuda", "bf16", marks=pytest.mark.gpu),
        pytest.param("cuda", "fp16", marks=pytest.mark.gpu),
    ],
)
def test_soup_streamed_training_matches_lora_and_reloads_adapter(
    tmp_path,
    monkeypatch,
    quantization,
    source,
    device,
    precision,
):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, TrainerCallback

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_shard, layer_stream

    if precision:
        # Exercise both native arithmetic modes on one Ampere GPU. Override
        # only the dtype policy, not CUDA kernels or capability reports.
        if precision == "bf16":
            assert torch.cuda.is_bf16_supported(including_emulation=False)
        original_dtype = layer_stream.resolve_stream_dtype
        monkeypatch.setattr(
            layer_stream,
            "resolve_stream_dtype",
            lambda target: (
                {"bf16": "bfloat16", "fp16": "float16"}[precision]
                if str(target).startswith("cuda")
                else original_dtype(target)
            ),
        )
        monkeypatch.setattr(
            SFTTrainerWrapper,
            "_resolve_mixed_precision",
            lambda *_args: (precision == "bf16", precision == "fp16"),
        )

    if quantization == "4bit":
        pytest.importorskip("bitsandbytes")
    torch.set_num_threads(1)
    base = _tiny_streaming_base(tmp_path)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda _base: str(tmp_path / "shards"))
    rows = [
        {
            "messages": [
                {"role": "user", "content": "the cat sat on the mat"},
                {"role": "assistant", "content": "hello world"},
            ]
        }
    ] * 4
    results = []
    runtime_stats = []
    for enabled in (False, True):
        output = tmp_path / ("fast" if enabled else "ordinary")
        wrapper = SFTTrainerWrapper(
            _stream_config(base, output, enabled=enabled, quantization=quantization, source=source),
            device=device,
        )
        try:
            wrapper.setup({"train": rows})
            runtime = wrapper._stream_runtime
            layers = wrapper.model.get_base_model().model.layers
            assert len(layers) == 4 and runtime.pool.n == 2
            assert all(layer.use_checkpoint for layer in layers)
            assert runtime.tier == source
            # The frozen decoder placeholders must stay meta, not materialize a
            # second resident base to make Fast-LoRA work.
            assert layers[0].inner.self_attn.q_proj.get_base_layer().weight.is_meta
            adapters = {
                name: parameter
                for name, parameter in wrapper.model.named_parameters()
                if "lora_" in name
            }
            assert len(adapters) == 56
            generator = torch.Generator().manual_seed(123)
            with torch.no_grad():
                for name, parameter in adapters.items():
                    if "lora_B" in name:
                        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)
            initial = {
                name: parameter.detach().cpu().clone() for name, parameter in adapters.items()
            }
            gradients = []
            logits = []

            def observe(_module, _inputs, result):
                logits.append(result.logits.detach().cpu().clone())

            handle = wrapper.model.register_forward_hook(observe)

            class Capture(TrainerCallback):
                def on_pre_optimizer_step(self, _args, _state, _control, model=None, **_kwargs):
                    gradients.append(
                        {
                            name: parameter.grad.detach().cpu().clone()
                            for name, parameter in model.named_parameters()
                            if "lora_" in name
                        }
                    )
                    assert all(
                        parameter.grad is None
                        for name, parameter in model.named_parameters()
                        if "lora_" not in name
                    )

            wrapper.trainer.add_callback(Capture())
            wrapper.trainer.args.report_to = []
            wrapper.trainer.args.disable_tqdm = True
            if precision:
                assert wrapper.trainer.args.bf16 == (precision == "bf16")
                assert wrapper.trainer.args.fp16 == (precision == "fp16")
            with torch.autograd.profiler.profile(use_device=None) as trace:
                wrapper.train()
            handle.remove()
            events = {event.name for event in trace.function_events if "FastLora" in event.name}
            if enabled:
                assert wrapper._fast_lora_patches == {"mlp": 4, "qkv": 4, "single": 16}
                for marker in ("SwiGLU", "QKV", "SingleProjection"):
                    assert any(marker in name and "Backward" in name for name in events)
            else:
                assert wrapper._fast_lora_patches == {} and not events
            assert runtime.stats()["layer_loads"] > 4 * len(rows)
            assert len(gradients) == 2 and len(logits) == 4
            assert all(
                torch.isfinite(gradient).all() for step in gradients for gradient in step.values()
            )
            assert any(
                value.abs().sum() > 0 for name, value in gradients[0].items() if "lora_A" in name
            )
            final = {name: parameter.detach().cpu().clone() for name, parameter in adapters.items()}
            assert any(not torch.equal(value, initial[name]) for name, value in final.items())
            # Periodic and final saves must contain all canonical tensors; the
            # public PEFT loader must accept the final adapter on a resident base.
            for directory in (output, output / "checkpoint-1", output / "checkpoint-2"):
                saved = load_file(str(directory / "adapter_model.safetensors"))
                assert len(saved) == 56 and all(".inner." not in name for name in saved)
            loaded = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(base),
                str(output),
                is_trainable=True,
            )
            reloaded = {
                name: value.detach() for name, value in loaded.named_parameters() if "lora_" in name
            }
            assert set(reloaded) == set(final)
            assert all(torch.equal(value, reloaded[name]) for name, value in final.items())
            losses = [row["loss"] for row in wrapper.trainer.state.log_history if "loss" in row]
            results.append((initial, gradients, final, losses, logits))
            runtime_stats.append(runtime.stats())
        finally:
            wrapper._close_stream_runtime()
    assert all(torch.equal(value, results[1][0][name]) for name, value in results[0][0].items())
    rtol, atol = (1e-5, 1e-7) if device == "cpu" else (1e-2, 1e-5)
    for first, second in zip(results[0][1], results[1][1], strict=True):
        for name, gradient in first.items():
            torch.testing.assert_close(gradient, second[name], rtol=rtol, atol=atol)
    for name, parameter in results[0][2].items():
        torch.testing.assert_close(parameter, results[1][2][name], rtol=rtol, atol=atol)
    for first, second in zip(results[0][4], results[1][4], strict=True):
        torch.testing.assert_close(first, second, rtol=rtol, atol=max(atol, 1e-6))
    assert len(results[0][3]) == len(results[1][3]) == 2
    assert all(
        f"{first:.3f}" == f"{second:.3f}"
        for first, second in zip(results[0][3], results[1][3], strict=True)
    )
    if os.environ.get("SOUP_D2_STREAM_EVIDENCE"):
        from safetensors.torch import save_file

        destination = Path(os.environ["SOUP_D2_STREAM_EVIDENCE"])
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"{device}-{precision or 'fp32'}-{source}-{quantization}"
        tensors = {}
        for arm, result in zip(("lora", "fast"), results, strict=True):
            for group, position in (("initial", 0), ("final", 2)):
                for name, value in result[position].items():
                    tensors[f"{arm}/{group}/{name}"] = value.contiguous().clone()
            for step, gradients in enumerate(result[1]):
                for name, value in gradients.items():
                    tensors[f"{arm}/gradient/{step}/{name}"] = value.contiguous().clone()
            for step, value in enumerate(result[4]):
                tensors[f"{arm}/logits/{step}"] = value.contiguous().clone()
            tensors[f"{arm}/loss"] = torch.tensor(result[3], dtype=torch.float64)
        save_file(tensors, str(destination / f"{stem}.safetensors"))
        (destination / f"{stem}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "device": device,
                    "precision": precision or "fp32",
                    "dtype_policy_override": precision is not None,
                    "quantization": quantization,
                    "source": source,
                    "rtol": rtol,
                    "atol": atol,
                    "runtime_stats": runtime_stats,
                    "losses": {arm: result[3] for arm, result in zip(("lora", "fast"), results)},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def test_failed_fast_lora_install_closes_disk_reader(tmp_path, monkeypatch):
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import fast_lora_training, layer_shard

    base = _tiny_streaming_base(tmp_path)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda _base: str(tmp_path / "shards"))
    wrapper = SFTTrainerWrapper(
        _stream_config(base, tmp_path / "output", enabled=True, quantization="none", source="disk"),
        device="cpu",
    )

    def fail_install(_model, *, streaming=False):
        assert streaming
        raise RuntimeError("deliberate Fast-LoRA installation failure")

    monkeypatch.setattr(fast_lora_training, "apply_fast_lora_training", fail_install)
    with pytest.raises(RuntimeError, match="deliberate"):
        wrapper.setup({"train": []})
    runtime = wrapper._stream_runtime
    assert runtime.hook is None
    assert runtime.source._closed and not runtime.source._thread.is_alive()


def test_stream_probe_runs_after_fast_lora_install(tmp_path, monkeypatch):
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_shard

    base = _tiny_streaming_base(tmp_path)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda _base: str(tmp_path / "shards"))
    wrapper = SFTTrainerWrapper(
        _stream_config(base, tmp_path / "output", enabled=True, quantization="none"),
        device="cpu",
    )
    plan = object()
    monkeypatch.setattr(wrapper, "_stream_budget_lines", lambda *_args, **_kwargs: ([], plan))

    def inspect_probe(model, actual_plan):
        assert actual_plan is plan
        assert model._soup_fast_lora_training_counts == {"mlp": 4, "qkv": 4, "single": 16}
        raise RuntimeError("probe reached the patched model")

    monkeypatch.setattr(wrapper, "_run_stream_vram_probe", inspect_probe)
    try:
        with pytest.raises(RuntimeError, match="probe reached"):
            wrapper.setup({"train": []})
    finally:
        wrapper._close_stream_runtime()


def test_non_llama_refused_before_stream_sharding(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from transformers import AutoConfig

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import spectrum_scan

    base = _tiny_streaming_base(tmp_path)
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *_args, **_kwargs: SimpleNamespace(model_type="qwen2")
    )

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("unsupported Fast-LoRA architecture reached weight resolution")

    monkeypatch.setattr(spectrum_scan, "resolve_model_weights", unexpected_load)
    wrapper = SFTTrainerWrapper(
        _stream_config(base, tmp_path / "output", enabled=True, quantization="none"),
        device="cpu",
    )
    with pytest.raises(ValueError, match="Llama"):
        wrapper.setup({"train": []})
