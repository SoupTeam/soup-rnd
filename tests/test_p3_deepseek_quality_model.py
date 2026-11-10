"""Native checkpoint/cache and strict shared-adapter contracts, qualified remotely only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import p3_deepseek_quality_attention as attention
from benchmarks.harness import p3_deepseek_quality_model as quality


def _training_stack(device: str) -> Any:
    if device == "cuda":
        import bitsandbytes as bnb
        import peft  # noqa: F401
        import torch
        import transformers  # noqa: F401
    else:
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        bnb = pytest.importorskip("bitsandbytes")
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        attention.require_ampere_cuda(device)
    if device == "cpu":
        try:
            packed, state = bnb.functional.quantize_4bit(
                torch.ones((64, 64), dtype=torch.bfloat16), quant_type="nf4",
                compress_statistics=False,
            )
            bnb.functional.dequantize_4bit(packed, state)
        except (NotImplementedError, RuntimeError) as error:
            message = str(error).lower()
            unsupported = ("not implemented", "no kernel", "not available for this backend",
                           "compiled without gpu support", "requires cuda")
            if any(message_part in message for message_part in unsupported):
                pytest.skip(f"installed bitsandbytes lacks CPU NF4: {error}")
            raise
    return torch


def _checkpoint(root: Path, *, exact_mla: bool = False) -> Path:
    from benchmarks.harness.b13_moe_fixture import moe_config, moe_weights_dir

    shape = ({"num_attention_heads": 128, "num_key_value_heads": 128,
              "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128}
             if exact_mla else {})
    return moe_weights_dir(root, moe_config(
        hidden_size=128, intermediate_size=128, moe_intermediate_size=64,
        num_hidden_layers=2, q_lora_rank=32, kv_lora_rank=32,
        max_position_embeddings=4096, attention_dropout=0.0, **shape,
    ), seed=23)


def _published_config_checkpoint(root: Path, **declared: Any) -> Path:
    """Keep tiny weights but use the pinned V3 Hub config shape, not a native save."""
    source = _checkpoint(root)
    path = source / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for name in ("dtype", "head_dim", "qk_head_dim", "rope_parameters", "rope_interleave",
                 "output_router_logits", "pad_token_id", "pretraining_tp"):
        config.pop(name, None)
    # Pinned V3 names Hub code but does not declare either auxiliary coefficient.
    config.update(
        auto_map={
            "AutoConfig": "configuration_deepseek.DeepseekV3Config",
            "AutoModel": "modeling_deepseek.DeepseekV3Model",
            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
        },
        ep_size=1, moe_layer_freq=1, scoring_func="sigmoid", topk_method="noaux_tc",
        max_position_embeddings=163840, use_cache=True, rope_theta=10000.0,
        torch_dtype="bfloat16", transformers_version="4.33.1",
        rope_scaling={
            "beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 1.0,
            "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096, "type": "yarn",
        },
        quantization_config={
            "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    config.update(declared)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return source


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_same_native_nf4_disk_path_materializes_every_attention_adapter(
    tmp_path: Path, device: str, dtype: str,
) -> None:
    torch = _training_stack(device)
    source = _checkpoint(tmp_path)
    cache = tmp_path / "cache"
    with quality.open_session(source, cache, device=device, dtype=dtype, prepare=True) as session:
        params = quality.trainable_parameters(session.model)
        assert len(params) == 20
        assert all(value.dtype == torch.float32 and not value.is_meta for value in params.values())
        assert all(".self_attn." in name and ".inner." not in name for name in params)
        assert session.runtime.stats()["tier"] == "disk"
        assert session.runtime.expert_pool.dtype == dtype
        assert session.metadata["cache"]["quant"] == "nf4"
        assert session.metadata["cache"]["double_quant"] is False
        assert session.metadata["cache"]["expert_layout"] == "per_expert"
        assert session.metadata["precision"]["attention_implementation"] == "sdpa"
        assert session.metadata["source"]["production_qualified"] is False
        with pytest.raises(ValueError, match="production"):
            quality.require_production_source(session.metadata["source"])
        if device == "cuda":
            ids = torch.tensor([[1, 2, 3, 4]], device=device)
            with attention.attention_execution("EFFICIENT_ATTENTION", device=device) as observed:
                loss = session.model(input_ids=ids, labels=ids).loss
                loss.backward()
                observed.require_complete(dtype=dtype, min_calls=2)
            assert bool(torch.isfinite(loss))
        # The CPU case verifies only cache and adapter serialization, not attention numerics.
        identity = session.metadata
    with quality.open_session(source, cache, device=device, dtype=dtype) as reopened:
        assert reopened.metadata == identity
    assert session.closed is True


@pytest.mark.gpu
def test_auto_map_pinned_v3_shape_preserves_native_objective_and_both_arms(
    tmp_path: Path,
) -> None:
    device = "cuda"
    # Loading Hub defaults, stripping source fields, or changing one arm's objective fails here.
    torch = _training_stack(device)
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    from benchmarks.harness.p3_deepseek_quality_train import run_window
    from soup_cli.utils.l2l_step import model_parts

    source = _published_config_checkpoint(tmp_path)
    config_path = source / "config.json"
    source_bytes = config_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_config = json.loads(source_bytes)
    assert "aux_loss_alpha" not in source_config and "router_aux_loss_coef" not in source_config
    with quality.open_session(
        source, tmp_path / "cache", device=device, dtype="bfloat16", prepare=True,
    ) as session:
        causal, owner = model_parts(session.model)
        assert type(causal) is DeepseekV3ForCausalLM
        for config in (session.config, causal.config, owner.config):
            assert type(config) is DeepseekV3Config
            assert config.auto_map == source_config["auto_map"]
            assert getattr(config, "aux_loss_alpha", 0) in (0, None)
            assert getattr(config, "router_aux_loss_coef", 0) in (0, None)
            assert config.output_router_logits is False
        assert session.metadata["source"]["config_sha256"] == source_hash
        initial = tmp_path / "initial.safetensors"
        quality.write_initial_adapter(session.model, initial)
        ids = torch.tensor([[1, 2, 3, 4]], device=device)
        with torch.no_grad(), attention.attention_execution(
            "EFFICIENT_ATTENTION", device=device,
        ) as observed:
            output = session.model(input_ids=ids, labels=ids)
            observed.require_complete(dtype="bfloat16", backward=False, min_calls=2)
        expected_ce = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].float().reshape(-1, session.config.vocab_size),
            ids[:, 1:].reshape(-1),
        )
        torch.testing.assert_close(output.loss, expected_ce, rtol=1e-6, atol=1e-6)
        assert output.router_logits is None and getattr(output, "aux_loss", None) is None
        window = {
            "window_id": "legacy-auto-map",
            "micro_batches": [{"input_ids": [[1, 2, 3, 4]], "labels": [[-100, 2, 3, 4]]}],
            "valid_target_count": 3, "input_token_count": 4,
        }
        records = {}
        for arm in ("bf16_l2l", "oracle_block64_e4m3"):
            quality.load_initial_adapter(session.model, initial)
            record, gradients = run_window(session, window, arm=arm, capture_gradients=True)
            assert record["status"] == "COMPLETE"
            assert record["routing_guard"]["outcome"] == "PASSED"
            attention.require_execution_evidence(
                record["attention"], backend="EFFICIENT_ATTENTION", dtype="bfloat16", min_calls=2,
            )
            assert set(gradients) == set(quality.trainable_parameters(session.model))
            assert all(bool(torch.isfinite(value).all()) for value in gradients.values())
            records[arm] = record
        baseline, candidate = records["bf16_l2l"], records["oracle_block64_e4m3"]
        assert baseline["adapter_before_sha256"] == candidate["adapter_before_sha256"]
        assert candidate["loss"] == baseline["loss"]
        assert candidate["oracle_before_close"]["complete"] is True
        assert candidate["oracle_before_close"]["counts"] == {
            "native_forward": 1, "oracle_recompute": 1, "oracle_vjp": 1,
        }
    assert config_path.read_bytes() == source_bytes
    assert quality.sha256(config_path) == source_hash


@pytest.mark.parametrize("declared", [
    # r1-config-kaggle-v1 recovered Input /2 SHA4422016f...a27 with these declarations.
    {"aux_loss_alpha": 0.001, "seq_aux": True, "pretraining_tp": 1,
     "transformers_version": "4.46.3"},
    {"router_aux_loss_coef": 0.125},
    {"output_router_logits": True},
])
def test_auto_map_source_declared_consumers_are_not_cleared_before_oracle(
    tmp_path: Path, declared: dict[str, Any],
) -> None:
    _training_stack("cpu")
    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe
    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    source = _published_config_checkpoint(tmp_path, **declared)
    source_bytes = (source / "config.json").read_bytes()
    with quality.open_session(source, tmp_path / "cache", device="cpu", prepare=True) as session:
        for field, value in declared.items():
            assert getattr(session.config, field) == value
        budget = HostMemoryBudget(1 << 26)
        with pytest.raises(RuntimeError, match="auxiliary|router-logit"):
            DeepseekOracleProbe(session.model, [(1, 4, 128)], budget)
        assert budget.used_bytes == 0
        for field, value in declared.items():
            assert getattr(session.config, field) == value
    assert (source / "config.json").read_bytes() == source_bytes


def test_prepare_does_not_replace_a_mismatched_committed_cache(tmp_path: Path) -> None:
    _training_stack("cpu")
    source = _checkpoint(tmp_path)
    cache = tmp_path / "cache"
    with quality.open_session(source, cache, device="cpu", prepare=True):
        pass
    before = (cache / "index.json").read_bytes()
    with pytest.raises(ValueError, match="cache"):
        quality.open_session(source, cache, device="cpu", dtype="float32", prepare=True)
    assert (cache / "index.json").read_bytes() == before


def test_cache_is_required_unless_preparation_was_explicit(tmp_path: Path) -> None:
    _training_stack("cpu")
    source = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="cache"):
        quality.open_session(source, tmp_path / "absent", device="cpu")
    assert not (tmp_path / "absent").exists()


def test_seeded_shared_adapter_uses_frozen_nonzero_b_scale_and_exclusive_file(
    tmp_path: Path,
) -> None:
    torch = _training_stack("cpu")
    from safetensors.torch import load_file

    source = _checkpoint(tmp_path)
    cache = tmp_path / "cache"
    initial = tmp_path / "initial.safetensors"
    with quality.open_session(source, cache, device="cpu", prepare=True) as first:
        receipt = quality.write_initial_adapter(first.model, initial)
        saved = load_file(str(initial))
        generator = torch.Generator().manual_seed(23)
        for name, value in sorted(saved.items()):
            if "lora_B" in name:
                expected = torch.randn(value.shape, generator=generator) * 0.02
                assert torch.equal(value, expected)
        repeated = tmp_path / "initial-repeated.safetensors"
        repeated_receipt = quality.write_initial_adapter(first.model, repeated)
        assert repeated_receipt["parameter_sha256"] == receipt["parameter_sha256"]
        assert all(torch.equal(value, load_file(str(repeated))[name])
                   for name, value in saved.items())
        assert receipt["sha256"] == hashlib.sha256(initial.read_bytes()).hexdigest()
        with pytest.raises(FileExistsError):
            quality.write_initial_adapter(first.model, initial)
    with quality.open_session(source, cache, device="cpu") as second:
        loaded_hash = quality.load_initial_adapter(second.model, initial)
        assert loaded_hash == receipt["sha256"]
        assert all(torch.equal(value.detach().cpu(), saved[name])
                   for name, value in quality.trainable_parameters(second.model).items())


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "dtype", "nonfinite"])
def test_shared_adapter_refuses_partial_or_invalid_load_without_mutation(
    tmp_path: Path, damage: str,
) -> None:
    torch = _training_stack("cpu")
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    source = _checkpoint(tmp_path)
    with quality.open_session(source, tmp_path / "cache", device="cpu", prepare=True) as session:
        good = tmp_path / "good.safetensors"
        quality.write_initial_adapter(session.model, good)
        params = quality.trainable_parameters(session.model)
        before = {name: value.detach().clone() for name, value in params.items()}
        tensors = load_file(str(good))
        with safe_open(str(good), framework="pt") as handle:
            metadata = handle.metadata()
        key = sorted(tensors)[0]
        if damage == "missing":
            tensors.pop(key)
        elif damage == "extra":
            tensors["unexpected"] = torch.ones(1)
        elif damage == "shape":
            tensors[key] = tensors[key].flatten()
        elif damage == "dtype":
            tensors[key] = tensors[key].to(torch.bfloat16)
        else:
            tensors[key].flatten()[0] = float("nan")
        bad = tmp_path / "bad.safetensors"
        save_file(tensors, str(bad), metadata=metadata)
        with pytest.raises(ValueError):
            quality.load_initial_adapter(session.model, bad)
        assert all(torch.equal(value, before[name]) for name, value in params.items())


def test_unpinned_source_receipt_cannot_claim_production(tmp_path: Path) -> None:
    _training_stack("cpu")
    source = _checkpoint(tmp_path)
    receipt = {
        "model_id": "deepseek-ai/DeepSeek-V3", "revision": "other",
        "source_files": {path.name: {"bytes": path.stat().st_size,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in source.iterdir() if path.is_file()},
    }
    (source / "checkpoint-identity.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        quality.open_session(source, tmp_path / "cache", device="cpu", prepare=True)


def test_source_receipt_rejects_path_escape_before_reading_file(tmp_path: Path) -> None:
    _training_stack("cpu")
    source = _checkpoint(tmp_path)
    receipt = {
        "model_id": "deepseek-ai/DeepSeek-V3", "revision": quality.REVISION,
        "source_files": {"../outside": {"bytes": 1, "sha256": "a" * 64}},
    }
    (source / "checkpoint-identity.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        quality.open_session(source, tmp_path / "cache", device="cpu", prepare=True)


def test_production_flag_without_inventory_never_admits_training() -> None:
    metadata = {
        "production_qualified": True, "model_id": quality.MODEL_ID,
        "revision": quality.REVISION, "shape": dict(quality._FULL_SHAPE),
        "weight_files": 163, "weight_bytes": quality.SOURCE_BYTES,
        "source_verification": "hub_commit_etag_and_size_plus_cache_source_fingerprint",
        "receipt_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="inventory"):
        quality.require_production_source(metadata)


def test_cache_missing_an_expert_shard_is_refused_without_repair(tmp_path: Path) -> None:
    _training_stack("cpu")
    from soup_cli.utils.layer_shard import expert_paths, read_shard_index

    source = _checkpoint(tmp_path)
    cache = tmp_path / "cache"
    with quality.open_session(source, cache, device="cpu", prepare=True):
        pass
    index = read_shard_index(str(cache))
    expert = Path(next(iter(expert_paths(str(cache), index).values())))
    expert.unlink()
    with pytest.raises(ValueError, match="missing"):
        quality.open_session(source, cache, device="cpu", prepare=True)
    assert not expert.exists()


def test_attention_dropout_is_refused_before_creating_cache(tmp_path: Path) -> None:
    _training_stack("cpu")
    source = _checkpoint(tmp_path)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["attention_dropout"] = 0.1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="dropout"):
        quality.open_session(source, cache, device="cpu", prepare=True)
    assert not cache.exists()


@pytest.mark.parametrize("damage", [
    "missing", "schema", "seed", "scale", "initializer", "digest", "changed_tensor",
    "wrong_seed_forged_metadata", "trained_forged_digest", "trained_a_forged_digest", "checkpoint",
])
def test_initial_adapter_provenance_refuses_mutation_before_any_copy(
    tmp_path: Path, damage: str,
) -> None:
    torch = _training_stack("cpu")
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    source = _checkpoint(tmp_path)
    with quality.open_session(source, tmp_path / "cache", device="cpu", prepare=True) as session:
        good = tmp_path / "initial.safetensors"
        quality.write_initial_adapter(session.model, good)
        params = quality.trainable_parameters(session.model)
        before = {name: value.detach().clone() for name, value in params.items()}
        tensors = load_file(str(good))
        with safe_open(str(good), framework="pt") as handle:
            metadata = handle.metadata()
        if damage == "missing":
            metadata = {}
        elif damage == "schema":
            metadata["schema"] = "unrelated-adapter-v1"
        elif damage == "seed":
            metadata["seed"] = "24"
        elif damage == "scale":
            metadata["nonzero_B_scale"] = "0.05"
        elif damage == "initializer":
            metadata["initializer"] = "different.initializer"
        elif damage == "digest":
            metadata["parameter_sha256"] = "f" * 64
        elif damage == "changed_tensor":
            tensors[sorted(tensors)[0]].flatten()[0] += 1
        elif damage == "wrong_seed_forged_metadata":
            generator = torch.Generator().manual_seed(24)
            for name, tensor in sorted(tensors.items()):
                if "lora_B" in name:
                    tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.02)
            metadata["parameter_sha256"] = quality.adapter_digest(tensors)[0]
        elif damage == "trained_forged_digest":
            name = next(name for name in tensors if "lora_B" in name)
            tensors[name].flatten()[0] += 0.125
            metadata["parameter_sha256"] = quality.adapter_digest(tensors)[0]
        elif damage == "trained_a_forged_digest":
            name = next(name for name in tensors if "lora_A" in name)
            tensors[name].flatten()[0] += 0.125
            metadata["parameter_sha256"] = quality.adapter_digest(tensors)[0]
        else:
            metadata = {"schema": "quality-checkpoint-adapter-v1"}
        bad = tmp_path / "bad.safetensors"
        save_file(tensors, str(bad), metadata=metadata)
        with pytest.raises(ValueError):
            quality.load_initial_adapter(session.model, bad)
        assert all(torch.equal(value, before[name]) for name, value in params.items())


def test_checkpoint_adapter_loader_is_distinct_from_seeded_initial_admission(
    tmp_path: Path,
) -> None:
    torch = _training_stack("cpu")
    from safetensors.torch import load_file, save_file

    source = _checkpoint(tmp_path)
    with quality.open_session(source, tmp_path / "cache", device="cpu", prepare=True) as session:
        initial = tmp_path / "initial.safetensors"
        quality.write_initial_adapter(session.model, initial)
        trained = load_file(str(initial))
        for value in trained.values():
            value.add_(0.125)
        checkpoint = tmp_path / "adapter.safetensors"
        save_file(trained, str(checkpoint), metadata={"schema": "quality-checkpoint-adapter-v1"})
        digest = quality.load_checkpoint_adapter(session.model, checkpoint)
        assert digest == quality.sha256(checkpoint)
        assert all(torch.equal(value.cpu(), trained[name])
                   for name, value in quality.trainable_parameters(session.model).items())
        with pytest.raises(ValueError):
            quality.load_checkpoint_adapter(session.model, initial)
        assert quality.load_initial_adapter(session.model, initial) == quality.sha256(initial)
        original = load_file(str(initial))
        assert all(torch.equal(value.cpu(), original[name])
                   for name, value in quality.trainable_parameters(session.model).items())
