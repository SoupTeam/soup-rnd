"""Whole-model integration controls; tiny CPU fixtures are not a trained CUDA gate."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/harness"))
SPEC = importlib.util.spec_from_file_location(
    "d3_full_model", ROOT / "benchmarks/harness/d3_full_model.py",
)
FULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL)


@pytest.fixture
def prepared(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    torch.manual_seed(921)
    config = OlmoeConfig(
        vocab_size=48, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = OlmoeForCausalLM(config).requires_grad_(False).eval()
    state = {name: value for name, value in model.state_dict().items()
             if ".mlp.experts." not in name}

    def weights():
        for index, layer in enumerate(model.model.layers):
            yield index, zip(layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj)

    folder = tmp_path / "model"
    FULL.prepare_bundle(config, state, weights(), folder, codec="dense", quant_device="cpu",
                        provenance={"fixture": "tiny synthetic CPU model"})
    return model, folder


@pytest.mark.parametrize("backend", ["sequential", "grouped-restore-control"])
@pytest.mark.parametrize("tokens", [1, 17, 64])
def test_native_full_logits_and_single_pool_across_all_layers(prepared, backend, tokens):
    import torch

    original, folder = prepared
    ids = torch.arange(tokens).remainder(original.config.vocab_size).reshape(1, -1)
    with FULL.open_streamed_model(folder, device="cpu", backend=backend) as run, torch.no_grad():
        expected = original(ids, use_cache=False).logits
        observed = run.forward(ids)
        assert torch.equal(observed, expected)
        assert torch.isfinite(observed).all()
        assert observed.shape == (1, tokens, original.config.vocab_size)
        assert run.model.config.use_cache is False
        assert all(layer.mlp.experts.pool is run.pool for layer in run.model.model.layers)
        traces = run.traces()
        for index, rows in enumerate(traces):
            assert rows
            assert [row["expert"] for row in rows] == sorted({row["expert"] for row in rows})
            assert all(row["layer"] == index and row["status"] == "ok" for row in rows)
            assert all(row["logical_bytes_read"] > 0 for row in rows)
        assert run.pool.fetches == sum(len(rows) for rows in traces)
        assert not any(value.is_meta for value in run.model.parameters())
        assert not any(value.is_meta for value in run.model.buffers())
        assert run.model.model.rotary_emb.inv_freq.dtype == torch.float32
        for name, value in run.model.state_dict().items():
            assert ".mlp.experts." not in name
            assert torch.equal(value, original.state_dict()[name])
    assert run.pool._closed
    assert all(reader._handle is None for reader in run.readers)


@pytest.mark.parametrize("backend", ["sequential", "grouped-restore-control"])
def test_full_logits_active_weight_negative_and_restored_positive(prepared, backend):
    import torch

    from soup_cli.utils.expert_shard import expert_key

    original, folder = prepared
    ids = torch.arange(17).reshape(1, -1)
    with FULL.open_streamed_model(folder, device="cpu", backend=backend) as run, torch.no_grad():
        expected = original(ids, use_cache=False).logits
        assert torch.equal(run.forward(ids), expected)
        target = run.traces()[0][0]["expert"]
        reader = run.readers[0]
        read = reader.read_host

        def corrupt(expert, staging):
            values = read(expert, staging)
            if expert == target:
                values[expert_key(expert, "down")].view(-1)[0].add_(1)
            return values

        reader.read_host = corrupt
        assert not torch.equal(run.forward(ids), expected)
        reader.read_host = read
        assert torch.equal(run.forward(ids), expected)


def test_training_and_autograd_are_not_silently_enabled(prepared):
    import torch

    _, folder = prepared
    with FULL.open_streamed_model(folder, device="cpu", backend="sequential") as run:
        with pytest.raises(RuntimeError, match="forward-only"):
            run.forward(torch.tensor([[2, 3]]))
        run.model.train()
        with torch.no_grad(), pytest.raises(RuntimeError, match="eval mode"):
            run.forward(torch.tensor([[2, 3]]))


def test_independent_full_resident_native_oracle_matches_original(prepared):
    import torch

    original, folder = prepared
    ids = torch.arange(17).reshape(1, -1)
    with FULL.open_resident_model(folder, device="cpu") as run, torch.no_grad():
        expected = original(ids, use_cache=False).logits
        assert torch.equal(run.forward(ids), expected)
        assert len(run.resident_experts) == original.config.num_hidden_layers
        assert all(rows for rows in run.traces())
        expert = run.traces()[0][0]["expert"]
        down = run.resident_experts[0].down_proj[expert]
        saved = down.view(-1)[0].clone()
        down.view(-1)[0].add_(1)
        assert not torch.equal(run.forward(ids), expected)
        down.view(-1)[0].copy_(saved)
        assert torch.equal(run.forward(ids), expected)


@pytest.mark.parametrize("codec", ["nf4", "nf4-dq"])
def test_full_quantized_model_matches_independent_resident_dequantization(
    prepared, tmp_path, codec,
):
    import torch

    pytest.importorskip("bitsandbytes")
    model, _ = prepared
    state = {name: value for name, value in model.state_dict().items()
             if ".mlp.experts." not in name}

    def layers():
        for index, layer in enumerate(model.model.layers):
            yield index, zip(layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj)

    folder = tmp_path / codec
    FULL.prepare_bundle(model.config, state, layers(), folder, codec=codec, quant_device="cpu",
                        provenance={"fixture": "tiny synthetic quantized model"})
    ids = torch.arange(17).reshape(1, -1)
    with FULL.open_resident_model(folder, device="cpu") as resident, torch.no_grad():
        expected = resident.forward(ids)
        for backend in ["sequential", "grouped-restore-control"]:
            with FULL.open_streamed_model(folder, device="cpu", backend=backend) as run:
                assert torch.equal(run.forward(ids), expected)


def test_actual_grouped_compute_cannot_be_replaced_by_cpu_control(prepared):
    _, folder = prepared
    with pytest.raises(ValueError, match="native BF16 CUDA"):
        FULL.open_streamed_model(folder, device="cpu", backend="grouped-compute")


def test_closed_whole_model_owner_is_not_reusable(prepared):
    import torch

    _, folder = prepared
    run = FULL.open_streamed_model(folder, device="cpu", backend="sequential")
    run.close()
    assert run.model is None
    with torch.no_grad(), pytest.raises(RuntimeError, match="closed"):
        run.forward(torch.tensor([[2, 3]]))


@pytest.mark.parametrize("codec", ["dense", "nf4", "nf4-dq"])
def test_tiny_cuda_full_model_uses_actual_grouped_backend(prepared, tmp_path, codec):
    """Development smoke only; this does not satisfy the trained 16 GB+ gate."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA development smoke requires a GPU")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("native BF16 required")
    library = os.environ.get("D3_CUBLAS_LIBRARY")
    if not library:
        pytest.skip("provide the explicit D3_CUBLAS_LIBRARY for the CUDA development smoke")
    from d3_cublas_grouped_probe import reference_policy

    source, _ = prepared
    source.to(dtype=torch.bfloat16)
    state = {name: value for name, value in source.state_dict().items()
             if ".mlp.experts." not in name}

    def layers():
        for index, layer in enumerate(source.model.layers):
            yield index, zip(layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj)

    folder = tmp_path / ("cuda-" + codec)
    FULL.prepare_bundle(source.config, state, layers(), folder, codec=codec, quant_device="cuda",
                        provenance={"scope": "tiny synthetic CUDA integration smoke"})
    ids = torch.arange(64, device="cuda").remainder(source.config.vocab_size).reshape(1, -1)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with reference_policy("cublaslt-no-splitk"), torch.no_grad():
            with FULL.open_resident_model(folder, device="cuda") as native:
                expected = native.forward(ids).cpu()
            with FULL.open_streamed_model(folder, device="cuda", backend="sequential") as run:
                assert torch.equal(run.forward(ids).cpu(), expected)
            with FULL.open_resident_model(folder, device="cuda", backend="grouped-compute",
                                           library=Path(library),
                                           native_olmoe_rows_max=128) as run:
                assert torch.equal(run.forward(ids).cpu(), expected)
            with FULL.open_streamed_model(folder, device="cuda", backend="grouped-compute",
                                           library=Path(library),
                                           native_olmoe_rows_max=128) as run:
                assert torch.equal(run.forward(ids).cpu(), expected)
                assert run.gemm.stats()["grouped_matrices"] > 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


@pytest.mark.parametrize("name", ["nonexpert.safetensors", "experts_001.safetensors"])
def test_modified_checkpoint_is_rejected_before_opening_model(prepared, name):
    _, folder = prepared
    path = folder / name
    value = bytearray(path.read_bytes())
    value[-1] ^= 1
    path.write_bytes(value)
    with pytest.raises(ValueError, match="hash differs"):
        FULL.open_streamed_model(folder, device="cpu", backend="sequential")


def test_missing_layer_is_not_a_whole_model(prepared):
    _, folder = prepared
    path = folder / "bundle.json"
    data = json.loads(path.read_bytes())
    data["layers"].pop()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="layer count"):
        FULL.open_streamed_model(folder, device="cpu", backend="sequential")


def test_reordered_layers_are_rejected(prepared):
    _, folder = prepared
    path = folder / "bundle.json"
    data = json.loads(path.read_bytes())
    data["layers"].reverse()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="reordered"):
        FULL.validate_bundle(folder)


def test_budget_refusal_before_expert_reads(prepared, monkeypatch):
    from soup_cli.utils.expert_shard import ExpertShardReader

    _, folder = prepared
    calls = []
    monkeypatch.setattr(ExpertShardReader, "read_host", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="budget"):
        FULL.open_streamed_model(folder, device="cpu", backend="sequential",
                                 weight_budget_bytes=1)
    assert calls == []


def test_preparation_failure_is_retained_not_relabelled_complete(prepared, tmp_path):
    model, _ = prepared
    output = tmp_path / "incomplete"
    state = {name: value for name, value in model.state_dict().items()
             if ".mlp.experts." not in name}
    with pytest.raises(ValueError, match="expert layer is missing"):
        FULL.prepare_bundle(model.config, state, iter(()), output, codec="dense",
                            quant_device="cpu", provenance={"fixture": "incomplete"})
    assert json.loads((output / "bundle.json").read_bytes())["complete"] is False
    with pytest.raises(ValueError, match="did not complete"):
        FULL.validate_bundle(output)


@pytest.mark.parametrize("name", ["../elsewhere", "/tmp/elsewhere"])
def test_bundle_path_containment(tmp_path, name):
    with pytest.raises(ValueError, match="escapes"):
        FULL.contained(tmp_path, name)
