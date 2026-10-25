"""Moonlight's full-rank query must qualify through the same four worker arms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize("dtype,quant", [("bfloat16", "none"), ("float32", "nf4")])
def test_full_rank_query_preserves_four_arm_worker_trajectories(
    tmp_path: Path, dtype: str, quant: str,
) -> None:
    _full_query_trajectories(tmp_path, dtype=dtype, quant=quant, device="cpu", bitwise=True)


@pytest.mark.gpu(reason="Full-rank query, CUDA NF4 and BF16 autocast need a real GPU")
def test_real_cuda_nf4_full_rank_query_bf16_worker_trajectories(
    tmp_path: Path, monkeypatch: Any, record_property: Any,
) -> None:
    """Cheap Moonlight-shaped attention preflight before downloading real weights."""
    import torch

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reports = _full_query_trajectories(
            tmp_path, dtype="bfloat16", quant="nf4", device="cuda", bitwise=False,
            evidence_dir=tmp_path / "gpu-evidence",
        )
        # Bitwise equality is reported, never an extra CUDA acceptance gate.
        record_property("cuda_four_arm_comparisons", json.dumps(reports, sort_keys=True))
        record_property("cuda_evidence_directory", str(tmp_path / "gpu-evidence"))
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def _full_query_trajectories(
    tmp_path: Path, *, dtype: str, quant: str, device: str, bitwise: bool,
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType
    from transformers import DeepseekV3ForCausalLM

    from benchmarks.harness.two_read_cache import attention_targets, validate_attention_targets
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )
    from tests.test_expert_stream import _shard, _tiny_deepseek_v3

    # Rebuild only this test's checkpoint; do not change the shared rank-24 fixture.
    weights, original = _tiny_deepseek_v3(tmp_path)
    config = original.config
    del original
    config.q_lora_rank = None
    torch.manual_seed(11)
    resident = DeepseekV3ForCausalLM(config).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for layer in resident.model.layers[1:]:
            bias = layer.mlp.gate.e_score_correction_bias
            bias.copy_(torch.randn(bias.shape, generator=generator))
    resident.save_pretrained(weights)
    del resident

    targets = attention_targets(config)
    assert targets == ["q_proj", "kv_b_proj", "o_proj"]
    skeleton = build_meta_skeleton(weights, dtype=dtype, quant=quant)
    validate_attention_targets(skeleton, targets)
    suffixes = quantised_layer_suffixes(skeleton) if quant == "nf4" else ()
    del skeleton
    index = _shard(
        tmp_path, weights, dtype=dtype, quant=quant, expert_layout="per_expert",
        quant_suffixes=suffixes, quant_device=device if quant == "nf4" else None,
    )

    def build(two_read: bool) -> tuple[Any, Any]:
        lora = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            target_modules=targets, task_type=TaskType.CAUSAL_LM,
        )
        model, runtime = build_streamed_model(
            model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
            lora_config=lora, device=device, dtype=dtype, quant=quant, buffers=2,
            pin=False, seed=3, two_read_experts=two_read,
        )
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * .02)
        return model.eval(), runtime

    ordinary, default_runtime = build(False)
    try:
        two, two_runtime = build(True)
    except BaseException:
        default_runtime.close()
        raise
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(29))
    return _qualify_full_query_captures(
        ordinary, default_runtime, two, two_runtime, ids, bitwise=bitwise,
        evidence_dir=evidence_dir,
    )


def _qualify_full_query_captures(
    ordinary: Any, default_runtime: Any, two: Any, two_runtime: Any, ids: Any,
    *, bitwise: bool, evidence_dir: Path | None,
) -> dict[str, Any]:
    import torch

    from benchmarks.harness.two_read_cache_common import (
        atomic_json,
        compare_captures,
        sha256_file,
    )
    from benchmarks.harness.two_read_cache_worker import _adapter_values, _adapters, _run_accuracy

    spec = {"accuracy_updates": 2, "accumulation": 2}
    batches = torch.stack((ids[:1], ids[1:], ids[1:], ids[:1]))
    initial = _adapter_values(ordinary)
    reference = None
    reports = {}
    try:
        for name, model, runtime, cache in (
            ("ordinary", ordinary, default_runtime, False),
            ("two_read", two, two_runtime, False),
            ("cache", ordinary, default_runtime, True),
            ("two_read_cache", two, two_runtime, True),
        ):
            model.zero_grad(set_to_none=True)
            adapters = _adapters(model)
            assert adapters.keys() == initial.keys()
            with torch.no_grad():
                for key, parameter in adapters.items():
                    parameter.copy_(initial[key])
            assert all(torch.equal(value, initial[key])
                       for key, value in _adapter_values(model).items())
            runtime.expert_fetcher.configure_cache(
                host_cache_bytes=2**20 if cache else 0,
                device_cache_bytes=2**15 if cache else 0,
            )
            capture = _run_accuracy(model, runtime, batches, spec)
            assert len(capture["microbatches"]) == 4
            assert len(capture["updates"]) == 2
            assert all(torch.equal(item["input_ids"], batches[index])
                       for index, item in enumerate(capture["microbatches"]))
            if reference is None:
                reference = capture
            comparison = compare_captures(reference, capture)
            stats = runtime.expert_fetcher.cache_stats()
            reports[name] = {"comparison": comparison, "expert_cache": stats}
            if evidence_dir is not None:
                evidence_dir.mkdir(parents=True, exist_ok=True)
                path = evidence_dir / f"{name}.pt"
                torch.save(capture, path)
                reports[name]["capture_sha256"] = sha256_file(path)
                atomic_json(evidence_dir / "comparisons.json", reports)
            assert comparison["passed"], (name, comparison)
            if bitwise:
                assert comparison["bitwise"], (name, comparison)
            if cache:
                assert stats["host"]["hits"] > 0, (name, stats)
                assert stats["device"]["hits"] > 0, (name, stats)
    finally:
        default_runtime.close()
        two_runtime.close()
    return reports
