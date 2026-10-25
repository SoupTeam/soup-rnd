"""Tiny native controls for the official Kanana-2 30B-A3B configuration.

Source facts (dimensions are reduced locally, no checkpoint download in tests):
https://huggingface.co/kakaocorp/kanana-2-30b-a3b-base-2601/blob/
0d31365acf656e5f9a593cacc4913e9c0d36cb1c/config.json
No real 30B capacity, training quality or speed claim follows from these cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_two_read_cache_full_query import _full_query_trajectories


def _kanana_overrides() -> dict[str, Any]:
    return {
        "q_lora_rank": None, "rope_interleave": True, "n_shared_experts": 2,
        "n_group": 1, "topk_group": 1, "num_experts_per_tok": 6,
        "topk_method": "noaux_tc", "scoring_func": "sigmoid", "norm_topk_prob": True,
        "routed_scaling_factor": 2.448, "first_k_dense_replace": 1, "moe_layer_freq": 1,
        "attention_bias": False, "attention_dropout": 0.0, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 1000000., "rope_scaling": None,
        # Transformers 5 stores the effective RoPE parameters here; replacing
        # only the legacy theta would leave the old fixture's derived value.
        "rope_parameters": {"rope_theta": 1000000., "rope_type": "default"},
        "tie_word_embeddings": False,
    }


def _assert_saved_native_config(weights: Path) -> Any:
    import torch
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    config = DeepseekV3Config.from_pretrained(weights, local_files_only=True)
    expected = _kanana_overrides()
    for name in ("q_lora_rank", "rope_interleave", "n_shared_experts", "n_group",
                 "topk_group", "num_experts_per_tok", "norm_topk_prob",
                 "routed_scaling_factor", "first_k_dense_replace", "moe_layer_freq",
                 "attention_bias", "attention_dropout", "hidden_act", "rms_norm_eps",
                 "tie_word_embeddings", "scoring_func", "topk_method"):
        assert getattr(config, name) == expected[name], name
    assert config.rope_parameters == expected["rope_parameters"]
    assert config.num_hidden_layers == 3 and config.n_routed_experts == 8
    assert config.n_group != 2  # Shared DeepSeek fixture must not leak its grouped routing.
    model = DeepseekV3ForCausalLM.from_pretrained(weights, local_files_only=True,
                                               dtype=torch.float32).eval()
    model.set_experts_implementation("eager")
    layers = model.model.layers
    assert sum(not hasattr(layer.mlp, "experts") for layer in layers) == 1
    assert not hasattr(layers[0].mlp, "experts")
    for layer in layers:
        assert hasattr(layer.self_attn, "q_proj") and layer.self_attn.q_lora_rank is None
        assert layer.self_attn.config.rope_interleave is True
    for layer in layers[1:]:
        assert layer.mlp.shared_experts.gate_proj.out_features == 2 * config.moe_intermediate_size
        assert layer.mlp.shared_experts.down_proj.in_features == 2 * config.moe_intermediate_size
        router = layer.mlp.gate
        assert router.num_group == router.topk_group == 1
        assert router.top_k == 6 and router.norm_topk_prob is True
        hidden = torch.randn(1, 12, config.hidden_size, generator=torch.Generator().manual_seed(9))
        with torch.no_grad():
            logits, weights, indices = router(hidden)
            expected_logits = torch.nn.functional.linear(hidden.view(-1, config.hidden_size),
                                                         router.weight.float())
            sigmoid = expected_logits.sigmoid()
            # With one selected group every expert remains eligible. This is
            # the noaux correction-bias selection, not softmax or group-two routing.
            expected_indices = (sigmoid + router.e_score_correction_bias).topk(
                6, dim=-1, sorted=False,
            ).indices
            expected_weights = sigmoid.gather(1, expected_indices)
            expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20
            expected_weights *= 2.448
        assert torch.equal(logits, expected_logits)
        assert torch.equal(indices, expected_indices)
        assert torch.equal(weights, expected_weights)
    return config


@pytest.mark.parametrize("dtype,quant", [("bfloat16", "none"), ("float32", "nf4")])
def test_kanana_native_config_preserves_four_arm_cpu_trajectories(
    tmp_path: Path, dtype: str, quant: str,
) -> None:
    reports = _full_query_trajectories(
        tmp_path, dtype=dtype, quant=quant, device="cpu", bitwise=True,
        host_cache_policy="static", config_overrides=_kanana_overrides(),
    )
    assert set(reports) == {"ordinary", "two_read", "cache", "two_read_cache"}
    assert all(report["comparison"]["passed"] and report["comparison"]["bitwise"]
               for report in reports.values())
    _assert_saved_native_config(tmp_path / "model")


@pytest.mark.gpu(reason="Kanana full-query/group-one/shared-two CUDA NF4 BF16 parity needs a GPU")
def test_real_cuda_nf4_bf16_kanana_worker_trajectories(
    tmp_path: Path, monkeypatch: Any, record_property: Any,
) -> None:
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
            host_cache_policy="static", config_overrides=_kanana_overrides(),
            evidence_dir=tmp_path / "kanana-gpu-evidence",
        )
        _assert_saved_native_config(tmp_path / "model")
        record_property("kanana_four_arm_comparisons", json.dumps(reports, sort_keys=True))
        record_property("kanana_gpu_evidence_directory", str(tmp_path / "kanana-gpu-evidence"))
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
