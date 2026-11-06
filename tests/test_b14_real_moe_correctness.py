"""B1.4 ticket 08: harness that compares per_expert streaming with a resident model.

The harness (``benchmarks/harness/b14_real_moe_correctness.py``) takes any native
DeepSeek-V3 checkpoint directory. The real run uses a published trained checkpoint on a
Kaggle notebook and on a vast.ai machine. These tests run the same entry point on the tiny
synthetic checkpoint of ``b13_moe_fixture`` on CPU, so they check the wiring of the
comparison and its sensitivity, not the behavior of real weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness import b14_real_moe_correctness as harness
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fixture.moe_weights_dir(tmp_path_factory.mktemp("b14-real-moe-weights"))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_per_expert_matches_resident_model_on_a_checkpoint_directory(
    tiny_weights: Path, tmp_path: Path, dtype: str,
) -> None:
    result = harness.run_correctness(tiny_weights, tmp_path / "work", dtype=dtype)

    assert result["verdict"] == "PASS", result
    assert result["bitwise"] is True
    assert result["grad_mismatches"] == []
    # The comparison must not be vacuous: LoRA gradients arrive and are not all zero,
    # and the router sent tokens to several experts of every MoE layer.
    assert result["grads_compared"] > 0
    assert result["grads_nonzero"] == result["grads_compared"]
    assert all(count >= 2 for count in result["experts_hit_by_layer"].values())
    assert len(result["experts_hit_by_layer"]) == 2
    # After the comparison the harness corrupts one resident expert and compares again:
    # a comparison that cannot see that corruption would pass for any model.
    assert result["control_detected"] is True


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_one_wrong_expert_in_the_resident_model_fails_the_verdict(
    tiny_weights: Path, tmp_path: Path, dtype: str,
) -> None:
    result = harness.run_correctness(
        tiny_weights, tmp_path / "work", dtype=dtype, perturb_resident_expert=True,
    )

    assert result["verdict"] == "FAIL", result
    assert result["logits_equal"] is False
    assert result["bitwise"] is False


# --------------------------------------------------------------------------
# The definition of "real small MoE" (ticket 08), checked on plain data.
# --------------------------------------------------------------------------
def _config(**overrides: Any) -> Dict[str, Any]:
    fields = dict(
        model_type="deepseek_v3", num_hidden_layers=4, first_k_dense_replace=1,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        routed_scaling_factor=2.5, initializer_range=0.02,
    )
    fields.update(overrides)
    return fields


def _names(config: Dict[str, Any], *, drop: Iterable[str] = ()) -> List[str]:
    names = ["model.embed_tokens.weight", "lm_head.weight"]
    for layer in range(config["first_k_dense_replace"], config["num_hidden_layers"]):
        for expert in range(config["n_routed_experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.append(f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight")
    return [name for name in names if name not in drop]


def _check(
    config: Optional[Dict[str, Any]] = None,
    *,
    names: Optional[List[str]] = None,
    size: int = 9_000_000_000,
    expert_std: float = 0.023,
) -> List[str]:
    config = config if config is not None else _config()
    return harness.check_definition(
        config, names if names is not None else _names(config), size, expert_std,
    )


def test_a_trained_deepseek_v3_checkpoint_meets_the_definition() -> None:
    assert _check() == []


def test_weights_whose_spread_equals_the_init_range_are_random_init() -> None:
    assert _check(expert_std=0.0200) == ["6-trained-weights"]


def test_a_missing_expert_projection_or_id_gap_fails_the_expert_key_layout() -> None:
    config = _config()
    gap = "model.layers.2.mlp.experts.1.up_proj.weight"
    fused = [n for n in _names(config) if ".mlp.experts." not in n] + [
        "model.layers.1.mlp.experts.gate_up_proj", "model.layers.2.mlp.experts.gate_up_proj",
        "model.layers.3.mlp.experts.gate_up_proj",
    ]

    assert _check(config, names=_names(config, drop={gap})) == ["3-expert-keys"]
    assert _check(config, names=fused) == ["3-expert-keys"]


def test_an_mtp_layer_beyond_the_layer_count_is_not_an_expert_key_failure() -> None:
    config = _config()
    mtp = [
        f"model.layers.4.mlp.experts.{e}.{p}_proj.weight"
        for e in range(4) for p in ("gate", "up", "down")
    ]

    assert _check(config, names=_names(config) + mtp) == []


def test_a_size_over_the_limit_fails() -> None:
    assert _check(size=20_000_000_001) == ["4-size"]
    assert _check(size=20_000_000_000) == []


def test_fewer_than_two_moe_layers_or_no_shared_expert_or_top_k_at_expert_count_fails() -> None:
    assert _check(_config(first_k_dense_replace=3)) == ["5-moe-shape"]
    assert _check(_config(n_shared_experts=0)) == ["5-moe-shape"]
    assert _check(_config(num_experts_per_tok=4)) == ["5-moe-shape"]


def test_a_config_the_loader_cannot_read_unedited_fails() -> None:
    # GigaChat3 ships routed_scaling_factor as the int 1; transformers 5.20.2 rejects it.
    assert _check(_config(routed_scaling_factor=1)) == ["1-config"]
    assert _check(_config(model_type="olmoe")) == ["1-config"]
