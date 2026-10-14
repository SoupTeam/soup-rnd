"""Ticket 16: real ``model_type=kimi_k2`` under layer streaming.

Seams: ``load_stream_config`` / ``load_kimi_k2_config`` (config without Hub code),
``register_kimi_k2_conversion`` (checkpoint conversion), ``stream_arch_of`` (allowlist),
``shard_checkpoint`` and ``build_streamed_model`` (through the parity harness of
``test_b2_deepseek_v3_stream``).

The real K2 ``config.json`` files are not in the repo. ``_K2_CONFIG`` holds the fields
that ``.scratch/b2-deepseek-v3-stream/research.md`` section 1 recorded from
``Kimi-K2-Base`` and ``Kimi-K2-Instruct`` (they share one file) plus the structural fields
transformers needs to build the class. The tokenizer is not covered here.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator

import pytest

from tests.test_b2_deepseek_v3_stream import (
    _ATTENTION_LINEARS,
    _DENSE_MLP_LINEARS,
    _DEVICES,
    _EXPERT_IMPLEMENTATIONS,
    _REQUIRES_TRAINING_FLOOR,
    _SHARED_EXPERT_LINEARS,
    _assert_parity,
    _moe_lora_config,
    _run_parity,
    _zero_router_bias,
    deterministic,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

_ = deterministic  # a pytest fixture, imported for the tests below

_K2_CONFIG: Dict[str, Any] = {
    "model_type": "kimi_k2",
    "architectures": ["DeepseekV3ForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_deepseek.DeepseekV3Config",
        "AutoModel": "modeling_deepseek.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
    },
    "vocab_size": 163840,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 61,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 1,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.827,
    "rms_norm_eps": 1e-06,
    "rope_theta": 50000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 32.0,
        "beta_fast": 1.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
    },
    "max_position_embeddings": 131072,
    "num_nextn_predict_layers": 0,
    "tie_word_embeddings": False,
    "aux_loss_alpha": 0.001,
    "seq_aux": True,
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    },
}

_K2_INT4_QUANTIZATION = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 32}}},
}


def _write_config(folder: Path, **overrides: Any) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(_K2_CONFIG)
    payload.update(overrides)
    (folder / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(folder)


@pytest.fixture(autouse=True)
def _clean_kimi_registration() -> Iterator[None]:
    """Each test starts and ends with transformers not knowing ``kimi_k2``."""
    _forget_kimi_k2_conversion()
    yield
    _forget_kimi_k2_conversion()


def _forget_kimi_k2_conversion() -> None:
    from peft.utils import transformers_weight_conversion as peft_conversion
    from transformers import conversion_mapping

    conversion_mapping._MODEL_TO_CONVERSION_PATTERN.pop("kimi_k2", None)
    peft_conversion._MODEL_TO_CONVERSION_PATTERN.pop("kimi_k2", None)
    cache = conversion_mapping._checkpoint_conversion_mapping_cache
    if cache is not None:
        cache.pop("kimi_k2", None)
    conversion_mapping.USER_REGISTERED_MAPPINGS.discard("kimi_k2")


# ==========================================================================
# config without Hub code
# ==========================================================================


def test_native_autoconfig_cannot_read_a_k2_config(tmp_path: Path) -> None:
    """The control for the loader below: transformers 5.19 alone refuses this folder."""
    from transformers import AutoConfig

    folder = _write_config(tmp_path / "k2")
    with pytest.raises(ValueError, match="custom code"):
        AutoConfig.from_pretrained(folder, trust_remote_code=False)


def test_real_k2_config_identities_load_without_hub_code(tmp_path: Path) -> None:
    from transformers import DeepseekV3Config

    from soup_cli.utils.kimi_k2 import load_stream_config

    config = load_stream_config(_write_config(tmp_path / "k2"), trust_remote_code=False)

    assert isinstance(config, DeepseekV3Config)
    assert config.model_type == "kimi_k2"
    expected = {
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "num_nextn_predict_layers": 0,
        "num_hidden_layers": 61,
        "num_experts_per_tok": 8,
        "routed_scaling_factor": 2.827,
        "vocab_size": 163840,
        "max_position_embeddings": 131072,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
    }
    for name, value in expected.items():
        assert getattr(config, name) == value, name
    rope = config.rope_parameters
    assert (rope["rope_type"], rope["factor"], rope["rope_theta"]) == ("yarn", 32.0, 50000.0)
    assert (rope["beta_fast"], rope["beta_slow"]) == (1.0, 1.0)
    assert config.quantization_config["quant_method"] == "fp8"
    assert config.quantization_config["weight_block_size"] == [128, 128]


def test_k2_instruct_0905_style_rope_and_context_are_read_not_defaulted(tmp_path: Path) -> None:
    from soup_cli.utils.kimi_k2 import load_stream_config

    scaling = dict(_K2_CONFIG["rope_scaling"], factor=64.0)
    folder = _write_config(
        tmp_path / "k2", rope_scaling=scaling, max_position_embeddings=262144, rms_norm_eps=1e-5
    )
    config = load_stream_config(folder, trust_remote_code=False)

    assert config.rope_parameters["factor"] == 64.0
    assert config.max_position_embeddings == 262144
    assert config.rms_norm_eps == 1e-5


def test_other_model_types_still_go_through_autoconfig(tmp_path: Path) -> None:
    from transformers import DeepseekV3Config

    from soup_cli.utils.kimi_k2 import load_stream_config

    folder = tmp_path / "v3"
    folder.mkdir()
    DeepseekV3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=2, n_routed_experts=4, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=24, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    ).save_pretrained(folder)

    config = load_stream_config(str(folder), trust_remote_code=False)

    assert config.model_type == "deepseek_v3"
    assert config.n_routed_experts == 4


def test_a_k2_config_with_trust_remote_code_is_refused(tmp_path: Path) -> None:
    """Moonshot's modeling file keeps experts in a module list and cannot train."""
    from soup_cli.utils.kimi_k2 import load_stream_config

    folder = _write_config(tmp_path / "k2")

    with pytest.raises(ValueError, match="trust-remote-code"):
        load_stream_config(folder, trust_remote_code=True)


def test_the_aux_loss_fields_are_recorded_as_ignored(tmp_path: Path) -> None:
    from soup_cli.utils.kimi_k2 import ignored_aux_loss_fields, load_stream_config

    config = load_stream_config(_write_config(tmp_path / "k2"), trust_remote_code=False)

    assert ignored_aux_loss_fields(config) == {"aux_loss_alpha": 0.001, "seq_aux": True}
    assert ignored_aux_loss_fields(SimpleNamespace(model_type="deepseek_v3")) == {}


# ==========================================================================
# conversion registration
# ==========================================================================


def _mapping(model_type: str) -> Any:
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    return get_checkpoint_conversion_mapping(model_type)


def test_registering_the_conversion_gives_kimi_k2_the_deepseek_v3_rules() -> None:
    from soup_cli.utils.kimi_k2 import register_kimi_k2_conversion

    assert _mapping("kimi_k2") is None
    register_kimi_k2_conversion()

    kimi, deepseek = _mapping("kimi_k2"), _mapping("deepseek_v3")
    assert kimi and [repr(rule) for rule in kimi] == [repr(rule) for rule in deepseek]


def test_registering_twice_is_a_no_op_and_touches_no_other_model_type() -> None:
    from transformers import conversion_mapping

    from soup_cli.utils.kimi_k2 import register_kimi_k2_conversion

    others = ("deepseek_v3", "deepseek_v32", "deepseek_v2", "kimi_k25", "qwen2_moe", "mixtral")
    before = {name: repr(_mapping(name)) for name in others}
    patterns = dict(conversion_mapping._MODEL_TO_CONVERSION_PATTERN)
    register_kimi_k2_conversion()
    first = repr(_mapping("kimi_k2"))
    register_kimi_k2_conversion()

    assert first != "None"
    assert repr(_mapping("kimi_k2")) == first
    assert {name: repr(_mapping(name)) for name in others} == before
    assert _mapping("not_a_model_type") is None
    assert conversion_mapping._MODEL_TO_CONVERSION_PATTERN == {**patterns, "kimi_k2": "qwen2_moe"}


def test_registering_does_not_replace_a_mapping_someone_else_registered() -> None:
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    from soup_cli.utils.kimi_k2 import register_kimi_k2_conversion

    custom: list = []
    register_checkpoint_conversion_mapping("kimi_k2", custom)

    with pytest.raises(ValueError, match="already registered"):
        register_kimi_k2_conversion()
    assert _mapping("kimi_k2") == []


# ==========================================================================
# a tiny kimi_k2 checkpoint: load, save, aux loss, router
# ==========================================================================

_N_EXPERTS = 8


def _tiny_config(model_type: str = "kimi_k2", **overrides: Any) -> Any:
    """K2's routing (one expert group) and yarn rope at a size a CPU test can afford."""
    from transformers import DeepseekV3Config

    fields: Dict[str, Any] = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=_N_EXPERTS,
        n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1, q_lora_rank=24,
        kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        rope_parameters={
            "rope_type": "yarn", "factor": 4.0, "beta_fast": 1.0, "beta_slow": 1.0,
            "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 32,
            "rope_theta": 50000.0,
        },
    )
    fields.update(overrides)
    config = DeepseekV3Config(**fields)
    config.model_type = model_type
    if model_type == "kimi_k2":
        config.aux_loss_alpha = 0.001
        config.seq_aux = True
    return config


def _save_tiny(folder: Path, model_type: str = "kimi_k2", inv_freq: bool = False) -> Any:
    """Save a fp32 checkpoint with per-expert keys and a spread router bias.

    ``inv_freq`` adds the per-layer rotary keys K2 checkpoints carry.
    """
    from safetensors.torch import load_file, save_file
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.kimi_k2 import register_kimi_k2_conversion

    if model_type == "kimi_k2":
        register_kimi_k2_conversion()
    torch.manual_seed(7)
    config = _tiny_config(model_type)
    model = DeepseekV3ForCausalLM(config).to(torch.float32)
    with torch.no_grad():
        for layer in model.model.layers[config.first_k_dense_replace:]:
            layer.mlp.gate.e_score_correction_bias.normal_(0.0, 0.01)
    model.save_pretrained(folder, safe_serialization=True)
    # the model rebuilds its config from the class, so the saved file says deepseek_v3
    saved = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    saved.update(model_type=model_type, **(
        {"aux_loss_alpha": 0.001, "seq_aux": True} if model_type == "kimi_k2" else {}
    ))
    (folder / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    if inv_freq:
        path = folder / "model.safetensors"
        tensors = load_file(str(path))
        for idx in range(config.num_hidden_layers):
            tensors[f"model.layers.{idx}.self_attn.rotary_emb.inv_freq"] = torch.arange(
                config.qk_rope_head_dim // 2, dtype=torch.float32
            )
        save_file(tensors, str(path), metadata={"format": "pt"})
    return model


def _load(folder: Path) -> Any:
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.kimi_k2 import load_stream_config

    config = load_stream_config(str(folder), trust_remote_code=False)
    return DeepseekV3ForCausalLM.from_pretrained(
        str(folder), config=config, dtype=torch.float32, output_loading_info=True
    )


def _logits(model: Any) -> Any:
    ids = torch.arange(16).reshape(2, 8) % 64
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids).logits


def _saved_keys(folder: Path) -> set:
    from safetensors import safe_open

    with safe_open(str(folder / "model.safetensors"), framework="pt") as handle:
        return set(handle.keys())


def test_a_kimi_k2_checkpoint_loads_complete_with_the_deepseek_v3_logits(tmp_path: Path) -> None:
    twin = _save_tiny(tmp_path / "twin", "deepseek_v3", inv_freq=True)
    _save_tiny(tmp_path / "k2", "kimi_k2", inv_freq=True)
    # same weights, only the model type differs
    (tmp_path / "k2" / "model.safetensors").write_bytes(
        (tmp_path / "twin" / "model.safetensors").read_bytes()
    )

    model, info = _load(tmp_path / "k2")

    assert model.config.model_type == "kimi_k2"
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert torch.equal(_logits(model), _logits(twin))


def test_without_the_conversion_a_kimi_k2_load_leaves_the_experts_at_init(
    tmp_path: Path,
) -> None:
    """The control: it shows the load test above can detect a missing conversion."""
    from transformers import DeepseekV3ForCausalLM

    twin = _save_tiny(tmp_path / "twin", "deepseek_v3")
    folder = tmp_path / "k2"
    _save_tiny(folder, "kimi_k2")
    twin_weights = (tmp_path / "twin" / "model.safetensors").read_bytes()
    (folder / "model.safetensors").write_bytes(twin_weights)
    _forget_kimi_k2_conversion()
    config = _tiny_config("kimi_k2")

    model, info = DeepseekV3ForCausalLM.from_pretrained(
        str(folder), config=config, dtype=torch.float32, output_loading_info=True
    )

    assert any("experts.gate_up_proj" in key for key in info["missing_keys"])
    assert not torch.equal(_logits(model), _logits(twin))


def test_saving_a_kimi_k2_model_writes_per_expert_keys_and_reloads_identically(
    tmp_path: Path,
) -> None:
    _save_tiny(tmp_path / "k2", "kimi_k2", inv_freq=True)
    model, _ = _load(tmp_path / "k2")
    out = tmp_path / "resaved"

    model.save_pretrained(out, safe_serialization=True)

    keys = _saved_keys(out)
    last = f"model.layers.2.mlp.experts.{_N_EXPERTS - 1}"
    assert f"{last}.gate_proj.weight" in keys and f"{last}.down_proj.weight" in keys
    assert not any(key.endswith("experts.gate_up_proj") for key in keys)
    assert not any(key.endswith("experts.down_proj") for key in keys)
    again, info = _load(out)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert torch.equal(_logits(again), _logits(model))


def test_the_aux_loss_fields_change_nothing_under_a_frozen_router() -> None:
    """The native classes never read ``aux_loss_alpha`` or ``seq_aux``."""
    from transformers import DeepseekV3ForCausalLM

    def run(alpha: float, seq_aux: bool) -> Any:
        torch.manual_seed(3)
        config = _tiny_config()
        config.aux_loss_alpha, config.seq_aux = alpha, seq_aux
        model = DeepseekV3ForCausalLM(config).to(torch.float32)
        for name, param in model.named_parameters():
            param.requires_grad = ".mlp.gate." not in name
        ids = torch.arange(16).reshape(2, 8) % 64
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        router = [p for n, p in model.named_parameters() if ".mlp.gate." in n]
        return out.loss.detach(), grads, router

    base_loss, base_grads, router = run(0.001, True)
    other_loss, other_grads, _ = run(0.9, False)

    assert torch.equal(base_loss, other_loss)
    assert base_grads.keys() == other_grads.keys()
    assert all(torch.equal(base_grads[name], other_grads[name]) for name in base_grads)
    assert router and all(param.grad is None for param in router)


def _moonshot_choice(router: Any, hidden: Any) -> Any:
    """Moonshot's ``MoEGate.forward`` selection, re-stated from the Hub file.

    It masks excluded groups with ``0.0``. The native router masks them with ``-inf``.
    """
    scores = torch.sigmoid(torch.nn.functional.linear(hidden.float(), router.weight.float()))
    for_choice = scores + router.e_score_correction_bias.float()
    n_group, topk_group, top_k = router.num_group, router.topk_group, router.top_k
    n_tokens = scores.shape[0]
    group_scores = (
        for_choice.view(n_tokens, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1).expand(n_tokens, n_group, router.num_experts // n_group)
        .reshape(n_tokens, -1)
    )
    masked = for_choice.masked_fill(~score_mask.bool(), 0.0)
    return torch.topk(masked, k=top_k, dim=-1, sorted=False)[1]


def _native_choice(router: Any, hidden: Any) -> Any:
    return router(hidden)[2]


def _router_with(config: Any, bias_shift: float) -> Any:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    torch.manual_seed(11)
    router = DeepseekV3TopkRouter(config)
    with torch.no_grad():
        router.weight.normal_(0.0, 1.0)
        router.e_score_correction_bias.copy_(torch.randn(config.n_routed_experts) + bias_shift)
    return router


@pytest.mark.parametrize("bias_shift", [0.0, -3.0, 2.0])
def test_the_router_selects_the_same_experts_as_moonshots_zero_mask_with_one_group(
    bias_shift: float,
) -> None:
    config = _tiny_config(n_routed_experts=16, num_experts_per_tok=4)
    router = _router_with(config, bias_shift)
    hidden = torch.randn(64, config.hidden_size, generator=torch.Generator().manual_seed(5))

    native = _native_choice(router, hidden).sort(dim=-1)[0]
    moonshot = _moonshot_choice(router, hidden).sort(dim=-1)[0]

    assert torch.equal(native, moonshot)


def test_the_zero_mask_does_differ_when_groups_are_excluded() -> None:
    """The control: with real group routing and negative scores, the two masks disagree."""
    config = _tiny_config(n_routed_experts=16, num_experts_per_tok=4, n_group=4, topk_group=1)
    router = _router_with(config, -3.0)
    hidden = torch.randn(64, config.hidden_size, generator=torch.Generator().manual_seed(5))

    native = _native_choice(router, hidden).sort(dim=-1)[0]
    moonshot = _moonshot_choice(router, hidden).sort(dim=-1)[0]

    assert not torch.equal(native, moonshot)


# ==========================================================================
# the streaming allowlist
# ==========================================================================


def test_kimi_k2_streams_as_a_deepseek_v3_alias(tmp_path: Path) -> None:
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_stream import _STREAM_ARCH_ALIASES, stream_arch_of

    config = load_stream_config(_write_config(tmp_path / "k2"), trust_remote_code=False)

    assert stream_arch_of(config) == "deepseek_v3"
    assert _STREAM_ARCH_ALIASES["kimi_k2"] == "deepseek_v3"
    assert stream_arch_of(SimpleNamespace(model_type="kimi_k2")) == "deepseek_v3"


def test_the_kimi_k25_wrapper_stays_refused_although_its_text_tower_is_kimi_k2() -> None:
    """K2.5's ``text_config.model_type`` is ``kimi_k2``. Only K2-Base and K2-Instruct are in."""
    from soup_cli.utils.layer_stream import SUPPORTED_STREAM_ARCHS, stream_arch_of

    config = SimpleNamespace(
        model_type="kimi_k25", text_config=SimpleNamespace(model_type="kimi_k2")
    )

    with pytest.raises(ValueError) as excinfo:
        stream_arch_of(config)

    assert str(excinfo.value) == (
        "layer streaming does not support model_type='kimi_k25'. "
        f"Supported: {', '.join(SUPPORTED_STREAM_ARCHS)}."
    )


def test_an_int4_kimi_k2_checkpoint_is_refused_naming_the_remaining_limits(
    tmp_path: Path,
) -> None:
    """K2-Thinking stores its routed experts as INT4 compressed-tensors."""
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_stream import stream_arch_of

    folder = _write_config(tmp_path / "k2", quantization_config=_K2_INT4_QUANTIZATION)
    config = load_stream_config(folder, trust_remote_code=False)

    with pytest.raises(ValueError) as excinfo:
        stream_arch_of(config)

    message = str(excinfo.value)
    assert "model_type='kimi_k2'" in message
    assert "INT4" in message and "compressed-tensors" in message
    assert "block-FP8 Kimi-K2-Base and Kimi-K2-Instruct" in message
    assert "sub-layer streaming" in message
    assert "Hub code" not in message and "random values" not in message
    assert "FP8, which layer streaming cannot read" not in message


def test_a_block_fp8_kimi_k2_config_is_not_refused(tmp_path: Path) -> None:
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_stream import stream_arch_of

    config = load_stream_config(_write_config(tmp_path / "k2"), trust_remote_code=False)

    assert config.quantization_config["quant_method"] == "fp8"
    assert stream_arch_of(config) == "deepseek_v3"


# ==========================================================================
# the sharder and the streamed model on a real kimi_k2 loading path
# ==========================================================================


def _make_kimi_k2(weights: Path) -> None:
    """Turn the harness's saved fixture into a ``kimi_k2`` checkpoint: its config file and
    the per-layer rotary keys K2 checkpoints carry."""
    from safetensors.torch import load_file, save_file

    saved = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    saved.update(model_type="kimi_k2", aux_loss_alpha=0.001, seq_aux=True)
    (weights / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    path = weights / "model.safetensors"
    tensors = load_file(str(path))
    for idx in range(saved["num_hidden_layers"]):
        tensors[f"model.layers.{idx}.self_attn.rotary_emb.inv_freq"] = torch.arange(
            saved["qk_rope_head_dim"] // 2, dtype=torch.float32
        )
    save_file(tensors, str(path), metadata={"format": "pt"})


def test_the_sharder_drops_the_rotary_keys_k2_checkpoints_carry(tmp_path: Path) -> None:
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import layer_paths, shard_checkpoint

    weights = tmp_path / "weights"
    _save_tiny(weights, "kimi_k2", inv_freq=True)
    assert sum("rotary_emb.inv_freq" in key for key in _saved_keys(weights)) == 3

    index = shard_checkpoint(str(weights), str(tmp_path / "shards"), dtype="float32",
                             arch="deepseek_v3")

    stored = set()
    for path in layer_paths(str(tmp_path / "shards"), index):
        with safe_open(path, framework="pt") as handle:
            stored |= set(handle.keys())
    assert stored and not any("inv_freq" in key for key in stored)
    assert any(key.endswith("mlp.experts.gate_up_proj") for key in stored)




@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_kimi_k2_matches_resident(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """Forward and backward on a checkpoint whose config says ``kimi_k2``, loaded through
    ``load_stream_config``, with K2's one expert group, yarn rope and rotary keys."""
    parity = _run_parity(
        tmp_path, device=device, experts=experts, config=_tiny_config(),
        edit_checkpoint=_make_kimi_k2,
    )

    assert parity.config.model_type == "kimi_k2"
    _assert_parity(parity)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("device", _DEVICES)
def test_control_a_kimi_k2_stream_with_a_zero_router_bias_differs(
    tmp_path: Path, device: str, deterministic: None
) -> None:
    parity = _run_parity(
        tmp_path, device=device, experts="grouped_mm", config=_tiny_config(),
        edit_checkpoint=_make_kimi_k2, edit_streamed=_zero_router_bias,
    )

    assert not parity.forward_equal(), parity.report()
    assert parity.backward_mismatches(), parity.report()


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_kimi_k2_from_the_disk_tier_matches_resident(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    parity = _run_parity(
        tmp_path, device=device, experts=experts, config=_tiny_config(),
        edit_checkpoint=_make_kimi_k2, tier="disk",
    )

    assert (parity.tier, parity.source) == ("disk", "AsyncDiskSource")
    _assert_parity(parity)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("tier", ["ram", "disk"])
@pytest.mark.parametrize(
    "device", [pytest.param("cuda", marks=pytest.mark.gpu(reason="the 4-bit kernels are GPU-only"))]
)
def test_streamed_kimi_k2_nf4_matches_resident_nf4(
    tmp_path: Path, device: str, tier: str, experts: str, deterministic: None
) -> None:
    pytest.importorskip("bitsandbytes")

    parity = _run_parity(
        tmp_path, device=device, experts=experts, config=_tiny_config(),
        edit_checkpoint=_make_kimi_k2, quant="nf4", tier=tier,
    )

    expected = {
        0: _ATTENTION_LINEARS | _DENSE_MLP_LINEARS,
        1: _ATTENTION_LINEARS | _SHARED_EXPERT_LINEARS,
        2: _ATTENTION_LINEARS | _SHARED_EXPERT_LINEARS,
    }
    assert parity.quantized == expected and parity.resident_quantized == expected
    _assert_parity(parity)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_kimi_k2_moe_lora_matches_resident(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """The routed-expert adapters of ``moe_lora: true``, which need the registered conversion."""
    parity = _run_parity(
        tmp_path, device=device, experts=experts, config=_tiny_config(),
        edit_checkpoint=_make_kimi_k2, lora_config=_moe_lora_config,
    )

    assert any(".mlp.experts." in name for name in parity.lora_names)
    _assert_parity(parity)


def test_peft_rewrites_expert_targets_onto_the_fused_experts_only_once_registered() -> None:
    """``moe_lora`` on ``kimi_k2`` is routed-expert LoRA only through the registration."""
    from accelerate import init_empty_weights
    from peft import LoraConfig
    from peft.utils.transformers_weight_conversion import convert_peft_config_for_transformers
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.kimi_k2 import register_kimi_k2_conversion

    with init_empty_weights():
        model = DeepseekV3ForCausalLM(_tiny_config())

    def converted() -> Any:
        config = LoraConfig(r=2, target_modules=["q_a_proj", "gate_proj", "up_proj", "down_proj"])
        convert_peft_config_for_transformers(config, model, None)
        return config

    assert not converted().target_parameters
    register_kimi_k2_conversion()
    after = converted()
    assert set(after.target_parameters) == {"gate_up_proj", "down_proj"}
    assert "gate_proj" not in after.target_modules


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_kimi_k2_shared_expert_lora_matches_resident(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """Ticket 15's shared-expert targeting on the ``kimi_k2`` model type."""
    from tests.test_b2_shared_expert_lora import _SHARED_TARGETS, _assert_shared_parity, _config

    parity = _run_parity(
        tmp_path, device=device, experts=experts, config=_tiny_config(),
        edit_checkpoint=_make_kimi_k2, lora_config=lambda weights: _config(_SHARED_TARGETS),
    )

    _assert_shared_parity(parity)


def test_an_int4_text_config_under_an_unknown_wrapper_is_refused_too() -> None:
    from soup_cli.utils.layer_stream import stream_arch_of

    config = SimpleNamespace(
        model_type="some_wrapper",
        text_config=SimpleNamespace(
            model_type="kimi_k2", quantization_config=_K2_INT4_QUANTIZATION
        ),
    )

    with pytest.raises(ValueError, match="INT4"):
        stream_arch_of(config)
