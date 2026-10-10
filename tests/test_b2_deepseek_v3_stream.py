"""B2 ticket 06: DeepSeek-V3 streams, and a parity control proves it on a tiny model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest
import torch

from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

# ==========================================================================
# the architecture gate
# ==========================================================================


def test_deepseek_v3_is_accepted_under_its_own_name() -> None:
    from soup_cli.utils.layer_stream import _STREAM_ARCH_ALIASES, stream_arch_of

    assert stream_arch_of(SimpleNamespace(model_type="deepseek_v3")) == "deepseek_v3"
    assert "deepseek_v3" not in _STREAM_ARCH_ALIASES
    assert "deepseek_v3" not in _STREAM_ARCH_ALIASES.values()


def test_kimi_k2_is_refused_naming_what_is_missing() -> None:
    from soup_cli.utils.layer_stream import stream_arch_of

    with pytest.raises(ValueError) as excinfo:
        stream_arch_of(SimpleNamespace(model_type="kimi_k2"))

    message = str(excinfo.value)
    assert "model_type='kimi_k2'" in message
    assert "does not know the kimi_k2 config type" in message
    assert "tokenizer is Hub code" in message
    assert "no conversion of Kimi K2's per-expert weights into fused experts" in message
    assert "checkpoint is FP8" in message
    assert "too large to stream as a single layer" in message


@pytest.mark.parametrize("model_type", ["deepseek_v32", "deepseek_v2", "kimi_k25"])
def test_other_deepseek_and_kimi_types_get_the_ordinary_refusal(model_type: str) -> None:
    from soup_cli.utils.layer_stream import SUPPORTED_STREAM_ARCHS, stream_arch_of

    with pytest.raises(ValueError) as excinfo:
        stream_arch_of(SimpleNamespace(model_type=model_type))

    assert str(excinfo.value) == (
        f"layer streaming does not support model_type={model_type!r}. "
        f"Supported: {', '.join(SUPPORTED_STREAM_ARCHS)}."
    )


# ==========================================================================
# the parity harness: resident against streamed on a tiny DeepSeek-V3
# ==========================================================================
#
# Tickets 07 to 11 add arms to this harness. An arm is a fixture config plus,
# optionally, an edit of the saved checkpoint; `_run_parity` does the rest.

_BIAS = "mlp.gate.e_score_correction_bias"
_N_EXPERTS = 16
_EXPERT_IMPLEMENTATIONS = ("grouped_mm", "eager")
_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.gpu(reason="the CUDA fp32 parity control")),
]


def _tiny_deepseek_v3_config(**overrides: Any) -> Any:
    from transformers import DeepseekV3Config

    fields: Dict[str, Any] = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=_N_EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=4,
        topk_group=2,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    fields.update(overrides)
    return DeepseekV3Config(**fields)


def _moe_layers(config: Any) -> range:
    return range(config.first_k_dense_replace, config.num_hidden_layers)


def _save_fixture(weights: Path, config: Any) -> None:
    """Save a fp32 checkpoint with a spread router bias and per-expert keys on disk."""
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    torch.manual_seed(61)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    with torch.no_grad():
        for idx in _moe_layers(config):
            model.model.layers[idx].mlp.gate.e_score_correction_bias.normal_(0.0, 0.01)
    model.save_pretrained(weights, safe_serialization=True)
    last = config.num_hidden_layers - 1
    with safe_open(str(weights / "model.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())
    assert f"model.layers.{last}.mlp.experts.{config.n_routed_experts - 1}.gate_proj.weight" in keys
    assert f"model.layers.{last}.mlp.experts.gate_up_proj" not in keys


def _lora() -> Any:
    """The adapter Soup builds for ``target_modules: auto`` on DeepSeek-V3.

    Those are the five MLA projections. Naming ``gate_proj``, ``up_proj`` or
    ``down_proj`` here would not reach the shared experts: peft 0.21 rewrites
    those names on ``deepseek_v3`` into ``target_parameters`` on the fused
    routed experts, which is ``moe_lora`` (ticket 10).
    """
    from peft import LoraConfig, TaskType

    from soup_cli.utils.peft_wiring import MOE_TEXT_LORA_TARGETS

    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(MOE_TEXT_LORA_TARGETS["deepseek_v3"]),
        task_type=TaskType.CAUSAL_LM,
    )


def _strip_stream_wrapper(name: str) -> str:
    """A streamed layer's name without the ``.inner.`` its wrapper adds."""
    return name.replace(".inner.", ".")


def _lora_params(model: Any) -> Dict[str, Any]:
    return {
        _strip_stream_wrapper(name): param
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def _randomise_lora_b(model: Any) -> None:
    """PEFT starts ``lora_B`` at zero, which makes a detached adapter look equal."""
    gen = torch.Generator().manual_seed(67)
    with torch.no_grad():
        for name, param in sorted(_lora_params(model).items()):
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen).to(param) * 0.05)


def _copy_lora(src: Any, dst: Any) -> None:
    source, target = _lora_params(src), _lora_params(dst)
    assert source and set(source) == set(target)
    with torch.no_grad():
        for name, param in target.items():
            param.copy_(source[name])


def _record_expert_hits(model: Any, hits: Dict[str, set]) -> List[Any]:
    """Hook every router; ``hits`` maps a router's name to the experts it chose."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, DeepseekV3TopkRouter):
            seen = hits.setdefault(_strip_stream_wrapper(name), set())

            def hook(_module: Any, _args: Any, output: Any, seen: set = seen) -> None:
                seen.update(output[2].flatten().tolist())

            handles.append(module.register_forward_hook(hook))
    return handles


@dataclass
class _Pass:
    logits: Any
    #: Every LoRA parameter's gradient by name; ``None`` where none reached it.
    grads: Dict[str, Any]
    expert_hits: Dict[str, set]


def _forward_backward(model: Any, input_ids: Any) -> _Pass:
    hits: Dict[str, set] = {}
    handles = _record_expert_hits(model, hits)
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    grads = {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in _lora_params(model).items()
    }
    return _Pass(out.logits.detach().clone(), grads, hits)


@dataclass
class _Parity:
    config: Any
    resident: _Pass
    streamed: _Pass
    meta_violations: List[str]
    streamed_weights: Dict[int, set]
    lora_names: List[str]

    def forward_equal(self) -> bool:
        return torch.equal(self.streamed.logits, self.resident.logits)

    def backward_mismatches(self) -> List[str]:
        names = set(self.streamed.grads) | set(self.resident.grads)
        return sorted(
            name
            for name in names
            if self.streamed.grads.get(name) is None
            or self.resident.grads.get(name) is None
            or not torch.equal(self.streamed.grads[name], self.resident.grads[name])
        )

    def report(self) -> str:
        if self.forward_equal():
            forward = "forward: logits equal"
        else:
            diff = (self.streamed.logits - self.resident.logits).abs().max().item()
            forward = f"forward: logits differ, max |diff| {diff:.3e}"
        bad = self.backward_mismatches()
        backward = (
            f"backward: {len(bad)} of {len(self.lora_names)} LoRA gradients differ: {bad[:4]}"
            if bad
            else f"backward: all {len(self.lora_names)} LoRA gradients equal"
        )
        return f"{forward}; {backward}"


def _run_parity(
    tmp_path: Path,
    *,
    device: str,
    experts: str,
    edit_checkpoint: Optional[Callable[[Path], None]] = None,
    edit_streamed: Optional[Callable[[Any], None]] = None,
    config: Any = None,
    notices: Optional[List[str]] = None,
) -> _Parity:
    """Shard and stream the fixture the way the training setup does, then run
    the resident model and the streamed model on one batch."""
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model, decoder_owner

    config = config if config is not None else _tiny_deepseek_v3_config()
    weights = tmp_path / "weights"
    _save_fixture(weights, config)
    if edit_checkpoint is not None:
        edit_checkpoint(weights)

    arch = stream_arch_of(AutoConfig.from_pretrained(str(weights)))
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(
        str(weights),
        shards,
        dtype="float32",
        arch=arch,
        notify=notices.append if notices is not None else None,
    )
    streamed, runtime = build_streamed_model(
        model_id=str(weights),
        shard_dir=shards,
        index=index,
        lora_config=_lora(),
        device=device,
        dtype="float32",
        buffers=2,
        pin=False,
        seed=71,
    )
    try:
        streamed.get_base_model().set_experts_implementation(experts)
        assert streamed.get_base_model().config._experts_implementation == experts
        _randomise_lora_b(streamed)

        resident_base: Any = AutoModelForCausalLM.from_pretrained(
            str(weights), dtype=torch.float32, experts_implementation=experts
        )
        resident_base.to(device)
        assert resident_base.config._experts_implementation == experts
        resident = get_peft_model(resident_base, _lora())
        _copy_lora(streamed, resident)
        if edit_streamed is not None:
            edit_streamed(streamed)

        gen = torch.Generator().manual_seed(73)
        input_ids = torch.randint(0, config.vocab_size, (4, 64), generator=gen).to(device)
        streamed_pass = _forward_backward(streamed, input_ids)
        resident_pass = _forward_backward(resident, input_ids)
        layers = decoder_owner(streamed).layers
        base_weights = {
            idx: dict(
                (name, param)
                for name, param in layer.inner.named_parameters()
                if "lora_" not in name
            )
            for idx, layer in enumerate(layers)
        }
        meta_violations = [
            f"layer {idx}: {name}"
            for idx, weights_of in base_weights.items()
            for name, param in weights_of.items()
            if not param.is_meta
        ]
        streamed_weights = {idx: set(weights_of) for idx, weights_of in base_weights.items()}
        lora_names = sorted(_lora_params(resident))
    finally:
        runtime.close()
    return _Parity(
        config, resident_pass, streamed_pass, meta_violations, streamed_weights, lora_names
    )


def _assert_parity(parity: _Parity) -> None:
    """Every check an arm makes: coverage, meta, a live adapter, then equality,
    with forward and backward asserted separately."""
    config = parity.config
    for side in (parity.resident, parity.streamed):
        assert len(side.expert_hits) == len(_moe_layers(config))
        for router, seen in side.expert_hits.items():
            unused = set(range(config.n_routed_experts)) - seen
            assert not unused, f"{router} never chose experts {sorted(unused)}"
    for idx in _moe_layers(config):
        assert {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"} <= parity.streamed_weights[idx]
    assert parity.meta_violations == [], f"streamed weights left meta: {parity.meta_violations}"
    layer0 = [name for name in parity.lora_names if ".layers.0." in name]
    assert layer0
    for name in layer0:
        grad = parity.streamed.grads[name]
        assert grad is not None and bool(grad.ne(0).any()), f"{name}: no or zero gradient"
    assert parity.forward_equal(), parity.report()
    assert not parity.backward_mismatches(), parity.report()


@pytest.fixture
def deterministic(request: Any) -> Iterator[None]:
    """On CUDA, the parity runs under deterministic algorithms.

    cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when it creates its handle, so the
    variable must be set before the process first touches CUDA and must stay set
    for later tests. It is left in the environment for that reason. A process that
    touched CUDA without it skips, instead of running with no effect.
    """
    import os

    if request.node.callspec.params.get("device") != "cuda":
        yield
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        if torch.cuda.is_initialized():
            pytest.skip(
                "CUDA was initialised before CUBLAS_WORKSPACE_CONFIG was set; run with "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 exported"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_deepseek_v3_matches_resident(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """Parity arm 1: the base fixture."""
    _assert_parity(_run_parity(tmp_path, device=device, experts=experts))


def _zero_router_bias(streamed: Any) -> None:
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    zeroed = 0
    with torch.no_grad():
        for layer in decoder_owner(streamed).layers:
            for name, buf in layer.inner.named_buffers():
                if name == _BIAS:
                    assert bool(buf.ne(0).any())
                    buf.zero_()
                    zeroed += 1
    assert zeroed == len(_moe_layers(streamed.get_base_model().config))


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("device", _DEVICES)
def test_control_streamed_model_with_a_zero_router_bias_differs(
    tmp_path: Path, device: str, deterministic: None
) -> None:
    """The parity check can see a routing error: the bias the runtime fills matters."""
    parity = _run_parity(
        tmp_path, device=device, experts="grouped_mm", edit_streamed=_zero_router_bias
    )

    assert not parity.forward_equal(), parity.report()
    assert parity.backward_mismatches(), parity.report()


def _add_mtp_layer(weights: Path) -> int:
    """Write an MTP layer into the checkpoint after the last decoder layer, as a real
    DeepSeek-V3 checkpoint carries one: a copy of the last decoder layer plus the
    MTP projections and norms. Returns the number of keys added."""
    from safetensors.torch import load_file, save_file

    path = str(weights / "model.safetensors")
    state = load_file(path)
    mtp = {
        key.replace("model.layers.2.", "model.layers.3.", 1): value.clone()
        for key, value in state.items()
        if key.startswith("model.layers.2.")
    }
    hidden = state["model.norm.weight"].shape[0]
    mtp.update(
        {
            "model.layers.3.enorm.weight": torch.ones(hidden),
            "model.layers.3.hnorm.weight": torch.ones(hidden),
            "model.layers.3.eh_proj.weight": torch.randn(hidden, 2 * hidden),
            "model.layers.3.shared_head.norm.weight": torch.ones(hidden),
            "model.layers.3.shared_head.head.weight": state["lm_head.weight"].clone(),
        }
    )
    save_file({**state, **mtp}, path, metadata={"format": "pt"})
    return len(mtp)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_an_mtp_layer_is_dropped_and_the_streamed_model_still_matches(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """Parity arm 7: the sharder drops the MTP layer's keys and says how many."""
    added: List[int] = []
    notices: List[str] = []

    parity = _run_parity(
        tmp_path,
        device=device,
        experts=experts,
        edit_checkpoint=lambda weights: added.append(_add_mtp_layer(weights)),
        notices=notices,
    )

    assert added and added[0] > 3 * _N_EXPERTS
    assert [notice for notice in notices if "Dropped" in notice] == [
        f"[yellow]Dropped {added[0]} checkpoint keys of decoder layers at or beyond "
        "the model's 3 layers (an MTP layer, for example).[/]"
    ]
    _assert_parity(parity)
