"""B1.3 ticket 02: the streaming runtime installs streamed routed experts.

A per-expert (``expert_layout="per_expert"``) DeepSeek-V3 cache is streamed through
``build_streamed_model``: one reader per MoE layer, one bounded one-slot pool, installed
before layer weights are mapped, owned and closed by ``StreamRuntime``. LoRA sits on the
non-expert (MLA) projections; the expected values always come from an independent resident
eager model built from the same checkpoint with the same LoRA weights.

Reusable helpers (the L2L tickets import these; plain functions, no fixtures):

``moe_config(**overrides) -> DeepseekV3Config``
    1 dense layer + 2 MoE layers, 8 routed experts, 1 shared, router top-k 2, fp32 sizes.
``moe_weights_dir(tmp_path, config=None, *, seed=61) -> Path``
    Saves a randomised fp32 checkpoint (router weight, correction bias and routed experts
    randomised so routing differs between tokens and small inputs leave experts unused).
``moe_batch(config, *, seed=0, rows=2, seq=6) -> dict``
    ``input_ids``/``labels`` for one micro-batch; vary ``seed`` to vary the routing.
``moe_lora_config(*, dropout=0.0) -> LoraConfig``
    LoRA on the MLA projections only, never on routed experts.
``shard_moe(weights, shards, *, storage="dense", dtype="float32", layout="per_expert")``
    Shard the checkpoint. ``storage``: ``"dense"``, ``"nf4"`` or ``"nf4-nested"``;
    ``layout``: ``"per_expert"`` or ``"fused"``. Returns the ``ShardIndex``.
``build_moe_streamed(weights, shards, *, storage="dense", dtype="float32", layout="per_expert",
        lora_dropout=0.0, lora_config=None, device="cpu", index=None, randomize_b=True)``
    ``build_streamed_model`` on the cache (sharded here unless ``index`` is given). Returns a
    ``MoeStreamed`` (``model``, ``runtime``, ``index``, ``lora_config``, ``weights``,
    ``shards``, ``storage``, ``dtype``); always call its ``close()``.
``moe_resident_reference(weights, lora_config, *, shards=None, index=None, lora_from=None,
        dtype="float32")``
    ``from_pretrained`` + eager experts + the same LoRA (weights copied from ``lora_from``).
    For an NF4 cache every stored NF4 weight (attention, shared experts, routed experts) is
    overwritten with its independent dequantization, so the reference holds exactly the
    values the cache stores.
``randomize_lora_b(model, *, seed=67)`` / ``copy_lora_weights(src, dst)`` / ``lora_parameters``
``run_moe_pass(model, batch, *, backward=True) -> MoePass``  (``logits``, ``loss``, ``grads``)
``grads_differ(got, want) -> list[str]``  names whose gradient is missing or not ``torch.equal``
``routed_expert_hits(model, batch) -> {layer: set(expert)}``  what each router chose (no grad)
``streamed_experts_modules(model) -> {layer: StreamedExperts}``
``record_expert_reads(model) -> {layer: [expert, ...]}``  every ``reader.read_host`` call, in order
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

N_EXPERTS = 8
MLA_LORA_TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
_STORAGES = ("dense", "nf4", "nf4-nested")
_LAYOUTS = ("per_expert", "fused")


# ==========================================================================
# helpers
# ==========================================================================
def moe_config(**overrides: Any) -> Any:
    from transformers import DeepseekV3Config

    fields: Dict[str, Any] = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=N_EXPERTS,
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


def _moe_layer_ids(config: Any) -> range:
    return range(config.first_k_dense_replace, config.num_hidden_layers)


def moe_weights_dir(tmp_path: Path, config: Any = None, *, seed: int = 61) -> Path:
    """Save a randomised fp32 checkpoint whose routed experts are separate keys on disk."""
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    config = config if config is not None else moe_config()
    weights = tmp_path / "weights"
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for idx in _moe_layer_ids(config):
            mlp = model.model.layers[idx].mlp
            mlp.gate.weight.normal_(0.0, 0.6, generator=gen)
            mlp.gate.e_score_correction_bias.normal_(0.0, 0.3, generator=gen)
            mlp.experts.gate_up_proj.normal_(0.0, 0.2, generator=gen)
            mlp.experts.down_proj.normal_(0.0, 0.2, generator=gen)
    model.save_pretrained(weights, safe_serialization=True)
    last = config.num_hidden_layers - 1
    with safe_open(str(weights / "model.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())
    assert f"model.layers.{last}.mlp.experts.{config.n_routed_experts - 1}.gate_proj.weight" in keys
    assert f"model.layers.{last}.mlp.experts.gate_up_proj" not in keys
    return weights


def moe_batch(config: Any, *, seed: int = 0, rows: int = 2, seq: int = 6) -> Dict[str, Any]:
    gen = torch.Generator().manual_seed(1000 + seed)
    ids = torch.randint(0, config.vocab_size, (rows, seq), generator=gen)
    return {"input_ids": ids, "labels": ids.clone()}


def moe_lora_config(*, dropout: float = 0.0) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(MLA_LORA_TARGETS),
        task_type=TaskType.CAUSAL_LM,
    )


def shard_moe(
    weights: Path,
    shards: Path,
    *,
    storage: str = "dense",
    dtype: str = "float32",
    layout: str = "per_expert",
) -> Any:
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    assert storage in _STORAGES and layout in _LAYOUTS
    arch = stream_arch_of(load_stream_config(str(weights), trust_remote_code=False))
    quant_args: Dict[str, Any] = {}
    if storage != "dense":
        quant_args = dict(
            quant="nf4",
            quant_suffixes=quantised_layer_suffixes(
                build_meta_skeleton(str(weights), dtype=dtype, quant="nf4")
            ),
            quant_device="cpu",
            double_quant=storage == "nf4-nested",
        )
    return shard_checkpoint(
        str(weights),
        str(shards),
        dtype=dtype,
        arch=arch,
        expert_layout="per_expert" if layout == "per_expert" else "",
        **quant_args,
    )


def lora_parameters(model: Any) -> Dict[str, Any]:
    return {
        name.replace(".inner.", "."): param
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def randomize_lora_b(model: Any, *, seed: int = 67) -> None:
    """PEFT starts ``lora_B`` at zero, which would make every adapter gradient degenerate."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(lora_parameters(model).items()):
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen).to(param) * 0.05)


def copy_lora_weights(src: Any, dst: Any) -> None:
    source, target = lora_parameters(src), lora_parameters(dst)
    assert source and set(source) == set(target)
    with torch.no_grad():
        for name, param in target.items():
            param.copy_(source[name])


@dataclass
class MoeStreamed:
    model: Any
    runtime: Any
    index: Any
    lora_config: Any
    weights: Path
    shards: Path
    storage: str
    dtype: str

    def close(self) -> None:
        self.runtime.close()


def build_moe_streamed(
    weights: Path,
    shards: Path,
    *,
    storage: str = "dense",
    dtype: str = "float32",
    layout: str = "per_expert",
    lora_dropout: float = 0.0,
    lora_config: Any = None,
    device: str = "cpu",
    index: Any = None,
    randomize_b: bool = True,
) -> MoeStreamed:
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    if index is None:
        index = shard_moe(weights, shards, storage=storage, dtype=dtype, layout=layout)
    lora = lora_config if lora_config is not None else moe_lora_config(dropout=lora_dropout)
    model, runtime = build_streamed_model(
        model_id=str(weights),
        shard_dir=str(shards),
        index=index,
        lora_config=lora,
        device=device,
        dtype=dtype,
        buffers=2,
        pin=False,
        seed=71,
        quant="none" if storage == "dense" else "nf4",
        double_quant=storage == "nf4-nested",
    )
    if randomize_b:
        randomize_lora_b(model)
    return MoeStreamed(model, runtime, index, lora, weights, shards, storage, dtype)


def _apply_stored_nf4(base: Any, shards: Path, index: Any) -> None:
    """Overwrite every NF4-stored weight of ``base`` with its independent dequantization."""
    from bitsandbytes.functional import dequantize_4bit
    from safetensors.torch import load_file

    from soup_cli.utils.expert_shard import EXPERT_PARTS, expert_key
    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        expert_paths,
        extras_shard_path,
        layer_paths,
    )
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    extras = load_file(extras_shard_path(str(shards)))
    codes = {key: extras[key] for key in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY) if key in extras}
    layers = base.model.layers
    with torch.no_grad():
        for idx, path in enumerate(layer_paths(str(shards), index)):
            stored = load_file(path)
            for short, spec in index.quant_specs.items():
                if short not in stored:
                    continue
                state = rebuild_quant_state(short, stored, spec, codes)
                layers[idx].get_parameter(short).copy_(
                    dequantize_4bit(stored[short], quant_state=state)
                )
        for idx, path in expert_paths(str(shards), index).items():
            stored = load_file(path)
            experts = layers[idx].mlp.experts
            for expert in range(index.expert_count):
                for part, target in zip(EXPERT_PARTS, (experts.gate_up_proj, experts.down_proj)):
                    key = expert_key(expert, part)
                    state = rebuild_quant_state(key, stored, index.expert_quant_specs[part], codes)
                    target[expert].copy_(dequantize_4bit(stored[key], quant_state=state))


def moe_resident_reference(
    weights: Path,
    lora_config: Any,
    *,
    shards: Optional[Path] = None,
    index: Any = None,
    lora_from: Any = None,
    dtype: str = "float32",
) -> Any:
    """Resident eager model: the checkpoint, the stored NF4 values (if any), the same LoRA."""
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.peft_wiring import attach_adapter

    config = load_stream_config(str(weights), trust_remote_code=False)
    config.use_cache = False
    base = AutoModelForCausalLM.from_pretrained(
        str(weights), config=config, dtype=getattr(torch, dtype), experts_implementation="eager"
    )
    assert base.config._experts_implementation == "eager"
    if index is not None and index.quant == "nf4":
        assert shards is not None
        _apply_stored_nf4(base, shards, index)
    resident = attach_adapter(base, lora_config)
    if lora_from is not None:
        copy_lora_weights(lora_from, resident)
    return resident


@dataclass
class MoePass:
    logits: Any
    loss: Any
    #: every LoRA gradient by name (``None`` where none arrived); empty for forward only.
    grads: Dict[str, Any]


def run_moe_pass(model: Any, batch: Dict[str, Any], *, backward: bool = True) -> MoePass:
    model.train()
    model.zero_grad(set_to_none=True)
    if not backward:
        with torch.no_grad():
            out = model(**batch)
        return MoePass(out.logits.detach().clone(), out.loss.detach().clone(), {})
    out = model(**batch)
    out.loss.backward()
    grads = {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in lora_parameters(model).items()
    }
    return MoePass(out.logits.detach().clone(), out.loss.detach().clone(), grads)


def grads_differ(got: Dict[str, Any], want: Dict[str, Any]) -> List[str]:
    return sorted(
        name
        for name in set(got) | set(want)
        if got.get(name) is None
        or want.get(name) is None
        or not torch.equal(got[name], want[name])
    )


def routed_expert_hits(model: Any, batch: Dict[str, Any]) -> Dict[int, set]:
    """Experts each MoE layer's router chose for ``batch`` (the oracle for expert reads)."""
    import re

    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    hits: Dict[int, set] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, DeepseekV3TopkRouter):
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            seen = hits.setdefault(layer, set())

            def hook(_module: Any, _args: Any, output: Any, seen: set = seen) -> None:
                seen.update(output[2].flatten().tolist())

            handles.append(module.register_forward_hook(hook))
    try:
        run_moe_pass(model, batch, backward=False)
    finally:
        for handle in handles:
            handle.remove()
    return hits


def streamed_experts_modules(model: Any) -> Dict[int, Any]:
    from soup_cli.utils.expert_stream import streamed_experts_class

    cls = streamed_experts_class()
    return {module.layer_idx: module for module in model.modules() if isinstance(module, cls)}


def record_expert_reads(model: Any) -> Dict[int, List[int]]:
    """Wrap each streamed layer's ``reader.read_host``; the lists fill as experts are read."""
    log: Dict[int, List[int]] = {}
    for layer, module in streamed_experts_modules(model).items():
        reads = log.setdefault(layer, [])
        original = module.reader.read_host

        def wrapped(expert: int, staging: Any, *, _original: Any = original, _reads: Any = reads):
            _reads.append(expert)
            return _original(expert, staging)

        module.reader.read_host = wrapped
    return log


def _stored_bytes_of_one_expert(shards: Path, index: Any) -> int:
    """Bytes one expert occupies in its experts file, counted from the file itself."""
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import expert_paths

    stored = load_file(next(iter(expert_paths(str(shards), index).values())))
    return sum(t.numel() * t.element_size() for key, t in stored.items() if key.startswith("0."))


# ==========================================================================
# the fixture itself is non-degenerate (independent of the streaming runtime)
# ==========================================================================
def test_reference_routes_unevenly_and_every_lora_gradient_is_non_zero(tmp_path: Path) -> None:
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    lora = moe_lora_config()
    resident = moe_resident_reference(weights, lora)
    randomize_lora_b(resident)
    batch = moe_batch(config)

    hits = routed_expert_hits(resident, batch)
    assert sorted(hits) == list(_moe_layer_ids(config))
    for layer, seen in hits.items():
        assert 2 <= len(seen) < N_EXPERTS, f"layer {layer} hit {sorted(seen)}"
    assert routed_expert_hits(resident, moe_batch(config, seed=1)) != hits

    grads = run_moe_pass(resident, batch).grads
    assert grads
    zero = [name for name, grad in grads.items() if grad is None or not bool(grad.ne(0).any())]
    assert not zero, f"degenerate gradients: {zero[:4]}"


# ==========================================================================
# forward: logits and loss equal the resident eager model
# ==========================================================================
_FORWARD_CASES = [
    pytest.param("dense", "float32", True, id="dense-fp32"),
    pytest.param("nf4", "float32", True, id="nf4-fp32"),
    pytest.param("nf4-nested", "float32", True, id="nf4-nested-fp32"),
    pytest.param("dense", "bfloat16", True, id="dense-bf16"),
    # bf16 NF4 runs with the freshly initialised (zero) lora_B. The CPU resident reference
    # cannot be NF4 (the 4-bit kernels are GPU-only), so it holds the dequantised attention
    # weights in plain ``Linear`` layers, where peft rounds a non-zero adapter output
    # differently from bnb's ``Linear4bit``. Measured: dense layer 0, which has no routed
    # experts, already differs by ~4e-4 there. A zero adapter removes that reference artefact
    # and leaves the bf16 NF4 routed experts fully compared (bitwise, below).
    pytest.param("nf4", "bfloat16", False, id="nf4-bf16-zero-lora-b"),
    pytest.param("nf4-nested", "bfloat16", False, id="nf4-nested-bf16-zero-lora-b"),
]


@pytest.mark.parametrize(("storage", "dtype", "randomize_b"), _FORWARD_CASES)
def test_streamed_forward_equals_resident_eager(
    tmp_path: Path, storage: str, dtype: str, randomize_b: bool
) -> None:
    """Logits and loss are bitwise the resident eager model's.

    The expert pool computes at the cache's dtype, which is the layer pool's dtype, the
    dtype hidden states arrive in and the dtype every expert NF4 spec dequantises to; the
    bf16 cases would not be equal if any of them differed.
    """
    if storage != "dense":
        pytest.importorskip("bitsandbytes")
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    streamed = build_moe_streamed(
        weights, tmp_path / "shards", storage=storage, dtype=dtype, randomize_b=randomize_b
    )
    try:
        assert streamed.index.dtype == dtype
        assert streamed.runtime.expert_pool.dtype == dtype
        resident = moe_resident_reference(
            weights,
            streamed.lora_config,
            shards=streamed.shards,
            index=streamed.index,
            lora_from=streamed.model,
            dtype=dtype,
        )
        batch = moe_batch(config)
        want = run_moe_pass(resident, batch, backward=False)
        got = run_moe_pass(streamed.model, batch, backward=False)
        assert got.logits.dtype == getattr(torch, dtype)
        assert bool(want.logits.abs().sum() > 0)
        assert torch.equal(got.logits, want.logits), (
            (got.logits.float() - want.logits.float()).abs().max()
        )
        assert torch.equal(got.loss, want.loss)
    finally:
        streamed.close()


def test_forward_comparison_sees_a_wrong_routed_expert(tmp_path: Path) -> None:
    """Control: the equality above is not vacuous; the streamed forward really reads experts."""
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    streamed = build_moe_streamed(weights, tmp_path / "shards")
    try:
        resident = moe_resident_reference(
            weights, streamed.lora_config, lora_from=streamed.model
        )
        batch = moe_batch(config)
        hits = routed_expert_hits(resident, batch)
        assert torch.equal(
            run_moe_pass(streamed.model, batch, backward=False).logits,
            run_moe_pass(resident, batch, backward=False).logits,
        )
        layer, expert = 1, min(hits[1])
        with torch.no_grad():
            resident.get_base_model().model.layers[layer].mlp.experts.down_proj[expert].add_(0.05)
        assert not torch.equal(
            run_moe_pass(streamed.model, batch, backward=False).logits,
            run_moe_pass(resident, batch, backward=False).logits,
        )
    finally:
        streamed.close()


# ==========================================================================
# the layer pool never holds routed experts; the experts are one bounded arena
# ==========================================================================
def test_layer_pool_holds_no_routed_expert_weights(tmp_path: Path) -> None:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3MLP,
        DeepseekV3TopkRouter,
    )

    from soup_cli.utils.layer_stream_runtime import decoder_owner

    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    per_expert = build_moe_streamed(weights, tmp_path / "per_expert")
    fused = build_moe_streamed(weights, tmp_path / "fused", layout="fused")
    try:
        routed = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
        # Control: a fused cache puts them in the layer pool, so the check can see them.
        assert all(key in slot for slot in fused.runtime.pool.buffers for key in routed)

        for slot in per_expert.runtime.pool.buffers:
            assert not [key for key in slot if key.startswith("mlp.experts")]
        for layer in decoder_owner(per_expert.model).layers:
            assert not [key for key in layer.name_map.values() if key.startswith("mlp.experts")]

        # The same checkpoint's layer pool is smaller by exactly the routed experts.
        hidden, inter = config.hidden_size, config.moe_intermediate_size
        expert_bytes = config.n_routed_experts * (2 * inter * hidden + hidden * inter) * 4
        pool = per_expert.runtime.pool
        assert fused.runtime.pool.nbytes - pool.nbytes == pool.n * expert_bytes
        assert per_expert.runtime.stats()["buffer_bytes"] < fused.runtime.stats()["buffer_bytes"]

        modules = streamed_experts_modules(per_expert.model)
        assert sorted(modules) == list(_moe_layer_ids(config))
        for layer, module in modules.items():
            assert not list(module.parameters())
            block = decoder_owner(per_expert.model).layers[layer].inner.mlp
            assert block.experts is module
            assert type(block.gate) is DeepseekV3TopkRouter
            assert type(block.shared_experts) is DeepseekV3MLP
    finally:
        per_expert.close()
        fused.close()


def test_one_bounded_slot_and_one_reader_per_moe_layer(tmp_path: Path) -> None:
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    streamed = build_moe_streamed(weights, tmp_path / "shards")
    try:
        runtime, index = streamed.runtime, streamed.index
        modules = streamed_experts_modules(streamed.model)
        pool = runtime.expert_pool
        assert {id(module.pool) for module in modules.values()} == {id(pool)}
        assert len(runtime.expert_readers) == len(index.expert_layers) == len(modules) == 2
        assert {id(module.reader) for module in modules.values()} == {
            id(reader) for reader in runtime.expert_readers
        }
        # One expert's dense weights, from the config, whatever the expert count is.
        hidden, inter = config.hidden_size, config.moe_intermediate_size
        assert pool.plan.dense_bytes == (2 * inter * hidden + hidden * inter) * 4
        assert pool.weight_budget_bytes == pool.plan.weight_bytes
        assert pool.stats()["slot_count"] == 1
        stats = runtime.stats()
        assert stats["expert_buffer_bytes"] == pool.plan.weight_bytes
        assert pool.device.type == "cpu"
    finally:
        streamed.close()


def test_a_fused_cache_installs_no_expert_streaming(tmp_path: Path) -> None:
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    fused = build_moe_streamed(weights, tmp_path / "fused", layout="fused")
    try:
        assert fused.index.expert_layout == ""
        assert fused.runtime.expert_readers == () and fused.runtime.expert_pool is None
        assert not streamed_experts_modules(fused.model)
        run_moe_pass(fused.model, moe_batch(config), backward=False)
        stats = fused.runtime.stats()
        assert (stats["expert_fetches"], stats["expert_bytes_read"]) == (0, 0)
        assert stats["expert_buffer_bytes"] == 0
        assert {
            "n_layers", "buffers", "buffer_bytes", "large_buffer_bytes", "store_bytes", "pinned",
            "pinned_bytes", "tier", "read_ahead", "disk_bytes", "layer_loads", "layer_restores",
            "large_loads", "device", "total_params",
        } <= set(stats)
        assert stats["layer_loads"] > 0
    finally:
        fused.close()


# ==========================================================================
# observable expert reads
# ==========================================================================
@pytest.mark.parametrize("storage", ["dense", "nf4-nested"])
def test_expert_fetch_and_byte_counters_follow_the_experts_the_router_chose(
    tmp_path: Path, storage: str
) -> None:
    if storage != "dense":
        pytest.importorskip("bitsandbytes")
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    streamed = build_moe_streamed(weights, tmp_path / "shards", storage=storage)
    try:
        resident = moe_resident_reference(
            weights, streamed.lora_config, shards=streamed.shards, index=streamed.index,
            lora_from=streamed.model,
        )
        batch = moe_batch(config)
        hits = routed_expert_hits(resident, batch)
        assert any(len(seen) < N_EXPERTS for seen in hits.values())

        before = streamed.runtime.stats()
        assert (before["expert_fetches"], before["expert_bytes_read"]) == (0, 0)
        reads = record_expert_reads(streamed.model)
        run_moe_pass(streamed.model, batch, backward=False)

        # One read per expert the router chose, in ascending order, none for unused experts.
        assert reads == {layer: sorted(seen) for layer, seen in hits.items()}
        expected = sum(len(seen) for seen in hits.values())
        stats = streamed.runtime.stats()
        assert stats["expert_fetches"] == expected
        assert stats["expert_bytes_read"] == expected * _stored_bytes_of_one_expert(
            streamed.shards, streamed.index
        )
        assert sum(reader.fetches for reader in streamed.runtime.expert_readers) == expected
        assert sum(reader.bytes_read for reader in streamed.runtime.expert_readers) == (
            stats["expert_bytes_read"]
        )
        assert stats["layer_loads"] > 0
    finally:
        streamed.close()


# ==========================================================================
# the runtime owns and closes what it opened
# ==========================================================================
def test_close_releases_the_expert_readers_and_pool(tmp_path: Path) -> None:
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    streamed = build_moe_streamed(weights, tmp_path / "shards")
    runtime = streamed.runtime
    try:
        run_moe_pass(streamed.model, moe_batch(config), backward=False)
        assert all(reader._handle is not None for reader in runtime.expert_readers)
        assert not runtime.expert_pool.stats()["closed"]
    finally:
        streamed.close()
    assert all(reader._handle is None for reader in runtime.expert_readers)
    assert runtime.expert_pool.stats()["closed"]
    streamed.close()  # idempotent


def test_close_attempts_every_release_even_when_the_pool_refuses(tmp_path: Path) -> None:
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    streamed = build_moe_streamed(weights, tmp_path / "shards")
    runtime = streamed.runtime
    run_moe_pass(streamed.model, moe_batch(config), backward=False)
    pool, first = runtime.expert_pool, runtime.expert_readers[0]
    try:
        with pool.lease(first, 0):
            with pytest.raises(RuntimeError, match="active lease"):
                runtime.close()
            assert all(reader._handle is None for reader in runtime.expert_readers)
            assert runtime.hook is None
    finally:
        runtime.close()
    assert pool.stats()["closed"]


def _spy_expert_resources(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    """Record every expert reader and pool the runtime creates, and whether it was closed."""
    from soup_cli.utils import expert_shard, expert_stream

    made: Dict[str, List[Any]] = {"readers": [], "pools": []}

    class SpyReader(expert_shard.ExpertShardReader):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            made["readers"].append(self)

        def close(self) -> None:
            if hasattr(self, "close_calls"):
                self.close_calls += 1
            super().close()

    class SpyPool(expert_stream.ExpertBufferPool):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            made["pools"].append(self)

    monkeypatch.setattr(expert_shard, "ExpertShardReader", SpyReader)
    monkeypatch.setattr(expert_stream, "ExpertBufferPool", SpyPool)
    return made


def test_a_failed_expert_install_closes_the_readers_and_pool_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index drops MoE layer 2, whose model layer still has routed experts."""
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    shards = tmp_path / "shards"
    index = shard_moe(weights, shards)
    bad = dataclasses.replace(index, expert_layers=(1,))
    made = _spy_expert_resources(monkeypatch)

    with pytest.raises(ValueError, match="model/cache experts disagree"):
        build_moe_streamed(weights, shards, index=bad)

    assert len(made["readers"]) == 1 and len(made["pools"]) == 1
    assert made["pools"][0].stats()["closed"]
    assert made["readers"][0].close_calls >= 1


def test_a_missing_experts_file_is_refused_by_name_and_closes_the_readers_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from soup_cli.utils.layer_shard import expert_paths

    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    shards = tmp_path / "shards"
    index = shard_moe(weights, shards)
    Path(expert_paths(str(shards), index)[2]).unlink()
    made = _spy_expert_resources(monkeypatch)

    with pytest.raises(FileNotFoundError, match="experts_002"):
        build_moe_streamed(weights, shards, index=index)

    assert len(made["readers"]) == 1 and made["pools"] == []
    assert made["readers"][0].close_calls >= 1


def test_a_failure_after_the_experts_were_installed_still_closes_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layer 0 is the dense layer: nothing reads its file before the layer wiring does, which
    runs after the routed experts are installed and their readers opened."""
    from soup_cli.utils.layer_shard import layer_paths

    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    shards = tmp_path / "shards"
    index = shard_moe(weights, shards)
    Path(layer_paths(str(shards), index)[0]).unlink()
    made = _spy_expert_resources(monkeypatch)

    with pytest.raises(OSError, match="layer_000"):
        build_moe_streamed(weights, shards, index=index)

    assert len(made["readers"]) == 2 and len(made["pools"]) == 1
    assert made["pools"][0].stats()["closed"]
    assert all(reader.close_calls >= 1 and reader._handle is None for reader in made["readers"])


# ==========================================================================
# adapters on routed experts are refused before any expert is read
# ==========================================================================
def _expert_adapter_configs() -> Dict[str, Any]:
    from peft import LoraConfig, TaskType

    common = dict(r=2, lora_alpha=4, lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM)
    return {
        # peft turns bare gate/up/down targets into parameter targets on the fused experts.
        "gate-up-down-modules": LoraConfig(
            target_modules=["gate_proj", "up_proj", "down_proj"], **common
        ),
        "expert-parameters": LoraConfig(
            target_modules=list(MLA_LORA_TARGETS),
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            **common,
        ),
    }


@pytest.mark.parametrize("name", sorted(_expert_adapter_configs()))
def test_adapters_on_routed_experts_are_refused_before_any_expert_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from soup_cli.utils import expert_shard

    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    shards = tmp_path / "shards"
    index = shard_moe(weights, shards)
    made = _spy_expert_resources(monkeypatch)
    reads: List[int] = []
    original = expert_shard.ExpertShardReader.read_host

    def counted(self: Any, expert: int, staging: Any) -> Any:
        reads.append(expert)
        return original(self, expert, staging)

    monkeypatch.setattr(expert_shard.ExpertShardReader, "read_host", counted)

    with pytest.raises(ValueError, match="adapters"):
        build_moe_streamed(
            weights, shards, index=index, lora_config=_expert_adapter_configs()[name]
        )

    assert reads == []
    assert made["pools"] and all(pool.stats()["closed"] for pool in made["pools"])
    assert made["readers"] and all(reader.close_calls >= 1 for reader in made["readers"])


# ==========================================================================
# E2: ordinary checkpointed streamed backward equals the resident eager model
# ==========================================================================
def test_e2_checkpointed_backward_lora_gradients_equal_resident_eager(tmp_path: Path) -> None:
    """E2 (spec): CPU fp32 ``torch.equal`` on logits, loss and every LoRA gradient."""
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    streamed = build_moe_streamed(weights, tmp_path / "shards")
    try:
        resident = moe_resident_reference(
            weights, streamed.lora_config, lora_from=streamed.model
        )
        batch = moe_batch(config)
        want = run_moe_pass(resident, batch)
        got = run_moe_pass(streamed.model, batch)

        assert set(got.grads) == set(want.grads) and got.grads
        for name, grad in got.grads.items():
            assert grad is not None and bool(grad.ne(0).any()), f"{name}: no or zero gradient"
        assert torch.equal(got.logits, want.logits)
        assert torch.equal(got.loss, want.loss)
        assert grads_differ(got.grads, want.grads) == []
    finally:
        streamed.close()


def test_checkpointed_backward_reads_each_hit_expert_three_times(tmp_path: Path) -> None:
    """Forward, checkpoint recompute and the expert's own backward: three reads per visit."""
    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    streamed = build_moe_streamed(weights, tmp_path / "shards")
    try:
        resident = moe_resident_reference(
            weights, streamed.lora_config, lora_from=streamed.model
        )
        batch = moe_batch(config)
        hits = routed_expert_hits(resident, batch)
        reads = record_expert_reads(streamed.model)
        run_moe_pass(streamed.model, batch)

        assert sorted(reads) == sorted(hits)
        for layer, seen in hits.items():
            assert sorted(reads[layer]) == sorted(list(seen) * 3), f"layer {layer}"
        stats = streamed.runtime.stats()
        assert stats["expert_fetches"] == 3 * sum(len(seen) for seen in hits.values())
        assert stats["expert_bytes_read"] == stats["expert_fetches"] * _stored_bytes_of_one_expert(
            streamed.shards, streamed.index
        )
    finally:
        streamed.close()
