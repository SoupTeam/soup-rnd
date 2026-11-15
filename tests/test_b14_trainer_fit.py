"""B1.4: the trainer's VRAM preflight must not charge a per_expert run for the whole layer.

Seam: the inputs ``_setup_streaming_transformers`` hands to ``_stream_budget_lines``, built
from a real shard cache. The setup method itself loads a tokenizer and a checkpoint, so the
composition of those inputs is repeated here, next to the helpers it calls.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.test_b1_expert_shard import _checkpoint, _shard

SMALL, LARGE = 12, 96


ROWS, COLS, BIG_EXPERTS = 192, 256, 96
# gate (ROWS, COLS), up (ROWS, COLS), down (COLS, ROWS), fp32: about 56.6 MB per layer, more
# than the 33.5 MB expert arena (one slot plus the CUDA allocator allowance) and than the
# 47.6 MB the preflight needs for a per_expert run, measured on this fixture.
EXPERT_LAYER_BYTES = BIG_EXPERTS * 3 * ROWS * COLS * 4


def _big_checkpoint(root):
    import torch
    from safetensors.torch import save_file

    root.mkdir(parents=True)
    tensors = {"model.norm.weight": torch.ones(4)}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "self_attn.q_proj.weight"] = torch.eye(4)
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = torch.zeros(BIG_EXPERTS)
        tensors[prefix + "mlp.shared_experts.down_proj.weight"] = torch.ones(4, 3)
        for expert in range(BIG_EXPERTS):
            for projection, shape in (
                ("gate_proj", (ROWS, COLS)), ("up_proj", (ROWS, COLS)), ("down_proj", (COLS, ROWS)),
            ):
                tensors[prefix + f"mlp.experts.{expert}.{projection}.weight"] = torch.zeros(shape)
    save_file(tensors, str(root / "model.safetensors"))


def _inputs_for(tmp_path, source, name, expert_layout):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin, _expert_buffer_requirements
    from soup_cli.utils.layer_shard import layer_paths
    from soup_cli.utils.layer_stream_runtime import RamSource, extras_resident_bytes

    cache = tmp_path / name
    kwargs = {"expert_layout": expert_layout} if expert_layout else {}
    index = _shard(source, cache, **kwargs)
    specs = RamSource.layer_specs_from_paths(layer_paths(str(cache), index))
    expert_weight_bytes, _ = _expert_buffer_requirements(str(cache), index, device="cuda:0")
    return SimpleNamespace(
        index=index,
        layer_bytes=StreamingSetupMixin._stream_layer_budget_bytes(specs),
        embed_bytes=extras_resident_bytes(str(cache)),
        expert_weight_bytes=expert_weight_bytes,
    )


def _preflight_inputs(tmp_path, *, count, expert_layout):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source"
    _checkpoint(source, count=count)
    return _inputs_for(tmp_path, source, "cache", expert_layout)


def _min_free_vram(inputs, monkeypatch):
    """Smallest free-VRAM figure the real preflight accepts, found by bisection."""
    import torch

    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils import layer_stream_runtime

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1, 1))
    monkeypatch.setattr(layer_stream_runtime, "measure_gemm_tflops", lambda **kwargs: None)
    wrapper = StreamingSetupMixin()
    wrapper.device = "cpu"
    cfg = SimpleNamespace(data=SimpleNamespace(max_length=8))
    shape = SimpleNamespace(vocab_size=8, hidden_size=4, intermediate_size=8)

    def fits(free):
        tcfg = SimpleNamespace(
            batch_size=1, stream_buffers=2, stream_vram_override=free, stream_vram_probe=False,
            gradient_accumulation_steps=1,
        )
        try:
            wrapper._stream_budget_lines(
                cfg, tcfg, model_config=shape, layer_bytes=inputs.layer_bytes,
                embed_bytes=inputs.embed_bytes, index=inputs.index, on_cuda=True,
                expert_weight_bytes=inputs.expert_weight_bytes,
                adapter_budget=SimpleNamespace(stored_parameters=0, total=None),
            )
        except ValueError as error:
            assert "predicted to need" in str(error)
            return False
        return True

    low, high = 0, 2**40
    assert fits(high) and not fits(low)
    while high - low > 1:
        mid = (low + high) // 2
        low, high = (low, mid) if fits(mid) else (mid, high)
    return high


def test_layer_bigger_than_free_vram_is_accepted_per_expert_and_refused_fused(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    _big_checkpoint(source)
    per_expert = _min_free_vram(
        _inputs_for(tmp_path, source, "per_expert", "per_expert"), monkeypatch
    )
    fused = _min_free_vram(_inputs_for(tmp_path, source, "fused", ""), monkeypatch)
    assert per_expert < EXPERT_LAYER_BYTES <= fused


def test_per_expert_layer_budget_grows_only_by_router_bias_and_fused_by_weights(tmp_path):
    # From the ``_checkpoint`` fixture: the router bias is one fp32 element per expert and
    # stays in the dense layer file; one expert holds gate (3, 4), up (3, 4), down (4, 3).
    bias_bytes_per_expert = 4
    weight_bytes_per_expert = (12 + 12 + 12) * 4
    extra = LARGE - SMALL

    small = _preflight_inputs(tmp_path / "small", count=SMALL, expert_layout="per_expert")
    large = _preflight_inputs(tmp_path / "large", count=LARGE, expert_layout="per_expert")
    assert large.layer_bytes - small.layer_bytes == extra * bias_bytes_per_expert
    assert large.expert_weight_bytes == small.expert_weight_bytes > 0

    fused_small = _preflight_inputs(tmp_path / "fs", count=SMALL, expert_layout="")
    fused_large = _preflight_inputs(tmp_path / "fl", count=LARGE, expert_layout="")
    assert fused_large.layer_bytes - fused_small.layer_bytes >= extra * weight_bytes_per_expert
