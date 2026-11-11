"""ZERO: L2L on the hybrid ``qwen3_5`` decoder (Gated DeltaNet + gated attention, 3:1).

Its text model hands the two layer types different arguments (a recurrent mask for
``linear_attention``, a causal mask for ``full_attention``). An L2L step that replays
layer 0's arguments to every layer would train the attention layers on the wrong mask
without raising, so the step must replay the captured arguments per layer type, and an
L2L step must equal gradient accumulation on a model with both types.
"""
from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, hermetic_ram

VOCAB = 64
HIDDEN = 64


def _tiny_qwen35_dir(tmp_path, n_layers: int = 4):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(11)
    config = AutoConfig.for_model(
        "qwen3_5_text", vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, full_attention_interval=4, tie_word_embeddings=False,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights), config


@pytest.fixture()
def qwen35_streamed(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights, config = _tiny_qwen35_dir(tmp_path)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="qwen3")
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=lora,
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
    )
    hermetic_ram(monkeypatch, tmp_path)
    yield model, runtime, config
    runtime.close()


def _store(k, n_layers=4):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    return ActivationStore(
        n_layers=n_layers, k=k, chunk_shape=(1, SEQ, HIDDEN),
        dtype=torch.float32, device="cpu", pin=False,
    )


def _micro_batches(k, seed=29):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (1, SEQ), generator=gen) for _ in range(k)]


def test_fixture_has_both_layer_types(qwen35_streamed):
    _model, _runtime, config = qwen35_streamed
    assert set(config.layer_types) == {"linear_attention", "full_attention"}


def test_capture_records_arguments_per_layer_type(qwen35_streamed):
    """for_layer(i) returns what HF passes to layer i, for every layer of both types."""
    import torch

    from soup_cli.utils.l2l_step import capture_layer_call, model_parts

    model, _runtime, config = qwen35_streamed
    ids = _micro_batches(1)[0]
    call = capture_layer_call(model, ids, draws=False)
    assert set(call.typed) == {"linear_attention", "full_attention"}
    assert call.layer_types == tuple(config.layer_types)

    _causal, owner = model_parts(model)
    actual = {}

    def make(i):
        def hook(_m, args, kwargs):
            actual[i] = (tuple(args[1:]), dict(kwargs))
        return hook

    handles = [layer.register_forward_pre_hook(make(i), with_kwargs=True)
               for i, layer in enumerate(owner.layers)]
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    def same(a, b):
        if torch.is_tensor(a) or torch.is_tensor(b):
            return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a is b or a == b

    for i in range(len(owner.layers)):
        args, kwargs = call.for_layer(i)
        want_args, want_kwargs = actual[i]
        want_kwargs.pop("num_items_in_batch", None)
        assert same(args, want_args), i
        assert set(kwargs) == set(want_kwargs), i
        for key in kwargs:
            if key == "past_key_values":
                continue
            assert same(kwargs[key], want_kwargs[key]), (i, key)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_l2l_step_equals_gradient_accumulation_on_qwen35(qwen35_streamed, k):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(k)

    harness.zero_grads(model)
    ga_losses = harness.ga_step(model, mbs)
    ga = harness.collect_grads(model)

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    l2l_losses = step.run(mbs, _store(k))
    l2l = harness.collect_grads(model)

    result = harness.compare(ga, l2l)
    assert result["all_zero"] == []
    assert result["equal"], result
    assert harness.losses_equal(ga_losses, l2l_losses)


def test_trainer_admits_qwen35():
    from types import SimpleNamespace

    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    for arch in ("qwen3_5", "qwen3_5_text"):
        refuse_unsupported_l2l(SimpleNamespace(model_type=arch), n_gpu=1)
