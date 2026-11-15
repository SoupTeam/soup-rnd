"""ZERO-2: training.lora_layers — LoRA adapters trained on a subset of the decoder layers."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

N_LAYERS = 6
VOCAB = 64


def _model(kind: str = "llama"):
    pytest.importorskip("peft")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(7)
    if kind == "llama":
        config = AutoConfig.for_model(
            "llama", vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
            num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        targets = ["q_proj", "v_proj", "down_proj"]
    else:
        config = AutoConfig.for_model(
            "qwen3_5_text", vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
            linear_value_head_dim=16, full_attention_interval=4, tie_word_embeddings=False,
            max_position_embeddings=64,
        )
        targets = ["q_proj", "v_proj", "in_proj_qkv", "out_proj", "down_proj"]
    model = AutoModelForCausalLM.from_config(config).float()
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets,
                                            lora_dropout=0.0))


def _batches(n=2, seq=12, seed=3):
    import torch

    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(0, VOCAB, (2, seq), generator=gen)
        out.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids})
    return out


def _trainable_layers(model):
    from soup_cli.utils.layer_select import _LAYER_RE

    return sorted({int(_LAYER_RE.search(n).group(1)) for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n})


def test_adapters_are_grouped_by_layer():
    from soup_cli.utils.layer_select import adapter_params_by_layer

    by_layer = adapter_params_by_layer(_model())
    assert sorted(by_layer) == list(range(N_LAYERS))
    assert all("lora_" in name for params in by_layer.values() for name, _p in params)


def test_qwen35_layers_are_found_in_both_layer_types():
    from soup_cli.utils.layer_select import adapter_params_by_layer

    by_layer = adapter_params_by_layer(_model("qwen3_5"))
    assert sorted(by_layer) == [0, 1, 2, 3]
    names = {n for params in by_layer.values() for n, _p in params}
    assert any("in_proj_qkv" in n for n in names) and any("q_proj" in n for n in names)


def test_no_adapter_is_refused():
    import torch

    from soup_cli.utils.layer_select import adapter_params_by_layer

    with pytest.raises(ValueError, match="no trainable LoRA"):
        adapter_params_by_layer(torch.nn.Linear(2, 2))


def test_mask_freezes_the_other_layers_and_refuses_unknown_ones():
    from soup_cli.utils.layer_select import adapter_params_by_layer, apply_layer_mask

    model = _model()
    by_layer = adapter_params_by_layer(model)
    count = apply_layer_mask(by_layer, [1, 4])
    assert _trainable_layers(model) == [1, 4]
    assert count == sum(p.numel() for idx in (1, 4) for _n, p in by_layer[idx])
    apply_layer_mask(by_layer, [0])
    assert _trainable_layers(model) == [0]
    with pytest.raises(ValueError, match="no adapter"):
        apply_layer_mask(by_layer, [N_LAYERS])


def test_frozen_layers_get_no_gradient():
    from soup_cli.utils.layer_select import adapter_params_by_layer, apply_layer_mask

    model = _model()
    by_layer = adapter_params_by_layer(model)
    apply_layer_mask(by_layer, [2, 3])
    model(**_batches(1)[0]).loss.backward()
    for idx, params in by_layer.items():
        got = [p.grad is not None for _n, p in params]
        assert all(got) if idx in (2, 3) else not any(got), idx


def test_top_k_and_draw():
    from soup_cli.utils.layer_select import draw, top_k

    assert top_k({0: 0.1, 1: 0.9, 2: 0.5, 3: 0.9}, 2) == [1, 3]
    assert top_k({0: 1.0, 1: 1.0}, 1) == [0]  # ties: the lower index
    assert top_k({0: 1.0}, 3) == [0]
    picked = draw(range(10), 3, random.Random(0))
    assert picked == draw(range(10), 3, random.Random(0)) and len(set(picked)) == 3


def test_grad_scores_cover_every_layer_and_leave_no_grad():
    from soup_cli.utils.layer_select import adapter_params_by_layer, grad_scores

    model = _model()
    by_layer = adapter_params_by_layer(model)
    scores = grad_scores(model, by_layer, _batches(), "cpu")
    assert sorted(scores) == list(range(N_LAYERS))
    assert all(v > 0 for v in scores.values())
    assert all(p.grad is None for p in model.parameters())


def test_angular_scores_are_distances():
    from soup_cli.utils.layer_select import angular_scores

    scores = angular_scores(_model(), N_LAYERS, _batches(), "cpu")
    assert sorted(scores) == list(range(N_LAYERS))
    assert all(0.0 <= v <= 1.0 for v in scores.values())


@pytest.mark.parametrize("method", ["grad", "angular"])
def test_scored_choice_restores_the_rng_and_the_mode(method):
    import torch

    from soup_cli.utils.layer_select import LayerSelectPolicy, adapter_params_by_layer, choose

    model = _model()
    model.train()
    by_layer = adapter_params_by_layer(model)
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    sel = choose(LayerSelectPolicy(method, k=2), model, by_layer, _batches(), "cpu",
                 random.Random(0))
    assert torch.equal(torch.random.get_rng_state(), before)
    assert model.training
    assert len(sel.layers) == 2 and set(sel.scores) == set(range(N_LAYERS))
    assert sel.layers == sorted(sel.layers, key=lambda i: i)
    assert set(sel.layers) == set(sorted(sel.scores, key=lambda i: -sel.scores[i])[:2])


def test_policy_validation():
    from soup_cli.utils.layer_select import LayerSelectPolicy

    with pytest.raises(ValueError):
        LayerSelectPolicy("best", k=2)
    with pytest.raises(ValueError):
        LayerSelectPolicy("fixed")
    with pytest.raises(ValueError):
        LayerSelectPolicy("grad")
    LayerSelectPolicy("fixed", layers=(1, 2))


# -- the callback ------------------------------------------------------------------------------
def _callback(method, tmp_path, **kw):
    from soup_cli.utils import layer_select

    dataset = [{"input_ids": b["input_ids"][0].tolist()} for b in _batches(3)]

    def collator(items):
        import torch

        ids = torch.tensor([it["input_ids"] for it in items])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}

    policy = layer_select.LayerSelectPolicy(method, **kw)
    return layer_select.LayerSelectCallback(policy, dataset=dataset, collator=collator,
                                            batch_size=2, output_dir=str(tmp_path))


@pytest.mark.parametrize("method,kw", [
    ("fixed", {"layers": (0, 5)}), ("random", {"k": 3}), ("grad", {"k": 2}),
    ("angular", {"k": 2}), ("lisa", {"k": 2}),
])
def test_callback_selects_at_train_begin_and_records(tmp_path, method, kw):
    model = _model()
    cb = _callback(method, tmp_path, **kw)
    args = SimpleNamespace(device="cpu")
    cb.on_train_begin(args, SimpleNamespace(global_step=0), None, model=model)
    layers = _trainable_layers(model)
    assert layers == cb.selection.layers
    assert len(layers) == (2 if method == "fixed" else kw["k"])
    record = json.loads((tmp_path / "lora_layers.json").read_text())
    assert record["method"] == method and record["layers"] == layers
    if method in ("grad", "angular"):
        assert len(record["scores"]) == N_LAYERS


def test_lisa_redraws_on_the_interval_and_clears_optimizer_state(tmp_path):
    import torch

    model = _model()
    cb = _callback("lisa", tmp_path, k=2, interval_steps=3, seed=5)
    args = SimpleNamespace(device="cpu")
    cb.on_train_begin(args, SimpleNamespace(global_step=0), None, model=model)
    first = list(cb.selection.layers)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if "lora_" in
                                   next(n for n, q in model.named_parameters() if q is p)])
    for idx in first:
        for _n, p in cb.by_layer[idx]:
            optimizer.state[p] = {"step": torch.tensor(1.0)}
    cb.on_step_end(args, SimpleNamespace(global_step=2), None, model=model, optimizer=optimizer)
    assert cb.selection.layers == first  # not on the interval
    seen = set()
    for step in (3, 6, 9, 12):
        cb.on_step_end(args, SimpleNamespace(global_step=step), None, model=model,
                       optimizer=optimizer)
        assert _trainable_layers(model) == cb.selection.layers
        seen.add(tuple(cb.selection.layers))
    assert len(seen) > 1
    for idx in set(first) - set(cb.selection.layers):
        for _n, p in cb.by_layer[idx]:
            assert optimizer.state[p] == {}
    record = json.loads((tmp_path / "lora_layers.json").read_text())
    assert [h["step"] for h in record["history"]] == [0, 3, 6, 9, 12]


def test_non_lisa_methods_never_redraw(tmp_path):
    model = _model()
    cb = _callback("random", tmp_path, k=2)
    args = SimpleNamespace(device="cpu")
    cb.on_train_begin(args, SimpleNamespace(global_step=0), None, model=model)
    first = list(cb.selection.layers)
    cb.on_step_end(args, SimpleNamespace(global_step=20), None, model=model)
    assert cb.selection.layers == first


# -- schema ------------------------------------------------------------------------------------
def _cfg(**training):
    from soup_cli.config.schema import SoupConfig

    settings = {"lora": {"r": 4, "target_modules": ["q_proj"]}}
    settings.update(training)
    return SoupConfig.model_validate({
        "base": "example/model", "data": {"train": "train.jsonl", "chat_template": "chatml"},
        "training": settings,
    })


def test_schema_accepts_each_method():
    assert _cfg(lora_layers={"method": "grad", "k": 8}).training.lora_layers.k == 8
    assert _cfg(lora_layers={"method": "fixed", "layers": [1, 2]}).training.lora_layers.layers
    assert _cfg(lora_layers={"method": "lisa", "k": 2, "interval_steps": 5})
    assert _cfg().training.lora_layers is None


@pytest.mark.parametrize("bad", [
    {"method": "grad"}, {"method": "fixed", "k": 2}, {"method": "grad", "k": 2, "layers": [1]},
    {"method": "fixed", "layers": [-1]}, {"method": "grad", "k": 0}, {"method": "best", "k": 2},
    {"method": "grad", "k": 2, "extra": 1},
])
def test_schema_refuses_bad_blocks(bad):
    with pytest.raises(Exception):
        _cfg(lora_layers=bad)


def test_schema_refuses_full_finetuning_and_other_freezes():
    with pytest.raises(Exception, match="lora.r=0"):
        _cfg(lora={"r": 0}, lora_layers={"method": "grad", "k": 2})
    with pytest.raises(Exception, match="freeze_layers"):
        _cfg(freeze_layers=2, lora_layers={"method": "grad", "k": 2})


def test_attach_reads_the_config(tmp_path):
    from soup_cli.utils.layer_select import attach_layer_select_callback

    added = []
    trainer = SimpleNamespace(add_callback=added.append, train_dataset=[], data_collator=None,
                              args=SimpleNamespace(per_device_train_batch_size=4))
    tcfg = _cfg(lora_layers={"method": "lisa", "k": 3, "interval_steps": 7}).training
    assert attach_layer_select_callback(trainer, tcfg, str(tmp_path))
    assert added[0].policy.k == 3 and added[0].policy.interval_steps == 7
    assert added[0].batch_size == 4
    assert not attach_layer_select_callback(trainer, _cfg().training, str(tmp_path))


def test_frozen_adapters_still_count_as_the_saved_adapter(tmp_path):
    """The streamed save guard expects every adapter tensor, frozen by selection or not."""
    from soup_cli.utils.layer_select import adapter_params_by_layer, apply_layer_mask
    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    model = _model()
    apply_layer_mask(adapter_params_by_layer(model), [1])
    model.save_pretrained(str(tmp_path))
    assert_streamed_adapter_saved(model, str(tmp_path))
