"""ZERO-2 x E2: the frozen bottom's output reused across epochs (utils/prefix_cache.py) and the
progressive freeze of ``training.lora_layers`` that feeds it."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_layer_select import _cfg, _model

torch = pytest.importorskip("torch")


def _rows(n=3, seq=10, seed=0, batch=1):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 64, (batch, seq), generator=gen) for _ in range(n)]


def _prepare(kind, trainable_from, gc=False):
    from soup_cli.utils.layer_select import (
        adapter_params_by_layer,
        apply_layer_mask,
        cut_input_grads,
    )

    model = _model(kind)
    if gc:
        model.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    by_layer = adapter_params_by_layer(model)
    apply_layer_mask(by_layer, [i for i in by_layer if i >= trainable_from])
    cut_input_grads(model)
    model.train()
    return model


def _train(model, rows, epochs=3, masks=None):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(epochs):
        for i, ids in enumerate(rows):
            mask = torch.ones_like(ids) if masks is None else masks[i]
            labels = ids.masked_fill(mask == 0, -100)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach())
    return losses, {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("kind", ["llama", "qwen3_5"])
@pytest.mark.parametrize("gc", [False, True])
def test_cached_training_is_bit_identical(kind, gc):
    """Batch 1: three epochs with the cache give the same losses and adapters, bit for bit."""
    from soup_cli.utils.prefix_cache import PrefixCache

    rows = _rows()
    plain_losses, plain_state = _train(_prepare(kind, 2, gc), rows)
    model = _prepare(kind, 2, gc)
    cache = PrefixCache(model, 2, 1 << 30).install()
    losses, state = _train(model, rows)
    assert all(torch.equal(a, b) for a, b in zip(plain_losses, losses))
    assert state.keys() == plain_state.keys()
    assert all(torch.equal(state[n], plain_state[n]) for n in state)
    assert (cache.stats.misses, cache.stats.hits, cache.stats.rows_stored) == (3, 6, 3)


@pytest.mark.parametrize("kind", ["llama", "qwen3_5"])
def test_a_hit_skips_the_bottom_layers(kind):
    import types

    from soup_cli.utils.prefix_cache import PrefixCache, find_decoder

    model = _prepare(kind, 2)
    decoder = find_decoder(model)
    ran = []
    for idx in range(2):  # count the real computation of the two bottom layers
        layer = decoder.layers[idx]
        real = layer.forward

        def spy(*a, _real=real, _idx=idx, **k):
            ran.append(_idx)
            return _real(*a, **k)

        layer.forward = spy
    PrefixCache(model, 2, 1 << 30).install()
    assert isinstance(decoder.layers[0].forward, types.MethodType)
    ids = _rows(1)[0]
    model(input_ids=ids, labels=ids)
    assert ran == [0, 1]  # miss: the bottom runs
    ran.clear()
    model(input_ids=ids, labels=ids)
    assert ran == []  # hit: neither bottom layer computes
    model.eval()
    model(input_ids=ids, labels=ids)
    assert ran == [0, 1]  # evaluation runs the whole model


def test_padded_batches_match_the_uncached_loss():
    """Right padding, batch 2: the loss with the cache matches the uncached loss on real tokens."""
    from soup_cli.utils.prefix_cache import PrefixCache

    rows = _rows(n=2, seq=10, batch=2)
    masks = []
    for ids in rows:
        mask = torch.ones_like(ids)
        mask[1, 7:] = 0
        masks.append(mask)
    plain, plain_state = _train(_prepare("qwen3_5", 2), rows, epochs=2, masks=masks)
    model = _prepare("qwen3_5", 2)
    cache = PrefixCache(model, 2, 1 << 30).install()
    cached, state = _train(model, rows, epochs=2, masks=masks)
    assert cache.stats.hits == 2
    for a, b in zip(plain, cached):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    for name in state:
        assert torch.allclose(state[name], plain_state[name], atol=1e-5, rtol=1e-4), name


def test_budget_limits_what_is_kept_and_stays_correct():
    from soup_cli.utils.prefix_cache import PrefixCache

    rows = _rows()
    plain, _ = _train(_prepare("llama", 2), rows, epochs=2)
    model = _prepare("llama", 2)
    one_row = 10 * 32 * 4
    cache = PrefixCache(model, 2, one_row).install()
    cached, _ = _train(model, rows, epochs=2)
    assert cache.stats.rows_stored == 1 and cache.stats.rows_not_stored >= 2
    assert all(torch.equal(a, b) for a, b in zip(plain, cached))


def test_guards():
    from soup_cli.utils.layer_select import adapter_params_by_layer, apply_layer_mask
    from soup_cli.utils.prefix_cache import PrefixCache, find_decoder

    model = _model("llama")
    apply_layer_mask(adapter_params_by_layer(model), [1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="frozen"):
        PrefixCache(model, 2, 1 << 20)
    with pytest.raises(ValueError, match="cutoff"):
        PrefixCache(_prepare("llama", 2), 0, 1 << 20)

    model = _prepare("llama", 2)
    find_decoder(model).layers[0].self_attn.q_proj.lora_dropout["default"] = torch.nn.Dropout(0.1)
    with pytest.raises(ValueError, match="dropout"):
        PrefixCache(model, 2, 1 << 20).install()

    model = _prepare("llama", 2)
    cache = PrefixCache(model, 2, 1 << 20).install()
    ids = _rows(1)[0]
    model(input_ids=ids, labels=ids)
    with torch.no_grad():
        find_decoder(model).layers[0].mlp.down_proj.base_layer.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="changed in place"):
        model(input_ids=ids, labels=ids)
    cache.remove()
    model(input_ids=ids, labels=ids)  # restored forwards run normally
    assert all("forward" not in vars(layer) for layer in find_decoder(model).layers)


# ------------------------------------------------------------------ progressive


def _callback(tmp_path, **kw):
    from soup_cli.utils import layer_select

    policy = layer_select.LayerSelectPolicy("progressive", **kw)
    return layer_select.LayerSelectCallback(policy, output_dir=str(tmp_path))


def _trainable(model):
    from tests.test_layer_select import _trainable_layers

    return _trainable_layers(model)


def test_progressive_schedule_freezes_at_the_step_and_installs_the_cache(tmp_path):
    import json

    model = _model("llama")
    cb = _callback(tmp_path, freeze_step=2, freeze_below=3, prefix_cache=True)
    args, state = SimpleNamespace(device="cpu"), SimpleNamespace(global_step=0)
    cb.on_train_begin(args, state, None, model=model)
    assert _trainable(model) == [0, 1, 2, 3, 4, 5] and cb.cache is None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    for step in (1, 2):
        ids = _rows(1, seed=step)[0]
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        cb.on_step_end(args, SimpleNamespace(global_step=step), None, optimizer=opt)
    assert _trainable(model) == [3, 4, 5]
    assert cb.cache is not None and cb.cache.cutoff == 3
    frozen = [p for n, p in model.named_parameters() if "lora_" in n and ".layers.0." in n]
    assert all(opt.state.get(p) in (None, {}) for p in frozen)
    cb.on_train_end(args, state, None)
    record = json.loads((tmp_path / "lora_layers.json").read_text())
    assert record["layers"] == [3, 4, 5]
    assert [h["step"] for h in record["history"]] == [0, 2]
    assert record["prefix_cache"][0]["cutoff"] == 3


def test_progressive_rule_freezes_stable_layers_from_the_bottom(tmp_path):
    model = _model("llama")
    cb = _callback(tmp_path, interval_steps=2, freeze_threshold=0.1, max_freeze_below=4)
    args = SimpleNamespace(device="cpu")
    cb.on_train_begin(args, SimpleNamespace(global_step=0), None, model=model)
    # layers 0, 1 steady; layer 2 grows: the cut lands at 2
    norms = {0: [1.0, 1.0, 1.0, 1.0], 1: [2.0, 2.0, 2.05, 2.05], 2: [1.0, 1.0, 3.0, 3.0],
             3: [1.0] * 4, 4: [1.0] * 4, 5: [1.0] * 4}
    for step in range(1, 5):
        for idx, params in cb.by_layer.items():
            first = True
            for _n, p in params:
                p.grad = torch.zeros_like(p)
                if first:
                    p.grad.view(-1)[0] = norms[idx][step - 1]
                    first = False
        cb.on_pre_optimizer_step(args, SimpleNamespace(global_step=step), None)
        cb.on_step_end(args, SimpleNamespace(global_step=step), None, optimizer=None)
    assert _trainable(model) == [2, 3, 4, 5]
    assert cb.selection.history[-1] == (4, [2, 3, 4, 5])


def test_progressive_rule_respects_the_cap(tmp_path):
    model = _model("llama")
    cb = _callback(tmp_path, interval_steps=1, max_freeze_below=2)
    args = SimpleNamespace(device="cpu")
    cb.on_train_begin(args, SimpleNamespace(global_step=0), None, model=model)
    for step in (1, 2):
        for params in cb.by_layer.values():
            for _n, p in params:
                p.grad = torch.ones_like(p)
        cb.on_pre_optimizer_step(args, SimpleNamespace(global_step=step), None)
        cb.on_step_end(args, SimpleNamespace(global_step=step), None, optimizer=None)
    assert _trainable(model) == [2, 3, 4, 5]


def test_static_method_installs_the_cache_at_train_begin(tmp_path):
    from soup_cli.utils import layer_select

    model = _model("qwen3_5")
    policy = layer_select.LayerSelectPolicy("fixed", layers=(2, 3), prefix_cache=True)
    cb = layer_select.LayerSelectCallback(policy, output_dir=str(tmp_path))
    cb.on_train_begin(SimpleNamespace(device="cpu"), SimpleNamespace(global_step=0), None,
                      model=model)
    assert cb.cache is not None and cb.cache.cutoff == 2


def test_progressive_with_cache_matches_without_through_the_trainer(tmp_path):
    """HF Trainer, 3 epochs, freeze after step 3: the cache changes nothing in the result."""
    transformers = pytest.importorskip("transformers")
    from soup_cli.utils import layer_select

    rows = [r[0].tolist() for r in _rows(n=3)]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return {"input_ids": rows[i]}

    def collate(items):
        ids = torch.tensor([it["input_ids"] for it in items])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}

    def run(cache, out):
        model = _model("qwen3_5")
        args = transformers.TrainingArguments(
            output_dir=str(out), per_device_train_batch_size=1, num_train_epochs=3,
            learning_rate=1e-2, logging_steps=1, save_strategy="no", report_to=[],
            use_cpu=True, seed=0, data_seed=0, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        trainer = transformers.Trainer(model=model, args=args, train_dataset=DS(),
                                       data_collator=collate)
        policy = layer_select.LayerSelectPolicy("progressive", freeze_step=3, freeze_below=2,
                                                prefix_cache=cache)
        cb = layer_select.LayerSelectCallback(policy, output_dir=str(out))
        trainer.add_callback(cb)
        trainer.train()
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        state = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
        return losses, state, cb.selection

    plain_losses, plain_state, _ = run(False, tmp_path / "a")
    losses, state, selection = run(True, tmp_path / "b")
    assert losses == plain_losses
    assert all(torch.equal(state[n], plain_state[n]) for n in state)
    stats = selection.cache[-1]
    assert stats["cutoff"] == 2 and stats["hits"] >= 3


# ----------------------------------------------------------------------- schema


def test_schema_progressive_and_cache():
    lora = {"r": 4, "target_modules": ["q_proj"], "dropout": 0.0}
    cfg = _cfg(lora=lora, lora_layers={"method": "progressive"})
    assert cfg.training.lora_layers.freeze_threshold == 0.10
    cfg = _cfg(lora=lora, lora_layers={"method": "progressive", "freeze_step": 100,
                                       "freeze_below": 8, "prefix_cache": True})
    assert cfg.training.lora_layers.prefix_cache
    assert _cfg(lora=lora, lora_layers={"method": "fixed", "layers": [8, 9],
                                        "prefix_cache": True})


@pytest.mark.parametrize("block,match", [
    ({"method": "progressive", "k": 2}, "neither k"),
    ({"method": "progressive", "freeze_step": 5}, "together"),
    ({"method": "grad", "k": 2, "freeze_below": 4}, "only apply"),
    ({"method": "lisa", "k": 2, "prefix_cache": True}, "lisa"),
    ({"method": "progressive", "freeze_threshold": 1.5}, "less than"),
])
def test_schema_refuses_bad_progressive_blocks(block, match):
    with pytest.raises(Exception, match=match):
        _cfg(lora={"r": 4, "target_modules": ["q_proj"], "dropout": 0.0}, lora_layers=block)


def test_schema_cache_needs_no_dropout_and_no_streaming():
    with pytest.raises(Exception, match="dropout"):
        _cfg(lora={"r": 4, "target_modules": ["q_proj"], "dropout": 0.05},
             lora_layers={"method": "progressive", "prefix_cache": True})
    with pytest.raises(Exception, match="stream_layers"):
        _cfg(lora={"r": 4, "target_modules": ["q_proj"], "dropout": 0.0}, stream_layers=True,
             lora_layers={"method": "progressive", "prefix_cache": True})
