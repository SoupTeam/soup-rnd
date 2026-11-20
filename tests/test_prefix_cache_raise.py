"""ZERO-2 x E2: a cache that keeps its rows when the cut rises, the rule's deadline and
smoothing, and the per-window gradient norms in lora_layers.json."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_l2l_probe_harness import _load, hermetic_ram
from tests.test_l2l_qwen35 import _store
from tests.test_layer_select import _model
from tests.test_prefix_cache import _prepare, _rows, _trainable
from tests.test_prefix_cache_stream import _adapters, _streamed

torch = pytest.importorskip("torch")

#: (epoch at whose start the cut rises, new cut); epoch 0 starts at cut 1
RAISES = {1: 2, 2: 3}


def _freeze_to(model, cut):
    from soup_cli.utils.layer_select import adapter_params_by_layer, apply_layer_mask

    by_layer = adapter_params_by_layer(model)
    apply_layer_mask(by_layer, [i for i in by_layer if i >= cut])


def _train_decoder(model, rows, cache, epochs=4):
    from soup_cli.utils.prefix_cache import PrefixCache

    _freeze_to(model, 1)
    pc = PrefixCache(model, 1, 1 << 30).install() if cache else None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses, stats = [], []
    for epoch in range(epochs):
        if epoch in RAISES:
            _freeze_to(model, RAISES[epoch])
            if pc is not None:
                stats.append(pc.stats.to_json())
                pc.raise_cutoff(RAISES[epoch])
        for ids in rows:
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach())
    if pc is not None:
        stats.append(pc.stats.to_json())
    state = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    return losses, state, stats


@pytest.mark.parametrize("kind", ["llama", "qwen3_5"])
def test_raising_the_cut_keeps_rows_and_stays_bit_identical(kind):
    rows = _rows()
    plain, plain_state, _ = _train_decoder(_prepare(kind, 0), rows, cache=False)
    cached, state, stats = _train_decoder(_prepare(kind, 0), rows, cache=True)
    assert all(torch.equal(a, b) for a, b in zip(plain, cached))
    assert all(torch.equal(state[n], plain_state[n]) for n in state)
    # epoch 0 at cut 1: 3 misses; epoch 1 at cut 2: every row resumes from level 1;
    # epoch 2 at cut 3: from level 2; epoch 3: hits
    assert (stats[0]["misses"], stats[1]["partial"], stats[2]["partial"], stats[2]["hits"]) == (
        3, 3, 3, 3)


def test_a_partial_row_runs_only_the_newly_frozen_layers():
    import types

    from soup_cli.utils.prefix_cache import PrefixCache, find_decoder

    model = _prepare("llama", 0)
    _freeze_to(model, 1)
    decoder = find_decoder(model)
    ran = []
    for idx in range(4):
        layer = decoder.layers[idx]
        real = layer.forward

        def spy(*a, _real=real, _idx=idx, **k):
            ran.append(_idx)
            return _real(*a, **k)

        layer.forward = spy
    pc = PrefixCache(model, 1, 1 << 30).install()
    assert isinstance(decoder.layers[0].forward, types.MethodType)
    ids = _rows(1)[0]
    model(input_ids=ids, labels=ids)
    _freeze_to(model, 3)
    pc.raise_cutoff(3)
    ran.clear()
    model(input_ids=ids, labels=ids)
    assert ran[:3] == [1, 2, 3]  # layer 0 comes from the cache, 1-2 run once, then the top
    ran.clear()
    model(input_ids=ids, labels=ids)
    assert ran[0] == 3  # now kept at the new cut
    with pytest.raises(ValueError, match="not above"):
        pc.raise_cutoff(2)


def _train_streamed(tmp_path, monkeypatch, name, cache, l2l):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.prefix_cache import PrefixCache

    hermetic_ram(monkeypatch, tmp_path)
    harness = _load("l2l_schedule")
    gen = torch.Generator().manual_seed(4)
    rows = [torch.randint(0, 64, (1, n), generator=gen) for n in (5, 8, 6, 7)]
    windows = [rows[0:2], rows[2:4]]
    model, runtime = _streamed(tmp_path, name)
    try:
        _freeze_to(model, 1)
        pc = PrefixCache(model, 1, 1 << 30).install() if cache else None
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        losses = []
        for epoch in range(4):
            if epoch in RAISES:
                _freeze_to(model, RAISES[epoch])
                if pc is not None:
                    pc.raise_cutoff(RAISES[epoch])
            for mbs in windows:
                harness.zero_grads(model)
                if l2l:
                    calls = [capture_layer_call(model, mb) for mb in mbs]
                    losses.extend(L2LStep(model, runtime, calls[0]).run(mbs, _store(2),
                                                                        calls=calls))
                else:
                    for ids in mbs:
                        loss = model(input_ids=ids, labels=ids).loss
                        loss.backward()
                        losses.append(loss.detach())
                opt.step()
                opt.zero_grad(set_to_none=True)
        return losses, _adapters(model), pc
    finally:
        runtime.close()


@pytest.mark.parametrize("l2l", [False, True])
def test_raising_the_cut_under_streaming_and_l2l_is_bit_identical(tmp_path, monkeypatch, l2l):
    plain, plain_state, _ = _train_streamed(tmp_path, monkeypatch, f"p{l2l}", False, l2l)
    cached, state, pc = _train_streamed(tmp_path, monkeypatch, f"c{l2l}", True, l2l)
    assert all(torch.equal(a, b) for a, b in zip(plain, cached))
    assert all(torch.equal(state[n], plain_state[n]) for n in state)
    assert pc.cutoff == 3 and all(level == 3 for level, _t in pc.store.values())


# ------------------------------------------------------------------------- the rule


def _drive(cb, model, norms, steps):
    args = SimpleNamespace(device="cpu")
    for step in range(1, steps + 1):
        for idx, params in cb.by_layer.items():
            first = True
            for _n, p in params:
                p.grad = torch.zeros_like(p)
                if first:
                    p.grad.view(-1)[0] = norms(idx, step)
                    first = False
        cb.on_pre_optimizer_step(args, SimpleNamespace(global_step=step), None)
        cb.on_step_end(args, SimpleNamespace(global_step=step), None, optimizer=None)


def _callback(tmp_path, **kw):
    from soup_cli.utils import layer_select

    cb = layer_select.LayerSelectCallback(layer_select.LayerSelectPolicy("progressive", **kw),
                                          output_dir=str(tmp_path))
    return cb


def test_deadline_freezes_to_the_minimum_when_the_rule_has_not(tmp_path):
    model = _model("llama")
    cb = _callback(tmp_path, interval_steps=1, freeze_deadline=6, min_freeze_below=3)
    cb.on_train_begin(SimpleNamespace(device="cpu"), SimpleNamespace(global_step=0), None,
                      model=model)
    _drive(cb, model, lambda idx, step: float(step * (idx + 1)), 5)  # never settles
    assert _trainable(model) == [0, 1, 2, 3, 4, 5]
    args = SimpleNamespace(device="cpu")
    cb.on_pre_optimizer_step(args, SimpleNamespace(global_step=6), None)
    cb.on_step_end(args, SimpleNamespace(global_step=6), None, optimizer=None)
    assert _trainable(model) == [3, 4, 5]
    assert cb.selection.history[-1][0] == 6


def test_deadline_leaves_a_rule_that_already_went_higher(tmp_path):
    model = _model("llama")
    cb = _callback(tmp_path, interval_steps=1, freeze_deadline=4, min_freeze_below=2,
                   max_freeze_below=4)
    cb.on_train_begin(SimpleNamespace(device="cpu"), SimpleNamespace(global_step=0), None,
                      model=model)
    _drive(cb, model, lambda idx, step: 1.0, 4)  # settled at once: up to the cap
    assert _trainable(model) == [4, 5]


def test_smoothing_rides_over_one_outlier_window(tmp_path):
    """Layer 0 is steady but window 2 is a 60 % outlier. The plain rule compares 2 with 1 and 3
    with 2 and is held back both times; the moving average (beta 0.8) moves 2 % at window 3."""
    def norms(idx, step):
        return 1.6 if (idx == 0 and step == 2) else 1.0

    plain_model, smooth_model = _model("llama"), _model("llama")
    plain = _callback(tmp_path / "a", interval_steps=1, freeze_threshold=0.10, max_freeze_below=2)
    smooth = _callback(tmp_path / "b", interval_steps=1, freeze_threshold=0.10,
                       max_freeze_below=2, freeze_smoothing=0.8)
    for cb, model in ((plain, plain_model), (smooth, smooth_model)):
        cb.on_train_begin(SimpleNamespace(device="cpu"), SimpleNamespace(global_step=0), None,
                          model=model)
        _drive(cb, model, norms, 3)
    assert min(_trainable(plain_model)) == 0
    assert min(_trainable(smooth_model)) == 2
    _drive(plain, plain_model, norms, 4)  # window 4 vs 3 is steady: the plain rule catches up
    assert min(_trainable(plain_model)) == 2


def test_norms_are_recorded_per_window(tmp_path):
    model = _model("llama")
    cb = _callback(tmp_path, freeze_step=100, freeze_below=2, interval_steps=2)
    cb.on_train_begin(SimpleNamespace(device="cpu"), SimpleNamespace(global_step=0), None,
                      model=model)
    _drive(cb, model, lambda idx, step: float(idx + 1), 4)
    cb.on_train_end(SimpleNamespace(device="cpu"), SimpleNamespace(global_step=4), None)
    record = json.loads((tmp_path / "lora_layers.json").read_text())
    assert [w["step"] for w in record["norms"]] == [2, 4]
    assert record["norms"][0]["layers"]["5"] == pytest.approx(2 * 6.0)


def test_schema_rule_fields():
    from tests.test_layer_select import _cfg

    lora = {"r": 4, "target_modules": ["q_proj"], "dropout": 0.0}
    cfg = _cfg(lora=lora, lora_layers={"method": "progressive", "freeze_deadline": 80,
                                       "min_freeze_below": 8, "freeze_smoothing": 0.8})
    assert cfg.training.lora_layers.freeze_deadline == 80
    with pytest.raises(Exception, match="together"):
        _cfg(lora=lora, lora_layers={"method": "progressive", "freeze_deadline": 80})
    with pytest.raises(Exception, match="rule"):
        _cfg(lora=lora, lora_layers={"method": "progressive", "freeze_deadline": 80,
                                     "min_freeze_below": 8, "freeze_step": 5,
                                     "freeze_below": 4})
