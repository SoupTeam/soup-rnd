"""ZERO-2 x E2: the frozen-prefix cache on a streamed model, with and without L2L.

On a hit the bottom layers are neither loaded nor computed: the streamed step primes its first
load at the cache's cutoff, the L2L window starts there. Cached and uncached training must agree
bit for bit on the hybrid qwen3_5 decoder.
"""
from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, hermetic_ram
from tests.test_l2l_qwen35 import _store, _tiny_qwen35_dir

CUTOFF = 2
EPOCHS = 3


def _streamed(tmp_path, name):
    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_select import (
        adapter_params_by_layer,
        apply_layer_mask,
        cut_input_grads,
    )
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    root = tmp_path / name
    weights, _config = _tiny_qwen35_dir(root)
    index = shard_checkpoint(weights, str(root / "shards"), dtype="float32", arch="qwen3")
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=str(root / "shards"), index=index, lora_config=lora,
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
    )
    _load("l2l_schedule").make_non_vacuous(model)
    by_layer = adapter_params_by_layer(model)
    apply_layer_mask(by_layer, [i for i in by_layer if i >= CUTOFF])
    cut_input_grads(model)
    model.train()
    return model, runtime


def _rows(n=3, seed=5):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 64, (1, SEQ), generator=gen) for _ in range(n)]


def _adapters(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _record_loads(runtime):
    """Every layer index the pool loads, in order."""
    pool = runtime.pool
    loads = []
    original = pool.load_async

    def spy(idx, *a, **k):
        loads.append(idx)
        return original(idx, *a, **k)

    pool.load_async = spy
    return loads


def _train_streamed(model, runtime, rows, cache):
    import torch

    from soup_cli.utils.prefix_cache import PrefixCache

    pc = PrefixCache(model, CUTOFF, 1 << 30).install() if cache else None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loads = _record_loads(runtime)
    losses, per_epoch_loads = [], []
    for _ in range(EPOCHS):
        start = len(loads)
        for ids in rows:
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach())
        per_epoch_loads.append(loads[start:])
    return losses, _adapters(model), per_epoch_loads, pc


def test_streamed_training_with_the_cache_is_bit_identical_and_skips_the_bottom(
        tmp_path, monkeypatch):
    import torch

    hermetic_ram(monkeypatch, tmp_path)
    rows = _rows()
    model, runtime = _streamed(tmp_path, "a")
    try:
        plain, plain_state, plain_loads, _ = _train_streamed(model, runtime, rows, cache=False)
    finally:
        runtime.close()
    model, runtime = _streamed(tmp_path, "b")
    try:
        cached, state, loads, pc = _train_streamed(model, runtime, rows, cache=True)
    finally:
        runtime.close()
    assert all(torch.equal(a, b) for a, b in zip(plain, cached))
    assert state.keys() == plain_state.keys()
    assert all(torch.equal(state[n], plain_state[n]) for n in state)
    assert (pc.stats.misses, pc.stats.hits) == (3, 6)
    # Epoch 1 computes the bottom; epochs 2 and 3 never load a layer below the cutoff.
    assert any(i < CUTOFF for i in loads[0])
    for epoch in (1, 2):
        assert loads[epoch] and all(i >= CUTOFF for i in loads[epoch]), loads[epoch]
        assert len(loads[epoch]) < len(plain_loads[epoch])


def test_streamed_eval_runs_the_whole_model(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.prefix_cache import PrefixCache

    hermetic_ram(monkeypatch, tmp_path)
    model, runtime = _streamed(tmp_path, "c")
    try:
        PrefixCache(model, CUTOFF, 1 << 30).install()
        ids = _rows(1)[0]
        model(input_ids=ids, labels=ids).loss.backward()  # stores the row
        loads = _record_loads(runtime)
        model.eval()
        with torch.no_grad():
            model(input_ids=ids, labels=ids)
        assert 0 in loads
        assert runtime.prefetcher.prime_start == 0
    finally:
        runtime.close()


# --------------------------------------------------------------------------------- L2L


def _train_l2l(model, runtime, windows, cache):
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.prefix_cache import PrefixCache

    harness = _load("l2l_schedule")
    pc = PrefixCache(model, CUTOFF, 1 << 30).install() if cache else None
    step = L2LStep(model, runtime, capture_layer_call(model, windows[0][0]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses, grads, loads = [], [], []
    for _ in range(EPOCHS):
        for mbs in windows:
            harness.zero_grads(model)
            losses.extend(step.run(mbs, _store(len(mbs))))
            grads.append({n: p.grad.clone() for n, p in model.named_parameters()
                          if p.grad is not None})
            loads.append(step.counters.layer_loads)
            opt.step()
            opt.zero_grad(set_to_none=True)
    return losses, grads, loads, _adapters(model), pc


def test_l2l_with_the_cache_is_bit_identical_and_loads_less(tmp_path, monkeypatch):
    import torch

    hermetic_ram(monkeypatch, tmp_path)
    rows = _rows(4)
    windows = [rows[0:2], rows[2:4]]
    model, runtime = _streamed(tmp_path, "d")
    try:
        p_losses, p_grads, p_loads, p_state, _ = _train_l2l(model, runtime, windows, False)
    finally:
        runtime.close()
    model, runtime = _streamed(tmp_path, "e")
    try:
        losses, grads, loads, state, pc = _train_l2l(model, runtime, windows, True)
    finally:
        runtime.close()
    assert all(torch.equal(a, b) for a, b in zip(p_losses, losses))
    for a, b in zip(p_grads, grads):
        assert a.keys() == b.keys() and a
        assert all(torch.equal(a[n], b[n]) for n in a)
    assert all(torch.equal(state[n], p_state[n]) for n in state)
    assert (pc.stats.l2l_misses, pc.stats.l2l_hits) == (2, 4)
    assert pc.stats.hits == 0  # only capture_layer_call's forward went through the decoder
    # epoch 1 computes the bottom (the backward no longer prefetches the frozen layer under
    # the cutoff, so it may load one layer less); epochs 2 and 3 skip the bottom's loads
    assert all(c <= p for c, p in zip(loads[:2], p_loads[:2]))
    assert all(c < p for c, p in zip(loads[2:], p_loads[2:])), (loads, p_loads)


def test_l2l_window_with_one_new_row_computes_the_bottom(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.prefix_cache import PrefixCache

    hermetic_ram(monkeypatch, tmp_path)
    model, runtime = _streamed(tmp_path, "f")
    try:
        pc = PrefixCache(model, CUTOFF, 1 << 30).install()
        a, b, c = _rows(3)
        step = L2LStep(model, runtime, capture_layer_call(model, a))
        step.run([a, b], _store(2))
        step.run([a, c], _store(2))  # c is new: computed, then kept
        step.run([c, b], _store(2))  # both kept now
        assert (pc.stats.l2l_misses, pc.stats.l2l_hits) == (2, 1)
        assert pc.stats.rows_stored == 3
    finally:
        runtime.close()


# ------------------------------------------------------------- through the HF Trainer


@pytest.mark.parametrize("l2l", [False, True])
def test_progressive_freeze_with_cache_through_the_trainer(tmp_path, monkeypatch, l2l):
    """Streamed qwen3_5, HF Trainer (plain or L2L), 3 epochs, progressive freeze after step 2:
    the cache changes neither a logged loss nor an adapter bit, and serves later epochs."""
    import torch
    from transformers import Trainer, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class
    from soup_cli.utils import layer_select
    from tests.test_l2l_trainer_mode import _Rows

    hermetic_ram(monkeypatch, tmp_path)

    def run(cache, name):
        model, runtime = _streamed(tmp_path, name)
        # undo _streamed's fixed mask: progressive starts with every layer trainable
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        trainer = None
        try:
            args = TrainingArguments(
                output_dir=str(tmp_path / f"out-{name}"), per_device_train_batch_size=1,
                gradient_accumulation_steps=2, learning_rate=0.01, optim="adamw_torch",
                lr_scheduler_type="constant", num_train_epochs=EPOCHS, save_strategy="no",
                report_to=[], use_cpu=True, seed=1, data_seed=1, max_grad_norm=0,
                disable_tqdm=True, logging_steps=1,
            )
            cls = make_l2l_trainer_class(Trainer) if l2l else Trainer
            trainer = cls(model=model, args=args, train_dataset=_Rows(4))
            if l2l:
                attach_l2l_state(trainer, runtime, dtype=torch.float32)
            policy = layer_select.LayerSelectPolicy(
                "progressive", freeze_step=2, freeze_below=CUTOFF, prefix_cache=cache)
            cb = layer_select.LayerSelectCallback(policy, output_dir=str(tmp_path / name))
            trainer.add_callback(cb)
            trainer.train()
            losses = [r["loss"] for r in trainer.state.log_history if "loss" in r]
            state = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
            return losses, state, cb.selection
        finally:
            if trainer is not None and l2l:
                trainer._soup_l2l.close()
            runtime.close()

    plain_losses, plain_state, _ = run(False, f"plain-{l2l}")
    losses, state, selection = run(True, f"cache-{l2l}")
    assert len(losses) == 2 * EPOCHS and losses == plain_losses
    assert all(torch.equal(state[n], plain_state[n]) for n in state)
    stats = selection.cache[-1]
    assert stats["cutoff"] == CUTOFF
    served = stats["l2l_hits"] if l2l else stats["hits"]
    assert served >= 2, stats


def test_l2l_captures_of_other_rows_never_corrupt_the_cache(tmp_path, monkeypatch):
    """The L2L trainer captures layer calls per micro-batch shape: a decoder forward in training
    mode that capture_layer_call stops early with an exception. That forward must not leave the
    cache armed with its row's keys (the Kaggle failure: a later window stored another row's
    output under the captured row's key, then a hit returned a tensor of the wrong length)."""
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.prefix_cache import PrefixCache

    hermetic_ram(monkeypatch, tmp_path)
    gen = torch.Generator().manual_seed(9)
    rows = [torch.randint(0, 64, (1, n), generator=gen) for n in (5, 8, 6, 7)]
    windows = [rows[0:2], rows[2:4]]
    probe = torch.randint(0, 64, (1, 4), generator=gen)  # a row no window holds

    def train(cache, name):
        harness = _load("l2l_schedule")
        model, runtime = _streamed(tmp_path, name)
        try:
            pc = PrefixCache(model, CUTOFF, 1 << 30).install() if cache else None
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
            losses = []
            for _ in range(EPOCHS):
                for mbs in windows:
                    capture_layer_call(model, probe)  # an interrupted training-mode forward
                    calls = [capture_layer_call(model, mb) for mb in mbs]
                    step = L2LStep(model, runtime, calls[0])
                    harness.zero_grads(model)
                    losses.extend(step.run(mbs, _store(2), calls=calls))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            return losses, _adapters(model), pc
        finally:
            runtime.close()

    plain, plain_state, _ = train(False, "g")
    cached, state, pc = train(True, "h")
    assert all(torch.equal(a, b) for a, b in zip(plain, cached))
    assert all(torch.equal(state[n], plain_state[n]) for n in state)
    # The captures run the bottom too, so they keep every window row (and the probe) before the
    # first window: then every window, from epoch 1 on, reads the cache.
    assert pc.stats.rows_stored == 5
    assert (pc.stats.l2l_hits, pc.stats.l2l_misses) == (2 * EPOCHS, 0)
