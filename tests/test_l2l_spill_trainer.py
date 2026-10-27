"""The activation spill through the real HF trainer loop (CPU).

Spill is forced only by the real budget path: ``layer_stream.free_ram_bytes`` is
monkeypatched and the cgroup roots point at an empty tree. The fixture is deeper than
``tests/test_l2l_probe_harness.py::build_streamed``: with ring 4 the rings are 8 buffers, so
spill pays only when ``n_layers x k x chunk > 8 x (slot + 4096)``. Eight layers of hidden 64
and a store sized to ``max_seq`` 64 give a 16 KiB float32 chunk, where it does.
"""

from __future__ import annotations

import errno
import io
import os
import time
from typing import Any, List, NamedTuple

import pytest

from tests.test_l2l_probe_harness import SEQ, _load
from tests.test_l2l_trainer_labels import _MaskedRows

N_LAYERS = 8
#: The store is sized to this many tokens; the rows are SEQ (8) tokens or up to 64.
MAX_SEQ = 64
#: Free RAM that leaves a budget of 240,000 bytes (below 1.79 GB the budget is what the 512 MiB
#: reserve leaves). A chunk is 64 x 64 x 4 = 16 KiB and the rings 8 x 20 KiB = 160 KiB, so 6
#: of 8 layers spill at k=2 (all resident: 256 KiB) and 7 at k=4 (all resident: 512 KiB).
SMALL = (512 << 20) + 240_000
PLENTY = 10**12
PLAN_KEYS = {
    "spill_layers", "n_layers", "k", "chunk_bytes", "slot_bytes", "ring", "pin", "host_bytes",
    "all_resident_bytes", "budget_bytes", "available_bytes", "limiting_level",
    "limiting_numbers", "spill_dir", "spill_path", "spill_file_bytes", "explicit_dir",
}


def build_deep(tmp_path, n_layers=N_LAYERS):
    """``build_streamed`` with more layers; the caller closes ``runtime``."""
    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_v07204 import _tiny_llama_dir

    weights, _model, _config = _tiny_llama_dir(tmp_path, n_layers=n_layers, tie=False)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type=TaskType.CAUSAL_LM,
    )
    return build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=lora, device="cpu",
        dtype="float32", buffers=2, pin=False, seed=3,
    )


@pytest.fixture()
def deep(tmp_path):
    model, runtime = build_deep(tmp_path)
    yield model, runtime
    runtime.close()


@pytest.fixture()
def ram(tmp_path, monkeypatch):
    """Sets the free RAM the budget sees; no cgroup limits, no mountinfo."""
    from soup_cli.utils import l2l_spill

    empty = tmp_path / "no-cgroup"
    empty.mkdir()
    for name in ("PROC_ROOT", "CGROUP_V2_ROOT", "CGROUP_V1_ROOT"):
        monkeypatch.setattr(l2l_spill, name, str(empty))
    probe = tmp_path / "probe"
    probe.mkdir()
    problem = l2l_spill.filesystem_refusal(os.path.realpath(probe))
    problem = problem or l2l_spill._round_trip(str(probe))
    if problem:
        pytest.skip(f"the temp directory cannot hold an unbuffered spill: {problem}")

    def set_free(nbytes):
        monkeypatch.setattr("soup_cli.utils.layer_stream.free_ram_bytes", lambda: nbytes)

    set_free(PLENTY)
    return set_free


class Run(NamedTuple):
    logs: List[dict]
    grads: List[dict]
    stores: List[Any]
    held: List[int]
    state: Any
    trainer: Any


def _held_bytes(store) -> int:
    """Host bytes the store's own allocations hold, each at its real storage size."""
    held = sum(t.untyped_storage().nbytes() for t in store._resident.values())
    if store.spill_layers:
        for ring in (store._write_ring, store._read_ring):
            deadline = time.time() + 5
            while ring._free.qsize() < 4 and time.time() < deadline:
                time.sleep(0.005)
            held += sum(buf.untyped_storage().nbytes() for buf, _ in list(ring._free.queue))
    return held


def _run(model, runtime, tmp_path, data, *, k, name, free, set_free, max_seq=MAX_SEQ,
         spill_dir=None, resume=None, callbacks=(), **overrides) -> Run:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    grads, stores, held = [], [], []
    holder = {}

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append({
                n: p.grad.detach().clone()
                for n, p in model.named_parameters() if "lora_" in n and p.grad is not None
            })

        def on_step_end(self, args, state, control, **kw):
            store = holder["state"].store
            stores.append(store)
            held.append(_held_bytes(store))

    options = dict(
        output_dir=str(tmp_path / name), per_device_train_batch_size=1,
        gradient_accumulation_steps=k, learning_rate=0.5, optim="sgd",
        lr_scheduler_type="constant", num_train_epochs=1, logging_steps=1, save_strategy="no",
        report_to=[], use_cpu=True, seed=1, data_seed=1, max_grad_norm=0.0, disable_tqdm=True,
    )
    options.update(overrides)
    trainer = make_l2l_trainer_class(Trainer)(
        model=model, args=TrainingArguments(**options), train_dataset=data,
        callbacks=[Recorder(), *callbacks],
    )
    state = attach_l2l_state(
        trainer, runtime, dtype=torch.float32, spill_dir=spill_dir, max_seq=max_seq
    )
    holder["state"] = state
    set_free(free)
    trainer.train(resume_from_checkpoint=resume)
    logs = [row for row in trainer.state.log_history if "loss" in row]
    return Run(logs, grads, stores, held, state, trainer)


def _pair(deep, tmp_path, set_free, data, k, **overrides):
    """The same rows from the same adapters, without and with spill."""
    harness = _load("l2l_schedule")
    model, runtime = deep
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    resident = _run(model, runtime, tmp_path, data, k=k, name="resident", free=PLENTY,
                    set_free=set_free, **overrides)
    harness.restore_adapters(model, start)
    spilled = _run(model, runtime, tmp_path, data, k=k, name="spilled", free=SMALL,
                   set_free=set_free, **overrides)
    return resident, spilled


def _assert_equal_runs(resident, spilled, windows):
    import torch

    assert len(resident.grads) == len(spilled.grads) == windows
    for ours, theirs in zip(spilled.grads, resident.grads):
        assert ours.keys() == theirs.keys()
        assert any(bool(g.abs().sum() > 0) for g in ours.values())
        for name in theirs:
            assert torch.equal(ours[name], theirs[name]), name
    # a resumed trainer's history starts with the checkpoint's rows: compare this run's own
    resident_logs, spilled_logs = resident.logs[-windows:], spilled.logs[-windows:]
    assert [row["loss"] for row in spilled_logs] == [row["loss"] for row in resident_logs]
    # the spill really happened, and the twin really did not
    assert all(row["l2l_spill_layers"] == 0 for row in resident_logs)
    layers = spilled.state.spill_plan["spill_layers"]
    assert 0 < layers < N_LAYERS
    assert [row["l2l_spill_layers"] for row in spilled_logs] == [layers] * windows
    stats = spilled.stores[-1].stats
    assert stats.bytes_written > 0 and stats.bytes_read > 0
    assert resident.stores[-1].stats.bytes_written == 0


def _masked(lengths):
    return _MaskedRows(lengths, [[(0, i % 3)] if i % 3 else [] for i in range(len(lengths))])


# --------------------------------------------------------------------------
# equality
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k,layers", [(2, 6), (4, 7)])
def test_spill_gives_the_losses_and_gradients_of_the_resident_store(
    deep, tmp_path, ram, k, layers
):
    resident, spilled = _pair(deep, tmp_path, ram, _masked([SEQ] * 8), k)
    _assert_equal_runs(resident, spilled, 8 // k)
    assert spilled.state.spill_plan["spill_layers"] == layers


@pytest.mark.parametrize("k", [2, 4])
def test_rows_of_different_lengths_spill_and_stay_equal(deep, tmp_path, ram, k):
    lengths = [5, 8, 3, 6, 60, 9, 4, 33]
    resident, spilled = _pair(deep, tmp_path, ram, _masked(lengths), k)
    _assert_equal_runs(resident, spilled, 8 // k)


def test_a_short_last_window_spills_and_stays_equal(deep, tmp_path, ram):
    resident, spilled = _pair(deep, tmp_path, ram, _masked([SEQ] * 7), 3)
    _assert_equal_runs(resident, spilled, 3)  # windows 3, 3, 1


def test_resume_with_spill_equals_resume_without(deep, tmp_path, ram, monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    harness = _load("l2l_schedule")
    model, runtime = deep
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    data = _masked([SEQ] * 7)
    options = dict(k=3, num_train_epochs=2, lr_scheduler_type="linear", warmup_steps=1,
                   set_free=ram)
    _run(model, runtime, tmp_path, data, name="full", free=PLENTY, save_strategy="steps",
         save_steps=1, **options)
    checkpoint = str(tmp_path / "full" / "checkpoint-3")
    runs = {}
    for name, free in (("resident", PLENTY), ("spilled", SMALL)):
        harness.restore_adapters(model, start)
        runs[name] = _run(model, runtime, tmp_path, data, name=name, free=free,
                          resume=checkpoint, **options)
    assert runs["spilled"].trainer.state.global_step == 6
    _assert_equal_runs(runs["resident"], runs["spilled"], 3)


# --------------------------------------------------------------------------
# one store and one file per run, within the plan
# --------------------------------------------------------------------------
def test_one_store_and_one_file_serve_windows_of_every_length(deep, tmp_path, ram, monkeypatch):
    from soup_cli.utils import l2l_activations, l2l_spill

    created = []

    class Spy(l2l_activations.DirectSpillFile):
        def __init__(self, path, size, **kw):
            if os.path.basename(path).startswith(l2l_spill.SPILL_FILE_PREFIX):
                created.append((path, size))
            super().__init__(path, size, **kw)

    monkeypatch.setattr(l2l_activations, "DirectSpillFile", Spy)
    buffer = io.StringIO()
    from rich.console import Console

    monkeypatch.setattr(l2l_spill, "console", Console(file=buffer, width=80))
    model, runtime = deep
    _load("l2l_schedule").make_non_vacuous(model)
    lengths = [3, 8, 60, 9, 4, 33, 5, 2, 17, 64]
    run = _run(model, runtime, tmp_path, _masked(lengths), k=2, name="spilled", free=SMALL,
               set_free=ram, num_train_epochs=2)

    plan = run.state.spill_plan
    assert PLAN_KEYS <= set(plan)
    assert plan["spill_layers"] == 6 and plan["k"] == 2 and plan["ring"] == 4
    assert plan["chunk_bytes"] == 1 * MAX_SEQ * 64 * 4
    assert plan["limiting_level"] == "psutil"
    assert plan["limiting_numbers"] == {"psutil_available": SMALL}
    assert plan["explicit_dir"] is False
    assert plan["spill_dir"] == os.path.realpath(tmp_path / "spilled" / "_l2l_spill")
    # one file, of the planned size, for ten windows of different lengths
    assert created == [(plan["spill_path"], plan["spill_file_bytes"])]
    assert len(run.stores) == 10 and len({id(store) for store in run.stores}) == 1
    # the store's real allocations are the plan's, within the budget, at every step
    assert run.held == [plan["host_bytes"]] * 10
    assert plan["host_bytes"] <= plan["budget_bytes"]
    # removed at the end, and the state no longer holds the store
    assert run.state.store is None
    assert not os.path.exists(plan["spill_path"])
    # one console line before the first step: layers, GB, file, budget, limiting level
    lines = [line for line in buffer.getvalue().splitlines() if "spilling" in line]
    assert len(lines) == 1
    line = lines[0]
    assert "6 of 8 layers" in line and " GB" in line and plan["spill_path"] in line
    assert "RAM budget" in line and "psutil available" in line


def test_activations_that_fit_change_nothing(deep, tmp_path, ram, monkeypatch):
    from soup_cli.utils import l2l_spill

    def forbidden(*a, **kw):
        raise AssertionError("the spill directory was checked")

    monkeypatch.setattr(l2l_spill, "prepare_spill_dir", forbidden)
    model, runtime = deep
    _load("l2l_schedule").make_non_vacuous(model)
    run = _run(model, runtime, tmp_path, _masked([SEQ] * 4), k=2, name="fits", free=PLENTY,
               set_free=ram)
    assert [row["l2l_spill_layers"] for row in run.logs] == [0, 0]
    assert run.state.spill_plan["spill_path"] is None
    assert not (tmp_path / "fits" / "_l2l_spill").exists()


def test_a_budget_below_every_layer_spilled_stops_before_allocating(
    deep, tmp_path, ram, monkeypatch
):
    from soup_cli.utils import l2l_activations

    def forbidden(*a, **kw):
        raise AssertionError("the store was allocated")

    monkeypatch.setattr(l2l_activations, "ActivationStore", forbidden)
    model, runtime = deep
    with pytest.raises(ValueError) as caught:
        _run(model, runtime, tmp_path, _masked([SEQ] * 4), k=2, name="short", free=1000,
             set_free=ram)
    message = str(caught.value)
    for fix in ("gradient_accumulation_steps", "data.max_length", "batch_size", "free RAM"):
        assert fix in message


# --------------------------------------------------------------------------
# cleanup
# --------------------------------------------------------------------------
def _raising_at_step(exc):
    from transformers import TrainerCallback

    seen = {}

    class Boom(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            path = seen["state"].spill_plan["spill_path"]
            seen["existed"] = os.path.exists(path)
            raise exc

    return Boom(), seen


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_an_exception_or_ctrl_c_removes_the_spill_file(deep, tmp_path, ram, monkeypatch, exc):
    from soup_cli.trainer import l2l_trainer
    from soup_cli.utils.l2l_spill import spill_files

    callback, seen = _raising_at_step(exc)
    real_attach = l2l_trainer.attach_l2l_state

    def attach(*a, **kw):
        seen["state"] = real_attach(*a, **kw)
        return seen["state"]

    monkeypatch.setattr(l2l_trainer, "attach_l2l_state", attach)
    model, runtime = deep
    with pytest.raises(type(exc)):
        _run(model, runtime, tmp_path, _masked([SEQ] * 4), k=2, name="boom", free=SMALL,
             set_free=ram, callbacks=[callback])
    assert seen["existed"] is True
    assert seen["state"].store is None
    assert spill_files(str(tmp_path / "boom" / "_l2l_spill")) == []


def test_a_full_drive_mid_step_names_size_and_directory_and_removes_the_file(
    deep, tmp_path, ram, monkeypatch
):
    from soup_cli.utils import l2l_activations
    from soup_cli.utils.l2l_spill import SPILL_FILE_PREFIX, SpillDiskFullError, spill_files

    real_write = l2l_activations.DirectSpillFile.write

    def write(self, offset, buf):
        if os.path.basename(self.path).startswith(SPILL_FILE_PREFIX):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(self, offset, buf)

    monkeypatch.setattr(l2l_activations.DirectSpillFile, "write", write)
    model, runtime = deep
    with pytest.raises(SpillDiskFullError) as caught:
        _run(model, runtime, tmp_path, _masked([SEQ] * 4), k=2, name="full", free=SMALL,
             set_free=ram)
    directory = os.path.realpath(tmp_path / "full" / "_l2l_spill")
    message = str(caught.value)
    file_bytes = 6 * 2 * MAX_SEQ * 64 * 4
    assert directory in message and f"{file_bytes:,} bytes" in message
    assert spill_files(directory) == []


def test_train_sweeps_files_of_dead_runs_first(deep, tmp_path, ram):
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    directory = tmp_path / "swept" / "_l2l_spill"
    directory.mkdir(parents=True)
    stale = directory / f"soup-l2l-spill-{proc.pid}-deadbeef.bin"
    stale.write_bytes(b"x" * 4096)
    mine = directory / f"soup-l2l-spill-{os.getpid()}-0badf00d.bin"
    mine.write_bytes(b"x")
    model, runtime = deep
    run = _run(model, runtime, tmp_path, _masked([SEQ] * 2), k=2, name="swept", free=PLENTY,
               set_free=ram)
    assert run.state.stale_removed == [os.path.realpath(stale)]
    assert not stale.exists() and mine.exists()


# --------------------------------------------------------------------------
# the store's capacity and the spill directory setting
# --------------------------------------------------------------------------
def _stepper(state, model, seq):
    import torch

    from soup_cli.trainer.l2l_trainer import _l2l_step_for

    return _l2l_step_for(state, model, (1, seq), torch.zeros((1, seq), dtype=torch.long))


def test_without_max_seq_only_a_larger_window_rebuilds_and_makes_a_second_file(
    deep, tmp_path, ram, monkeypatch
):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils import l2l_spill

    opened = []
    real_open = l2l_spill.open_spill_file
    monkeypatch.setattr(
        l2l_spill, "open_spill_file", lambda path, size: opened.append(path) or real_open(
            path, size)
    )
    # budget 400,000 bytes at k=4 (what the 512 MiB reserve leaves): the 64-token store spills
    # 5 layers, the 128-token one all 8 (its rings alone are 8 x 36 KiB)
    ram((512 << 20) + 400_000)
    model, runtime = deep
    state = L2LState(runtime=runtime, dtype=torch.float32, output_dir=str(tmp_path / "out"),
                     max_rows=1, max_k=4)
    try:
        first = _store_for(state, _stepper(state, model, 64), (1, 64), 4, "cpu")
        assert first.shape == (1, 64, 64) and first.spill_layers == 5
        assert _store_for(state, _stepper(state, model, 8), (1, 8), 2, "cpu") is first
        bigger = _store_for(state, _stepper(state, model, 128), (1, 128), 4, "cpu")
        assert bigger is not first and bigger.shape == (1, 128, 64)
        assert bigger.spill_layers == 8
        assert len(opened) == 2
        assert not os.path.exists(opened[0]) and os.path.exists(opened[1])
    finally:
        state.close()
    assert not os.path.exists(opened[1])


def test_the_trainer_max_length_sizes_the_store_when_max_seq_is_not_given(deep, ram):
    import types

    import torch

    from soup_cli.trainer.l2l_trainer import _store_for, attach_l2l_state

    model, runtime = deep
    trainer = types.SimpleNamespace(args=types.SimpleNamespace(
        max_length=100, per_device_train_batch_size=2, gradient_accumulation_steps=3,
        output_dir="out"))
    state = attach_l2l_state(trainer, runtime, dtype=torch.float32)
    try:
        store = _store_for(state, _stepper(state, model, 8), (1, 8), 1, "cpu")
        assert store.shape == (2, 128, 64) and store.k == 3
    finally:
        state.close()


def test_an_explicit_spill_dir_is_used(deep, tmp_path, ram):
    model, runtime = deep
    _load("l2l_schedule").make_non_vacuous(model)
    target = tmp_path / "elsewhere" / "spill"
    run = _run(model, runtime, tmp_path, _masked([SEQ] * 4), k=2, name="explicit", free=SMALL,
               set_free=ram, spill_dir=str(target))
    plan = run.state.spill_plan
    assert plan["explicit_dir"] is True
    assert plan["spill_dir"] == os.path.realpath(target)
    assert os.path.dirname(plan["spill_path"]) == os.path.realpath(target)
    assert not (tmp_path / "explicit" / "_l2l_spill").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_the_disk_tier_keeps_the_default_dir_off_the_weights_disk(
    deep, tmp_path, ram, monkeypatch, explicit
):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(l2l_spill, "physical_disks", lambda path: (frozenset({"sda"}), "sda"))
    lines = []
    monkeypatch.setattr(l2l_spill, "say", lambda line, notify=None: lines.append(line))
    ram(SMALL)
    model, runtime = deep
    monkeypatch.setattr(runtime, "tier", "disk")
    state = L2LState(
        runtime=runtime, dtype=torch.float32, output_dir=str(tmp_path / "out"), max_rows=1,
        max_k=2, spill_dir=str(tmp_path / "mine") if explicit else None,
        weight_dirs=(str(tmp_path / "shards"),),
    )
    try:
        if not explicit:
            with pytest.raises(l2l_spill.SpillRefusedError, match="training.l2l_spill_dir"):
                _store_for(state, _stepper(state, model, 64), (1, 64), 2, "cpu")
            return
        store = _store_for(state, _stepper(state, model, 64), (1, 64), 2, "cpu")
        assert store.spill_layers > 0
        assert any("13% slower" in line for line in lines)
    finally:
        state.close()
