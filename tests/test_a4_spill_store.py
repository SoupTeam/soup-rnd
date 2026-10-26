"""CPU tests for A4's budget-obeying activation store (benchmarks/harness/a4_spill_store.py).

What a GPU run cannot show by itself: the store holds exactly the planned bytes;
a full spill volume ends in a named error with the arithmetic and no file left
behind; an exception or a Ctrl-C in the middle of a step releases the file, the
threads and the buffers; and L2L through a real spill file is bit-exact to the
all-pinned store and to gradient accumulation, while a corrupted spill read is
caught by the same comparison.
"""

from __future__ import annotations

import errno
import os
import signal
import sys
import threading

import pytest

from tests.test_l2l_probe_harness import _load

SEQ = 16  # 16 x 64 x float32 = 4096 B: one sector per chunk, so the planner allows a spill
HIDDEN = 64
N_LAYERS = 3


@pytest.fixture()
def bud():
    return _load("a4_activation_budget")


@pytest.fixture()
def spill():
    return _load("a4_spill_store")


class _Releases:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return 0


def _plan(bud, budget_bytes, *, k=3, n_layers=N_LAYERS, ring=1):
    return bud.plan_activations(budget_bytes, n_layers=n_layers, k=k,
                                chunk_bytes=SEQ * HIDDEN * 4, ring=ring, pin=False)


def _store(spill, plan, tmp_path, **over):
    import torch

    kwargs = dict(chunk_shape=(1, SEQ, HIDDEN), dtype=torch.float32, device="cpu",
                  spill_dir=str(tmp_path / "spill"), backend="buffered", release=_Releases())
    kwargs.update(over)
    return spill.BudgetedStore(plan, **kwargs)


def _shipped_acts():
    """The module ``ActivationStore`` really runs in. The harness ``l2l_activations`` only
    re-exports it, so a patch on the harness copy never reaches the store's own calls."""
    import importlib

    return importlib.import_module("soup_cli.utils.l2l_activations")


def _chunk(seed):
    import torch

    return torch.randn((1, SEQ, HIDDEN), generator=torch.Generator().manual_seed(seed))


def _no_spill_threads():
    return not any(t.name.startswith("l2l-spill-") for t in threading.enumerate())


def _open_spill_handles(spill, spill_dir):
    """Spill files this process holds open. On Linux the name is unlinked right after the
    file is opened (``open_spill``), so a directory listing cannot see an open spill file;
    its handles in /proc/self/fd can. On Windows the name lives exactly while it is open.
    None where neither is observable (other POSIX systems)."""
    if sys.platform.startswith("linux"):
        prefix = os.path.realpath(str(spill_dir)) + os.sep
        found = []
        for fd in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                continue
            if target.startswith(prefix) and "a4-spill-" in target:
                found.append(target)
        return found
    if os.name == "nt":
        return spill.leftover_spill_files(str(spill_dir))
    return None


def _released(spill, path):
    """No name and no open handle left for a spill file."""
    assert not os.path.exists(path)
    handles = _open_spill_handles(spill, os.path.dirname(path))
    assert not handles, handles


# ---------------------------------------------------------------------------
# the store obeys the plan, and the cap check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("budget,spilled", [(16384, 3), (28672, 2), (36864, 0)])
def test_the_store_holds_exactly_the_planned_bytes(bud, spill, tmp_path, budget, spilled):
    plan = _plan(bud, budget)
    assert plan.spill_layers == spilled
    with _store(spill, plan, tmp_path) as store:
        assert store.cap["ok"], store.cap
        assert store.held_bytes == plan.host_bytes <= budget
        assert store.stats.spill_layers == spilled
        assert store.stats.resident_chunks == (N_LAYERS - spilled) * 3
        if spilled:
            assert store.spill.open_file_bytes() == plan.spill_file_bytes
            assert _open_spill_handles(spill, tmp_path / "spill") != []
    assert store.closed


def test_round_trip_through_the_spill_is_lossless(bud, spill, tmp_path):
    import torch

    with _store(spill, _plan(bud, 16384), tmp_path) as store:
        for step in range(2):
            for layer in range(N_LAYERS):
                for mb in range(3):
                    store.put(layer, mb, _chunk(step * 1000 + layer * 10 + mb))
            store.begin_backward()
            for layer in reversed(range(N_LAYERS)):
                for mb in range(3):
                    assert torch.equal(store.get(layer, mb), _chunk(step * 1000 + layer * 10 + mb))
            store.end_step()
        io = store.io()
    assert io["bytes_written"] == io["bytes_read"] == 2 * 9 * 4096
    assert io["write_s"] > 0 and io["read_s"] > 0


def test_the_cap_check_catches_a_store_that_allocates_more_than_planned(bud, spill, tmp_path,
                                                                       monkeypatch):
    """If the staging buffers ever grow (a second sector of slack, say), the plan's bytes
    are no longer the real ones: refuse, and leave no file behind."""
    acts = _shipped_acts()
    real = acts.aligned_buffer

    def wider(nbytes, *, pin):
        return real(nbytes + acts.SECTOR_BYTES, pin=pin)[:nbytes]

    monkeypatch.setattr(acts, "aligned_buffer", wider)
    monkeypatch.setattr(spill, "l2l_activations", acts)
    releases = _Releases()
    with pytest.raises(spill.PlanMismatchError,
                       match="the store holds 24,576 B, the plan says 16,384 B"):
        _store(spill, _plan(bud, 16384), tmp_path, release=releases)
    assert spill.leftover_spill_files(str(tmp_path / "spill")) == []
    assert releases.calls == 1
    assert _no_spill_threads()


def test_a_chunk_of_another_size_is_refused(bud, spill, tmp_path):
    import torch

    with pytest.raises(spill.PlanMismatchError, match="2,048 B, but the plan was made for 4,096 B"):
        _store(spill, _plan(bud, 16384), tmp_path, dtype=torch.float16)


def test_a_spilling_plan_needs_a_directory(bud, spill, tmp_path):
    with pytest.raises(ValueError, match="spill_dir"):
        _store(spill, _plan(bud, 16384), tmp_path, spill_dir=None)


# ---------------------------------------------------------------------------
# a full disk, errors, Ctrl-C: named, and nothing left behind
# ---------------------------------------------------------------------------
def _full_after(acts, good_writes):
    class FullDisk(acts.BufferedSpillFile):
        writes = 0

        def write(self, offset, buf):
            if FullDisk.writes >= good_writes:
                raise OSError(errno.ENOSPC, "No space left on device")
            FullDisk.writes += 1
            super().write(offset, buf)

    return FullDisk


def test_enospc_mid_step_is_named_with_the_arithmetic_and_cleans_up(bud, spill, tmp_path):
    acts = _load("l2l_activations")
    full = _full_after(acts, good_writes=2)
    releases = _Releases()
    store = _store(spill, _plan(bud, 16384), tmp_path, release=releases,
                   opener=lambda kind, path, size: full(path, size), free_bytes_at_plan=10 ** 9)
    path = store.path
    with pytest.raises(spill.SpillDiskFullError) as info:
        with store:
            for layer in range(N_LAYERS):
                for mb in range(3):
                    store.put(layer, mb, _chunk(layer * 10 + mb))
            store.begin_backward()
            for layer in reversed(range(N_LAYERS)):
                for mb in range(3):
                    store.get(layer, mb)
    text = str(info.value)
    assert text.startswith("SPILL_ENOSPC")
    assert "3 layers x k=3 x 4,096 B = 36,864 B" in text
    assert "8,192 B (0.00 GiB) were written before the drive refused" in text
    assert "at offset 8,192 (4,096 B write)" in text
    assert "1,000,000,000 B" in text
    assert "the spill file was removed" in text
    _released(spill, path)
    assert store.closed and releases.calls >= 1
    assert _no_spill_threads()


def test_enospc_while_creating_the_file_is_named_too(bud, spill, tmp_path):
    def no_room(kind, path, size):
        raise OSError(errno.ENOSPC, "No space left on device", path)

    with pytest.raises(spill.SpillDiskFullError, match="while creating the spill file"):
        _store(spill, _plan(bud, 16384), tmp_path, opener=no_room)
    assert spill.leftover_spill_files(str(tmp_path / "spill")) == []


def test_a_spill_file_that_cannot_be_removed_is_reported_by_path(bud, spill, tmp_path,
                                                                 monkeypatch):
    acts = _load("l2l_activations")
    full = _full_after(acts, good_writes=0)
    store = _store(spill, _plan(bud, 16384), tmp_path,
                   opener=lambda kind, path, size: full(path, size))

    def held(path):
        raise PermissionError(13, "in use by another process", path)

    monkeypatch.setattr(acts.os, "remove", held)
    with pytest.raises(spill.SpillDiskFullError) as info:
        for layer in range(N_LAYERS):
            for mb in range(3):
                store.put(layer, mb, _chunk(mb))
    monkeypatch.undo()
    assert f"could NOT be removed and is left at {store.path}" in str(info.value)
    assert store.left_behind == store.path
    os.remove(store.path)


def test_a_failure_after_the_file_is_opened_removes_it(bud, spill, tmp_path, monkeypatch):
    """ActivationStore opens the spill file, then pins its rings. A ring that fails to pin
    (#901's page-lock refusal) must not leave the file behind."""
    acts = _shipped_acts()

    def refuse(nbytes, *, pin):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(acts, "aligned_buffer", refuse)
    monkeypatch.setattr(spill, "l2l_activations", acts)
    releases = _Releases()
    with pytest.raises(RuntimeError, match="out of memory"):
        _store(spill, _plan(bud, 16384), tmp_path, release=releases)
    assert spill.leftover_spill_files(str(tmp_path / "spill")) == []
    assert releases.calls == 1


def test_an_exception_mid_step_releases_everything(bud, spill, tmp_path):
    releases = _Releases()
    store = _store(spill, _plan(bud, 28672), tmp_path, release=releases)
    path = store.path
    with pytest.raises(ValueError, match="boom"):
        with store:
            store.put(0, 0, _chunk(1))
            raise ValueError("boom")
    _released(spill, path)
    assert store.closed and releases.calls == 1
    assert store.stats.spill_left_behind is None
    assert _no_spill_threads()
    store.close()  # idempotent
    assert releases.calls == 1


def test_ctrl_c_in_a_step_releases_the_file_and_the_threads(bud, spill, tmp_path):
    """A real SIGINT, raised from inside put() the way Ctrl-C lands in the main thread."""

    class Interrupted(spill.BudgetedStore):
        def put(self, layer, mb, tensor):
            if (layer, mb) == (1, 1):
                signal.raise_signal(signal.SIGINT)
            super().put(layer, mb, tensor)

    import torch

    releases = _Releases()
    store = Interrupted(_plan(bud, 16384), chunk_shape=(1, SEQ, HIDDEN), dtype=torch.float32,
                        device="cpu", spill_dir=str(tmp_path / "spill"), backend="buffered",
                        release=releases)
    path = store.path
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            with store:
                for layer in range(N_LAYERS):
                    for mb in range(3):
                        store.put(layer, mb, _chunk(mb))
    finally:
        signal.signal(signal.SIGINT, previous)
    _released(spill, path)
    assert store.closed and releases.calls == 1
    assert _no_spill_threads()


class _ReleaseProbe:
    """A ``release`` that records how many of the store's buffers are still reachable when
    it runs. The cache release can only return a page-locked block nothing references."""

    def __init__(self):
        self.calls = 0
        self.refs = []
        self.alive = []

    def __call__(self):
        import gc

        gc.collect()
        self.calls += 1
        self.alive.append(sum(1 for alive in self.refs if alive()))
        return 0


def _storage_ref(tensor):
    from torch.multiprocessing.reductions import StorageWeakRef

    ref = StorageWeakRef(tensor.untyped_storage())
    return lambda: not ref.expired()


def _watch(probe, store):
    inner = store._store
    buffers = list(inner._resident.values())
    if inner.spill_layers:
        for ring in (inner._write_ring, inner._read_ring):
            buffers += [buf for buf, _event in list(ring._free.queue)]
    probe.refs = [_storage_ref(buf) for buf in buffers]
    return len(buffers)


def _one_step(store):
    for layer in range(N_LAYERS):
        for mb in range(3):
            store.put(layer, mb, _chunk(layer * 10 + mb))
    store.begin_backward()
    for layer in reversed(range(N_LAYERS)):
        for mb in range(3):
            store.get(layer, mb)
    store.end_step()


def _raise_in_the_body(store):
    store.put(0, 0, _chunk(1))
    raise ValueError("boom")


def _interrupt_inside_put(store):
    """Ctrl-C lands inside ActivationStore.put after it took a staging buffer."""
    def typed(buf):
        raise KeyboardInterrupt

    store._store._typed = typed
    store.put(0, 0, _chunk(1))


def _interrupt_inside_get(store):
    """Ctrl-C lands inside ActivationStore.get after it popped a read buffer."""
    for layer in range(N_LAYERS):
        for mb in range(3):
            store.put(layer, mb, _chunk(layer * 10 + mb))
    store.begin_backward()

    def typed(buf):
        raise KeyboardInterrupt

    store._store._typed = typed
    store.get(N_LAYERS - 1, 0)


def _failing_writes(acts, err):
    class Failing(acts.BufferedSpillFile):
        def write(self, offset, buf):
            raise err

    return lambda kind, path, size: Failing(path, size)


@pytest.mark.parametrize("scenario,raised", [
    ("normal", None), ("exception", ValueError), ("ctrl_c_in_put", KeyboardInterrupt),
    ("ctrl_c_in_get", KeyboardInterrupt), ("enospc_writer", "SpillDiskFullError"),
    ("eio_writer", RuntimeError),
])
def test_no_store_buffer_is_reachable_when_the_pinned_cache_is_released(bud, spill, tmp_path,
                                                                        scenario, raised):
    """Before the fix, a Ctrl-C inside put/get and a writer failing with anything but ENOSPC
    kept one staging buffer alive in a traceback frame past close(), so its page-locked
    block could not go back to the OS."""
    acts = _load("l2l_activations")
    openers = {"enospc_writer": _failing_writes(acts, OSError(errno.ENOSPC, "No space left")),
               "eio_writer": _failing_writes(acts, OSError(errno.EIO, "I/O error"))}
    bodies = {"exception": _raise_in_the_body, "ctrl_c_in_put": _interrupt_inside_put,
              "ctrl_c_in_get": _interrupt_inside_get}
    probe = _ReleaseProbe()
    extra = {"opener": openers[scenario]} if scenario in openers else {}
    store = _store(spill, _plan(bud, 16384), tmp_path, release=probe, **extra)
    assert _watch(probe, store) == 2  # ring 1: one write and one read staging buffer
    path = store.path
    expected = getattr(spill, raised) if isinstance(raised, str) else raised
    if expected is None:
        with store:
            _one_step(store)
    else:
        with pytest.raises(expected):
            with store:
                bodies.get(scenario, _one_step)(store)
    assert probe.calls >= 1 and probe.alive[0] == 0, probe.alive
    _released(spill, path)
    assert _no_spill_threads()


def test_a_ring_that_fails_to_pin_after_the_file_is_open_leaves_nothing(bud, spill, tmp_path,
                                                                       monkeypatch):
    """Production-harness finding 3, injected deterministically: ActivationStore.__init__
    pins the resident chunks, opens the spill file (l2l_activations.py:318), then pins the
    rings (:320-321). The second ring buffer fails, as #901's page-lock refusal would.
    The wrapper must remove the open file, start no thread, and leave neither the
    resident chunks nor the first ring buffer reachable when it releases the cache."""
    acts = _shipped_acts()
    real_buffer, real_store = acts.aligned_buffer, acts.ActivationStore
    made, seen_files, instances = [], [], []

    def second_fails(nbytes, *, pin):
        seen_files.extend(_open_spill_handles(spill, tmp_path / "spill") or [])
        if made:
            raise RuntimeError("CUDA error: out of memory (injected page-lock failure)")
        made.append(real_buffer(nbytes, pin=pin))
        return made[-1]

    class Recorded(real_store):
        def __init__(self, **kwargs):
            import weakref

            instances.append(weakref.ref(self))
            super().__init__(**kwargs)

    monkeypatch.setattr(acts, "aligned_buffer", second_fails)
    monkeypatch.setattr(acts, "ActivationStore", Recorded)
    monkeypatch.setattr(spill, "l2l_activations", acts)
    probe = _ReleaseProbe()

    def watch_then_release():
        probe.refs = [_storage_ref(made[0]), lambda: instances[0]() is not None]
        made.clear()
        return probe()

    plan = _plan(bud, 28672)  # 2 layers spilled, 1 resident layer of 3 chunks
    assert plan.spill_layers == 2
    with pytest.raises(RuntimeError, match="injected page-lock failure"):
        _store(spill, plan, tmp_path, release=watch_then_release)
    assert seen_files, "the spill file was not open yet when the ring was pinned"
    assert not _open_spill_handles(spill, tmp_path / "spill")
    assert probe.calls == 1 and probe.alive == [0]
    assert _no_spill_threads()


def test_an_opener_that_fails_after_creating_the_file_does_not_leave_it(bud, spill, tmp_path):
    """Windows: ``truncate()`` extends a file by WRITING zeros (measured on this NTFS
    volume: a 64 MiB truncate wrote 67,108,864 B and took 64 MiB of free space), before
    DirectSpillFile's own cleanup guard starts. A full volume then raises out of the
    constructor with the partly written file on disk."""
    def full_while_extending(kind, path, size):
        with open(path, "wb") as handle:
            handle.write(b"\0" * 8192)
        raise OSError(errno.ENOSPC, "There is not enough space on the disk", path)

    with pytest.raises(spill.SpillDiskFullError, match="the spill file was removed"):
        _store(spill, _plan(bud, 16384), tmp_path, opener=full_while_extending)
    assert spill.leftover_spill_files(str(tmp_path / "spill")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX unlinks the open spill file")
def test_posix_spill_file_has_no_name_while_open(spill, tmp_path):
    made = spill.open_spill("buffered", str(tmp_path / "x.bin"), 8192)
    try:
        assert not os.path.exists(tmp_path / "x.bin")
        assert spill.open_file_bytes(made) == 8192  # the file lives through its handle
    finally:
        made.close()
    assert spill.open_file_bytes(made) is None


def test_disk_full_is_found_through_the_cause_chain(spill):
    inner = OSError(errno.ENOSPC, "No space left on device")
    try:
        try:
            raise inner
        except OSError as exc:
            raise RuntimeError("activation spill failed") from exc
    except RuntimeError as outer:
        assert spill.is_disk_full(outer) is inner
    assert spill.is_disk_full(RuntimeError("other")) is None
    assert spill.is_disk_full(OSError(errno.EACCES, "denied")) is None


# ---------------------------------------------------------------------------
# bit-exactness through L2LStep on a tiny streamed model
# ---------------------------------------------------------------------------
@pytest.fixture()
def sched():
    return _load("l2l_schedule")


@pytest.fixture()
def tiny(tmp_path):
    """The tiny untied 3-layer Llama of test_l2l_probe_harness, streamed on CPU."""
    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_v07204 import _tiny_llama_dir

    weights, _model, _config = _tiny_llama_dir(tmp_path, n_layers=N_LAYERS, tie=False,
                                               hidden=HIDDEN)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type=TaskType.CAUSAL_LM)
    model, runtime = build_streamed_model(model_id=weights, shard_dir=shards, index=index,
                                          lora_config=lora, device="cpu", dtype="float32",
                                          buffers=2, pin=False, seed=3)
    yield model, runtime
    runtime.close()


def _mbs(k, seed=17):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 64, (1, SEQ), generator=gen) for _ in range(k)]


def _l2l(sched, step, model, mbs, store):
    sched.zero_grads(model)
    with store:
        losses = step.run(mbs, store)
    return losses, sched.collect_grads(model)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_spill_at_every_budget_is_bit_exact_to_all_pinned_and_to_ga(bud, spill, sched, tiny,
                                                                   tmp_path, k):
    model, runtime = tiny
    sched.make_non_vacuous(model)
    mbs = _mbs(k)
    sched.zero_grads(model)
    ga_losses = sched.ga_step(model, mbs)
    ga = sched.collect_grads(model)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))

    layout = bud.StoreLayout(n_layers=N_LAYERS, k=k, chunk_bytes=SEQ * HIDDEN * 4, ring=1,
                             pin=False)
    pinned_losses, pinned = _l2l(sched, step, model, mbs,
                                 _store(spill, _plan(bud, layout.all_pinned_bytes, k=k), tmp_path))
    assert sched.compare(ga, pinned)["equal"]
    assert sched.losses_equal(ga_losses, pinned_losses)
    # Every budget that spills a different number of layers: the minimum, then one more
    # resident layer at a time, up to all-pinned.
    per_layer = k * SEQ * HIDDEN * 4
    depths = []
    for resident in range(N_LAYERS):
        budget = layout.minimum_bytes + resident * per_layer
        if budget >= layout.all_pinned_bytes:
            break
        plan = _plan(bud, budget, k=k)
        depths.append(plan.spill_layers)
        losses, grads = _l2l(sched, step, model, mbs, _store(spill, plan, tmp_path))
        backward = sched.compare(pinned, grads)
        assert backward["all_zero"] == [] and backward["equal"], (budget, backward)
        assert sched.losses_equal(pinned_losses, losses), budget  # the forward
    # k=1: the two rings cost more than all three layers, so no budget spills.
    assert depths == {1: [], 2: [3], 3: [3, 2]}[k]
    assert spill.leftover_spill_files(str(tmp_path / "spill")) == []


def test_a_corrupted_spill_read_is_caught_by_the_same_comparison(bud, spill, sched, tiny,
                                                                 tmp_path):
    """The control expected to differ, in the same process: one bit flipped in one spilled
    chunk on its way back. The forward (losses) is unaffected, because the loss is computed
    from the exact activations; the backward (grads) must differ."""
    acts = _load("l2l_activations")

    class Corrupt(acts.BufferedSpillFile):
        def read(self, offset, buf):
            super().read(offset, buf)
            if offset == 0:
                buf[0] ^= 1

    model, runtime = tiny
    sched.make_non_vacuous(model)
    mbs = _mbs(3)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    layout = bud.StoreLayout(n_layers=N_LAYERS, k=3, chunk_bytes=SEQ * HIDDEN * 4, ring=1,
                             pin=False)
    pinned_losses, pinned = _l2l(sched, step, model, mbs,
                                 _store(spill, _plan(bud, layout.all_pinned_bytes), tmp_path))
    plan = _plan(bud, layout.minimum_bytes)
    clean_losses, clean = _l2l(sched, step, model, mbs, _store(spill, plan, tmp_path))
    bad_losses, bad = _l2l(sched, step, model, mbs, _store(
        spill, plan, tmp_path, opener=lambda kind, path, size: Corrupt(path, size)))
    assert sched.compare(pinned, clean)["equal"]
    assert not sched.compare(pinned, bad)["equal"]
    assert sched.losses_equal(pinned_losses, bad_losses)
    assert sched.losses_equal(pinned_losses, clean_losses)


def test_another_input_seed_differs_in_both_directions(bud, spill, sched, tiny, tmp_path):
    model, runtime = tiny
    sched.make_non_vacuous(model)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, _mbs(2)[0]))
    layout = bud.StoreLayout(n_layers=N_LAYERS, k=2, chunk_bytes=SEQ * HIDDEN * 4, ring=1,
                             pin=False)
    plan = _plan(bud, layout.minimum_bytes, k=2)
    a_losses, a = _l2l(sched, step, model, _mbs(2, seed=17), _store(spill, plan, tmp_path))
    b_losses, b = _l2l(sched, step, model, _mbs(2, seed=18), _store(spill, plan, tmp_path))
    assert not sched.compare(a, b)["equal"]
    assert not sched.losses_equal(a_losses, b_losses)


def test_ctrl_c_in_the_middle_of_an_l2l_backward_leaves_nothing(bud, spill, sched, tiny,
                                                               tmp_path):
    class Interrupted(spill.BudgetedStore):
        def get(self, layer, mb):
            if layer == 1:
                raise KeyboardInterrupt
            return super().get(layer, mb)

    import torch

    model, runtime = tiny
    mbs = _mbs(2)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    layout = bud.StoreLayout(n_layers=N_LAYERS, k=2, chunk_bytes=SEQ * HIDDEN * 4, ring=1,
                             pin=False)
    releases = _Releases()
    store = Interrupted(_plan(bud, layout.minimum_bytes, k=2), chunk_shape=(1, SEQ, HIDDEN),
                        dtype=torch.float32, device="cpu", spill_dir=str(tmp_path / "spill"),
                        backend="buffered", release=releases)
    with pytest.raises(KeyboardInterrupt):
        with store:
            step.run(mbs, store)
    assert spill.leftover_spill_files(str(tmp_path / "spill")) == []
    assert releases.calls == 1
    assert _no_spill_threads()
    assert all(layer.use_checkpoint for layer in step.layers)
