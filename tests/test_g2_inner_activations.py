"""G2: keeping each layer's saved tensors instead of recomputing them, on CPU.

The streamed runtime refills its weight slots while earlier layers' graphs are
alive, so these tests pin the three things that make the store safe: losses and
LoRA gradients bit-equal to GA and to the recompute L2L step, no streamed weight
byte inside a record, and the guards that turn a recycled slot or an in-place
change into an error instead of a silently wrong gradient.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "harness"
SEQ = 16


def _load(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def g2():
    return _load("g2_inner_activations")


@pytest.fixture()
def sched():
    return _load("l2l_schedule")


@pytest.fixture()
def acts():
    return _load("l2l_activations")


def _streamed(tmp_path, quant="none"):
    pytest.importorskip("peft")
    from tests.test_v07203 import _tiny_stream

    model, runtime, _weights = _tiny_stream(tmp_path, quant=quant)
    return model, runtime


@pytest.fixture()
def streamed(tmp_path):
    model, runtime = _streamed(tmp_path)
    yield model, runtime
    runtime.close()


def _micro_batches(k, seed=17):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 64, (1, SEQ), generator=gen) for _ in range(k)]


def _reference(sched, acts, model, runtime, mbs):
    import torch

    sched.make_non_vacuous(model)
    sched.zero_grads(model)
    ga_losses = sched.ga_step(model, mbs)
    ga = sched.collect_grads(model)
    call = sched.capture_layer_call(model, mbs[0])
    store = acts.ActivationStore(n_layers=3, k=len(mbs), chunk_shape=(1, SEQ, 64),
                                 dtype=torch.float32, device="cpu", pin=False)
    sched.zero_grads(model)
    l2l_losses = sched.L2LStep(model, runtime, call).run(mbs, store)
    l2l = sched.collect_grads(model)
    return call, ga_losses, ga, l2l_losses, l2l


def _check_exact(sched, ref_losses, ref, losses, grads):
    result = sched.compare(ref, grads)
    assert result["all_zero"] == []
    assert result["equal"], result
    assert sched.losses_equal(ref_losses, losses)


# --------------------------------------------------------------------------
# exactness on the shipped streamed runtime
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3])
def test_ram_records_are_bit_exact_to_ga_and_to_recompute(g2, sched, acts, streamed, k):
    model, runtime = streamed
    mbs = _micro_batches(k)
    call, ga_losses, ga, l2l_losses, l2l = _reference(sched, acts, model, runtime, mbs)
    step = g2.G2Step(model, runtime, call, verify=True,
                     store_factory=g2.ram_store_factory("cpu", pin=False))
    try:
        sched.zero_grads(model)
        losses = step.run(mbs)
        grads = sched.collect_grads(model)
    finally:
        step.close()
    _check_exact(sched, ga_losses, ga, losses, grads)
    _check_exact(sched, l2l_losses, l2l, losses, grads)
    assert all(row["weight_bytes_stored"] == 0 for row in step.record_stats.values())
    # Every layer's weights were saved by reference and resolved against the slot.
    assert step.packer.unpacked["weight"] > 0


@pytest.mark.parametrize("k", [1, 2])
def test_profiled_recompute_is_exact_with_current_l2l_step(g2, sched, acts, streamed, k):
    import torch

    model, runtime = streamed
    mbs = _micro_batches(k)
    call, ga_losses, ga, l2l_losses, l2l = _reference(sched, acts, model, runtime, mbs)
    step = g2.ProfiledL2LStep(model, runtime, call)
    store = acts.ActivationStore(n_layers=3, k=k, chunk_shape=(1, SEQ, 64),
                                 dtype=torch.float32, device="cpu", pin=False)
    try:
        sched.zero_grads(model)
        losses = step.run(mbs, store)
        grads = sched.collect_grads(model)
    finally:
        store.close()
    _check_exact(sched, ga_losses, ga, losses, grads)
    _check_exact(sched, l2l_losses, l2l, losses, grads)
    assert step.timing["forward_layers_s"] > 0
    assert step.timing["recompute_s"] > 0
    assert step.timing["backward_layers_s"] > 0


def test_spilled_records_are_bit_exact_and_move_exactly_the_plan(g2, sched, acts, streamed,
                                                                 tmp_path):
    model, runtime = streamed
    k = 2
    mbs = _micro_batches(k)
    call, ga_losses, ga, _l2l_losses, _l2l = _reference(sched, acts, model, runtime, mbs)
    factory = g2.spill_store_factory("cpu", pin=False, spill_layers=2,
                                     path=str(tmp_path / "records.bin"), direct=False)
    step = g2.G2Step(model, runtime, call, store_factory=factory)
    try:
        sched.zero_grads(model)
        losses = step.run(mbs)
        grads = sched.collect_grads(model)
        stats = step.store.stats
        planned = 2 * k * step.capacity
        assert stats.bytes_written == planned and stats.bytes_read == planned
        # A second step reuses the store and moves the same bytes again.
        sched.zero_grads(model)
        step.run(mbs)
        assert step.store.stats.bytes_written == 2 * planned
        assert sched.compare(ga, sched.collect_grads(model))["equal"]
    finally:
        step.close()
    _check_exact(sched, ga_losses, ga, losses, grads)
    assert step.capacity % 4096 == 0
    assert not (tmp_path / "records.bin").exists()


def test_nf4_records_rebuild_the_dequantised_weight_instead_of_storing_it(g2, sched, acts,
                                                                          tmp_path):
    pytest.importorskip("bitsandbytes")
    assert g2.install_derived_tracking()
    model, runtime = _streamed(tmp_path, quant="nf4")
    try:
        mbs = _micro_batches(2)
        sched.make_non_vacuous(model)
        sched.zero_grads(model)
        ga_losses = sched.ga_step(model, mbs)
        ga = sched.collect_grads(model)
        call = sched.capture_layer_call(model, mbs[0])
        step = g2.G2Step(model, runtime, call, verify=True,
                         store_factory=g2.ram_store_factory("cpu", pin=False))
        try:
            sched.zero_grads(model)
            losses = step.run(mbs)
            grads = sched.collect_grads(model)
        finally:
            step.close()
    finally:
        runtime.close()
    _check_exact(sched, ga_losses, ga, losses, grads)
    row = next(iter(step.record_stats.values()))
    assert row["saved"]["derived"] > 0, "the dense dequantised weights were not recognised"
    assert row["weight_bytes_stored"] == 0 and row["weight_like_items"] == 0


# --------------------------------------------------------------------------
# the controls: each must be able to fail
# --------------------------------------------------------------------------
def test_identity_hooks_without_a_reload_give_wrong_gradients(g2, sched, acts, streamed):
    """Scratch exp2's hazard on this fixture: 3 layers, 2 weight slots. Keeping the
    autograd graph with identity hooks bypasses the version counter and the
    backward reads weights the pool has since overwritten."""
    model, runtime = streamed
    mbs = _micro_batches(2)
    call, _ga_losses, ga, _l2l_losses, _l2l = _reference(sched, acts, model, runtime, mbs)
    naive = g2.G2Step(model, runtime, call, store_factory=None, naive_identity=True)
    sched.zero_grads(model)
    naive.run(mbs)
    assert not sched.compare(ga, sched.collect_grads(model))["equal"]


def test_a_corrupted_record_changes_grads_but_not_losses(g2, sched, acts, streamed):
    model, runtime = streamed
    mbs = _micro_batches(2)
    call, ga_losses, ga, _l2l_losses, _l2l = _reference(sched, acts, model, runtime, mbs)
    step = g2.G2Step(model, runtime, call, store_factory=g2.ram_store_factory("cpu", pin=False))
    step.corrupt = (1, 0)
    try:
        sched.zero_grads(model)
        losses = step.run(mbs)
    finally:
        step.close()
    assert sched.losses_equal(ga_losses, losses)
    assert not sched.compare(ga, sched.collect_grads(model))["equal"]


# --------------------------------------------------------------------------
# the guards, on the packer itself
# --------------------------------------------------------------------------
class _Pool:
    def __init__(self, buffers, owner):
        self.buffers = buffers
        self.owner = owner


def test_a_weight_unpacked_from_a_recycled_slot_raises(g2):
    import torch

    weight = torch.randn(8, 8)
    pool = _Pool([{"w": weight}], owner=[0])
    packer = g2.Packer(weights=g2.streamed_weight_sites(pool), pool=pool)
    packer.begin((0, 0), None)
    ref = packer.pack(weight.t())
    packer.seal()
    packer.release()
    packer.begin_replay((0, 0), torch.zeros(4096, dtype=torch.uint8))
    assert torch.equal(packer.unpack(ref), weight.t())
    pool.owner[0] = 2
    with pytest.raises(RuntimeError, match="recycled weight slot"):
        packer.unpack(ref)


def test_a_saved_activation_changed_in_place_before_sealing_raises(g2):
    import torch

    packer = g2.Packer(weights={})
    act = torch.randn(4, 4)
    packer.begin((0, 0), None)
    packer.pack(act)
    act.add_(1.0)
    with pytest.raises(RuntimeError, match="modified in place"):
        packer.seal()


def test_a_held_tensor_changed_before_its_backward_raises(g2):
    import torch

    lora = torch.randn(4, 4, requires_grad=True)
    packer = g2.Packer(weights={}, params={g2._sid(lora): lora})
    packer.begin((0, 0), None)
    ref = packer.pack(lora)
    packer.seal()
    with torch.no_grad():
        lora.add_(1.0)
    with pytest.raises(RuntimeError, match="modified before its backward"):
        packer.unpack(ref)


def test_a_transposed_square_derived_weight_comes_back_transposed(g2):
    """A recomputed square weight has the saved view's size; only its stride says
    it was saved transposed, so the saved geometry must always be re-applied."""
    import torch

    base = torch.arange(16.0).view(4, 4)
    packer = g2.Packer(weights={})
    packer.begin((0, 0), None)
    packer.register_derived(base, torch.empty(0), lambda: base.clone())
    ref = packer.pack(base.t())
    packer.seal()
    packer.begin_replay((0, 0), torch.zeros(4096, dtype=torch.uint8))
    assert torch.equal(packer.unpack(ref), base.t())


def test_strided_views_of_one_storage_share_one_record_item(g2):
    import torch

    act = torch.randn(6, 8)
    packer = g2.Packer(weights={})
    packer.begin((0, 0), None)
    refs = [packer.pack(act), packer.pack(act.t()[2:5]), packer.pack(act[1:3])]
    stats, items = packer.seal()
    assert len(items) == 1 and stats.payload_bytes == act.numel() * 4
    record = torch.zeros(g2.align_up(stats.padded_bytes, 4096), dtype=torch.uint8)
    raw = torch.empty((0,), dtype=torch.uint8).set_(act.untyped_storage(), 0,
                                                    (items[0].nbytes,), (1,))
    record[items[0].offset:items[0].offset + items[0].nbytes].copy_(raw)
    packer.release()
    act.zero_()  # the record is a copy: later writes to the source must not matter
    packer.begin_replay((0, 0), record)
    expected = torch.empty((0,), dtype=torch.uint8).set_(record.untyped_storage(), 0,
                                                         (items[0].nbytes,), (1,))
    original = expected.view(torch.float32).view(6, 8)
    for ref, view in zip(refs, (original, original.t()[2:5], original[1:3])):
        assert torch.equal(packer.unpack(ref), view)


def test_weight_bytes_are_detected_by_address_not_by_classification(g2):
    """An item living inside a weight buffer is counted even if the classifier
    missed it (here: the buffer is not registered as a weight site)."""
    import torch

    buf = torch.randn(64)
    lo = buf.untyped_storage().data_ptr()
    packer = g2.Packer(weights={}, weight_ranges=[(lo, lo + buf.untyped_storage().nbytes())])
    packer.begin((0, 0), None)
    packer.pack(buf[8:16])
    stats, _items = packer.seal()
    assert stats.weight_bytes_stored == 64 * 4


def test_counting_a_plain_layer_matches_the_streamed_record(g2, sched, acts, streamed, tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    model, runtime = streamed
    mbs = _micro_batches(1)
    call = sched.capture_layer_call(model, mbs[0])
    step = g2.G2Step(model, runtime, call, store_factory=g2.ram_store_factory("cpu", pin=False))
    try:
        sched.zero_grads(model)
        step.run(mbs)
    finally:
        step.close()
    stored = step.record_stats[(0, 0)]

    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType, get_peft_model

    plain = AutoModelForCausalLM.from_pretrained(str(tmp_path / "model"),
                                                 dtype=torch.float32).eval()
    for param in plain.parameters():
        param.requires_grad = False
    plain = get_peft_model(plain, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                             target_modules=["q_proj", "v_proj"],
                                             task_type=TaskType.CAUSAL_LM))
    counted = g2.count_layer_saved_bytes(plain, mbs[0])
    assert counted["payload_bytes"] == stored["payload_bytes"]
    assert counted["boundary_bytes"] == stored["boundary_bytes"] == SEQ * 64 * 4
    assert counted["saved"] == stored["saved"]
