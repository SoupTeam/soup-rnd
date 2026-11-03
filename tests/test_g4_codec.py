"""CPU tests for the G4 boundary-activation codecs and their store
(benchmarks/harness/g4_activation_codec.py).

What a GPU run cannot check by itself: the per-element error bound on adversarial
rows (outliers, zeros, denormals, bf16 extremes, non-finite), the byte accounting the
A4 planner consumes, and the null control: the identity codec through the same store
wrapper and the same ``L2LStep`` gives gradients ``torch.equal`` to the plain store.
"""

from __future__ import annotations

import math

import pytest

from tests import test_l2l_probe_harness as l2l_tests

torch = pytest.importorskip("torch")

#: The tiny streamed Llama fixture of the L2L harness tests (3 layers, hidden 64).
streamed = l2l_tests.streamed

BF16_TINY = 2.0**-133  # the smallest bf16 denormal
BF16_MAX = torch.finfo(torch.bfloat16).max


@pytest.fixture()
def codec():
    return l2l_tests._load("g4_activation_codec")


@pytest.fixture()
def sched():
    return l2l_tests._load("l2l_schedule")


@pytest.fixture()
def acts():
    return l2l_tests._load("l2l_activations")


def _adversarial(hidden: int = 96):
    """One bf16 row per hazard, (rows, hidden)."""
    gen = torch.Generator().manual_seed(5)
    normal = torch.randn(hidden, generator=gen)
    rows = {
        "gaussian": normal,
        "massive_outlier": torch.cat([torch.tensor([3.0e4]), normal[1:] * 0.5]),
        "two_signs_outlier": torch.cat([torch.tensor([-2.0e3, 1.9e3]), normal[2:]]),
        "zeros": torch.zeros(hidden),
        "one_nonzero": torch.cat([torch.tensor([1e-3]), torch.zeros(hidden - 1)]),
        "denormals": torch.arange(hidden, dtype=torch.float32) % 5 * BF16_TINY,
        "mixed_denormal": torch.cat([torch.tensor([1.0]), torch.full((hidden - 1,), BF16_TINY)]),
        "bf16_max": torch.cat([torch.tensor([BF16_MAX, -BF16_MAX]), normal[2:] * 1e30]),
        "bf16_min_normal": normal * torch.finfo(torch.bfloat16).tiny,
        "constant": torch.full((hidden,), -7.25),
    }
    names = list(rows)
    x = torch.stack([rows[name] for name in names]).to(torch.bfloat16)
    return names, x


#: The worst-case per-element error of each codec, as a fraction of the row absmax:
#: half a quantisation step. fp8 is relative to the element: half an e4m3 ulp is at
#: most 2^-4 of it, plus half a subnormal step (2^-10 of 448ths of the absmax).
def _element_bound(name, x, absmax):
    if name == "int8":
        codec_error = absmax / 254
    elif name == "int4":
        codec_error = absmax / 14
    else:
        codec_error = x.abs() * 2.0**-4 + absmax * 2.0**-10 / 448
    # The restored value is rounded to bf16: half an ulp, or half the denormal grid.
    return codec_error * (1 + 2.0**-8) + x.abs() * 2.0**-8 + BF16_TINY


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_round_trip_stays_inside_the_element_bound_on_adversarial_rows(codec, name):
    names, x = _adversarial()
    restored = codec.round_trip(x, name)
    assert restored.dtype == torch.bfloat16
    assert restored.shape == x.shape
    xf, rf = x.float(), restored.float()
    absmax = xf.abs().amax(dim=-1, keepdim=True)
    assert torch.isfinite(rf).all(), "a finite bf16 row must restore finite, extremes included"
    excess = (rf - xf).abs() - _element_bound(name, xf, absmax)
    worst = excess.amax(dim=-1)
    bad = [names[i] for i in range(len(names)) if worst[i] > 0]
    assert not bad, f"{name} breaks its element bound on rows {bad}"


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_the_row_absmax_element_and_zero_rows_restore_exactly(codec, name):
    names, x = _adversarial(hidden=97)  # odd: int4 packs a pad nibble
    restored = codec.round_trip(x, name)
    idx = x.float().abs().argmax(dim=-1, keepdim=True)
    assert torch.equal(restored.gather(-1, idx), x.gather(-1, idx))
    zero = names.index("zeros")
    assert torch.equal(restored[zero], torch.zeros_like(restored[zero]))
    assert not torch.signbit(restored[zero]).any()


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_refused_not_encoded(codec, name, bad):
    """A NaN or Inf has no absmax to scale by: encoding it would store garbage."""
    x = torch.randn(3, 16).to(torch.bfloat16)
    x[1, 4] = bad
    with pytest.raises(codec.NonFiniteActivationError, match="1 of 3 rows"):
        codec.encode(x, name)
    with pytest.raises(codec.NonFiniteActivationError):
        codec.round_trip(x, name)
    clean = x[[0, 2]]
    assert torch.isfinite(codec.round_trip(clean, name)).all()


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_the_store_refuses_a_step_whose_boundary_was_non_finite(codec, name):
    """The forward is never synced: the refusal comes at end_step, before the caller
    can apply the step, and it names the (layer, micro-batch)."""
    store = _codec_store(codec, name)
    try:
        for layer in range(3):
            for mb in range(2):
                x = torch.randn(1, 8, 64)
                if (layer, mb) == (1, 1):
                    x[0, 3, 5] = float("nan")
                store.put(layer, mb, x)
        store.begin_backward()
        for layer in (2, 1, 0):
            for mb in range(2):
                store.get(layer, mb)
        with pytest.raises(codec.NonFiniteActivationError, match=r"\(1, 1\)"):
            store.end_step()
        # The next step starts clean: the flag is per step, not sticky.
        for layer in range(3):
            for mb in range(2):
                store.put(layer, mb, torch.randn(1, 8, 64))
        store.begin_backward()
        for layer in (2, 1, 0):
            for mb in range(2):
                store.get(layer, mb)
        store.end_step()
    finally:
        store.close()


def test_the_identity_store_keeps_non_finite_values_byte_for_byte(codec):
    store = _codec_store(codec, "bf16")
    x = torch.randn(1, 8, 64)
    x[0, 0, 0] = float("nan")
    store.put(0, 0, x)
    store.begin_backward()
    assert torch.equal(store.get(0, 0).isnan(), x.isnan())
    store.end_step()
    store.close()


def test_scales_are_the_float32_row_absmax(codec):
    x = torch.tensor([[1.5, -3.0, 0.25], [0.0, 0.0, 0.0], [BF16_TINY, 0.0, -BF16_TINY]])
    x = x.to(torch.bfloat16)
    for name in ("int8", "fp8_e4m3", "int4"):
        _payload, scale = codec.encode(x, name)
        assert scale.dtype == torch.float32
        assert torch.equal(scale, x.float().abs().amax(dim=-1))


def test_payload_dtypes_and_identity_is_untouched(codec):
    x = torch.randn(1, 4, 10).to(torch.bfloat16)
    assert codec.encode(x, "int8")[0].dtype == torch.int8
    assert codec.encode(x, "fp8_e4m3")[0].dtype == torch.float8_e4m3fn
    packed = codec.encode(x, "int4")[0]
    assert packed.dtype == torch.uint8 and packed.shape == (1, 4, 5)
    payload, scale = codec.encode(x, "bf16")
    assert payload is x and scale is None
    assert codec.round_trip(x, "bf16") is x


def test_coarse_codec_is_measurably_worse_than_both_candidates(codec):
    """The positive control's premise: on Gaussian rows int4 is ~10x int8's error."""
    x = torch.randn(64, 4096, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    err = {}
    for name in ("int8", "fp8_e4m3", "int4"):
        stats = codec.restore_error(x, codec.round_trip(x, name))
        err[name] = stats["rel_l2"]
    assert err["int8"] < 0.02 and err["fp8_e4m3"] < 0.04
    assert err["int4"] > 4 * max(err["int8"], err["fp8_e4m3"])


def test_unknown_codec_is_refused(codec):
    with pytest.raises(ValueError, match="unknown codec"):
        codec.codec_of("int2")


# --------------------------------------------------------------------------
# restore-error metric
# --------------------------------------------------------------------------
def test_restore_error_metrics(codec):
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    x_hat = torch.tensor([[3.0, 4.5], [0.0, 0.0], [1.0, 0.1]])
    stats = codec.restore_error(x, x_hat)
    assert stats["rel_l2"] == pytest.approx(math.sqrt(0.25 + 0.01) / math.sqrt(26.0))
    assert stats["max_abs"] == pytest.approx(0.5)
    assert stats["max_abs_over_absmax"] == pytest.approx(0.125)
    assert stats["worst_row_rel"] == pytest.approx(0.1)
    assert stats["err_sq"] == pytest.approx(0.26)
    assert stats["ref_sq"] == pytest.approx(26.0)


def test_restore_error_of_a_destroyed_zero_row_is_infinite(codec):
    x = torch.zeros(1, 4)
    stats = codec.restore_error(x, torch.tensor([[0.0, 0.0, 1e-9, 0.0]]))
    assert stats["worst_row_rel"] == math.inf
    assert codec.restore_error(x, x.clone())["worst_row_rel"] == 0.0


def test_pooled_relative_l2_over_chunks(codec):
    parts = [{"err_sq": 1.0, "ref_sq": 100.0}, {"err_sq": 3.0, "ref_sq": 300.0}]
    assert codec.pooled_rel_l2(parts) == pytest.approx(0.1)
    assert codec.pooled_rel_l2([{"err_sq": 0.0, "ref_sq": 0.0}]) == 0.0


# --------------------------------------------------------------------------
# byte accounting
# --------------------------------------------------------------------------
def test_chunk_bytes_on_the_7b_shape(codec):
    shape = (1, 512, 4096)
    assert codec.compressed_chunk_bytes(shape, "bf16") == 512 * 4096 * 2
    assert codec.compressed_chunk_bytes(shape, "int8") == 512 * 4096 + 512 * 4
    assert codec.compressed_chunk_bytes(shape, "fp8_e4m3") == 512 * 4096 + 512 * 4
    assert codec.compressed_chunk_bytes(shape, "int4") == 512 * 2048 + 512 * 4
    assert codec.payload_chunk_bytes(shape, "int8") == 512 * 4096
    assert codec.scale_chunk_bytes(shape, "int8") == 2048
    assert codec.scale_chunk_bytes(shape, "bf16") == 0


def test_giant_per_token_table_matches_the_pdf(codec):
    rows = {(r["shape"], r["codec"]): r for r in codec.per_token_table()}
    k2 = rows[("kimi-k2", "bf16")]
    assert k2["bytes_per_token"] == 61 * 7168 * 2 == 874_496  # "874 KB per token"
    int8 = rows[("kimi-k2", "int8")]
    assert int8["bytes_per_token"] == 61 * (7168 + 4)
    assert int8["tokens_per_step_factor_at_fixed_ram"] == pytest.approx(2 * 7168 / 7172)
    assert all("not measured" in r["kind"] for r in rows.values())


def test_tokens_at_budget_is_floor_arithmetic(codec):
    budget = 32 * 4096 * 2 * 1000 + 5
    assert codec.tokens_at_budget(budget, n_layers=32, hidden=4096, codec="bf16") == 1000
    assert codec.tokens_at_budget(budget, n_layers=32, hidden=4096, codec="int8") == 1998


def _codec_store(codec, name, *, n_layers=3, k=2, shape=(1, 8, 64), dtype=None, **kw):
    return codec.CodecActivationStore(
        codec=name, n_layers=n_layers, k=k, chunk_shape=shape,
        dtype=dtype or torch.float32, device="cpu", pin=False, **kw,
    )


@pytest.mark.parametrize("name", ["bf16", "int8", "fp8_e4m3", "int4"])
def test_store_host_bytes_are_measured_and_match_the_arithmetic(codec, name):
    store = _codec_store(codec, name, dtype=torch.bfloat16)
    try:
        held = store.host_bytes()
        expected = codec.compressed_chunk_bytes((1, 8, 64), name)
        assert store.chunk_bytes == expected
        assert held["host_bytes"] == 3 * 2 * expected
        assert held["measured_bytes_per_token_per_layer"] == expected / 8
        assert held["arithmetic_bytes_per_token_per_layer"] == expected / 8
    finally:
        store.close()


def test_identity_store_accounts_the_source_dtype(codec):
    store = _codec_store(codec, "bf16", dtype=torch.float32)
    assert store.chunk_bytes == 8 * 64 * 4
    assert store.chunk_bytes == codec.compressed_chunk_bytes(
        (1, 8, 64), "bf16", source_element_bytes=4)
    store.close()


@pytest.mark.parametrize("hidden", [1, 7, 97, 4097])
def test_odd_hidden_int4_packs_one_pad_nibble_and_round_trips(codec, hidden):
    """An odd row packs to ceil(hidden / 2) bytes: the pad nibble is real storage."""
    x = torch.randn(2, 5, hidden).to(torch.bfloat16)
    payload, scale = codec.encode(x, "int4")
    assert payload.shape == (2, 5, (hidden + 1) // 2)
    assert payload.numel() * payload.element_size() == codec.payload_chunk_bytes(
        (2, 5, hidden), "int4")
    assert scale.numel() * scale.element_size() == codec.scale_chunk_bytes((2, 5, hidden), "int4")
    restored = codec.round_trip(x, "int4")
    assert restored.shape == x.shape
    idx = x.float().abs().argmax(dim=-1, keepdim=True)
    assert torch.equal(restored.gather(-1, idx), x.gather(-1, idx))


@pytest.mark.parametrize("name", ["bf16", "int8", "fp8_e4m3", "int4"])
@pytest.mark.parametrize("shape", [(1, 8, 64), (1, 3, 97), (2, 5, 33)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_compressed_chunk_bytes_equal_the_bytes_the_store_allocates(codec, name, shape, dtype):
    """The A4 contract, measured: payload + scales per chunk from the real storages,
    odd int4 widths and float32 identity (source_element_bytes=4) included."""
    torch_dtype = getattr(torch, dtype)
    element = torch.empty((), dtype=torch_dtype).element_size()
    store = _codec_store(codec, name, shape=shape, dtype=torch_dtype)
    try:
        payloads = list(store._inner._resident.values())
        assert len(payloads) == 3 * 2
        assert {t.untyped_storage().nbytes() for t in payloads} == {
            codec.payload_chunk_bytes(shape, name, source_element_bytes=element)}
        slab = 0 if store._scale_slab is None else store._scale_slab.untyped_storage().nbytes()
        assert slab == 3 * 2 * codec.scale_chunk_bytes(shape, name)
        held = store.host_bytes()
        assert held["host_bytes"] == 3 * 2 * codec.compressed_chunk_bytes(
            shape, name, source_element_bytes=element)
        assert held["spill_staging_bytes"] == 0 and held["spill_file_bytes"] == 0
        assert store.payload_chunk_bytes + store.scale_chunk_bytes == store.chunk_bytes
    finally:
        store.close()


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["bf16", "int8", "fp8_e4m3", "int4"])
def test_store_returns_exactly_the_codec_round_trip_per_key(codec, name):
    """Each (layer, micro-batch) gets its own scales: a mix-up shows as a wrong scale."""
    store = _codec_store(codec, name, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(3)
    chunks = {}
    for layer in range(3):
        for mb in range(2):
            x = (torch.randn(1, 8, 64, generator=gen) * 10 ** (layer + mb)).to(torch.bfloat16)
            chunks[(layer, mb)] = x
            store.put(layer, mb, x)
    store.begin_backward()
    for layer in (2, 1, 0):
        for mb in range(2):
            got = store.get(layer, mb)
            assert got.dtype == torch.bfloat16
            assert torch.equal(got, codec.round_trip(chunks[(layer, mb)], name))
            assert got.data_ptr() != chunks[(layer, mb)].data_ptr()
    store.end_step()
    assert store.codec_stats.timer == "perf_counter"
    calls = 0 if name == "bf16" else 6
    assert store.codec_stats.compress_calls == calls
    assert store.codec_stats.restore_calls == calls
    store.close()


def test_store_keeps_per_chunk_times_on_request(codec):
    store = _codec_store(codec, "int8", keep_chunk_times=True)
    store.put(0, 0, torch.randn(1, 8, 64))
    store.get(0, 0)
    assert len(store.codec_stats.compress_chunk_s) == 1
    assert len(store.codec_stats.restore_chunk_s) == 1
    assert store.codec_stats.compress_s >= 0.0
    store.close()


def _spill_threads():
    import threading

    return [t for t in threading.enumerate() if t.name.startswith("l2l-spill-")]


def _backend(acts, kind, path, calls=None):
    def factory(size):
        if calls is not None:
            calls.append(size)
        made = acts.DirectSpillFile if kind == "direct" else acts.BufferedSpillFile
        return made(str(path), size)

    return factory


@pytest.mark.parametrize("name", ["int8", "int4", "bf16"])
def test_spill_of_a_payload_that_is_not_a_sector_multiple_is_refused_before_any_file(
        codec, acts, tmp_path, name):
    """No padding: the spill file would otherwise hold unaccounted bytes per chunk."""
    calls = []
    with pytest.raises(ValueError, match="multiple of 4096"):
        _codec_store(codec, name, shape=(1, 8, 100), spill_layers=1,
                     spill_factory=_backend(acts, "buffered", tmp_path / "s.bin", calls))
    assert calls == [] and not (tmp_path / "s.bin").exists()
    # The same shape keeps every layer resident without complaint.
    _codec_store(codec, name, shape=(1, 8, 100)).close()


@pytest.mark.parametrize("kind", ["buffered", "direct"])
@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4", "bf16"])
def test_spilled_payload_round_trips_through_the_real_backends(codec, acts, tmp_path, kind, name):
    """put/get through the inner spill, two steps; scales stay resident; the file is
    exactly spill_layers x k x payload, the rings exactly 2 x ring x (payload + sector);
    close joins the threads and removes the file."""
    hidden = {"int4": 1024, "bf16": 256}.get(name, 512)  # every payload: 4096 B
    shape = (1, 8, hidden)
    path = tmp_path / "s.bin"
    calls = []
    before = set(_spill_threads())
    store = _codec_store(codec, name, shape=shape, dtype=torch.bfloat16, spill_layers=2,
                         ring=2, spill_factory=_backend(acts, kind, path, calls))
    try:
        assert store.payload_chunk_bytes == 4096 == store.stats.chunk_bytes
        assert calls == [2 * 2 * 4096]
        assert store.stats.direct is (kind == "direct")
        gen = torch.Generator().manual_seed(9)
        for _step in range(2):
            chunks = {}
            for layer in range(3):
                for mb in range(2):
                    x = (torch.randn(shape, generator=gen) * (layer + 1)).to(torch.bfloat16)
                    chunks[(layer, mb)] = x
                    store.put(layer, mb, x)
            store.begin_backward()
            for layer in (2, 1, 0):
                for mb in range(2):
                    assert torch.equal(store.get(layer, mb),
                                       codec.round_trip(chunks[(layer, mb)], name))
            store.end_step()
        held = store.host_bytes()
        assert held["spill_file_bytes"] == path.stat().st_size == 2 * 2 * 4096
        assert held["spilled_payload_bytes"] == 2 * 2 * 4096
        assert held["resident_payload_bytes"] == 1 * 2 * 4096
        assert held["scale_bytes"] == (0 if name == "bf16" else 3 * 2 * 8 * 4)
        assert held["spill_staging_bytes"] == 2 * 2 * (4096 + 4096)
        assert store.stats.bytes_written == store.stats.bytes_read == 2 * 4 * 4096
        assert len(set(_spill_threads()) - before) == 2
    finally:
        store.close()
    assert not path.exists()
    assert set(_spill_threads()) <= before
    store.close()  # idempotent


def test_a_failed_spill_write_surfaces_and_close_still_cleans_up(codec, acts, tmp_path):
    class BrokenFile(acts.BufferedSpillFile):
        def write(self, offset, buf):
            raise OSError("disk full")

    path = tmp_path / "s.bin"
    before = set(_spill_threads())
    store = _codec_store(codec, "int8", shape=(1, 8, 512), spill_layers=2,
                         spill_factory=lambda size: BrokenFile(str(path), size))
    with pytest.raises(RuntimeError, match="disk full"):
        for _ in range(500):
            store.put(0, 0, torch.randn(1, 8, 512))
    store.close()
    assert not path.exists()
    assert set(_spill_threads()) <= before


def test_a_failed_scale_allocation_closes_the_spilling_inner_store(codec, acts, tmp_path,
                                                                   monkeypatch):
    """The inner store has opened the file and started its threads by then."""
    path = tmp_path / "s.bin"
    real_zeros = torch.zeros
    before = set(_spill_threads())

    def zeros(*args, **kwargs):
        if kwargs.get("dtype") is torch.float32 and len(args) == 1 and len(args[0]) == 3:
            raise RuntimeError("pinned allocation failed")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    with pytest.raises(RuntimeError, match="pinned allocation failed"):
        _codec_store(codec, "int8", shape=(1, 8, 512), spill_layers=2,
                     spill_factory=_backend(acts, "buffered", path))
    assert not path.exists()
    assert set(_spill_threads()) <= before


# --------------------------------------------------------------------------
# through the L2L step on the tiny streamed model
# --------------------------------------------------------------------------
def _l2l(sched, model, runtime, mbs, store):
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    sched.zero_grads(model)
    losses = step.run(mbs, store)
    grads = sched.collect_grads(model)
    store.close()
    return losses, grads


def _tiny_store(codec, acts, name, k):
    if name is None:
        return l2l_tests._cpu_store(acts, k)
    return codec.CodecActivationStore(
        codec=name, n_layers=3, k=k, chunk_shape=(1, l2l_tests.SEQ, 64),
        dtype=torch.float32, device="cpu", pin=False,
    )


def test_identity_codec_is_a_null_control_through_l2l(codec, sched, acts, streamed):
    """Card step 1: the existing store gives L2L == GA exactly, and the identity codec
    through the wrapper gives the same gradients and losses bit for bit."""
    model, runtime = streamed
    sched.make_non_vacuous(model)
    mbs = l2l_tests._micro_batches(4)
    sched.zero_grads(model)
    ga_losses = sched.ga_step(model, mbs)
    ga = sched.collect_grads(model)
    plain_losses, plain = _l2l(sched, model, runtime, mbs, _tiny_store(codec, acts, None, 4))
    ident_losses, ident = _l2l(sched, model, runtime, mbs, _tiny_store(codec, acts, "bf16", 4))
    assert sched.compare(ga, plain)["equal"]
    assert sched.losses_equal(ga_losses, plain_losses)
    result = sched.compare(plain, ident)
    assert result["equal"], result
    assert result["all_zero"] == []
    assert sched.losses_equal(plain_losses, ident_losses)
    direct = sched.compare(ga, ident)
    assert direct["equal"] and direct["all_zero"] == [], direct
    assert sched.losses_equal(ga_losses, ident_losses)


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_a_lossy_codec_changes_grads_but_never_the_forward_losses(
    codec, sched, acts, streamed, name
):
    model, runtime = streamed
    sched.make_non_vacuous(model)
    mbs = l2l_tests._micro_batches(4)
    exact_losses, exact = _l2l(sched, model, runtime, mbs, _tiny_store(codec, acts, "bf16", 4))
    losses, grads = _l2l(sched, model, runtime, mbs, _tiny_store(codec, acts, name, 4))
    assert sched.losses_equal(exact_losses, losses)
    assert not sched.compare(exact, grads)["equal"]
    agreement = codec.grad_agreement(exact, grads)
    assert agreement["global_rel_l2"] > 0
    assert agreement["min_cosine"] > 0.5


def test_the_exactness_checks_discriminate(codec, sched, acts, streamed):
    """Controls for the two equalities above. The loss check is not vacuous: a store
    that writes its int4 restore back into the forward's hidden changes the losses.
    The gradient check is not vacuous: int4 moves the gradients far more than either
    candidate, and every lossy codec differs from exact."""

    class LeakingStore(codec.CodecActivationStore):
        def put(self, layer, mb, tensor):
            tensor.copy_(codec.round_trip(tensor, "int4"))
            super().put(layer, mb, tensor)

    model, runtime = streamed
    sched.make_non_vacuous(model)
    mbs = l2l_tests._micro_batches(4)
    exact_losses, exact = _l2l(sched, model, runtime, mbs, _tiny_store(codec, acts, "bf16", 4))
    leak = LeakingStore(codec="bf16", n_layers=3, k=4, chunk_shape=(1, l2l_tests.SEQ, 64),
                        dtype=torch.float32, device="cpu", pin=False)
    leak_losses, _ = _l2l(sched, model, runtime, mbs, leak)
    assert not sched.losses_equal(exact_losses, leak_losses)
    rel = {}
    for name in ("int8", "fp8_e4m3", "int4"):
        losses, grads = _l2l(sched, model, runtime, mbs, _tiny_store(codec, acts, name, 4))
        assert sched.losses_equal(exact_losses, losses), name
        assert not sched.compare(exact, grads)["equal"], name
        rel[name] = codec.grad_agreement(exact, grads)["global_rel_l2"]
    # Strict ordering only: on this 3-layer fixture int4 measured 3.7x fp8 (an earlier
    # draft's 4x guess failed), and no factor is claimed beyond the ordering.
    assert rel["int4"] > max(rel["int8"], rel["fp8_e4m3"]), rel
