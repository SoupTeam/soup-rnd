"""ZERO: one L2L step split over several devices in one process (``training.l2l_split_gpus``).

The step reads each streamed layer once into the main device's pool, copies it to the other
devices and runs a share of the window's micro-batches on each, so two GPUs share the
compute of one read. The adapters are replicated on the extra devices for the step and
their gradients are added into the main device's adapters at the end.

On CPU both "devices" are ``cpu``: the replica path (weights copied out of the pool,
``functional_call`` with replica adapters, the gradient merge) runs for real, and the step
must give the single-device step's losses and gradients.
"""
# ruff: noqa: F811  (fixtures are imported and taken as arguments)
from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load, streamed  # noqa: F401
from tests.test_l2l_qwen35 import _micro_batches, _store, qwen35_streamed  # noqa: F401


def _run(model, runtime, mbs, store, split=(), head=None):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]), split_devices=split)
    return step, step.run(mbs, store, head=head)


def _assert_close(single, split):
    import torch

    assert set(single) == set(split)
    for name in single:
        torch.testing.assert_close(split[name], single[name], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_split_step_equals_single_device_on_qwen35(qwen35_streamed, k):
    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(k)

    harness.zero_grads(model)
    _step, single_losses = _run(model, runtime, mbs, _store(k))
    single = harness.collect_grads(model)

    harness.zero_grads(model)
    _step, split_losses = _run(model, runtime, mbs, _store(k), split=("cpu",))
    split = harness.collect_grads(model)

    assert harness.compare(single, split)["all_zero"] == []
    _assert_close(single, split)
    assert harness.losses_equal(single_losses, split_losses)


def test_split_step_with_the_chunked_head(qwen35_streamed):
    from soup_cli.utils.l2l_step import label_tokens, sft_chunked_head

    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(4)
    head = sft_chunked_head(mbs, label_tokens(mbs), chunk=3)

    harness.zero_grads(model)
    _run(model, runtime, mbs, _store(4), head=head)
    single = harness.collect_grads(model)
    harness.zero_grads(model)
    _run(model, runtime, mbs, _store(4), split=("cpu",), head=head)
    _assert_close(single, harness.collect_grads(model))


def test_split_step_equals_single_device_on_llama(streamed):
    from tests.test_l2l_probe_harness import SEQ

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(3)
    hidden = int(model.config.hidden_size)

    def store():
        import torch

        from soup_cli.utils.l2l_activations import ActivationStore

        return ActivationStore(n_layers=3, k=3, chunk_shape=(1, SEQ, hidden),
                               dtype=torch.float32, device="cpu", pin=False)

    harness.zero_grads(model)
    _run(model, runtime, mbs, store())
    single = harness.collect_grads(model)
    harness.zero_grads(model)
    _run(model, runtime, mbs, store(), split=("cpu",))
    _assert_close(single, harness.collect_grads(model))


def test_replicas_follow_the_adapters_between_steps(qwen35_streamed):
    """An optimizer step between two windows: the second split window sees the new values."""
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(2)
    call = capture_layer_call(model, mbs[0])
    split_step = L2LStep(model, runtime, call, split_devices=("cpu",))

    harness.zero_grads(model)
    split_step.run(mbs, _store(2))
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(0.01)

    harness.zero_grads(model)
    L2LStep(model, runtime, call).run(mbs, _store(2))
    single = harness.collect_grads(model)
    harness.zero_grads(model)
    split_step.run(mbs, _store(2))
    _assert_close(single, harness.collect_grads(model))


def test_grads_add_to_existing_main_grads(qwen35_streamed):
    """The merge adds into ``.grad`` like accumulation does; it does not overwrite it."""
    import torch

    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(2)

    harness.zero_grads(model)
    _run(model, runtime, mbs, _store(2))
    once = harness.collect_grads(model)
    harness.zero_grads(model)
    _run(model, runtime, mbs, _store(2), split=("cpu",))
    _run(model, runtime, mbs, _store(2), split=("cpu",))
    twice = harness.collect_grads(model)
    for name in once:
        torch.testing.assert_close(twice[name], 2 * once[name], rtol=1e-5, atol=1e-7)


def test_split_refuses_a_reference_pass(qwen35_streamed):
    from soup_cli.utils.l2l_step import ReferencePass

    model, runtime, _config = qwen35_streamed
    mbs = _micro_batches(2)
    reference = ReferencePass(head=lambda *_a: None, adapters_off=True)
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]), split_devices=("cpu",))
    with pytest.raises(ValueError, match="split"):
        step.run(mbs, _store(2), reference=reference)


def test_split_refuses_a_trainable_embedding(qwen35_streamed, monkeypatch):
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, model_parts

    model, runtime, _config = qwen35_streamed
    mbs = _micro_batches(2)
    _causal, owner = model_parts(model)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]), split_devices=("cpu",))
    adapter = torch.nn.Parameter(torch.zeros(1))
    monkeypatch.setattr(owner.embed_tokens, "parameters", lambda *a, **k: iter([adapter]))
    with pytest.raises(ValueError, match="embedding"):
        step.run(mbs, _store(2))


def test_to_device_moves_nested_tensors_only():
    import torch

    from soup_cli.utils.l2l_split import to_device

    a, b = torch.ones(2), torch.zeros(1)
    moved = to_device(((a, b), {"mask": None, "pos": a, "flag": True}, [b]), "cpu")
    (x, y), kwargs, (z,) = moved
    assert torch.equal(x, a) and torch.equal(y, b) and torch.equal(z, b)
    assert kwargs["mask"] is None and kwargs["flag"] is True


def test_assignment_spreads_micro_batches_round_robin():
    from soup_cli.utils.l2l_split import assign

    assert assign(4, 2) == [0, 1, 0, 1]
    assert assign(3, 2) == [0, 1, 0]
    assert assign(1, 2) == [0]
    assert assign(5, 1) == [0] * 5


# -- launch ----------------------------------------------------------------------------------
def _sft(**training):
    from soup_cli.config.schema import SoupConfig

    settings = {
        "stream_layers": True, "stream_source": "disk", "quantization": "4bit",
        "batch_size": 1, "gradient_accumulation_steps": 4, "l2l": True,
        "lora": {"r": 4, "dropout": 0, "target_modules": ["q_proj"]},
    }
    settings.update(training)
    return SoupConfig.model_validate({
        "base": "example/model", "data": {"train": "train.jsonl", "chat_template": "chatml"},
        "training": settings,
    })


def test_schema_takes_split_gpus_with_l2l():
    assert _sft(l2l_split_gpus=2).training.l2l_split_gpus == 2
    assert _sft().training.l2l_split_gpus == 1


@pytest.mark.parametrize("value", [0, 9, -1])
def test_schema_refuses_out_of_range_split(value):
    with pytest.raises(ValueError, match="l2l_split_gpus"):
        _sft(l2l_split_gpus=value)


def test_schema_refuses_split_without_l2l():
    with pytest.raises(ValueError, match="l2l_split_gpus"):
        _sft(l2l=False, l2l_split_gpus=2)


def test_split_devices_lists_the_other_cards():
    from soup_cli.utils.l2l_split import split_devices

    assert split_devices("cuda:0", 1) == ()
    assert split_devices("cuda", 2) == ("cuda:1",)
    assert split_devices("cuda:0", 3) == ("cuda:1", "cuda:2")


def test_split_devices_refuses_too_few_cards(monkeypatch):
    import torch

    from soup_cli.utils.l2l_split import check_split

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="2 GPUs"):
        check_split("cuda", 2)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="--gpus"):
        check_split("cuda", 2)


def test_split_refuses_a_cpu_run():
    from soup_cli.utils.l2l_split import check_split

    with pytest.raises(ValueError, match="CUDA"):
        check_split("cpu", 2)
    check_split("cpu", 1)


def test_data_parallel_guard_admits_a_split_run(monkeypatch):
    import torch

    from soup_cli.trainer.stream_setup import refuse_if_data_parallel

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="DataParallel"):
        refuse_if_data_parallel("cuda")
    refuse_if_data_parallel("cuda", split_gpus=2)


def test_one_hf_gpu_for_a_split_run():
    from types import SimpleNamespace

    from soup_cli.utils.l2l_split import keep_hf_on_one_gpu

    args = SimpleNamespace(_n_gpu=2)
    keep_hf_on_one_gpu(args, 2)
    assert args._n_gpu == 1
    args = SimpleNamespace(_n_gpu=2)
    keep_hf_on_one_gpu(args, 1)
    assert args._n_gpu == 2


def test_trainer_state_builds_a_split_step(qwen35_streamed):
    from types import SimpleNamespace

    from soup_cli.trainer.l2l_trainer import _l2l_step_for, attach_l2l_state

    model, runtime, _config = qwen35_streamed
    trainer = SimpleNamespace(args=SimpleNamespace(output_dir="out"))
    state = attach_l2l_state(trainer, runtime, dtype=None, split_devices=("cpu",))
    state.has_dropout = False
    stepper, _calls = _l2l_step_for(state, model, _micro_batches(2))
    assert stepper.split_devices == ("cpu",)


def test_overlap_is_opt_in_and_cuda_only(monkeypatch):
    from soup_cli.utils.l2l_split import OVERLAP_ENV, Transfers, overlap_enabled

    monkeypatch.delenv(OVERLAP_ENV, raising=False)
    assert not overlap_enabled("cpu")
    assert not overlap_enabled("cuda:0")
    monkeypatch.setenv(OVERLAP_ENV, "1")
    assert overlap_enabled("cuda:0")
    assert not overlap_enabled("cpu")
    xfer = Transfers("cpu", ["cpu"])
    assert not xfer.enabled and xfer.side("acts", "cpu") is None


def test_weight_copy_without_overlap_is_a_plain_copy():
    import torch

    from soup_cli.utils.l2l_split import Transfers, copy_weights

    buffers = {"w": torch.arange(4.0)}
    copy = copy_weights(Transfers("cpu", ["cpu"]), buffers, "cpu")
    assert torch.equal(copy.moved["w"], buffers["w"])
    assert copy.src_done is None and copy.dst_done is None


def test_backward_hands_autograd_one_device_at_a_time(qwen35_streamed, monkeypatch):
    """One micro-batch per autograd call: two devices' graphs in one call run on two engine
    threads, and Triton's autotuner is not safe to tune one kernel from both (part E)."""
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime, _config = qwen35_streamed
    mbs = _micro_batches(4)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]), split_devices=("cpu",))
    sizes = []
    real = torch.autograd.backward

    def spy(tensors, *args, **kwargs):
        sizes.append(len(tensors) if isinstance(tensors, (list, tuple)) else 1)
        return real(tensors, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", spy)
    step.run(mbs, _store(4))
    assert sizes and set(sizes) == {1}
