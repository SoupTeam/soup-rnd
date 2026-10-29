"""Ticket 03: `training.l2l` in the SFT trainer.

Seams: the config schema (`load_config_from_string`), the launch guard
(`soup_cli.trainer.l2l_trainer.refuse_unsupported_l2l`) and the trainer subclass
(`make_l2l_trainer_class`) driven through `training_step`, the way HF's loop drives it.
"""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _yaml(task="sft", lora_dropout=0.0, **training):
    fields = {
        "batch_size": 1,
        "quantization": "none",
        "stream_layers": True,
        "gradient_accumulation_steps": 2,
        "lora": {"r": 4, "alpha": 8, "dropout": lora_dropout},
    }
    fields.update(training)
    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": task,
            "data": {"train": "data.jsonl"},
            "training": fields,
        }
    )


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def test_l2l_defaults_to_false_and_leaves_existing_configs_alone():
    cfg = load_config_from_string(_yaml())
    assert cfg.training.l2l is False


def test_l2l_loads_with_stream_layers_and_accumulation():
    cfg = load_config_from_string(_yaml(l2l=True))
    assert cfg.training.l2l is True
    assert cfg.training.gradient_accumulation_steps == 2


def test_l2l_with_k_equal_one_is_accepted():
    cfg = load_config_from_string(_yaml(l2l=True, gradient_accumulation_steps=1))
    assert cfg.training.l2l is True


def test_l2l_without_stream_layers_is_refused():
    with pytest.raises(ValueError, match=r"l2l.*stream_layers"):
        load_config_from_string(_yaml(l2l=True, stream_layers=False))


def test_l2l_with_nonzero_lora_dropout_is_accepted():
    cfg = load_config_from_string(_yaml(l2l=True, lora_dropout=0.05))
    assert cfg.training.l2l is True
    assert cfg.training.lora.dropout == 0.05


def test_l2l_for_a_task_other_than_sft_is_refused():
    with pytest.raises(ValueError, match=r"l2l.*sft"):
        load_config_from_string(_yaml(task="dpo", l2l=True))


def test_lora_dropout_is_only_checked_when_l2l_is_on():
    cfg = load_config_from_string(_yaml(lora_dropout=0.05))
    assert cfg.training.l2l is False


# --------------------------------------------------------------------------
# launch guard
# --------------------------------------------------------------------------
class _Cfg:
    def __init__(self, model_type="llama", attention_dropout=0.0):
        self.model_type = model_type
        self.attention_dropout = attention_dropout


def _guard(model_type="llama", attention_dropout=0.0, n_gpu=1, fp16=False):
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    refuse_unsupported_l2l(_Cfg(model_type, attention_dropout), n_gpu=n_gpu, fp16=fp16)


@pytest.mark.parametrize("arch", ["llama", "mistral"])
def test_guard_accepts_llama_and_mistral(arch):
    _guard(model_type=arch)


def test_guard_refuses_another_architecture():
    with pytest.raises(ValueError, match=r"qwen2.*llama.*mistral"):
        _guard(model_type="qwen2")


def test_guard_accepts_attention_dropout():
    _guard(attention_dropout=0.1)


def test_guard_refuses_several_gpus():
    with pytest.raises(ValueError, match=r"2 GPU"):
        _guard(n_gpu=2)


def test_guard_refuses_fp16_because_the_grad_scaler_is_bypassed():
    with pytest.raises(ValueError, match="fp16"):
        _guard(fp16=True)


# --------------------------------------------------------------------------
# the trainer subclass, driven by HF's own loop
# --------------------------------------------------------------------------
from tests.test_l2l_probe_harness import SEQ, _load, streamed  # noqa: E402,F401


class _Rows:
    def __init__(self, n, vocab=64, seed=5, lengths=None):
        import torch

        gen = torch.Generator().manual_seed(seed)
        self.rows = []
        for i in range(n):
            ids = torch.randint(0, vocab, (lengths[i] if lengths else SEQ,), generator=gen)
            self.rows.append(
                {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _train(
    base_cls, model, runtime, tmp_path, *, l2l, k, rows=4, lengths=None, batch_size=1, **targs
):
    import torch
    from transformers import TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    class Count(TrainerCallback):
        optimizer_steps = 0

        def on_optimizer_step(self, args, state, control, **kw):
            Count.optimizer_steps += 1

    args = TrainingArguments(
        output_dir=str(tmp_path / ("l2l" if l2l else "ga")),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=k,
        learning_rate=0.5,
        optim="sgd",
        lr_scheduler_type="constant",
        num_train_epochs=1,
        max_steps=-1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=1,
        data_seed=1,
        max_grad_norm=0.0,
        disable_tqdm=True,
        **targs,
    )
    cls = make_l2l_trainer_class(base_cls) if l2l else base_cls
    data = _Rows(rows, lengths=lengths)
    trainer = cls(model=model, args=args, train_dataset=data, callbacks=[Count()])
    if l2l:
        attach_l2l_state(trainer, runtime, dtype=torch.float32)
    trainer.train()
    return trainer, Count.optimizer_steps


def _adapters(model):
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if "lora_" in name and p.requires_grad
    }


@pytest.mark.parametrize("k", [1, 2])
def test_one_optimizer_step_per_window_and_same_adapters_as_accumulation(
    streamed, tmp_path, k  # noqa: F811
):
    import torch
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    before = _adapters(model)

    _ga_trainer, ga_steps = _train(Trainer, model, runtime, tmp_path, l2l=False, k=k)
    ga = _adapters(model)

    harness.restore_adapters(model, start)
    l2l_trainer, l2l_steps = _train(Trainer, model, runtime, tmp_path, l2l=True, k=k)
    l2l = _adapters(model)

    assert ga_steps == l2l_steps == 4 // k
    assert l2l_trainer.state.global_step == 4 // k
    assert ga.keys() == l2l.keys()
    for name in ga:
        assert torch.allclose(ga[name], l2l[name], atol=1e-6, rtol=1e-5), name
    assert any(not torch.equal(l2l[n], before[n]) for n in l2l)


def _same_adapters_as_accumulation(streamed, tmp_path, lengths=None, **targs):  # noqa: F811
    import torch
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    _train(Trainer, model, runtime, tmp_path, l2l=False, k=2, lengths=lengths, **targs)
    ga = _adapters(model)
    harness.restore_adapters(model, start)
    _train(Trainer, model, runtime, tmp_path, l2l=True, k=2, lengths=lengths, **targs)
    l2l = _adapters(model)
    for name in ga:
        assert torch.allclose(ga[name], l2l[name], atol=1e-6, rtol=1e-5), name


def test_bf16_autocast_of_the_trainer_reaches_the_l2l_step(streamed, tmp_path):  # noqa: F811
    _same_adapters_as_accumulation(streamed, tmp_path, bf16=True)


def test_neftune_noise_sequence_is_the_one_accumulation_draws(streamed, tmp_path):  # noqa: F811
    _same_adapters_as_accumulation(streamed, tmp_path, neftune_noise_alpha=5.0)


@pytest.fixture()
def dropout_model(tmp_path):
    """The ``streamed`` model with ``lora.dropout`` 0.05, closed after the test."""
    from tests.test_l2l_probe_harness import build_streamed

    model, runtime = build_streamed(tmp_path, lora_dropout=0.05)
    yield model, runtime
    runtime.close()


def _set_attention_dropout(model, value, *, eager=False):
    """``eager`` switches the attention to the eager implementation, whose dropout is replayable."""
    for module in model.modules():
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = value
        config = getattr(module, "config", None)
        if eager and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def _dropout_run_matches_accumulation(model_and_runtime, tmp_path, *, lengths=None, k=2):
    """(accumulation adapters, L2L adapters, equal) from one common starting point."""
    import torch

    harness = _load("l2l_schedule")
    model, runtime = model_and_runtime
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    from transformers import Trainer

    base = Trainer
    _train(base, model, runtime, tmp_path, l2l=False, k=k, lengths=lengths)
    ga = _adapters(model)
    harness.restore_adapters(model, start)
    _train(base, model, runtime, tmp_path, l2l=True, k=k, lengths=lengths)
    l2l = _adapters(model)
    equal = all(torch.allclose(ga[n], l2l[n], atol=1e-6, rtol=1e-5) for n in ga)
    return ga, l2l, equal, start


def test_trainer_with_lora_dropout_matches_accumulation_at_k2(dropout_model, tmp_path):
    import torch

    ga, l2l, same, start = _dropout_run_matches_accumulation(dropout_model, tmp_path)
    assert same
    assert any(not torch.equal(l2l[n], start[n]) for n in l2l)


def test_lora_dropout_changes_the_adapters_the_trainer_trains(streamed, dropout_model, tmp_path):  # noqa: F811
    """Control: the equality above is not vacuous, dropout 0.05 moves the adapters."""
    import torch
    from transformers import Trainer

    results = []
    for model, runtime in (streamed, dropout_model):
        harness = _load("l2l_schedule")
        harness.make_non_vacuous(model)
        _train(Trainer, model, runtime, tmp_path, l2l=False, k=2)
        results.append(_adapters(model))
    plain, dropped = results
    assert any(not torch.allclose(plain[n], dropped[n], atol=1e-6, rtol=1e-5) for n in plain)


def test_eager_attention_dropout_alone_matches_accumulation_at_k2(streamed, tmp_path):  # noqa: F811
    import torch

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    from transformers import Trainer

    base = Trainer

    _set_attention_dropout(model, 0.0, eager=True)
    _train(base, model, runtime, tmp_path, l2l=False, k=2)
    ga_without = _adapters(model)

    _set_attention_dropout(model, 0.1, eager=True)
    harness.restore_adapters(model, start)
    _train(base, model, runtime, tmp_path, l2l=False, k=2)
    ga = _adapters(model)
    harness.restore_adapters(model, start)
    _train(base, model, runtime, tmp_path, l2l=True, k=2)
    l2l = _adapters(model)

    assert all(torch.allclose(ga[n], l2l[n], atol=1e-6, rtol=1e-5) for n in ga)
    # control: from the same start, attention dropout alone changed the result
    assert any(not torch.allclose(ga[n], ga_without[n], atol=1e-6, rtol=1e-5) for n in ga)


def test_has_dropout_sees_lora_attention_and_config_dropout(dropout_model, streamed):  # noqa: F811
    import types

    from soup_cli.trainer.l2l_trainer import _has_dropout

    assert _has_dropout(dropout_model[0])
    plain = streamed[0]
    assert not _has_dropout(plain)
    _set_attention_dropout(plain, 0.1)
    assert _has_dropout(plain)
    _set_attention_dropout(plain, 0.0)
    config_only = types.SimpleNamespace(
        config=types.SimpleNamespace(attention_dropout=0.1), modules=lambda: []
    )
    assert _has_dropout(config_only)


def test_neftune_with_dropout_is_refused(dropout_model, tmp_path):
    from transformers import Trainer

    model, runtime = dropout_model
    _load("l2l_schedule").make_non_vacuous(model)
    with pytest.raises(ValueError, match=r"(?i)neftune.*dropout"):
        _train(Trainer, model, runtime, tmp_path, l2l=True, k=2, neftune_noise_alpha=5.0)


def test_mixed_length_windows_with_lora_dropout_are_refused(dropout_model, tmp_path):
    from transformers import Trainer

    model, runtime = dropout_model
    _load("l2l_schedule").make_non_vacuous(model)
    with pytest.raises(ValueError, match=r"(?i)dropout.*(length|padd)"):
        _train(Trainer, model, runtime, tmp_path, l2l=True, k=2, lengths=[5, 8, 3, 6])


def test_sdpa_attention_dropout_is_refused_before_the_first_step(streamed, tmp_path):  # noqa: F811
    from transformers import Trainer

    model, runtime = streamed
    _load("l2l_schedule").make_non_vacuous(model)
    _set_attention_dropout(model, 0.1)
    with pytest.raises(ValueError, match=r"(?i)scaled_dot_product_attention"):
        _train(Trainer, model, runtime, tmp_path, l2l=True, k=2)


def test_mixed_length_windows_with_attention_dropout_only_are_refused(tmp_path, streamed):  # noqa: F811
    from transformers import Trainer

    model, runtime = streamed
    _load("l2l_schedule").make_non_vacuous(model)
    _set_attention_dropout(model, 0.1)
    with pytest.raises(ValueError, match=r"(?i)dropout.*(length|padd)"):
        _train(Trainer, model, runtime, tmp_path, l2l=True, k=2, lengths=[5, 8, 3, 6])


def test_activation_store_is_refused_before_it_is_allocated(streamed, tmp_path, monkeypatch):  # noqa: F811
    from transformers import Trainer

    from soup_cli.utils import l2l_activations

    def forbidden(*args, **kwargs):
        raise AssertionError("the store was allocated")

    monkeypatch.setattr(l2l_activations, "ActivationStore", forbidden)
    monkeypatch.setattr("soup_cli.utils.layer_stream.free_ram_bytes", lambda: 1000)
    model, runtime = streamed
    _load("l2l_schedule").make_non_vacuous(model)
    with pytest.raises(ValueError, match=r"activation.*gradient_accumulation_steps"):
        _train(Trainer, model, runtime, tmp_path, l2l=True, k=2)


def test_activation_budget_accepts_what_fits_and_skips_without_psutil():
    from soup_cli.trainer.l2l_trainer import check_activation_budget

    kw = dict(n_layers=2, k=3, chunk_bytes=100, pinned=False)
    check_activation_budget(free_ram=1000, **kw)
    check_activation_budget(free_ram=None, **{**kw, "chunk_bytes": 10**15})
    with pytest.raises(ValueError, match="activation"):
        check_activation_budget(free_ram=1000, **{**kw, "k": 4, "chunk_bytes": 100})


def test_activation_budget_charges_pinned_chunks_at_the_next_power_of_two():
    from soup_cli.trainer.l2l_trainer import check_activation_budget

    mib, gib = 2**20, 2**30
    # bf16, 80 layers, hidden 8192, seq 1088, k=16: a 17 MiB chunk occupies a 32 MiB
    # pinned block, so the store needs 40 GiB, not the 21.25 GiB of its logical bytes.
    kw = dict(free_ram=32 * gib, n_layers=80, k=16, chunk_bytes=17 * mib)
    check_activation_budget(pinned=False, **kw)
    with pytest.raises(ValueError, match="pinned"):
        check_activation_budget(pinned=True, **kw)


def test_neftune_with_mixed_length_windows_is_refused(streamed, tmp_path):  # noqa: F811
    from transformers import Trainer

    model, runtime = streamed
    _load("l2l_schedule").make_non_vacuous(model)
    with pytest.raises(ValueError, match=r"(?i)neftune.*(length|padd)"):
        _train(
            Trainer, model, runtime, tmp_path, l2l=True, k=2,
            lengths=[5, 8, 3, 6], neftune_noise_alpha=5.0,
        )


def test_mixed_length_windows_without_neftune_still_train(streamed, tmp_path):  # noqa: F811
    _same_adapters_as_accumulation(streamed, tmp_path, lengths=[5, 8, 3, 6])


def test_counters_reach_the_training_log(streamed, tmp_path):  # noqa: F811
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    trainer, _ = _train(Trainer, model, runtime, tmp_path, l2l=True, k=2)
    logged = [row for row in trainer.state.log_history if "loss" in row]
    assert logged
    last = logged[-1]
    assert last["l2l_layer_loads"] > 0
    assert last["l2l_large_loads"] >= 0
    assert last["l2l_weight_restores"] == 0
    assert last["l2l_steps"] == 2


def test_windows_with_rows_of_different_lengths_match_accumulation(streamed, tmp_path):  # noqa: F811
    import torch
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    lengths = [5, 8, 3, 6]

    _train(Trainer, model, runtime, tmp_path, l2l=False, k=2, lengths=lengths)
    ga = _adapters(model)
    harness.restore_adapters(model, start)
    _train(Trainer, model, runtime, tmp_path, l2l=True, k=2, lengths=lengths)
    l2l = _adapters(model)

    for name in ga:
        diff = (ga[name] - l2l[name]).abs().max().item()
        assert torch.allclose(ga[name], l2l[name], atol=2e-5, rtol=1e-4), (name, diff)


def test_left_padded_rows_are_refused():
    import torch

    from soup_cli.trainer.l2l_trainer import _window_tensors

    window = [
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[0, 0, 1, 1]]),
        }
    ]
    with pytest.raises(ValueError, match="right-padded"):
        _window_tensors(window, "cpu")


def test_without_attached_state_the_subclass_is_the_base_trainer(streamed, tmp_path):  # noqa: F811
    from transformers import Trainer

    from soup_cli.trainer.l2l_trainer import make_l2l_trainer_class

    cls = make_l2l_trainer_class(Trainer)
    assert issubclass(cls, Trainer)
    assert make_l2l_trainer_class(Trainer) is cls
    assert cls._soup_l2l is None


# --------------------------------------------------------------------------
# the production path: SFTTrainerWrapper -> TRL SFTTrainer
# --------------------------------------------------------------------------
def _wrapper(tmp_path, monkeypatch, **training):
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_v07201 import _tiny_llama_dir, _write_tiny_tokenizer

    weights = _tiny_llama_dir(tmp_path, n_layers=2)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    # HF picks mps on an Apple GPU box while the streamed model sits on cpu
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.chdir(tmp_path)
    fields = {
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "quantization": "none",
        "stream_layers": True,
        "l2l": True,
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "gradient_checkpointing": True,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }
    fields.update(training)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": fields,
                "output": str(tmp_path / "out"),
            }
        )
    )
    dataset = {
        "train": [
            {
                "messages": [
                    {"role": "user", "content": "hi" * (i + 1)},
                    {"role": "assistant", "content": "hello world"},
                ]
            }
            for i in range(4)
        ]
    }
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup(dataset)
    return wrapper


def test_sft_wrapper_builds_the_l2l_trainer_and_logs_counters(tmp_path, monkeypatch):
    wrapper = _wrapper(tmp_path, monkeypatch)
    assert getattr(wrapper.trainer, "soup_l2l", False)
    wrapper.trainer.train()
    rows = [row for row in wrapper.trainer.state.log_history if "loss" in row]
    assert wrapper.trainer.state.global_step == 2
    assert rows[-1]["l2l_steps"] == 2
    assert rows[-1]["l2l_layer_loads"] > 0


def test_sft_wrapper_without_l2l_builds_the_ordinary_trainer(tmp_path, monkeypatch):
    wrapper = _wrapper(tmp_path, monkeypatch, l2l=False)
    assert not getattr(wrapper.trainer, "soup_l2l", False)


def test_sft_wrapper_refuses_an_unsupported_model_at_launch(tmp_path, monkeypatch):
    from soup_cli.trainer import l2l_trainer

    def refuse(*args, **kwargs):
        raise ValueError("training.l2l does not support model_type 'qwen2'")

    monkeypatch.setattr(l2l_trainer, "refuse_unsupported_l2l", refuse)
    with pytest.raises(ValueError, match="qwen2"):
        _wrapper(tmp_path, monkeypatch)
