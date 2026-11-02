"""Ticket A6-02: ORPO on L2L (CPU).

The reference is TRL's own ``ORPOTrainer`` under gradient accumulation, built through the
real ``ORPOTrainerWrapper.setup()``. The L2L run is the same wrapper with ``l2l: true``.
Both start from the same adapters and the same data order; the LoRA gradient of every
optimizer step and the logged loss must be equal with ``torch.equal``.

ORPO has no reference model, so the layers are walked only by the policy's
``[chosen; rejected]`` micro-batch. TRL's NLL runs over the whole chosen sequence
(prompt included) and is divided by the token count of the micro-batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_l2l_probe_harness import _load
from tests.test_v07204 import _stream_cfg, _tiny_llama_dir, _write_tiny_tokenizer

ROOT = Path(__file__).resolve().parent.parent

#: Pairs of one length: no padding inside a micro-batch.
UNIFORM_ROWS = [
    {"prompt": "hi", "chosen": f" good answer {i}", "rejected": f" bad answer {i}"}
    for i in range(10)
]
#: Pairs of different lengths: chosen longer than rejected, and the other way round.
UNEVEN_ROWS = [
    {"prompt": "hi", "chosen": " good answer the cat sat on mat", "rejected": " bad"},
    {"prompt": "the cat", "chosen": " good", "rejected": " bad answer the cat sat on the mat"},
    {"prompt": "hello world yo", "chosen": " good answer", "rejected": " bad answer the soup"},
    {"prompt": "hi", "chosen": " good answer the soup", "rejected": " bad"},
    {"prompt": "the cat sat", "chosen": " good", "rejected": " bad answer"},
    {"prompt": "yo", "chosen": " good answer the cat", "rejected": " bad answer the cat sat on"},
    {"prompt": "hi", "chosen": " good", "rejected": " bad answer the"},
    {"prompt": "hello", "chosen": " good answer the cat sat", "rejected": " bad"},
    {"prompt": "soup", "chosen": " good answer", "rejected": " bad the cat sat on mat"},
    {"prompt": "the", "chosen": " good the", "rejected": " bad answer the cat"},
]


def _wrapper(tmp_path, monkeypatch, name, *, l2l, k, rows, batch_size=2, **training):
    """A streamed ORPO wrapper through the real ``setup()``, on CPU in float32."""
    from soup_cli.trainer.orpo import ORPOTrainerWrapper

    root = tmp_path / name
    root.mkdir()
    weights, _model, _config = _tiny_llama_dir(root, n_layers=3, tie=False)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(root / "cache"))
    # TrainingArguments would pick MPS or CUDA by itself; the model here is on the CPU.
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.chdir(root)
    options = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": k,
        "seed": 1,
        "lr": 0.01,
        "optimizer": "sgd",
        "scheduler": "constant",
        "warmup_ratio": 0.0,
        "orpo_beta": 0.3,
    }
    if l2l:
        options["l2l"] = True
    options.update(training)
    cfg = _stream_cfg(weights, root / "out", task="orpo", **options)
    wrapper = ORPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": list(rows)})
    return wrapper


def _train(wrapper):
    """Train through the trainer; return the LoRA gradient of every optimizer step."""
    from transformers import TrainerCallback

    grads = []

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append(
                {
                    name: p.grad.detach().clone()
                    for name, p in model.named_parameters()
                    if "lora_" in name and p.grad is not None
                }
            )

    wrapper.trainer.add_callback(Recorder())
    wrapper.trainer.train()
    logs = [row for row in wrapper.trainer.state.log_history if "loss" in row]
    return grads, logs


def _pair(tmp_path, monkeypatch, *, k, rows, **options):
    """The accumulation run and the L2L run from the same start; returns both results."""
    harness = _load("l2l_schedule")
    ga = _wrapper(tmp_path, monkeypatch, "ga", l2l=False, k=k, rows=rows, **options)
    l2l = _wrapper(tmp_path, monkeypatch, "l2l", l2l=True, k=k, rows=rows, **options)
    try:
        harness.make_non_vacuous(ga.model)
        start = harness.snapshot_adapters(ga.model)
        harness.restore_adapters(l2l.model, start)
        ga_result = _train(ga)
        l2l_result = _train(l2l)
        return ga, l2l, ga_result, l2l_result
    finally:
        for wrapper in (ga, l2l):
            wrapper._close_stream_runtime()


def _assert_equal_runs(ga_result, l2l_result, windows):
    import math

    import torch

    ga_grads, ga_logs = ga_result
    l2l_grads, l2l_logs = l2l_result
    assert len(ga_grads) == len(l2l_grads) == windows
    for gradient_ga, gradient_l2l in zip(ga_grads, l2l_grads):
        assert gradient_ga.keys() == gradient_l2l.keys()
        assert any(bool(g.abs().sum() > 0) for g in gradient_l2l.values())
        for name in gradient_ga:
            assert torch.isfinite(gradient_l2l[name]).all(), name
            assert torch.equal(gradient_ga[name], gradient_l2l[name]), name
    assert all(math.isfinite(row["loss"]) for row in l2l_logs)
    assert [row["loss"] for row in l2l_logs] == [row["loss"] for row in ga_logs]


# 10 rows at batch 2 are 5 micro-batches: k=2 gives windows of 2, 2, 1; k=4 gives 4, 1
@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_orpo_l2l_equals_accumulation_for_pairs_of_one_length(
    tmp_path, monkeypatch, k, windows
):
    _ga, l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=k, rows=UNIFORM_ROWS)
    _assert_equal_runs(ga_result, l2l_result, windows)
    assert l2l.trainer._soup_l2l.steps == windows


@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_orpo_l2l_equals_accumulation_for_pairs_of_different_lengths(
    tmp_path, monkeypatch, k, windows
):
    _ga, l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=k, rows=UNEVEN_ROWS)
    _assert_equal_runs(ga_result, l2l_result, windows)
    # the micro-batches really differ in shape, so the store and the calls are per shape
    assert len(l2l.trainer._soup_l2l.calls_by_shape) > 1


def test_orpo_l2l_equals_accumulation_for_a_batch_of_one_pair(tmp_path, monkeypatch):
    _ga, _l2l, ga_result, l2l_result = _pair(
        tmp_path, monkeypatch, k=2, rows=UNEVEN_ROWS[:5], batch_size=1
    )
    _assert_equal_runs(ga_result, l2l_result, 3)


def test_the_layer_load_counters_are_in_the_training_log(tmp_path, monkeypatch):
    _ga, _l2l, _ga_result, (_grads, logs) = _pair(
        tmp_path, monkeypatch, k=2, rows=UNIFORM_ROWS
    )
    assert [row["l2l_steps"] for row in logs] == [1, 2, 3]
    for key in ("l2l_layer_loads", "l2l_large_loads"):
        assert logs[0][key] > 0
        assert all(later[key] > earlier[key] for earlier, later in zip(logs, logs[1:]))
    assert "l2l_weight_restores" in logs[0]


# --------------------------------------------------------------------------
# which trainer class runs
# --------------------------------------------------------------------------
def test_the_trainer_is_the_l2l_subclass_only_with_l2l_true(tmp_path, monkeypatch):
    from soup_cli.trainer._trl_compat import resolve_trl_symbol

    base = resolve_trl_symbol("ORPOTrainer", "trl.experimental.orpo")
    plain = _wrapper(tmp_path, monkeypatch, "plain", l2l=False, k=2, rows=UNIFORM_ROWS)
    streamed = _wrapper(tmp_path, monkeypatch, "streamed", l2l=True, k=2, rows=UNIFORM_ROWS)
    try:
        assert type(plain.trainer) is base
        assert not hasattr(plain.trainer, "soup_l2l")
        assert type(streamed.trainer) is not base
        assert isinstance(streamed.trainer, base)
        assert streamed.trainer.soup_l2l is True
        assert streamed.trainer._soup_l2l.loss.divide_by == "micro_batches"
    finally:
        plain._close_stream_runtime()
        streamed._close_stream_runtime()


# --------------------------------------------------------------------------
# reading a micro-batch: chosen_* and rejected_* fields, not input_ids
# --------------------------------------------------------------------------
def _collated(chosen, rejected):
    """What TRL's ``DPODataCollatorWithPadding`` hands the trainer: no ``input_ids``."""
    import torch

    def rows(seqs, pad):
        width = max(len(s) for s in seqs)
        ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs])
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
        labels = torch.tensor([[-100] + s[1:] + [-100] * (width - len(s)) for s in seqs])
        return ids, mask, labels

    c_ids, c_mask, c_labels = rows(chosen, 3)
    r_ids, r_mask, r_labels = rows(rejected, 3)
    return {
        "prompt": ["a", "b"][: len(chosen)],
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
    }


def test_the_window_reader_stacks_chosen_over_rejected_padded_on_the_right():
    import torch

    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    batch = _collated(chosen=[[1, 5, 6, 7], [1, 8]], rejected=[[1, 9], [1, 4, 5]])
    assert "input_ids" not in batch
    ids_list, _extras = strategy.read_window(
        [batch], "cpu", neftune=False, has_dropout=False
    )
    assert len(ids_list) == 1
    expected = torch.tensor(
        [[1, 5, 6, 7], [1, 8, 3, 3], [1, 9, 3, 3], [1, 4, 5, 3]]  # chosen rows, rejected rows
    )
    assert torch.equal(ids_list[0], expected)


def test_the_window_reader_gives_each_micro_batch_its_own_width():
    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    short = _collated(chosen=[[1, 5]], rejected=[[1, 9]])
    long = _collated(chosen=[[1, 5, 6, 7, 8]], rejected=[[1, 9]])
    ids_list, _ = strategy.read_window([short, long], "cpu", neftune=False, has_dropout=False)
    assert [tuple(ids.shape) for ids in ids_list] == [(2, 2), (2, 5)]


def test_the_window_reader_refuses_left_padding():
    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    batch = _collated(chosen=[[1, 5, 6], [1, 8]], rejected=[[1, 9], [1, 4]])
    batch["chosen_attention_mask"][1] = batch["chosen_attention_mask"][1].flip(0)
    with pytest.raises(ValueError, match="right-padded"):
        strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)


def test_the_window_reader_keeps_the_mixed_shape_refusals_of_sft():
    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    window = [
        _collated(chosen=[[1, 5]], rejected=[[1, 9]]),
        _collated(chosen=[[1, 5, 6]], rejected=[[1, 9]]),
    ]
    with pytest.raises(ValueError, match="neftune_alpha"):
        strategy.read_window(window, "cpu", neftune=True, has_dropout=False)
    with pytest.raises(ValueError, match="dropout"):
        strategy.read_window(window, "cpu", neftune=False, has_dropout=True)


def test_orpo_divides_the_window_by_k_and_needs_no_item_count():
    from soup_cli.trainer.l2l_orpo import orpo_window_loss

    strategy = orpo_window_loss(beta=0.1, padding_value=3)
    assert strategy.divisor(3, None) == 3


# --------------------------------------------------------------------------
# refusals at launch and in the schema
# --------------------------------------------------------------------------
def _bare_orpo_wrapper(model_type="llama"):
    from soup_cli.trainer.orpo import ORPOTrainerWrapper

    wrapper = ORPOTrainerWrapper.__new__(ORPOTrainerWrapper)
    wrapper.device = "cpu"
    wrapper.model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    wrapper._stream_runtime = object()
    return wrapper


def _training_args(n_gpu=1, fp16=False):
    return SimpleNamespace(n_gpu=n_gpu, fp16=fp16, gradient_accumulation_steps=2)


def test_an_unsupported_architecture_is_refused_before_a_trainer_is_built():
    wrapper = _bare_orpo_wrapper("qwen2")
    with pytest.raises(ValueError, match="model_type 'qwen2'"):
        wrapper._build_l2l_trainer(object, {}, _training_args())


def test_several_visible_gpus_are_refused_before_a_trainer_is_built():
    wrapper = _bare_orpo_wrapper()
    with pytest.raises(ValueError, match="single-GPU, but 2 GPUs"):
        wrapper._build_l2l_trainer(object, {}, _training_args(n_gpu=2))


def _l2l_yaml(task):
    import yaml

    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": task,
            "data": {"train": "data.jsonl", "format": "dpo"},
            "training": {
                "batch_size": 1,
                "quantization": "none",
                "stream_layers": True,
                "l2l": True,
                "gradient_accumulation_steps": 2,
                "lora": {"r": 4, "alpha": 8},
            },
        }
    )


def test_the_schema_accepts_orpo_with_l2l():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS

    assert "orpo" in L2L_SUPPORTED_TASKS
    cfg = load_config_from_string(_l2l_yaml("orpo"))
    assert cfg.task == "orpo" and cfg.training.l2l is True


def test_the_schema_still_refuses_the_tasks_that_have_no_window_loss_yet():
    from soup_cli.config.loader import load_config_from_string

    for task in ("dpo", "simpo"):
        with pytest.raises(ValueError, match=f"got task='{task}'"):
            load_config_from_string(_l2l_yaml(task))


def test_the_schema_refuses_orpo_with_l2l_when_streaming_is_off():
    from soup_cli.config.loader import load_config_from_string

    text = _l2l_yaml("orpo").replace("stream_layers: true", "stream_layers: false")
    with pytest.raises(ValueError, match="requires training.stream_layers"):
        load_config_from_string(text)


# --------------------------------------------------------------------------
# the short-training config and its synthetic pairs
# --------------------------------------------------------------------------
CONFIG = ROOT / "examples" / "configs" / "orpo_l2l_short.yaml"
PAIRS = ROOT / "examples" / "data" / "orpo_pairs_synthetic.jsonl"


def test_the_short_config_passes_the_schema_check():
    from soup_cli.config.loader import load_config

    cfg = load_config(CONFIG)
    assert cfg.task == "orpo"
    assert cfg.training.l2l is True and cfg.training.stream_layers is True
    assert cfg.data.train == "examples/data/orpo_pairs_synthetic.jsonl"
    assert cfg.training.epochs == 1


def test_the_short_config_names_a_supported_architecture():
    import yaml

    text = CONFIG.read_text(encoding="utf-8")
    base = yaml.safe_load(text)["base"]
    assert "SmolLM2" in base or "Llama" in base or "llama" in base
    assert "synthetic" in text.lower()


def test_the_pairs_are_small_valid_and_labelled_synthetic():
    rows = [json.loads(line) for line in PAIRS.read_text(encoding="utf-8").splitlines() if line]
    assert 8 <= len(rows) <= 40
    for row in rows:
        assert set(row) == {"prompt", "chosen", "rejected"}
        assert all(isinstance(row[key], str) and row[key].strip() for key in row)
        assert row["chosen"] != row["rejected"]
    readme = (ROOT / "examples" / "data" / "README.md").read_text(encoding="utf-8")
    line = next(line for line in readme.splitlines() if "orpo_pairs_synthetic.jsonl" in line)
    assert "synthetic" in line.lower()
