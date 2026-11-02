"""Ticket A6-03: SimPO on L2L (CPU).

SimPO is TRL's ``CPOTrainer`` with ``loss_type="simpo"``. The reference is that trainer under
gradient accumulation, built through the real ``SimPOTrainerWrapper.setup()``. The L2L run is
the same wrapper with ``l2l: true``. Both start from the same adapters and the same data
order; the LoRA gradient of every optimizer step and the logged loss must be equal with
``torch.equal``.

SimPO has no reference model, so the layers are walked only by the policy's
``[chosen; rejected]`` micro-batch. The per-row log-probability is the mean over the response
tokens, and the NLL term runs over the response tokens of the chosen side only (the prompt is
masked), unlike ORPO's NLL.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_l2l_orpo import (
    UNEVEN_ROWS,
    UNIFORM_ROWS,
    _assert_equal_runs,
    _collated,
    _l2l_yaml,
    _train,
)
from tests.test_l2l_probe_harness import _load
from tests.test_v07204 import _stream_cfg, _tiny_llama_dir, _write_tiny_tokenizer

ROOT = Path(__file__).resolve().parent.parent

#: Soup's SimPO knobs, set away from their defaults so that neither term vanishes.
GAMMA = 0.7
CPO_ALPHA = 0.6


def _wrapper(tmp_path, monkeypatch, name, *, l2l, k, rows, batch_size=2, **training):
    """A streamed SimPO wrapper through the real ``setup()``, on CPU in float32."""
    from soup_cli.trainer.simpo import SimPOTrainerWrapper

    root = tmp_path / name
    root.mkdir()
    weights, _model, _config = _tiny_llama_dir(root, n_layers=3, tie=False)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(root / "cache"))
    # TrainingArguments would pick MPS or CUDA by itself; the model here is on the CPU.
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    # CPOTrainer, unlike ORPOTrainer, does not move its loss to args.device: on an Apple GPU
    # box HF would pick mps for the loss accumulator while the model sits on the CPU.
    monkeypatch.setattr("transformers.training_args.is_torch_mps_available", lambda *a, **k: False)
    monkeypatch.chdir(root)
    options = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": k,
        "seed": 1,
        "lr": 0.01,
        "optimizer": "sgd",
        "scheduler": "constant",
        "warmup_ratio": 0.0,
        "simpo_gamma": GAMMA,
        "cpo_alpha": CPO_ALPHA,
    }
    if l2l:
        options["l2l"] = True
    options.update(training)
    cfg = _stream_cfg(weights, root / "out", task="simpo", **options)
    wrapper = SimPOTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": list(rows)})
    return wrapper


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


# --------------------------------------------------------------------------
# losses and gradients equal gradient accumulation
# --------------------------------------------------------------------------
# 10 rows at batch 2 are 5 micro-batches: k=2 gives windows of 2, 2, 1; k=4 gives 4, 1
@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_simpo_l2l_equals_accumulation_for_pairs_of_one_length(tmp_path, monkeypatch, k, windows):
    _ga, l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=k, rows=UNIFORM_ROWS)
    _assert_equal_runs(ga_result, l2l_result, windows)
    assert l2l.trainer._soup_l2l.steps == windows


@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_simpo_l2l_equals_accumulation_for_pairs_of_different_lengths(
    tmp_path, monkeypatch, k, windows
):
    _ga, l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=k, rows=UNEVEN_ROWS)
    _assert_equal_runs(ga_result, l2l_result, windows)
    # the micro-batches really differ in shape, so the store and the calls are per shape
    assert len(l2l.trainer._soup_l2l.calls_by_shape) > 1


def test_simpo_l2l_equals_accumulation_for_a_batch_of_one_pair(tmp_path, monkeypatch):
    _ga, _l2l, ga_result, l2l_result = _pair(
        tmp_path, monkeypatch, k=2, rows=UNEVEN_ROWS[:5], batch_size=1
    )
    _assert_equal_runs(ga_result, l2l_result, 3)


def test_the_gamma_and_nll_terms_are_part_of_the_compared_loss(tmp_path, monkeypatch):
    """A changed gamma or cpo_alpha changes the L2L loss: neither term is vacuous."""
    (tmp_path / "base").mkdir()
    (tmp_path / "shifted").mkdir()
    base = _pair(tmp_path / "base", monkeypatch, k=2, rows=UNIFORM_ROWS)[3]
    shifted = _pair(
        tmp_path / "shifted", monkeypatch, k=2, rows=UNIFORM_ROWS, simpo_gamma=0.0, cpo_alpha=2.0
    )[3]
    assert [row["loss"] for row in base[1]] != [row["loss"] for row in shifted[1]]


def test_the_layer_load_counters_are_in_the_training_log(tmp_path, monkeypatch):
    _ga, _l2l, _ga_result, (_grads, logs) = _pair(tmp_path, monkeypatch, k=2, rows=UNIFORM_ROWS)
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

    base = resolve_trl_symbol("CPOTrainer", "trl.experimental.cpo")
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


def test_simpo_keeps_beta_at_the_cpo_default_and_the_cpo_alpha_positive(tmp_path, monkeypatch):
    wrapper = _wrapper(tmp_path, monkeypatch, "defaults", l2l=True, k=2, rows=UNIFORM_ROWS)
    try:
        args = wrapper.trainer.args
        assert args.beta == pytest.approx(0.1)
        assert args.loss_type == "simpo"
        assert args.cpo_alpha == pytest.approx(CPO_ALPHA) and args.cpo_alpha > 0
        assert args.simpo_gamma == pytest.approx(GAMMA)
    finally:
        wrapper._close_stream_runtime()


# --------------------------------------------------------------------------
# truncation: the refusal of #1208 still runs on the L2L path
# --------------------------------------------------------------------------
#: 80 answer tokens against 3 at max_length 64: TRL cuts both to 64 - 80 / 64 - 0, and the
#: short answer ends with no token, so its mean log-probability would be 0/0.
LOPSIDED_ROWS = [
    {"prompt": "hi", "chosen": " good" * 80, "rejected": " bad bad soup"},
] + list(UNIFORM_ROWS[:3])


@pytest.mark.parametrize("l2l", [False, True])
def test_a_pair_that_truncation_empties_is_refused_with_and_without_l2l(
    tmp_path, monkeypatch, l2l
):
    with pytest.raises(ValueError, match="SimPO: 1 train row.*zero completion tokens"):
        _wrapper(tmp_path, monkeypatch, "lopsided", l2l=l2l, k=2, rows=LOPSIDED_ROWS)


# --------------------------------------------------------------------------
# the loss itself, against the formula
# --------------------------------------------------------------------------
def _hand_loss(logits, labels, len_chosen, beta, gamma, cpo_alpha):
    """``-logsigmoid(beta*(p_c - p_r) - gamma)`` + ``cpo_alpha * NLL`` from the spec, in plain
    torch: p is the mean log-probability of the response tokens of a row; the NLL is the mean
    over the response tokens of the chosen rows."""
    import torch

    shifted = labels[:, 1:]
    logp = torch.log_softmax(logits[:, :-1].double(), dim=-1)
    mask = shifted != -100
    token_logp = logp.gather(-1, shifted.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
    row_logp = token_logp.sum(-1) / mask.sum(-1)
    margin = beta * (row_logp[:len_chosen] - row_logp[len_chosen:]) - gamma
    nll = -token_logp[:len_chosen].sum() / mask[:len_chosen].sum()
    return -torch.nn.functional.logsigmoid(margin).mean() + cpo_alpha * nll


def test_the_simpo_loss_matches_the_formula_on_random_logits():
    import torch

    from soup_cli.trainer._trl_compat import resolve_trl_symbol
    from soup_cli.trainer.l2l_simpo import simpo_loss

    torch.manual_seed(0)
    trainer_cls = resolve_trl_symbol("CPOTrainer", "trl.experimental.cpo")
    logits = torch.randn(4, 6, 11)
    # two pairs; rows differ in the length of the response, prompts are masked with -100
    labels = torch.tensor(
        [
            [-100, -100, 4, 5, 6, 7],
            [-100, 3, 4, 5, -100, -100],
            [-100, -100, 8, 9, -100, -100],
            [-100, -100, -100, 2, 3, 4],
        ]
    )
    got = simpo_loss(
        logits, labels, 2, beta=0.1, gamma=GAMMA, cpo_alpha=CPO_ALPHA, trainer_cls=trainer_cls
    )
    want = _hand_loss(logits, labels, 2, 0.1, GAMMA, CPO_ALPHA)
    assert torch.isfinite(got)
    assert got.item() == pytest.approx(want.item(), rel=1e-5)


def test_the_nll_term_ignores_the_prompt_unlike_orpo():
    import torch

    from soup_cli.trainer._trl_compat import resolve_trl_symbol
    from soup_cli.trainer.l2l_simpo import simpo_loss

    trainer_cls = resolve_trl_symbol("CPOTrainer", "trl.experimental.cpo")
    logits = torch.randn(2, 5, 9)
    labels = torch.tensor([[-100, -100, 4, 5, 6], [-100, -100, 7, 8, 2]])
    with_nll = simpo_loss(
        logits, labels, 1, beta=0.1, gamma=0.0, cpo_alpha=1.0, trainer_cls=trainer_cls
    )
    without = simpo_loss(
        logits, labels, 1, beta=0.1, gamma=0.0, cpo_alpha=0.0, trainer_cls=trainer_cls
    )
    hand_nll = _hand_loss(logits, labels, 1, 0.1, 0.0, 1.0) - _hand_loss(
        logits, labels, 1, 0.1, 0.0, 0.0
    )
    assert (with_nll - without).item() == pytest.approx(hand_nll.item(), rel=1e-5)


# --------------------------------------------------------------------------
# reading a micro-batch: chosen_* and rejected_* fields, not input_ids
# --------------------------------------------------------------------------
def _strategy():
    from soup_cli.trainer.l2l_simpo import simpo_window_loss

    return simpo_window_loss(beta=0.1, gamma=GAMMA, cpo_alpha=CPO_ALPHA, padding_value=3)


def test_the_window_reader_stacks_chosen_over_rejected_padded_on_the_right():
    import torch

    batch = _collated(chosen=[[1, 5, 6, 7], [1, 8]], rejected=[[1, 9], [1, 4, 5]])
    assert "input_ids" not in batch
    ids_list, _extras = _strategy().read_window([batch], "cpu", neftune=False, has_dropout=False)
    assert len(ids_list) == 1
    expected = torch.tensor(
        [[1, 5, 6, 7], [1, 8, 3, 3], [1, 9, 3, 3], [1, 4, 5, 3]]  # chosen rows, rejected rows
    )
    assert torch.equal(ids_list[0], expected)


def test_the_window_reader_gives_each_micro_batch_its_own_width():
    short = _collated(chosen=[[1, 5]], rejected=[[1, 9]])
    long = _collated(chosen=[[1, 5, 6, 7, 8]], rejected=[[1, 9]])
    ids_list, _ = _strategy().read_window([short, long], "cpu", neftune=False, has_dropout=False)
    assert [tuple(ids.shape) for ids in ids_list] == [(2, 2), (2, 5)]


def test_the_window_reader_refuses_left_padding():
    batch = _collated(chosen=[[1, 5, 6], [1, 8]], rejected=[[1, 9], [1, 4]])
    batch["chosen_attention_mask"][1] = batch["chosen_attention_mask"][1].flip(0)
    with pytest.raises(ValueError, match="right-padded"):
        _strategy().read_window([batch], "cpu", neftune=False, has_dropout=False)


def test_the_window_reader_keeps_the_mixed_shape_refusals_of_sft():
    window = [
        _collated(chosen=[[1, 5]], rejected=[[1, 9]]),
        _collated(chosen=[[1, 5, 6]], rejected=[[1, 9]]),
    ]
    with pytest.raises(ValueError, match="neftune_alpha"):
        _strategy().read_window(window, "cpu", neftune=True, has_dropout=False)
    with pytest.raises(ValueError, match="dropout"):
        _strategy().read_window(window, "cpu", neftune=False, has_dropout=True)


def test_simpo_divides_the_window_by_k_and_needs_no_item_count():
    assert _strategy().divisor(3, None) == 3


# --------------------------------------------------------------------------
# other CPO loss types are refused at config load, never swapped for simpo (the schema has
# no loss_type field, and the loader rejects unknown keys)
# --------------------------------------------------------------------------
def test_a_config_that_names_a_cpo_loss_type_is_refused_at_load():
    import yaml

    from soup_cli.config.loader import load_config_from_string

    text = yaml.safe_load(_l2l_yaml("simpo"))
    text["training"]["loss_type"] = "hinge"
    with pytest.raises(ValueError, match="unknown config key 'training.loss_type'"):
        load_config_from_string(yaml.safe_dump(text))


# --------------------------------------------------------------------------
# refusals at launch and in the schema
# --------------------------------------------------------------------------
def _bare_simpo_wrapper(model_type="llama"):
    from soup_cli.trainer.simpo import SimPOTrainerWrapper

    wrapper = SimPOTrainerWrapper.__new__(SimPOTrainerWrapper)
    wrapper.device = "cpu"
    wrapper.model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    wrapper._stream_runtime = object()
    return wrapper


def _training_args(n_gpu=1, fp16=False):
    return SimpleNamespace(n_gpu=n_gpu, fp16=fp16, gradient_accumulation_steps=2)


def test_an_unsupported_architecture_is_refused_before_a_trainer_is_built():
    wrapper = _bare_simpo_wrapper("qwen2")
    with pytest.raises(ValueError, match="model_type 'qwen2'"):
        wrapper._build_l2l_trainer(object, {}, _training_args())


def test_several_visible_gpus_are_refused_before_a_trainer_is_built():
    wrapper = _bare_simpo_wrapper()
    with pytest.raises(ValueError, match="single-GPU, but 2 GPUs"):
        wrapper._build_l2l_trainer(object, {}, _training_args(n_gpu=2))


def test_the_schema_accepts_simpo_with_l2l():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS

    assert "simpo" in L2L_SUPPORTED_TASKS
    cfg = load_config_from_string(_l2l_yaml("simpo"))
    assert cfg.task == "simpo" and cfg.training.l2l is True


def test_the_schema_still_refuses_the_tasks_that_have_no_window_loss_yet():
    from soup_cli.config.loader import load_config_from_string

    for task in ("dpo", "ipo"):
        with pytest.raises(ValueError, match=f"got task='{task}'"):
            load_config_from_string(_l2l_yaml(task))


def test_the_schema_refuses_simpo_with_l2l_when_streaming_is_off():
    from soup_cli.config.loader import load_config_from_string

    text = _l2l_yaml("simpo").replace("stream_layers: true", "stream_layers: false")
    with pytest.raises(ValueError, match="requires training.stream_layers"):
        load_config_from_string(text)


# --------------------------------------------------------------------------
# the short-training config and its synthetic pairs
# --------------------------------------------------------------------------
CONFIG = ROOT / "examples" / "configs" / "simpo_l2l_short.yaml"
PAIRS = ROOT / "examples" / "data" / "simpo_pairs_synthetic.jsonl"


def test_the_short_config_passes_the_schema_check():
    from soup_cli.config.loader import load_config

    cfg = load_config(CONFIG)
    assert cfg.task == "simpo"
    assert cfg.training.l2l is True and cfg.training.stream_layers is True
    assert cfg.data.train == "examples/data/simpo_pairs_synthetic.jsonl"
    assert cfg.training.epochs == 1
    assert cfg.training.cpo_alpha > 0


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
    line = next(line for line in readme.splitlines() if "simpo_pairs_synthetic.jsonl" in line)
    assert "synthetic" in line.lower()
