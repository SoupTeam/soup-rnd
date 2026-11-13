"""B1.4 ticket 04: the L2L run recipe, the effective k and the snapshot on disk.

Seam: ``soup_cli.utils.l2l_recipe`` (pure logic and one JSON file in the output directory).
"""

import json

import pytest


def _tcfg(**training):
    from soup_cli.config.schema import SoupConfig

    settings = {
        "stream_layers": True, "stream_source": "disk", "stream_expert_layout": "per_expert",
        "l2l": True, "quantization": "none", "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "lora": {"r": 4, "dropout": 0, "target_modules": ["q_proj"]},
    }
    settings.update(training)
    return SoupConfig.model_validate({
        "base": "example/model", "task": "sft",
        "data": {"train": "train.jsonl", "chat_template": "chatml"},
        "training": settings,
    }).training


def test_default_recipe_uses_the_accumulation_steps_as_effective_k():
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe

    recipe = resolve_l2l_recipe(_tcfg(gradient_accumulation_steps=16))
    assert (recipe.requested_k, recipe.effective_k) == (16, 16)
    assert (recipe.activation_codec, recipe.auto_k) == ("none", False)


def test_unavailable_codec_is_refused_at_launch_by_name():
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe

    with pytest.raises(ValueError, match="fp8_e4m3.*not available"):
        resolve_l2l_recipe(_tcfg(l2l_activation_codec="fp8_e4m3"))


@pytest.mark.parametrize("effective_k", [None, 0, 5])
def test_auto_k_requires_a_resolved_window_within_the_requested_limit(effective_k):
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe

    with pytest.raises(ValueError, match="l2l_auto_k.*requires a resolved"):
        resolve_l2l_recipe(_tcfg(l2l_auto_k=True), effective_k=effective_k)


def test_recipe_is_written_and_read_back(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_recipe import RECIPE_NAME, load_recipe, resolve_l2l_recipe, write_recipe

    monkeypatch.chdir(tmp_path)
    recipe = resolve_l2l_recipe(_tcfg(gradient_accumulation_steps=8))
    path = write_recipe(tmp_path / "out", recipe)
    assert path.endswith(RECIPE_NAME)
    saved = json.loads((tmp_path / "out" / RECIPE_NAME).read_text(encoding="utf-8"))
    assert saved["effective_k"] == 8
    assert saved["activation_codec"] == "none"
    assert load_recipe(tmp_path / "out") == recipe


def test_load_recipe_without_a_file_is_none(tmp_path):
    from soup_cli.utils.l2l_recipe import load_recipe

    assert load_recipe(tmp_path) is None


def test_resume_with_the_same_recipe_or_no_snapshot_is_accepted(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe, validate_resume, write_recipe

    monkeypatch.chdir(tmp_path)
    recipe = resolve_l2l_recipe(_tcfg())
    validate_resume(tmp_path / "out", recipe)
    write_recipe(tmp_path / "out", recipe)
    validate_resume(tmp_path / "out", recipe)


def test_resume_with_another_effective_k_is_refused(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe, validate_resume, write_recipe

    monkeypatch.chdir(tmp_path)
    write_recipe(tmp_path / "out", resolve_l2l_recipe(_tcfg(gradient_accumulation_steps=4)))
    with pytest.raises(ValueError, match="effective_k.*4.*8"):
        validate_resume(
            tmp_path / "out", resolve_l2l_recipe(_tcfg(gradient_accumulation_steps=8))
        )


def test_resume_with_another_codec_is_refused(tmp_path, monkeypatch):
    from dataclasses import replace

    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe, validate_resume, write_recipe

    monkeypatch.chdir(tmp_path)
    recipe = resolve_l2l_recipe(_tcfg())
    write_recipe(tmp_path / "out", replace(recipe, activation_codec="fp8_e4m3"))
    with pytest.raises(ValueError, match="activation_codec.*fp8_e4m3.*none"):
        validate_resume(tmp_path / "out", recipe)


# --------------------------------------------------------------------------
# SFTTrainerWrapper: the effective k reaches every consumer
# --------------------------------------------------------------------------
def _force_effective_k(monkeypatch, k):
    """Make the recipe's effective k differ from gradient_accumulation_steps.

    Exercise the existing dense Llama path and its C3 reservation independently
    of the automatic selector, which only supports DeepSeek.
    """
    from dataclasses import replace

    from soup_cli.utils import l2l_recipe

    real = l2l_recipe.resolve_l2l_recipe
    monkeypatch.setattr(
        l2l_recipe, "resolve_l2l_recipe", lambda tcfg: replace(real(tcfg), effective_k=k)
    )


def test_wrapper_gives_the_effective_k_to_hf_the_window_the_store_and_the_log(
    tmp_path, monkeypatch
):
    from transformers import TrainerCallback

    from tests.test_l2l_trainer_mode import _wrapper

    _force_effective_k(monkeypatch, 1)
    wrapper = _wrapper(tmp_path, monkeypatch, gradient_accumulation_steps=2)
    assert wrapper.trainer.args.gradient_accumulation_steps == 1
    capacities = []

    class ObserveStore(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            capacities.append(len(wrapper.trainer._soup_l2l.store.mb_shapes))

    wrapper.trainer.add_callback(ObserveStore())
    wrapper.trainer.train()
    rows = [row for row in wrapper.trainer.state.log_history if "loss" in row]
    assert wrapper.trainer.state.global_step == 4
    assert capacities == [1, 1, 1, 1]
    assert {row["l2l_effective_k"] for row in rows} == {1}


def test_wrapper_gives_the_effective_k_to_the_c3_reservation(tmp_path, monkeypatch):
    from soup_cli.config.schema import SoupConfig
    from tests.test_v07200 import TestStreamingEndToEndSetup

    _force_effective_k(monkeypatch, 1)
    wrapper, dataset = TestStreamingEndToEndSetup()._wrapper(tmp_path, monkeypatch)
    config = wrapper.config.model_dump()
    config["training"].update(
        l2l=True, gradient_accumulation_steps=2, stream_source="disk", stream_pin=False,
        stream_disk_kind="nvme",
        stream_partial_ram=dict(limit_bytes=10_000_000, reserved_bytes=100_000, cache_bytes=45_000),
    )
    wrapper.config = SoupConfig.model_validate(config)
    try:
        wrapper.setup(dataset)
        assert wrapper._c3_l2l_plan.max_microbatches == 1
    finally:
        wrapper._close_stream_runtime()


def test_wrapper_without_l2l_keeps_the_accumulation_steps(tmp_path, monkeypatch):
    from tests.test_l2l_trainer_mode import _wrapper

    wrapper = _wrapper(tmp_path, monkeypatch, l2l=False, gradient_accumulation_steps=2)
    assert wrapper.trainer.args.gradient_accumulation_steps == 2
    assert wrapper._l2l_recipe is None


def test_wrapper_writes_the_recipe_before_training(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_recipe import load_recipe
    from tests.test_l2l_trainer_mode import _wrapper

    wrapper = _wrapper(tmp_path, monkeypatch)
    wrapper.train()
    saved = load_recipe(wrapper._output_dir)
    assert (saved.requested_k, saved.effective_k) == (2, 2)
    assert saved.activation_codec == "none"


@pytest.mark.parametrize("change, name", [
    ({"l2l_activation_codec": "fp8_e4m3"}, "fp8_e4m3.*not available"),
    ({"l2l_auto_k": True}, "l2l_auto_k.*requires DeepSeek"),
])
def test_wrapper_refuses_unavailable_options_before_loading_the_model(
    tmp_path, monkeypatch, change, name
):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from tests.test_l2l_trainer_mode import _wrapper

    def loaded(*args, **kwargs):
        raise AssertionError("the model load started before the refusal")

    monkeypatch.setattr(StreamingSetupMixin, "_setup_streaming_transformers", loaded)
    with pytest.raises(ValueError, match=name):
        _wrapper(tmp_path, monkeypatch, **change)


def test_wrapper_refuses_to_resume_with_another_effective_k(tmp_path, monkeypatch):
    from dataclasses import replace

    from soup_cli.utils.l2l_recipe import write_recipe
    from tests.test_l2l_trainer_mode import _wrapper

    wrapper = _wrapper(tmp_path, monkeypatch)
    write_recipe(wrapper._output_dir, replace(wrapper._l2l_recipe, effective_k=3))
    with pytest.raises(ValueError, match="cannot resume.*effective_k=3.*effective_k=2"):
        wrapper.train(resume_from_checkpoint=str(tmp_path / "out" / "checkpoint-1"))


@pytest.mark.parametrize(
    ("available", "expected_k"),
    [(215_000, 2), (8_000_000, 64), (120_000, None), (None, None)],
    ids=["shrinks-to-two", "keeps-64", "too-small", "unknown"],
)
def test_public_deepseek_auto_k_reserves_and_uses_one_window(
    tmp_path, monkeypatch, available, expected_k,
):
    from transformers import TrainerCallback

    from benchmarks.harness.b13_moe_fixture import moe_weights_dir
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import l2l_spill
    from soup_cli.utils.budgeted_activations import BudgetedActivationStore
    from soup_cli.utils.l2l_recipe import load_recipe
    from tests.test_l2l_probe_harness import hermetic_ram
    from tests.test_v07201 import _write_tiny_tokenizer

    weights = moe_weights_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    hermetic_ram(monkeypatch, tmp_path)
    monkeypatch.setattr(
        l2l_spill, "ram_budget",
        lambda: l2l_spill.RamBudget(None, available, None, {}, None),
    )
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.chdir(tmp_path)
    cfg = SoupConfig.model_validate({
        "base": str(weights), "output": str(tmp_path / "out"),
        "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
        "training": {
            "stream_layers": True, "stream_expert_layout": "per_expert", "stream_source": "disk",
            "stream_disk_kind": "nvme", "stream_pin": False, "quantization": "4bit",
            "bnb_4bit_use_double_quant": False,
            "l2l": True, "l2l_auto_k": True, "batch_size": 1, "gradient_accumulation_steps": 64,
            "epochs": 1, "logging_steps": 1, "save_steps": 1000,
            "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_a_proj"]},
        },
    })
    row_count = 64 if expected_k == 64 else 5
    dataset = {"train": [
        {"messages": [{"role": "user", "content": "hi " * (row % 5 + 1)},
                      {"role": "assistant", "content": "hello world"}]}
        for row in range(row_count)
    ]}
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    if available is None:
        with pytest.raises(ValueError, match="could not establish a safe host RAM budget"):
            wrapper.setup(dataset)
        assert wrapper._stream_runtime.source._closed
        return
    if available == 120_000:
        with pytest.raises(MemoryError, match="k=1 needs"):
            wrapper.setup(dataset)
        assert wrapper._stream_runtime.source._closed
        return

    assert expected_k is not None
    expected_steps = (row_count + expected_k - 1) // expected_k
    try:
        wrapper.setup(dataset)
        recipe = wrapper._l2l_recipe
        plan = wrapper._c3_l2l_plan
        assert recipe.requested_k == cfg.training.gradient_accumulation_steps == 64
        assert recipe.effective_k == plan.max_microbatches == expected_k
        assert plan.activations.limit_bytes == expected_k * 3 * 64 * 32 * 4
        assert plan.weights.reserved_bytes <= available
        assert wrapper.trainer.args.gradient_accumulation_steps == expected_k
        assert wrapper.trainer._soup_l2l.c3_plan is plan
        capacities = []

        class ObserveStore(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                store = wrapper.trainer._soup_l2l.store
                assert isinstance(store, BudgetedActivationStore)
                assert 0 < plan.activations.used_bytes <= plan.activations.limit_bytes
                capacities.append(store.k)

        wrapper.trainer.add_callback(ObserveStore())
        wrapper.train()
        assert wrapper.trainer.state.global_step == expected_steps
        assert capacities == [expected_k] * expected_steps
        assert plan.activations.used_bytes == 0
        assert load_recipe(wrapper._output_dir) == recipe
        rows = [row for row in wrapper.trainer.state.log_history if "loss" in row]
        assert {row["l2l_effective_k"] for row in rows} == {expected_k}
    finally:
        wrapper._close_stream_runtime()
