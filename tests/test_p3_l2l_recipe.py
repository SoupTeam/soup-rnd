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


def test_auto_k_is_refused_at_launch_until_the_planner_is_wired():
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe

    with pytest.raises(ValueError, match="l2l_auto_k.*not available"):
        resolve_l2l_recipe(_tcfg(l2l_auto_k=True))


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

    Nothing in the launch can do that yet (``l2l_auto_k`` is refused), and the point of the
    recipe is that every consumer reads the effective value, not the requested one.
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
    ({"l2l_activation_codec": "fp8_e4m3"}, "fp8_e4m3"),
    ({"l2l_auto_k": True}, "l2l_auto_k"),
])
def test_wrapper_refuses_unavailable_options_before_loading_the_model(
    tmp_path, monkeypatch, change, name
):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from tests.test_l2l_trainer_mode import _wrapper

    def loaded(*args, **kwargs):
        raise AssertionError("the model load started before the refusal")

    monkeypatch.setattr(StreamingSetupMixin, "_setup_streaming_transformers", loaded)
    with pytest.raises(ValueError, match=f"{name}.*not available"):
        _wrapper(tmp_path, monkeypatch, **change)


def test_wrapper_refuses_to_resume_with_another_effective_k(tmp_path, monkeypatch):
    from dataclasses import replace

    from soup_cli.utils.l2l_recipe import write_recipe
    from tests.test_l2l_trainer_mode import _wrapper

    wrapper = _wrapper(tmp_path, monkeypatch)
    write_recipe(wrapper._output_dir, replace(wrapper._l2l_recipe, effective_k=3))
    with pytest.raises(ValueError, match="cannot resume.*effective_k=3.*effective_k=2"):
        wrapper.train(resume_from_checkpoint=str(tmp_path / "out" / "checkpoint-1"))
