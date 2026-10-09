
"""Integration tests for the E2 runner lifecycle."""

import pytest

from soup_cli.config.loader import load_config
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.sft import SFTTrainerWrapper


@pytest.fixture
def e2_wrapper(tmp_path):
    cfg = load_config("experiments/e2_medium_cached.yaml")
    cfg = cfg.model_copy(update={"output": str(tmp_path / "output")})

    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup(load_dataset(cfg.data))

    wrapper.trainer.args.max_steps = 2
    wrapper.trainer.args.save_strategy = "no"

    return wrapper


def test_e2_runner_cleared_after_success(e2_wrapper, monkeypatch):
    wrapper = e2_wrapper
    seen = []

    original_train = wrapper.trainer.train

    def observed_train(*args, **kwargs):
        seen.append(wrapper._e2_runner)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(wrapper.trainer, "train", observed_train)

    wrapper.train()

    assert len(seen) == 1
    assert seen[0] is not None
    assert wrapper._e2_runner is None


def test_e2_runner_cleared_after_failure(e2_wrapper, monkeypatch):
    wrapper = e2_wrapper

    def failing_train(*args, **kwargs):
        assert wrapper._e2_runner is not None
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(wrapper.trainer, "train", failing_train)

    with pytest.raises(RuntimeError, match="simulated training failure"):
        wrapper.train()

    assert wrapper._e2_runner is None

