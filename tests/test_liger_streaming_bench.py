"""The GPU plan's config overrides must not bypass Soup's compatibility gates."""

import pytest

from benchmarks.harness.liger_streaming_bench import build_config
from soup_cli.config.schema import SoupConfig
from tests.test_liger_streaming import _payload


def _local_config(tmp_path):
    payload = _payload()
    base = tmp_path / "snapshot"
    base.mkdir()
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    payload.update(base=str(base))
    payload["data"]["train"] = str(data)
    return SoupConfig.model_validate(payload)


@pytest.mark.parametrize("arm", ["baseline", "rms_norm", "swiglu", "rms_norm+swiglu"])
def test_benchmark_arms_revalidate_and_leave_original_config_unchanged(tmp_path, arm):
    original = _local_config(tmp_path)
    frozen = original.model_dump()
    config = build_config(original, arm=arm, output=str(tmp_path / "output"))
    assert original.model_dump() == frozen
    assert config.training.use_liger == (arm != "baseline")
    assert not config.training.fast_lora
    assert config.training.liger_components == (None if arm == "baseline" else arm.split("+"))
    assert config.data.val_split == 0
    assert config.training.save_steps == 1_000_000_000


def test_benchmark_refuses_fast_lora_instead_of_silently_disabling_it(tmp_path):
    payload = _local_config(tmp_path).model_dump()
    payload["training"].update(use_liger=False, fast_lora=True)
    with pytest.raises(ValueError, match="fast_lora=false"):
        build_config(SoupConfig.model_validate(payload), arm="rms_norm", output="unused")


def test_benchmark_requires_local_frozen_inputs(tmp_path):
    payload = _local_config(tmp_path).model_dump()
    payload["base"] = "remote/model"
    with pytest.raises(ValueError, match="snapshot"):
        build_config(SoupConfig.model_validate(payload), arm="baseline", output="unused")
