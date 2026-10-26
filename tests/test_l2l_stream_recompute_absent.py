"""A5 ticket 01: `training.stream_recompute` and `training.l2l`.

The mode comes from MakazhanAlpamys/Soup #2152 and is not in this repository. A config that
sets it is refused at load, with or without `l2l`, so tickets 03 and 04 cannot measure a
mixture of modes by accident. If a merge brings the field in, this test fails and the
decision on how it combines with `l2l` has to be made then.
"""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _yaml(**training):
    fields = {
        "batch_size": 1,
        "quantization": "none",
        "stream_layers": True,
        "gradient_accumulation_steps": 2,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
    }
    fields.update(training)
    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": "sft",
            "data": {"train": "data.jsonl"},
            "training": fields,
        }
    )


@pytest.mark.parametrize("value", ["on", "off", "auto"])
@pytest.mark.parametrize("l2l", [True, False])
def test_stream_recompute_is_refused_at_load(value, l2l):
    with pytest.raises(ValueError, match=r"unknown config key 'training\.stream_recompute'"):
        load_config_from_string(_yaml(l2l=l2l, stream_recompute=value))
