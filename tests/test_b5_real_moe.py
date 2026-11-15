"""B5 ticket 03: the run harness on the tiny MoE, on CPU, and the soup.yaml of the real run.

The harness is ``benchmarks/harness/b5_real_moe.py``. Here it trains the tiny native DeepSeek-V3
of ``b13_moe_fixture`` through the real ``per_expert`` runtime and writes its journal, which
``b5_judge`` reads. The weights are random, so the journal is signed ``synthetic``. NF4, CUDA and
the real Moonlight checkpoint are not covered; they are measured on the rented card.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pytest

from benchmarks.harness import b5_judge, b5_real_moe
from benchmarks.harness import b13_moe_fixture as fixture
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

PROTOCOL = Path(__file__).resolve().parent.parent / "benchmarks/protocols/b5-moonlight-20261109"
TINY_CONFIG = """\
base: tiny-deepseek-v3-moe
task: sft
backend: transformers
data:
  train: data.json
  format: plaintext
  max_length: 64
training:
  quantization: none
  epochs: 1
  batch_size: 1
  lr: 1.0e-3
  stream_layers: true
  stream_source: disk
  stream_expert_layout: per_expert
  lora:
    r: 2
    alpha: 4
    dropout: 0.0
    target_modules: [q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj]
"""


def _tiny_args(root: Path, **overrides: Any) -> argparse.Namespace:
    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(root / "src", config)
    (root / "soup.yaml").write_text(TINY_CONFIG, encoding="utf-8")
    values = dict(
        config=str(root / "soup.yaml"), source=str(weights), cache=str(root / "cache"),
        output=str(root / "out"), data=None, kind="synthetic", model="tiny-deepseek-v3-moe",
        revision=None, generator="b13_moe_fixture.moe_weights_dir seed=61", steps=4,
        device="cpu", tier="ram", shard_threads=1,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _tiny_batches(steps: int = 4) -> List[Dict[str, Any]]:
    config = fixture.moe_config()
    return [fixture.moe_batch(config, seed=step, rows=1, seq=64) for step in range(steps)]


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    root = tmp_path_factory.mktemp("b5-real-moe-tiny")
    b5_real_moe.run(_tiny_args(root), batches=_tiny_batches())
    return {"rows": b5_judge.read_journal(root / "out" / "journal.jsonl"), "root": root}


def test_the_judge_passes_a_per_expert_run_of_the_harness(tiny_run: Dict[str, Any]) -> None:
    result = b5_judge.judge([tiny_run["rows"]])

    assert result["passed"] is True, result["failures"]
    assert result["kind"] == "synthetic"
    assert result["steps"] == 4


def _events(rows: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row["event"] == name]


def test_every_step_records_tokens_expert_loads_and_peak_memory_fields(
    tiny_run: Dict[str, Any],
) -> None:
    steps = _events(tiny_run["rows"], "step")

    assert [row["tokens"] for row in steps] == [64] * 4
    for row in steps:
        assert row["expert_fetches"] > 0
        assert row["expert_bytes_read"] > 0
        assert row["tokens_per_second"] > 0
        assert "peak_allocated" in row and "peak_reserved" in row  # null on CPU


def test_the_journal_names_the_setup_the_acceptance_rule_reads(tiny_run: Dict[str, Any]) -> None:
    (environment,) = _events(tiny_run["rows"], "environment")

    assert environment["expert_layout"] == "per_expert"
    assert environment["double_quant"] is False
    assert environment["trust_remote_code"] is False
    assert environment["quant"] == "none"  # the tiny run is not NF4
    assert environment["versions"]["torch"]
    assert environment["lora_targets"] == ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa",
                                           "kv_b_proj", "o_proj"]
    assert [row["phase"] for row in _events(tiny_run["rows"], "phase_end")] == [
        "preparation", "training"]
    assert _events(tiny_run["rows"], "complete")


def test_an_existing_output_folder_is_refused(tmp_path: Path) -> None:
    args = _tiny_args(tmp_path)
    Path(args.output).mkdir(parents=True)

    with pytest.raises(FileExistsError):
        b5_real_moe.run(args, batches=_tiny_batches())


def test_the_real_run_config_is_accepted_by_the_schema_and_matches_the_rule() -> None:
    import yaml

    from benchmarks.harness import b5_acceptance
    from soup_cli.config.schema import SoupConfig

    config = SoupConfig.model_validate(
        yaml.safe_load((PROTOCOL / "soup.yaml").read_text(encoding="utf-8")))

    assert config.base == b5_acceptance.MODEL
    assert config.data.max_length == b5_acceptance.TOKENS_PER_STEP
    assert config.training.stream_expert_layout == "per_expert"
    assert config.training.stream_source == "disk"
    assert config.training.quantization == "4bit"
    assert config.training.batch_size == 1
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.lora.target_modules == [
        "q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    assert not config.training.l2l


def test_the_real_run_data_has_a_text_for_every_step() -> None:
    import json

    texts = json.loads((PROTOCOL / "data.json").read_text(encoding="utf-8"))["train"]

    assert len(texts) >= 4
    assert all(len(text) > 2000 for text in texts)  # about 500 tokens at 4 characters each
