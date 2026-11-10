"""Resource-only native inference checks on approved remote hardware."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tests.test_p3_deepseek_quality_eval import _tokenizer as tokenizer_fixture
from tests.test_p3_deepseek_quality_eval import _twins as twins_fixture


@pytest.fixture(name="tokenizer")
def _tokenizer():
    return tokenizer_fixture.__wrapped__()


@pytest.fixture(name="twins")
def _twins(tmp_path, tokenizer):
    yield from twins_fixture.__wrapped__(tmp_path, tokenizer)


def test_resource_chunks_preserve_adapter_and_modes_without_targets(twins, tokenizer):
    from benchmarks.harness import p3_deepseek_quality_timing as timing
    from benchmarks.harness.p3_deepseek_quality_model import adapter_digest, trainable_parameters

    parameters = trainable_parameters(twins.model)
    before = adapter_digest(parameters)[0]
    modes = [(module, module.training) for _, module in timing.scoring._module_nodes(twins.model)]
    row = {"input_ids": [2, 3, 4, 5, 6, 7, 8, 9], "positions": [1, 2, 3, 4, 5, 6],
           "targets": object()}
    vocabulary = twins.model.config.vocab_size
    limit = timing.frozen._logits_bound(1, 2, vocabulary, 1)
    observations = timing.time_inputs(
        twins, tokenizer, [row], max_tokens_per_batch=8, deadline=time.monotonic() + 300,
        logits_memory_limit_bytes=limit, reduction_chunk_tokens=1, test_fixture=True,
    )
    assert observations[0]["position_chunk_sizes"] == [2, 2, 2]
    assert observations[0]["native_forward_calls"] == 3
    assert observations[0]["largest_native_logits_bytes"] == 2 * vocabulary * 4
    assert observations[0]["logits_estimated_peak_bytes"] <= limit
    assert adapter_digest(parameters)[0] == before
    assert all(parameter.grad is None for parameter in parameters.values())
    assert all(module.training == mode for module, mode in modes)
    if os.environ.get("P3_QUALITY_EVIDENCE"):
        root = Path(os.environ["P3_QUALITY_EVIDENCE"]) / "resource-timing"
        root.mkdir(parents=True, exist_ok=True)
        (root / "native-forward.json").write_text(json.dumps({
            "adapter_unchanged_sha256": before, "quality_outcomes_computed": False,
            "observations": observations,
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_expired_timing_deadline_refuses_and_restores_caller(twins, tokenizer):
    from benchmarks.harness import p3_deepseek_quality_timing as timing
    from benchmarks.harness.p3_deepseek_quality_model import adapter_digest, trainable_parameters

    before = adapter_digest(trainable_parameters(twins.model))[0]
    mode = twins.model.training
    with pytest.raises(ValueError, match="deadline"):
        timing.time_inputs(twins, tokenizer, [{"input_ids": [2, 3], "positions": [1]}],
                           max_tokens_per_batch=2, deadline=0, test_fixture=True)
    assert twins.model.training == mode
    assert adapter_digest(trainable_parameters(twins.model))[0] == before


@pytest.mark.parametrize("positions", [[2], [-1], [1, 1]])
def test_invalid_native_positions_refused_before_inference(twins, tokenizer, positions):
    from benchmarks.harness import p3_deepseek_quality_timing as timing

    with pytest.raises(ValueError, match="invalid resource input"):
        timing.time_inputs(twins, tokenizer, [{"input_ids": [2, 3], "positions": positions}],
                           max_tokens_per_batch=2, deadline=time.monotonic() + 60,
                           test_fixture=True)


def test_later_forward_failure_keeps_completed_and_failed_batch_evidence(
    twins, tokenizer, monkeypatch, tmp_path,
):
    from benchmarks.harness import p3_deepseek_quality_timing as timing

    original = timing.scoring._ScoringModel.__call__
    calls = 0

    def fail_second(model, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected forward OOM")
        return original(model, **kwargs)

    monkeypatch.setattr(timing.scoring._ScoringModel, "__call__", fail_second)
    observations = []
    rows = [
        {"input_ids": [2, 3], "positions": [1]},
        {"input_ids": [2, 3, 4], "positions": [1]},
    ]
    with pytest.raises(RuntimeError, match="injected forward OOM"):
        timing.time_inputs(
            twins, tokenizer, rows, max_tokens_per_batch=3,
            deadline=time.monotonic() + 300, test_fixture=True, observations=observations,
            evidence_prefix=tmp_path / "dev",
        )
    assert [row["status"] for row in observations] == ["COMPLETE", "FAILED"]
    assert [row["native_forward_calls"] for row in observations] == [1, 0]
    assert sorted(index for row in observations for index in row["input_indices"]) == [0, 1]
    assert observations[1]["failure"] == {
        "type": "RuntimeError", "message": "injected forward OOM",
    }
    assert all("process_rss_sampled_peak_bytes" in row["memory"] for row in observations)
    completed = json.loads((tmp_path / "dev-batch-0001.json").read_text())
    failed = json.loads((tmp_path / "dev-batch-0002.json").read_text())
    assert completed["admission_sufficient"] is False
    assert completed["observation"] == observations[0]
    assert failed["observation"]["failure"]["message"] == "injected forward OOM"
