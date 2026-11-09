"""B5 ticket 02: the journal recorder and the judge on a tiny MoE, on CPU.

The model is the tiny native DeepSeek-V3 of ``b13_moe_fixture``, run through the real
``per_expert`` streaming runtime (checkpoint sharded per expert, experts read back through the
expert pool, backward through the streamed experts). The weights are random, so the journal is
signed ``synthetic``. These tests show that what the recorder writes is what the judge reads,
including the digest of the adapter after it is saved and loaded again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from benchmarks.harness import b5_judge, b5_recorder
from benchmarks.harness import b13_moe_fixture as fixture
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

STEPS = 3


def _train_save_reload(root: Path, journal_path: Path, *, restore: bool = True) -> Dict[str, Any]:
    """Train a few steps with the recorder, save the adapter, load it into a fresh model."""
    import torch
    from peft import load_peft_weights, set_peft_model_state_dict

    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    first = fixture.build_fixture(root / "first")
    second = None
    try:
        recorder = b5_recorder.AdapterRecorder(journal_path)
        recorder.signature("synthetic", "tiny-deepseek-v3-moe",
                           generator="b13_moe_fixture.build_fixture seed=61")
        model = first.streamed
        params = b5_recorder.adapter_params(model)
        optimizer = torch.optim.AdamW(list(params.values()), lr=1e-3)
        recorder.start(params)
        for step in range(STEPS):
            batch = fixture.moe_batch(first.config, seed=step)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            recorder.gradients(params)
            optimizer.step()
            recorder.step(step, float(loss.detach()), params)
        adapter = root / "adapter"
        model.save_pretrained(adapter)
        assert_streamed_adapter_saved(model, str(adapter))

        second = fixture.build_fixture(root / "second")
        if restore:
            set_peft_model_state_dict(second.streamed, load_peft_weights(str(adapter),
                                                                         device="cpu"))
        recorder.reload(b5_recorder.adapter_params(second.streamed))
        return {"first_digest": b5_recorder.adapter_digest(params),
                "second_digest": b5_recorder.adapter_digest(
                    b5_recorder.adapter_params(second.streamed))}
    finally:
        first.close()
        if second is not None:
            second.close()


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    root = tmp_path_factory.mktemp("b5-tiny-moe")
    journal = root / "events.jsonl"
    digests = _train_save_reload(root, journal)
    return {"journal": journal, **digests}


def test_the_judge_passes_a_real_per_expert_training_run_on_cpu(tiny_run: Dict[str, Any]) -> None:
    result = b5_judge.judge([b5_judge.read_journal(tiny_run["journal"])])

    assert result["passed"] is True, result
    assert result["kind"] == "synthetic"
    assert result["steps"] == STEPS


def test_the_reloaded_adapter_has_the_digest_of_the_trained_one(tiny_run: Dict[str, Any]) -> None:
    assert tiny_run["second_digest"] == tiny_run["first_digest"]
    rows = b5_judge.read_journal(tiny_run["journal"])
    reload_row = [row for row in rows if row["event"] == "reload"][0]
    assert reload_row["parameter_sha256"] == tiny_run["first_digest"]


def test_the_journal_records_nonzero_gradients_on_every_trainable_tensor(
    tiny_run: Dict[str, Any],
) -> None:
    rows = b5_judge.read_journal(tiny_run["journal"])
    steps: List[Dict[str, Any]] = [row for row in rows if row["event"] == "step"]
    initial = [row for row in rows if row["event"] == "adapter_initial"][0]

    assert initial["tensors"] == steps[0]["trainable_tensors"] > 0
    for row in steps:
        assert row["gradient_tensors"] == row["trainable_tensors"]
        assert row["gradient_nonfinite_tensors"] == 0
        assert row["gradient_norm"] > b5_judge.GRADIENT_NORM_MIN
    assert steps[-1]["distance_from_initial"] > 0


def test_a_save_that_is_not_loaded_back_fails_the_reload_check(tmp_path: Path) -> None:
    journal = tmp_path / "events.jsonl"
    digests = _train_save_reload(tmp_path, journal, restore=False)

    result = b5_judge.judge([b5_judge.read_journal(journal)])

    assert digests["second_digest"] != digests["first_digest"]
    assert result["passed"] is False
    assert any(failure.startswith("reload:") for failure in result["failures"]), result


def test_the_recorder_writes_no_nan_and_never_overwrites_a_journal(tmp_path: Path) -> None:
    journal = tmp_path / "events.jsonl"
    recorder = b5_recorder.AdapterRecorder(journal)

    with pytest.raises(ValueError):
        recorder.event("step", loss=float("nan"))
    with pytest.raises(FileExistsError):
        b5_recorder.AdapterRecorder(journal)
    assert all(json.loads(line) for line in journal.read_text().splitlines())


def test_a_nan_gradient_leaves_an_abort_row_and_the_judge_names_it(tmp_path: Path) -> None:
    import torch

    journal = tmp_path / "events.jsonl"
    recorder = b5_recorder.AdapterRecorder(journal)
    recorder.signature("synthetic", "toy", generator="test")
    params = {"w": torch.nn.Parameter(torch.ones(3))}
    recorder.start(params)
    params["w"].grad = torch.tensor([1.0, float("nan"), 1.0])
    recorder.gradients(params)

    with pytest.raises(FloatingPointError):
        recorder.step(0, 1.0, params)

    rows = b5_judge.read_journal(journal)
    assert [row["event"] for row in rows][-1] == "abort"
    result = b5_judge.judge([rows])
    assert any(failure.startswith("aborted:") for failure in result["failures"]), result
