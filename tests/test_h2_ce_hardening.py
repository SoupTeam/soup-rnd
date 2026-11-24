"""Fail-closed controls for the CE hardening addendum: R7 alarm, R2 denominator, R3b, R4."""

import ast
import copy
import inspect
import json
import signal
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))
sys.path.insert(0, str(Path(__file__).parent))
import h2_ce_confirmation as runner  # noqa: E402
from h2_ce_training import Recorder, review_training, seal, window_targets  # noqa: E402
from test_h2_ce_prepared_batches import fixture_rows, trainer_for  # noqa: E402
from test_h2_ce_training import POLICY, fixture_run  # noqa: E402

WORKSET = json.loads((runner.SPEC / "ce_confirmation_workset.json").read_text())


# R7: the child alarm comes from the shared deadline and cannot interrupt evidence writing.
@pytest.mark.parametrize("ahead,expected", [(2400, 600), (300, 300 - 55), (60, 5)])
def test_alarm_follows_shared_deadline(ahead, expected):
    now = 1_000_000.0
    contract = {"max_seconds": 600, "deadline_unix_s": now + ahead}
    assert runner.alarm_seconds(contract, now) == expected


@pytest.mark.parametrize("ahead", [57, 55, 0, -30])
def test_alarm_refused_when_deadline_leaves_no_budget(ahead):
    now = 1_000_000.0
    with pytest.raises(ValueError, match="no child budget"):
        runner.alarm_seconds({"max_seconds": 600, "deadline_unix_s": now + ahead}, now)


def test_disarmed_alarm_never_raises_but_armed_alarm_does():
    try:
        runner.arm_alarm(600)
        handler = signal.getsignal(signal.SIGALRM)
        with pytest.raises(TimeoutError):
            handler(signal.SIGALRM, None)
        runner.disarm_alarm()
        handler(signal.SIGALRM, None)  # A pending signal after disarm is a no-op.
        assert signal.alarm(0) == 0
    finally:
        runner.disarm_alarm()
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


def test_real_signal_is_cancelled_by_disarm():
    try:
        runner.arm_alarm(1)
        runner.disarm_alarm()
        time.sleep(1.3)  # Would raise TimeoutError if the timer survived.
        runner.arm_alarm(1)
        with pytest.raises(TimeoutError):
            time.sleep(3)
    finally:
        runner.disarm_alarm()
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


def test_child_disarms_before_controls_stop_or_any_evidence_write():
    tree = ast.parse(inspect.getsource(runner.run_child).lstrip())
    function = tree.body[0]
    statement = next(n for n in function.body if isinstance(n, ast.Try))  # Outer try.
    first = statement.finalbody[0]
    assert isinstance(first.value, ast.Call) and first.value.func.id == "disarm_alarm"
    assert isinstance(statement.handlers[0].body[0].value.func, ast.Name)
    assert statement.handlers[0].body[0].value.func.id == "disarm_alarm"
    raw = [n for n in ast.walk(function) if isinstance(n, ast.Attribute) and n.attr == "alarm"]
    assert not raw, "run_child must use arm_alarm/disarm_alarm only"


# R2: the denominator must be an int equal to its window's target count; None is refused.
def recorder_for(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, bias=False)
    batch = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]], "labels": [[-100, 2, 3]]}
    work = [batch] * 16
    recorder = Recorder(
        tmp_path / "rec", model, ["weight"], work, accumulation=4, compute_policy="fp32-cpu-test"
    )
    inputs = {k: torch.tensor(v) for k, v in batch.items()}
    return recorder, inputs, torch.tensor(1.0)


def test_window_targets_counts_shifted_labels():
    assert window_targets([{"labels": [[-100, 2, 3]]}] * 16, 5) == 8


@pytest.mark.parametrize("denominator", [None, 0, -8, 7, 9, 8.0, True, "8"])
def test_recorder_rejects_missing_or_wrong_denominator(tmp_path, denominator):
    recorder, inputs, loss = recorder_for(tmp_path)
    with pytest.raises(ValueError, match="normalization denominator"):
        recorder.batch(inputs, loss, denominator)


def test_frozen_gpu_workset_has_24_targets_in_every_window():
    # The real GPU denominator: 6 shifted targets per row x 4 microbatches (plan: 96 per arm).
    batches = WORKSET["training_batches"]
    assert [window_targets(batches, i) for i in range(16)] == [24] * 16


def test_recorder_accepts_exact_int_denominator(tmp_path):
    recorder, inputs, loss = recorder_for(tmp_path)
    recorder.batch(inputs, loss, 8)
    assert recorder.batches[0]["num_items_in_batch"] == 8


@pytest.mark.parametrize("value", [None, 7])
def test_review_rejects_recorded_none_or_wrong_denominator(tmp_path, value):
    work = fixture_run(tmp_path / "left")
    fixture_run(tmp_path / "right")
    path = tmp_path / "right/training.json"
    data = json.loads(path.read_text())
    data["batches"][0]["num_items_in_batch"] = value
    path.write_text(json.dumps(data))
    seal(tmp_path / "right")
    with pytest.raises(ValueError, match="normalization denominator"):
        review_training(tmp_path / "left", tmp_path / "right", POLICY, work)


# R3b: the audit receipt records the dataset columns and collator config; missing config fails.
@pytest.mark.parametrize("liger", [False, True])
def test_audit_records_columns_and_collator(liger):
    rows = fixture_rows()
    audit = runner.validate_prepared_batches(trainer_for(rows, liger=liger), rows)
    assert audit["dataset_columns"] == (
        ["assistant_masks", "input_ids"]
        if liger
        else ["assistant_masks", "attention_mask", "input_ids", "labels"]
    )
    assert audit["collator"] == {
        "class": "trl.trainer.sft_trainer.DataCollatorForLanguageModeling",
        "pad_token_id": 128009,
        "padding_free": False,
        "completion_only_loss": False,
        "pad_to_multiple_of": None,
    }


@pytest.mark.parametrize(
    "missing", ["pad_token_id", "padding_free", "completion_only_loss", "pad_to_multiple_of"]
)
def test_audit_refuses_collator_without_declared_config(missing):
    rows = fixture_rows()
    trainer = trainer_for(rows, liger=True)
    real = trainer.data_collator
    state = {
        n: getattr(real, n)
        for n in ("pad_token_id", "padding_free", "completion_only_loss", "pad_to_multiple_of")
        if n != missing
    }

    class Bare:
        def __call__(self, items):
            return real(items)

    bare = Bare()
    bare.__dict__.update(state)
    trainer.data_collator = bare
    with pytest.raises(ValueError, match="Collator config field missing: " + missing):
        runner.validate_prepared_batches(trainer, rows)


# R4: every training batch must be the audited prepared row it names.
@pytest.mark.parametrize("liger", [False, True])
def test_workset_is_linked_to_prepared_rows(liger):
    rows = fixture_rows()
    result = runner.validate_workset_link(trainer_for(rows, liger=liger), WORKSET, rows)
    assert result["rows"] == WORKSET["training_row_indices"]


@pytest.mark.parametrize(
    "fault",
    [
        "batch_label",
        "batch_token",
        "swapped_indices",
        "duplicate_index",
        "out_of_range",
        "short",
        "dataset_row",
        "expected_row",
    ],
)
def test_workset_link_faults_rejected(fault):
    rows = fixture_rows()
    trainer = trainer_for(rows, liger=True)
    work = copy.deepcopy(WORKSET)
    expected = rows
    if fault == "batch_label":
        work["training_batches"][3]["labels"][0][-1] += 1
    elif fault == "batch_token":
        work["training_batches"][0]["input_ids"][0][0] += 1
    elif fault == "swapped_indices":
        indices = work["training_row_indices"]
        indices[0], indices[1] = indices[1], indices[0]
    elif fault == "duplicate_index":
        work["training_row_indices"][1] = work["training_row_indices"][0]
    elif fault == "out_of_range":
        work["training_row_indices"][0] = len(rows)
    elif fault == "short":
        work["training_row_indices"].pop()
    elif fault == "dataset_row":
        trainer.train_dataset[work["training_row_indices"][5]]["assistant_masks"][-1] = 0
    else:
        expected = copy.deepcopy(rows)
        expected[work["training_row_indices"][7]]["input_ids"][0] += 1
    with pytest.raises(ValueError, match="Invalid training row indices|differs from row"):
        runner.validate_workset_link(trainer, work, expected)


def test_recorder_requires_24_for_the_frozen_workset(tmp_path):
    torch = pytest.importorskip("torch")
    batches = WORKSET["training_batches"]
    first = {k: torch.tensor(v) for k, v in batches[0].items()}
    for denominator, ok in ((None, False), (23, False), (24.0, False), (24, True)):
        recorder = Recorder(
            tmp_path / f"r{denominator}",
            torch.nn.Linear(2, 2, bias=False),
            ["weight"],
            batches,
            accumulation=4,
            compute_policy="fp32-cpu-test",
        )
        if ok:
            recorder.batch(first, torch.tensor(1.0), denominator)
        else:
            with pytest.raises(ValueError, match="normalization denominator"):
                recorder.batch(first, torch.tensor(1.0), denominator)


def test_run_child_links_workset_before_replacing_the_training_dataset():
    function = ast.parse(inspect.getsource(runner.run_child).lstrip()).body[0]
    link = replace = None
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "validate_workset_link"
        ):
            target = node.targets[0]
            assert isinstance(target, ast.Subscript) and target.slice.value == "workset_link"
            assert target.value.id == "result"
            link = node.lineno
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "train_dataset"
        ):
            replace = node.lineno
    assert link is not None and replace is not None and link < replace
