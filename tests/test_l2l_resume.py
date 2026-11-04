"""CPU tests for the L2L resume supervisor (benchmarks/harness/l2l_resume.py).

HF writes ``checkpoint-N`` in place and ``trainer_state.json`` last, and ``soup train
--resume auto`` takes the highest N without looking inside. A run killed during a save
leaves a half-written ``checkpoint-N``, so the supervisor picks the checkpoint itself and
hands ``soup train`` an explicit path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def resume():
    return _load("l2l_resume")


def _checkpoint(output, step, *, complete=True):
    folder = output / f"checkpoint-{step}"
    folder.mkdir(parents=True)
    (folder / "adapter_model.safetensors").write_bytes(b"w")
    (folder / "optimizer.pt").write_bytes(b"o")
    if complete:
        (folder / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return folder


# --------------------------------------------------------------------------
# which checkpoint a restart may use
# --------------------------------------------------------------------------
def test_no_output_directory_gives_no_checkpoint(resume, tmp_path):
    assert resume.latest_complete_checkpoint(tmp_path / "missing") is None


def test_highest_complete_checkpoint_wins(resume, tmp_path):
    _checkpoint(tmp_path, 2)
    _checkpoint(tmp_path, 10)
    _checkpoint(tmp_path, 4)
    assert resume.latest_complete_checkpoint(tmp_path).name == "checkpoint-10"


def test_numbers_compare_as_numbers_not_as_text(resume, tmp_path):
    _checkpoint(tmp_path, 9)
    _checkpoint(tmp_path, 100)
    assert resume.latest_complete_checkpoint(tmp_path).name == "checkpoint-100"


def test_a_checkpoint_killed_mid_save_is_skipped(resume, tmp_path):
    _checkpoint(tmp_path, 4)
    _checkpoint(tmp_path, 6, complete=False)  # no trainer_state.json: HF writes it last
    assert resume.latest_complete_checkpoint(tmp_path).name == "checkpoint-4"


def test_a_truncated_trainer_state_is_skipped(resume, tmp_path):
    _checkpoint(tmp_path, 4)
    broken = _checkpoint(tmp_path, 6, complete=False)
    (broken / "trainer_state.json").write_text('{"global_st')
    assert resume.latest_complete_checkpoint(tmp_path).name == "checkpoint-4"


def test_a_trainer_state_of_another_step_is_skipped(resume, tmp_path):
    _checkpoint(tmp_path, 4)
    odd = _checkpoint(tmp_path, 6, complete=False)
    (odd / "trainer_state.json").write_text(json.dumps({"global_step": 5}))
    assert resume.latest_complete_checkpoint(tmp_path).name == "checkpoint-4"


def test_only_incomplete_checkpoints_give_none(resume, tmp_path):
    _checkpoint(tmp_path, 3, complete=False)
    assert resume.latest_complete_checkpoint(tmp_path) is None


def test_other_entries_are_ignored(resume, tmp_path):
    _checkpoint(tmp_path, 3)
    (tmp_path / "checkpoint-final").mkdir()
    (tmp_path / "checkpoint-7").write_text("a file, not a directory")
    (tmp_path / "_l2l_spill").mkdir()
    assert resume.latest_complete_checkpoint(tmp_path).name == "checkpoint-3"


def test_step_of_a_checkpoint(resume, tmp_path):
    assert resume.checkpoint_step(_checkpoint(tmp_path, 12)) == 12


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------
def test_fresh_command_has_no_resume_flag(resume):
    command = resume.arm_command("l2l.yaml", "attempt-1.json", None, python="py")
    assert "--resume-from" not in command
    assert command[:2] == ["py", str(resume.HARNESS / "l2l_ticket04_arm.py")]
    assert command[command.index("--config") + 1] == "l2l.yaml"
    assert command[command.index("--out") + 1] == "attempt-1.json"


def test_resumed_command_names_the_checkpoint(resume, tmp_path):
    command = resume.arm_command("l2l.yaml", "a.json", tmp_path / "checkpoint-4", python="py")
    assert command[command.index("--resume-from") + 1] == str(tmp_path / "checkpoint-4")


def test_soup_train_command_resumes_from_an_explicit_path(resume):
    assert resume.soup_train_command("c.yaml", None, None) == [
        "train", "--config", "c.yaml", "--yes",
    ]
    assert resume.soup_train_command("c.yaml", "n", "out/checkpoint-4") == [
        "train", "--config", "c.yaml", "--yes", "--name", "n", "--resume", "out/checkpoint-4",
    ]


# --------------------------------------------------------------------------
# identity: a restart must continue the same run
# --------------------------------------------------------------------------
def _files(tmp_path, config="a: 1\n", data="row\n"):
    (tmp_path / "c.yaml").write_text(config)
    (tmp_path / "d.jsonl").write_text(data)
    return tmp_path / "c.yaml", tmp_path / "d.jsonl"


def test_identity_changes_with_config_or_data(resume, tmp_path):
    config, data = _files(tmp_path)
    first = resume.run_identity(config, data)
    config.write_text("a: 2\n")
    second = resume.run_identity(config, data)
    data.write_text("other\n")
    third = resume.run_identity(config, data)
    assert len({json.dumps(x, sort_keys=True) for x in (first, second, third)}) == 3


def test_first_attempt_records_identity_and_a_later_one_accepts_it(resume, tmp_path):
    config, data = _files(tmp_path)
    out = tmp_path / "out"
    resume.check_identity(out, resume.run_identity(config, data))
    resume.check_identity(out, resume.run_identity(config, data))  # same run: no error


def test_changed_data_is_refused(resume, tmp_path):
    config, data = _files(tmp_path)
    out = tmp_path / "out"
    resume.check_identity(out, resume.run_identity(config, data))
    data.write_text("changed\n")
    with pytest.raises(ValueError, match="differs"):
        resume.check_identity(out, resume.run_identity(config, data))


# --------------------------------------------------------------------------
# the supervisor loop
# --------------------------------------------------------------------------
class _Fake:
    """A run that dies at the given steps; each attempt saves a checkpoint every 2 steps."""

    def __init__(self, deaths, total=10):
        self.deaths, self.total, self.saved, self.started = list(deaths), total, 0, []

    def checkpoint(self):
        return self.saved or None

    def attempt(self, checkpoint):
        start = checkpoint or 0
        self.started.append(start)
        end = self.deaths.pop(0) if self.deaths else self.total
        if end < self.total:
            self.saved = max(self.saved, end - end % 2)
            return 137
        self.saved = self.total
        return 0


def _supervise(resume, fake, **kw):
    return resume.supervise(
        fake.attempt, fake.checkpoint, max_attempts=kw.pop("max_attempts", 5),
        step_of=lambda checkpoint: checkpoint, **kw,
    )


def test_a_clean_run_takes_one_attempt_without_resume(resume):
    fake = _Fake([])
    result = _supervise(resume, fake)
    assert result["status"] == "completed"
    assert [a["resumed_from"] for a in result["attempts"]] == [None]


def test_a_killed_run_resumes_from_its_last_checkpoint(resume):
    fake = _Fake([5])
    result = _supervise(resume, fake)
    assert result["status"] == "completed"
    assert [a["resumed_from"] for a in result["attempts"]] == [None, 4]
    assert [a["returncode"] for a in result["attempts"]] == [137, 0]


def test_several_kills_resume_each_time_further_on(resume):
    fake = _Fake([3, 7])
    result = _supervise(resume, fake)
    assert result["status"] == "completed"
    assert fake.started == [0, 2, 6]


def test_a_kill_before_any_checkpoint_restarts_from_scratch(resume):
    fake = _Fake([1])  # dies at step 1, nothing saved yet
    result = _supervise(resume, fake)
    assert result["status"] == "completed"
    assert [a["resumed_from"] for a in result["attempts"]] == [None, None]


def test_the_attempt_cap_stops_the_loop(resume):
    fake = _Fake([3, 5, 7, 9])
    result = _supervise(resume, fake, max_attempts=2)
    assert result["status"] == "gave_up"
    assert len(result["attempts"]) == 2
    assert result["reason"] == "attempt cap"


def test_two_attempts_without_progress_stop_the_loop(resume):
    fake = _Fake([4, 4, 4, 4])  # dies at the same step every time: a bug, not a kill
    result = _supervise(resume, fake, max_attempts=9)
    assert result["status"] == "gave_up"
    assert result["reason"] == "no progress"
    assert len(result["attempts"]) == 3


def test_a_kill_before_the_first_checkpoint_twice_counts_as_no_progress(resume):
    fake = _Fake([1, 1, 1, 1])
    result = _supervise(resume, fake, max_attempts=9)
    assert result["status"] == "gave_up" and result["reason"] == "no progress"


# --------------------------------------------------------------------------
# the command line, with the arm replaced by a fake process
# --------------------------------------------------------------------------
def test_main_restarts_a_killed_arm_from_the_complete_checkpoint(resume, tmp_path, monkeypatch):
    import subprocess

    data = tmp_path / "d.jsonl"
    data.write_text("row\n")
    out = tmp_path / "out"
    config = tmp_path / "c.yaml"
    config.write_text(
        f"output: {out}\ndata:\n  train: {data}\ntraining:\n  save_steps: 2\n"
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:  # killed after saving step 4, and while saving step 6
            _checkpoint(out, 4)
            _checkpoint(out, 6, complete=False)
            return subprocess.CompletedProcess(command, 137)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = resume.main(["--config", str(config), "--results", str(tmp_path / "res")])
    assert status == 0
    assert "--resume-from" not in commands[0]
    assert commands[1][commands[1].index("--resume-from") + 1] == str(out / "checkpoint-4")
    saved = json.loads((tmp_path / "res" / "attempts.json").read_text())
    assert [a["resumed_from"] for a in saved["attempts"]] == [None, 4]
    assert (tmp_path / "res" / "attempt-2.log").exists()


# --------------------------------------------------------------------------
# review of PR 55: where the checkpoints are, what the data is, what a restart adopts
# --------------------------------------------------------------------------
def test_checkpoints_live_below_the_experiment_name(resume):
    assert resume.checkpoint_directory({"output": "out", "experiment_name": "exp"}) == (
        Path("out") / "exp"
    )


def test_output_defaults_to_the_schema_default(resume):
    assert resume.checkpoint_directory({}) == Path("./output")
    assert resume.checkpoint_directory({"experiment_name": "exp"}) == Path("./output") / "exp"


def test_a_list_of_data_files_is_hashed_file_by_file(resume, tmp_path):
    config, first = _files(tmp_path)
    second = tmp_path / "e.jsonl"
    second.write_text("more\n")
    both = resume.run_identity(config, [str(first), str(second)])
    assert both == resume.run_identity(config, [str(first), str(second)])
    second.write_text("changed\n")
    assert resume.run_identity(config, [str(first), str(second)]) != both


def test_the_order_of_the_data_files_is_part_of_the_identity(resume, tmp_path):
    config, first = _files(tmp_path)
    second = tmp_path / "e.jsonl"
    second.write_text("more\n")
    assert resume.run_identity(config, [str(first), str(second)]) != resume.run_identity(
        config, [str(second), str(first)]
    )


def test_a_dataset_name_is_identified_by_its_text(resume, tmp_path):
    config, _ = _files(tmp_path)
    assert resume.run_identity(config, "org/name") == resume.run_identity(config, "org/name")
    assert resume.run_identity(config, "org/name") != resume.run_identity(config, "org/other")


def test_identity_survives_the_json_round_trip(resume, tmp_path):
    config, data = _files(tmp_path)
    identity = resume.run_identity(config, [str(data), "org/name"])
    resume.check_identity(tmp_path / "out", identity)
    resume.check_identity(tmp_path / "out", identity)  # read back from the file: equal


def test_checkpoints_without_an_identity_file_are_refused(resume, tmp_path):
    config, data = _files(tmp_path)
    out = tmp_path / "out"
    _checkpoint(out, 4)  # written by some other run
    with pytest.raises(ValueError, match="adopt"):
        resume.check_identity(out, resume.run_identity(config, data))


def test_checkpoints_can_be_adopted_on_request(resume, tmp_path):
    config, data = _files(tmp_path)
    out = tmp_path / "out"
    _checkpoint(out, 4)
    identity = resume.run_identity(config, data)
    resume.check_identity(out, identity, adopt=True)
    resume.check_identity(out, identity)  # now recorded: no flag needed


def _main_with_fake_arm(resume, tmp_path, monkeypatch, config_text, *, results="res", kill=None):
    import subprocess

    config = tmp_path / "c.yaml"
    config.write_text(config_text)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if kill and len(commands) == 1:
            kill()
            return subprocess.CompletedProcess(command, 137)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = resume.main(["--config", str(config), "--results", str(tmp_path / results)])
    return status, commands


def test_main_finds_checkpoints_below_the_experiment_name(resume, tmp_path, monkeypatch):
    data = tmp_path / "d.jsonl"
    data.write_text("row\n")
    out = tmp_path / "out"
    status, commands = _main_with_fake_arm(
        resume, tmp_path, monkeypatch,
        f"output: {out}\nexperiment_name: exp\ndata:\n  train: {data}\n",
        kill=lambda: _checkpoint(out / "exp", 4),
    )
    assert status == 0
    assert commands[1][commands[1].index("--resume-from") + 1] == str(out / "exp" / "checkpoint-4")


def test_main_takes_a_list_of_data_files_and_no_output_key(resume, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "d.jsonl"
    data.write_text("row\n")
    status, commands = _main_with_fake_arm(
        resume, tmp_path, monkeypatch, f"data:\n  train:\n    - {data}\n    - org/name\n",
        kill=lambda: _checkpoint(tmp_path / "output", 2),
    )
    assert status == 0
    assert commands[1][commands[1].index("--resume-from") + 1] == str(
        Path("output") / "checkpoint-2"
    )


def test_save_steps_warning_follows_the_schema_default(resume, tmp_path, monkeypatch, capsys):
    data = tmp_path / "d.jsonl"
    data.write_text("row\n")
    base = f"output: {tmp_path / 'out'}\ndata:\n  train: {data}\n"
    _main_with_fake_arm(resume, tmp_path, monkeypatch, base)
    assert "save_steps" not in capsys.readouterr().out  # absent key means the default, 100
    other = f"output: {tmp_path / 'out2'}\ndata:\n  train: {data}\ntraining:\n  save_steps: 0\n"
    _main_with_fake_arm(resume, tmp_path, monkeypatch, other, results="res2")
    assert "save_steps" in capsys.readouterr().out


def test_a_second_invocation_continues_the_attempt_numbers(resume, tmp_path, monkeypatch):
    data = tmp_path / "d.jsonl"
    data.write_text("row\n")
    results = tmp_path / "res"
    results.mkdir()
    (results / "attempt-1.log").write_text("earlier invocation")
    (results / "attempts.json").write_text(json.dumps({
        "status": "gave_up", "attempts": [{"attempt": 1, "resumed_from": None, "returncode": 137}],
    }))
    status, commands = _main_with_fake_arm(
        resume, tmp_path, monkeypatch, f"output: {tmp_path / 'out'}\ndata:\n  train: {data}\n",
    )
    assert status == 0
    assert (results / "attempt-2.log").exists()
    assert commands[0][commands[0].index("--out") + 1] == str(results / "attempt-2.json")
    saved = json.loads((results / "attempts.json").read_text())
    assert [a["attempt"] for a in saved["attempts"]] == [1, 2]
    assert saved["status"] == "completed"


def test_supervise_numbers_attempts_from_the_first_number(resume):
    fake = _Fake([])
    result = resume.supervise(
        fake.attempt, fake.checkpoint, max_attempts=3, first_number=4, step_of=lambda c: c,
    )
    assert [a["attempt"] for a in result["attempts"]] == [4]
