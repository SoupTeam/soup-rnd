"""Tests for the G7 visit profile harness (benchmarks/harness/visit_profile.py, G7 ticket 02).

The worker, the watch and the fixture writer are fakes: no test builds a model, starts a
process, reads a sensor or writes into benchmarks/results/. Step times and path counts
are synthetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def profile():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_visit_profile_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "visit_profile.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


PATHS = {"fused_layers": 7, "old_layers": 0, "fused_calls": 14, "old_calls": 0}


def _steps(*seconds):
    queue = list(seconds)
    return lambda: queue.pop(0)


# --- one run ---------------------------------------------------------------------------


def test_run_records_the_median_of_the_timed_steps_only(profile):
    # Three warm-up steps are slow; the five timed steps have the median 0.30.
    step = _steps(9.0, 8.0, 7.0, 0.50, 0.30, 0.20, 0.31, 0.29)
    result = profile.run_arm(step, lambda: PATHS, warmup=3, timed=5)

    assert result["median_step_s"] == pytest.approx(0.30)
    assert result["step_s"] == [0.50, 0.30, 0.20, 0.31, 0.29]
    assert result["timed_steps"] == 5
    assert result["warmup_steps"] == 3


def test_run_records_how_many_linears_took_each_path(profile):
    result = profile.run_arm(_steps(1.0, 1.0), lambda: PATHS, warmup=1, timed=1)

    assert result["linear_paths"] == PATHS


def test_run_counts_paths_after_the_timed_steps(profile):
    order = []

    def step():
        order.append("step")
        return 1.0

    def count_paths():
        order.append("count")
        return PATHS

    profile.run_arm(step, count_paths, warmup=1, timed=2)

    assert order == ["step", "step", "step", "count"]


def test_run_needs_at_least_one_timed_step(profile):
    with pytest.raises(ValueError, match="timed"):
        profile.run_arm(_steps(1.0), lambda: PATHS, warmup=1, timed=0)


# --- the fixture -----------------------------------------------------------------------


class FakeRunner:
    """Logs the command and environment it was asked to start and returns."""

    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call

    def __call__(self, command, env):
        self.calls.append((list(command), dict(env)))
        if self.on_call:
            self.on_call(command, env)


def test_fixture_is_written_with_the_gpu_hidden_on_a_disk_under_root(profile, tmp_path):
    runner = FakeRunner()
    fixture = profile.build_fixture(tmp_path, runner=runner, disk_device=lambda path: "nvme0n1")

    command, env = runner.calls[0]
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert command[command.index("--shape") + 1] == "llama-8b"
    assert Path(command[command.index("--out") + 1]) == fixture.weights
    assert fixture.weights.parent == tmp_path
    assert fixture.shards.parent == tmp_path


def test_fixture_refuses_a_root_that_is_not_on_a_disk(profile, tmp_path):
    runner = FakeRunner()
    with pytest.raises(ValueError, match="block device"):
        profile.build_fixture(tmp_path, runner=runner, disk_device=lambda path: None)

    assert runner.calls == []


# --- one bare arm under the watch ------------------------------------------------------


class FakeWatch:
    """Stands in for ``G7ArmWatch``: keeps the arguments it was made with."""

    made = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.record = None
        FakeWatch.made.append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.record = {
            "arm": self.kwargs["arm"],
            "round": self.kwargs["round"],
            "run": self.kwargs["run"],
            "label": self.kwargs["label"],
            "required": list(self.kwargs["required"]),
            "outcome": "ok",
            "checks": {},
        }


def _bare_arm(profile, tmp_path, **overrides):
    FakeWatch.made = []

    def worker(command, env):
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps({"median_step_s": 0.3, "linear_paths": PATHS}), encoding="utf-8")

    kwargs = dict(
        seq=16,
        fixture=profile.Fixture(weights=tmp_path / "w", shards=tmp_path / "shards"),
        root=tmp_path,
        out_dir=tmp_path / "out",
        runner=FakeRunner(worker),
        watch=FakeWatch,
    )
    kwargs.update(overrides)
    return profile.run_bare_arm(**kwargs), kwargs["runner"]


def test_bare_arm_runs_under_a_watch_with_the_label_the_shards_and_five_checks(profile, tmp_path):
    _bare_arm(profile, tmp_path)

    (made,) = FakeWatch.made
    assert made["label"] == "RTX 3050, synthetic"
    assert made["model_path"] == str(tmp_path / "shards")
    assert made["required"] == ["suspend", "power", "foreign_reads", "gpu_power_limit", "swap"]
    assert made["arm"] == "bare-16"


def test_bare_arm_starts_a_fresh_process_with_the_token_count_and_the_root_as_tmpdir(
    profile, tmp_path
):
    _, runner = _bare_arm(profile, tmp_path, seq=64)

    ((command, env),) = runner.calls
    assert command[command.index("--seq") + 1] == "64"
    assert command[command.index("--batch") + 1] == "1"
    assert env["TMPDIR"] == str(tmp_path)


def test_bare_arm_writes_the_run_json_and_the_arm_record(profile, tmp_path):
    result, _ = _bare_arm(profile, tmp_path)

    run_file = tmp_path / "out" / "bare-16.run.json"
    arm_file = tmp_path / "out" / "bare-16.arm.json"
    assert json.loads(run_file.read_text(encoding="utf-8"))["median_step_s"] == 0.3
    assert json.loads(arm_file.read_text(encoding="utf-8"))["outcome"] == "ok"
    assert result["run"]["linear_paths"] == PATHS
    assert result["arm"]["label"] == "RTX 3050, synthetic"


def test_bare_arm_runs_the_worker_inside_the_watch(profile, tmp_path):
    seen = []

    class OrderedWatch(FakeWatch):
        def __enter__(self):
            seen.append("enter")
            return super().__enter__()

        def __exit__(self, *exc_info):
            seen.append("exit")
            super().__exit__(*exc_info)

    def worker(command, env):
        seen.append("worker")
        Path(command[command.index("--out") + 1]).write_text("{}", encoding="utf-8")

    _bare_arm(profile, tmp_path, runner=FakeRunner(worker), watch=OrderedWatch)

    assert seen == ["enter", "worker", "exit"]


@pytest.mark.parametrize("seq", [0, 8, 256, 512])
def test_bare_arm_takes_only_16_32_64_or_128_tokens(profile, tmp_path, seq):
    with pytest.raises(ValueError, match="16, 32, 64 or 128"):
        _bare_arm(profile, tmp_path, seq=seq)


def test_bare_arm_creates_the_shard_folder_before_the_watch_starts(profile, tmp_path):
    seen = []

    class CheckingWatch(FakeWatch):
        def __enter__(self):
            seen.append((tmp_path / "shards").is_dir())
            return super().__enter__()

    _bare_arm(profile, tmp_path, watch=CheckingWatch)

    assert seen == [True]


# --- one timer run ---------------------------------------------------------------------


class FakeRecorder:
    """Stands in for ``visit_timer.Recorder``: returns a canned summary and LoRA report."""

    def __init__(self):
        self.calls = []

    def summary(self, warmup, timed):
        self.calls.append(("summary", warmup, timed))
        return {"forward_visit_ms": {"median": 1.5, "values": [1.5]}}


def test_timer_run_has_every_field_of_a_bare_run_plus_the_timer_summary(profile):
    recorder = FakeRecorder()
    lora = {"inside": 3, "outside": 0, "state": "inside"}

    result = profile.run_timer_arm(
        _steps(1.0, 0.5, 0.7),
        lambda: PATHS,
        recorder,
        lambda: lora,
        warmup=1,
        timed=2,
    )

    bare = profile.run_arm(_steps(1.0, 0.5, 0.7), lambda: PATHS, warmup=1, timed=2)
    assert set(bare) <= set(result)
    assert result["median_step_s"] == bare["median_step_s"]
    assert result["timer"] == recorder.summary(1, 2)
    assert result["lora_in_window"] == lora


def test_timer_run_summarises_the_timed_steps_before_the_extra_steps(profile):
    order = []
    recorder = FakeRecorder()
    original = recorder.summary
    recorder.summary = lambda warmup, timed: (order.append("summary"), original(warmup, timed))[1]

    def count_paths():
        order.append("count")
        return PATHS

    def check_lora():
        order.append("lora")
        return {"inside": 0, "outside": 0, "state": "inside"}

    profile.run_timer_arm(_steps(1.0, 1.0), count_paths, recorder, check_lora, warmup=1, timed=1)

    assert order == ["count", "summary", "lora"]


# --- the timer arm under the watch -----------------------------------------------------


def test_timer_arm_is_a_fresh_process_named_timer_with_its_own_files(profile, tmp_path):
    result, runner = _timer_arm(profile, tmp_path, seq=32)

    ((command, _),) = runner.calls
    assert command[command.index("--arm") + 1] == "timer"
    assert command[command.index("--seq") + 1] == "32"
    assert FakeWatch.made[0]["arm"] == "timer-32"
    assert (tmp_path / "out" / "timer-32.run.json").is_file()
    assert (tmp_path / "out" / "timer-32.arm.json").is_file()
    assert result["arm"]["label"] == "RTX 3050, synthetic"
    assert FakeWatch.made[0]["required"] == [
        "suspend", "power", "foreign_reads", "gpu_power_limit", "swap",
    ]  # fmt: skip


def test_bare_and_timer_arms_of_one_round_leave_separate_jsons(profile, tmp_path):
    _bare_arm(profile, tmp_path, round=1)
    _timer_arm(profile, tmp_path, round=1)

    names = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert names == [
        "bare-16.arm.json", "bare-16.run.json", "timer-16.arm.json", "timer-16.run.json",
    ]  # fmt: skip


def test_the_bare_arm_tells_its_worker_to_run_bare(profile, tmp_path):
    _, runner = _bare_arm(profile, tmp_path)

    ((command, _),) = runner.calls
    assert command[command.index("--arm") + 1] == "bare"


def _timer_arm(profile, tmp_path, **overrides):
    FakeWatch.made = []

    def worker(command, env):
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps({"median_step_s": 0.3, "arm": "timer"}), encoding="utf-8")

    kwargs = dict(
        seq=16,
        fixture=profile.Fixture(weights=tmp_path / "w", shards=tmp_path / "shards"),
        root=tmp_path,
        out_dir=tmp_path / "out",
        runner=FakeRunner(worker),
        watch=FakeWatch,
    )
    kwargs.update(overrides)
    return profile.run_timer_process(**kwargs), kwargs["runner"]


# --- the profile arm -------------------------------------------------------------------


def test_profile_run_profiles_only_the_timed_steps_and_names_the_trace(profile):
    order = []

    def step():
        order.append("step")
        return 0.5

    def capture(body):
        order.append("capture")
        return body()

    result = profile.run_profile_arm(
        step, lambda: PATHS, capture, warmup=2, timed=3, trace_file="/x/profile-16.trace.json"
    )

    assert order == ["step", "step", "capture", "step", "step", "step"]
    assert result["step_s"] == [0.5, 0.5, 0.5]
    assert result["median_step_s"] == 0.5
    assert result["linear_paths"] == PATHS
    assert result["trace_file"] == "/x/profile-16.trace.json"


ANALYSIS = {
    "summary": {"forward": {"count": 96}},
    "reconciliation": {"ok": True, "gap_fraction": 0.01},
    "memcpy_sources": {"aten::clone > aten::copy_": 4},
}


def _profile_arm(profile, tmp_path, analyse=None, **overrides):
    FakeWatch.made = []
    seen = []

    class OrderedWatch(FakeWatch):
        def __exit__(self, *exc_info):
            seen.append("watch closed")
            super().__exit__(*exc_info)

    def worker(command, env):
        out = Path(command[command.index("--out") + 1])
        trace = out.with_name("profile-16.trace.json")
        trace.write_text("{}", encoding="utf-8")
        out.write_text(
            json.dumps({"median_step_s": 0.3, "trace_file": str(trace)}), encoding="utf-8"
        )

    def fake_analyse(path):
        seen.append(("analyse", Path(path).name))
        return ANALYSIS

    kwargs = dict(
        seq=16,
        fixture=profile.Fixture(weights=tmp_path / "w", shards=tmp_path / "shards"),
        root=tmp_path,
        out_dir=tmp_path / "out",
        runner=FakeRunner(worker),
        watch=OrderedWatch,
        analyse=analyse or fake_analyse,
    )
    kwargs.update(overrides)
    return profile.run_arm_process(arm="profile", **kwargs), kwargs["runner"], seen


def test_profile_arm_profiles_three_steps_by_default_in_a_fresh_process(profile, tmp_path):
    _, runner, _ = _profile_arm(profile, tmp_path)

    ((command, _),) = runner.calls
    assert command[command.index("--arm") + 1] == "profile"
    assert command[command.index("--steps") + 1] == "3"
    assert FakeWatch.made[0]["arm"] == "profile-16"


def test_profile_arm_analyses_the_trace_after_the_watch_has_closed(profile, tmp_path):
    result, _, seen = _profile_arm(profile, tmp_path)

    assert seen == ["watch closed", ("analyse", "profile-16.trace.json")]
    assert result["run"]["profile"] == ANALYSIS
    saved = json.loads((tmp_path / "out" / "profile-16.run.json").read_text(encoding="utf-8"))
    assert saved["profile"] == ANALYSIS


def test_profile_arm_names_the_device_to_device_copy_source_in_the_notes(profile, tmp_path):
    result, _, _ = _profile_arm(profile, tmp_path)

    notes = " ".join(result["run"]["notes"])
    assert "aten::clone > aten::copy_" in notes
    assert "device-to-device" in notes


def test_profile_arm_notes_name_the_peft_clone_as_the_source_of_the_copies(profile, tmp_path):
    result, _, _ = _profile_arm(profile, tmp_path)

    notes = " ".join(result["run"]["notes"])
    assert "peft/tuners/lora/bnb.py" in notes
    assert "result.clone()" in notes


def test_profile_arm_deletes_the_trace_unless_asked_to_keep_it(profile, tmp_path):
    result, _, _ = _profile_arm(profile, tmp_path)
    assert not (tmp_path / "out" / "profile-16.trace.json").exists()
    assert "trace_file" not in result["run"]

    kept, _, _ = _profile_arm(profile, tmp_path, keep_trace=True)
    assert (tmp_path / "out" / "profile-16.trace.json").is_file()
    assert kept["run"]["trace_file"].endswith("profile-16.trace.json")


def test_the_trace_file_loader_reads_the_trace_events_list(profile, tmp_path):
    path = tmp_path / "t.trace.json"
    events = [
        {"ph": "X", "cat": "user_annotation", "name": "g7::forward_visit", "ts": 0, "dur": 500,
         "tid": 1, "args": {}},
    ]  # fmt: skip
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")

    analysis = profile.analyse_trace_file(path)

    assert analysis["summary"]["forward"]["host_ms"] == pytest.approx(0.5)


# --- control arm ----------------------------------------------------------------------


def _control_arm(profile, tmp_path, **overrides):
    overrides.setdefault("seq", 16)
    arm = overrides.pop("arm", "control")
    FakeWatch.made = []

    def worker(command, env):
        out = Path(command[command.index("--out") + 1])
        trace = out.with_name(f"{arm}-16.trace.json")
        trace.write_text("{}", encoding="utf-8")
        out.write_text(
            json.dumps({"median_step_s": 0.3, "trace_file": str(trace)}), encoding="utf-8"
        )

    kwargs = dict(
        fixture=profile.Fixture(weights=tmp_path / "w", shards=tmp_path / "shards"),
        root=tmp_path,
        out_dir=tmp_path / "out",
        runner=FakeRunner(worker),
        watch=FakeWatch,
        analyse=lambda path: ANALYSIS,
    )
    kwargs.update(overrides)
    return profile.run_arm_process(arm=arm, **kwargs), kwargs["runner"]


def test_control_is_one_of_the_arms(profile):
    assert "control" in profile.ARMS


def test_control_arm_runs_in_a_fresh_process_named_control(profile, tmp_path):
    result, runner = _control_arm(profile, tmp_path)
    command = runner.calls[0][0]
    assert command[command.index("--arm") + 1] == "control"
    assert (tmp_path / "out" / "control-16.run.json").exists()
    assert (tmp_path / "out" / "control-16.arm.json").exists()


def test_control_arm_profiles_three_steps_and_analyses_the_trace(profile, tmp_path):
    result, runner = _control_arm(profile, tmp_path)
    command = runner.calls[0][0]
    assert command[command.index("--steps") + 1] == str(profile.PROFILE_STEPS)
    assert result["run"]["profile"] == ANALYSIS
    assert "trace_file" not in result["run"]


def test_control_arm_runs_at_16_tokens_only(profile, tmp_path):
    for seq in (32, 64, 128):
        with pytest.raises(ValueError, match="16"):
            _control_arm(profile, tmp_path, seq=seq)


def test_control_run_has_the_timer_summary_beside_the_profile_fields(profile):
    recorder = FakeRecorder()
    captured = []

    def capture(body):
        captured.append(body())
        return captured[0]

    result = profile.run_profile_arm(
        _steps(1.0, 0.5, 0.7, 0.6),
        lambda: PATHS,
        capture,
        warmup=1,
        timed=3,
        trace_file="t.json",
        recorder=recorder,
    )

    assert result["timer"] == {"forward_visit_ms": {"median": 1.5, "values": [1.5]}}
    assert recorder.calls == [("summary", 1, 3)]
    assert result["trace_file"] == "t.json"


def test_a_profile_run_without_a_recorder_has_no_timer_field(profile):
    result = profile.run_profile_arm(
        _steps(1.0, 0.5),
        lambda: PATHS,
        lambda body: body(),
        warmup=1,
        timed=1,
        trace_file="t.json",
    )
    assert "timer" not in result
