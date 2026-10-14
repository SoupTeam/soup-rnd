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


def test_bare_arm_runs_under_a_watch_with_the_label_the_shards_and_five_checks(
    profile, tmp_path
):
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

    (command, env), = runner.calls
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
