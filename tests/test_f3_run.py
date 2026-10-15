"""Tests for the F3 run lifecycle (benchmarks/harness/f3_run.py, F3 ticket 06).

Everything is faked: arms, upload, destroy and the clock. No test opens the network,
touches a GPU or calls vast.ai.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
TOKEN = "hf_SECRETTOKENVALUE1234567890"


@pytest.fixture(scope="module")
def run_mod():
    name = "_f3_run_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "f3_run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Env:
    """Fake world: records the order of upload and destroy calls, and runs arms."""

    def __init__(self, clock, seconds=600.0, fail=(), upload_fail=()):
        self.clock = clock
        self.seconds = seconds
        self.fail = set(fail)
        self.upload_fail = set(upload_fail)
        self.events = []
        self.logs = []

    def run_arm(self, name):
        self.events.append(("run", name))
        self.clock.now += self.seconds
        if name in self.fail:
            raise RuntimeError(f"{name} crashed with {TOKEN}")
        return {"arm": name, "valid": True}

    def upload(self, name, path):
        self.events.append(("upload", name))
        if name in self.upload_fail:
            raise OSError("network down")
        assert Path(path).exists()

    def destroy(self):
        self.events.append(("destroy", None))

    def log(self, line):
        self.logs.append(line)


def plans(run_mod, *names, estimate=600.0):
    return [run_mod.ArmPlan(name, estimate) for name in names]


def go(run_mod, tmp_path, env, clock, arms, **kwargs):
    options = dict(
        arms=arms,
        run_arm=env.run_arm,
        upload=env.upload,
        destroy=env.destroy,
        workdir=tmp_path,
        clock=clock,
        log=env.log,
        rate_per_hour=0.80,
        ceiling=2.50,
        time_limit_seconds=3 * 3600,
        secrets=(TOKEN,),
    )
    options.update(kwargs)
    return run_mod.run_lifecycle(**options)


def test_failed_arm_does_not_stop_the_others(run_mod, tmp_path):
    clock = Clock()
    env = Env(clock, fail={"b"})
    result = go(run_mod, tmp_path, env, clock, plans(run_mod, "a", "b", "c"))
    assert [name for kind, name in env.events if kind == "run"] == ["a", "b", "c"]
    assert result.outcomes == {"a": "done", "b": "failed", "c": "done"}
    record = json.loads((tmp_path / "b.json").read_text())
    assert record["status"] == "failed"


def test_restart_skips_uploaded_arms(run_mod, tmp_path):
    clock = Clock()
    first = Env(clock, upload_fail={"b"})
    go(run_mod, tmp_path, first, clock, plans(run_mod, "a", "b"))
    second = Env(clock)
    go(run_mod, tmp_path, second, clock, plans(run_mod, "a", "b"))
    assert [e for e in second.events if e[0] == "run"] == [("run", "b")]


def test_upload_after_every_arm_and_destroy_last(run_mod, tmp_path):
    clock = Clock()
    env = Env(clock, fail={"a"})
    go(run_mod, tmp_path, env, clock, plans(run_mod, "a", "b"))
    uploads = [i for i, e in enumerate(env.events) if e[0] == "upload"]
    runs = [i for i, e in enumerate(env.events) if e[0] == "run"]
    assert runs[0] < uploads[0] < runs[1] < uploads[1]
    assert env.events[-1] == ("destroy", None)
    assert [e for e in env.events if e[0] == "destroy"] == [("destroy", None)]


def test_failed_upload_keeps_the_instance_alive(run_mod, tmp_path):
    clock = Clock()
    env = Env(clock, upload_fail={"a"})
    result = go(run_mod, tmp_path, env, clock, plans(run_mod, "a"))
    assert ("destroy", None) not in env.events
    assert result.pending_uploads == ["a"]
    assert result.destroyed is False


def test_stops_before_the_ceiling_is_exceeded(run_mod, tmp_path):
    clock = Clock()
    env = Env(clock, seconds=3600.0)
    arms = [run_mod.ArmPlan(name, 3600.0) for name in ("a", "b", "c", "d")]
    # 0.80 $/h: three one-hour arms cost 2.40, a fourth would reach 3.20.
    result = go(run_mod, tmp_path, env, clock, arms, time_limit_seconds=10 * 3600)
    assert [n for k, n in env.events if k == "run"] == ["a", "b", "c"]
    assert result.outcomes["d"] == "skipped: budget"
    assert result.spent <= 2.50
    assert env.events[-1] == ("destroy", None)


def test_time_limit_uploads_and_ends(run_mod, tmp_path):
    clock = Clock()
    env = Env(clock, seconds=3600.0)
    arms = [run_mod.ArmPlan(name, 3600.0) for name in ("a", "b", "c", "d")]
    result = go(
        run_mod, tmp_path, env, clock, arms, rate_per_hour=0.10, time_limit_seconds=2.5 * 3600
    )
    assert [n for k, n in env.events if k == "run"] == ["a", "b"]
    assert result.outcomes["c"] == "skipped: time limit"
    assert ("upload", "a") in env.events and ("upload", "b") in env.events
    assert env.events[-1] == ("destroy", None)


def test_spend_is_logged_against_the_ceiling(run_mod, tmp_path):
    clock = Clock()
    env = Env(clock)
    go(run_mod, tmp_path, env, clock, plans(run_mod, "a"))
    assert any("2.50" in line and "$" in line for line in env.logs)


def test_token_never_reaches_logs_or_files(run_mod, tmp_path):
    clock = Clock()
    env = Env(clock, fail={"a"})
    go(run_mod, tmp_path, env, clock, plans(run_mod, "a"))
    assert not any(TOKEN in line for line in env.logs)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert TOKEN not in path.read_text()


def test_module_source_holds_no_token(run_mod):
    source = (HARNESS / "f3_run.py").read_text()
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source)


def test_default_destroy_command_uses_no_account_key(run_mod):
    cmd = run_mod.destroy_command({"CONTAINER_ID": "123", "CONTAINER_API_KEY": "k"})
    assert cmd[:3] == ["vastai", "destroy", "instance"]
    assert "123" in cmd
    assert "VAST_API_KEY" not in " ".join(cmd)


def test_destroy_command_requires_container_id(run_mod):
    with pytest.raises(RuntimeError):
        run_mod.destroy_command({})


def test_restart_reports_failed_arm_as_failed(run_mod, tmp_path):
    clock = Clock()
    go(run_mod, tmp_path, Env(clock, fail={"a"}), clock, plans(run_mod, "a"))
    again = go(run_mod, tmp_path, Env(clock), clock, plans(run_mod, "a"))
    assert again.outcomes["a"] == "failed"


def test_budget_stop_skips_later_shorter_arms(run_mod, tmp_path):
    clock = Clock()
    env = Env(clock, seconds=3600.0)
    arms = [
        run_mod.ArmPlan("a", 3600.0),
        run_mod.ArmPlan("big", 99 * 3600.0),
        run_mod.ArmPlan("small", 60.0),
    ]
    result = go(run_mod, tmp_path, env, clock, arms, time_limit_seconds=1000 * 3600)
    assert result.outcomes["small"] == "skipped: budget"


def test_destroy_needs_the_container_key(run_mod):
    with pytest.raises(RuntimeError):
        run_mod.destroy_command({"CONTAINER_ID": "1"})
