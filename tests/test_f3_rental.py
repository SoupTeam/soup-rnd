"""Tests for the F3 rental entry point (benchmarks/harness/f3_rental.py, F3 ticket 08).

Fakes only: no network, no GPU, no vast.ai.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def rental():
    sys.path.insert(0, str(HARNESS))
    name = "_f3_rental_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "f3_rental.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        sys.path.remove(str(HARNESS))
    return module


def test_arm_command_uses_batch_16_and_the_venv_python(rental):
    command = rental.arm_command("unsloth", Path("/tmp/out.json"), steps=20)
    assert command[0].endswith("f3-unsloth/bin/python")
    assert command[command.index("--batch") + 1] == "16"
    assert command[command.index("--model") + 1] == "Qwen/Qwen2.5-0.5B"
    assert command[command.index("--steps") + 1] == "20"
    assert "--control" not in command


def test_c1_specs_audit_unsloth_and_scope_out_chronicals(rental):
    specs = rental.c1_specs()
    assert [s.arm for s in specs] == ["A-unsloth", "B-chronicals"]
    assert specs[0].uses_trainer is True
    assert specs[1].uses_trainer is False
    assert "custom loop" in specs[1].scope_reason


def test_sleep_observation_records_which_checks_answer(rental):
    class Watched:
        record = {
            "outcome": "valid",
            "checks": {
                "suspend": {"outcome": "valid", "reason": "", "evidence": {}},
                "power": {"outcome": "unknown", "reason": "no battery", "evidence": {}},
                "foreign_reads": {"outcome": "unknown", "reason": "no device", "evidence": {}},
            },
        }

    class Watch:
        def __enter__(self):
            return Watched()

        def __exit__(self, *exc):
            return None

    slept = []
    result = rental.observe_sleep(lambda **kw: Watch(), slept.append, seconds=60)
    assert slept == [60]
    assert result["answering"] == ["suspend"]
    assert result["not_answering"] == ["power", "foreign_reads"]
    assert result["label"] == "A100-PCIE-80GB vast.ai, real"


def test_sleep_observation_without_a_record_answers_nothing(rental):
    class Watch:
        def __enter__(self):
            return type("W", (), {"record": None})()

        def __exit__(self, *exc):
            return None

    result = rental.observe_sleep(lambda **kw: Watch(), lambda s: None)
    assert result["answering"] == []
    assert result["outcome"] == "unknown"


# --- cmd_c1 through its public entry point (review of PR 18) -----------------------------


def refusing(check_names=("grad_norm", "parameters_changed")):
    return {
        "valid": False,
        "failures": [{"check": name, "message": f"{name} refused"} for name in check_names],
        "checks": {"grad_norm": "failed", "parameters_changed": False},
        "provenance": {"weight_decay": 0.0},
    }


def training_report(path):
    return {
        "valid": True,
        "failures": [],
        "checks": {"trainable_parameters": 290, "grad_norm": "finite", "parameters_changed": True},
        "timing": {"median_seconds": 0.5},
        "report_path": str(path),
    }


class World:
    """Fake instance: training, watch, upload and destroy are recorded in one event list."""

    def __init__(self, rental, monkeypatch, tmp_path, *, ceiling=2.5, upload_fail=()):
        import json

        self.events = []
        self.uploaded = {}
        self.upload_fail = set(upload_fail)
        self.network_down = False
        self.workdir = tmp_path / "f3-results"
        self.workdir.mkdir()
        for path in ("native", "unsloth"):
            (self.workdir / f"control-{path}.json").write_text(json.dumps(refusing()))
        world = self

        def run_arm(path, out, steps=20):
            world.events.append(("train", out.name))
            out.parent.mkdir(parents=True, exist_ok=True)
            # The call count makes every report's bytes differ, so a replaced report shows.
            report = dict(training_report(out), marker=len(world.trained()))
            out.write_text(json.dumps(report))
            return report

        class Watched:
            record = None

            def __init__(self, kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.record = {"outcome": "ok", "arm": self.kwargs["arm"], "checks": {}}

        def upload_factory(repo):
            def upload(name, path):
                if world.network_down or name in world.upload_fail:
                    raise OSError("network down")
                world.events.append(("upload", name))
                world.uploaded[name] = path.read_text(encoding="utf-8")

            return upload

        monkeypatch.setattr(rental, "WORKDIR", self.workdir)
        monkeypatch.setattr(rental, "CEILING", ceiling)
        monkeypatch.setattr(rental, "LEASE_FILE", tmp_path / "no-lease-file")
        monkeypatch.setattr(rental, "run_arm", run_arm)
        monkeypatch.setattr(rental, "_watch", lambda **kw: Watched(kw))
        monkeypatch.setattr(rental.f3_run, "default_upload_factory", upload_factory)
        monkeypatch.setattr(
            rental.f3_run, "default_destroy", lambda: world.events.append(("destroy", None))
        )
        monkeypatch.setenv("F3_HF_REPO", "someone/private-repo")

    def trained(self):
        return [name for kind, name in self.events if kind == "train"]


def test_a_zero_ceiling_trains_nothing_and_still_destroys(rental, monkeypatch, tmp_path):
    world = World(rental, monkeypatch, tmp_path, ceiling=0.0)
    assert rental.cmd_c1(1.0) == 0
    assert world.trained() == []
    assert world.events[-1] == ("destroy", None)
    import json

    summary = json.loads(world.uploaded["summary"])
    assert summary["outcomes"]["A-unsloth"] == "skipped: budget"


def test_the_paid_lease_start_counts_against_the_ceiling(rental, monkeypatch, tmp_path):
    import time

    world = World(rental, monkeypatch, tmp_path, ceiling=2.5)
    # 2.45 h already paid at 1.00 $/h: the 300 s arm would pass the 2.50 $ ceiling.
    assert rental.cmd_c1(1.0, lease_start=time.time() - 2.45 * 3600) == 0
    assert world.trained() == []


def test_the_lease_start_before_the_arm_lets_a_cheap_lease_train(rental, monkeypatch, tmp_path):
    import time

    world = World(rental, monkeypatch, tmp_path, ceiling=2.5)
    assert rental.cmd_c1(1.0, lease_start=time.time() - 3600) == 0
    assert world.trained() == ["A-unsloth-a1-p1.json", "A-unsloth-a1-p2.json"]


def test_a_broken_control_file_still_ends_in_upload_and_destroy(rental, monkeypatch, tmp_path):
    world = World(rental, monkeypatch, tmp_path)
    (world.workdir / "control-native.json").write_text("{not json")
    assert rental.cmd_c1(1.0) == 0
    assert world.trained() == []
    assert world.events[-1] == ("destroy", None)
    assert "A-unsloth" in world.uploaded


def test_the_raw_evidence_is_uploaded_before_destroy(rental, monkeypatch, tmp_path):
    import json

    world = World(rental, monkeypatch, tmp_path)
    log = tmp_path / "c1.log"
    log.write_text("training log")
    assert rental.cmd_c1(1.0, logs=(log,)) == 0
    for name in (
        "c1/A-unsloth-a1-p1.json",
        "c1/A-unsloth-a1-p2.json",
        "audit-c1.json",
        "refusals.jsonl",
        "control-native.json",
        "control-unsloth.json",
        "c1.log",
    ):
        assert name in world.uploaded, name
    arm = json.loads(world.uploaded["A-unsloth"])
    assert len(arm["records"]) == 2
    assert arm["records"][0]["watch"]["outcome"] == "ok"
    assert arm["records"][0]["report"]["valid"] is True
    audit = json.loads(world.uploaded["audit-c1.json"])
    assert [v["verdict"] for v in audit["verdicts"]] == ["pass", "out of scope"]
    assert len(audit["records"]) == 2
    uploads = [i for i, (kind, _) in enumerate(world.events) if kind == "upload"]
    assert max(uploads) < world.events.index(("destroy", None))


def test_a_failed_evidence_upload_keeps_the_instance(rental, monkeypatch, tmp_path):
    world = World(rental, monkeypatch, tmp_path, upload_fail={"c1/A-unsloth-a1-p2.json"})
    rental.cmd_c1(1.0)
    assert ("destroy", None) not in world.events


def test_a_resume_after_an_upload_outage_keeps_the_original_reports(
    rental, monkeypatch, tmp_path
):
    world = World(rental, monkeypatch, tmp_path)
    world.network_down = True
    rental.cmd_c1(1.0)
    assert ("destroy", None) not in world.events
    raw = sorted((world.workdir / "c1").glob("A-unsloth-*.json"))
    originals = {f"c1/{path.name}": path.read_bytes() for path in raw}
    assert sorted(originals) == [
        "c1/A-unsloth-a1-p1.json", "c1/A-unsloth-a1-p2.json", "c1/A-unsloth-a1.json"
    ]
    world.network_down = False
    rental.cmd_c1(1.0)
    assert len(world.trained()) == 2
    for name, data in originals.items():
        assert (world.workdir / name).read_bytes() == data
        assert world.uploaded[name] == data.decode("utf-8")
    assert world.events[-1] == ("destroy", None)


def test_an_explicit_rerun_writes_a_new_attempt_and_keeps_the_old_one(
    rental, monkeypatch, tmp_path
):
    world = World(rental, monkeypatch, tmp_path)
    rental.cmd_c1(1.0)
    raw = world.workdir / "c1"
    names = ("A-unsloth-a1-p1.json", "A-unsloth-a1.json")
    first = {name: (raw / name).read_bytes() for name in names}
    (world.workdir / "lifecycle" / "state.json").unlink()
    rental.cmd_c1(1.0)
    assert world.trained() == [
        "A-unsloth-a1-p1.json",
        "A-unsloth-a1-p2.json",
        "A-unsloth-a2-p1.json",
        "A-unsloth-a2-p2.json",
    ]
    for name, data in first.items():
        assert (raw / name).read_bytes() == data
    import json

    record = json.loads(first["A-unsloth-a1.json"])
    assert record["attempt"] == 1
    assert [r["watch"]["outcome"] for r in record["records"]] == ["ok", "ok"]
    assert json.loads(world.uploaded["c1/A-unsloth-a2.json"])["attempt"] == 2


def test_a_crash_inside_an_arm_reruns_it_as_a_new_attempt(rental, monkeypatch, tmp_path):
    world = World(rental, monkeypatch, tmp_path)
    train = rental.run_arm

    def killed_after_the_first_process(path, out, steps=20):
        report = train(path, out, steps)
        if len(world.trained()) == 1:
            raise KeyboardInterrupt  # the process dies, nothing reaches the state file
        return report

    monkeypatch.setattr(rental, "run_arm", killed_after_the_first_process)
    with pytest.raises(KeyboardInterrupt):
        rental.cmd_c1(1.0)
    partial = (world.workdir / "c1" / "A-unsloth-a1-p1.json").read_bytes()
    rental.cmd_c1(1.0)
    assert world.trained() == [
        "A-unsloth-a1-p1.json", "A-unsloth-a2-p1.json", "A-unsloth-a2-p2.json"
    ]
    assert (world.workdir / "c1" / "A-unsloth-a1-p1.json").read_bytes() == partial
    assert "c1/A-unsloth-a1-p1.json" in world.uploaded
    assert world.events[-1] == ("destroy", None)


def test_run_arm_never_overwrites_a_report(rental, monkeypatch, tmp_path):
    out = tmp_path / "c1" / "A-unsloth-a1-p1.json"
    out.parent.mkdir()
    out.write_text("original")
    calls = []
    monkeypatch.setattr(rental.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(FileExistsError):
        rental.run_arm("unsloth", out)
    assert calls == []
    assert out.read_text() == "original"
