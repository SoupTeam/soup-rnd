"""Regression checks for portable A0 measurement and its fail-closed session runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.test_l2l_probe_harness import _corr, _load, _timing, _write


def _observed_timing():
    timing = _timing()
    for row in timing["records"]:
        row["suspend"] = {"samples": 2, "unknown_power_samples": 0, "battery_samples": 0,
                          "max_gap_s": 2, "void": False}
        row["foreign_readers"] = []
        row["box_before"] = {"pid": 7, "gpu_pids": [7]}
        row["box_after"] = {"pid": 7, "gpu_pids": [7]}
    return timing


def test_strict_rule_accepts_observed_clean_arms(tmp_path):
    rule = _load("l2l_rule")
    result = rule.evaluate(_write(tmp_path, timing=_observed_timing()), strict=True)
    assert result["verdict"] == "BUILD STEP 1"


@pytest.mark.parametrize("missing", ["suspend", "foreign_readers", "box_before", "box_after"])
def test_missing_telemetry_cannot_pass_a_new_session(tmp_path, missing):
    timing = _observed_timing()
    for row in timing["records"]:
        if (row["mode"], row["arm"], row["k"]) == ("l2l", "A", 16):
            row.pop(missing)
    rule = _load("l2l_rule")
    result = rule.evaluate(_write(tmp_path, timing=timing), strict=True)
    assert all(result["gates"][key]["status"] == "no verdict" for key in ("G2", "G3", "G4"))


@pytest.mark.parametrize("change", ["power", "gap", "peer", "nan", "duplicate", "tokens"])
def test_bad_arm_cannot_pass_strict_validity(tmp_path, change):
    timing = _observed_timing()
    row = next(row for row in timing["records"]
               if (row["mode"], row["arm"], row["k"], row["round"]) == ("l2l", "A", 16, 1))
    if change == "power":
        row["suspend"]["unknown_power_samples"] = 1
    elif change == "gap":
        row["suspend"]["max_gap_s"] = 31
    elif change == "peer":
        row["box_after"]["gpu_pids"] = [7, 9]
    elif change == "nan":
        row["step_s_mean"] = float("nan")
    elif change == "duplicate":
        row["round"] = 0
    else:
        row["tokens_per_step"] = 512
    result = _load("l2l_rule").evaluate(_write(tmp_path, timing=timing), strict=True)
    assert all(result["gates"][key]["status"] == "no verdict" for key in ("G2", "G3", "G4"))


def test_linux_memory_preserves_zero_and_does_not_use_commit_limit():
    box = _load("l2l_box")
    result = box.linux_memory_status("MemTotal: 32768000 kB\nMemAvailable: 0 kB\n")
    assert result["total_phys_gb"] == pytest.approx(32768000 * 1024 / 1e9)
    assert result["avail_phys_gb"] == 0
    assert result["commit_avail_gb"] is None


def test_container_memory_is_capped_by_its_cgroup(monkeypatch):
    box = _load("l2l_box")
    monkeypatch.setattr(box.sys, "platform", "linux")
    files = {"/proc/meminfo": "MemTotal: 256000000 kB\nMemAvailable: 128000000 kB\n",
             "/sys/fs/cgroup/memory.max": "32000000000",
             "/sys/fs/cgroup/memory.current": "12000000000"}

    def read_text(path, *args, **kwargs):
        if path.as_posix() not in files:
            raise FileNotFoundError(path)
        return files[path.as_posix()]

    monkeypatch.setattr(Path, "read_text", read_text)
    result = box.memory_status()
    assert result["total_phys_gb"] == 32
    assert result["avail_phys_gb"] == 20


@pytest.mark.parametrize(("v1", "stats", "available_kb", "expected"), [
    (False, "inactive_file 22000000000\nactive_file 2000000000\n", 128000000, 30),
    (True, "total_inactive_file 22000000000\ninactive_file 1000\n", 128000000, 30),
    (False, "inactive_file invalid\n", 128000000, 8),
    (False, "active_file 22000000000\n", 128000000, 8),
    (False, "inactive_file 99000000000\n", 128000000, 32),
    (False, "inactive_file 22000000000\n", 1000000, 1.024),
    (False, "inactive_file 22000000000\n", 0, 0),
])
def test_cgroup_reclaimable_cache_preserves_host_and_container_limits(
        monkeypatch, v1, stats, available_kb, expected):
    box = _load("l2l_box")
    monkeypatch.setattr(box.sys, "platform", "linux")
    parent = "/sys/fs/cgroup/memory" if v1 else "/sys/fs/cgroup"
    limit = "memory.limit_in_bytes" if v1 else "memory.max"
    used = "memory.usage_in_bytes" if v1 else "memory.current"
    files = {"/proc/meminfo": f"MemTotal: 256000000 kB\nMemAvailable: {available_kb} kB\n",
             f"{parent}/{limit}": "32000000000", f"{parent}/{used}": "24000000000",
             f"{parent}/memory.stat": stats}

    def read_text(path, *args, **kwargs):
        if path.as_posix() not in files:
            raise FileNotFoundError(path)
        return files[path.as_posix()]

    monkeypatch.setattr(Path, "read_text", read_text)
    result = box.memory_status()
    assert result["total_phys_gb"] == 32
    assert result["avail_phys_gb"] == pytest.approx(expected)


def test_unknown_gpu_process_list_refuses_v3(monkeypatch):
    probe = _load("l2l_probe")
    monkeypatch.setattr(probe.l2l_box, "memory_status",
                        lambda: {"avail_phys_gb": 30, "commit_avail_gb": None})
    monkeypatch.setattr(probe.l2l_box, "gpu_compute_pids", lambda: None)
    assert probe.v3_check(1000, 0)["ok"] is False


def test_nvidia_smi_failure_is_unknown(monkeypatch):
    box = _load("l2l_box")
    monkeypatch.setattr(box, "_nvidia_smi", lambda query: "[Not Supported]\n")
    assert box.gpu_compute_pids() is None


def test_cloud_power_declaration_does_not_hide_a_discharging_battery(tmp_path, monkeypatch):
    box = _load("l2l_box")
    monkeypatch.setattr(box.sys, "platform", "linux")
    monkeypatch.setenv("SOUP_L2L_POWER_SOURCE", "server")
    supply = tmp_path / "BAT0"
    supply.mkdir()
    (supply / "status").write_text("Discharging")
    monkeypatch.setattr(Path, "iterdir", lambda path: iter([supply]))
    assert box.on_ac_power() is False


def _session_args(tmp_path):
    return ["--f1", str(tmp_path / "f1"), "--f2", str(tmp_path / "f2"),
            "--f1-shards", str(tmp_path / "cache1"), "--f2-shards", str(tmp_path / "cache2"),
            "--spill-dir", str(tmp_path / "spill"), "--out", str(tmp_path / "attempt")]


def test_planning_never_executes_a_child_or_overwrites_an_attempt(tmp_path, monkeypatch):
    session = _load("l2l_session")
    monkeypatch.setattr(session, "git_head", lambda: "base-commit")
    monkeypatch.setattr(session, "source_hashes", lambda: {})
    monkeypatch.setattr(session, "versions", lambda: {})
    monkeypatch.setattr(session, "run_child", lambda *args: pytest.fail("planning executed CUDA"))
    args = _session_args(tmp_path)
    assert session.main(args) == 0
    protocol_path = tmp_path / "attempt" / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["protocol"]["profile"] == "a0-strict-v1"
    assert protocol["plan"][4]["name"] == "timing"
    original = protocol_path.read_bytes()
    with pytest.raises(FileExistsError):
        session.main(args)
    assert protocol_path.read_bytes() == original


def test_child_skip_without_json_is_not_success(tmp_path, monkeypatch):
    session = _load("l2l_session")
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(session.l2l_box, "box_stamp", lambda: {})
    monkeypatch.setattr(session.l2l_box, "process_read_bytes", lambda: {})
    entry = {"name": "skipped", "output": str(tmp_path / "missing.json"),
             "command": [sys.executable, "-c", "pass"]}
    with pytest.raises(FileNotFoundError):
        session.run_child(entry, tmp_path, dict(session.os.environ), session.time.monotonic() + 30)


def test_session_stops_after_a_crashed_child_and_records_incomplete(tmp_path, monkeypatch):
    session = _load("l2l_session")
    monkeypatch.setattr(session, "git_head", lambda: "base-commit")
    monkeypatch.setattr(session, "source_hashes", lambda: {})
    monkeypatch.setattr(session, "versions", lambda: {})
    monkeypatch.setattr(session.subprocess, "run",
                        lambda *args, **kwargs: type("Result", (), {"returncode": 0})())
    calls = []

    def fake_child(entry, out, env, deadline):
        calls.append(entry["name"])
        if len(calls) == 2:
            raise RuntimeError("crashed before output")
        session.write_json(Path(entry["output"]), _corr("F1"))
        return {"name": entry["name"], "returncode": 0, "oom": False}

    monkeypatch.setattr(session, "run_child", fake_child)
    assert session.main([*_session_args(tmp_path), "--execute"]) == 3
    assert calls == ["correctness_f1", "correctness_f2"]
    result = json.loads((tmp_path / "attempt" / "verdict.json").read_text())
    assert result["verdict"].startswith("SESSION INCOMPLETE")
    assert result["gates"]["G1"]["status"] == "no verdict"


def test_container_host_pid_is_not_a_foreign_gpu_process(tmp_path):
    timing = _observed_timing()
    for row in timing["records"]:
        for key in ("box_before", "box_after"):
            row[key].update(own_pids=[7, 9007], gpu_pids=[9007])
    result = _load("l2l_rule").evaluate(_write(tmp_path, timing=timing), strict=True)
    assert result["verdict"] == "BUILD STEP 1"


@pytest.mark.parametrize(("before", "after", "active", "single", "owned"), [
    ([], [9007], True, True, True),
    (None, [9007], True, True, False),
    ([9], [9007], True, True, False),
    ([], None, True, True, False),
    ([], [9007, 8], True, True, False),
    ([], [], True, True, False),
    ([], [9007], False, True, False),
    ([], [9007], True, False, False),
])
def test_gpu_host_pid_attribution_requires_exclusive_live_context(
        monkeypatch, before, after, active, single, owned):
    box = _load("l2l_box")
    monkeypatch.setattr(box.sys, "platform", "linux")
    monkeypatch.setattr(box, "_GPU_OWNER_PID", 42)
    monkeypatch.setattr(box, "own_process_ids", lambda: [7])
    monkeypatch.setattr(box, "gpu_compute_pids", lambda: after)
    evidence = box.confirm_gpu_owner(before, context_active=active, single_device=single)
    assert box.own_gpu_process_ids() == ([7, 9007] if owned else [7])
    assert box.own_process_ids() == [7]
    assert evidence["before"] == before and evidence["after"] == after
    assert (evidence["method"] == "exclusive-local-CUDA-initialization") is owned


def test_strict_evaluator_accepts_gpu_alias_without_changing_io_owner(tmp_path):
    timing = _observed_timing()
    for row in timing["records"]:
        for key in ("box_before", "box_after"):
            row[key].update(own_pids=[7], gpu_own_pids=[7, 9007], gpu_pids=[9007])
    result = _load("l2l_rule").evaluate(_write(tmp_path, timing=timing), strict=True)
    assert result["verdict"] == "BUILD STEP 1"


def test_prepare_refuses_cpu_before_any_download(tmp_path, monkeypatch):
    import huggingface_hub
    import torch

    prepare = _load("l2l_prepare")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda **kwargs: pytest.fail("downloaded without CUDA"))
    out = tmp_path / "prepare"
    assert prepare.main(["--out", str(out)]) == 3
    result = json.loads((out / "preparation.json").read_text())
    assert result["state"] == "stopped"
    assert "CUDA unavailable" in result["reason"]


def test_aligned_direct_io_preparation_probe(tmp_path):
    prepare = _load("l2l_prepare")
    prepare.direct_io_probe(tmp_path)
    assert not (tmp_path / "direct-io-probe.bin").exists()


@pytest.mark.parametrize("device_ooms,expect_bound", [(0, False), (1, True)])
def test_batch_bound_requires_a_real_cuda_allocator_oom(tmp_path, device_ooms, expect_bound):
    session = _load("l2l_session")
    timing = _observed_timing()
    completed = []
    for rnd in (0, 1):
        row = next(row.copy() for row in timing["records"]
                   if (row["mode"], row["arm"], row["k"], row["round"])
                   == ("l2l", "A", 1, rnd))
        name = f"batch_1_r{rnd}"
        session.write_json(tmp_path / f"{name}.json",
                           {"meta": {"direct_io": True, "v3": {"ok": True}}, "records": [row]})
        completed.append({"name": name, "oom": False})
    session.write_json(tmp_path / "batch_2_r0.json",
                       {"records": [{"kind": "oom", "memory": {"num_ooms": device_ooms}}]})
    completed.append({"name": "batch_2_r0", "oom": True})
    assert session.batch_summary(tmp_path, completed)["maximum_established"] is expect_bound


def test_session_retains_status_when_a_crash_leaves_invalid_json(tmp_path, monkeypatch):
    session = _load("l2l_session")
    monkeypatch.setattr(session, "git_head", lambda: "base-commit")
    monkeypatch.setattr(session, "source_hashes", lambda: {})
    monkeypatch.setattr(session, "versions", lambda: {})
    monkeypatch.setattr(session.subprocess, "run",
                        lambda *args, **kwargs: type("Result", (), {"returncode": 0})())

    def crash(entry, out, env, deadline):
        Path(entry["output"]).write_text("{")
        raise RuntimeError("interrupted while writing")

    monkeypatch.setattr(session, "run_child", crash)
    assert session.main([*_session_args(tmp_path), "--execute"]) == 3
    result = json.loads((tmp_path / "attempt" / "verdict.json").read_text())
    assert result["session"] == "stopped"
    assert result["verdict"].startswith("SESSION INCOMPLETE")
    assert "analysis_error" in result
