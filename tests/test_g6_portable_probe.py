"""Portable admission fails closed and preserves the frozen scientific comparison."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BASE = ROOT / "benchmarks/g6"
sys.path.insert(0, str(BASE))
LOADER = importlib.util.spec_from_file_location(
    "portable_probe", BASE / "portable_generation_probe.py"
)
PROBE = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(PROBE)


def inputs():
    spec = {"total_s": 2400}
    launch = {"account": "test-project", "instance_id": 123, "gpu_uuid": "GPU-example",
              "account_floor_usd": 7.5, "stop_only_this_instance_after_preservation": True,
              "contract_sha256": "science"}
    receipt = {"account": "test-project", "mode": "generation-probe",
               "contract_sha256": "science", "launch_config_sha256": "launch",
               "observed_unix_s": 1000, "balance_usd": 12,
               "total_account_rate_usd_h": .6, "instance_id": 123,
               "gpu_uuid": "GPU-example", "authorized_for_this_stage": True}
    return spec, launch, receipt


@pytest.mark.parametrize("field,value", [
    ("balance_usd", 8), ("balance_usd", float("nan")), ("balance_usd", True),
    ("total_account_rate_usd_h", 0), ("total_account_rate_usd_h", float("inf")),
    ("observed_unix_s", 699), ("observed_unix_s", 1001),
    ("instance_id", 456), ("gpu_uuid", "GPU-other"), ("mode", "train"),
    ("account", "other-project"), ("authorized_for_this_stage", False),
    ("contract_sha256", "other-science"), ("launch_config_sha256", "other-launch"),
])
def test_fresh_stage_is_bound_to_approved_private_settings(field, value):
    spec, launch, receipt = inputs()
    PROBE.admit_probe(spec, launch, receipt, "science", "launch", 1000)
    receipt[field] = value
    with pytest.raises(ValueError):
        PROBE.admit_probe(spec, launch, receipt, "science", "launch", 1000)


@pytest.mark.parametrize("field,value", [
    ("account", ""), ("instance_id", 0), ("instance_id", True),
    ("gpu_uuid", "unknown"), ("account_floor_usd", -1),
    ("account_floor_usd", float("nan")), ("account_floor_usd", True),
    ("stop_only_this_instance_after_preservation", False),
])
def test_missing_or_invalid_authority_has_no_permissive_default(field, value):
    _, launch, _ = inputs()
    launch[field] = value
    with pytest.raises(ValueError):
        PROBE.validate_launch(launch)
    launch.pop(field)
    with pytest.raises(ValueError):
        PROBE.validate_launch(launch)


def test_changed_private_floor_and_science_need_a_new_admission():
    spec, launch, receipt = inputs()
    launch["account_floor_usd"] = 11.9
    with pytest.raises(ValueError, match="balance floor"):
        PROBE.admit_probe(spec, launch, receipt, "science", "launch", 1000)
    launch["account_floor_usd"] = 7.5
    with pytest.raises(ValueError, match="configuration mismatch"):
        PROBE.admit_probe(spec, launch, receipt, "science", "changed-launch-sha", 1000)
    launch["contract_sha256"] = "different"
    with pytest.raises(ValueError, match="configuration mismatch"):
        PROBE.admit_probe(spec, launch, receipt, "science", "launch", 1000)


@pytest.mark.parametrize("duration", [-1, True, 2401])
def test_execution_deadline_cannot_expand(duration):
    spec, launch, receipt = inputs()
    spec["total_s"] = duration
    with pytest.raises(ValueError):
        PROBE.admit_probe(spec, launch, receipt, "science", "launch", 1000)


def test_private_file_tampering_is_rejected_before_model_loading(tmp_path):
    spec, launch, _ = inputs()
    spec_path = tmp_path / "science.json"
    launch_path = tmp_path / "private.json"
    spec_path.write_text(json.dumps(spec))
    launch_path.write_text(json.dumps(launch))
    launch_sha = hashlib.sha256(launch_path.read_bytes()).hexdigest()
    launch["account_floor_usd"] = 0
    launch_path.write_text(json.dumps(launch))
    args = argparse.Namespace(spec=spec_path,
                              spec_sha=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                              launch_config=launch_path, launch_config_sha=launch_sha)
    with pytest.raises(ValueError, match="Private launch configuration SHA mismatch"):
        PROBE.execute(args)


def test_launcher_requires_explicit_host_and_gpu_before_python():
    env = dict(os.environ)
    for key in ("SOUP_G6_INSTANCE_ID", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    result = subprocess.run(["sh", str(BASE / "launch_portable_probe.sh")], env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert "approved instance ID" in result.stderr


def test_generation_and_stopping_match_historical_science():
    trees = [ast.parse((BASE / name).read_text()) for name in
             ["paired_generation_probe.py", "portable_generation_probe.py"]]
    for name in ("stopping", "generate_pass"):
        functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == name) for tree in trees]
        assert ast.dump(functions[0]) == ast.dump(functions[1])


def test_public_contract_keeps_science_and_separates_operational_fields():
    old = json.loads((BASE / "spec/paired_generation_probe_56900314.json").read_text())
    new = json.loads((BASE / "spec/portable_paired_generation.json").read_text())
    private_fields = {"instance_id", "gpu_uuid", "budget"}
    assert not private_fields.intersection(new)
    for key in old.keys() - private_fields - {"version", "source_sha256", "before_launch"}:
        assert new[key] == old[key], key
    for name, digest in new["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
