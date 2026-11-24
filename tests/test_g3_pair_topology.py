"""Reject false two-drive claims before any prospective G3 concurrent run."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))
from g3_pair_topology import assess_pair  # noqa: E402


def records():
    return [{"boot_id": "12345678-1234-1234-1234-123456789abc",
             "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc", "unix_s": 1000,
             "data_path": f"/data/{index}", "namespace": f"nvme{index}n1",
             "controller": f"nvme{index}", "serial": f"SERIAL-{index}",
             "controller_sysfs": f"/sys/devices/pci0000:00/0000:0{index}:00.0",
             "mount_major_minor": f"259:{index}", "fstype": "xfs",
             "topology_status": "ATTRIBUTABLE_LOCAL_NVME"} for index in range(2)]


def test_distinct_drives_are_only_candidates():
    result = assess_pair(records(), now=1010)
    assert result["status"] == "TWO_DRIVE_CANDIDATE_REQUIRES_LIVE_REVIEW"
    assert result["execution_authorized"] is False
    assert result["performance_verdict"] == "NO_VERDICT"


@pytest.mark.parametrize("field", ["data_path", "namespace", "controller", "serial",
                                  "controller_sysfs", "mount_major_minor"])
def test_alias_or_shared_physical_identity_rejected(field):
    rows = records()
    rows[1][field] = rows[0][field]
    assert assess_pair(rows, now=1010)["status"] == "BLOCKED_PAIR_TOPOLOGY"


@pytest.mark.parametrize("field,value", [("boot_id", "other"), ("gpu_uuid", "other"),
                                        ("serial", ""), ("serial", "unknown"),
                                        ("serial", " SERIAL-1 "),
                                        ("fstype", "overlay"), ("unix_s", 1),
                                        ("unix_s", 9999), ("unix_s", float("nan")),
                                        ("unix_s", True), ("controller_sysfs", "/tmp/fake"),
                                        ("namespace", "loop0"),
                                        ("topology_status", "UNKNOWN")])
def test_incomplete_or_mismatched_evidence_rejected(field, value):
    rows = records()
    rows[1][field] = value
    assert assess_pair(rows, now=1010)["status"] == "BLOCKED_PAIR_TOPOLOGY"


def test_namespace_controller_mismatch_and_partition_rejected():
    for namespace in ["nvme5n1", "nvme1n1p1"]:
        rows = records()
        rows[1]["namespace"] = namespace
        assert assess_pair(rows, now=1010)["status"] == "BLOCKED_PAIR_TOPOLOGY"


def test_one_drive_two_namespaces_is_not_two_drives():
    rows = records()
    rows[1].update(namespace="nvme0n2", controller="nvme0")
    assert assess_pair(rows, now=1010)["status"] == "BLOCKED_PAIR_TOPOLOGY"


def test_inputs_are_not_mutated_and_wrong_cardinality_fails():
    rows = records()
    before = copy.deepcopy(rows)
    assess_pair(rows, now=1010)
    assert rows == before
    for invalid in [[], rows[:1], rows + rows[:1]]:
        assert assess_pair(invalid, now=1010)["status"] == "BLOCKED_PAIR_TOPOLOGY"
