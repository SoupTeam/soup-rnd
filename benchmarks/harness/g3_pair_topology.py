"""Offline, conservative two-drive topology gate. Never launches CUDA or a lease.

Input records are historical, unauthenticated evidence. A positive result only
qualifies a candidate for live review; it cannot prove direct transport or speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

from rich.console import Console


def assess_pair(rows: list[dict], *, now: float) -> dict:
    """Require separate namespaces, controllers, serials and mount identities."""
    reasons = []
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError("Finite observation time required")
    if not isinstance(rows, list) or len(rows) != 2:
        reasons.append("Exactly two records required")
    else:
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                reasons.append(f"Record {index} is not an object")
                continue
            stamp = row.get("unix_s")
            if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                    or not 0 <= now - stamp <= 300):
                reasons.append(f"Record {index}: stale, future or invalid timestamp")
            for field in ("boot_id", "gpu_uuid", "serial", "data_path", "namespace",
                          "controller", "controller_sysfs", "mount_major_minor"):
                value = row.get(field)
                if (not isinstance(value, str) or not value.strip() or value != value.strip()
                        or value.strip().lower() in ("unknown", "none", "n/a", "unconfigured")):
                    reasons.append(f"Record {index}: missing {field}")
            for field, pattern in (("boot_id", r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}"),
                                   ("gpu_uuid", r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}"
                                                r"-[a-fA-F0-9]{12}")):
                value = row.get(field)
                if not isinstance(value, str) or not re.fullmatch(pattern, value):
                    reasons.append(f"Record {index}: invalid {field}")
            namespace = row.get("namespace", "")
            controller = row.get("controller", "")
            if (not isinstance(namespace, str) or not isinstance(controller, str)
                    or not re.fullmatch(r"nvme[0-9]+", controller)
                    or not re.fullmatch(re.escape(controller) + r"n[0-9]+", namespace)):
                reasons.append(f"Record {index}: unresolved NVMe controller/namespace")
            if row.get("fstype") not in ("ext4", "xfs"):
                reasons.append(f"Record {index}: filesystem is not ext4/xfs")
            if row.get("topology_status") != "ATTRIBUTABLE_LOCAL_NVME":
                reasons.append(f"Record {index}: underlying NVMe mapping missing")
            for field in ("data_path", "controller_sysfs"):
                value = row.get(field)
                if (not isinstance(value, str) or not value.startswith("/")
                        or str(PurePosixPath(value)) != value or ".." in value.split("/")):
                    reasons.append(f"Record {index}: noncanonical {field}")
            sysfs = row.get("controller_sysfs", "")
            if not isinstance(sysfs, str) or not sysfs.startswith("/sys/devices/pci"):
                reasons.append(f"Record {index}: physical PCI controller path missing")
            mount = row.get("mount_major_minor", "")
            if not isinstance(mount, str) or not re.fullmatch(r"[0-9]+:[0-9]+", mount):
                reasons.append(f"Record {index}: invalid mount device identity")
        if all(isinstance(row, dict) for row in rows):
            for field in ("boot_id", "gpu_uuid"):
                if rows[0].get(field) != rows[1].get(field):
                    reasons.append(f"Records disagree on {field}")
            for field in ("data_path", "namespace", "controller", "serial",
                          "controller_sysfs", "mount_major_minor"):
                if rows[0].get(field) == rows[1].get(field):
                    reasons.append(f"Shared {field}; two physical drives unproven")
    return {
        "status": ("BLOCKED_PAIR_TOPOLOGY" if reasons else
                   "TWO_DRIVE_CANDIDATE_REQUIRES_LIVE_REVIEW"),
        "reasons": reasons, "execution_authorized": False,
        "performance_verdict": "NO_VERDICT", "direct_path_verified": False,
        "remaining": ["Live sysfs and mount recollection, same boot/GPU",
                      "Independent F1 calibration on each disk",
                      "Audited direct capability on each disk",
                      "Frozen bounded concurrent runner and overlap evidence",
                      "Independent byte and transport audit for both readers"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--observed-unix-s", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.records.read_bytes()
    result = assess_pair(json.loads(raw), now=args.observed_unix_s)
    result.update(input_sha256=hashlib.sha256(raw).hexdigest(),
                  gate_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  observed_unix_s=args.observed_unix_s)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    Console().print(result["status"])


if __name__ == "__main__":
    main()
