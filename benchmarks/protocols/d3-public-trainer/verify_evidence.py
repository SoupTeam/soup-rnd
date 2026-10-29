"""Read-only verification of prospective preparation; never run a model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            result.update(block)
    return result.hexdigest()


def check_archive(path):
    count, nested = 0, 0
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        if set(archive.namelist()) != set(manifest) | {"MANIFEST.json"}:
            raise ValueError("archive payload set differs: " + str(path))
        for name, expected in manifest.items():
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("archive hash differs: " + name)
            count += 1
            if name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as child:
                    if child.testzip() is not None:
                        raise ValueError("nested archive CRC failed: " + name)
                nested += 1
        if archive.testzip() is not None:
            raise ValueError("archive CRC failed: " + str(path))
    return {"payload_hashes": count, "nested_crcs": nested, "crc": "passed"}


def verify(folder, backup_directory=None):
    root = folder.parents[2]
    sys.path.insert(0, str(root))
    from benchmarks.harness.d3_trainer_protocol import contained, launch_blockers, read_rule

    manifest = json.loads((folder / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        if digest(contained(folder, name)) != expected:
            raise ValueError("published hash differs: " + name)
    blockers = launch_blockers(read_rule(root), root, check_git=False)
    if blockers:
        raise ValueError("preparation/runtime no longer matches: " + "; ".join(blockers))
    host = json.loads((folder / "final-host-check.json").read_text())
    if host["status"] != "prepared_data_and_host_identities_verified" or (
        host["prepared_sha256"] != read_rule(root)["preparation"]["shards"]["sha256"]
    ):
        raise ValueError("final host/preparation receipt differs")
    rule = read_rule(root)
    prior = rule.get("prior_formal_attempt", {})
    result = {
        "published_hashes": len(manifest),
        "setup": check_archive(folder / "setup-evidence.zip"),
        "cpu": check_archive(folder / "cpu-evidence.zip"),
        "preparation_runtime_bindings": "passed",
        "new_revision_formal_execution_started": False,
        "pretrained_trainer_verdict": (
            f"prior session {prior.get('git_head', 'unknown')} is "
            f"{prior.get('status', 'unrecorded')}; no timing verdict; "
            f"revision {rule.get('protocol_revision')} not measured"
        ),
        "full_task_accepted": False,
    }
    if read_rule(root).get("protocol_revision", 1) >= 2:
        result["corrective_untied"] = check_archive(folder / "untied-preflight.zip")
        result["corrective_cpu"] = check_archive(folder / "r2-cpu-evidence.zip")
        previous = json.loads((folder / "previous-attempt.json").read_text())
        if previous["status"] != "fail" or previous["accepted_cells"] != 0:
            raise ValueError("the failed first session was reclassified")
        result["previous_attempt_retained"] = {
            "first_formal_failure": previous["formal_source"],
            "latest_incomplete_session": prior.get("git_head"),
            "latest_status": prior.get("status"),
            "latest_cells_ok": prior.get("cells_ok"),
        }
    if (folder / "ci-repair-evidence.zip").is_file():
        result["ci_repair"] = check_archive(folder / "ci-repair-evidence.zip")
        result["linux_ci_reproduction"] = check_archive(folder / "d3-ci-linux-evidence.zip")
        result["restart_rediscovery"] = check_archive(folder / "d3-restart-evidence.zip")
    if backup_directory is not None:
        with zipfile.ZipFile(folder / "setup-evidence.zip") as archive:
            full = json.loads(archive.read("FULL-ARCHIVE-RETENTION.json"))["retained_full_archive"]
        native = json.loads((folder / "native-supplement.json").read_text())
        result["local_backups"] = {}
        receipts = [full, native]
        if read_rule(root).get("protocol_revision", 1) >= 2:
            receipts += [
                json.loads((folder / "untied-full-archive.json").read_text()),
                previous["full_archive"],
            ]
        for receipt in receipts:
            path = contained(backup_directory, receipt["filename"])
            if digest(path) != receipt["sha256"]:
                raise ValueError("local backup hash differs: " + str(path))
            result["local_backups"][path.name] = check_archive(path)
    return result


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-directory", type=Path)
    args = parser.parse_args()
    Console().print_json(
        json.dumps(verify(Path(__file__).resolve().parent, args.backup_directory), indent=2)
    )
