"""Build a reviewable evidence packet from one preserved H2-close archive (layout of C3's packets).

  python benchmarks/harness/h2_close_packet.py --extracted DIR --manifest preserved_manifest.json \
      --archive-sha256 HEX --name NAME --out benchmarks/h2/reports/20261102_h2_close/evidence/NAME

Every archive member is first checked against the preserved manifest (bytes and SHA256). Members
are then copied byte for byte, except:

* tensor and optimizer payloads, the public tokenizer and Unsloth's generated trainer cache are
  excluded by size; each keeps its bytes and SHA256 in PACKET.json (the per-tensor digests that
  the stage U gates read are inside training.json, which is included);
* kernel traces are stored gzip-compressed, with the original SHA256 recorded;
* ``rental.json`` (SSH endpoint, budget, local paths) becomes ``rental_public.json``;
* ``.py`` members are stored as ``.py.txt`` so repository lint does not run on frozen bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

PRIVATE_RENTAL_KEYS = {"ssh_host", "ssh_port", "budget_cap_usd", "cleanup_traffic_reserve_usd",
                       "local_export_directory", "lease_deadline_unix_s", "protected_resources"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusion(name: str) -> str | None:
    base = name.rsplit("/", 1)[-1]
    if base.endswith(".safetensors"):
        return "tensor payload; per-tensor SHA256 of every adapter state is in training.json"
    if base == "optimizer.pt":
        return "Trainer optimizer checkpoint; AdamW moments are also in optimizer_final.safetensors"
    if base == "tokenizer.json":
        return "public tokenizer of the pinned model revision"
    if "/unsloth_compiled_cache/" in name:
        return "Unsloth-generated trainer code written at import; not read by any gate"
    return None


def build(extracted: Path, manifest: dict, archive_sha256: str, name: str, out: Path) -> dict:
    members = {p.relative_to(extracted).as_posix(): p for p in extracted.rglob("*") if p.is_file()}
    members.pop("preserved_manifest.json", None)
    if set(members) != set(manifest):
        raise ValueError("Extracted members differ from the preserved manifest")
    for member, path in members.items():
        if path.stat().st_size != manifest[member]["bytes"] or sha_file(path) != manifest[member][
            "sha256"
        ]:
            raise ValueError("Member differs from the preserved manifest: " + member)
    out.mkdir(parents=True, exist_ok=False)
    excluded, renamed, derived = {}, {}, {}
    for member in sorted(members):
        source = members[member]
        reason = exclusion(member)
        if reason:
            excluded[member] = {**manifest[member], "reason": reason}
            continue
        target = out / member
        target.parent.mkdir(parents=True, exist_ok=True)
        if member.endswith("rental.json"):
            data = json.loads(source.read_text())
            public = {k: v for k, v in data.items() if k not in PRIVATE_RENTAL_KEYS}
            target = target.with_name(target.name.replace("rental.json", "rental_public.json"))
            target.write_text(json.dumps(public, indent=2) + "\n")
            excluded[member] = {**manifest[member], "reason": "SSH endpoint, budget and paths"}
            derived[target.relative_to(out).as_posix()] = "rental.json without " + ", ".join(
                sorted(PRIVATE_RENTAL_KEYS)
            )
            continue
        if member.endswith("kernel_trace.json"):
            target = target.with_name(target.name + ".gz")
            with open(source, "rb") as src, gzip.GzipFile(target, "wb", mtime=0) as dst:
                shutil.copyfileobj(src, dst)
            derived[target.relative_to(out).as_posix()] = "gzip of " + member
            continue
        if member.endswith(".py"):
            target = target.with_name(target.name + ".txt")
            renamed[member] = target.relative_to(out).as_posix()
        shutil.copyfile(source, target)
    packet = {
        "name": name,
        "archive_sha256": archive_sha256,
        "archive_members": len(manifest),
        "included_members": len(manifest) - len(excluded),
        "excluded_members": excluded,
        "renamed_members": renamed,
        "renamed_reason": "archived scripts are stored as .py.txt so repository lint skips them",
        "derived_files": derived,
        "archive_manifest": "archive_manifest.json (the preserved manifest, unchanged)",
    }
    (out / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "PACKET.json").write_text(json.dumps(packet, indent=2) + "\n")
    return packet


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    packet = build(args.extracted, json.loads(args.manifest.read_text()), args.archive_sha256,
                   args.name, args.out)
    print(args.name, packet["included_members"], "of", packet["archive_members"], "members")


if __name__ == "__main__":
    main()
