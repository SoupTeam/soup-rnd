"""Verify an extracted timing bundle and restore byte-identical tensor aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from rich.console import Console


def materialize(root: Path) -> int:
    root = Path(os.path.realpath(root))
    aliases = json.loads((root / "dedup-map.json").read_text())
    manifest = json.loads((root / "evidence-manifest.json").read_text())

    def contained(name: str) -> Path:
        path = Path(os.path.realpath(root / name))
        if os.path.commonpath([str(root), str(path)]) != str(root):
            raise ValueError("evidence path escapes extraction root")
        return path

    for name, source in aliases.items():
        target, original = contained(name), contained(source)
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if digest != manifest[name]["sha256"]:
            raise ValueError("alias payload hash mismatch")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    checked = 0
    for name, entry in manifest.items():
        if entry["included"]:
            payload = contained(name)
            if hashlib.sha256(payload.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"payload hash mismatch: {name}")
            checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    Console().print("Verified", materialize(args.directory), "payloads and restored tensor aliases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
