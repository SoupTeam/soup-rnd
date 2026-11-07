"""Verify and join the exact evidence ZIP, stored in repository-sized parts."""

import argparse
import hashlib
import json
from pathlib import Path

from rich.console import Console


def join(output: Path) -> None:
    root = Path(__file__).parent
    manifest = json.loads((root / "evidence-manifest.json").read_bytes())
    chunks = []
    for part in manifest["parts"]:
        name = part["name"]
        if Path(name).name != name:
            raise ValueError("Invalid archive-part name")
        raw = (root / name).read_bytes()
        if len(raw) != part["bytes"] or hashlib.sha256(raw).hexdigest() != part["sha256"]:
            raise ValueError("Archive part failed verification: " + name)
        chunks.append(raw)
    archive = b"".join(chunks)
    expected = manifest["archive"]
    if (
        len(archive) != expected["bytes"]
        or hashlib.sha256(archive).hexdigest() != expected["sha256"]
    ):
        raise ValueError("Reassembled archive differs from verified original")
    # Never overwrite an existing file, even when the requested destination exists.
    with output.open("xb") as target:
        target.write(archive)
    Console().print({"archive": str(output), "bytes": len(archive), "sha256": expected["sha256"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    join(parser.parse_args().output)
