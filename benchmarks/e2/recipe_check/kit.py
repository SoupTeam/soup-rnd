"""Build or verify the source kit the Kaggle notebook runs (PROTOCOL.txt 2).

  python benchmarks/e2/recipe_check/kit.py build OUT.zip      # from a clean checkout
  python benchmarks/e2/recipe_check/kit.py verify ROOT        # inside the notebook

The kit is `git archive HEAD` plus KIT.json: the commit and the SHA256 of every file.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(out: Path) -> dict:
    if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                      check=True).stdout.strip():
        raise RuntimeError("Working tree is not clean; commit first")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", "HEAD"])
    files = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for info in source.infolist():
            if not info.is_dir():
                files[info.filename] = _sha(source.read(info))
    manifest = {"schema": "e2-recipe-kit-v1", "commit": commit, "files": files}
    with zipfile.ZipFile(io.BytesIO(archive)) as source, \
            zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            target.writestr(info, source.read(info))
        target.writestr("KIT.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return {"commit": commit, "files": len(files), "zip_sha256": _sha(out.read_bytes())}


def _hidden(name: str) -> bool:
    return any(part.startswith(".") for part in name.split("/"))


def verify(root: Path) -> dict:
    """Every kit file must be present and unchanged.

    Kaggle unpacks an uploaded zip and may drop dot-files (.gitignore, .github/...);
    those are not used to run an arm, so a missing dot-file is reported, not fatal.
    A changed file, or any other missing file, is fatal.
    """
    manifest = json.loads((root / "KIT.json").read_text(encoding="utf-8"))
    bad, dropped = [], []
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.is_file():
            (dropped if _hidden(name) else bad).append(name)
        elif _sha(path.read_bytes()) != digest:
            bad.append(name)
    if bad:
        raise RuntimeError(f"{len(bad)} kit file(s) differ, first: {bad[0]}")
    return {"commit": manifest["commit"], "files": len(manifest["files"]),
            "missing_dot_files": dropped}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in {"build", "verify"}:
        raise SystemExit(__doc__)
    action, path = argv[0], Path(argv[1])
    print(json.dumps(build(path) if action == "build" else verify(path)))


if __name__ == "__main__":
    main()
