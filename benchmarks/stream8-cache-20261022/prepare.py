"""Rebuild a local research bundle from pinned Git and the preserved legacy overlay.

No network, hardware provisioning, dependency installation, or model download.
Use the generated code manually on an already qualified Linux CUDA host.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from rich.console import Console

BASELINE = "d500ae0dca7c4da905f8d59c9866af2f8aaa824b"


def contained(root, name):
    absolute = os.path.realpath(root / name)
    if os.path.commonpath([os.path.realpath(root), absolute]) != os.path.realpath(root):
        raise ValueError("path outside output")
    return Path(absolute)


def prepare(output):
    here = Path(__file__).parent
    repo = here.parents[1]
    output.mkdir(parents=True, exist_ok=False)
    tree = output / "repo"
    tree.mkdir()
    archive = output / "upstream.tar"
    subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=tar",
                    "--output", str(archive), BASELINE],
                   cwd=repo, check=True)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            contained(tree, member.name)
            if member.issym() or member.islnk():
                raise ValueError("unexpected source link")
        source.extractall(tree)
    legacy = (repo / "benchmarks/evidence/streaming-residency-compile-8gb-20261021"
              / "planned-harness.zip")
    with zipfile.ZipFile(legacy) as source:
        manifest = json.loads(source.read("manifest.json"))
        for name, metadata in manifest.items():
            data = source.read(name)
            assert len(data) == metadata["bytes"]
            assert hashlib.sha256(data).hexdigest() == metadata["sha256"]
            if name.startswith("overlay/"):
                path = contained(tree, name.removeprefix("overlay/"))
            elif name in {"inputs/constraints.txt", "inputs/prepare_model.py",
                          "inputs/pin_entry.py", "inputs/telemetry.py"}:
                path = output / Path(name).name
                data = data.replace(b"/root/liger", b"/root/stream8-cache")
            else:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    for name in ("bench.py", "candidate.py", "worker.py", "protocol.json"):
        shutil.copyfile(here / name, output / name)
    test = (repo / "tests/test_stream8_cache_20261022.py").read_text()
    old = 'Path(__file__).parents[1] / "benchmarks/stream8-cache-20261022/candidate.py"'
    test = test.replace(old, 'Path(__file__).parent / "candidate.py"')
    (output / "cache_test.py").write_text(test)
    # Verify the complete production tree against Git's own blob identities.
    count = 0
    for line in subprocess.check_output(["git", "ls-tree", "-r", BASELINE, "src/soup_cli"],
                                        cwd=repo, text=True).splitlines():
        metadata, name = line.split("\t")
        data = (tree / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == metadata.split()[2], name
        count += 1
    proof = {"baseline": BASELINE, "production_files": count, "production_all_exact": True,
             "legacy_overlay_sha256": hashlib.sha256(legacy.read_bytes()).hexdigest(),
             "status": "PREPARED_NOT_GPU_VALIDATED"}
    (output / "source-audit.json").write_text(json.dumps(proof, indent=2))
    Console().print(proof)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output.absolute())
