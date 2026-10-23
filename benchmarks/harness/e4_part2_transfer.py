"""E4 part 2: move the Mac-side quantization outputs to the instance and verify every SHA-256.

    push (Mac):      e4_part2_transfer.py push --out ~/e4-part2-data/out \\
                         --manifest benchmarks/results/e4-part2-quant/quant-manifest.json
    pull (instance): e4_part2_transfer.py pull --data-root /workspace/e4

``push`` uploads ``out/`` (not ``source/``) and the manifest to a private dataset repo; it
needs a Hugging Face token with write access (``hf auth login``). ``pull`` downloads that
repo into ``DATA_ROOT``, downloads the source model from HF at the pinned revision into
``DATA_ROOT/source`` and checks every file against ``quant-manifest.json`` (U4): the 78 output
files and the 9 source files. Any missing, extra-named-in-manifest-but-absent or differing
file makes it exit 1; nothing is repaired or re-downloaded silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import e4_part2_common as common

DEFAULT_REPO = "e4-part2-quant"
MANIFEST_NAME = "quant-manifest.json"
VERIFIED_NAME = "verified.json"
#: Files a Hugging Face dataset repo gets on creation; the only extra the push check allows.
HF_AUTO_FILES = frozenset({".gitattributes"})
SOURCE_PATTERNS = ["*.safetensors", "*.json", "*.txt"]


def verify_tree(root: Path, expected: Dict[str, Dict[str, Any]], label: str,
                log=print, record: Optional[Dict[str, Any]] = None, prefix: str = ""
                ) -> List[str]:
    """SHA-256 and size of every file in ``expected`` under ``root``. Returns the problems.

    With ``record`` given, the hash of every file that matches is stored there under
    ``prefix + name``.
    """
    problems = []
    for index, (name, want) in enumerate(sorted(expected.items()), 1):
        path = root / name
        if not path.is_file():
            problems.append(f"{label}: {name} is missing")
            continue
        if "bytes" in want and path.stat().st_size != want["bytes"]:
            problems.append(f"{label}: {name} is {path.stat().st_size} bytes, "
                            f"manifest says {want['bytes']}")
            continue
        got = common.sha256_file(path)
        status = "ok" if got == want["sha256"] else "DIFFERS"
        if got != want["sha256"]:
            problems.append(f"{label}: {name} SHA-256 {got} differs from {want['sha256']}")
        elif record is not None:
            record[prefix + name] = {"sha256": got, "bytes": path.stat().st_size}
        log(f"[{index}/{len(expected)}] {label} {name} {status}")
    return problems


def unlisted_files(root: Path, expected: Dict[str, Any], ignore: tuple = ()) -> List[str]:
    """Files under ``root`` the manifest does not name (reported, not an error)."""
    found = {p.relative_to(root).as_posix() for p in root.rglob("*")
             if p.is_file() and not p.relative_to(root).parts[0].startswith(".")}
    return sorted(found - set(expected) - set(ignore))


def resolve_repo_id(api, repo: str) -> str:
    return repo if "/" in repo else f"{api.whoami()['name']}/{repo}"


def push(out: Path, manifest: Path, repo: str, dry_run: bool, api: Optional[Any] = None,
         log=print) -> int:
    """Upload exactly the manifest's files (plus the manifest) and check the repo afterwards."""
    from huggingface_hub import HfApi

    api = api or HfApi()
    body = json.loads(manifest.read_text())
    names = sorted(body["files"])
    problems = verify_tree(out, body["files"], "out", log=lambda *_: None)
    if problems:
        log("refusing to upload; local files do not match the manifest:\n  " +
            "\n  ".join(problems[:10]))
        return 1
    total = sum(v["bytes"] for v in body["files"].values())
    repo_id = resolve_repo_id(api, repo)
    log(f"{len(names)} files, {total / 1e9:.1f} GB -> dataset {repo_id} (private)")
    if dry_run:
        log("dry run: nothing uploaded")
        return 0
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        log("the repo exists and is not private; refusing to upload")
        return 1
    api.upload_file(path_or_fileobj=str(manifest), path_in_repo=MANIFEST_NAME,
                    repo_id=repo_id, repo_type="dataset")
    groups: Dict[str, List[str]] = {}
    for name in names:
        groups.setdefault(name.split("/")[0] if "/" in name else "", []).append(name)
    for group, members in sorted(groups.items()):
        if not group:
            for name in members:
                api.upload_file(path_or_fileobj=str(out / name), path_in_repo=name,
                                repo_id=repo_id, repo_type="dataset")
            continue
        # one commit per directory, only the manifest's files: no .cache/, no strays
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out / group),
                          path_in_repo=group, commit_message=f"{group}",
                          allow_patterns=[n[len(group) + 1:] for n in members],
                          ignore_patterns=[".cache/**", ".cache"])
        log(f"uploaded {group} ({len(members)} files)")
    expected = set(names) | {MANIFEST_NAME}
    listed = set(api.list_repo_files(repo_id, repo_type="dataset")) - HF_AUTO_FILES
    extra, missing = sorted(listed - expected), sorted(expected - listed)
    if extra or missing:
        log(f"REPO FILE SET DIFFERS from the manifest: extra {extra[:10]}, missing {missing[:10]}")
        return 1
    log(f"repo file set equals the manifest ({len(names)} files) plus {MANIFEST_NAME}; pull "
        f"on the instance with: e4_part2_transfer.py pull --repo {repo_id} --data-root DIR")
    return 0


def pull(data_root: Path, repo: str, api: Optional[Any] = None, log=print) -> int:
    """Download both repos, verify all files once and record that on the instance."""
    from huggingface_hub import HfApi, snapshot_download

    api = api or HfApi()
    repo_id = resolve_repo_id(api, repo)
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / VERIFIED_NAME).unlink(missing_ok=True)
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(data_root))
    manifest = json.loads((data_root / MANIFEST_NAME).read_text())
    if manifest["model"] != {"repo": common.MODEL_REPO, "revision": common.MODEL_REV}:
        log(f"manifest model {manifest['model']} is not the pinned one")
        return 1
    source = data_root / "source"
    snapshot_download(common.MODEL_REPO, revision=common.MODEL_REV, local_dir=str(source),
                      allow_patterns=SOURCE_PATTERNS)
    hashes: Dict[str, Any] = {}
    problems = verify_tree(data_root, manifest["files"], "out", log, hashes)
    problems += verify_tree(source, manifest["source_files"], "source", log, hashes, "source/")
    extra = [f for f in unlisted_files(data_root, manifest["files"],
                                       ignore=(MANIFEST_NAME, VERIFIED_NAME))
             if not f.startswith("source/")]
    if extra:
        log(f"files not named by the manifest (ignored): {extra[:10]}")
    if problems:
        log("VERIFY FAILED:\n  " + "\n  ".join(problems))
        return 1
    (data_root / VERIFIED_NAME).write_text(json.dumps({
        "verified_utc": common.now_utc(),
        "manifest_sha256": common.sha256_file(data_root / MANIFEST_NAME),
        "instance_id": common.instance_id(),
        "files": hashes,
    }, indent=1, sort_keys=True) + "\n")
    log(f"verified {len(manifest['files'])} output files and "
        f"{len(manifest['source_files'])} source files against {MANIFEST_NAME}; "
        f"recorded in {VERIFIED_NAME}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("pull")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args(argv)
    if args.cmd == "push":
        return push(args.out, args.manifest, args.repo, args.dry_run)
    return pull(args.data_root, args.repo)


if __name__ == "__main__":
    sys.exit(main())
