"""Download the pinned public F2 checkpoint only after checking disk headroom."""

import argparse
import json
import shutil
import time
from pathlib import Path

from f2_reference import MODEL, REVISION
from rich.console import Console


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.destination).free
    if free < 60e9:
        raise ValueError("need 60 GB free for 29.6 GB source, NF4 cache and scratch")
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(MODEL, revision=REVISION)
    if info.sha != REVISION:
        raise ValueError("Hub revision did not resolve to the pinned checkpoint")
    began = time.time()
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=args.destination,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
        max_workers=2,
    )
    record = {
        "model": MODEL,
        "revision": info.sha,
        "unix_s": began,
        "download_seconds": time.time() - began,
        "free_bytes_before": free,
    }
    (args.destination / "f2-source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    Console().print(record)


if __name__ == "__main__":
    main()
