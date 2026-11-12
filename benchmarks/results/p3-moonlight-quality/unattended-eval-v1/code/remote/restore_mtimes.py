"""Restore recorded shard mtimes (identity input) after download, before convert."""
import json
import os
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(os.path.realpath(sys.argv[2]))
files = protocol["source"]["files"]
for name, item in sorted(files.items()):
    path = source / name
    assert path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"], name
    os.utime(path, ns=(item["mtime_ns"], item["mtime_ns"]))
    assert path.stat().st_mtime_ns == item["mtime_ns"], f"filesystem rounded mtime: {name}"
print("restored", len(files), "mtimes")
