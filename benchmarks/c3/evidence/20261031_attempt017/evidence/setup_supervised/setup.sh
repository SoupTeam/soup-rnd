#!/bin/bash
# Run once under the existing setup_supervisor, in a new owned /data/c3 root.
set -euo pipefail
umask 077
cd /data/c3
export PYTHONPATH=/data/c3/repo
export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never UV_NO_CONFIG=1
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
python3 - <<'PY'
import hashlib,json,os,platform,tarfile
from pathlib import Path
from benchmarks.c3.offline_bootstrap import verify_bundle
p=Path('/data/c3');repo=p/'repo/benchmarks/c3';cache=p/'offline'
assert platform.system()=='Linux' and platform.machine()=='x86_64'
assert platform.libc_ver()[0]=='glibc' and tuple(map(int,platform.libc_ver()[1].split('.'))) >= (2,39)
assert os.path.realpath(p)==str(p)
for name in ['tools','env','python_runtime']:
 assert not (p/name).exists(), 'fresh setup only; no retry'
m=json.loads((repo/'offline_artifacts.json').read_text())
for name,digest in m['lock_sha256'].items():
 assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest
for row in m['files']:verify_bundle(cache/'wheels',row)
a=cache/m['python']['name']
assert a.is_file() and not a.is_symlink() and a.stat().st_size==m['python']['size']
with a.open('rb') as f:assert 'sha256:'+hashlib.file_digest(f,'sha256').hexdigest()==m['python']['digest']
with tarfile.open(a) as t:
 assert all(x.name=='python' or x.name.startswith('python/') for x in t.getmembers())
 t.extractall(p/'python_runtime',filter='data')
PY
python3 -m venv /data/c3/tools
/data/c3/tools/bin/python -m pip --isolated install --no-index --no-deps --require-hashes --find-links /data/c3/offline/wheels -r /data/c3/repo/benchmarks/c3/bootstrap_uv.lock
/data/c3/tools/bin/uv --version
/data/c3/python_runtime/python/bin/python3.11 -c 'import sys; assert sys.version_info[:3] == (3,11,16)'
/data/c3/tools/bin/uv venv --python /data/c3/python_runtime/python/bin/python3.11 /data/c3/env
/data/c3/tools/bin/uv pip sync --python /data/c3/env/bin/python --no-index --find-links /data/c3/offline/wheels --require-hashes --no-cache /data/c3/repo/benchmarks/c3/runtime.lock
/data/c3/tools/bin/uv pip install --python /data/c3/env/bin/python --no-index --find-links /data/c3/offline/wheels --require-hashes --no-cache -r /data/c3/repo/benchmarks/c3/test_runtime.lock
/data/c3/tools/bin/uv pip check --python /data/c3/env/bin/python
/data/c3/tools/bin/uv pip freeze --python /data/c3/env/bin/python > /data/c3/evidence/packages.txt
