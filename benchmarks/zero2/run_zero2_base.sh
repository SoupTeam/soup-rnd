#!/usr/bin/env bash
# ZERO-2 stage 2 on a Vast instance (one 24 GB GPU, >= 64 GB RAM, >= 120 GB disk).
# Usage: bash run_zero2_base.sh        (from the directory holding zero2_base.py)
set -euo pipefail
python -V
pip install -q --ignore-requires-python \
  "soup-cli[train] @ git+https://github.com/SoupTeam/soup-rnd@63443f153a9fe5accaed0490df63571a01d3cfd7"
pip install -q "transformers==5.22.2" "flash-linear-attention==0.6.0" "fla-core==0.6.0" \
  "peft==0.21.2" "bitsandbytes==0.50.2" "accelerate==1.15.0" "trl==0.29.1" "datasets==5.2.0" \
  "lm-eval" pandas pyarrow
pip uninstall -y torchao || true
python -c "from soup_cli.utils.layer_select import LayerSelectPolicy; print('soup branch ok')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python - <<'PY'
import hashlib
h = hashlib.sha256(open('zero2_base.py','rb').read()).hexdigest()
print(h)
assert h == 'd1511826bfbc9b8bed29ae8cab11decb14dd3cf134bf62a202b17ad963f486f2', 'runner differs from the protocol copy'
PY
export ZERO_OUT="$PWD/zero2_base_results"
python zero2_base.py 2>&1 | tee zero2_base_runner.log
TS=$(date -u +%Y%m%dT%H%M%SZ)
tar czf "zero2_base_${TS}.tgz" --exclude='zero2_base_results/work/out_*' \
  zero2_base_results zero2_base.py zero2_base_runner.log $( [ -d .soup-crashes ] && echo .soup-crashes )
ls -la "zero2_base_${TS}.tgz"
