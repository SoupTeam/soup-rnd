#!/usr/bin/env bash
# ZERO-2 stage 2b on a Vast instance (one 24 GB GPU, >= 64 GB RAM, >= 120 GB disk).
# Usage: bash run_zero2_2b.sh        (from the directory holding zero2_2b.py)
set -euo pipefail
python -V
pip install -q --ignore-requires-python \
  "soup-cli[train] @ git+https://github.com/SoupTeam/soup-rnd@6ac802385acba2250f7a09d84d5bf04a81356d6a"
pip install -q "transformers==5.22.2" "flash-linear-attention==0.6.0" "fla-core==0.6.0" \
  "peft==0.21.2" "bitsandbytes==0.50.2" "accelerate==1.15.0" "trl==0.29.1" "datasets==5.2.0" \
  "lm-eval==0.4.13" pandas pyarrow
pip uninstall -y torchao || true
python -c "from soup_cli.utils.nf4_triton import enable; from soup_cli.trainer.kbit_lean import prepare_kbit; print('soup branch ok')"
python -c "import triton; print('triton', triton.__version__)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
sha256sum zero2_2b.py
export ZERO_OUT="$PWD/zero2_2b_results"
python zero2_2b.py 2>&1 | tee zero2_2b_runner.log
TS=$(date -u +%Y%m%dT%H%M%SZ)
tar czf "zero2_2b_${TS}.tgz" --exclude='zero2_2b_results/work/out_*' \
  zero2_2b_results zero2_2b.py zero2_2b_runner.log $( [ -d .soup-crashes ] && echo .soup-crashes )
ls -la "zero2_2b_${TS}.tgz"
