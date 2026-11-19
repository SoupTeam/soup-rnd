#!/usr/bin/env bash
# ZERO-2 stage 4 on a Vast instance (one 24 GB GPU, >= 64 GB RAM, >= 120 GB disk).
# Usage: bash run_zero2_stage4.sh [part ...]   (from the directory holding zero2_stage4.py)
set -euo pipefail
python -V
pip install -q --ignore-requires-python \
  "soup-cli[train] @ git+https://github.com/SoupTeam/soup-rnd@b3bca25cda116bf843b4e76aa59f228e4cbbc0ed"
pip install -q "transformers==5.22.2" "flash-linear-attention==0.6.0" "fla-core==0.6.0" \
  "peft==0.21.2" "bitsandbytes==0.50.2" "accelerate==1.15.0" "trl==0.29.1" "datasets==5.2.0" \
  "lm-eval==0.4.13" "evaluate==0.4.6" pandas pyarrow
pip uninstall -y torchao || true
python -c "from soup_cli.utils.layer_select import cut_input_grads; from soup_cli.trainer.kbit_lean import embedding_on_host; print('soup branch ok')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
sha256sum zero2_stage4.py
export ZERO_OUT="$PWD/zero2_stage4_results"
python zero2_stage4.py "$@" 2>&1 | tee -a zero2_stage4_runner.log
TS=$(date -u +%Y%m%dT%H%M%SZ)
tar czf "zero2_stage4_${TS}.tgz" --exclude='zero2_stage4_results/work/out_*' \
  zero2_stage4_results zero2_stage4.py zero2_stage4_runner.log $( [ -d .soup-crashes ] && echo .soup-crashes )
ls -la "zero2_stage4_${TS}.tgz"
