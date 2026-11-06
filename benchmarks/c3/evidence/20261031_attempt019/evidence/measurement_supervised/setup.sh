#!/bin/bash
set -euo pipefail
cd /data/c3/repo
export PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
exec /data/c3/env/bin/python -m benchmarks.c3.isolated_compare --mode cuda-representative --abba-blocks 20 --prepared /data/c3/repo/.c3_prepared/prepared.json --output /data/c3/evidence/comparison --deadline-unix-s 1793445153.6265
