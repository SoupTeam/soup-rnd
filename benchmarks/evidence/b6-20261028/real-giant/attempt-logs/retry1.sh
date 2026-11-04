#!/bin/bash
set -euo pipefail
export PYTHONPATH=/root/b6/repo:/root/b6/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /root/b6/repo
trap 'code=$?; echo "$code" > /root/b6/results/retry1-exit-code.txt; if test "$code" = 0; then echo 0 > /root/b6/results/exit-code.txt; else echo "$code" > /root/b6/results/failure-code.txt; while ! test -f /root/b6/abort; do sleep 5; done; echo "$code" > /root/b6/results/exit-code.txt; fi' EXIT
/root/b6/venv/bin/python benchmarks/harness/b6_real_moe.py --source /root/b6/source --cache /root/b6/cache --output /root/b6/results/real-retry1 --data examples/b6-pilot-data.json --revision e815299b0bcbac849fa540c768ef21845365c9eb --steps 2 --max-length 64 | tee /root/b6/results/real-retry1.log
