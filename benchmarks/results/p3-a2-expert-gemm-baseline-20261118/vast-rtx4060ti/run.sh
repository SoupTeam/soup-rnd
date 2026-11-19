#!/usr/bin/env bash
# A2 stage 0 on an sm_80+ card: bits (all dtypes, all configs), timing, repeat.
set -uo pipefail
W=/root/a2; OUT=$W/out; V=/root/py312
mkdir -p $OUT; cd $W
exec > >(tee $OUT/run.log) 2>&1
echo "harness $(sha256sum p3_expert_gemm_baseline.py)"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh; }
export PATH=$HOME/.local/bin:$PATH
command -v gcc >/dev/null && test -f /usr/include/python3.12/Python.h || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.12-dev python3.12-venv gcc >/dev/null; }
test -f /usr/include/python3.12/Python.h || { echo "no Python.h"; touch $OUT/DONE; exit 1; }
[ -x $V/bin/python ] || uv venv --python /usr/bin/python3.12 --seed $V
export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2
for try in 1 2 3 4; do uv pip install --python $V/bin/python torch==2.11.0 rich==13.9.4 numpy==2.1.3 && break; echo "install retry $try"; sleep 20; done
$V/bin/python -c 'import torch' || { echo 'torch install failed'; touch $OUT/DONE; exit 1; }
$V/bin/python -c 'import torch,triton;print(torch.__version__,torch.version.cuda,triton.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 COLUMNS=220
H="$V/bin/python p3_expert_gemm_baseline.py --device cuda"
declare -A codes
$H --no-timing --output $OUT/bits.json; codes[bits]=$?
$H --dtypes float16,bfloat16 --output $OUT/timing.json; codes[timing]=$?
$H --no-timing --dtypes float16,bfloat16 --output $OUT/bits-repeat.json; codes[bits_repeat]=$?
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
for k in "${!codes[@]}"; do echo "exit $k=${codes[$k]}"; done | tee $OUT/exit-codes.txt
touch $OUT/DONE
