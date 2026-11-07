#!/usr/bin/env bash
# Runs on the rented instance for the A6 readiness rerun (rule: benchmarks/l2l-preference-readiness-rule-a6.md).
# /root/soup is the tree to measure. The host libcuda comes first (error 803 otherwise).
# Writes /root/results/ready/ (records, per-arm logs and watch records) and /root/results/done.
cd /root/soup
apt-get update -qq; apt-get install -y -qq python3.12-venv rsync git > /root/apt-venv.log 2>&1
bash benchmarks/a0-bootstrap.sh > /root/bootstrap.log 2>&1; echo EXIT=$? >> /root/bootstrap.log
.venv/bin/pip install --no-cache-dir 'typer>=0.9,<0.21' plotext 'trl>=0.29,<1' 'datasets>=2.14' \
  > /root/extras.log 2>&1; echo EXIT=$? >> /root/extras.log
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/src
export HF_HOME=/root/hf TMPDIR=/root/tmp
mkdir -p "$HF_HOME" "$TMPDIR" /root/results
.venv/bin/pip freeze > /root/packages.txt
nvidia-smi > /root/nvidia-smi.txt 2>&1
nvidia-smi -q -d CLOCK,PERFORMANCE > /root/nvidia-smi-q.txt 2>&1
lscpu > /root/lscpu.txt 2>&1
free -b > /root/free.txt
uptime > /root/uptime.txt
# The weights come down before the first arm, so no arm waits on the network.
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('HuggingFaceTB/SmolLM2-135M'))" > /root/results/download.log 2>&1
.venv/bin/python benchmarks/harness/l2l_pref_diagnose.py ready --device cuda --out /root/results/ready \
  --label "${CARD:-unknown card}, synthetic data, SmolLM2-135M" --model-path "$HF_HOME" \
  > /root/results/ready.log 2>&1
echo EXIT=$? >> /root/results/ready.log
echo ALLDONE > /root/results/done
