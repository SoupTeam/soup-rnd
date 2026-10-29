#!/usr/bin/env bash
# Runs on the rented instance for A5 ticket 11. /root/soup is the tree to measure (git archive).
# CARD (8gb or 24gb) labels the result. MAXBLOCKS (default 7) caps the blocks of the repeats rule. The host libcuda comes first (error 803 otherwise).
cd /root/soup
apt-get update -qq; apt-get install -y -qq python3.12-venv rsync git > /root/apt-venv.log 2>&1
bash benchmarks/a0-bootstrap.sh > /root/bootstrap.log 2>&1; echo EXIT=$? >> /root/bootstrap.log
.venv/bin/pip install --no-cache-dir 'typer>=0.9,<0.21' plotext 'trl>=0.29,<1' 'datasets>=2.14' \
  > /root/extras.log 2>&1; echo EXIT=$? >> /root/extras.log
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/src
.venv/bin/pip freeze > /root/packages.txt
nvidia-smi > /root/nvidia-smi.txt 2>&1
nvidia-smi -q -d CLOCK,PERFORMANCE > /root/nvidia-smi-q.txt 2>&1
lscpu > /root/lscpu.txt 2>&1
free -b > /root/free.txt; cat /sys/fs/cgroup/memory.max >> /root/free.txt 2>&1
nproc >> /root/free.txt; cat /sys/fs/cgroup/cpu.max >> /root/free.txt 2>&1
mkdir -p /root/work/data /root/results
.venv/bin/python benchmarks/harness/l2l_ticket04/prepare_data.py --out /root/work/data \
  > /root/results/prepare_data.log 2>&1; echo EXIT=$? >> /root/results/prepare_data.log
.venv/bin/python benchmarks/harness/l2l_ticket11_campaign.py --work /root/work \
  --out /root/results/campaign --card "$CARD" --max-blocks "${MAXBLOCKS:-7}" > /root/results/campaign.log 2>&1
echo EXIT=$? >> /root/results/campaign.log
echo ALLDONE > /root/results/done
