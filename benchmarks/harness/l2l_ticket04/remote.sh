#!/usr/bin/env bash
# Runs on the rented instance, from the repo root (/root/soup): bootstrap, data, then the
# A5 ticket 04 campaign. CARD (8gb or 24gb) labels the result. The host libcuda comes first
# (error 803 otherwise), as in tickets 03 and 07.
cd /root/soup
apt-get update -qq; apt-get install -y -qq python3.12-venv rsync git > /root/apt-venv.log 2>&1
bash benchmarks/a0-bootstrap.sh > /root/bootstrap.log 2>&1; echo EXIT=$? >> /root/bootstrap.log
.venv/bin/pip install --no-cache-dir 'typer>=0.9,<0.21' plotext 'trl>=0.29,<1' 'datasets>=2.14' \
  > /root/extras.log 2>&1; echo EXIT=$? >> /root/extras.log
export PYTHONPATH=$PWD/src LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
.venv/bin/pip freeze > /root/packages.txt
nvidia-smi > /root/nvidia-smi.txt 2>&1
free -b > /root/free.txt; cat /sys/fs/cgroup/memory.max >> /root/free.txt 2>&1
df -h /root /root/.cache > /root/df.txt 2>&1
mkdir -p /root/work/data /root/results
.venv/bin/python benchmarks/harness/l2l_ticket04/prepare_data.py --out /root/work/data \
  > /root/results/prepare_data.log 2>&1; echo EXIT=$? >> /root/results/prepare_data.log
.venv/bin/python benchmarks/harness/l2l_ticket04_campaign.py --work /root/work \
  --out /root/results/campaign --card "$CARD" > /root/results/campaign.log 2>&1
echo EXIT=$? >> /root/results/campaign.log
echo ALLDONE > /root/results/done
