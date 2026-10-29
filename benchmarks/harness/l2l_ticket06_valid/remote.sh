#!/usr/bin/env bash
# Runs on the rented instance for the A5 rerun where the required foreign_reads check answers
# (rule: benchmarks/l2l-readiness-rule-a5.md, last section). The vast local volume is mounted at
# /root/volume; HF_HOME and the data directory live on it, so the model path maps to a block
# device. /root/soup is the tree to measure. CARD labels the result. The host libcuda comes first
# (error 803 otherwise). MAXBLOCKS04 caps the blocks of the ticket-04 repeats rule (default 7);
# SKIP_T11=1 stops after ticket 04.
cd /root/soup
apt-get update -qq; apt-get install -y -qq python3.12-venv rsync git > /root/apt-venv.log 2>&1
bash benchmarks/a0-bootstrap.sh > /root/bootstrap.log 2>&1; echo EXIT=$? >> /root/bootstrap.log
.venv/bin/pip install --no-cache-dir 'typer>=0.9,<0.21' plotext 'trl>=0.29,<1' 'datasets>=2.14' \
  > /root/extras.log 2>&1; echo EXIT=$? >> /root/extras.log
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/src
export HF_HOME=/root/volume/hf TMPDIR=/root/volume/tmp
mkdir -p "$HF_HOME" "$TMPDIR" /root/volume/work/data /root/results
.venv/bin/pip freeze > /root/packages.txt
nvidia-smi > /root/nvidia-smi.txt 2>&1
nvidia-smi -q -d CLOCK,PERFORMANCE > /root/nvidia-smi-q.txt 2>&1
lscpu > /root/lscpu.txt 2>&1
free -b > /root/free.txt; cat /sys/fs/cgroup/memory.max >> /root/free.txt 2>&1
df -h /root /root/volume > /root/df.txt 2>&1; grep volume /proc/mounts >> /root/df.txt
.venv/bin/python benchmarks/harness/l2l_ticket04/prepare_data.py --out /root/volume/work/data \
  > /root/results/prepare_data.log 2>&1; echo EXIT=$? >> /root/results/prepare_data.log
.venv/bin/python benchmarks/harness/l2l_ticket04_campaign.py --work /root/volume/work \
  --out /root/results/t04 --card "$CARD" --max-blocks "${MAXBLOCKS04:-7}" \
  > /root/results/t04.log 2>&1
echo EXIT=$? >> /root/results/t04.log
echo T04DONE > /root/results/t04-done
[ -n "$SKIP_T11" ] && { echo ALLDONE > /root/results/done; exit 0; }
.venv/bin/python benchmarks/harness/l2l_ticket11_campaign.py --work /root/volume/work \
  --out /root/results/t11 --card "$CARD" --max-blocks "${MAXBLOCKS:-7}" > /root/results/t11.log 2>&1
echo EXIT=$? >> /root/results/t11.log
echo ALLDONE > /root/results/done
