#!/usr/bin/env bash
# A5 ticket 05, F1 pass. Runs on the rented instance, from the repo root (/root/soup): bootstrap, the synthetic 70B
# checkpoint, the plan check, A-B-B-A blocks of L2L watched by F1. Mechanics and memory only.
cd /root/soup
mkdir -p /root/results
apt-get update -qq; apt-get install -y -qq python3.12-venv rsync git > /root/apt-venv.log 2>&1
bash benchmarks/a0-bootstrap.sh > /root/bootstrap.log 2>&1; echo EXIT=$? >> /root/bootstrap.log
.venv/bin/pip install --no-cache-dir 'typer>=0.9,<0.21' plotext 'trl>=0.29,<1' 'datasets>=2.14' \
  psutil > /root/extras.log 2>&1; echo EXIT=$? >> /root/extras.log
export PYTHONPATH=$PWD/src LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
.venv/bin/pip freeze > /root/results/packages.txt
nvidia-smi > /root/results/nvidia-smi.txt 2>&1
free -b > /root/results/free.txt; cat /sys/fs/cgroup/memory.max >> /root/results/free.txt 2>&1
df -h /root > /root/results/df.txt 2>&1
mkdir -p /root/work/data
.venv/bin/python benchmarks/harness/l2l_ticket04/prepare_data.py --out /root/work/data --rows 16 \
  > /root/results/prepare_data.log 2>&1; echo EXIT=$? >> /root/results/prepare_data.log
.venv/bin/python benchmarks/harness/l2l_synth70b.py plan --k 4 --seq 256 --tier disk --path /root/work \
  > /root/results/plan.json 2>&1; echo EXIT=$? >> /root/results/plan.json
.venv/bin/python benchmarks/harness/synth_checkpoint.py --shape llama-70b --vocab 32768 \
  --out /root/work/synth-llama-70b > /root/results/synth_checkpoint.log 2>&1
echo EXIT=$? >> /root/results/synth_checkpoint.log
.venv/bin/python - > /root/results/tokenizer.log 2>&1 <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("mistralai/Mistral-7B-v0.1", local_dir="/root/work/synth-llama-70b",
                  allow_patterns=["tokenizer*", "special_tokens_map.json"])
print("tokenizer ok")
PY
df -h /root > /root/results/df-after-synth.txt 2>&1
cd /root/work
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
/root/soup/.venv/bin/python /root/soup/benchmarks/harness/l2l_synth70b_campaign.py \
  --work /root/work --out /root/results/campaign --card "$CARD" --base /root/work/synth-llama-70b \
  --max-blocks "${MAX_BLOCKS:-5}" > /root/results/campaign.log 2>&1
echo EXIT=$? >> /root/results/campaign.log
echo ALLDONE > /root/results/done
