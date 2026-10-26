#!/usr/bin/env bash
# Runs on the instance: three runs with data seeds 5, 6, 7 in fresh processes.
# The host libcuda comes first: the image's cuda-compat libcuda (580.65) gives error 803.
cd /root/soup
export PYTHONPATH=$PWD/src LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
rm -rf /root/runs; mkdir -p /root/runs
for s in 5 6 7; do
  .venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 8gb --data-seed $s --out /root/runs/fwdbwd-8gb-seed$s > /root/runs/fwdbwd-8gb-seed$s.log 2>&1
  echo EXIT=$? >> /root/runs/fwdbwd-8gb-seed$s.log
done
echo ALLDONE > /root/runs/done
