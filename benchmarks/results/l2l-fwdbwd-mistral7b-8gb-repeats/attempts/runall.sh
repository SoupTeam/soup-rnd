#!/usr/bin/env bash
# Runs on the instance: bootstrap, then three runs with data seeds 5, 6, 7 in fresh processes.
cd /root/soup
apt-get update -qq; apt-get install -y -qq python3.12-venv rsync git > /root/apt-venv.log 2>&1
bash benchmarks/a0-bootstrap.sh > /root/bootstrap.log 2>&1; echo EXIT=$? >> /root/bootstrap.log
export PYTHONPATH=$PWD/src
.venv/bin/pip freeze > /root/packages.txt
mkdir -p /root/runs
for s in 5 6 7; do
  .venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card 8gb --data-seed $s --out /root/runs/fwdbwd-8gb-seed$s > /root/runs/fwdbwd-8gb-seed$s.log 2>&1
  echo EXIT=$? >> /root/runs/fwdbwd-8gb-seed$s.log
done
echo ALLDONE > /root/runs/done
