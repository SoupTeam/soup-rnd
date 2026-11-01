#!/usr/bin/env bash
# Runs on the instance: bootstrap, then the equal-length padding control (230 tokens) for each
# data seed given as an argument, in fresh processes. The host libcuda comes first (error 803).
cd /root/soup
apt-get update -qq; apt-get install -y -qq python3.12-venv rsync git > /root/apt-venv.log 2>&1
bash benchmarks/a0-bootstrap.sh > /root/bootstrap.log 2>&1; echo EXIT=$? >> /root/bootstrap.log
export PYTHONPATH=$PWD/src LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
.venv/bin/pip freeze > /root/packages.txt
mkdir -p /root/runs
for s in "$@"; do
  .venv/bin/python benchmarks/harness/l2l_fwdbwd.py --card "$CARD" --equal-length 230 --data-seed $s --out /root/runs/ctl-$CARD-seed$s > /root/runs/ctl-$CARD-seed$s.log 2>&1
  echo EXIT=$? >> /root/runs/ctl-$CARD-seed$s.log
done
echo ALLDONE > /root/runs/done
