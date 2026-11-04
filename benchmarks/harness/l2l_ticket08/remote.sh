#!/usr/bin/env bash
# A5 ticket 08, on the rented instance. /root/soup is this checkout with the fix,
# /root/soup-before the same tree with l2l_step.py from before the fix (copied by the caller).
# Bootstrap as in ticket 04, then four runs of the probe on the ticket-04 configs (Mistral-7B
# NF4, 64 alpaca rows, k=4, 16 steps, SDPA MATH, deterministic): GA twice, L2L before the fix,
# L2L after it. Then the gradient comparisons. Logs in /root/results.
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

PY=/root/soup/.venv/bin/python
probe() {  # probe <tree> <config> <name>
  local dir=/root/work/$3
  rm -rf "$dir"; mkdir -p "$dir"; ln -s /root/work/data "$dir/data"
  ( cd "$dir" && "$PY" "$1/benchmarks/harness/l2l_ticket08_gpu_probe.py" run \
      --config "$1/benchmarks/harness/l2l_ticket04/$2" --out /root/results/$3 \
      > /root/results/$3.log 2>&1; echo EXIT=$? >> /root/results/$3.log )
}
probe /root/soup ga.yaml ga1
probe /root/soup ga.yaml ga2
probe /root/soup-before l2l.yaml l2l-before
probe /root/soup l2l.yaml l2l-after
for pair in "ga1 ga2" "ga1 l2l-before" "ga1 l2l-after"; do
  set -- $pair
  "$PY" /root/soup/benchmarks/harness/l2l_ticket08_gpu_probe.py compare /root/results/$1 \
    /root/results/$2 > /root/results/compare-$1-$2.txt 2>&1
done
echo ALLDONE > /root/results/done
