#!/bin/bash
set -euo pipefail
export PYTHONPATH=/root/b6/repo-revised:/root/b6/repo-revised/src
mkdir -m 700 /root/b6/results/attempt2
trap 'code=$?; if test "$code" != 0; then echo "$code" > /root/b6/results/attempt2/failure-code.txt; until test -f /root/b6/abort-attempt2; do sleep 5; done; fi; echo "$code" > /root/b6/results/exit-code.txt' EXIT
cp /root/b6/experiment-revised.sh /root/b6/results/attempt2/experiment.sh
cp /root/b6/revised-runner.sh /root/b6/results/attempt2/runner.sh
cp /root/b6/hardware-probe.json /root/b6/results/attempt2/hardware-probe.json
systemctl show b6-pilot.scope -p MemoryMax -p MemorySwapMax > /root/b6/results/attempt2/initial-scope-properties.txt
echo download > /root/b6/phase2.txt
/root/b6/venv/bin/python /root/b6/repo-revised/benchmarks/harness/b6_ram.py \
 --output /root/b6/results/attempt2 --phase-file /root/b6/phase2.txt --interval 2 \
 -- bash /root/b6/experiment-revised.sh
