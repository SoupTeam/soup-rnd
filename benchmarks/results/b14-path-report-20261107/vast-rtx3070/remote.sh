#!/bin/bash
# B1.4 ticket 09 on the rented RTX 3070: tiny, oversized, real, one process each.
cd /root/soup
export B14_SRC=/root/soup B14_OUT=/root/out B14_WORK=/root/work B14_SEQS=16,512,2048 B14_TIER=ram B14_DEVICE=cuda
export PYTHONPATH=/root/soup/src
mkdir -p /root/out
for stage in tiny oversized real; do
  rm -rf /root/work
  B14_STAGE=$stage python3 benchmarks/harness/b14_path_run.py > /root/out/run-$stage.log 2>&1
  echo "$stage exit $?" >> /root/out/stages.txt
done
rm -rf /root/work
echo done >> /root/out/stages.txt
