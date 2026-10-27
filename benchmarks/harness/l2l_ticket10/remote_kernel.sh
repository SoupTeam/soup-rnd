#!/usr/bin/env bash
# Third pass for A5 ticket 10: one A-B-B-A block with step 3 of every run under torch.profiler,
# to see how long the GPU was busy against the step's wall time. /root/soup3 is the tree.
cd /root/soup3
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/src
/root/soup/.venv/bin/python benchmarks/harness/l2l_ticket10_campaign.py --work /root/work \
  --out /root/results/kernel --card "$CARD" --tree /root/soup3 --a ga --b after \
  --max-blocks 1 --profile --kernel-step 3 > /root/results/kernel.log 2>&1
echo EXIT=$? >> /root/results/kernel.log
echo DONE3 > /root/results/done3
