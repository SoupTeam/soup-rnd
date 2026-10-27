#!/usr/bin/env bash
# Second profile pass for A5 ticket 10, with the step parts on the GPU timeline. Runs after
# remote.sh on the same instance; /root/soup2 is the newer tree (git archive), the venv and the
# data are the ones remote.sh made. CARD (8gb or 24gb) labels the result.
cd /root/soup2
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/src
/root/soup/.venv/bin/python benchmarks/harness/l2l_ticket10_campaign.py --work /root/work \
  --out /root/results/profile2 --card "$CARD" --tree /root/soup2 --a ga --b after \
  --max-blocks 2 --profile > /root/results/profile2.log 2>&1
echo EXIT=$? >> /root/results/profile2.log
echo DONE2 > /root/results/done2
