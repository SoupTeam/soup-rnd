#!/usr/bin/env bash
# G5: the twelve mixed-arm runs of the rule in a fixed order, one process each.
#
#   g5_train_all.sh DATA_DIR OUT_DIR SELECTION_DIR <c5_train.py model args...>
#   g5_train_all.sh data runs/mixed sel --model allenai/OLMoE-1B-7B-0924 \
#       --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --device cuda --gradient-checkpointing
#
# SELECTION_DIR is the output of g5_select.py (its maps/ folder). The order is
# seed by seed (42, 7, 123), and within a seed the four arms of the rule. As in
# c5_train_all.sh, a run whose directory already holds run.json is skipped,
# whatever its status, so a spoiled attempt is never overwritten. Its one
# repeat is run by hand, with its log kept the same way:
#   python g5_train.py --map SELECTION_DIR/maps/ARM.json --seed SEED --attempt 2 \
#       --data-dir DATA_DIR --out-dir OUT_DIR/ARM-sSEED-a2 <model args...> \
#       2>&1 | tee OUT_DIR/ARM-sSEED-a2.log
# A failed command or incomplete log stops the loop, keeping the failed attempt on disk.
set -u -o pipefail
if [ "$#" -lt 3 ]; then
  echo "usage: $0 DATA_DIR OUT_DIR SELECTION_DIR <model args...>" >&2
  exit 2
fi
data=$1
out=$2
sel=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd) || exit 1
mkdir -p "$out" || exit 1
for seed in 42 7 123; do
  for arm in mixed_traffic_f25 mixed_sensitivity_f25 mixed_traffic_f50 mixed_sensitivity_f50; do
    dir="$out/$arm-s$seed-a1"
    if [ -e "$dir/run.json" ]; then
      echo "== skip $dir: run.json exists"
      continue
    fi
    echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $arm seed $seed"
    python "$here/g5_train.py" --map "$sel/maps/$arm.json" --seed "$seed" --data-dir "$data" \
      --out-dir "$dir" "$@" 2>&1 | tee "$out/$arm-s$seed-a1.log"
    statuses=("${PIPESTATUS[@]}")
    if [ "${statuses[1]}" != 0 ]; then
      # the run's log is incomplete (disk full, ...): stop, do not start the next arm
      echo "== stop: tee could not write $arm-s$seed-a1.log (exit ${statuses[1]})" >&2
      exit 1
    fi
    if [ "${statuses[0]}" != 0 ]; then
      echo "== stop: $arm seed $seed exited ${statuses[0]}; kept its record and log" >&2
      exit "${statuses[0]}"
    fi
    if [ ! -e "$dir/run.json" ]; then
      echo "== stop: $arm seed $seed wrote no run.json" >&2
      exit 1
    fi
  done
done
