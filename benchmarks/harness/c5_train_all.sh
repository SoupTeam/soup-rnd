#!/usr/bin/env bash
# C5 ticket 04: the seven runs of rule 03 in a fixed order, one process each.
#
#   c5_train_all.sh DATA_DIR OUT_DIR <c5_train.py model args...>
#   c5_train_all.sh data runs --model allenai/OLMoE-1B-7B-0924 \
#       --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --device cuda
#
# A run whose directory already holds run.json is skipped, whatever its status,
# so a spoiled attempt is never overwritten. Its one repeat is run by hand, with
# its log kept the same way:
#   python c5_train.py --arm ARM --seed SEED --attempt 2 --data-dir DATA_DIR \
#       --out-dir OUT_DIR/ARM-sSEED-a2 <model args...> 2>&1 | tee OUT_DIR/ARM-sSEED-a2.log
set -u -o pipefail
data=$1
out=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
for run in nf4:42 q2_0:42 nf4:7 q2_0:7 nf4:123 q2_0:123 bf16:42; do
  arm=${run%%:*}
  seed=${run##*:}
  dir="$out/$arm-s$seed-a1"
  if [ -e "$dir/run.json" ]; then
    echo "== skip $dir: run.json exists"
    continue
  fi
  echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $arm seed $seed"
  python "$here/c5_train.py" --arm "$arm" --seed "$seed" --data-dir "$data" \
    --out-dir "$dir" "$@" 2>&1 | tee "$out/$arm-s$seed-a1.log" \
    || echo "== $arm seed $seed exited non-zero; its run.json records why"
done
