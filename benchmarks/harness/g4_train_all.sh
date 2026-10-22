#!/usr/bin/env bash
# G4 ticket 05: the seven runs of benchmarks/gate-g4-nvfp4.md in a fixed order, one process each.
#
#   g4_train_all.sh DATA_DIR OUT_DIR <g4_train.py args...>
#   g4_train_all.sh data runs --device cuda
#
# A run whose directory already holds run.json is skipped, whatever its status, so a spoiled
# attempt is never overwritten. Its one repeat is run by hand, with its log kept the same way:
#   python g4_train.py --arm ARM --seed SEED --attempt 2 --data-dir DATA_DIR \
#       --out-dir OUT_DIR/ARM-sSEED-a2 <args> 2>&1 | tee OUT_DIR/ARM-sSEED-a2.log
set -u -o pipefail
data=$1
out=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
for run in nf4:42 nvfp4:42 nf4:7 nvfp4:7 nf4:123 nvfp4:123 bf16:42; do
  arm=${run%%:*}
  seed=${run##*:}
  dir="$out/$arm-s$seed-a1"
  if [ -e "$dir/run.json" ]; then
    echo "== skip $dir: run.json exists"
    continue
  fi
  echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $arm seed $seed"
  python "$here/g4_train.py" --arm "$arm" --seed "$seed" --data-dir "$data" \
    --out-dir "$dir" "$@" 2>&1 | tee "$out/$arm-s$seed-a1.log" \
    || echo "== $arm seed $seed exited non-zero; its run.json records why"
done
