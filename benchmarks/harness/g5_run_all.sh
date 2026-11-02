#!/usr/bin/env bash
# G5: the whole check under the rule, in its order, one process per step.
#
#   g5_run_all.sh real  DATA_DIR OUT_DIR   # OLMoE-1B-7B on CUDA (the GPU runbook)
#   g5_run_all.sh pilot DATA_DIR OUT_DIR   # granite fixture on CPU, SYNTHETIC
#
# DATA_DIR is the output of c5_data.py. Steps (benchmarks/gate-g5-mixed-precision-experts.md):
#   0. provenance: C5's code here must hash as in C5's NF4 seed 42 record (LF checkout);
#      a real run stops here if not, a pilot records it and goes on;
#   1. selection: routing statistics and the sensitivity proxy on the train split, the maps;
#   2. bytes: full cache bytes of all-NF4, all-Q2_0 and the four maps;
#   3. control: C5's NF4 seed 42 on this box, compared with C5's record;
#   4. baselines: if the control fails, C5's seven arms are re-run here; else C5's are reused;
#   5. the twelve mixed arms;
#   6. the verdict and the quality-vs-bytes chart.
# Existing outputs resume the session; cached selection and bytes must still pass their checks.
# Any command or log failure stops before the next step, keeping every attempt on disk.
set -u -o pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 real|pilot DATA_DIR OUT_DIR" >&2
  exit 2
fi
mode=$1
data=$2
out=$3
here=$(cd "$(dirname "$0")" && pwd) || exit 1
repo=$(cd "$here/../.." && pwd) || exit 1
case "$mode" in
  real)
    model=(--model allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5)
    device=(--device cuda)
    train_extra=(--gradient-checkpointing)
    select_extra=()
    provenance_required=1
    c5_runs=$repo/benchmarks/results/c5-train/runs
    bytes_extra=(--cache-json "$repo/benchmarks/results/c5-cache/cache.json")
    verdict_extra=(--swap-record "$repo/benchmarks/results/c5-olmoe-swap/nf4.json"
                   --swap-record "$repo/benchmarks/results/c5-olmoe-swap/q2_0.json")
    ;;
  pilot)
    model=(--fixture granite)
    device=(--device cpu)
    train_extra=(--train-limit 64 --panel-limit 32)
    select_extra=(--calibration-rows 32)
    provenance_required=0
    c5_runs=$repo/benchmarks/results/c5-train-synthetic
    bytes_extra=()
    verdict_extra=(--synthetic)
    ;;
  *)
    echo "usage: $0 real|pilot DATA_DIR OUT_DIR" >&2
    exit 2
    ;;
esac
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
need_runs() {  # need_runs DIR NAME...: every DIR/NAME/run.json exists, else a stop
  local dir=$1 missing=() name
  shift
  for name in "$@"; do
    [ -e "$dir/$name/run.json" ] || missing+=("$name")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "== $(stamp) stop: no run.json in $dir for ${missing[*]} (a run that never" \
      "started, or was killed before writing its record)" >&2
    return 1
  fi
}
tee_ok() {  # tee_ok STATUS LOGNAME: a log that tee could not write is a stop (disk full, ...)
  if [ "$1" != 0 ]; then
    echo "== $(stamp) stop: tee could not write $2 (exit $1): the step's log is incomplete" >&2
    return 1
  fi
}
check_cached() {
  python - "$1" "$mode" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
synthetic = sys.argv[2] == "pilot"
if path.name == "selection.json":
    kind = "g5_selection"
    required = ("model_unchanged_by_selection", "every_token_routed_top_k",
                "forward_check_within_limit", "maps_equal_budget_per_fraction")
else:
    kind = "g5_bytes"
    required = ("mixed_arms_of_one_fraction_equal_bytes",)
    if not synthetic:
        required += ("all_nf4_layer_files_match_c5", "all_nf4_ssd_bytes_match_c5",
                     "all_q2_0_layer_files_match_c5", "all_q2_0_ssd_bytes_match_c5")
checks = record.get("checks")
if (record.get("kind") != kind or record.get("synthetic") is not synthetic
        or not isinstance(checks, dict) or not all(checks.get(key) is True for key in required)
        or not all(value is True for value in checks.values())):
    sys.stderr.write(f"stop: cached {path} failed its checks or belongs to another mode\n")
    sys.exit(1)
PY
}
run_baseline() {
  local arm=$1 seed=$2 dir="$out/baselines/$1-s$2-a1"
  local -a statuses
  if [ -e "$dir/run.json" ]; then
    echo "== skip $dir: run.json exists"
    return 0
  fi
  echo "== $(stamp) baseline: $arm seed $seed"
  python "$here/c5_train.py" --arm "$arm" --seed "$seed" "${model[@]}" "${device[@]}" \
    "${train_extra[@]}" --data-dir "$data" --out-dir "$dir" 2>&1 \
    | tee "$out/baselines/$arm-s$seed-a1.log"
  statuses=("${PIPESTATUS[@]}")
  tee_ok "${statuses[1]}" "baselines/$arm-s$seed-a1.log" || return 1
  if [ "${statuses[0]}" != 0 ]; then
    echo "== $(stamp) stop: $arm seed $seed exited ${statuses[0]}; kept its record and log" >&2
    return "${statuses[0]}"
  fi
  need_runs "$out/baselines" "$arm-s$seed-a1"
}
mkdir -p "$out" || exit 1
commit=$(git -C "$repo" rev-parse HEAD) || exit 1
echo "== $(stamp) G5 $mode, commit $commit"

echo "== $(stamp) provenance"
git -C "$repo" ls-files --eol -- 'benchmarks/harness/c5_*' benchmarks/harness/q2_0_codec.py \
  'benchmarks/harness/g5_*' > "$out/eol.txt" || exit 1
python "$here/g5_verdict.py" provenance "$c5_runs/nf4-s42-a1/run.json" \
  --out "$out/provenance.json" 2>&1 | tee "$out/provenance.log"
statuses=("${PIPESTATUS[@]}")
provenance_status=${statuses[0]}
tee_ok "${statuses[1]}" provenance.log || exit 1
case "$provenance_status" in
  0) ;;
  1)
    if [ "$provenance_required" = 1 ]; then
      echo "== $(stamp) stop: C5's code here is not the code of C5's records (see provenance.log);" \
        "check out the rule's commit with LF files, as .gitattributes sets them" >&2
      exit 1
    fi
    echo "== $(stamp) provenance differs (pilot): recorded, the control will fail same_c5_code"
    ;;
  *)
    echo "== $(stamp) stop: the provenance check itself failed (exit $provenance_status)" >&2
    exit 1
    ;;
esac

sel=$out/selection
if [ ! -e "$sel/selection.json" ]; then
  echo "== $(stamp) selection"
  python "$here/g5_select.py" "${model[@]}" "${device[@]}" "${select_extra[@]}" \
    --data-dir "$data" --out-dir "$sel" 2>&1 | tee "$out/selection.log" || exit 1
else
  check_cached "$sel/selection.json" || exit 1
fi

if [ ! -e "$sel/bytes.json" ]; then
  echo "== $(stamp) bytes"
  maps=()
  for arm in mixed_traffic_f25 mixed_sensitivity_f25 mixed_traffic_f50 mixed_sensitivity_f50; do
    maps+=(--map "$sel/maps/$arm.json")
  done
  python "$here/g5_bytes.py" "${model[@]}" "${maps[@]}" "${bytes_extra[@]}" \
    --out "$sel/bytes.json" 2>&1 | tee "$out/bytes.log" || exit 1
else
  check_cached "$sel/bytes.json" || exit 1
fi

control=$out/baselines/nf4-s42-a1
mkdir -p "$out/baselines" || exit 1
run_baseline nf4 42 || exit $?
python "$here/g5_verdict.py" control "$control/run.json" "$c5_runs/nf4-s42-a1/run.json" \
  --out "$out/control.json" 2>&1 | tee "$out/control.log"
statuses=("${PIPESTATUS[@]}")
control_status=${statuses[0]}
tee_ok "${statuses[1]}" control.log || exit 1
case "$control_status" in
  0)
    echo "== $(stamp) control passed: C5's baselines are reused"
    baselines=$c5_runs
    ;;
  1)
    echo "== $(stamp) control failed: C5's seven arms are re-run on this box"
    # C5's historical wrapper masks command failures; keep its code and run order unchanged.
    for run in nf4:42 q2_0:42 nf4:7 q2_0:7 nf4:123 q2_0:123 bf16:42; do
      run_baseline "${run%%:*}" "${run##*:}" || exit $?
    done
    baselines=$out/baselines
    need_runs "$baselines" nf4-s42-a1 q2_0-s42-a1 nf4-s7-a1 q2_0-s7-a1 nf4-s123-a1 \
      q2_0-s123-a1 bf16-s42-a1 || exit 1
    ;;
  *)
    echo "== $(stamp) stop: the control check itself failed (exit $control_status)" >&2
    exit 1
    ;;
esac

echo "== $(stamp) mixed arms"
bash "$here/g5_train_all.sh" "$data" "$out/mixed" "$sel" "${model[@]}" "${device[@]}" \
  "${train_extra[@]}" || { echo "== $(stamp) stop: g5_train_all.sh failed" >&2; exit 1; }
mixed_dirs=()
for seed in 42 7 123; do
  for arm in mixed_traffic_f25 mixed_sensitivity_f25 mixed_traffic_f50 mixed_sensitivity_f50; do
    mixed_dirs+=("$arm-s$seed-a1")
  done
done
need_runs "$out/mixed" "${mixed_dirs[@]}" || exit 1

echo "== $(stamp) verdict"
python "$here/g5_verdict.py" "$out/mixed" --baselines "$baselines" \
  --control "$control/run.json" --control-reference "$c5_runs/nf4-s42-a1/run.json" \
  --selection "$sel/selection.json" --bytes "$sel/bytes.json" "${verdict_extra[@]}" \
  --out "$out/verdict.json" 2>&1 | tee "$out/verdict.log"
statuses=("${PIPESTATUS[@]}")
verdict_status=${statuses[0]}
tee_ok "${statuses[1]}" verdict.log || exit 1
if [ "$verdict_status" != 0 ]; then
  echo "== $(stamp) stop: the verdict failed (exit $verdict_status); see verdict.log" >&2
  exit "$verdict_status"
fi
echo "== $(stamp) G5 $mode done"
