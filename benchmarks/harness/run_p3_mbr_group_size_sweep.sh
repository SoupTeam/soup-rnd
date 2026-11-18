#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/p3-baseline
REPO="$ROOT/repo-p3-mbr-20261117"
RUNNER="$REPO/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py"
OUT_ROOT="$ROOT/results/mbr_group_size_sweep_20261117"
STATUS="$OUT_ROOT/status.txt"

mkdir -p "$OUT_ROOT"
if [[ -e "$STATUS" ]]; then
  echo "STOP: status file already exists: $STATUS" >&2
  exit 2
fi

python3 - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
p = pathlib.Path('/sys/fs/cgroup')
limit = int((p / 'memory.max').read_text().strip())
current = int((p / 'memory.current').read_text().strip())
swap = int((p / 'memory.swap.max').read_text().strip())
ballast = json.loads(pathlib.Path('/tmp/p3-cgroup-ram-ballast.json').read_text())
assert ballast['status'] == 'holding', ballast
assert swap == 0, f'swap must remain disabled: {swap}'
assert limit - current >= 32 * 2**30, (limit, current)
print(json.dumps({
    'cgroup_limit_bytes': limit,
    'cgroup_current_bytes': current,
    'cgroup_headroom_bytes': limit-current,
    'swap_max_bytes': swap,
    'ballast_pid': ballast['pid'],
    'ballast_bytes': ballast['ballast_bytes'],
}, indent=2))
PY

export PYTHONPATH="$REPO/src:$REPO/benchmarks/harness${PYTHONPATH:+:$PYTHONPATH}"
source /venv/main/bin/activate

for group_size in 1 2 4 8; do
  output="$OUT_ROOT/group_${group_size}"
  log="$OUT_ROOT/group_${group_size}.log"
  if [[ -e "$output" ]]; then
    echo "STOP: output already exists: $output" >&2
    exit 2
  fi
  echo "START group_size=$group_size $(date -Is)" | tee -a "$STATUS"
  python "$RUNNER" \
    --root "$ROOT" \
    --source "$ROOT/source" \
    --cache "$ROOT/cache" \
    --data "$ROOT/repo/benchmarks/protocols/b5-moonlight-20261109/data.json" \
    --output "$output" \
    --mbr-group-size "$group_size" >"$log" 2>&1
  echo "DONE group_size=$group_size $(date -Is) result=$output/result.json" | tee -a "$STATUS"
done

python3 - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for group_size in (1, 2, 4, 8):
    path = root / f'group_{group_size}' / 'result.json'
    report = json.loads(path.read_text())
    if report.get('status') != 'PASS':
        raise SystemExit(f'FAIL: {path} status={report.get("status")}')
    timing = report['timing_summary']
    diag = report['diagnostic_gate'] if 'diagnostic_gate' in report else json.loads((path.parent/'diagnostic_gate.json').read_text())
    rows.append({
        'group_size_microbatches': group_size,
        'grouped_dispatches_per_layer': report['mbr_grouped_dispatches_per_layer'],
        'l2l_seconds_median': timing['l2l_seconds_median'],
        'mbr_seconds_median': timing['mbr_seconds_median'],
        'speedup_vs_l2l': timing['speedup_l2l_over_mbr'],
        'expert_fetches_l2l': report['measured'][0]['runtime_total']['expert_fetches'],
        'expert_fetches_mbr': report['measured'][1]['runtime_total']['expert_fetches'],
        'expert_bytes_l2l': report['measured'][0]['runtime_total']['expert_bytes_read'],
        'expert_bytes_mbr': report['measured'][1]['runtime_total']['expert_bytes_read'],
        'loss_delta_mbr_minus_l2l': report['l2l_vs_mbr_first_pair']['loss_delta_mbr_minus_l2l'],
        'gradient_rel_l2': report['l2l_vs_mbr_first_pair']['gradient']['rel_l2_vs_left'],
        'route_slot_mismatch_fraction': report['l2l_vs_mbr_first_pair']['routes']['slot_mismatch_fraction'],
        'memory_max_bytes': report['host_preflight']['cgroup_memory_max_bytes'],
        'memory_current_at_preflight_bytes': report['host_preflight']['cgroup_memory_current_bytes'],
        'p3_ram_limit_pass': report['host_preflight']['p3_ram_limit_pass'],
    })
summary = {
    'status': 'complete',
    'scope': 'Moonlight Soup bridge; fixed total 8x64 input window; routed-FFN group size sweep; not quality or P3 acceptance',
    'rows': rows,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY

echo "COMPLETE $(date -Is) summary=$OUT_ROOT/summary.json" | tee -a "$STATUS"
