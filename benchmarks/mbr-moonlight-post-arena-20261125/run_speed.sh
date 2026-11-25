#!/usr/bin/env bash
# Ticket 17, item 2b: G against E and C after the arena, C-E-G-G-E-C on (8,64) and (8,512).
#
# mbr_measure ceg-speed: 5 blocks of C-E-G-G-E-C on the two k=8 windows, every run in a fresh
# process inside ArmWatch with the suspend, power and foreign_reads checks required
# (speed-rule.md), the ticket 13 health gate before every run, and all three hands under the one
# memory recipe of ticket 13 phase 2 (expandable_segments plus the lt-fused arena sized for
# k*S = 64*64 = 4096 rows). G is E plus async_lease plus fast_row_gather; C is the plain ordered
# hand. A0 has no role: this ticket asks where G sits between E and C.
#
# Set OUT to a fresh campaign root. RESUME=1 repairs an interrupted run created by this launcher.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-post-arena-20261125}"
RESUME="${RESUME:-0}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"
GATE="$REPO/benchmarks/mbr-moonlight-emem2-20261124/gate.py"
ACAD=/sys/class/power_supply/ACAD/online

export PATH="$VENV/bin:$PATH"
if ! command -v systemd-inhibit > /dev/null; then
  echo "refusing to run: systemd-inhibit is required" >&2
  exit 1
fi
mkdir -p "$OUT"
exec 9> "$OUT/.speed.lock"
if ! flock -n 9; then
  echo "refusing to run: another campaign holds $OUT/.speed.lock" >&2
  exit 1
fi
if [[ "$RESUME" = 1 ]]; then
  if [[ ! -f "$OUT/speed/.ticket17-active" ]]; then
    echo "refusing to resume: no active campaign at $OUT/speed" >&2
    exit 1
  fi
elif [[ -e "$OUT/speed" || -L "$OUT/speed" ]]; then
  echo "refusing to overwrite $OUT/speed; use a fresh OUT" >&2
  exit 1
fi

CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export CUDA_HOME
export PYTHONPATH="$REPO/src:$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export A0_SCRATCH_PREALLOC_ROWS=4096
export A0_SCRATCH_PREALLOC_HID=2048
export A0_SCRATCH_PREALLOC_INTER=1408
export A0_SCRATCH_WHEN=lazy
export A0_SCRATCH_PREALLOC_GATHER=1

INHIBIT=(systemd-inhibit --what=sleep --why="ticket 17 C-E-G-G-E-C speed blocks")
if [[ "$(cat "$ACAD" 2>/dev/null)" != 1 ]]; then
  echo "Refusing the speed pass: AC power is not online." >&2
  exit 1
fi

if [[ "$RESUME" != 1 ]]; then
  mkdir "$OUT/speed"
  : > "$OUT/speed/.ticket17-active"
  echo "== C-E-G-G-E-C pilot start $(date +%T)"
  "${INHIBIT[@]}" "$VENV/bin/python" -u "$HARNESS" ceg-speed --require-complete \
    --gate "$VENV/bin/python" "$GATE" \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" --out "$OUT/speed" \
    > "$OUT/speed.log" 2>&1
  if [[ ! -s "$OUT/speed/rows.jsonl" || ! -s "$OUT/speed/arm_records.jsonl" ]]; then
    echo "missing speed sweep output; see $OUT/speed.log" >&2
    exit 1
  fi
fi

echo "== replacements and top-up start $(date +%T)"
"${INHIBIT[@]}" "$VENV/bin/python" -u \
  "$REPO/benchmarks/mbr-moonlight-post-arena-20261125/replace_void.py" \
  --out "$OUT/speed" --gate "$GATE" \
  --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
  --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" >> "$OUT/replace.log" 2>&1
if [[ ! -s "$OUT/speed/rows.jsonl" || ! -s "$OUT/speed/arm_records.jsonl" ]]; then
  echo "missing speed output; see $OUT/replace.log" >&2
  exit 1
fi
echo "== speed complete $(date +%T)"
