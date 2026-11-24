#!/usr/bin/env bash
# Ticket 16: the three candidate default hands in one campaign, C-S-E-E-S-C.
#
# mbr_measure esc-speed: 5 blocks of C-S-E-E-S-C on the three windows that fit 4 GiB, (8,64),
# (8,512) and (64,64), every run in a fresh process inside ArmWatch with the suspend, power and
# foreign_reads checks required (speed-rule.md). A0 has no role: C, S and E exist on every
# window, so the palindrome is whole and the ticket's 90 runs are all planned.
#
# Every hand runs with the same allocator config: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# for all three (C and S included, so the memory of the three hands is comparable), and the
# lt-fused scratch arena sized for the largest visit of the campaign (k*S = 64*64 = 4096 rows,
# lazily allocated) for E. The arena options are inert for C and S: neither loads the extension.
#
# The ticket 13 health gate (AC online, SM clock >= 1500 MHz under load, enforced power limit
# >= 50 W) runs before every single run; a refusal stops the sweep instead of writing a row.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default this campaign
# folder), BLOCKS, START_BLOCK (resume after a gate refusal).
# Refuses to run over existing rows unless START_BLOCK > 0.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-esc-speed-20261125}"
BLOCKS="${BLOCKS:-5}"
START_BLOCK="${START_BLOCK:-0}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"
GATE="$REPO/benchmarks/mbr-moonlight-emem2-20261124/gate.py"
ACAD=/sys/class/power_supply/ACAD/online

if [ "$START_BLOCK" = "0" ] && [ -e "$OUT/speed/rows.jsonl" ]; then
  echo "refusing to run: $OUT/speed/rows.jsonl exists. Move it away first." >&2
  exit 1
fi
mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
# The memory recipe of ticket 13 phase 2, identical for the three hands.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export A0_SCRATCH_PREALLOC_ROWS=4096
export A0_SCRATCH_PREALLOC_HID=2048
export A0_SCRATCH_PREALLOC_INTER=1408
export A0_SCRATCH_WHEN=lazy
export A0_SCRATCH_PREALLOC_GATHER=1

# The host must not suspend mid-block: ArmWatch would void the arm, and the campaign must survive
# (systemd-inhibit sleep for the whole campaign, replacements included).
INHIBIT=(systemd-inhibit --what=sleep --why="ticket 16 C-S-E-E-S-C speed blocks")
command -v systemd-inhibit > /dev/null || INHIBIT=()

exec 9> "$OUT/.speed.lock"
if ! flock -n 9; then
  echo "refusing to run: another campaign holds $OUT/.speed.lock" >&2
  exit 1
fi
if [[ "$(cat "$ACAD" 2>/dev/null)" != 1 ]]; then
  echo "Refusing the speed pass: AC power is not online." >&2
  exit 1
fi

echo "== esc-speed C-S-E-E-S-C, blocks $BLOCKS from $START_BLOCK, start $(date +%T)"
"${INHIBIT[@]}" "$VENV/bin/python" -u "$HARNESS" esc-speed --blocks "$BLOCKS" \
  --start-block "$START_BLOCK" --gate "$VENV/bin/python" "$GATE" \
  --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
  --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" --out "$OUT/speed" \
  > "$OUT/speed.log" 2>&1
echo "== esc-speed exit=$? $(date +%T)"

echo "== void replacements start $(date +%T)"
if [ -e "$OUT/speed/rows.jsonl" ]; then
  if [[ "$(cat "$ACAD" 2>/dev/null)" != 1 ]]; then
    echo "Refusing the replacements: AC power is not online." >&2
    exit 1
  fi
  "${INHIBIT[@]}" "$VENV/bin/python" -u \
    "$REPO/benchmarks/mbr-moonlight-esc-speed-20261125/replace_void.py" \
    --out "$OUT/speed" --blocks "$BLOCKS" --gate "$GATE" \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" \
    > "$OUT/replace.log" 2>&1
  echo "== void replacements exit=$? $(date +%T)"
else
  echo "== void replacements skipped: no speed rows $(date +%T)" | tee "$OUT/replace.log"
fi
