#!/usr/bin/env bash
# Ticket 14 tail: the price of the segmented expert GEMM (arm S) against C and A0.
#
# mbr_measure seg-speed: 5 blocks of A0-C-S-S-C-A0 on the three windows that fit 4 GiB,
# (8,64), (8,512) and (64,64), every run in a fresh process inside ArmWatch with the suspend,
# power and foreign_reads checks required (speed-rule.md). A0 does not run at k=64, so (64,64)
# keeps C-S-S-C. The paired gain per block is the block mean of S over the block means of C and
# A0; the ticket reads the price of segmentation against C and the leftover speed against A0.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default this campaign
# folder: the raw rows and arm records stay inside the repository).
# Refuses to run over existing rows.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-seg-speed-20261124}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"
ACAD=/sys/class/power_supply/ACAD/online

if [ -e "$OUT/speed/rows.jsonl" ]; then
  echo "refusing to run: $OUT/speed/rows.jsonl exists. Move it away first." >&2
  exit 1
fi
mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

# The host must not suspend mid-block: ArmWatch would void the arm, and the campaign must survive
# (systemd-inhibit sleep for the whole campaign, replacements included).
INHIBIT=(systemd-inhibit --what=sleep --why="ticket 14 segmented speed blocks")
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

echo "== seg-speed A0-C-S-S-C-A0, 5 blocks, start $(date +%T)"
"${INHIBIT[@]}" "$VENV/bin/python" -u "$HARNESS" seg-speed --blocks 5 \
  --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
  --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" --out "$OUT/speed" \
  > "$OUT/speed.log" 2>&1
echo "== seg-speed exit=$? $(date +%T)"

echo "== void replacements start $(date +%T)"
if [ -e "$OUT/speed/rows.jsonl" ]; then
  if [[ "$(cat "$ACAD" 2>/dev/null)" != 1 ]]; then
    echo "Refusing the replacements: AC power is not online." >&2
    exit 1
  fi
  "${INHIBIT[@]}" "$VENV/bin/python" -u \
    "$REPO/benchmarks/mbr-moonlight-seg-speed-20261124/replace_void.py" \
    --out "$OUT/speed" --blocks 5 \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" \
    > "$OUT/replace.log" 2>&1
  echo "== void replacements exit=$? $(date +%T)"
else
  echo "== void replacements skipped: no speed rows $(date +%T)" | tee "$OUT/replace.log"
fi
