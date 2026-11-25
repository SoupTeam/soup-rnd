#!/usr/bin/env bash
# Ticket 17, item 1 and the bit halves of items 2 and 3: the E ladder after the arena, G at
# k=2, and D with segmented against A0.
#
# Five bits sweeps, deterministic mode, every run a fresh process inside ArmWatch with the
# suspend, power and foreign_reads checks required, the ticket 13 health gate before every run,
# and the memory recipe of ticket 13 phase 2 with the arena sized for the widest visit
# (k*S = 64*64 = 4096 rows):
#
#   s64        A0 and E at k=1,2,4,8, S=64 (the row (8,64) and (64,64) share; ticket 03)
#   s512-math  A0 and E at k=1,2,4,8, S=512, math SDPA (the only valid S=512 row, ticket 03)
#   g-k2-s64   A0 and G at k=2, S=64: G's bits under the new memory path
#   d-k2-s64   A0 and D at k=2, S=64: the same-day plain-D control for the DS cell
#   ds-k2-s64  A0 and DS at k=2, S=64: D's decoupled step with expert_gemm=segmented
#
# The E, G and D ladders are `lever-bits` (the `bits` plan carries only A0/B0/C); `--arms`
# keeps the reference and the one hand under test. `ds-bits` plans its A0-DS pair itself.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default this folder).
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-post-arena-20261125}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"
GATE="$REPO/benchmarks/mbr-moonlight-emem2-20261124/gate.py"
ACAD=/sys/class/power_supply/ACAD/online

mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export A0_SCRATCH_PREALLOC_ROWS=4096
export A0_SCRATCH_PREALLOC_HID=2048
export A0_SCRATCH_PREALLOC_INTER=1408
export A0_SCRATCH_WHEN=lazy
export A0_SCRATCH_PREALLOC_GATHER=1

INHIBIT=(systemd-inhibit --what=sleep --why="ticket 17 post-arena bits blocks")
command -v systemd-inhibit > /dev/null || INHIBIT=()

exec 9> "$OUT/.bits.lock"
if ! flock -n 9; then
  echo "refusing to run: another campaign holds $OUT/.bits.lock" >&2
  exit 1
fi
if compgen -G "$OUT/bits/*/rows.jsonl" > /dev/null; then
  echo "refusing to run: $OUT/bits already has rows. Move them away first." >&2
  exit 1
fi
if [[ "$(cat "$ACAD" 2>/dev/null)" != 1 ]]; then
  echo "Refusing the bits pass: AC power is not online." >&2
  exit 1
fi

failed=0
sweep() {  # sweep <label> <bits|lever-bits|ds-bits> <args...>
  local label="$1" command="$2"
  shift 2
  if [[ "$(cat "$ACAD" 2>/dev/null)" != 1 ]]; then
    echo "Refusing $label: AC power is not online." >&2
    return 1
  fi
  echo "== $label start $(date +%T)"
  "${INHIBIT[@]}" "$VENV/bin/python" -u "$HARNESS" "$command" "$@" \
    --gate "$VENV/bin/python" "$GATE" \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" --out "$OUT/bits/$label" \
    > "$OUT/bits-$label.log" 2>&1
  local code=$?
  echo "== $label exit=$code $(date +%T)"
  return $code
}

sweep s64 lever-bits --seq 64 --ladder-ks 1,2,4,8 --arms A0,E || (( failed += 1 ))
sweep s512-math lever-bits --seq 512 --ladder-ks 1,2,4,8 --arms A0,E --math-attention \
  || (( failed += 1 ))
sweep g-k2-s64 lever-bits --seq 64 --ladder-ks 2 --arms A0,G || (( failed += 1 ))
sweep d-k2-s64 lever-bits --seq 64 --ladder-ks 2 --arms A0,D || (( failed += 1 ))
sweep ds-k2-s64 ds-bits --seq 64 --arms A0,DS || (( failed += 1 ))
echo "== bits done, failed sweeps: $failed $(date +%T)"
exit $(( failed > 0 ))
