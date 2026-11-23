#!/usr/bin/env bash
# Ticket 06 part 2: the repeat of every cell the first bits pass reported as ``differs``.
#
# The campaign rule: one run per cell decides, and a ``differs`` cell is repeated in a new
# folder before it is written down (ticket 03's rule). This repeats the whole S=64 ladder, the
# k=64 cell and the S=512 math cell with fresh processes; the comparison script then checks each
# repeat against the first pass and each other pairwise.
#
# Inputs, overridable: VENV, MOONLIGHT_ROOT, OUT. Refuses to run over existing rows.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-decoupled-20261124}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"
MODEL=(--source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache")
ACAD=/sys/class/power_supply/ACAD/online

for folder in repeat-s64 repeat-k64 repeat-s512-math; do
  if [ -e "$OUT/$folder/rows.jsonl" ]; then
    echo "refusing to run: $OUT/$folder/rows.jsonl exists. Move it away first." >&2
    exit 1
  fi
done

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
INHIBIT=(systemd-inhibit --what=sleep --why="ticket 06 decoupled bits repeats")
command -v systemd-inhibit > /dev/null || INHIBIT=()

run_pass() {
  local name="$1"; shift
  if [[ "$(cat "$ACAD" 2>/dev/null)" != 1 ]]; then
    printf 'Refusing %s: AC power is not online.\n' "$name" >&2
    return 1
  fi
  echo "== $name start $(date +%T)"
  "${INHIBIT[@]}" "$VENV/bin/python" -u "$HARNESS" "$@" "${MODEL[@]}" \
    --lora-targets "$TARGETS" --steps 3 --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local status=$?
  echo "== $name exit=$status $(date +%T)"
  return "$status"
}

failed=0
run_pass repeat-s64 lever-bits --seq 64 --ladder-ks 1,2,4,8 --arms A0,C,D || (( failed += 1 ))
run_pass repeat-k64 lever-bits --seq 64 --ladder-ks 64 --arms C,D || (( failed += 1 ))
run_pass repeat-s512-math lever-bits --seq 512 --ladder-ks 8 --arms A0,C,D --math-attention \
  || (( failed += 1 ))

echo "repeats done, failed passes: $failed $(date +%T)"
exit "$failed"
