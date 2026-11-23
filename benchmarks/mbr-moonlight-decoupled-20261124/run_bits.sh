#!/usr/bin/env bash
# Ticket 06 part 2: the bits ladder of arm D (decoupled MBR) against the reference.
#
# Reference rule of the campaign: A0 at k <= 8, C above it (mbr_measure.reference_arm).
#   1. S=64, default backend, k = 1,2,4,8: A0, C and D.  -> (1..8,64)
#   2. S=64, default backend, k = 64:      C and D.      -> (64,64), recorded, not blocking
#   3. S=512, math backend, k = 8:         A0, C and D.  -> (8,512) math
#   4. S=512, deterministic, k = 64:       C and D.      -> (64,512), the large-window event
# Deterministic mode, three steps per run (one warm-up). torch.equal on the saved loss and
# the 216 LoRA gradients decides; the harness repeats nothing by itself, a differs cell is
# repeated by hand in a new folder.
#
# Inputs, overridable: VENV, MOONLIGHT_ROOT, OUT (default this campaign folder), TENSOR_DIR
# (default $MOONLIGHT_ROOT/results/decoupled-20261124; the raw tensors are git-ignored inside
# OUT because each is ~25 MB).
# Refuses to run over existing rows.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-decoupled-20261124}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"
MODEL=(--source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache")
ACAD=/sys/class/power_supply/ACAD/online

for folder in s64 k64 s512-math k64-s512; do
  if [ -e "$OUT/$folder/rows.jsonl" ]; then
    echo "refusing to run: $OUT/$folder/rows.jsonl exists. Move it away first." >&2
    exit 1
  fi
done
mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
INHIBIT=(systemd-inhibit --what=sleep --why="ticket 06 decoupled bits")
command -v systemd-inhibit > /dev/null || INHIBIT=()

{
  "$VENV/bin/python" -c "from soup_cli.utils.expert_gemm import detect_environment; print(detect_environment())"
  "$VENV/bin/python" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda)'
  nvidia-smi --query-gpu=name,memory.total,clocks.sm,power.limit,driver_version --format=csv
} > "$OUT/environment.txt" 2>&1

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
run_pass s64 lever-bits --seq 64 --ladder-ks 1,2,4,8 --arms A0,C,D || (( failed += 1 ))
run_pass k64 lever-bits --seq 64 --ladder-ks 64 --arms C,D || (( failed += 1 ))
run_pass s512-math lever-bits --seq 512 --ladder-ks 8 --arms A0,C,D --math-attention \
  || (( failed += 1 ))
run_pass k64-s512 lever-bits --seq 512 --ladder-ks 64 --arms C,D || (( failed += 1 ))

echo "bits done, failed passes: $failed $(date +%T)"
exit "$failed"
