#!/usr/bin/env bash
# Run the Moonlight ticket 12 matrix. A failed measurement must not stop the next one.
set -uo pipefail

if [[ ${1:-} == --help ]]; then
  printf '%s\n' "Usage: [VENV=DIR] [MOONLIGHT_ROOT=DIR] [OUT=DIR] $0"
  exit 0
fi
if (( $# )); then
  printf '%s\n' "Unexpected arguments. See $0 --help." >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)" || exit 1
REPO="$(cd "$HERE/../.." && pwd)" || exit 1
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$HERE/raw}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
ARM=A0
K=1

if [[ "$(cat /sys/class/power_supply/ACAD/online 2>/dev/null)" != 1 ]]; then
  printf '%s\n' "Refusing to start: AC power is not online." >&2
  exit 1
fi

# Each entry gives the tag, sequence length, repeat number, and whether to capture gradients.
plan=(
  "default 512 1 1"
  "default 512 2 1"
  "default 512 3 0"
  "math 512 1 0"
  "math 512 2 0"
  "default 64 1 0"
  "default 64 2 0"
  "strict 512 1 0"
)
run_name() {
  printf '%s-%s-k%s-rep%s-s%s' "$ARM" "$1" "$K" "$3" "$2"
}

destinations=(matrix.log probe)
for entry in "${plan[@]}"; do
  read -r tag seq rep capture <<< "$entry"
  destinations+=("$(run_name "$tag" "$seq" "$rep")")
done
for name in "${destinations[@]}"; do
  if [[ -e "$OUT/$name" || -L "$OUT/$name" ]]; then
    printf 'Refusing to start: %s already exists.\n' "$OUT/$name" >&2
    exit 1
  fi
done

mkdir -p -- "$OUT" || exit 1
matrix_log="$OUT/matrix.log"
if ! (set -o noclobber; : > "$matrix_log"); then
  printf 'Cannot create %s.\n' "$matrix_log" >&2
  exit 1
fi

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONWARNINGS=always

inhibited_python() {
  systemd-inhibit --what=sleep --why='mbr nondet ticket 12' "$VENV/bin/python" "$@"
}

CUDA_HOME="$(inhibited_python -c \
  'import os, sys, nvidia; sys.stdout.write(os.path.join(next(iter(nvidia.__path__)), "cu13"))')"
status=$?
printf 'cuda-home exit=%s\n' "$status" >> "$matrix_log"
if (( status )) || [[ -z "$CUDA_HOME" ]]; then
  printf '%s\n' "Cannot find CUDA_HOME in the venv's nvidia package." >&2
  exit 1
fi
export CUDA_HOME

run_logged() {
  local name=$1 runner=$2 status dir
  shift 2
  dir="$OUT/$name"
  if mkdir -- "$dir"; then
    inhibited_python -u "$HERE/$runner" "$@" --out "$dir" \
      --warnings-file "$dir/warnings.txt" > "$dir/run.log" 2>&1
    status=$?
  else
    status=1
  fi
  printf '%s exit=%s\n' "$name" "$status" >> "$matrix_log"
  return "$status"
}

# A strict run may fail only because torch refuses a non-deterministic operation; the log then
# names the operation. Anything else is a real failure of the measurement.
strict_determinism_op() {
  sed -nE \
    's/.*RuntimeError: ([^ ]+) does not have a deterministic implementation.*/\1/p' \
    "$1" | head -n 1
}

failed=0
run_logged probe probe_sdpa.py || failed=1
for entry in "${plan[@]}"; do
  read -r tag seq rep capture <<< "$entry"
  name="$(run_name "$tag" "$seq" "$rep")"
  backend=$tag
  extra=()
  if [[ $capture == 1 ]]; then
    extra+=(--capture)
  fi
  if [[ $tag == strict ]]; then
    backend=default
    extra+=(--strict-determinism)
  fi
  if run_logged "$name" run_capture.py --source "$MOONLIGHT_ROOT/source" \
      --cache "$MOONLIGHT_ROOT/cache" --arm "$ARM" --k "$K" --seq "$seq" \
      --steps 3 --lora-targets "$TARGETS" --backend "$backend" --deterministic \
      --tag "$name" "${extra[@]}"; then
    continue
  fi
  status=$?
  if [[ $tag != strict ]]; then
    failed=1
    printf 'Run %s failed (exit %s). See %s.\n' "$name" "$status" "$OUT/$name/run.log" >&2
    continue
  fi
  if op="$(strict_determinism_op "$OUT/$name/run.log")" && [[ -n $op ]]; then
    printf '%s deterministic-op-unsupported %s\n' "$name" "$op" >> "$matrix_log"
    continue
  fi
  failed=1
  printf 'Strict run %s failed (exit %s) without a deterministic-op error naming an operation in %s.\n' \
    "$name" "$status" "$OUT/$name/run.log" >&2
done
exit "$failed"
