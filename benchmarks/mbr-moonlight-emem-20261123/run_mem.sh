#!/usr/bin/env bash
# Ticket 13: measure Moonlight C and E in fresh processes. Keep going after a failed run.
set -u

if (( $# == 1 )) && [[ $1 == --help ]]; then
  printf '%s\n' 'Usage: [VENV=DIR] [MOONLIGHT_ROOT=DIR] [OUT=DIR] [REV=HEX] run_mem.sh'
  exit 0
fi
if (( $# )); then
  printf 'Unexpected arguments. See %s --help.\n' "$0" >&2
  exit 2
fi

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 1
REPO="$(cd -- "$HERE/../.." && pwd)" || exit 1
VENV="${VENV:-/home/m1kadoo/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-/home/m1kadoo/moonlight-local}"
OUT="${OUT:-$HERE}"
REV="${REV:-2a9a177f4841649f849ba941e41d2ee73c042314}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"

if [[ -e "$OUT/rows_canonical.jsonl" || -L "$OUT/rows_canonical.jsonl" ]]; then
  printf 'Refusing to start: %s/rows_canonical.jsonl already exists. Move it to raw/ first.\n' \
    "$OUT" >&2
  exit 1
fi
mkdir -p -- "$OUT/raw/logs" || exit 1

CUDA_HOME="$("$VENV/bin/python" -c \
  "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")" || exit 1
export CUDA_HOME
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

{
  "$VENV/bin/python" -c \
    "from soup_cli.utils.expert_gemm import detect_environment; print(detect_environment())"
  nvidia-smi --query-gpu=name,memory.total,clocks.sm,power.limit,driver_version --format=csv
  "$VENV/bin/python" -c \
    'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda)'
} > "$OUT/raw/environment.txt" 2>&1

failed=0
for window in 8:64 8:512 64:64; do
  k="${window%:*}"
  seq="${window#*:}"
  for arm in C E; do
    name="mem-$arm-k$k-s$seq"
    printf '== %s start %s\n' "$name" "$(date +%T)"
    if [[ "$(cat /sys/class/power_supply/ACAD/online 2>/dev/null)" == 1 ]]; then
      systemd-inhibit --what=sleep --why="ticket13 $arm ${k}x$seq" \
        "$VENV/bin/python" -u "$HERE/run_mem.py" \
        --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" --out "$OUT" \
        --arm "$arm" --k "$k" --seq "$seq" --steps 3 --model-revision "$REV" \
        --lora-targets "$TARGETS" > "$OUT/raw/logs/$name.log" 2>&1
      status=$?
    else
      printf 'Refusing %s: AC power is not online.\n' "$name" >&2
      status=1
    fi
    printf '== %s exit=%s %s\n' "$name" "$status" "$(date +%T)"
    if (( status != 0 )); then
      (( failed += 1 ))
    fi
  done
done
exit "$failed"
