#!/usr/bin/env bash
# One phase-2 cell: power/clock gate first, then the phase-1 measurement under systemd-inhibit.
# Caller sets experiment env (PYTORCH_CUDA_ALLOC_CONF, CUBLASLT_WORKSPACE_SIZE, A0_*).
set -u

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 1
REPO="$(cd -- "$HERE/../.." && pwd)" || exit 1
VENV="${VENV:-/home/m1kadoo/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-/home/m1kadoo/moonlight-local}"
REV="${REV:-2a9a177f4841649f849ba941e41d2ee73c042314}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"

lever="" arm="" k=8 seq=512 steps=3 note=""
while (( $# )); do
  case "$1" in
    --lever) lever="${2:?}"; shift 2 ;;
    --arm) arm="${2:?}"; shift 2 ;;
    --k) k="${2:?}"; shift 2 ;;
    --seq) seq="${2:?}"; shift 2 ;;
    --steps) steps="${2:?}"; shift 2 ;;
    --note) note="${2:-}"; shift 2 ;;
    *) printf 'Unknown argument %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ -z "$lever" || ( "$arm" != "C" && "$arm" != "E" ) ]]; then
  printf 'Usage: %s --lever NAME --arm C|E [--k 8] [--seq 512] [--steps 3] [--note TEXT]\n' "$0" >&2
  exit 2
fi

mkdir -p -- "$HERE/raw/logs" || exit 1
export CUDA_HOME="$("$VENV/bin/python" -c \
  "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")" || exit 1
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

name="$lever-$arm-k$k-s$seq"
printf '== %s gate %s\n' "$name" "$(date +%T)"
"$VENV/bin/python" "$HERE/gate.py" > "$HERE/raw/logs/$name.gate.log" 2>&1
gate_status=$?
cat "$HERE/raw/logs/$name.gate.log"
if (( gate_status != 0 )); then
  printf 'Refusing %s: the power/clock gate failed.\n' "$name" >&2
  exit "$gate_status"
fi

printf '== %s start %s\n' "$name" "$(date +%T)"
systemd-inhibit --what=sleep --why="ticket13-phase2 $name" \
  "$VENV/bin/python" -u "$HERE/run_mem2.py" \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" --out "$HERE" \
    --arm "$arm" --k "$k" --seq "$seq" --steps "$steps" --model-revision "$REV" \
    --lora-targets "$TARGETS" --lever "$lever" --note "$note" \
  > "$HERE/raw/logs/$name.log" 2>&1
status=$?
printf '== %s exit=%s %s\n' "$name" "$status" "$(date +%T)"
tail -n 2 "$HERE/raw/logs/$name.log"
exit "$status"
