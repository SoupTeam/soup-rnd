#!/usr/bin/env bash
# Ticket 15: why the segmented expert GEMM (arm S) is slower than C on this card.
#
# One traced step per cell and one py-spy snapshot per cell, on the two windows the ticket names:
# (8,64) at k=8 and (64,64) at k=64, arms C (native expert GEMM) and S (expert_gemm="segmented").
# The traced step writes the expert-GEMM counters and the kernel timeline into the row; py-spy
# writes the leaf shares of the Python dispatch against the kernels. Then analyze.py builds the
# layout and the attribution in analysis.json, layout.md and rows.md.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default this campaign
# folder: the raw rows, traces and profiles stay inside the repository).
# Refuses to run over existing rows.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-segprof-20261124}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
CAMPAIGN="$REPO/benchmarks/mbr-moonlight-segprof-20261124"
ACAD=/sys/class/power_supply/ACAD/online
# "k seq" of each profiled window: (8,64) at k=8 and (64,64) at k=64.
CELLS=("8 64" "64 64")

if [ -e "$OUT/runs/rows.jsonl" ]; then
  echo "refusing to run: $OUT/runs/rows.jsonl exists. Move it away first." >&2
  exit 1
fi
mkdir -p "$OUT/runs" "$OUT/trace" "$OUT/pyspy"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

# The host must not suspend mid-cell: ArmWatch would void the arm.
INHIBIT=(systemd-inhibit --what=sleep --why="ticket 15 segmented profile")
command -v systemd-inhibit > /dev/null || INHIBIT=()

exec 9> "$OUT/.profile.lock"
if ! flock -n 9; then
  echo "refusing to run: another campaign holds $OUT/.profile.lock" >&2
  exit 1
fi
if [[ "$(cat "$ACAD" 2>/dev/null)" != 1 ]]; then
  echo "Refusing the profile pass: AC power is not online." >&2
  exit 1
fi

echo "== traced steps start $(date +%T)"
for cell in "${CELLS[@]}"; do
  read -r k seq <<< "$cell"
  for arm in C S; do
    "${INHIBIT[@]}" "$VENV/bin/python" -u "$CAMPAIGN/profile_step.py" \
      --arm "$arm" --k "$k" --seq "$seq" --steps 4 --trace \
      --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
      --lora-targets "$TARGETS" --out "$OUT" \
      >> "$OUT/runs.log" 2>&1
    echo "== traced step arm=$arm k=$k seq=$seq exit=$? $(date +%T)"
  done
done

echo "== py-spy snapshots start $(date +%T)"
for cell in "${CELLS[@]}"; do
  read -r k seq <<< "$cell"
  for arm in C S; do
    "${INHIBIT[@]}" py-spy record --native --format speedscope \
      -o "$OUT/pyspy/$arm-k$k-s$seq.speedscope.json" \
      -- "$VENV/bin/python" -u "$CAMPAIGN/profile_step.py" \
      --arm "$arm" --k "$k" --seq "$seq" --steps 4 --pyspy \
      --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
      --lora-targets "$TARGETS" --out "$OUT/pyspy" \
      >> "$OUT/pyspy.log" 2>&1
    echo "== py-spy arm=$arm k=$k seq=$seq exit=$? $(date +%T)"
  done
done

PROFILES=("$OUT"/pyspy/*.speedscope.json)
if [ -e "${PROFILES[0]}" ]; then
  "$VENV/bin/python" -u "$CAMPAIGN/analyze_pyspy.py" --profile "${PROFILES[@]}" \
    --out "$OUT/pyspy/leaf-shares.json" > "$OUT/pyspy-analyze.log" 2>&1
  echo "== leaf shares exit=$? $(date +%T)"
else
  echo "== leaf shares skipped: no py-spy profile $(date +%T)"
fi

"$VENV/bin/python" -u "$CAMPAIGN/analyze.py" --root "$OUT" > "$OUT/analyze.log" 2>&1
echo "== analyze exit=$? $(date +%T)"
