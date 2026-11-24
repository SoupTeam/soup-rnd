#!/usr/bin/env bash
# Ticket 15: why the segmented expert GEMM (arm S) is slower than C on this card, memory-safe.
#
# The whole-step torch.profiler trace does not fit this host at (64,64): the CUDA-only profile of
# the whole step was OOM-killed at about 1.1M kernel events. So the campaign runs in five passes
# over the two windows the ticket names, (8,64) at k=8 and (64,64) at k=64, arms C (native expert
# GEMM) and S (expert_gemm="segmented"):
#
#   1. sweep_rows.py: the isolated per-M kernel cost of the two routed-expert projections, no model;
#   2. profile_step.py --count --gpu-util: the routed-expert F.linear counter of every step and
#      the card's utilisation, no profiler (this is also what sets each cell's step time);
#   3. profile_step.py --trace: one CUDA-only profiled step per arm, (8,64) only;
#   4. py-spy record --native: one snapshot per cell, into <out>/pyspy;
#   5. analyze.py: analysis.json, layout.md and rows.md.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default this campaign
# folder: the raw rows, sweeps and profiles stay inside the repository).
# Refuses to run over existing rows, off AC power, or while another campaign holds the lock.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-segprof-20261124}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
CAMPAIGN="$REPO/benchmarks/mbr-moonlight-segprof-20261124"
ACAD=/sys/class/power_supply/ACAD/online
# "k seq" of each window: (8,64) at k=8 and (64,64) at k=64.
CELLS=("8 64" "64 64")
# Only (8,64) is traced: the whole-step trace of (64,64) does not fit the host.
TRACED_CELLS=("8 64")
# The first step of every run is warm-up; three is the two timed steps the profile budget allows.
STEPS=3

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

echo "== sweep start $(date +%T)"
"$VENV/bin/python" -u "$CAMPAIGN/sweep_rows.py" --out "$OUT" >> "$OUT/sweep.log" 2>&1
echo "== sweep exit=$? $(date +%T)"

echo "== count and gpu utilisation start $(date +%T)"
for cell in "${CELLS[@]}"; do
  read -r k seq <<< "$cell"
  for arm in C S; do
    "${INHIBIT[@]}" "$VENV/bin/python" -u "$CAMPAIGN/profile_step.py" \
      --arm "$arm" --k "$k" --seq "$seq" --steps "$STEPS" --count --gpu-util \
      --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
      --lora-targets "$TARGETS" --out "$OUT" \
      >> "$OUT/runs.log" 2>&1
    echo "== count arm=$arm k=$k seq=$seq exit=$? $(date +%T)"
  done
done

echo "== traced steps start $(date +%T)"
for cell in "${TRACED_CELLS[@]}"; do
  read -r k seq <<< "$cell"
  for arm in C S; do
    "${INHIBIT[@]}" "$VENV/bin/python" -u "$CAMPAIGN/profile_step.py" \
      --arm "$arm" --k "$k" --seq "$seq" --steps "$STEPS" --trace \
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
      --arm "$arm" --k "$k" --seq "$seq" --steps "$STEPS" --pyspy \
      --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
      --lora-targets "$TARGETS" --out "$OUT/pyspy" \
      >> "$OUT/pyspy.log" 2>&1
    echo "== py-spy arm=$arm k=$k seq=$seq exit=$? $(date +%T)"
  done
done

"$VENV/bin/python" -u "$CAMPAIGN/analyze.py" --root "$OUT" > "$OUT/analyze.log" 2>&1
echo "== analyze exit=$? $(date +%T)"
