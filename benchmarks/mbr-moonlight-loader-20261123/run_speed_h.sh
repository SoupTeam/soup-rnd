#!/usr/bin/env bash
# Ticket 05 part 2: arm H (cpp_async_2slot) against arm G (async_lease), normal mode.
#
# mbr_measure loader-speed: 5 blocks of G-H-H-G on the three windows that fit 4 GiB,
# (8,64), (8,512) and (64,64), every run in a fresh process inside ArmWatch with the
# suspend, power and foreign_reads checks required (speed-rule.md). The paired gain per
# block is median(H) over median(G); the local gate wants >=3% and t > 2.78 in at least
# two of the three windows.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default this campaign
# folder: the raw rows and arm records stay inside the repository), NATIVE_EXT (default
# /tmp/opencode/native-ext).
# Refuses to run over existing rows.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-loader-20261123}"
NATIVE_EXT="${NATIVE_EXT:-/tmp/opencode/native-ext}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"

if [ -e "$OUT/speed/rows.jsonl" ]; then
  echo "refusing to run: $OUT/speed/rows.jsonl exists. Move it away first." >&2
  exit 1
fi
mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$NATIVE_EXT:$HOME/moonlight-local/native-ext:$REPO/src:$REPO"

# The host must not suspend mid-block: ArmWatch would void the arm, and the ticket asks for the
# run to survive (systemd-inhibit sleep for the whole campaign, replacements included).
INHIBIT=(systemd-inhibit --what=sleep --why="ticket 05 loader speed blocks")
command -v systemd-inhibit > /dev/null || INHIBIT=()

echo "== loader-speed G-H-H-G, 5 blocks, start $(date +%T)"
"${INHIBIT[@]}" "$VENV/bin/python" -u "$HARNESS" loader-speed --blocks 5 \
  --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
  --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" --out "$OUT/speed" \
  > "$OUT/speed.log" 2>&1
echo "== loader-speed exit=$? $(date +%T)"

echo "== void replacements start $(date +%T)"
if [ -e "$OUT/speed/rows.jsonl" ]; then
  "${INHIBIT[@]}" "$VENV/bin/python" -u "$REPO/benchmarks/mbr-moonlight-loader-20261123/replace_void.py" \
    --out "$OUT/speed" --blocks 5 \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --lora-targets "$TARGETS" --label "RTX 3050 Laptop, real" \
    > "$OUT/replace.log" 2>&1
  echo "== void replacements exit=$? $(date +%T)"
else
  echo "== void replacements skipped: no speed rows $(date +%T)" | tee "$OUT/replace.log"
fi
