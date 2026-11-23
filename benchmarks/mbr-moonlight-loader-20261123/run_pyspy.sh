#!/usr/bin/env bash
# Ticket 05, part 2: py-spy snapshot and the Python-controller share of a step.
#
# Profiles one normal-mode (8,64) run of each hand, in-process through profile_step.py under
# ArmWatch: H (cpp_async_2slot) is the hand under test, G (async_lease) is the contrast. Native
# frames are collected so a sample blocked inside the compiled stream is not counted as Python.
# The speedscope files feed analyze_pyspy.py; the flamegraph SVG is the human-readable snapshot.
#
# The ticket only asks for the cpp_async_2slot share; G costs one more run and shows where the
# pool's time went before. Inputs as in run_speed.sh; the profile and its rows stay inside this
# campaign folder in the repository; refuses to overwrite an existing profile.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
NATIVE_EXT="${NATIVE_EXT:-/tmp/opencode/native-ext}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-loader-20261123}"
PROFILE_OUT="$OUT/pyspy"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO:$NATIVE_EXT:$HOME/moonlight-local/native-ext"
mkdir -p "$PROFILE_OUT"

profile() {
  local name=$1 arm=$2 format=$3
  if [ -e "$PROFILE_OUT/$name" ]; then
    echo "refusing to run: $PROFILE_OUT/$name exists. Move it away first." >&2
    return 1
  fi
  echo "== py-spy $name arm=$arm $(date +%T)"
  "$VENV/bin/py-spy" record --format "$format" --rate 100 --native --threads \
    -o "$PROFILE_OUT/$name" -- \
    "$VENV/bin/python" -u "$REPO/benchmarks/mbr-moonlight-loader-20261123/profile_step.py" \
      --arm "$arm" --k 8 --seq 64 --steps 12 \
      --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
      --lora-targets q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj --out "$PROFILE_OUT/rows" \
    > "$PROFILE_OUT/$name.log" 2>&1
  echo "== py-spy $name exit=$? $(date +%T)"
}

profile H-k8-s64.speedscope.json H speedscope
profile H-k8-s64.svg H flamegraph
profile G-k8-s64.speedscope.json G speedscope

"$VENV/bin/python" "$REPO/benchmarks/mbr-moonlight-loader-20261123/analyze_pyspy.py" \
  --profile "$PROFILE_OUT/H-k8-s64.speedscope.json" \
  --profile "$PROFILE_OUT/G-k8-s64.speedscope.json" \
  --out "$PROFILE_OUT/controller-share.json"
echo "all done $(date +%T)"
