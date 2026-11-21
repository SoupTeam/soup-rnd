#!/usr/bin/env bash
# Ticket 02: which MBR windows fit Moonlight on the 4 GiB laptop card. Runs one window at a time.
set -u
R=/home/m1kadoo/Documents/soup-rnd
V=$HOME/.venvs/soup-rnd-cuda
L=$HOME/moonlight-local
OUT=$L/results/fit
mkdir -p "$OUT"
export CUDA_HOME=$($V/bin/python -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH=$V/bin:$PATH
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"

run() {
  local arm=$1 k=$2 seq=$3
  echo "== arm=$arm k=$k seq=$seq start $(date +%T)"
  PYTHONPATH=$R/src:$R $V/bin/python -u $R/benchmarks/harness/mbr_measure.py run \
    --arm "$arm" --k "$k" --seq "$seq" --kind speed \
    --source "$L/source" --cache "$L/cache" --out "$OUT" --lora-targets "$TARGETS" \
    >> "$OUT/fit.jsonl" 2>> "$OUT/fit.err"
  echo "== arm=$arm k=$k seq=$seq exit=$? $(date +%T)"
}

run C 8 64
run A0 8 64
run C 8 512
run A0 8 512
run C 64 64
echo "all done $(date +%T)"
