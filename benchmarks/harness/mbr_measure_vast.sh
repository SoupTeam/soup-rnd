#!/usr/bin/env bash
# Remote side of the single Vast run (RTX 4060 Ti 8 GB) of ticket 01. Runs ON the instance.
#
#   env: STAGES="probe bits profile speed"  BLOCKS=5  DEADLINE_HOURS=<hours for the sweeps>
#        VRAM_LIMIT_GIB=8  (allocator limit: the card may have more than the 8 GiB of the ticket)
#        TORCH_SPEC="torch==2.14.*"  (skipped when torch is already 2.14+)
#   in:  /root/soup/{src,benchmarks/harness}   (the tarball from the local launcher)
#   out: /root/mbr/out/*.json, summary.md, /root/mbr/results.tgz, /root/mbr/DONE
#
# A stage that fails is logged and the next one still runs; nothing here destroys the instance.
# Never edit this file while it runs (bash reads it as it goes).
set -u
ROOT=/root/soup
WORK=/root/mbr
OUT=$WORK/out
STAGES=${STAGES:-"probe bits profile speed"}
mkdir -p "$OUT"
exec > >(tee -a "$WORK/remote.log") 2>&1
echo "[$(date -u +%FT%TZ)] start"

# --- environment (lessons of earlier rentals: torchvision breaks transformers, Brotli breaks hf) ---
python3 - <<'PY' || NEED_TORCH=1
import sys, torch
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
print("torch", torch.__version__); sys.exit(0 if (major, minor) >= (2, 14) else 1)
PY
if [ "${NEED_TORCH:-0}" = 1 ]; then
  pip install -q "${TORCH_SPEC:-torch==2.14.*}" && pip uninstall -y torchvision torchaudio
fi
pip install -q "transformers==5.20.2" "peft==0.21.2" huggingface_hub safetensors rich pydantic \
  bitsandbytes pyyaml typer psutil datasets tiktoken "Brotli>=1.2"
pip uninstall -y torchao >/dev/null 2>&1
python3 -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__,torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uptime; lscpu | grep -E "Model name|^CPU\(s\)"; df -hT "$WORK" | tail -1
# O_DIRECT silently falls back on overlayfs: say so before the clock runs.
fs=$(df -T "$WORK" | tail -1 | awk '{print $2}')
[ "$fs" = overlay ] && echo "WARNING: $WORK is on overlay; direct_io may fall back (the reports flag it)"

# --- run ---
export MBR_SRC=$ROOT MBR_WORK=$WORK/data MBR_OUT=$OUT MBR_SKIP_PIP=1 CUDA_VISIBLE_DEVICES=0
export MBR_VRAM_LIMIT_GIB=${VRAM_LIMIT_GIB:-8}
export MBR_STAGES=${STAGES// /,} MBR_BLOCKS=${BLOCKS:-5} TMPDIR=$WORK/data
[ -n "${DEADLINE_HOURS:-}" ] && export MBR_DEADLINE_HOURS=$DEADLINE_HOURS
mkdir -p "$MBR_WORK"
python3 "$ROOT/benchmarks/harness/mbr_measure_kanana_kaggle.py"
echo "exit code $?"

# --- collect (small files only; the model and cache stay on the instance) ---
tar czf "$WORK/results.tgz" -C "$OUT" . "$WORK/remote.log" 2>/dev/null
touch "$WORK/DONE"
echo "[$(date -u +%FT%TZ)] done"
