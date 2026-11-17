#!/bin/sh
# Run only after the committed rule, runtime and fresh private admission are checked.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export G6_PROBE_EXTERNAL_TIMEOUT_S=2400
export SOUP_G6_INSTANCE_ID=56900314
export CUDA_VISIBLE_DEVICES=GPU-259fccc1-4cda-8c67-d13d-e08932415283
exec timeout -s TERM -k 10s 2400s python "$SCRIPT_DIR/paired_generation_probe.py" "$@"
