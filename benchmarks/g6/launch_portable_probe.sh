#!/bin/sh
# Private configuration and stage admission must be approved before this launcher.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${SOUP_G6_INSTANCE_ID:?Set the approved instance ID outside the repository}"
: "${CUDA_VISIBLE_DEVICES:?Set the approved GPU UUID outside the repository}"
export SOUP_G6_INSTANCE_ID CUDA_VISIBLE_DEVICES
export G6_PROBE_EXTERNAL_TIMEOUT_S=2400
exec timeout -s TERM -k 10s 2400s python "$SCRIPT_DIR/portable_generation_probe.py" "$@"
