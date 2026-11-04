#!/bin/sh
set -eu
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /data/c3/env/bin/python -m benchmarks.c3.isolated_arm --output /data/c3/evidence/comparison/arm_12_A --mode cuda-representative --arm A --prepared /data/c3/repo/.c3_prepared/prepared.json
