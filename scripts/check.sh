#!/usr/bin/env bash
# One gate before submit: lint every path given, run tests only for test paths.
#
# Usage:
#   scripts/check.sh src/soup_cli/utils/layer_stream.py tests/test_layer_stream.py
#   scripts/check.sh tests/test_layer_stream.py
#
# Refuses to run the full suite: pass explicit paths. A change without a test
# path is a finding, not a silent pass.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
  echo "usage: scripts/check.sh <path>... (source paths for ruff, test paths for pytest)" >&2
  exit 2
fi

VENV="${CHECK_VENV:-}"
if [ -z "$VENV" ]; then
  for candidate in .venv "$HOME/.venvs/soup-rnd" "$HOME/.venvs/soup-rnd-cuda"; do
    if [ -x "$candidate/bin/ruff" ]; then VENV="$candidate"; break; fi
  done
fi

if [ -n "$VENV" ] && [ -x "$VENV/bin/ruff" ]; then
  RUFF=("$VENV/bin/ruff")
elif command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
else
  echo "check: ruff not found; set CHECK_VENV or install dev deps" >&2
  exit 3
fi

if [ -n "$VENV" ] && [ -x "$VENV/bin/pytest" ]; then
  PYTEST=("$VENV/bin/pytest")
elif command -v pytest >/dev/null 2>&1; then
  PYTEST=(pytest)
else
  PYTEST=(python3 -m pytest)
fi

"${RUFF[@]}" check "$@"

tests=()
for path in "$@"; do
  case "$path" in
    tests/*) tests+=("$path") ;;
  esac
done

if [ "${#tests[@]}" -eq 0 ]; then
  echo "check: no test paths given; ruff passed, tests not run" >&2
  exit 0
fi

"${PYTEST[@]}" -q --no-cov "${tests[@]}"
