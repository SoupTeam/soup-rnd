# ruff: noqa
# One-off ticket A3-05 diagnostic, run on the Vast instance; kept as evidence, not linted.
"""Diagnostic attempt 02 (not part of the shipped stand): bitsandbytes' own 4-bit kernel is
switched off for BOTH arms so every gemm_4bit goes dequantize + linear. Everything else is the
stand unchanged."""
import sys
sys.path.insert(0, "/root/soup/benchmarks/harness")
import l2l_mistral_stand as stand

stand.set_deterministic_env()
from bitsandbytes.backends.cuda import ops

ops._gemm_4bit_use_custom_fn = lambda *a, **k: False
print("DIAG: custom gemm_4bit kernel disabled in both arms")
sys.exit(stand.main(sys.argv[1:]))
