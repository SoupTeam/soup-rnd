"""Shared test helpers for the E4 part 2 harness: harness import path and a tiny random OLMoE.

The tiny model has the real model's structure (fused 3-D experts, q/k norms, router) at a
size a CPU test can build in a second. It lives in ``e4_part2_dryrun`` (the dry run builds the
same model); this module re-exports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "benchmarks" / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from e4_part2_dryrun import (  # noqa: E402, F401
    TINY,
    tiny_config,
    tiny_model,
    write_legacy_checkpoint,
)


def stub_pynvml(monkeypatch):
    """`calibrate`/`run`/`drive` refuse without pynvml on a GPU device (nvml_problem). Tests
    that drive those commands past that check on a CPU box stub the module."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))


def stub_setup(monkeypatch):
    """`run`/`drive`/`calibrate` refuse a data root with no pull record (setup_problem). Tests
    that exercise another refusal or identity on a CPU box stub that one check; the setup check
    itself is tested in test_e4_part2_setup_preflight.py."""
    import e4_part2_calibration as cal

    monkeypatch.setattr(cal, "setup_problem", lambda *_a, **_k: None)
