"""Regression test for the exploratory Small MoE + LoRA PoC."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = (
    ROOT
    / "benchmarks/e2/moe/small_moe_lora_equivalence.py"
)

RESULT = (
    ROOT
    / "benchmarks/e2/moe/small_moe_lora_equivalence_result.json"
)


def test_small_moe_lora_training_equivalence():
    assert SCRIPT.is_file()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    process = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert process.returncode == 0, (
        process.stdout + "\n" + process.stderr
    )

    result = json.loads(RESULT.read_text())

    assert result["steps"] == 6
    assert result["unique_batches"] == 3
    assert result["cache_hits"] == 3
    assert result["cache_misses"] == 3
    assert result["frozen_layer_calls"] == 6
    assert result["skipped_layer_calls"] == 6

    assert result["exact_training_equivalence"] is True

    assert len(result["history"]) == 6

    for row in result["history"]:
        assert row["loss_equal"]
        assert row["gradients_equal"]
        assert row["weights_equal"]

    assert result["production_e2_supported"] is False
