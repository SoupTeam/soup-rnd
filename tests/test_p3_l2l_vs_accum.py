"""P3 B1.4 ticket 02: L2L against plain gradient accumulation at k = 1, 4, 16, 64.

Everything runs on the tiny native DeepSeek-V3 of B1.3 (3 decoder layers, per-expert shards,
LoRA on the MLA projections). The trainer under test is the real ``Trainer`` with and without
``make_l2l_trainer_class``; the reference is an independent resident eager model.

Already covered by B1.3 (``tests/test_b13_moe_trainer.py``), not repeated here:
  * k = 1, 2, 4 only; equal and uneven lengths; FP32 and BF16; three updates with a short
    last window; masked labels; a row that is right-padded inside the batch;
  * B1.3's own ``allclose`` limits: 0/0 (FP32, equal lengths), 1e-6/1e-5 (BF16, equal
    lengths), 2e-5/1e-4 (uneven lengths);
  * the boundary adapters (embedding, lm_head), dropout, the outer RNG state, the logged
    load/fetch counters.

Added by this file:
  1. k = 16 and k = 64, so a window holds 16 or 64 micro-batches (B1.3 stops at k = 4);
  2. the A3 verdict (``decide`` of the A3-04 stand, limits of A3-07): GA repeats bit for bit
     (A against A), the gradients are not all zero, then per update the relative L2 error of
     the gradients and the relative error of the loss are judged. FP32 limit: 1e-4. BF16 limit:
     1.5 x the noise floor. Both numbers are the stand's, none is new;
  3. a stricter claim on top of the A3 limit: gradients of every update, the summed loss of
     every update and the parameters after the last update are bit for bit equal, in FP32 and
     in BF16, with equal and with uneven lengths (B1.3 allowed a gap for BF16 and for uneven);
  4. the update count equals ceil(rows / k) for GA and for L2L at every k.

Noise floor of BF16. A3 measures it as GA on a second SDPA backend against GA on MATH. On CPU
this model has no second backend: FLASH_ATTENTION raises "No available kernel" and
EFFICIENT_ATTENTION is CUDA only. The floor used here is therefore the repeat of GA (A against
A), which is exactly 0, so 1.5 x the floor is 0 and BF16 has to match exactly. That is stricter
than A3, not looser. The real floor needs a GPU run.

Logged loss. Observed, cause not isolated: at k >= 16 the loss in the Trainer's log differs
between GA and L2L by one float32 ulp (about 1.1e-7 relative) in both dtypes, although the loss
of each update (the sum of the micro-batch losses, read from ``training_step``) is bit for bit
equal. In the L2L log the value equals that sum. The A3 loss check therefore runs on the exact
sum, and the logged loss is only held to the FP32 limit of A3 (1e-4) in both dtypes.

Because the BF16 floor is 0 here, ``status == "within_tolerance"`` for BF16 is the same claim as
the exact-equality asserts below it. The A3 tolerance itself is not exercised on CPU.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from tests.test_b13_moe_runtime import build_moe_streamed, moe_resident_reference
from tests.test_b13_moe_trainer import _train, _values
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_l2l_probe_harness import _load
from tests.test_l2l_trainer_labels import _MaskedRows

pytestmark = _REQUIRES_TRAINING_FLOOR

K_SWEEP = (1, 4, 16, 64)
WINDOWS = 3  # two full updates and one update of a single row


def _row_count(k: int) -> int:
    return 2 * k + 1


@pytest.fixture(scope="module")
def stand() -> Any:
    _load("l2l_schedule")
    return _load("l2l_mistral_stand")


def _rows(k: int, *, uneven: bool) -> Any:
    """Two full windows and a final window of one row; every row masks a label span."""
    rows = _row_count(k)
    lengths = ([5, 8, 3, 6, 9] * rows)[:rows] if uneven else [6] * rows
    return _MaskedRows(lengths, [[(0, 1 + index % 2)] for index in range(rows)])


def _record(stand: Any, run: dict[str, Any], *, logged_loss: bool = False) -> Any:
    """A run as the A3 stand reads it: the exact summed loss of each update, or the logged one."""
    logs = run["logs"] if logged_loss else [
        {"loss": float(loss), "learning_rate": row["learning_rate"]}
        for loss, row in zip(run["losses"], run["logs"])
    ]
    return stand.RunRecord(logs=logs, rates=run["rates"], grads=run["grads"], steps=run["steps"])


def _state_gap(want: dict[str, Any], got: dict[str, Any]) -> float:
    assert want.keys() == got.keys()
    return max(float((want[name] - got[name]).abs().max()) for name in want)


def _runs(tmp_path: Path, *, k: int, uneven: bool, bf16: bool) -> Any:
    """GA twice on independent resident models, then L2L; all from the same start."""
    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(tmp_path, config)
    moe = build_moe_streamed(weights, tmp_path / "shards")
    try:
        data = _rows(k, uneven=uneven)
        before = _values(moe.model)
        runs = []
        for name in ("ga1", "ga2"):
            resident = moe_resident_reference(weights, moe.lora_config, lora_from=moe.model)
            runs.append(_train(resident, None, tmp_path / name, data, k=k, l2l=False, bf16=bf16))
        l2l = _train(moe.model, moe.runtime, tmp_path / "l2l", data, k=k, l2l=True, bf16=bf16)
        return runs[0], runs[1], l2l, before
    finally:
        moe.close()


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("uneven", [False, True], ids=["equal-lengths", "uneven-lengths"])
@pytest.mark.parametrize("k", K_SWEEP)
def test_l2l_matches_accumulation(
    tmp_path: Path, record_property: Callable[..., None], stand: Any,
    k: int, uneven: bool, bf16: bool,
) -> None:
    ga1, ga2, l2l, before = _runs(tmp_path, k=k, uneven=uneven, bf16=bf16)
    assert ga1["steps"] == l2l["steps"] == l2l["l2l_steps"] == WINDOWS == math.ceil(
        _row_count(k) / k
    )
    assert l2l["window"] is None
    ga, ga_again, streamed = (_record(stand, run) for run in (ga1, ga2, l2l))
    verdict = stand.decide(
        ga, ga_again, streamed, mode="bf16" if bf16 else "fp32",
        floor=ga_again if bf16 else None,
    )
    tolerance = verdict["tolerance"]
    record_property("status", verdict["status"])
    record_property("max_grad_rel_l2", max(tolerance["grad_rel"]))
    record_property("max_loss_rel", max(tolerance["loss_rel"]))
    assert verdict["status"] == "within_tolerance", verdict["note"]
    assert len(tolerance["grad_rel"]) == len(tolerance["loss_rel"]) == WINDOWS
    # Bit for bit: gradients and summed loss of every update, parameters after the last one.
    assert verdict["l2l_vs_ga"]["grads_equal"] is True, verdict["l2l_vs_ga"]["worst"]
    assert verdict["l2l_vs_ga"]["losses_equal"] is True
    assert _state_gap(ga1["final"], l2l["final"]) == 0.0
    # The loss in the trainer's log: A3's FP32 limit, see "Logged loss" above.
    logged = stand.loss_rel_by_step(
        _record(stand, ga1, logged_loss=True), _record(stand, l2l, logged_loss=True),
    )
    record_property("max_logged_loss_rel", max(logged))
    assert len(logged) == WINDOWS
    assert all(error <= stand.FP32_REL_LIMIT for error in logged), logged
    assert len(before) == 30  # five MLA projections x A/B x three decoder layers
    assert any(not bool((before[name] == l2l["final"][name]).all()) for name in before)
