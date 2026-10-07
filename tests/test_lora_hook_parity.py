"""CPU checks for the decision rule of benchmarks/harness/lora_hook_parity.py.

The harness turns logits into verdicts (APPLIED, DROPPED, WRONG, ...). These
tests plant engine outputs with a known relation to the reference and check
that the rule in probe-lora-hook-engines.md §2 names them correctly.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "lora_hook_parity.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("_lora_hook_parity", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture()
def logits():
    """Reference logits without and with an adapter that moves them by ~10%."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(12, 50)).astype(np.float32)
    effect = 0.1 * rng.normal(size=base.shape).astype(np.float32)
    return base, base + effect


def _verdict_for(harness, ref0, ref1, eng0, eng1):
    metrics = harness.effect_metrics(ref0, ref1, eng0, eng1)
    return harness.verdict(
        base_ok=harness.base_error(eng0, ref0) <= harness.E_BASE_MAX,
        s_ref=harness.effect_size(ref0, ref1),
        convert_ok=True,
        load_ok=True,
        r=metrics["r"],
        rho=metrics["rho"],
    )


def test_engine_reproducing_the_effect_is_applied(harness, logits):
    ref0, ref1 = logits
    assert _verdict_for(harness, ref0, ref1, ref0.copy(), ref1.copy()) == "APPLIED"


def test_engine_ignoring_the_adapter_is_dropped(harness, logits):
    ref0, ref1 = logits
    assert _verdict_for(harness, ref0, ref1, ref0.copy(), ref0.copy()) == "DROPPED"


def test_engine_scaling_the_adapter_wrongly_is_wrong(harness, logits):
    ref0, ref1 = logits
    eng1 = ref0 + 1.1 * (ref1 - ref0)
    metrics = harness.effect_metrics(ref0, ref1, ref0, eng1)
    assert metrics["r"] == pytest.approx(0.1, rel=1e-4)
    assert _verdict_for(harness, ref0, ref1, ref0, eng1) == "WRONG"


def test_partially_applied_adapter_is_wrong_not_dropped(harness, logits):
    ref0, ref1 = logits
    eng1 = ref0 + 0.5 * (ref1 - ref0)
    assert _verdict_for(harness, ref0, ref1, ref0, eng1) == "WRONG"


def test_effect_is_measured_against_the_engines_own_base(harness, logits):
    """A tiny base offset shared by both engine runs must not count against the adapter."""
    ref0, ref1 = logits
    offset = np.float32(1e-5)
    assert _verdict_for(harness, ref0, ref1, ref0 + offset, ref1 + offset) == "APPLIED"


def test_base_mismatch_voids_even_a_perfect_adapter(harness, logits):
    ref0, ref1 = logits
    shift = 2e-3 * np.abs(ref0).max()
    assert _verdict_for(harness, ref0, ref1, ref0 + shift, ref1 + shift) == "VOID"


def test_rule_rows_apply_in_order(harness):
    # A weak adapter is reported as such before any engine outcome.
    assert harness.verdict(base_ok=True, s_ref=1e-3, convert_ok=False, load_ok=None) == "TOO WEAK"
    assert harness.verdict(base_ok=True, s_ref=0.5, convert_ok=False, load_ok=None) == (
        "CONVERT-FAILED"
    )
    assert harness.verdict(base_ok=True, s_ref=0.5, convert_ok=True, load_ok=False) == (
        "LOAD-FAILED"
    )
    # The APPLIED band includes its boundary; DROPPED is checked only when not APPLIED.
    boundary = harness.R_APPLIED_MAX
    assert harness.verdict(
        base_ok=True, s_ref=0.5, convert_ok=True, load_ok=True, r=boundary, rho=1.0
    ) == "APPLIED"
    assert harness.verdict(
        base_ok=True, s_ref=0.5, convert_ok=True, load_ok=True, r=1.0, rho=0.0
    ) == "DROPPED"


def _models(qwen_verdicts, dsv3_verdicts):
    def wrap(verdicts):
        return {"variants": {name: {"verdict": value} for name, value in verdicts.items()}}

    return {"qwen35moe-tiny": wrap(qwen_verdicts), "dsv3-tiny": wrap(dsv3_verdicts)}


DSV3_SINGLES = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "shared")
CONTROL_OK = {"q_proj": "APPLIED", "v_proj": "APPLIED", "shared": "APPLIED"}


def test_failed_positive_control_gives_no_verdict(harness):
    qwen = dict(CONTROL_OK, v_proj="DROPPED")
    dsv3 = {name: "APPLIED" for name in DSV3_SINGLES}
    assert harness.probe_verdict(_models(qwen, dsv3))["verdict"] == "NO VERDICT"


def test_hook_lists_exactly_the_modules_not_applied(harness):
    dsv3 = {name: "APPLIED" for name in DSV3_SINGLES}
    dsv3.update(q_a_proj="DROPPED", kv_b_proj="CONVERT-FAILED")
    outcome = harness.probe_verdict(_models(CONTROL_OK, dsv3))
    assert outcome["verdict"] == "HOOK NEEDED"
    assert outcome["hook_modules"] == {"q_a_proj": "DROPPED", "kv_b_proj": "CONVERT-FAILED"}


def test_all_modules_applied_needs_no_hook(harness):
    dsv3 = {name: "APPLIED" for name in DSV3_SINGLES}
    assert harness.probe_verdict(_models(CONTROL_OK, dsv3))["verdict"] == "NO HOOK NEEDED"
