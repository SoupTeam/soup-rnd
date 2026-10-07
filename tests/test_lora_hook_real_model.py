"""CPU checks for Part B's rule in benchmarks/harness/lora_hook_real_model.py.

On the real model both sides run bf16, so the rule lets the adapter's effect
differ by as much as the base models disagree. These tests plant engine outputs
with a known deviation and a known effect and check the verdicts.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "lora_hook_real_model.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("_lora_hook_real_model", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _case(noise_scale, effect_scale=0.1, engine_effect=1.0, seed=0):
    """The engine's deviation from the reference is deterministic: a fixed base gap, plus a
    further deviation of the same size once the adapter is applied. A dropped adapter computes
    the base again, so its logits equal the engine's base logits exactly."""
    rng = np.random.default_rng(seed)
    ref0 = rng.normal(size=(16, 64))
    effect = effect_scale * rng.normal(size=ref0.shape)
    eng0 = ref0 + noise_scale * rng.normal(size=ref0.shape)
    if engine_effect == 0.0:
        return ref0, ref0 + effect, eng0, eng0.copy()
    eng1 = eng0 + engine_effect * effect + noise_scale * rng.normal(size=ref0.shape)
    return ref0, ref0 + effect, eng0, eng1


def _verdict(harness, ref0, ref1, eng0, eng1):
    metrics = harness.parity.effect_metrics(ref0, ref1, eng0, eng1)
    return harness.verdict_b(
        base_ok=True,
        s_ref=harness.parity.effect_size(ref0, ref1),
        convert_ok=True,
        load_ok=True,
        r=metrics["r"],
        rho=metrics["rho"],
        floor=harness.noise_floor(eng0, ref0, ref1),
    )


def test_noise_floor_is_base_gap_over_effect(harness):
    ref0 = np.zeros((2, 3))
    ref1 = ref0 + 2.0
    eng0 = ref0 + 0.5
    assert harness.noise_floor(eng0, ref0, ref1) == pytest.approx(0.25)


def test_effect_reproduced_under_base_level_noise_is_applied(harness):
    assert _verdict(harness, *_case(noise_scale=0.01)) == "APPLIED"


def test_dropped_adapter_is_dropped(harness):
    assert _verdict(harness, *_case(noise_scale=0.01, engine_effect=0.0)) == "DROPPED"


def test_half_applied_adapter_is_wrong(harness):
    assert _verdict(harness, *_case(noise_scale=0.001, engine_effect=0.5)) == "WRONG"


def test_base_gap_too_large_for_the_effect_gives_no_verdict(harness):
    """Otherwise the APPLIED band would admit even a dropped adapter (r = 1)."""
    assert _verdict(harness, *_case(noise_scale=0.05, engine_effect=0.0)) == "TOO NOISY"


def test_base_disagreement_voids_before_anything_else(harness):
    verdict = harness.verdict_b(base_ok=False, s_ref=0.5, convert_ok=False, load_ok=None)
    assert verdict == "VOID"
