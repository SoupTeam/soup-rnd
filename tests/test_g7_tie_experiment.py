"""Tests for the G7 tie_weights experiment (benchmarks/harness/tie_experiment.py, ticket 06).

The gate comparison uses small CPU tensors. The block loop runs on fake runners. No test
starts a worker or reads a sensor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def tie():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_tie_experiment_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "tie_experiment.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def make_pass(loss=1.5, scale=1.0):
    import torch

    return {
        "loss": torch.tensor(loss),
        "grads": {
            "layers.0.q_proj.lora_A": torch.tensor([1.0, 2.0]) * scale,
            "layers.0.q_proj.lora_B": torch.tensor([0.5, -0.5]) * scale,
        },
    }


# --- the gate comparison ---------------------------------------------------------------


def test_gate_passes_when_everything_is_equal_and_the_control_differs(tie):
    result = tie.compare_passes(make_pass(), make_pass(), make_pass(loss=1.6, scale=2.0))

    assert result["passed"] is True
    assert result["loss_equal"] is True
    assert result["grads_equal"] is True
    assert result["control_differs"] is True
    assert result["grads_compared"] == 2


def test_gate_fails_on_a_different_loss(tie):
    result = tie.compare_passes(make_pass(), make_pass(loss=1.5000001), make_pass(loss=2.0))

    assert result["passed"] is False
    assert result["loss_equal"] is False
    assert "loss" in result["reason"]


def test_gate_fails_on_one_different_gradient_and_names_it(tie):
    changed = make_pass()
    changed["grads"]["layers.0.q_proj.lora_B"][0] += 1e-7

    result = tie.compare_passes(make_pass(), changed, make_pass(scale=2.0))

    assert result["passed"] is False
    assert result["grads_equal"] is False
    assert "layers.0.q_proj.lora_B" in result["reason"]


def test_gate_fails_when_the_gradient_names_differ(tie):
    other = make_pass()
    del other["grads"]["layers.0.q_proj.lora_A"]

    result = tie.compare_passes(make_pass(), other, make_pass(scale=2.0))

    assert result["passed"] is False
    assert "lora_A" in result["reason"]


def test_gate_fails_when_the_control_does_not_differ(tie):
    result = tie.compare_passes(make_pass(), make_pass(), make_pass())

    assert result["passed"] is False
    assert result["control_differs"] is False
    assert "control" in result["reason"]


def test_gate_fails_when_a_gradient_is_all_zero(tie):
    zero = make_pass()
    zero["grads"]["layers.0.q_proj.lora_B"] = zero["grads"]["layers.0.q_proj.lora_B"] * 0

    result = tie.compare_passes(zero, zero, make_pass(scale=2.0))

    assert result["passed"] is False
    assert "zero" in result["reason"]
    assert "layers.0.q_proj.lora_B" in result["reason"]


def test_gate_fails_with_no_gradients(tie):
    import torch

    empty = {"loss": torch.tensor(1.0), "grads": {}}
    result = tie.compare_passes(empty, empty, make_pass())

    assert result["passed"] is False


# --- the patch -------------------------------------------------------------------------


def test_untied_wrapper_calls_functional_call_with_tie_weights_false(tie):
    seen = {}

    def original(module, weights, args=None, kwargs=None, *, tie_weights=True, strict=False):
        seen.update(module=module, tie_weights=tie_weights, args=args, kwargs=kwargs)
        return "out"

    wrapped = tie.untied_wrapper(original)

    assert wrapped("layer", {"w": 1}, (1, 2), {"k": 3}) == "out"
    assert seen == {"module": "layer", "tie_weights": False, "args": (1, 2), "kwargs": {"k": 3}}


# --- the block loop --------------------------------------------------------------------


class FakeRuns:
    """Returns the next synthetic seconds for each run; run numbers in ``void`` come out void."""

    def __init__(self, values, void=()):
        self.values = list(values)
        self.void = set(void)
        self.calls = []

    def __call__(self, *, variant, round, run):
        self.calls.append((variant, round, run))
        outcome = "void" if run in self.void else "ok"
        return {"value": self.values.pop(0), "outcome": outcome, "record": {"outcome": outcome}}


def flat_blocks(count, a=1.0, b=0.95):
    return [value for _ in range(count) for value in (a, b, b, a)]


def test_blocks_run_a_b_b_a_with_a_shipped_and_b_untied(tie):
    runs = FakeRuns(flat_blocks(5))

    blocks = tie.run_blocks(runs)

    assert len(blocks) == 5
    variants = [variant for variant, _, _ in runs.calls[:4]]
    assert variants == ["shipped", "untied", "untied", "shipped"]
    assert [run["arm"] for run in blocks[0]] == ["A", "B", "B", "A"]
    assert [run for _, _, run in runs.calls] == list(range(1, 21))
    assert [round for _, round, _ in runs.calls[:8]] == [1, 1, 2, 2, 3, 3, 4, 4]


def test_a_block_with_a_void_run_is_kept_and_replaced(tie):
    runs = FakeRuns(flat_blocks(6), void={6})

    blocks = tie.run_blocks(runs)

    assert len(blocks) == 6
    assert any(run["outcome"] == "void" for run in blocks[1])


def test_blocks_go_past_the_pilot_when_the_spread_asks_for_more(tie):
    # A runs of 0.85 and 1.15 s give sigma 0.16 against a 0.1 s effect, so N is above 5.
    noisy = [0.85, 1.0, 1.0, 1.15] * 5
    runs = FakeRuns(noisy + [1.0, 1.0, 1.0, 1.0] * 40)

    blocks = tie.run_blocks(runs)

    assert len(blocks) > 5
    assert len(runs.calls) == 4 * len(blocks)


def test_blocks_give_up_after_max_blocks(tie):
    runs = FakeRuns(flat_blocks(6), void=set(range(1, 25)))

    with pytest.raises(RuntimeError, match="void"):
        tie.run_blocks(runs, max_blocks=3)


# --- the whole experiment --------------------------------------------------------------


def test_the_speed_comparison_does_not_run_when_the_gate_fails(tie):
    runs = FakeRuns(flat_blocks(5))
    gate = {"passed": False, "reason": "loss differs"}

    result = tie.run_experiment(lambda: gate, runs)

    assert runs.calls == []
    assert result["blocks"] == []
    assert result["gate"] == gate
    assert result["verdict"]["verdict"] == "no_verdict"
    assert "loss differs" in result["verdict"]["reason"]


def test_the_speed_comparison_runs_after_the_gate_passes_and_gets_a_verdict(tie):
    runs = FakeRuns([v + 0.001 * i for i, v in enumerate(flat_blocks(5))])
    gate = {"passed": True, "reason": "equal"}

    result = tie.run_experiment(lambda: gate, runs)

    assert len(runs.calls) == 20
    assert len(result["blocks"]) == 5
    assert result["verdict"]["verdict"] in {"pass", "fail"}


def test_run_one_starts_a_bare_arm_at_16_tokens_with_the_variant_and_a_run_tag(tie, tmp_path):
    seen = {}

    def run_arm_process(**kwargs):
        seen.update(kwargs)
        return {"run": {"median_step_s": 0.42}, "arm": {"outcome": "ok"}}

    run_one = tie.arm_runner(
        fixture="fx", root=tmp_path, out_dir=tmp_path / "out", run_arm_process=run_arm_process
    )
    result = run_one(variant="untied", round=2, run=7)

    assert seen["arm"] == "bare"
    assert seen["seq"] == 16
    assert seen["variant"] == "untied"
    assert seen["tag"] == "run007"
    assert (seen["round"], seen["run"]) == (2, 7)
    assert result == {"value": 0.42, "outcome": "ok", "record": {"outcome": "ok"}}
