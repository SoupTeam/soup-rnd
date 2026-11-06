"""CPU tests for the A6 ticket-06 CUDA check driver (benchmarks/harness/l2l_pref_check.py).

The GPU run cannot be tested here. These pin the rule of benchmarks/l2l-preference-check-rule-a6.md:
how a cell (method, k) is judged, how the layer loads of a method are compared with SFT, and
how a method's verdict comes from its three cells and the validity of their arms.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def check():
    return _load("l2l_pref_check")


def _arm(scale=1.0, steps=2):
    """A recorded arm: LoRA gradients of every optimizer step and the logged losses."""
    import torch

    grads = [
        {
            "base.default.lora_A": torch.tensor([1.0, 2.0, 3.0]) * scale * (i + 1),
            "base.default.lora_B": torch.tensor([0.5, -0.25]) * scale * (i + 1),
        }
        for i in range(steps)
    ]
    return {"grads": grads, "losses": [0.7 * scale * (i + 1) for i in range(steps)]}


def _nudge(arm, name="base.default.lora_A", step=0, by=1e-3, loss_by=0.0):
    import copy

    out = copy.deepcopy(arm)
    out["grads"][step][name] = out["grads"][step][name] + by
    out["losses"][step] += loss_by
    return out


# --------------------------------------------------------------------------
# judge_cell: bitwise first, A3-07 tolerance only when A against A is noisy
# --------------------------------------------------------------------------
def test_a_cell_equal_bit_for_bit_passes_tier_one(check):
    verdict = check.judge_cell(_arm(), _arm(), _arm(), windows=2)
    assert verdict["pass"] is True
    assert verdict["tier"] == 1
    assert verdict["a_vs_a_equal"] is True
    assert verdict["problems"] == []


def test_a_difference_that_ga_does_not_show_against_itself_fails_the_cell(check):
    verdict = check.judge_cell(_arm(), _arm(), _nudge(_arm()), windows=2)
    assert verdict["pass"] is False
    assert verdict["tier"] is None
    assert verdict["a_vs_a_equal"] is True
    assert any(
        "base.default.lora_A" in problem and "step 0" in problem for problem in verdict["problems"]
    )


def test_a_loss_that_differs_fails_the_cell_even_when_the_gradients_are_equal(check):
    verdict = check.judge_cell(_arm(), _arm(), _nudge(_arm(), by=0.0, loss_by=1e-4), windows=2)
    assert verdict["pass"] is False
    assert any("loss" in problem for problem in verdict["problems"])


def test_when_ga_is_noisy_against_itself_a_difference_inside_one_and_a_half_floors_passes_tier_two(
    check,
):
    # the floor: 1e-3 on lora_A at step 0, 1e-4 on the loss
    ga2 = _nudge(_arm(), by=1e-3, loss_by=1e-4)
    l2l = _nudge(_arm(), by=1.4e-3, loss_by=1.4e-4)
    verdict = check.judge_cell(_arm(), ga2, l2l, windows=2)
    assert verdict["a_vs_a_equal"] is False
    assert verdict["pass"] is True
    assert verdict["tier"] == 2
    assert verdict["problems"] == []


def test_a_difference_beyond_one_and_a_half_floors_fails_tier_two(check):
    ga2 = _nudge(_arm(), by=1e-3)
    l2l = _nudge(_arm(), by=1.6e-3)
    verdict = check.judge_cell(_arm(), ga2, l2l, windows=2)
    assert verdict["pass"] is False
    assert verdict["tier"] is None
    assert any("base.default.lora_A" in problem for problem in verdict["problems"])


def test_a_tensor_with_a_zero_floor_must_stay_bit_equal_under_tier_two(check):
    ga2 = _nudge(_arm(), name="base.default.lora_A", by=1e-3)  # lora_B has no noise
    l2l = _nudge(_arm(), name="base.default.lora_B", by=1e-4)
    verdict = check.judge_cell(_arm(), ga2, l2l, windows=2)
    assert verdict["pass"] is False
    assert any("base.default.lora_B" in problem for problem in verdict["problems"])


def test_a_loss_with_a_zero_floor_must_stay_equal_under_tier_two(check):
    ga2 = _nudge(_arm(), by=1e-3)  # gradient noise, loss identical
    l2l = _nudge(_arm(), by=0.0, loss_by=1e-6)
    verdict = check.judge_cell(_arm(), ga2, l2l, windows=2)
    assert verdict["pass"] is False
    assert any("loss" in problem for problem in verdict["problems"])


# --------------------------------------------------------------------------
# judge_cell: the equal arms have to be worth comparing
# --------------------------------------------------------------------------
def test_equal_arms_with_the_wrong_number_of_steps_fail(check):
    verdict = check.judge_cell(_arm(steps=2), _arm(steps=2), _arm(steps=2), windows=3)
    assert verdict["pass"] is False
    assert any("3 windows" in problem for problem in verdict["problems"])


def test_equal_arms_with_all_zero_gradients_fail(check):
    verdict = check.judge_cell(_arm(0.0), _arm(0.0), _arm(0.0), windows=2)
    assert verdict["pass"] is False
    assert any("zero" in problem for problem in verdict["problems"])


def test_equal_arms_with_a_non_finite_gradient_or_loss_fail(check):
    bad = _arm()
    bad["grads"][1]["base.default.lora_B"][0] = float("nan")
    verdict = check.judge_cell(bad, bad, bad, windows=2)
    assert verdict["pass"] is False
    assert any("finite" in problem for problem in verdict["problems"])

    bad_loss = _arm()
    bad_loss["losses"][0] = float("inf")
    verdict = check.judge_cell(bad_loss, bad_loss, bad_loss, windows=2)
    assert verdict["pass"] is False
    assert any("finite" in problem for problem in verdict["problems"])


def test_a_gradient_on_the_reference_adapter_fails_the_cell(check):
    import torch

    arm = _arm()
    arm["grads"][0]["base.ref.lora_A"] = torch.tensor([1.0])
    verdict = check.judge_cell(arm, arm, arm, windows=2)
    assert verdict["pass"] is False
    assert any(".ref" in problem for problem in verdict["problems"])


# --------------------------------------------------------------------------
# judge_loads: the method reads the layers no more often than L2L SFT with the same k
# --------------------------------------------------------------------------
def _logs(per_step, large=0, restores=0, captures=0):
    """Training-log rows with the running totals the trainer writes, from per-step loads."""
    rows, total = [], 0
    for index, loads in enumerate(per_step, start=1):
        total += loads
        rows.append(
            {
                "loss": 1.0,
                "l2l_steps": index,
                "l2l_layer_loads": total,
                "l2l_large_loads": large * index,
                "l2l_weight_restores": restores * index,
                "l2l_capture_layer_loads": captures,
            }
        )
    return rows


def test_equal_layer_loads_at_every_step_pass(check):
    verdict = check.judge_loads(_logs([6, 6, 6]), _logs([6, 6, 6]))
    assert verdict["pass"] is True
    assert verdict["layer_loads"] == {"method": [6, 6, 6], "sft": [6, 6, 6]}
    assert verdict["problems"] == []


def test_one_more_layer_load_at_one_step_fails(check):
    verdict = check.judge_loads(_logs([6, 7, 6]), _logs([6, 6, 6]))
    assert verdict["pass"] is False
    assert any("step 1" in problem and "layer loads" in problem for problem in verdict["problems"])


def test_fewer_layer_loads_than_sft_also_fails_because_the_rule_says_equal(check):
    verdict = check.judge_loads(_logs([5, 6, 6]), _logs([6, 6, 6]))
    assert verdict["pass"] is False


def test_more_large_loads_or_weight_restores_than_sft_fail_and_fewer_pass(check):
    assert check.judge_loads(_logs([6], large=2), _logs([6], large=1))["pass"] is False
    assert check.judge_loads(_logs([6], restores=3), _logs([6], restores=2))["pass"] is False
    fewer = check.judge_loads(_logs([6], large=1, restores=1), _logs([6], large=2, restores=2))
    assert fewer["pass"]


def test_a_different_number_of_steps_fails(check):
    verdict = check.judge_loads(_logs([6, 6]), _logs([6, 6, 6]))
    assert verdict["pass"] is False
    assert any("steps" in problem for problem in verdict["problems"])


def test_capture_loads_are_reported_but_not_compared(check):
    verdict = check.judge_loads(_logs([6, 6], captures=40), _logs([6, 6], captures=12))
    assert verdict["pass"] is True
    assert verdict["capture_layer_loads"] == {"method": 40, "sft": 12}


# --------------------------------------------------------------------------
# judge_method: three cells, up to three attempts each, validity of the arms
# --------------------------------------------------------------------------
OK_ARMS = {"ga1": "ok", "ga2": "ok", "l2l": "ok", "sft": "ok"}


def _attempt(equal=True, loads=True, arms=None, crashed=None):
    return {
        "arms": dict(OK_ARMS if arms is None else arms),
        "crashed": crashed,
        "equality": {"pass": equal, "tier": 1 if equal else None, "problems": []},
        "loads": {"pass": loads, "problems": []},
    }


def _cell(k, *attempts):
    return {"k": k, "attempts": list(attempts)}


def _good_cells():
    return [_cell(k, _attempt()) for k in (1, 2, 4)]


def test_a_method_with_three_passing_cells_passes(check):
    verdict = check.judge_method(_good_cells())
    assert verdict["verdict"] == "pass"
    assert [cell["verdict"] for cell in verdict["cells"]] == ["pass", "pass", "pass"]


def test_one_failing_cell_fails_the_method(check):
    cells = [_cell(1, _attempt()), _cell(2, _attempt(equal=False)), _cell(4, _attempt())]
    assert check.judge_method(cells)["verdict"] == "fail"


def test_a_cell_that_reads_more_layers_than_sft_fails_the_method(check):
    cells = [_cell(1, _attempt()), _cell(2, _attempt()), _cell(4, _attempt(loads=False))]
    assert check.judge_method(cells)["verdict"] == "fail"


def test_a_void_attempt_is_excluded_and_the_next_valid_attempt_decides(check):
    void = _attempt(arms={**OK_ARMS, "l2l": "void"}, equal=False)
    cells = [_cell(1, void, _attempt()), _cell(2, _attempt()), _cell(4, _attempt())]
    verdict = check.judge_method(cells)
    assert verdict["verdict"] == "pass"
    first = verdict["cells"][0]
    assert first["used_attempt"] == 1
    assert first["excluded"] == [{"attempt": 0, "reason": "arm l2l is void"}]


def test_a_failure_on_a_valid_attempt_is_not_replaced_by_a_later_pass(check):
    cells = [
        _cell(1, _attempt(equal=False), _attempt()),
        _cell(2, _attempt()),
        _cell(4, _attempt()),
    ]
    verdict = check.judge_method(cells)
    assert verdict["verdict"] == "fail"
    assert verdict["cells"][0]["used_attempt"] == 0


def test_a_cell_with_three_void_attempts_gives_no_verdict(check):
    void = _attempt(arms={**OK_ARMS, "ga1": "void"})
    cells = [_cell(1, void, void, void), _cell(2, _attempt()), _cell(4, _attempt())]
    verdict = check.judge_method(cells)
    assert verdict["verdict"] == "no_verdict"
    assert verdict["cells"][0]["verdict"] == "no_verdict"


def test_an_unknown_required_check_gives_no_verdict_and_is_not_a_fail(check):
    unknown = _attempt(arms={**OK_ARMS, "ga2": "unknown"}, equal=False)
    cells = [_cell(1, unknown), _cell(2, _attempt()), _cell(4, _attempt())]
    verdict = check.judge_method(cells)
    assert verdict["verdict"] == "no_verdict"
    assert "unknown" in verdict["cells"][0]["reason"]


def test_a_crashed_attempt_is_excluded_and_a_method_that_never_ran_has_no_verdict(check):
    crashed = _attempt(crashed="CUDA out of memory")
    cells = [_cell(1, crashed, _attempt()), _cell(2, _attempt()), _cell(4, _attempt())]
    assert check.judge_method(cells)["verdict"] == "pass"
    never = [_cell(1, crashed, crashed, crashed), _cell(2), _cell(4)]
    assert check.judge_method(never)["verdict"] == "no_verdict"


def test_a_fail_stands_even_when_another_cell_has_no_verdict(check):
    void = _attempt(arms={**OK_ARMS, "ga1": "void"})
    cells = [_cell(1, void, void, void), _cell(2, _attempt(equal=False)), _cell(4, _attempt())]
    assert check.judge_method(cells)["verdict"] == "fail"


def test_a_method_without_all_three_cells_gives_no_verdict(check):
    verdict = check.judge_method([_cell(1, _attempt()), _cell(2, _attempt())])
    assert verdict["verdict"] == "no_verdict"
    assert "k=4" in verdict["reason"]


# --------------------------------------------------------------------------
# judge_cell: arms that log fewer losses than steps, or train another adapter
# --------------------------------------------------------------------------
def test_an_arm_with_fewer_losses_than_steps_fails_even_when_the_rest_is_equal(check):
    short = _arm()
    short["losses"] = short["losses"][:1]
    verdict = check.judge_cell(short, short, short, windows=2)
    assert verdict["pass"] is False
    assert any("losses" in problem for problem in verdict["problems"])


def test_l2l_logging_fewer_losses_than_the_reference_fails(check):
    short = _arm()
    short["losses"] = short["losses"][:1]
    verdict = check.judge_cell(_arm(), _arm(), short, windows=2)
    assert verdict["pass"] is False


def test_a_gradient_outside_the_policy_adapter_fails_the_cell(check):
    import torch

    arm = _arm()
    arm["grads"][0]["base.other.lora_A"] = torch.tensor([1.0])
    verdict = check.judge_cell(arm, arm, arm, windows=2)
    assert verdict["pass"] is False
    assert any("policy adapter" in problem for problem in verdict["problems"])
