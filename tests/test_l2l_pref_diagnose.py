"""CPU tests for the A6 ticket-08 diagnosis judges (benchmarks/harness/l2l_pref_diagnose.py).

The GPU run cannot be tested here. These pin the rule
benchmarks/l2l-preference-diagnosis-rule-a6-ticket08.md:
when the float32 control may speak and what it says, how the noise-floor control reads the A3-07
factor, how the two controls map to a row of the outcome table, and which layer is named first.
"""

from __future__ import annotations

import copy

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def diag():
    return _load("l2l_pref_diagnose")


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
    out = copy.deepcopy(arm)
    out["grads"][step][name] = out["grads"][step][name] + by
    out["losses"][step] += loss_by
    return out


def _types(**overrides):
    """The dtype record of one arm when the float32 patch took effect."""
    record = {
        "stream_dtype": "float32",
        "train_bf16": False,
        "train_fp16": False,
        "mixed_precision": "no",
        "base_weight": "torch.float32",
        "lora_weight": "torch.float32",
        "layer_in": "torch.float32",
        "layer_out": "torch.float32",
        "grads": ["torch.float32"],
    }
    record.update(overrides)
    return record


def _all_types(**overrides):
    return {arm: _types(**overrides) for arm in ("f32_ga1", "f32_ga2", "f32_l2l")}


# --------------------------------------------------------------------------
# control 1: float32, bit for bit
# --------------------------------------------------------------------------
def test_float32_arms_equal_bit_for_bit_with_confirmed_types_are_equal(diag):
    verdict = diag.judge_float32(_arm(), _arm(), _arm(), _all_types())
    assert verdict["state"] == "equal"
    assert verdict["problems"] == []


def test_a_gradient_that_differs_in_float32_gives_not_equal_and_names_it(diag):
    verdict = diag.judge_float32(_arm(), _arm(), _nudge(_arm()), _all_types())
    assert verdict["state"] == "not_equal"
    assert any("step 0" in p and "base.default.lora_A" in p for p in verdict["problems"])


def test_a_loss_that_differs_in_float32_gives_not_equal(diag):
    verdict = diag.judge_float32(_arm(), _arm(), _nudge(_arm(), by=0.0, loss_by=1e-7), _all_types())
    assert verdict["state"] == "not_equal"
    assert any("loss" in p for p in verdict["problems"])


@pytest.mark.parametrize(
    "field, value",
    [
        ("base_weight", "torch.bfloat16"),
        ("lora_weight", "torch.bfloat16"),
        ("layer_in", "torch.bfloat16"),
        ("layer_out", "torch.bfloat16"),
        ("stream_dtype", "bfloat16"),
        ("train_bf16", True),
        ("train_fp16", True),
        ("mixed_precision", "bf16"),
        ("grads", ["torch.float32", "torch.bfloat16"]),
    ],
)
def test_float32_is_unavailable_when_the_types_do_not_confirm_the_patch(diag, field, value):
    types = _all_types()
    types["f32_l2l"] = _types(**{field: value})
    # even equal arms say nothing about float32 when the types show something else
    verdict = diag.judge_float32(_arm(), _arm(), _arm(), types)
    assert verdict["state"] == "unavailable"
    assert any("f32_l2l" in p and field in p for p in verdict["problems"])


def test_float32_is_unavailable_when_accumulation_does_not_repeat_itself(diag):
    verdict = diag.judge_float32(_arm(), _nudge(_arm(), by=1e-3), _arm(), _all_types())
    assert verdict["state"] == "unavailable"
    assert any("f32_ga1" in p and "f32_ga2" in p for p in verdict["problems"])


def test_float32_is_unavailable_without_a_valid_arm(diag):
    verdict = diag.judge_float32(_arm(), None, _arm(), _all_types())
    assert verdict["state"] == "unavailable"
    assert any("f32_ga2" in p and "no valid" in p for p in verdict["problems"])


def test_float32_is_unavailable_when_an_arm_has_no_type_record(diag):
    types = _all_types()
    del types["f32_ga1"]
    verdict = diag.judge_float32(_arm(), _arm(), _arm(), types)
    assert verdict["state"] == "unavailable"
    assert any("f32_ga1" in p and "type record" in p for p in verdict["problems"])


# --------------------------------------------------------------------------
# control 2: the second SDPA backend is the floor, A3-07 factor 1.5
# --------------------------------------------------------------------------
def test_l2l_inside_one_and_a_half_floors_is_within_on_step_zero_and_on_all_steps(diag):
    floor = _nudge(_arm(), by=1e-3, loss_by=1e-4)
    l2l = _nudge(_arm(), by=1.4e-3, loss_by=1.4e-4)
    verdict = diag.judge_floor(_arm(), floor, l2l)
    assert verdict["state"] == "within"
    assert verdict["step0_within"] is True
    assert verdict["problems"] == []
    assert verdict["max_floor"] == pytest.approx(1e-3, rel=1e-3)
    assert verdict["max_gap"] == pytest.approx(1.4e-3, rel=1e-3)


def test_l2l_outside_the_limit_is_outside_and_names_the_tensor(diag):
    floor = _nudge(_arm(), by=1e-3, loss_by=1e-4)
    l2l = _nudge(_arm(), by=1.6e-3, loss_by=1.4e-4)
    verdict = diag.judge_floor(_arm(), floor, l2l)
    assert verdict["state"] == "outside"
    assert verdict["step0_within"] is False
    assert any("step 0" in p and "base.default.lora_A" in p for p in verdict["problems"])


def test_a_gap_after_step_zero_keeps_step_zero_within_but_the_control_outside(diag):
    floor = _nudge(_nudge(_arm(), by=1e-3), step=1, by=1e-3)
    l2l = _nudge(_nudge(_arm(), by=1e-3), step=1, by=5e-3)
    verdict = diag.judge_floor(_arm(), floor, l2l)
    assert verdict["step0_within"] is True
    assert verdict["state"] == "outside"
    assert any("step 1" in p for p in verdict["problems"])


def test_a_floor_of_zero_leaves_no_room_for_l2l(diag):
    inside = diag.judge_floor(_arm(), _arm(), _arm())
    assert inside["state"] == "within"
    assert inside["max_floor"] == 0.0
    outside = diag.judge_floor(_arm(), _arm(), _nudge(_arm(), by=1e-6))
    assert outside["state"] == "outside"


def test_the_floor_control_has_no_data_without_one_of_its_arms(diag):
    for arms in ((None, _arm(), _arm()), (_arm(), None, _arm()), (_arm(), _arm(), None)):
        verdict = diag.judge_floor(*arms)
        assert verdict["state"] == "no_data"
        assert verdict["problems"]


# --------------------------------------------------------------------------
# the outcome table of the ticket
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "float32, floor, row",
    [
        ("equal", "within", 1),
        ("equal", "outside", 2),
        ("not_equal", "within", 3),
        ("not_equal", "outside", 3),
        ("unavailable", "within", 4),
        ("unavailable", "outside", 5),
    ],
)
def test_the_two_controls_pick_the_row_of_the_table(diag, float32, floor, row):
    assert diag.judge_outcome(float32, floor)["row"] == row


@pytest.mark.parametrize("float32", ["equal", "not_equal", "unavailable"])
def test_a_floor_control_without_data_leaves_the_outcome_undefined(diag, float32):
    outcome = diag.judge_outcome(float32, "no_data")
    assert outcome["row"] is None
    assert "undefined" in outcome["conclusion"]


def test_each_row_says_what_it_means_in_the_words_of_the_ticket(diag):
    said = {row: diag.judge_outcome(*pair)["conclusion"] for row, pair in {
        1: ("equal", "within"), 2: ("equal", "outside"), 3: ("not_equal", "within"),
        4: ("unavailable", "within"), 5: ("unavailable", "outside"),
    }.items()}
    assert "rounding" in said[1] and "no bug" in said[1]
    assert "user decides" in said[2]
    assert "bug" in said[3] and "separate ticket" in said[3]
    assert "precision not separated" in said[4]
    assert "cause not established" in said[5]


@pytest.mark.parametrize(
    "float32, needed, dtype",
    [
        ("equal", False, None),
        ("not_equal", True, "float32"),
        ("unavailable", True, "native"),
    ],
)
def test_the_layer_trace_is_needed_unless_float32_is_bit_equal(diag, float32, needed, dtype):
    outcome = diag.judge_outcome(float32, "within")
    assert outcome["trace_needed"] is needed
    assert outcome["trace_dtype"] == dtype


# --------------------------------------------------------------------------
# control 3: the first layer whose output differs
# --------------------------------------------------------------------------
def _trace(layers=3, scale=1.0):
    import torch

    return {
        "embed": torch.ones(2, 4, 3) * scale,
        "layers": [torch.ones(2, 4, 3) * scale * (i + 2) for i in range(layers)],
    }


def test_equal_traces_name_no_layer(diag):
    verdict = diag.first_diverging_layer(_trace(), _trace())
    assert verdict["state"] == "equal"
    assert verdict["first"] is None


def test_the_first_differing_layer_is_named_with_the_count_and_the_gap(diag):
    other = _trace()
    other["layers"][1] = other["layers"][1] + 0.5
    other["layers"][2] = other["layers"][2] + 0.25
    verdict = diag.first_diverging_layer(_trace(), other)
    assert verdict["state"] == "differs"
    assert verdict["first"] == 1
    assert verdict["differing"] == [1, 2]
    assert verdict["max_abs_at_first"] == pytest.approx(0.5)


def test_a_gap_in_the_embedding_is_named_before_any_layer(diag):
    other = _trace()
    other["embed"] = other["embed"] + 1e-3
    verdict = diag.first_diverging_layer(_trace(), other)
    assert verdict["first"] == "embed"


def test_traces_of_other_shapes_cannot_be_compared(diag):
    import torch

    other = _trace()
    other["layers"][0] = torch.ones(1, 4, 3)
    verdict = diag.first_diverging_layer(_trace(), other)
    assert verdict["state"] == "no_data"
    assert "shape" in verdict["reason"]


def test_a_missing_trace_is_no_data(diag):
    assert diag.first_diverging_layer(None, _trace())["state"] == "no_data"
