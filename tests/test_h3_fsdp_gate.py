"""CPU checks for H3 measurement logic; these do not validate distributed CUDA."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "benchmarks/harness/h3_fsdp_gate.py"
_SPEC = importlib.util.spec_from_file_location("h3_fsdp_gate", _PATH)
gate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gate
_SPEC.loader.exec_module(gate)


def test_batch_plan_preserves_global_batch():
    assert gate.accumulation(8, 1, 1) == 8
    assert gate.accumulation(8, 1, 2) == 4
    with pytest.raises(ValueError, match="divisible"):
        gate.accumulation(7, 1, 2)


def test_materialized_forward_precision_cannot_be_inferred_from_master_shards():
    modes = {"A": True, "B": True}
    gate.assert_forward_precision({"A": "torch.float32", "B": "torch.float32"}, modes, 2, True)
    for actual in ({"A": "torch.bfloat16", "B": "torch.bfloat16"},
                   {"A": "torch.float32"}, {}):
        with pytest.raises(ValueError, match="materialized LoRA"):
            gate.assert_forward_precision(actual, modes, 2, True)


def test_fp32_lora_without_actual_autocast_cannot_pass_a_bf16_control():
    dtypes = {"A": "torch.float32", "B": "torch.float32"}
    for modes in ({"A": False, "B": False}, {"A": True}, {}):
        with pytest.raises(ValueError, match="actual forward autocast"):
            gate.assert_forward_precision(dtypes, modes, 2, True)


@pytest.mark.parametrize("args", [(0, 1, 2), (8, 0, 2), (8, 1, 0), (True, 1, 2)])
def test_invalid_batch_plan(args):
    with pytest.raises(ValueError):
        gate.accumulation(*args)


def test_dtensor_uses_actual_local_storage_not_global_numel():
    class Local:
        def numel(self):
            return 3

    class DTensor:
        def numel(self):
            raise AssertionError("global numel is not resident storage")

        def to_local(self):
            return Local()

    assert gate.parameter_local_numel(DTensor()) == 3


def test_sharding_accepts_uneven_orig_parameter_views_but_refuses_replication():
    gate.assert_shards([5, 4], total=9, fsdp_units=[2, 2])
    with pytest.raises(ValueError, match="shard"):
        gate.assert_shards([9, 9], total=9, fsdp_units=[2, 2])
    with pytest.raises(ValueError, match="FSDP"):
        gate.assert_shards([5, 4], total=9, fsdp_units=[0, 0])


def test_data_audit_detects_repeated_rank_samples_and_incomplete_accumulation():
    gate.assert_data_window([["a", "b"], ["c", "d"]], global_batch=4)
    for rows in ([["a", "b"], ["a", "d"]], [["a"], ["b"]]):
        with pytest.raises(ValueError):
            gate.assert_data_window(rows, global_batch=4)


def test_measurement_uses_slowest_rank_and_actual_tokens():
    result = gate.throughput([[1.0, 2.0], [2.0, 2.0]], [4096, 4096])
    assert result["tokens_per_second"] == 2048
    assert result["seconds_per_optimizer_step"] == 2
    for times in ([[0]], [[float("nan")]], [[1], [1, 2]]):
        with pytest.raises(ValueError):
            gate.throughput(times, [1] * len(times))


def test_tensor_checks_allow_zero_first_step_a_but_require_some_update():
    import torch

    before = {"A": torch.ones(2), "B": torch.zeros(2)}
    after = {"A": torch.ones(2), "B": torch.ones(2) / 10}
    assert gate.adapter_delta(before, after)["changed_tensors"] == ["B"]
    with pytest.raises(ValueError, match="update"):
        gate.adapter_delta(before, before)
    with pytest.raises(ValueError, match="keys"):
        gate.adapter_delta(before, {"A": before["A"]})
    with pytest.raises(ValueError, match="finite"):
        gate.adapter_delta(before, {**after, "A": torch.tensor([float("nan"), 1])})


def test_reload_rejects_missing_keys_and_changed_values():
    import torch

    expected = {"a": torch.tensor([1.0, 2.0])}
    gate.assert_adapter_equal(expected, expected)
    with pytest.raises(ValueError, match="keys"):
        gate.assert_adapter_equal(expected, {})
    with pytest.raises(ValueError, match="values"):
        gate.assert_adapter_equal(expected, {"a": torch.zeros(2)})


def test_summary_refuses_failed_or_mismatched_arms():
    good = {"status": "passed", "identity": {"model": "abc"}, "world_size": 1,
            "tokens_per_second": 10}
    two = {**good, "world_size": 2, "tokens_per_second": 15}
    assert gate.scaling([good, two])["efficiency_2"] == 0.75
    for bad in ({**two, "status": "failed"}, {**two, "identity": {"model": "xyz"}}):
        with pytest.raises(ValueError):
            gate.scaling([good, bad])


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_scaling_refuses_invalid_throughput(rate):
    runs = [{"status": "passed", "identity": {}, "world_size": size,
             "tokens_per_second": rate if size == 2 else 10} for size in (1, 2)]
    with pytest.raises(ValueError):
        gate.scaling(runs)


def test_digest_handles_scalar_and_empty_rank_local_parameter_views():
    import torch

    assert gate.tensor_digest(torch.tensor(1.0)) != gate.tensor_digest(torch.tensor([1.0]))
    assert gate.tensor_digest(torch.empty(0)) == gate.tensor_digest(torch.empty(0))


def test_fsdp2_accounts_for_two_sharded_copies_of_a_previously_tied_weight():
    # Hardware failure: embed/lm_head become distinct DTensors/storage after
    # wrapping. Both shards are real; comparing their sum to pre-wrap unique
    # numel rejects valid sharding and hides the actual duplication.
    expected = {"embed": [10], "head": [10], "other": [6]}
    aliases = [["embed", "head"], ["other"]]
    rank = {name: {"global_shape": shape, "local_numel": shape[0] // 2,
                   "sharded": True, "dtype": "bf16"} for name, shape in expected.items()}
    with pytest.raises(ValueError, match="resident shard accounting"):
        gate.assert_shards([13, 13], total=16, fsdp_units=[2, 2])
    result = gate.assert_fsdp2_shards([rank, rank], expected, aliases)
    assert result["prewrap_unique_parameters"] == 16
    assert result["postwrap_registered_parameters"] == 26
    assert result["local_parameters_per_rank"] == [13, 13]
    assert result["extra_alias_parameters"] == 10


def test_fsdp2_accepts_an_alias_that_stays_tied_and_uneven_shards():
    expected = {"embed": [5], "head": [5]}
    ranks = [{"embed": {"global_shape": [5], "local_numel": size,
                        "sharded": True, "dtype": "bf16"}} for size in (3, 2)]
    assert gate.assert_fsdp2_shards(ranks, expected, [["embed", "head"]])[
        "extra_alias_parameters"] == 0


@pytest.mark.parametrize("change", ["replicated", "undercounted", "missing", "shape", "dtype"])
def test_fsdp2_refuses_false_sharding_claims(change):
    import copy

    expected = {"weight": [8], "other": [4]}
    rank = {name: {"global_shape": shape, "local_numel": shape[0] // 2,
                   "sharded": True, "dtype": "bf16"} for name, shape in expected.items()}
    ranks = [rank, copy.deepcopy(rank)]
    if change == "replicated":
        ranks[1]["weight"].update(sharded=False, local_numel=8)
    elif change == "undercounted":
        ranks[1]["weight"]["local_numel"] = 1
    elif change == "missing":
        ranks[1].pop("other")
    elif change == "shape":
        ranks[1]["weight"]["global_shape"] = [7]
    else:
        ranks[1]["weight"]["dtype"] = "fp32"
    with pytest.raises(ValueError):
        gate.assert_fsdp2_shards(ranks, expected, [["weight"], ["other"]])
