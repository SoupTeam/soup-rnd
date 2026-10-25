"""Targeting of the four uncalibrated gate/up fallback cells (T, --fallback-gateup).

Each call is the concatenation of the first captured SMOKE inputs of distinct sources, in unit
order, truncated to M. CPU only. The kernel threshold is read from the installed bitsandbytes
source, not reimplemented, and the test is skipped when bitsandbytes is not installed.
"""

import ast
import importlib.util
import types

import pytest

import tests.e4_part2_support  # noqa: F401  (puts the harness on sys.path)

torch = pytest.importorskip("torch")

import e4_part2_calibration as cal  # noqa: E402
import e4_part2_quant as quant  # noqa: E402


def unit(key, kind, layer, expert):
    return types.SimpleNamespace(key=key, kind=kind, layer=layer, expert=expert)


def rows(tag, n, width=3):
    """n distinct rows; each row encodes (tag, row index) so duplicates are detectable."""
    base = torch.arange(n, dtype=torch.float32).reshape(n, 1)
    return torch.cat([torch.full((n, 1), float(tag)), base, torch.zeros(n, width - 2)], dim=1)


def test_targets_are_in_the_fallback_range_and_one_per_call():
    assert len(cal.FALLBACK_GATEUP_M) == cal.CALLS_PER_UNIT
    assert all(129 <= m <= 512 for m in cal.FALLBACK_GATEUP_M)


def test_concatenates_distinct_sources_in_unit_order_and_truncates_the_last():
    order = [unit("a", "gate_proj", 0, 1), unit("b", "up_proj", 0, 2), unit("c", "gate_proj", 1, 0)]
    inputs = {"a": [rows(1, 4)], "b": [rows(2, 5)], "c": [rows(3, 6)]}
    x, used, total = cal.concat_sources(order, inputs, 12)
    assert total == 12 and used == ["a", "b", "c"]
    assert torch.equal(x[:4], rows(1, 4))
    assert torch.equal(x[4:9], rows(2, 5))
    assert torch.equal(x[9:12], rows(3, 6)[:3])  # truncated, not repeated


def test_no_row_is_used_twice_across_sources_within_a_call():
    order = [unit("a", "gate_proj", 0, 1), unit("b", "gate_proj", 0, 2)]
    inputs = {"a": [rows(1, 4)], "b": [rows(2, 4)]}
    x, _, total = cal.concat_sources(order, inputs, 8)
    assert total == 8 and len({tuple(r.tolist()) for r in x}) == 8


def test_gate_and_up_of_the_same_expert_are_one_source():
    # gate and up of one expert see the same routed tokens: count those rows once
    order = [unit("g", "gate_proj", 3, 7), unit("u", "up_proj", 3, 7), unit("v", "gate_proj", 3, 8)]
    inputs = {"g": [rows(1, 4)], "u": [rows(1, 4)], "v": [rows(2, 4)]}
    x, used, total = cal.concat_sources(order, inputs, 8)
    assert used == ["g", "v"] and total == 8


def test_a_shortage_is_reported_not_padded():
    order = [unit("a", "gate_proj", 0, 1)]
    _, used, total = cal.concat_sources(order, {"a": [rows(1, 4)]}, 9)
    assert total == 4 and used == ["a"]


def test_fallback_targets_are_the_gateup_units_with_one_call_per_target_m():
    units = quant.enumerate_units()
    full = cal.select_units(units, cal.T_SELECTION_SEED, 8)
    inputs = {u.key: [rows(i + 1, 40)] for i, us in enumerate(full.values()) for u in us}
    kept, tiled, sources = cal.fallback_targets(full, inputs)
    assert set(kept) == set(cal.FALLBACK_GATEUP_KINDS)
    assert [u.key for u in kept["gate_proj"]] == [u.key for u in full["gate_proj"]]
    for key, calls in tiled.items():
        assert [c.shape[0] for c in calls] == list(cal.FALLBACK_GATEUP_M)
        assert [s["M"] for s in sources[key]] == list(cal.FALLBACK_GATEUP_M)
        assert all(s["source_units"] for s in sources[key])


def test_a_pool_too_small_for_a_target_m_stops_the_run():
    units = quant.enumerate_units()
    full = cal.select_units(units, cal.T_SELECTION_SEED, 8)
    inputs = {u.key: [rows(1, 2)] for us in full.values() for u in us}
    with pytest.raises(cal.common.CalibrationStopped, match="cannot reach M"):
        cal.fallback_targets(full, inputs)


def _bitsandbytes_custom_fn():
    spec = importlib.util.find_spec("bitsandbytes")
    if spec is None:
        return None
    path = spec.submodule_search_locations[0] + "/backends/cuda/ops.py"
    src = open(path).read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_gemm_4bit_use_custom_cuda")
    ns = {"torch": torch, "_GEMM_4BIT_CUSTOM_FLOOR_M": 4,
          "_gpu_dispatch_props": lambda _i: (108, 8, 0)}  # A100 PCIe 40 GB: 108 SMs, sm80
    exec(ast.get_source_segment(src, fn), ns)
    return ns["_gemm_4bit_use_custom_cuda"]


@pytest.mark.skipif(_bitsandbytes_custom_fn() is None, reason="bitsandbytes not installed")
def test_the_installed_kernel_rule_sends_gateup_to_fallback_at_every_target_m():
    custom = _bitsandbytes_custom_fn()
    for m in cal.FALLBACK_GATEUP_M:
        assert custom(0, torch.bfloat16, m, 1024, 2048) is False  # gate/up: N=1024, K=2048
    assert custom(0, torch.bfloat16, 128, 1024, 2048) is True      # last fused M
