"""Targeting of the four uncalibrated gate/up fallback cells (T, --fallback-gateup).

CPU only. The kernel threshold is read from the installed bitsandbytes source, not
reimplemented, and is skipped when bitsandbytes is not installed.
"""

import ast
import importlib.util

import pytest

import tests.e4_part2_support  # noqa: F401  (puts the harness on sys.path)

torch = pytest.importorskip("torch")

import e4_part2_calibration as cal  # noqa: E402
import e4_part2_quant as quant  # noqa: E402


def test_targets_are_in_the_fallback_range_and_one_per_call():
    assert len(cal.FALLBACK_GATEUP_M) == cal.CALLS_PER_UNIT
    assert all(129 <= m <= 512 for m in cal.FALLBACK_GATEUP_M)


def test_tile_rows_repeats_in_order_to_exact_length():
    x = torch.arange(6.0).reshape(3, 2)
    y = cal.tile_rows(x, 7)
    assert y.shape == (7, 2)
    assert torch.equal(y[:3], x) and torch.equal(y[3:6], x) and torch.equal(y[6], x[0])


def test_gateup_selection_is_the_full_t_selection_restricted():
    units = quant.enumerate_units()
    full = cal.select_units(units, cal.T_SELECTION_SEED, 8)
    inputs = {u.key: [torch.zeros(5, 4)] for us in full.values() for u in us}
    kept, tiled = cal.fallback_targets(full, inputs)
    assert set(kept) == set(cal.FALLBACK_GATEUP_KINDS)
    assert [u.key for u in kept["gate_proj"]] == [u.key for u in full["gate_proj"]]
    assert [u.key for u in kept["up_proj"]] == [u.key for u in full["up_proj"]]
    for key, calls in tiled.items():
        assert [c.shape[0] for c in calls] == list(cal.FALLBACK_GATEUP_M)


def test_a_unit_without_captured_input_gets_no_calls():
    units = quant.enumerate_units()
    full = cal.select_units(units, cal.T_SELECTION_SEED, 8)
    _, tiled = cal.fallback_targets(full, {})
    assert tiled and all(calls == [] for calls in tiled.values())


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
