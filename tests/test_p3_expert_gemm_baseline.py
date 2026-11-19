"""A2 stage-0 harness (benchmarks/harness/p3_expert_gemm_baseline.py) on CPU.

The harness's verdict rests on three pieces: ``invariance`` says ``invariant`` only for a valid
product whose rows do not depend on M (each probe kind catches a function that breaks only
there; a wrong-but-constant function is not valid), the bit comparison catches a single
differing bit, and the fixed-config Triton GEMM computes ``a @ b`` for every M (row masks, ragged
K and N tails, transposed weight views). The Triton kernel runs under ``TRITON_INTERPRET=1``,
which only takes effect if set before Triton is first imported, so those checks run in a fresh
interpreter. On CPU they check the kernel's indexing; GPU arithmetic is measured on a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
HARNESS = "benchmarks/harness/p3_expert_gemm_baseline.py"


@pytest.fixture(scope="module")
def base():
    from benchmarks.harness import p3_expert_gemm_baseline

    return p3_expert_gemm_baseline


def _env(**extra):
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH")])
    )
    env.update(extra)
    return env


def _run(args, **extra):
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300,
        env=_env(**extra),
    )


def _probe(base, fn, *, rows=None):
    import torch

    ms, patterns = (1, 2, 16, 17), (base.equal_pattern(4, 3), base.RAGGED_PATTERN)
    rows = rows or base.reference_rows(ms, patterns)
    generator = torch.Generator().manual_seed(0)
    a = torch.randn((rows, 24), generator=generator)
    b = torch.randn((24, 8), generator=generator)
    return base.invariance(
        lambda x: fn(x, b), a, b, ms=ms, patterns=patterns, tolerance=1e-4
    )


def test_invariance_accepts_a_row_independent_product(base):
    result = _probe(base, lambda x, b: x @ b)
    assert result["invariant"] and result["valid"] and result["bit_equal_all"]
    assert [item["m"] for item in result["shifted"]] == [1, 2, 16, 17]
    assert [p["name"] for p in result["patterns"]] == ["4x3", "ragged"]
    assert result["rows"] > max(17, sum(base.RAGGED)) + base.ROW_OFFSET


def test_each_probe_kind_catches_its_own_break(base):
    import torch

    def nudge(out):  # one ulp on the first element
        out = out.clone()
        out.view(torch.int32)[0, 0] += 1
        return out

    full_rows = _probe(base, lambda x, b: x @ b)["rows"]

    def only_small(x, b):  # a different rounding for short calls, like a cuBLAS GEMV path
        return nudge(x @ b) if x.shape[0] <= 2 else x @ b

    def only_shifted(x, b):  # rows depend on where they start: probe rows 3.. but not 0..
        out = x @ b
        return nudge(out) if x.shape[0] == 16 and x.storage_offset() == 3 * x.shape[1] else out

    def only_mid_sizes(x, b):  # size 3 appears only in patterns (4x3, ragged), not as a prefix m
        return nudge(x @ b) if x.shape[0] in (3, 5) else x @ b

    small = _probe(base, only_small)
    assert not small["invariant"]
    assert {i["m"] for i in small["prefix"] if not i["bit_equal"]} == {1, 2}

    shifted = _probe(base, only_shifted)
    assert not shifted["invariant"]
    assert all(i["bit_equal"] for i in shifted["prefix"])
    assert [i["m"] for i in shifted["shifted"] if not i["bit_equal"]] == [16]

    mid = _probe(base, only_mid_sizes)
    assert not mid["invariant"]
    assert all(i["bit_equal"] for key in ("prefix", "shifted") for i in mid[key])
    assert {p["name"] for p in mid["patterns"] if not p["bit_equal"]} == {"4x3", "ragged"}
    assert full_rows == small["rows"]


def test_wrong_but_constant_output_is_not_valid(base):
    import torch

    zeros = _probe(base, lambda x, b: torch.zeros(x.shape[0], b.shape[1]))
    assert zeros["bit_equal_all"] and not zeros["valid"] and not zeros["invariant"]
    nans = _probe(base, lambda x, b: torch.full((x.shape[0], b.shape[1]), float("nan")))
    assert not nans["finite"] and not nans["valid"] and not nans["invariant"]
    assert nans["fp64_relative_l2"] is None  # stays standard JSON
    json.dumps(nans, allow_nan=False)


def test_invariance_refuses_a_reference_no_larger_than_the_probes(base):
    with pytest.raises(ValueError, match="reference rows"):
        _probe(base, lambda x, b: x @ b, rows=sum(base.RAGGED))


def test_bit_diff_counts_one_flipped_low_bit(base):
    import torch

    want = torch.randn(4, 8).to(torch.float16)
    got = want.clone()
    got.view(torch.int16)[2, 5] ^= 1
    diff = base.bit_diff(got, want)
    assert not diff["bit_equal"]
    assert (diff["diff_elements"], diff["diff_rows"]) == (1, 1)
    assert diff["max_abs"] > 0
    zero = torch.zeros(1, 1)
    assert not base.bit_diff(-zero, zero)["bit_equal"]  # -0.0 == 0.0, but not in bits


def test_speed_units_and_pattern_modes(base):
    import torch

    op = base.Op("probe", k=4, n=8, transposed=False)
    a = torch.zeros(64, 4)
    calls = []

    def timer(fn, graph=False):
        out = fn()
        calls.append((graph, len(out) if isinstance(out, list) else 1))
        return (2.0, 1.0, 3.0) if graph else (4.0, 3.0, 5.0)

    result = base.speed(
        lambda x: x, op, a, ms=(1, 16), patterns=(base.equal_pattern(4, 2),), timer=timer
    )
    first = result["sweep"][1]
    assert first["m"] == 16 and first["median_us"] == 4.0
    assert first["tflops"] == pytest.approx(2 * 16 * 4 * 8 / 4e-6 / 1e12)
    assert first["weight_gbps"] == pytest.approx(4 * 8 * 4 / 4e-6 / 1e9)
    pattern = result["patterns"][0]
    assert pattern["eager"]["separate_over_merged"] == 1.0
    assert pattern["graph"]["separate_us"] == 2.0
    assert (False, 4) in calls and (True, 4) in calls  # separate: one call per micro-batch


def test_fixed_gemm_under_interpreter_matches_reference_and_is_invariant():
    pytest.importorskip("triton")
    script = textwrap.dedent(
        """
        import torch
        from benchmarks.harness import p3_expert_gemm_baseline as base

        config = base.CONFIGS["m16n32k32w4s2"]
        generator = torch.Generator().manual_seed(3)
        k, n = 72, 40  # K and N not multiples of the 32-wide tiles
        for dtype in (torch.float16, torch.float32):
            for transposed in (True, False):
                for m in (1, 5, 16, 17, 33):
                    a = torch.randn((m, k), generator=generator).to(dtype)
                    stored = torch.randn((n, k) if transposed else (k, n), generator=generator)
                    b = stored.to(dtype).t() if transposed else stored.to(dtype)
                    got = base.fixed_gemm(a, b, config)
                    assert got.shape == (m, n) and got.dtype == dtype
                    torch.testing.assert_close(
                        got.double(), a.double() @ b.double(), rtol=2e-3, atol=2e-2
                    )
        op = base.Op("probe", k=96, n=48, transposed=True)
        patterns = (base.equal_pattern(4, 3), base.RAGGED_PATTERN)
        ms = (1, 2, 15, 16, 17, 64)
        a, weight = base.make_operands(op, base.reference_rows(ms, patterns), "float16", "cpu", 1)
        fn = base.kernel_fn("triton:m16n32k32w4s2", op, weight)
        result = base.invariance(fn, a, weight.t(), ms=ms, patterns=patterns, tolerance=2e-3)
        assert result["invariant"], result
        """
    )
    completed = _run(["-c", script], TRITON_INTERPRET="1")
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_cli_cpu_smoke_writes_complete_results(tmp_path):
    pytest.importorskip("triton")
    output = tmp_path / "baseline.json"
    completed = _run([
        HARNESS, "--device", "cpu", "--no-timing", "--models", "tiny",
        "--dtypes", "float16,bfloat16", "--configs", "m16n32k32w4s2", "--ms", "1,3,16",
        "--patterns", "4x2,ragged", "--output", str(output),
    ], TRITON_INTERPRET="1")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["complete"]
    rows = {(row["op"]["name"], row["dtype"], row["kernel"]): row for row in result["results"]}
    assert len(rows) == 4 * 2 * 3  # ops x dtypes x (cublas, cublas-norpr, one Triton config)
    for op in ("gate_up.fwd", "down.fwd", "down.dx", "gate_up.dx"):
        assert rows[(op, "float16", "triton:m16n32k32w4s2")]["invariance"]["invariant"]
        assert rows[(op, "float16", "cublas")]["invariance"]["valid"]
        assert "TRITON_INTERPRET" in rows[(op, "bfloat16", "triton:m16n32k32w4s2")]["unsupported"]


@pytest.mark.parametrize("extra, env", [
    (["--models", "tiny"], {}),  # CPU timing
    (["--no-timing", "--configs", "nope"], {}),
    (["--no-timing", "--models", "tiny", "--ms", "1"], {}),  # Triton on CPU without interpreter
    (["--device", "cuda", "--no-timing"], {"TRITON_INTERPRET": "1"}),
])
def test_cli_refuses_bad_runs(tmp_path, extra, env):
    output = tmp_path / "x.json"
    args = [HARNESS, "--device", "cpu", *extra, "--output", str(output)]
    completed = _run(args, **env)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert not output.exists()
