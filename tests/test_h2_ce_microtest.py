"""CPU controls for the attempt017 CE micro-test: math, labels, gates, ordering and in-place guards.

The real Liger kernel and the CUDA bf16 matmul are GPU-only. The two-phase flow is exercised in a
subprocess with a stand-in liger_kernel module (the H2-L1 pattern), so the global rebind cannot leak
into the test session.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/harness"))
import h2_ce_microtest as mt  # noqa: E402

torch = pytest.importorskip("torch")
WORKSET = mt.load_workset()
SMALL = mt.Config(vocab=64, hidden=16, label_mod=64, device="cpu")


# ------------------------------------------------------------------ cases and denominators
def test_case_grid_is_200_unique_cases_with_seed_outermost():
    cases = mt.all_cases()
    assert len(cases) == mt.EXPECTED_CASES == 200
    assert len({c.id for c in cases}) == 200
    assert [c.seed for c in cases] == sorted(c.seed for c in cases)
    assert {c.path for c in cases} == {"mean", "sum24"}
    assert {(c.kind, c.scale) for c in cases} == set(mt.REGIMES)


def test_denominators_follow_the_two_paths():
    assert mt.denominator("mean", 6) == 6
    assert mt.denominator("mean", 54) == 54
    assert mt.denominator("sum24", 6) == 24 and mt.denominator("sum24", 54) == 24
    with pytest.raises(ValueError):
        mt.denominator("other", 1)


# ------------------------------------------------------------------ reference and metrics
def brute_force(z, shifted, denom):
    import torch.nn.functional as functional

    grads = torch.zeros_like(z, dtype=torch.float64)
    for row in range(z.shape[0]):
        if shifted[row] == -100:
            continue
        probs = functional.softmax(z[row].double(), dim=-1)
        probs[shifted[row]] -= 1.0
        grads[row] = probs / denom
    return grads


@pytest.mark.parametrize("path", ["mean", "sum24"])
def test_reference_matches_brute_force_and_torch_fp64(path):
    torch.manual_seed(0)
    z = torch.randn(12, 50) * 3
    shifted = torch.tensor([3, -100, 7, 49, -100, 0, 11, 11, -100, 20, 5, 2])
    denom = mt.denominator(path, int((shifted != -100).sum()))
    ref, check = mt.reference_gradients(z, shifted, denom)
    assert check["ok"] and check["spread"] <= mt.REFERENCE_TOLERANCE
    assert torch.allclose(ref, brute_force(z, shifted, denom), rtol=0, atol=1e-14)
    assert (ref[shifted == -100] == 0).all()  # Ignored rows carry exactly zero gradient
    leaf = z.double().clone().requires_grad_(True)
    reduction = "mean" if path == "mean" else "sum"
    loss = torch.nn.functional.cross_entropy(leaf, shifted, reduction=reduction)
    (loss if path == "mean" else loss / 24).backward()
    assert torch.allclose(ref, leaf.grad, rtol=0, atol=1e-14)
    assert check["loss"] == pytest.approx(float((loss if path == "mean" else loss / 24).detach()))


def test_reference_self_check_fails_closed(monkeypatch):
    z = torch.randn(4, 20)
    shifted = torch.tensor([1, 2, -100, 3])
    monkeypatch.setattr(mt, "REFERENCE_TOLERANCE", -1.0)
    _, check = mt.reference_gradients(z, shifted, 3)
    assert check["ok"] is False
    with pytest.raises(ValueError):
        mt.reference_gradients(torch.randn(2, 5), torch.tensor([-100, -100]), 1)  # all ignored


def test_alt_route_agrees_with_the_reference_to_fp32_level():
    torch.manual_seed(1)
    z = torch.randn(10, 40) * 2
    shifted = torch.tensor([1, -100, 5, 7, 39, -100, 2, 2, 3, 4])
    ref, _ = mt.reference_gradients(z, shifted, 8)
    assert mt.metrics(mt.alt_gradient(z, shifted, 8), ref)["e"] < 1e-6


def test_metrics_known_values_and_degenerate_inputs():
    ref = torch.tensor([3.0, 4.0], dtype=torch.float64)
    same = mt.metrics(ref.float(), ref)
    assert same == {"e": 0.0, "m": 0.0, "c": pytest.approx(1.0)}
    shifted = mt.metrics(torch.tensor([3.0, 5.0]), ref)
    assert shifted["e"] == pytest.approx(1 / 5) and shifted["m"] == pytest.approx(1 / 4)
    with pytest.raises(ValueError):
        mt.metrics(torch.tensor([1.0, 1.0]), torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        mt.metrics(torch.tensor([float("inf"), 1.0]), ref)
    assert mt.relative_l2(torch.tensor([2.0]), torch.tensor([4.0])) == pytest.approx(0.5)


def test_hash_changes_on_one_bit_for_bf16_and_int64():
    a = torch.randn(8).bfloat16()
    b = a.clone()
    assert mt.sha256_tensor(a) == mt.sha256_tensor(b)
    b.view(torch.int16)[3] ^= 1
    assert mt.sha256_tensor(a) != mt.sha256_tensor(b)
    labels = torch.tensor([1, -100, 5])
    assert mt.sha256_tensor(labels) != mt.sha256_tensor(torch.tensor([1, -100, 6]))


# ------------------------------------------------------------------ inputs
@pytest.mark.parametrize("shape,targets", [(0, 6), (1, 12), (2, 24), (3, 54)])
def test_inputs_are_deterministic_and_keep_the_real_masks(shape, targets):
    for regime in range(len(mt.REGIMES)):
        case = mt.Case(shape, regime, 0, "mean")
        first = mt.make_inputs(case, SMALL, WORKSET)
        second = mt.make_inputs(case, SMALL, WORKSET)
        assert first["hashes"] == second["hashes"] and first["targets"] == targets
        assert (first["shifted"] != -100).sum() == targets
        assert first["z"].dtype == torch.bfloat16 and first["z"].shape[-1] == SMALL.vocab
        assert int(first["shifted"].max()) < SMALL.vocab
        assert first["labels"][..., 0].eq(-100).all()
    other = mt.make_inputs(mt.Case(shape, 0, 1, "mean"), SMALL, WORKSET)
    assert other["hashes"]["logits"] != first["hashes"]["logits"]


def test_logits_are_cpu_gaussian_without_any_matmul(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a matmul was used to build the logits")

    for name in ("matmul", "mm", "bmm", "einsum", "addmm"):
        monkeypatch.setattr(torch, name, boom)
    monkeypatch.setattr(torch.Tensor, "__matmul__", boom)
    monkeypatch.setattr(torch.Tensor, "matmul", boom)
    big = mt.Config(vocab=2000, hidden=16, label_mod=2000, device="cpu")
    for scale_index, scale in ((0, 1.0), (1, 2.0), (2, 3.0)):
        data = mt.make_inputs(mt.Case(3, scale_index, 0, "mean"), big, WORKSET)
        z = data["z"].float()
        assert data["z"].dtype == torch.bfloat16 and data["z"].device.type == "cpu"
        assert abs(float(z.mean())) < 0.05 * scale + 0.02
        assert float(z.std()) == pytest.approx(scale, rel=0.02)


def test_matmul_construction_is_a_different_tensor_than_the_shipped_one():
    case = mt.Case(2, 0, 0, "mean")
    shipped = mt.make_inputs(case, SMALL, WORKSET)
    probe = WORKSET["probes"][case.shape]
    rows = len(probe["labels"]) * len(probe["labels"][0])
    gen = torch.Generator().manual_seed(case.seed * 1_000_003 + case.shape * 1009)
    hidden = (torch.randn(rows, SMALL.hidden, generator=gen) / 4).bfloat16()
    matmul_based = (hidden @ mt.build_weight(SMALL).T).contiguous()
    assert mt.sha256_tensor(matmul_based) != shipped["hashes"]["logits"]


def test_weight_is_one_cpu_draw_from_its_own_seed():
    first, second = mt.build_weight(SMALL), mt.build_weight(SMALL)
    assert torch.equal(first, second) and first.dtype == torch.bfloat16
    assert tuple(first.shape) == (SMALL.vocab, SMALL.hidden) and mt.WEIGHT_SEED == 20261029


def test_protocol_lists_the_cpu_logit_construction_and_its_realism_caveats():
    text = (ROOT / "benchmarks/h2/CE_MICROTEST_PROTOCOL.txt").read_text()
    for needle in (
        "revision 2e",
        "not from a CUDA matmul",
        "independent across the vocabulary",
        "independent of the labels",
        "not low rank",
        "drawn on the CPU",
        "bf16 rounding differs",
    ):
        assert needle in text, needle


def test_self_consistent_targets_are_drawn_not_the_real_labels_but_keep_the_mask():
    real = mt.make_inputs(mt.Case(3, 0, 0, "mean"), SMALL, WORKSET)
    drawn = mt.make_inputs(mt.Case(3, 3, 0, "mean"), SMALL, WORKSET)
    assert torch.equal(real["shifted"] != -100, drawn["shifted"] != -100)
    assert not torch.equal(real["shifted"], drawn["shifted"])


# ------------------------------------------------------------------ labels, status, gate
def rec(e_p=1e-8, e_l=1e-8, rho=1.0, **extra):
    row = {
        "e_p": e_p,
        "e_l": e_l,
        "rho": rho,
        "id": extra.pop("id", "x"),
        "kind": "random",
        "scale": 1.0,
        "shape": 0,
        "path": "mean",
        "seed": 0,
        "flips_pl": 0,
        "e_alt": 1e-8,
        "d_pl": 1e-8,
        "d_p_alt": 1e-8,
        "d_l_alt": 1e-8,
        "dh_pl": 0.0,
        "p_repeat_equal": True,
        "l_repeat_equal": True,
    }
    row.update(extra)
    return row


def many(count=200, **first):
    rows = [rec(id=str(i)) for i in range(count)]
    for key, value in first.items():
        rows[0][key] = value
    return rows


A = mt.A


@pytest.mark.parametrize(
    "records,label,marginal",
    [
        (many(), "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC", False),
        (
            many(e_l=A, rho=mt.RHO),
            "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC",
            True,
        ),  # equality is not worse
        (many(e_l=A * 1.0000001, rho=1.0), "INCONCLUSIVE_AT_CE_LEVEL", True),
        (many(e_l=A * 0.5, rho=2.0000001), "INCONCLUSIVE_AT_CE_LEVEL", True),  # floor effect
        (many(e_l=A * 2, rho=2.5), "LIGER_WORSE_AT_CE_LEVEL_SYNTHETIC", False),
        (many(e_l=A * 1.05, rho=3.0), "LIGER_WORSE_AT_CE_LEVEL_SYNTHETIC", True),
        (
            many(e_l=A * 2, rho=2.0),
            "INCONCLUSIVE_AT_CE_LEVEL",
            True,
        ),  # rho at the limit: not worse
        (many(e_l=A * 0.8, rho=0.8), "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC", False),
        (many(e_l=A * 0.91, rho=1.0), "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC", True),
        (many(e_l=A * 0.89, rho=1.0), "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC", False),
        (many(e_l=A * 1.09, rho=1.0), "INCONCLUSIVE_AT_CE_LEVEL", True),
        (many(e_l=A * 1.11, rho=1.0), "INCONCLUSIVE_AT_CE_LEVEL", False),
        (many(rho=1.8), "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC", True),  # rho within 10% of Rho
        (many(rho=1.79), "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC", False),
    ],
)
def test_ce_label_table_at_and_around_the_constants(records, label, marginal):
    result = mt.ce_label(records)
    assert result["label"] == label and result["marginal"] is marginal, result


def both_outside(count):
    rows = [rec(id=str(i)) for i in range(200)]
    for row in rows[:count]:
        row.update(e_p=A * 3, e_l=A * 3, rho=1.0)
    return rows


@pytest.mark.parametrize(
    "count,label",
    [
        (100, "BOTH_OUTSIDE_A_AT_CE_LEVEL_SYNTHETIC"),
        (150, "BOTH_OUTSIDE_A_AT_CE_LEVEL_SYNTHETIC"),
        (99, "INCONCLUSIVE_AT_CE_LEVEL"),
    ],
)
def test_both_outside_needs_at_least_half_and_takes_precedence(count, label):
    assert mt.ce_label(both_outside(count))["label"] == label


def test_precedence_both_outside_beats_worse_and_worse_beats_inconclusive():
    rows = both_outside(100)
    rows[0]["rho"] = 10.0  # also a WORSE case: (a) still wins
    assert mt.ce_label(rows)["label"] == "BOTH_OUTSIDE_A_AT_CE_LEVEL_SYNTHETIC"
    rows = many(e_l=A * 2, rho=3.0)
    rows[1]["e_l"] = A * 1.5  # inconclusive-looking case alongside a worse one
    assert mt.ce_label(rows)["label"] == "LIGER_WORSE_AT_CE_LEVEL_SYNTHETIC"


def test_bothoutside_marginal_band_is_on_the_case_count():
    assert mt.ce_label(both_outside(105))["marginal"] is True
    assert mt.ce_label(both_outside(150))["marginal"] is False


def test_run_status_precedence_and_no_dropped_cases():
    rows = many()
    assert mt.run_status(rows, []) == "COMPLETE"
    assert mt.run_status(rows, ["bad hash"]) == "VOID"
    assert mt.run_status(rows[:199], []) == "INCOMPLETE"
    assert mt.run_status(rows[:199], ["bad hash"]) == "VOID"  # VOID is evaluated first
    duplicated = rows[:199] + [rows[0]]
    assert mt.run_status(duplicated, []) == "INCOMPLETE"
    assert mt.run_status([], []) == "INCOMPLETE"


def test_summary_carries_scope_and_never_labels_incomplete_or_void_runs():
    assert mt.summarize(many(150), [])["label"] == "NO_LABEL_INCOMPLETE"
    assert mt.summarize(many(), ["x"])["label"] == "NO_LABEL_VOID"
    full = mt.summarize(many(), [])
    assert "cannot turn attempt015/016 into PASS" in full["scope"]
    assert "does not imply the same bf16-propagated gradients" in full["scope"]
    assert full["constants"]["A"] == 17 * 2.0**-24 and full["constants"]["rho_limit"] == 2.0
    assert full["bf16_primary"]["statement"].startswith("fp32 accuracy parity")


def gate_records(high_flips=0, low_seeds=(5, 5, 5, 5), omit=None):
    rows = []
    for seed in mt.SEEDS:
        for shape in (0, 1):
            for path in mt.PATHS:
                rows.append(
                    rec(
                        kind="random",
                        scale=2.0,
                        shape=shape,
                        path=path,
                        seed=seed,
                        flips_pl=high_flips if (shape, path, seed) == (0, "mean", 0) else 0,
                    )
                )
        for index, (shape, path) in enumerate(
            [(2, "mean"), (2, "sum24"), (3, "mean"), (3, "sum24")]
        ):
            rows.append(
                rec(
                    kind="selfcons",
                    scale=1.5,
                    shape=shape,
                    path=path,
                    seed=seed,
                    flips_pl=1 if seed < low_seeds[index] else 0,
                )
            )
    return [r for r in rows if omit is None or not omit(r)]


def test_representativeness_gate_boundaries():
    assert mt.representativeness_gate(gate_records())["status"] == "REPRESENTATIVE"
    assert mt.representativeness_gate(gate_records(high_flips=1))["status"] == "NOT_REPRESENTATIVE"
    assert (
        mt.representativeness_gate(gate_records(low_seeds=(4, 5, 5, 5)))["status"]
        == "REPRESENTATIVE"
    )
    assert (
        mt.representativeness_gate(gate_records(low_seeds=(3, 5, 5, 5)))["status"]
        == "NOT_REPRESENTATIVE"
    )
    assert (
        mt.representativeness_gate(gate_records(low_seeds=(5, 5, 5, 0)))["status"]
        == "NOT_REPRESENTATIVE"
    )
    missing = gate_records(
        omit=lambda r: r["seed"] == 4 and r["kind"] == "selfcons" and r["shape"] == 3
    )
    assert mt.representativeness_gate(missing)["status"] == "NOT_EVALUATED_INCOMPLETE"


def test_explanatory_reading_follows_the_gate():
    gate = gate_records()
    filler = [rec(id=f"f{i}", kind="random", scale=3.0) for i in range(200 - len(gate))]
    for index, row in enumerate(gate):
        row["id"] = f"g{index}"
    summary = mt.summarize(gate + filler, [])
    assert summary["representativeness"]["status"] == "REPRESENTATIVE"
    assert summary["explanatory_reading"] == "allowed"
    failing = gate_records(high_flips=3)
    for index, row in enumerate(failing):
        row["id"] = f"g{index}"
    summary = mt.summarize(failing + filler, [])
    assert summary["representativeness"]["status"] == "NOT_REPRESENTATIVE"
    assert summary["explanatory_reading"] == "WITHHELD_NOT_REPRESENTATIVE"
    assert summary["label"] == summary["ce_level"]["label"]  # CE-level label is still reported
    assert mt.summarize(many(), [])["explanatory_reading"] == "WITHHELD_NOT_REPRESENTATIVE"


def test_anchor_summary_reports_spread_and_ratios():
    rows = [rec(e_l=2e-8, e_p=1e-8, id=str(i), d_pl=3e-8, d_p_alt=1e-8) for i in range(4)]
    anchor = mt.anchor_summary(rows)
    assert anchor["e_l_over_e_p"]["median"] == pytest.approx(2.0)
    assert anchor["d_pl_over_d_p_alt"]["max"] == pytest.approx(3.0)


# ------------------------------------------------------------------ kernel-ran and pins
def test_kernel_detection_uses_the_kernel_category_and_name():
    events = [{"cat": "kernel", "name": "liger_cross_entropy_kernel_0d1d"}]
    assert mt.liger_kernel_seen(events)
    assert not mt.liger_kernel_seen([{"cat": "cpu_op", "name": "liger_cross_entropy"}])
    assert not mt.liger_kernel_seen([{"cat": "kernel", "name": "cunn_SoftMaxForward"}])
    assert not mt.liger_kernel_seen([])


def test_pin_problems_lists_every_mismatch():
    ok = dict(mt.PINS)
    assert mt.pin_problems(ok) == []
    bad = dict(ok, torch="2.0.0", triton=None)
    assert len(mt.pin_problems(bad)) == 2


def test_counting_delegate_passes_through_and_counts():
    seen = []

    def real(*args, **kwargs):
        seen.append((args, kwargs))
        return 7

    delegate = mt.CountingDelegate(real)
    assert delegate(1, x=2) == 7 and delegate(3) == 7
    assert delegate.calls == 2 and delegate.__wrapped__ is real
    assert seen == [((1,), {"x": 2}), ((3,), {})]


def test_script_does_not_touch_the_runner_or_policy():
    source = (ROOT / "benchmarks/harness/h2_ce_microtest.py").read_text()
    for name in ("h2_ce_confirmation", "h2_gpu_correctness", "numerical_policy"):
        assert name not in source.replace("numerical_policy.json, the workset", "")
    assert "write_text" in source and "numerical_policy" not in source.split("def load_workset")[1]


# ------------------------------------------------------------------ the two-phase flow (subprocess)
CHILD = r"""
import json, sys, types
from pathlib import Path
import torch
import torch.nn.functional as F

mode, out, store = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
stock = F.cross_entropy
counter = {"n": 0}


def liger_cross_entropy(*args, **kwargs):
    if mode == "mutate_input":
        args[0].data[0, 0] += 1e-3  # in-place write over the input, like the real kernel
    result = stock(*args, **kwargs)
    if mode == "perturb":
        return result * (1 + 1e-3)  # a real, visible difference from the stock op
    if mode == "noisy":
        counter["n"] += 1
        if counter["n"] % 2 == 0:  # only the repeat of each pair is non-deterministic
            return result * (1 + 1e-4 * torch.rand(()))
    return result


liger_cross_entropy.__module__ = "liger_kernel.transformers.functional"


def apply_liger_kernel_to_llama(rope=True, cross_entropy=False, fused_linear_cross_entropy=True,
                                rms_norm=True, swiglu=True, model=None):
    assert (rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu) == (
        False, True, False, False, False), "Unexpected flags"
    if mode != "no_rebind":
        F.cross_entropy = liger_cross_entropy


def standin(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


standin("liger_kernel")
standin("liger_kernel.transformers")
standin("liger_kernel.transformers.monkey_patch",
        apply_liger_kernel_to_llama=apply_liger_kernel_to_llama)
patch = sys.modules["liger_kernel.transformers.monkey_patch"]
sys.modules["liger_kernel.transformers"].monkey_patch = patch

import h2_ce_microtest as mt

if mode == "prebound":
    F.cross_entropy = liger_cross_entropy
if mode == "reference_bad":
    mt.REFERENCE_TOLERANCE = -1.0
original_make = mt.make_inputs
calls = {}


def make_inputs(case, cfg, work):
    data = original_make(case, cfg, work)
    calls[case.id] = calls.get(case.id, 0) + 1
    if mode == "hash_mismatch" and calls[case.id] == 2:
        data["z"].view(torch.int16).view(-1)[0] ^= 1
        data["hashes"]["logits"] = mt.sha256_tensor(data["z"])
    return data


mt.make_inputs = make_inputs
original_run = mt.run_implementation


def run_implementation(master, labels, cfg, nib, counter=None, profiler=None):
    result = original_run(master, labels, cfg, nib, counter=counter, profiler=profiler)
    if mode == "binding_lost" and counter is not None:
        F.cross_entropy = stock  # something restores the stock function between cases
    if mode == "master_changed_p1" and counter is None:
        master.view(torch.int16).view(-1)[1] ^= 1
    if mode == "bf16_inconsistent" and counter is not None:
        result["bf16_consistent"] = False
    if mode == "master_changed" and counter is not None:
        master.view(torch.int16).view(-1)[1] ^= 1
    if mode == "p_modified" and counter is None:
        result["modified"] = True
    return result


mt.run_implementation = run_implementation
if mode == "double_call":
    import transformers.loss.loss_utils as loss_utils

    original_fixed = loss_utils.fixed_cross_entropy

    def twice(source, target, *args, **kwargs):
        loss_utils.nn.functional.cross_entropy(source.detach()[:1], target[:1])
        return original_fixed(source, target, *args, **kwargs)

    loss_utils.fixed_cross_entropy = twice
cfg = mt.Config(vocab=64, hidden=16, label_mod=64, device="cpu")
limit = 8 if mode != "full_grid" else None
result = mt.run(out, cfg, 0.0 if mode == "budget0" else 600.0, store, limit=limit)
print(json.dumps({"status": result["run_status"], "label": result["label"],
                  "void": result["void_reasons"], "cases": len(result["cases"])}))
"""


def run_child(tmp_path, mode):
    env = dict(
        os.environ,
        PYTHONPATH="src:benchmarks/harness",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    out = tmp_path / ("out_" + mode)
    done = subprocess.run(
        [sys.executable, "-B", "-c", CHILD, mode, str(out), str(tmp_path / ("store_" + mode))],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    summary = json.loads(done.stdout.strip().splitlines()[-1])
    return summary, json.loads((out / "result.json").read_text())


def test_two_phase_flow_completes_with_a_standin_and_records_every_check(tmp_path):
    summary, result = run_child(tmp_path, "ok")
    assert summary["status"] == "COMPLETE" and summary["cases"] == 8 and summary["void"] == []
    assert result["configuration"] == "TEST_CONFIGURATION_NOT_A_RESULT"
    for case in result["cases"]:
        assert case["e_p"] < 1e-5 and case["e_l"] < 1e-5  # stand-in delegates to the stock op
        assert case["l_input_modified_in_place"] is False
        assert case["p_repeat_equal"] and case["l_repeat_equal"]
        assert case["flips_pl"] >= 0 and math.isfinite(case["dh_pl"])
    assert result["receipt_start"]["device"] == "cpu" and "versions" in result["receipt_end"]


def test_in_place_writes_by_liger_are_recorded_and_never_touch_the_master(tmp_path):
    summary, result = run_child(tmp_path, "mutate_input")
    assert summary["status"] == "COMPLETE"  # clones protect the master; no VOID
    assert all(case["l_input_modified_in_place"] for case in result["cases"])
    assert all(case["e_p"] < 1e-5 for case in result["cases"])


@pytest.mark.parametrize(
    "mode,fragment",
    [
        ("no_rebind", "not rebound"),
        ("double_call", "call count"),
        ("prebound", "not stock before phase 1"),
        ("hash_mismatch", "input hash differs"),
        ("master_changed", "master logits changed during phase 2"),
        ("p_modified", "PyTorch modified its input"),
        ("binding_lost", "binding wrong before the case"),
        ("master_changed_p1", "master logits changed during phase 1"),
        ("bf16_inconsistent", "bf16 variant differs"),
        ("reference_bad", "reference self-check"),
    ],
)
def test_void_triggers_stop_the_run_with_no_label(tmp_path, mode, fragment):
    summary, result = run_child(tmp_path, mode)
    assert summary["status"] == "VOID" and summary["label"] == "NO_LABEL_VOID"
    assert any(fragment in reason for reason in summary["void"]), summary["void"]


def test_a_second_run_in_the_same_process_is_refused(tmp_path):
    mt._USED["value"] = False
    mt._USED["value"] = True
    try:
        with pytest.raises(ValueError, match="Fresh process"):
            mt.run(tmp_path / "again", SMALL, 10.0, tmp_path / "store")
    finally:
        mt._USED["value"] = False


def test_full_200_case_grid_runs_end_to_end_on_cpu_with_the_standin(tmp_path):
    summary, result = run_child(tmp_path, "full_grid")
    assert summary["status"] == "COMPLETE" and summary["cases"] == 200
    assert result["cases_expected"] == 200 and result["ce_level"]["label"] == summary["label"]
    assert result["representativeness"]["status"] in {"REPRESENTATIVE", "NOT_REPRESENTATIVE"}
    assert len({case["id"] for case in result["cases"]}) == 200


# ------------------------------------------------------------------ reviewer round 1 (C-1 .. C-3)
def test_profiler_subset_is_fixed_to_8_liger_and_2_pytorch_captures():
    liger, pytorch = mt.expected_profile_ids(mt.all_cases())
    assert len(liger) == 8 and len(pytorch) == 2
    assert {c.shape for c in mt.all_cases() if c.id in liger} == {0, 1, 2, 3}
    assert all(c.seed == 0 and c.regime == 0 for c in mt.all_cases() if c.id in liger)
    assert {c.path for c in mt.all_cases() if c.id in liger} == {"mean", "sum24"}
    by_id = {c.id: c for c in mt.all_cases()}
    assert sorted(by_id[i].shape for i in pytorch) == [0, 3]
    assert all(by_id[i].path == "mean" for i in pytorch)
    assert set(pytorch) <= set(liger)


def good_profiles(cases):
    liger, pytorch = mt.expected_profile_ids(cases)
    return {"liger": dict.fromkeys(liger, True), "pytorch": dict.fromkeys(pytorch, False)}


def test_profiler_checks_must_match_the_fixed_subset():
    cases = mt.all_cases()
    ids = {c.id for c in cases}
    assert mt.profiler_void_reasons(good_profiles(cases), cases, ids, ids) == []
    missing = good_profiles(cases)
    missing["liger"].pop(next(iter(missing["liger"])))
    assert any(
        "Liger profiler captures" in r for r in mt.profiler_void_reasons(missing, cases, ids, ids)
    )
    none = {"liger": {}, "pytorch": {}}  # nothing captured at all (a never-captured subset)
    assert len(mt.profiler_void_reasons(none, cases, ids, ids)) == 2
    extra = good_profiles(cases)
    extra["pytorch"][cases[5].id] = False
    assert any(
        "PyTorch profiler captures" in r for r in mt.profiler_void_reasons(extra, cases, ids, ids)
    )
    no_kernel = good_profiles(cases)
    no_kernel["liger"][next(iter(no_kernel["liger"]))] = False
    assert any(
        "no liger_cross_entropy kernel" in r
        for r in mt.profiler_void_reasons(no_kernel, cases, ids, ids)
    )
    seen = good_profiles(cases)
    seen["pytorch"][next(iter(seen["pytorch"]))] = True
    assert any(
        "seen in a PyTorch capture" in r for r in mt.profiler_void_reasons(seen, cases, ids, ids)
    )


def test_profiler_expectation_only_covers_cases_that_ran():
    cases = mt.all_cases()
    liger, pytorch = mt.expected_profile_ids(cases)
    ran_p2 = {c.id for c in cases[:1]}
    reasons = mt.profiler_void_reasons(
        {"liger": {}, "pytorch": dict.fromkeys(pytorch, False)},
        cases,
        {c.id for c in cases},
        ran_p2,
    )
    assert len(reasons) == 1 and "Liger profiler captures" in reasons[0]


def receipt_for(**extra):
    base = {"versions": dict(mt.PINS), "gpu_name": "NVIDIA GeForce RTX 4090", "gpu_processes": ""}
    base.update(extra)
    return base


REAL = mt.Config()
CPU_TEST = mt.Config(vocab=64, hidden=16, label_mod=64, device="cpu")


def test_preflight_void_reasons_branches():
    ok = receipt_for()
    assert mt.preflight_void_reasons(ok, REAL, "NVIDIA GeForce RTX 4090", None) == []
    wrong_pin = receipt_for(versions=dict(mt.PINS, torch="2.0.0"))
    assert any("torch" in r for r in mt.preflight_void_reasons(wrong_pin, REAL, None, None))
    missing_pin = receipt_for(versions=dict(mt.PINS, triton=None))
    assert any("triton" in r for r in mt.preflight_void_reasons(missing_pin, REAL, None, None))
    assert any("requires CUDA" in r for r in mt.preflight_void_reasons(ok, REAL, None, 8))
    cpu_real = mt.Config(device="cpu")
    assert any("requires CUDA" in r for r in mt.preflight_void_reasons(ok, cpu_real, None, None))
    assert any(
        "GPU class" in r
        for r in mt.preflight_void_reasons(
            receipt_for(gpu_name="A100"), REAL, "NVIDIA GeForce RTX 4090", None
        )
    )
    assert any(
        "Foreign GPU" in r
        for r in mt.preflight_void_reasons(
            receipt_for(gpu_processes="123, python"), REAL, None, None
        )
    )
    # Test configurations ignore the pins; a CPU device never checks GPU processes.
    assert (
        mt.preflight_void_reasons(receipt_for(versions={}, gpu_processes="x"), CPU_TEST, None, 8)
        == []
    )


def test_label_a_needs_both_implementations_outside_a():
    rows = [rec(id=str(i)) for i in range(200)]
    for row in rows[:150]:
        row.update(e_p=A * 3, e_l=1e-9)  # only P outside A
    assert mt.ce_label(rows)["label"] == "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC"
    for row in rows[:150]:
        row.update(e_p=1e-9, e_l=A * 3, rho=1.0)  # only L outside A, rho small
    assert mt.ce_label(rows)["label"] == "INCONCLUSIVE_AT_CE_LEVEL"


def test_rho_uses_the_unit_roundoff_floor():
    assert mt.rho_value(4e-8, 0.0) == pytest.approx(4e-8 / mt.U)
    assert mt.rho_value(4e-8, mt.U / 10) == pytest.approx(4e-8 / mt.U)
    assert mt.rho_value(4e-8, 2 * mt.U) == pytest.approx(
        2e-8 / mt.U * 1.0 * (4e-8 / (2 * mt.U)) / (2e-8 / mt.U)
    )


def test_reference_tolerance_constant_and_cosine_value():
    assert mt.REFERENCE_TOLERANCE == 1e-13
    ref = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert mt.metrics(torch.tensor([1.0, 1.0]), ref)["c"] == pytest.approx(2**-0.5)
    assert mt.metrics(torch.tensor([-1.0, 0.0]), ref)["c"] == pytest.approx(-1.0)


def test_alt_route_never_calls_cross_entropy(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("alt route called F.cross_entropy")

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", boom)
    z = torch.randn(6, 30)
    grad = mt.alt_gradient(z, torch.tensor([1, -100, 2, 3, -100, 4]), 4)
    assert grad.shape == z.shape and (grad[[1, 4]] == 0).all()


def test_ok_standin_has_no_bf16_flips_and_a_perturbed_one_does(tmp_path):
    _, ok = run_child(tmp_path, "ok")
    assert all(case["flips_pl"] == 0 for case in ok["cases"])
    _, perturbed = run_child(tmp_path, "perturb")
    assert all(case["flips_pl"] > 0 for case in perturbed["cases"])
    assert all(case["e_l"] > 1e-4 for case in perturbed["cases"])


def test_non_deterministic_repeats_are_reported_and_no_case_is_dropped(tmp_path):
    summary, result = run_child(tmp_path, "noisy")
    assert summary["status"] == "COMPLETE" and summary["cases"] == 8
    assert not any(case["l_repeat_equal"] for case in result["cases"])
    assert all(case["p_repeat_equal"] for case in result["cases"])
    assert (
        result["determinism"]["l_repeat_equal"] == 8
        and result["determinism"]["p_repeat_equal"] == 0
    )


def test_an_exhausted_phase_one_budget_is_incomplete_not_a_label(tmp_path):
    summary, result = run_child(tmp_path, "budget0")
    assert summary["status"] == "INCOMPLETE" and summary["label"] == "NO_LABEL_INCOMPLETE"
    assert summary["cases"] == 0 and result["void_reasons"] == []
    assert (
        list((tmp_path / "store_budget0").iterdir()) == []
    )  # phase 1 itself stopped on the budget
