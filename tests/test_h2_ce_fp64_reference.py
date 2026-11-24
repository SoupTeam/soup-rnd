"""CPU controls for the attempt018 fp64-reference arm: CE64, hooks, Rp, statistics, labels.

The CUDA path, peak memory and per-pass time are only observed at the run; everything below uses a
reduced-V configuration (TEST_CONFIGURATION_NOT_A_RESULT).
"""

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/harness"))
sys.path.insert(0, str(Path(__file__).parent))
import h2_ce_confirmation as runner  # noqa: E402
import h2_ce_ref64 as ce  # noqa: E402
import h2_ce_ref64_review as rv  # noqa: E402
import h2_operator_patch as patches  # noqa: E402

torch = pytest.importorskip("torch")
TARGET = torch.tensor([3, -100, 7, 49, -100, 0, 11, 11, -100, 20, 5, 2])


def fresh(vocab=50, real=False):
    state = ce.State(vocab, real)
    return ce.make_cross_entropy(state), state


def logits(seed=0, rows=12, vocab=50):
    torch.manual_seed(seed)
    return torch.randn(rows, vocab) * 3


# ------------------------------------------------------------------------------------ CE64
def test_ce64_matches_torch_fp64_autograd_including_ignored_rows():
    cross_entropy, state = fresh()
    x = logits().requires_grad_(True)
    loss = cross_entropy(x, TARGET)
    loss.backward()
    ref = x.detach().double().clone().requires_grad_(True)
    expected = torch.nn.functional.cross_entropy(ref, TARGET, ignore_index=-100, reduction="mean")
    expected.backward()
    assert loss.dtype == torch.float32 and loss.shape == torch.Size([])
    plain = torch.nn.functional.cross_entropy(x.detach(), TARGET, ignore_index=-100)
    assert plain.dtype == loss.dtype and plain.shape == loss.shape  # same type as PyTorch's
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), rel=1e-6)
    assert x.grad.dtype == torch.float32
    assert torch.equal(x.grad, ref.grad.float())  # fp32 rounding of the exact fp64 gradient
    assert (x.grad[TARGET == -100] == 0).all()
    assert state.calls == 1 and state.integrity == [{"ok": True, "rows": 12}]


def test_ce64_returns_the_fp32_rounding_not_a_direct_bf16_rounding():
    cross_entropy, _ = fresh()
    raw = logits(1).bfloat16().requires_grad_(True)
    out = cross_entropy(raw.float(), TARGET)
    out.backward()
    ref = raw.detach().float().double().requires_grad_(True)
    torch.nn.functional.cross_entropy(ref, TARGET, ignore_index=-100).backward()
    via_fp32 = ref.grad.float().bfloat16()
    assert raw.grad.dtype == torch.bfloat16 and torch.equal(raw.grad, via_fp32)
    assert raw.grad.dtype == torch.bfloat16 and out.dtype == torch.float32


@pytest.mark.parametrize(
    "case",
    [
        "bf16_input", "fp64_input", "three_d", "int32_target", "two_d_target", "sum", "none",
        "ignore_index", "weight", "label_smoothing", "size_average", "reduce", "vocab",
        "short_target", "all_ignored",
    ],
)  # fmt: skip
def test_ce64_refuses_everything_outside_its_specification(case):
    cross_entropy, state = fresh()
    x, t, kwargs = logits(), TARGET.clone(), {}
    if case == "bf16_input":
        x = x.bfloat16()
    elif case == "fp64_input":
        x = x.double()
    elif case == "three_d":
        x = x.view(1, 12, 50)
    elif case == "int32_target":
        t = t.int()
    elif case == "two_d_target":
        t = t.view(1, 12)
    elif case == "sum":
        kwargs["reduction"] = "sum"
    elif case == "none":
        kwargs["reduction"] = "none"
    elif case == "ignore_index":
        kwargs["ignore_index"] = 0
    elif case == "weight":
        kwargs["weight"] = torch.ones(50)
    elif case == "label_smoothing":
        kwargs["label_smoothing"] = 0.1
    elif case == "size_average":
        kwargs["size_average"] = True
    elif case == "reduce":
        kwargs["reduce"] = False
    elif case == "vocab":
        x = logits(vocab=51)
    elif case == "short_target":
        t = t[:5]
    elif case == "all_ignored":
        t = torch.full((12,), -100)
    with pytest.raises(ce.Ce64RefusedError):
        cross_entropy(x.requires_grad_(False), t, **kwargs)
    assert state.refusals or case == "all_ignored"


def test_ce64_real_configuration_requires_cuda_and_the_real_vocabulary():
    cross_entropy, state = fresh(vocab=ce.REAL_VOCAB, real=True)
    with pytest.raises(ce.Ce64RefusedError, match="128256"):
        cross_entropy(torch.randn(4, ce.REAL_VOCAB), torch.tensor([1, 2, 3, 4]))
    assert state.real and state.refusals


def test_ce64_result_is_unchanged_under_autocast():
    cross_entropy, _ = fresh()
    x = logits().requires_grad_(True)
    plain = cross_entropy(x, TARGET)
    plain.backward()
    plain_grad = x.grad.clone()
    x.grad = None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        auto = cross_entropy(x, TARGET)
        auto.backward()
    assert float(auto) == float(plain) and torch.equal(x.grad, plain_grad)


def test_ce64_detects_in_place_changes_and_counts_every_call():
    cross_entropy, state = fresh()
    x = logits().requires_grad_(True)
    target = TARGET.clone()
    loss = cross_entropy(x * 1.0, target)
    target[0] = 4  # the target changes between forward and backward
    with pytest.raises((ce.Ce64RefusedError, RuntimeError)):
        loss.backward()
    cross_entropy(logits(2), TARGET)
    assert state.calls == 2  # one call per loss evaluation, backward or not


def test_ce64_is_deterministic_and_does_not_touch_its_inputs():
    cross_entropy, _ = fresh()
    x = logits()
    before = ce.sha256_tensor(x), x._version
    grads = []
    for _ in range(2):
        leaf = x.clone().requires_grad_(True)
        cross_entropy(leaf, TARGET).backward()
        grads.append(leaf.grad.clone())
    assert torch.equal(grads[0], grads[1]) and (ce.sha256_tensor(x), x._version) == before


# ------------------------------------------------------------------------------------ Rp
def bf16_grad(seed=0, rows=12, vocab=64):
    torch.manual_seed(seed)
    return (torch.randn(rows, vocab) / 40).bfloat16()


def test_perturbation_seed_formula_and_grid_bounds():
    assert ce.perturbation_seed(0, 0) == 0 and ce.perturbation_seed(5, 2) == 5002
    assert ce.perturbation_seed(3, 1) == 3001
    for bad in ((6, 0), (0, 3), (-1, 0), (0, -1)):
        with pytest.raises(ValueError):
            ce.perturbation_seed(*bad)
    assert ce.GRID == (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4) and ce.SEEDS_PER_LEVEL == 3


@pytest.mark.parametrize("level", [1e-6, 3e-6, 1e-4, 3e-4, 0.01, 0.2])
def test_flip_count_is_exact_one_ulp_same_sign_finite_and_only_in_selected_rows(level):
    grad = bf16_grad()
    rows = torch.tensor([0, 2, 3, 5, 6, 8])
    out, info = ce.flip_one_ulp(grad, rows, level, seed=11)
    eligible = len(rows) * grad.shape[1]
    assert info["flipped"] == max(1, round(level * eligible)) and info["eligible_total"] == eligible
    before, after = grad.view(torch.int16), out.view(torch.int16)
    changed = (before != after).nonzero()
    assert len(changed) == info["flipped"]
    step = (before.int() & 0x7FFF) - (after.int() & 0x7FFF)
    assert (step[before != after].abs() == 1).all()
    assert ((before ^ after) & -32768 == 0).all()  # sign bit untouched
    assert torch.isfinite(out.float()).all() and (out.float() != 0).all()
    untouched = [r for r in range(grad.shape[0]) if r not in rows.tolist()]
    assert torch.equal(out[untouched], grad[untouched])


def test_flip_directions_are_random_and_the_seed_makes_it_reproducible():
    grad = bf16_grad(1, rows=40, vocab=200)
    rows = torch.arange(40)
    a, _ = ce.flip_one_ulp(grad, rows, 0.1, seed=3)
    b, _ = ce.flip_one_ulp(grad, rows, 0.1, seed=3)
    c, _ = ce.flip_one_ulp(grad, rows, 0.1, seed=4)
    assert torch.equal(a, b) and not torch.equal(a, c)
    moved = (a.float() - grad.float())[a != grad]
    assert (moved > 0).any() and (moved < 0).any()  # both directions occur


def test_flip_skips_zero_subnormal_and_maximal_elements():
    grad = torch.zeros(2, 8, dtype=torch.bfloat16)
    patterns = [0x0000, 0x0001, 0x007F, 0x7F7F, -32768, 0x0080, 0x7F7E, 0x3F80]
    grad.view(torch.int16)[0, :] = torch.tensor(patterns,
                                                dtype=torch.int16)  # fmt: skip
    rows = torch.tensor([0])
    with pytest.raises(ValueError, match="eligible"):
        ce.flip_one_ulp(grad, rows, 0.99, seed=1)  # would need 8 of the 3 eligible elements
    out, info = ce.flip_one_ulp(grad, rows, 0.25, seed=1)
    flipped = (grad.view(torch.int16) != out.view(torch.int16)).nonzero()[:, 1].tolist()
    assert info["flipped"] == 2 and set(flipped) <= {5, 6, 7}


def test_perturbation_inside_ce64_changes_exactly_the_requested_elements():
    cross_entropy, state = fresh(vocab=64)
    state.perturbation = {"level": 0.01, "seed": 9}
    x = logits(3, vocab=64).requires_grad_(True)
    base = x.detach().clone().requires_grad_(True)
    cross_entropy(x, TARGET).backward()
    plain, _ = fresh(vocab=64)
    plain_state = plain.ce64_state
    plain_state.perturbation = None
    plain(base, TARGET).backward()
    changed = int((x.grad.bfloat16() != base.grad.bfloat16()).sum())
    active = int((TARGET != -100).sum()) * 64
    assert changed == max(1, round(0.01 * active)) and state.flips[0]["flipped"] == changed
    assert torch.equal(x.grad, x.grad.bfloat16().float())  # exactly representable
    assert (x.grad[TARGET == -100] == 0).all()


def test_eligible_denominator_is_all_elements_of_non_ignored_rows():
    grad = bf16_grad(2, rows=12, vocab=64)
    rows = (TARGET != -100).nonzero()[:, 0]
    _, info = ce.flip_one_ulp(grad, rows, 3e-4, seed=1)
    assert info["eligible_total"] == int((TARGET != -100).sum()) * 64


# ------------------------------------------------------------------------------------ statistics
def synth_fit(beta=0.5, sigma=0.1, seed=1, intercept=-1.0):
    import random

    rng = random.Random(seed)
    levels = [f for f in rv.GRID for _ in range(rv.SEEDS)]
    dist = [math.exp(intercept + beta * math.log(f) + rng.gauss(0, sigma)) for f in levels]
    return rv.fit_loglog(levels, dist)


def manual_fit(alpha=0.0, beta=1.0, sigma=0.0, n=18, xbar=None, sxx=1.0):
    fit = {"alpha": alpha, "beta": beta, "sigma_hat": sigma, "n": n, "df": n - 2, "sxx": sxx,
           "xbar": math.log(1e-4) if xbar is None else xbar, "beta_ok": True}  # fmt: skip
    return fit


def test_t_quantile_and_cdf_match_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (3, 16, 30):
        for alpha in (0.05, 0.025, 0.1):
            assert rv.t_upper_quantile(alpha, df) == pytest.approx(
                scipy_stats.t.ppf(1 - alpha, df), abs=1e-9
            )
    assert rv.t_upper_quantile(0.05, 16) == pytest.approx(1.7458836762762504, abs=1e-9)
    for t in (-2.0, -0.3, 0.0, 1.0, 2.5):
        assert rv.t_cdf(t, 16) == pytest.approx(scipy_stats.t.cdf(t, 16), abs=1e-12)


def test_fit_recovers_a_known_line_and_matches_a_reference_ols():
    np = pytest.importorskip("numpy")
    fit = synth_fit(beta=0.7, sigma=0.2, seed=5, intercept=-2.0)
    levels = np.log(np.array([f for f in rv.GRID for _ in range(3)]))
    rng = __import__("random").Random(5)
    y = np.array([-2.0 + 0.7 * x + rng.gauss(0, 0.2) for x in levels])
    slope, icpt = np.polyfit(levels, y, 1)
    assert fit["beta"] == pytest.approx(slope, rel=1e-9) and fit["alpha"] == pytest.approx(
        icpt, rel=1e-9
    )
    resid = y - (icpt + slope * levels)
    assert fit["sigma_hat"] == pytest.approx(math.sqrt((resid**2).sum() / 16), rel=1e-9)
    assert fit["n"] == 18 and fit["df"] == 16 and fit["beta_ok"] is True


def test_fit_slope_test_branches_and_degenerate_inputs():
    flat = synth_fit(beta=0.0, sigma=0.5, seed=2)
    assert (
        flat["beta_ok"] is False or flat["t_beta"] > flat["t_critical"]
    )  # noise decides, no crash
    falling = synth_fit(beta=-0.8, sigma=0.05, seed=3)
    assert falling["beta"] < 0 and falling["beta_ok"] is False
    perfect = rv.fit_loglog(
        [f for f in rv.GRID for _ in range(3)], [f**0.5 for f in rv.GRID for _ in range(3)]
    )
    assert perfect["sigma_hat"] < 1e-12 and perfect["beta_ok"] is True
    with pytest.raises(ValueError, match="log G undefined"):
        rv.fit_loglog([1e-6, 1e-5, 1e-4], [1e-3, 0.0, 1e-2])
    with pytest.raises(ValueError):
        rv.fit_loglog([1e-6, 1e-6, 1e-6], [1e-3, 2e-3, 3e-3])
    with pytest.raises(ValueError):
        rv.fit_loglog([1e-6], [1e-3])


def test_t_critical_is_the_one_sided_five_percent_value_for_16_df():
    fit = synth_fit(beta=0.5, sigma=0.1)
    assert fit["t_critical"] == pytest.approx(1.7458836762762504, abs=1e-9) and fit["df"] == 16


def test_prediction_factor_is_minimal_at_the_mean_and_grows_away_from_it():
    fit = synth_fit()
    centre = math.exp(fit["xbar"])
    assert rv.prediction_factor(fit, centre) == pytest.approx(math.sqrt(1 + 1 / 18))
    assert rv.prediction_factor(fit, 1e-6) > rv.prediction_factor(fit, centre)
    assert rv.prediction_factor(fit, 3e-4) > rv.prediction_factor(fit, centre)
    assert 1.02 < rv.prediction_factor(fit, 1e-6) < 1.15


def test_level_summary_is_report_only_and_het_is_the_sd_ratio():
    levels = [f for f in rv.GRID for _ in range(3)]
    dist = [math.exp(math.log(f) * 0.5 + d) for f in rv.GRID for d in (-0.1, 0.0, 0.1)]
    summary = rv.level_summary(levels, dist)
    assert set(summary["levels"]) == set(rv.GRID) and summary["het"] == pytest.approx(1.0)
    dist2 = (
        dist[:3] + [math.exp(0.5 * math.log(1e-6) + d * 10) for d in (-0.1, 0.0, 0.1)] + dist[6:]
    )
    assert rv.level_summary(levels, dist2)["het"] == pytest.approx(10.0, rel=1e-6)


def test_fp32_spacing_and_loss_agreement_boundaries():
    assert rv.fp32_spacing(1.0) == 2.0**-23 and rv.fp32_spacing(10.0) == 2.0**-20
    assert rv.fp32_spacing(15.99) == 2.0**-20 and rv.fp32_spacing(16.0) == 2.0**-19
    base = 10.0
    step = rv.fp32_spacing(base)
    assert rv.loss_agrees(base, base) and rv.loss_agrees(base, base + 2 * step)
    assert not rv.loss_agrees(base, base + 3 * step) and rv.loss_agrees(base, base - 2 * step)


# ------------------------------------------------------------------------------------ arm status
def test_arm_status_branches():
    fit = manual_fit()
    assert rv.arm_status(0.0, 0.0, fit)["status"] == "NOISE_CONSISTENT"
    assert rv.arm_status(1e-9, 0.0, fit)["status"] == "INVALID_PHI0"
    assert rv.arm_status(0.1, 3e-4, fit)["status"] != "NOT_CALIBRATABLE"  # boundary inclusive
    assert rv.arm_status(0.1, 3.0000001e-4, fit)["status"] == "NOT_CALIBRATABLE"
    assert rv.arm_status(0.1, 1e-5, None)["status"] == "NOT_CALIBRATABLE"
    zero = rv.arm_status(0.0, 1e-5, fit)
    assert zero["status"] == "NOISE_CONSISTENT" and zero["detectable_factor"] is not None


def test_arm_status_bound_is_inclusive_and_strict_above():
    fit = manual_fit(alpha=0.0, beta=1.0, sigma=0.0)  # bound = log(phi)
    phi = 1e-5
    assert rv.arm_status(phi, phi, fit)["status"] == "NOISE_CONSISTENT"  # log d == bound
    assert rv.arm_status(phi * (1 + 1e-9), phi, fit)["status"] == "EXCESS"
    assert rv.arm_status(phi * (1 - 1e-9), phi, fit)["status"] == "NOISE_CONSISTENT"


def test_tiny_phi_is_evaluated_at_the_lowest_grid_level_and_marked_marginal():
    fit = manual_fit(alpha=-1.0, beta=0.5, sigma=0.1)
    tiny = rv.arm_status(1e-3, 5e-7, fit)
    low = rv.arm_status(1e-3, 1e-6, fit)
    assert tiny["tiny_phi"] and tiny["marginal"]
    assert tiny["bound"] == low["bound"] and tiny["pf"] == low["pf"]
    assert tiny["detectable_factor"] == low["detectable_factor"] and not low["tiny_phi"]


def test_distance_marginal_band_is_ten_percent_of_k_sigma_pf():
    fit = manual_fit(alpha=0.0, beta=1.0, sigma=0.1)
    phi = 1e-4
    first = rv.arm_status(phi, phi, fit)
    margin = rv.K * fit["sigma_hat"] * first["pf"]
    bound = math.log(phi) + margin
    inside = rv.arm_status(math.exp(bound - 0.09 * margin), phi, fit)
    edge_out = rv.arm_status(math.exp(bound - 0.11 * margin), phi, fit)
    over = rv.arm_status(math.exp(bound + 0.05 * margin), phi, fit)
    assert inside["marginal"] and not edge_out["marginal"]
    assert over["status"] == "EXCESS" and over["marginal"]


def test_detectable_factor_is_per_arm_at_its_own_phi_and_k_is_three():
    fit = synth_fit(beta=0.5, sigma=0.3)
    a = rv.arm_status(1e-3, 1e-5, fit)
    b = rv.arm_status(1e-3, 2e-4, fit)
    assert rv.K == 3.0 and rv.Q == 1.5 and rv.LOW_POWER_FACTOR == 5.0
    assert a["detectable_factor"] == pytest.approx(
        math.exp(3 * fit["sigma_hat"] * rv.prediction_factor(fit, 1e-5))
    )
    assert a["detectable_factor"] != b["detectable_factor"]


@pytest.mark.parametrize(
    "phi_l,phi_p,liger,pytorch,marginal",
    [
        (1.0, 1.5, True, False, True),  # exactly phi_P / Q is inclusive; r = 1/Q
        (1.0, 1.4999, False, False, True),
        (1.5, 1.0, False, True, True),  # exactly phi_L / Q
        (1.4999, 1.0, False, False, True),
        (0.0, 1.0, True, False, False),  # phi_L = 0 with phi_P > 0 holds for the Liger side
        (1.0, 0.0, False, True, False),
        (1.0, 1.0, False, False, False),
        (1.0, 1.8, True, False, False),
        (0.5, 1.5, True, False, False),
    ],
)  # fmt: skip
def test_flip_ratio_sides_and_their_marginal_band(phi_l, phi_p, liger, pytorch, marginal):
    assert rv.liger_side(phi_l, phi_p) is liger and rv.pytorch_side(phi_p, phi_l) is pytorch
    assert rv.ratio_marginal(phi_l, phi_p) is marginal


def test_ratio_marginal_band_edges():
    inv, q = 1 / rv.Q, rv.Q
    assert rv.ratio_marginal(inv * 1.0999, 1.0) and not rv.ratio_marginal(inv * 1.11, 1.0)
    assert rv.ratio_marginal(inv * 0.9001, 1.0) and not rv.ratio_marginal(inv * 0.89, 1.0)
    assert rv.ratio_marginal(q * 1.0999, 1.0) and not rv.ratio_marginal(q * 1.11, 1.0)
    assert rv.ratio_marginal(q * 0.9001, 1.0) and not rv.ratio_marginal(q * 0.89, 1.0)


# ------------------------------------------------------------------------------------ label table
GOOD = synth_fit(beta=0.5, sigma=0.05, seed=4, intercept=-1.0)  # small sigma: detectable factor < 5


def noise_d(phi, fit=GOOD, scale=1.0):
    return scale * math.exp(fit["alpha"] + fit["beta"] * math.log(phi))


def probe(phi_p=1e-4, phi_l=2e-5, d_p=None, d_l=None, fit=GOOD, identical=False, zero=False):
    return {"phi_p": phi_p, "phi_l": phi_l,
            "d_p": (noise_d(phi_p, fit) if fit else 1.0) if d_p is None else d_p,
            "d_l": (noise_d(phi_l, fit) if fit else 1.0) if d_l is None else d_l,
            "pl_identical": identical, "zero_pass": zero, "fit": fit}  # fmt: skip


def label(probes, void=(), incomplete=(), controls=()):
    return rv.classify(void_reasons=list(void), incomplete_reasons=list(incomplete),
                       failed_controls=list(controls), probes=probes)  # fmt: skip


def both(**kwargs):
    return {2: probe(**kwargs), 3: probe(**kwargs)}


def test_ordered_label_precedence():
    assert label(both(), void=["x"], incomplete=["y"], controls=["z"])["label"] == "VOID"
    assert label(both(), incomplete=["y"], controls=["z"])["label"] == "INCOMPLETE"
    assert label(both(identical=True), controls=["z"])["label"] == "REFERENCE_ARM_INVALID"
    assert label(both(identical=True, phi_p=1e-3))["label"] == "INCONCLUSIVE_NO_REPRODUCTION"
    assert label(both(phi_p=1e-3))["label"] == "NOT_CALIBRATABLE"
    assert label(both(zero=True, fit=None))["label"] == "NOT_CALIBRATABLE"
    assert label(both())["row"] == 6
    assert [label(both(), void=["x"])["row"], label(both(), incomplete=["y"])["row"]] == [1, 2]


@pytest.mark.parametrize("which", [2, 3])
def test_no_reproduction_in_either_decisive_probe(which):
    probes = both()
    probes[which] = probe(identical=True)
    assert label(probes)["label"] == "INCONCLUSIVE_NO_REPRODUCTION"


def test_row_5_triggers_and_the_exact_phi_edge():
    assert label(both(phi_p=3e-4, phi_l=1e-4))["row"] != 5
    assert label(both(phi_p=3.0000001e-4, phi_l=1e-4))["label"] == "NOT_CALIBRATABLE"
    assert label(both(phi_p=1e-4, phi_l=3.0000001e-4))["label"] == "NOT_CALIBRATABLE"
    flat = dict(GOOD, beta_ok=False)
    assert label(both(fit=flat))["label"] == "NOT_CALIBRATABLE"
    mixed = {2: probe(), 3: probe(fit=flat)}
    assert label(mixed)["label"] == "NOT_CALIBRATABLE"
    assert label({2: probe(zero=True, fit=None), 3: probe()})["label"] == "NOT_CALIBRATABLE"


def test_control_7_phi_zero_with_a_distance_is_reference_arm_invalid():
    probes = both(phi_p=1e-4, phi_l=0.0, d_l=1e-9)
    result = label(probes)
    assert result["label"] == "REFERENCE_ARM_INVALID" and "phi=0" in result["reasons"][0]
    clean = both(phi_p=1e-4, phi_l=0.0, d_l=0.0)
    assert label(clean)["label"].startswith("LIGER_CLOSER_BY_FLIPS_NOISE_CONSISTENT")


def test_closeness_rows_and_their_inclusive_boundaries():
    liger = label(both(phi_p=1.5e-5, phi_l=1e-5))
    assert liger["label"].startswith("LIGER_CLOSER_BY_FLIPS_NOISE_CONSISTENT")
    off = label(both(phi_p=1.4999e-5, phi_l=1e-5))
    assert off["label"].startswith("INDISTINGUISHABLE_NOISE_CONSISTENT")
    pytorch = label(both(phi_p=1e-5, phi_l=1.5e-5))
    assert pytorch["label"].startswith("PYTORCH_CLOSER_BY_FLIPS_NOISE_CONSISTENT")
    assert label(both(phi_p=1e-5, phi_l=1e-5))["label"].startswith(
        "INDISTINGUISHABLE_NOISE_CONSISTENT"
    )


def test_opposite_probes_and_one_probe_only_are_indistinguishable():
    opposite = {2: probe(phi_p=1e-4, phi_l=2e-5), 3: probe(phi_p=2e-5, phi_l=1e-4)}
    assert label(opposite)["label"].startswith("INDISTINGUISHABLE_NOISE_CONSISTENT")
    one_probe = {2: probe(phi_p=1e-4, phi_l=2e-5), 3: probe(phi_p=1e-5, phi_l=1e-5)}
    assert label(one_probe)["label"].startswith("INDISTINGUISHABLE_NOISE_CONSISTENT")


def test_d_equal_zero_with_positive_phi_is_noise_consistent():
    result = label(both(d_p=0.0, d_l=0.0))
    assert result["row"] in (6, 7) and result["excess_arms"] == []


@pytest.mark.parametrize(
    "excess_p,excess_l,name",
    [(True, False, "PYTORCH"), (False, True, "LIGER"), (True, True, "BOTH")],
)
def test_excess_names_the_arms(excess_p, excess_l, name):
    probes = both(
        d_p=noise_d(1e-4) * 1e3 if excess_p else None, d_l=noise_d(2e-5) * 1e3 if excess_l else None
    )
    result = label(probes)
    assert result["row"] == 8 and result["label"] == "EXCESS_OVER_FLIP_NOISE_" + name


def test_excess_in_different_probes_names_both_arms():
    probes = {2: probe(d_p=noise_d(1e-4) * 1e3), 3: probe(d_l=noise_d(2e-5) * 1e3)}
    assert label(probes)["label"].startswith("EXCESS_OVER_FLIP_NOISE_BOTH")


def test_excess_outranks_closeness_but_not_the_earlier_rows():
    excess = both(phi_p=1.5e-5, phi_l=1e-5, d_l=noise_d(1e-5) * 1e3)
    assert label(excess)["row"] == 8
    assert label({2: probe(identical=True), 3: excess[3]})["row"] == 4


def sloppy_fit(sigma=0.5):
    return manual_fit(alpha=-1.0, beta=0.5, sigma=sigma, xbar=math.log(1e-4), sxx=30.0)


def test_low_power_is_strictly_above_five_and_comes_before_marginal(monkeypatch):
    fit = sloppy_fit(sigma=0.5)
    factor = rv.arm_status(1e-3, 1e-5, fit)["detectable_factor"]
    assert factor > 5
    probes = both(
        phi_p=1.5e-5, phi_l=1e-5, fit=fit, d_p=noise_d(1.5e-5, fit), d_l=noise_d(1e-5, fit)
    )
    result = label(probes)
    assert result["label"] == "LIGER_CLOSER_BY_FLIPS_NOISE_CONSISTENT_LOW_POWER_MARGINAL"
    assert result["suffixes"] == ["_LOW_POWER", "_MARGINAL"]
    top = max(rv.arm_status(0.0, p, fit)["detectable_factor"] for p in (1.5e-5, 1e-5))
    monkeypatch.setattr(rv, "LOW_POWER_FACTOR", top)  # exactly equal: not low power
    assert "_LOW_POWER" not in label(probes)["label"]
    monkeypatch.setattr(rv, "LOW_POWER_FACTOR", top * (1 - 1e-12))
    assert "_LOW_POWER" in label(probes)["label"]


def test_low_power_triggers_on_any_arm_probe_and_marginal_alone():
    fit = sloppy_fit(sigma=0.5)
    mixed = {
        2: probe(),
        3: probe(
            phi_p=1.5e-5, phi_l=1e-5, fit=fit, d_p=noise_d(1.5e-5, fit), d_l=noise_d(1e-5, fit)
        ),
    }
    assert "_LOW_POWER" in label(mixed)["label"]
    tiny = both(phi_p=1.2e-6, phi_l=5e-7, d_l=1e-12)
    assert label(tiny)["label"].endswith("_MARGINAL") and "_LOW_POWER" not in label(tiny)["label"]
    ratio_edge = both(phi_p=1.5e-5, phi_l=1e-5)
    assert label(ratio_edge)["label"].endswith("_MARGINAL")  # r = 1/Q exactly


def test_excess_label_carries_marginal_but_never_low_power():
    fit = sloppy_fit(sigma=0.5)
    probes = both(fit=fit, d_p=noise_d(1e-4, fit) * 1e6)
    assert label(probes)["label"] == "EXCESS_OVER_FLIP_NOISE_PYTORCH"
    near = both(d_p=math.exp(rv.arm_status(1e-3, 1e-4, GOOD)["bound"] + 1e-9))
    assert label(near)["label"] == "EXCESS_OVER_FLIP_NOISE_PYTORCH_MARGINAL"


def test_every_label_text_carries_the_traveling_text():
    for result in (label(both()), label(both(), void=["x"]), label(both(), incomplete=["y"])):
        assert "Constants fixed after attempts 015-017 were seen" in result["text"]
        assert "BLOCKED outcomes of attempts 015/016 are unchanged" in result["text"]
        assert "exact-CE gradient with the rest of the model in bf16/NF4" in result["text"]
        assert "detectable_factor" in result["power_text"]


# ------------------------------------------------------------------------------------ controls
def good_controls():
    probes = range(4)
    return {
        "repeat_identical": dict.fromkeys(probes, True),
        "losses": {p: dict.fromkeys(probes, 10.0) for p in ("baseline_probes", "ce_only_probes",
                                                           "ref64_probes")},
        "forward_hashes": {i: {"P": "a", "L": "a", "R": "a"} for i in probes},
        "ce64_calls": {"baseline_probes": 0, "ce_only_probes": 0, "ref64_probes": 4,
                       "ref64_probes_repeat": 4,
                       "ref64_perturbed_probe02": 18, "ref64_perturbed_probe03": 18},
        "ce64_expected": {"baseline_probes": 0, "ce_only_probes": 0, "ref64_probes": 4,
                          "ref64_probes_repeat": 4,
                          "ref64_perturbed_probe02": 18, "ref64_perturbed_probe03": 18},
        "integrity_ok": {"ref64_probes": True, "ref64_probes_repeat": True},
        "liger_events": {"baseline_probes": 0, "ce_only_probes": 1, "ref64_probes": 0,
                         "ref64_probes_repeat": 0, "ref64_perturbed_probe02": 0,
                         "ref64_perturbed_probe03": 0},
        "receipts_equal": True, "identity_equal": True,
    }  # fmt: skip


def test_controls_pass_on_good_data_and_each_one_can_fail():
    assert rv.evaluate_controls(good_controls())["failed"] == []
    cases = {
        "1": lambda d: d["repeat_identical"].update({2: False}),
        "2": lambda d: d["losses"]["ce_only_probes"].update({1: 10.5}),
        "3": lambda d: d["forward_hashes"][3].update({"L": "b"}),
        "3b": lambda d: d.update(identity_equal=False),
        "4": lambda d: d["integrity_ok"].update({"ref64_probes": False}),
        "5a": lambda d: d["ce64_calls"].update({"ref64_probes": 3}),
        "5b": lambda d: d["ce64_calls"].update({"baseline_probes": 1}),
        "5c": lambda d: d["liger_events"].update({"ref64_probes": 2}),
        "5d": lambda d: d["liger_events"].update({"ce_only_probes": 0}),
        "5e": lambda d: d["liger_events"].update({"baseline_probes": 1}),
        "6": lambda d: d.update(receipts_equal=False),
    }
    for name, mutate in cases.items():
        data = good_controls()
        mutate(data)
        failed = rv.evaluate_controls(data)["failed"]
        assert failed and failed[0].startswith("control " + name[0]), (name, failed)


def test_loss_agreement_uses_two_fp32_spacings_for_the_loss_value():
    data = good_controls()
    step = rv.fp32_spacing(10.0)
    data["losses"]["baseline_probes"][0] = 10.0 + 2 * step
    assert rv.evaluate_controls(data)["failed"] == []
    data["losses"]["baseline_probes"][0] = 10.0 + 2.5 * step
    assert any("control 2" in f for f in rv.evaluate_controls(data)["failed"])


def test_history_control_is_descriptive_only():
    assert rv.describe_history(None, {})["status"] == "NOT_PROVIDED"
    assert rv.describe_history({"a": 1}, {"a": 1, "b": 2})["equal"] == {"a": True, "b": False}


# ---------------------------------------------------------------- hooks on a real model
HOOK_CHILD = r"""
import json, sys
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import h2_ce_ref64 as ce

torch.manual_seed(0)
config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                     num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
                     tie_word_embeddings=False)
model = LlamaForCausalLM(config)
ids = torch.randint(1, 64, (1, 12))
labels = ids.clone()
labels[0, :4] = -100


def step(hooks=None, key=None):
    model.zero_grad(set_to_none=True)
    if hooks:
        hooks.begin(key)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    if hooks:
        hooks.end()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


stock_loss, stock_grads = step()
wrapper = ce.install(vocab=64, real=False)
loss_a, grads_a = step()
hooks = ce.LmHeadHooks(model)
loss_b, grads_b = step(hooks, "probe_00")
loss_c, grads_c = step(hooks, "probe_01")
errors = []
try:
    hooks.begin("probe_00")
except ValueError as error:
    errors.append("reuse")
hooks.remove()
loss_d, grads_d = step()
dlog = hooks.dlogits
print(json.dumps({
    "same_with_and_without_hooks": all(
        torch.equal(grads_a[n], grads_b[n]) and torch.equal(grads_a[n], grads_d[n])
        for n in grads_a),
    "repeat_identical": all(torch.equal(grads_b[n], grads_c[n]) for n in grads_a),
    "loss_equal": loss_a == loss_b == loss_c == loss_d,
    "close_to_stock": abs(stock_loss - loss_a) < 1e-4 and all(
        torch.allclose(stock_grads[n], grads_a[n], rtol=0.05, atol=1e-4) for n in stock_grads),
    "calls": wrapper.ce64_state.calls, "integrity": len(wrapper.ce64_state.integrity),
    "sha_keys": sorted(hooks.sha), "dlogits_keys": sorted(dlog),
    "dlogits_dtype": str(dlog["probe_00"].dtype),
    "dlogits_shape": list(dlog["probe_00"].shape),
    "dlogits_same": bool(torch.equal(dlog["probe_00"], dlog["probe_01"])),
    "sha_same": hooks.sha["probe_00"] == hooks.sha["probe_01"], "reuse_refused": errors,
    "binding_module": torch.nn.functional.cross_entropy.__module__,
}))
"""


def run_python(code, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "benchmarks/harness"), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")  # fmt: skip
    done = subprocess.run([sys.executable, "-B", "-c", code, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)  # fmt: skip
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_ce64_is_reached_through_the_real_loss_function_and_hooks_change_nothing():
    result = run_python(HOOK_CHILD)
    assert result["binding_module"] == "h2_ce_ref64"
    assert (
        result["same_with_and_without_hooks"]
        and result["repeat_identical"]
        and result["loss_equal"]
    )
    assert result["close_to_stock"]
    assert result["calls"] == 4 and result["integrity"] == 4  # one per loss evaluation, all intact
    assert result["sha_keys"] == ["probe_00", "probe_01"] == result["dlogits_keys"]
    assert result["dlogits_dtype"] == "torch.bfloat16" and result["dlogits_shape"] == [1, 12, 64]
    assert result["dlogits_same"] and result["sha_same"] and result["reuse_refused"] == ["reuse"]


def test_hooks_store_clones_and_return_none(monkeypatch):
    out = torch.randn(1, 4, 8, requires_grad=True)
    holder = {}

    class Head(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = Head()

    model = Model()
    hooks = ce.LmHeadHooks(model)
    hooks.begin("probe_00")
    result = model.lm_head(out)
    result.sum().backward()
    stored = hooks.dlogits["probe_00"]
    assert torch.equal(stored, torch.ones(1, 4, 8)) and stored.data_ptr() != out.grad.data_ptr()
    stored.zero_()
    assert torch.equal(out.grad, torch.full((1, 4, 8), 2.0))  # the live gradient is untouched
    holder["calls"] = hooks.calls
    assert holder["calls"] == {"probe_00": 1}
    with pytest.raises(ValueError, match="More than one"):
        hooks.begin("probe_01")
        model.lm_head(out)
        model.lm_head(out)
    with pytest.raises(ValueError, match="Exactly one lm_head"):
        ce.LmHeadHooks(torch.nn.Linear(2, 2))


def test_hooks_without_gradient_storage_only_record_hashes():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 4)

    model = Model()
    hooks = ce.LmHeadHooks(model, keep_gradients=False)
    hooks.begin("pass_0_0")
    model.lm_head(torch.randn(2, 4, requires_grad=True)).sum().backward()
    assert "pass_0_0" in hooks.sha and hooks.dlogits == {}


# ------------------------------------------------------------------------------------ runner wiring
def test_new_phases_series_and_artifacts_are_registered():
    arms = runner.PHASE_ARMS
    assert arms["ref64_probes"] == arms["ref64_probes_repeat"] == ("ref64", "probes")
    assert (
        arms["ref64_perturbed_probe02"] == arms["ref64_perturbed_probe03"] == ("ref64", "perturbed")
    )
    assert runner.SERIES["ref64"] == runner.REF64_PHASES == rv.PHASES
    assert len(runner.REF64_PHASES) == 6 and runner.PERTURBED_PROBE == {
        "ref64_perturbed_probe02": 2, "ref64_perturbed_probe03": 3}  # fmt: skip
    assert runner.series_name({"series": "ref64"}) == "ref64"
    for name in ("benchmarks/harness/h2_ce_ref64.py", "benchmarks/harness/h2_ce_ref64_review.py",
                 "benchmarks/h2/CE_FP64_REFERENCE_ADDENDUM.txt"):  # fmt: skip
        assert name in runner.ARTIFACTS and (ROOT / name).exists()
    assert runner.PHASES == (
        "baseline_probes",
        "ce_only_probes",
        "baseline_training",
        "ce_only_training",
    )


def test_ref64_arm_is_not_in_the_diagnostic_matrix_and_has_no_flags():
    assert "ref64" not in patches.ARMS and patches.patch_flags("ref64") == dict.fromkeys(
        patches.FLAGS, False
    )
    patches.validate_counts("ref64", {"class": 0, "instance": 0})
    with pytest.raises(ValueError):
        patches.validate_counts("ref64", {"class": 1, "instance": 1})
    with pytest.raises(ValueError):
        patches.patch_flags("ref65")
    patches.validate_counts("ce_only", {"class": 1, "instance": 1})
    patches.validate_counts("baseline", {"class": 0, "instance": 0})


def test_patch_controller_verify_requires_the_ce64_wrapper_only_for_ref64():
    source = (ROOT / "benchmarks/harness/h2_operator_patch.py").read_text()
    assert "reference_ce" in source and "'h2_ce_ref64'" in source
    control = patches.PatchController.__new__(patches.PatchController)
    control.arm, control.flags, control.counts = (
        "ref64",
        patches.patch_flags("ref64"),
        {"class": 0, "instance": 0},
    )
    control.original, control.treatment = {key: object() for key in patches.FLAGS}, {}
    control.reference_ce = None
    with pytest.raises(ValueError, match="CE64 wrapper"):
        control.verify(object())
    control.reference_ce = lambda: None  # some other callable is refused as well
    with pytest.raises(ValueError, match="CE64 wrapper"):
        control.verify(object())
    with pytest.raises(ValueError, match="class patch"):
        control.class_patch("name")


def test_run_child_installs_ce64_only_for_the_ref64_arm_and_hooks_only_in_that_series():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(runner.run_child).lstrip())
    source = inspect.getsource(runner.run_child)
    installs = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "install"]  # fmt: skip
    assert len(installs) == 1
    guard = next(
        n for n in ast.walk(tree) if isinstance(n, ast.If) and installs[0] in list(ast.walk(n))
    )
    assert "'ref64'" in ast.unparse(guard.test) and "arm" in ast.unparse(guard.test)
    assert "if ref64_series:" in source and "LmHeadHooks" in source and "set_perturbation" in source
    assert "PERTURBED_PROBE[phase]" in source and "pass_0_0" in source
    assert source.count("collect(") >= 2 and "collect_probe(" in source


def test_perturbation_children_run_18_passes_with_the_trace_on_the_first():
    import inspect

    source = inspect.getsource(runner.run_child)
    assert (
        "range(len(h2_ce_ref64.GRID))" in source and "range(h2_ce_ref64.SEEDS_PER_LEVEL)" in source
    )
    assert "(level_index, seed_index) == (0, 0)" in source
    assert len(ce.GRID) * ce.SEEDS_PER_LEVEL == rv.PASSES == 18


def test_ref64_record_and_hook_evidence(tmp_path):
    assert runner.ref64_record(None) == {
        "ce64_calls": 0,
        "integrity_ok": True,
        "refusals": [],
        "flips": [],
    }
    cross_entropy, state = fresh()
    x = logits().requires_grad_(True)
    cross_entropy(x, TARGET).backward()
    record = runner.ref64_record(cross_entropy)
    assert record["ce64_calls"] == 1 and record["integrity_ok"] and record["config"] == "test"
    cross_entropy(logits(2), TARGET)  # a forward without a backward is not an intact evaluation
    assert runner.ref64_record(cross_entropy)["integrity_ok"] is False

    class Hooks:
        sha = {"probe_00": "ab"}
        dlogits = {"probe_00": torch.zeros(1, 2, 4, dtype=torch.bfloat16)}

    runner.write_hook_evidence(tmp_path / "hooks", Hooks())
    from safetensors.torch import load_file

    assert json.loads((tmp_path / "hooks/lm_head_sha256.json").read_text()) == {"probe_00": "ab"}
    assert (
        load_file(str(tmp_path / "hooks/probe_00.safetensors"))["dlogits"].dtype == torch.bfloat16
    )


VOID_CAUSES = [
    "ValueError: CE64 refused: input must be a 2-D fp32 tensor",
    "ValueError: CE64 refused: input or target changed during the evaluation",
    "OutOfMemoryError: CUDA out of memory. Tried to allocate 1.5 GiB",
    "ValueError: Nonfinite probe loss",
    "ValueError: Missing/nonfinite gradient: base_model.model.layers.0",
    "ValueError: Unexpected global patch binding",
    "ValueError: ref64 arm requires the installed CE64 wrapper",
    "ValueError: Unexpected actual model forward",
    "ValueError: Frozen study pin differs: versions",
    "ValueError: Missing or changed CE source artifact",
    "ValueError: Initial model differs from historical pinned state",
    "ValueError: One Ampere+ BF16 CUDA GPU with >=16GiB required",
    "ValueError: Child contract differs from whole-series authority",
    "ValueError: More than one lm_head forward for probe_00",
    "RuntimeError: some unforeseen failure",
    "",
]


@pytest.mark.parametrize("error", VOID_CAUSES)
def test_every_recorded_child_error_except_the_deadline_is_void(tmp_path, error):
    (tmp_path / "evidence.json").write_text(json.dumps({"error": error, "status": "FAILED"}))
    assert runner.ref64_failure_status(tmp_path) == "VOID"


def test_missing_unreadable_or_deadline_evidence_is_incomplete(tmp_path):
    assert runner.ref64_failure_status(tmp_path / "missing") == "INCOMPLETE"
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "evidence.json").write_text("{not json")
    assert runner.ref64_failure_status(broken) == "INCOMPLETE"
    timed = tmp_path / "timed"
    timed.mkdir()
    (timed / "evidence.json").write_text(
        json.dumps({"error": "TimeoutError: CE child deadline reached"})
    )
    assert runner.ref64_failure_status(timed) == "INCOMPLETE"
    no_error = tmp_path / "noerror"
    no_error.mkdir()
    (no_error / "evidence.json").write_text(json.dumps({"status": "FAILED", "controls_error": "x"}))
    assert runner.ref64_failure_status(no_error) == "VOID"


# ------------------------------------------------------- offline review on synthetic evidence
NAMES = [f"gradient.layer{i}.lora_B.weight" for i in range(224)]
VOCAB, ROWS = 50000, 24
ACTIVE = 6  # labels[0, 1:7] are targets, so six rows are non-ignored: 300000 active elements
ACTIVE_ELEMENTS = ACTIVE * VOCAB
PHASES = list(rv.PHASES)


def unit_direction(seed):
    gen = torch.Generator().manual_seed(seed)
    vec = torch.randn(len(NAMES) * 16, generator=gen)
    return (vec / vec.norm()).view(len(NAMES), 16)


def base_gradients(probe):
    gen = torch.Generator().manual_seed(100 + probe)
    return torch.randn(len(NAMES), 16, generator=gen)


def with_distance(probe, distance, seed):
    base = base_gradients(probe)
    moved = base + distance * float(base.norm()) * unit_direction(seed)
    return {name: moved[i].clone() for i, name in enumerate(NAMES)}


def flipped(tensor, count, offset):
    out = tensor.clone()
    flat = out.view(torch.int16).view(-1)
    positions = torch.arange(offset, offset + count) * 7 + 1
    flat[positions] += 1
    return out


def make_spec(**overrides):
    spec = {
        "flips": {(2, "P"): 30, (2, "L"): 6, (3, "P"): 30, (3, "L"): 6, (0, "P"): 3, (0, "L"): 3,
                  (1, "P"): 3, (1, "L"): 3},
        "dist_scale": {}, "identical_pl": set(), "r2_differs": None, "drop_pass": None,
        "ce_calls": {}, "refusal": None, "names": 224, "hash_change": None,
        "missing_phase": None, "kernel_events": {}, "integrity": {}, "receipt_change": None,
        "lora_a_nonzero": None,
    }  # fmt: skip
    spec.update(overrides)
    return spec


def lora_a(spec, phase, key):
    """LoRA-A gradients are exactly zero unless the scenario says otherwise."""
    tensor = torch.zeros(16)
    if spec["lora_a_nonzero"] == (phase, key):
        tensor[3] = 1e-3
    return {"gradient.layer0.lora_A.weight": tensor}


def build_tree(tmp_path, monkeypatch, spec):
    import h2_ce_ref64_evidence as gates
    from safetensors.torch import save_file

    # These are reduced-V statistics fixtures, not sealed real-GPU evidence.
    # The authority gate has separate sealed-tree regression tests.
    monkeypatch.setattr(rv, "validate_series_authority", lambda *args: None)
    monkeypatch.setattr(gates, "REAL_VOCAB", VOCAB)

    root = tmp_path / "series"
    root.mkdir()
    (root / "contract.json").write_text(json.dumps({"series": "ref64", "instance_id": 1}))
    labels = torch.full((1, ROWS), -100)
    labels[0, 1:7] = 5
    table = {}
    noise = __import__("random").Random(7)
    for phase in PHASES:
        if phase == spec["missing_phase"]:
            continue
        runner_dir = root / phase / "runner"
        (runner_dir / "hooks").mkdir(parents=True)
        shas = {}
        calls = {"baseline_probes": 0, "ce_only_probes": 0, "ref64_probes": 4,
                 "ref64_probes_repeat": 4,
                 "ref64_perturbed_probe02": 18, "ref64_perturbed_probe03": 18}[phase]  # fmt: skip
        calls = spec["ce_calls"].get(phase, calls)
        events = spec["kernel_events"].get(phase, 1 if phase == "ce_only_probes" else 0)
        evidence = {
            "status": "COMPLETE_CE_CAPTURE", "initial_identity": "abc",
            "determinism_receipt": {"x": 1 if phase != spec["receipt_change"] else 2},
            "kernels": {"family_events": {"cross_entropy": events}},
            "peak_memory_bytes": {"probe_00": 123},
            "ref64": {"ce64_calls": calls, "integrity_ok": spec["integrity"].get(phase, True),
                      "refusals": ["bad"] if spec["refusal"] == phase else [], "flips": []},
        }  # fmt: skip
        if "perturbed" in phase:
            evidence["ref64"]["flips"] = [
                {
                    "level": level,
                    "seed": ce.perturbation_seed(i, seed),
                    "flipped": max(1, round(level * ACTIVE_ELEMENTS)),
                    "eligible_total": ACTIVE_ELEMENTS,
                    "eligible_usable": ACTIVE_ELEMENTS,
                    "positions_sha256": "a" * 64 + ":" + "b" * 64,
                }
                for i, level in enumerate(rv.GRID)
                for seed in range(3)
            ]
        (runner_dir / "evidence.json").write_text(json.dumps(evidence))
        if phase in ("ref64_perturbed_probe02", "ref64_perturbed_probe03"):
            probe = int(phase[-2:])
            for level_index, level in enumerate(rv.GRID):
                for seed_index in range(3):
                    key = f"pass_{level_index}_{seed_index}"
                    if spec["drop_pass"] == (probe, key):
                        continue
                    (runner_dir / key / "model_probe").mkdir(parents=True)
                    distance = 0.05 * math.sqrt(level) * math.exp(0.05 * noise.gauss(0, 1))
                    distance *= spec["dist_scale"].get(("rp", probe), 1.0)
                    grads = with_distance(
                        probe, distance, 1000 * probe + 10 * level_index + seed_index
                    )
                    table[(phase, key)] = (
                        {"loss": 10.0 + probe},
                        {"input.labels": labels, **grads, **lora_a(spec, phase, key)},
                    )
                    shas[key] = hashlib.sha256(f"h{probe}".encode()).hexdigest()
            (runner_dir / "hooks" / "lm_head_sha256.json").write_text(json.dumps(shas))
            continue
        for probe in range(4):
            key = f"probe_{probe:02d}"
            ref = torch.randn(
                1, ROWS, VOCAB, generator=torch.Generator().manual_seed(500 + probe)
            ).bfloat16()
            ref[:, ACTIVE:] = 0  # ignored shifted-label rows have zero CE gradient
            if phase in ("ref64_probes", "ref64_probes_repeat"):
                dlogits = ref.clone()
                grads = {name: base_gradients(probe)[i].clone() for i, name in enumerate(NAMES)}
                if phase == "ref64_probes_repeat" and spec["r2_differs"] == probe:
                    dlogits = flipped(dlogits, 1, 5)
            else:
                tag = "P" if phase == "baseline_probes" else "L"
                dlogits = flipped(ref, spec["flips"][(probe, tag)], 0 if tag == "P" else 4000)
                phi = max(spec["flips"][(probe, tag)], 1) / ACTIVE_ELEMENTS
                distance = 0.05 * math.sqrt(phi) * spec["dist_scale"].get((probe, tag), 1.0)
                grads = with_distance(probe, distance, 10 * probe + (1 if tag == "P" else 2))
                if probe in spec["identical_pl"]:
                    grads = with_distance(
                        probe, 0.05 * math.sqrt(30 / ACTIVE_ELEMENTS), 10 * probe + 1
                    )
            table[(phase, key)] = (
                {"loss": 10.0 + probe},
                {"input.labels": labels, **grads, **lora_a(spec, phase, key)},
            )
            save_file({"dlogits": dlogits}, str(runner_dir / "hooks" / f"{key}.safetensors"))
            (runner_dir / key / "model_probe").mkdir(parents=True)
            shas[key] = hashlib.sha256(f"h{probe}".encode()).hexdigest()
            if spec["hash_change"] == (phase, probe):
                shas[key] = hashlib.sha256(b"different").hexdigest()
        (runner_dir / "hooks" / "lm_head_sha256.json").write_text(json.dumps(shas))

    def fake_read(path):
        return table[(path.parents[2].name, path.parent.name)]

    def fake_names(values):
        return NAMES[: spec["names"]]

    monkeypatch.setattr(rv, "read_probe_tree", fake_read)
    monkeypatch.setattr(rv, "lora_b_names", fake_names)
    return root


def review(tmp_path, monkeypatch, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return rv.review_ref64(build_tree(tmp_path, monkeypatch, make_spec(**overrides)))


def test_review_liger_closer_by_flips_when_everything_is_noise_consistent(tmp_path, monkeypatch):
    result = review(tmp_path, monkeypatch)
    assert result["label"].startswith("LIGER_CLOSER_BY_FLIPS_NOISE_CONSISTENT"), result
    probes = result["summary"]["per_probe"]
    assert probes[2]["phi_p"] == pytest.approx(30 / ACTIVE_ELEMENTS)
    assert probes[2]["phi_l"] == pytest.approx(6 / ACTIVE_ELEMENTS) and probes[2][
        "r_f"
    ] == pytest.approx(0.2)
    assert (
        probes[2]["active_elements"] == ACTIVE_ELEMENTS
        and probes[2]["flips"]["ce_only_probes"] == 6
    )
    assert result["controls"]["failed"] == [] and result["run_status"] == "COMPLETE"
    assert result["summary"]["per_probe"][2]["fit"]["n"] == 18 and not probes[2]["pl_identical"]
    assert probes[2]["d_p"] == pytest.approx(0.05 * math.sqrt(30 / ACTIVE_ELEMENTS), rel=1e-4)
    assert (
        len(probes[2]["per_tensor_p"]) == 224 and "Constants fixed after attempts" in result["text"]
    )
    json.dumps(rv.sanitize(result))


def test_review_other_labels(tmp_path, monkeypatch):
    flips = {
        (2, "P"): 6,
        (2, "L"): 30,
        (3, "P"): 6,
        (3, "L"): 30,
        (0, "P"): 3,
        (0, "L"): 3,
        (1, "P"): 3,
        (1, "L"): 3,
    }
    assert review(tmp_path / "a", monkeypatch, flips=flips)["label"].startswith(
        "PYTORCH_CLOSER_BY_FLIPS"
    )
    equal = {**flips, (2, "P"): 20, (2, "L"): 20, (3, "P"): 20, (3, "L"): 20}
    assert review(tmp_path / "b", monkeypatch, flips=equal)["label"].startswith(
        "INDISTINGUISHABLE_NOISE"
    )
    excess = review(tmp_path / "c", monkeypatch, dist_scale={(2, "L"): 1000.0})
    assert excess["label"].startswith("EXCESS_OVER_FLIP_NOISE_LIGER")
    big = {**flips, (2, "P"): 150}
    assert review(tmp_path / "d", monkeypatch, flips=big)["label"] == "NOT_CALIBRATABLE"
    assert (
        review(tmp_path / "e", monkeypatch, identical_pl={2})["label"]
        == "INCONCLUSIVE_NO_REPRODUCTION"
    )


def test_review_rows_for_controls_incomplete_and_void(tmp_path, monkeypatch):
    assert review(tmp_path / "a", monkeypatch, r2_differs=1)["label"] == "REFERENCE_ARM_INVALID"
    hashed = review(tmp_path / "b", monkeypatch, hash_change=("ce_only_probes", 2))
    assert hashed["label"] == "REFERENCE_ARM_INVALID" and any(
        "control 3" in r for r in hashed["reasons"]
    )
    calls = review(tmp_path / "c", monkeypatch, ce_calls={"ref64_probes": 3})
    assert calls["label"] == "REFERENCE_ARM_INVALID" and any(
        "control 5" in r for r in calls["reasons"]
    )
    events = review(tmp_path / "d", monkeypatch, kernel_events={"ref64_probes_repeat": 1})
    assert events["label"] == "REFERENCE_ARM_INVALID"
    integrity = review(tmp_path / "e", monkeypatch, integrity={"ref64_probes": False})
    assert integrity["label"] == "REFERENCE_ARM_INVALID"
    receipts = review(tmp_path / "f", monkeypatch, receipt_change="ref64_probes")
    assert receipts["label"] == "REFERENCE_ARM_INVALID"
    missing_pass = review(tmp_path / "g", monkeypatch, drop_pass=(3, "pass_5_2"))
    assert missing_pass["label"] == "INCOMPLETE" and missing_pass["run_status"] == "INCOMPLETE"
    missing_child = review(tmp_path / "h", monkeypatch, missing_phase="ref64_perturbed_probe02")
    assert missing_child["label"] == "INCOMPLETE"
    refusal = review(tmp_path / "i", monkeypatch, refusal="ref64_probes")
    assert refusal["label"] == "VOID" and "CE64 refusal" in refusal["reasons"][0]
    short = review(tmp_path / "j", monkeypatch, names=200)
    assert short["label"] == "VOID" and "224" in short["reasons"][0]


def test_review_zero_distance_pass_makes_the_probe_not_calibratable(tmp_path, monkeypatch):
    result = review(tmp_path, monkeypatch, dist_scale={("rp", 3): 0.0})
    assert result["label"] == "NOT_CALIBRATABLE" and any("zero" in r for r in result["reasons"])


def test_review_is_reproducible_and_offline_comparison_matches(tmp_path, monkeypatch):
    root = build_tree(tmp_path, monkeypatch, make_spec())
    first = rv.review_ref64(root)
    (root / "ref64_review.json").write_text(json.dumps(rv.sanitize(first)))
    again = rv.review_ref64(root)
    assert rv.compare_reviews(json.loads((root / "ref64_review.json").read_text()), again)["equal"]
    tampered = json.loads((root / "ref64_review.json").read_text())
    tampered["label"] = "INDISTINGUISHABLE_NOISE_CONSISTENT"
    assert not rv.compare_reviews(tampered, again)["equal"]
    assert rv.main([str(root)]) == 0


# ---------------------------------------------------------------- series and children (stubbed)
def ref64_contract(tmp_path):
    from test_h2_ce_confirmation import authority

    data = authority(tmp_path)
    data["series"] = "ref64"
    data["deadline_unix_s"] = data["series_started_unix_s"] + 2400
    return data


@pytest.mark.parametrize(
    "failure",
    [None, "child_killed", "child_timeout", "child_refusal", "child_binding", "review_error"],
)
def test_ref64_series_launches_six_fresh_children_and_maps_failures(tmp_path, monkeypatch, failure):
    import copy

    import h2_ce_ref64_review as module

    data = ref64_contract(tmp_path)
    launched, outs = [], []
    monkeypatch.setattr(runner, "validate_contract", lambda value: {})

    def child(argv, out, **kwargs):
        phase = argv[argv.index("--phase") + 1]
        launched.append(phase)
        outs.append(argv[argv.index("--out") + 1])
        target = Path(argv[argv.index("--out") + 1])
        target.mkdir(parents=True)
        failing = failure not in (None, "review_error") and len(launched) == 3
        error = {"child_refusal": "ValueError: CE64 refused: bad",
                 "child_binding": "ValueError: Unexpected global patch binding",
                 "child_timeout": "TimeoutError: CE child deadline reached"}  # fmt: skip
        error = error.get(failure) if failing else None
        if not (failing and failure == "child_killed"):  # a killed child leaves no evidence
            (target / "evidence.json").write_text(
                json.dumps({"status": "COMPLETE_CE_CAPTURE", "error": error})
            )
        return {"status": "CHILD_EXITED", "group_cleanup_confirmed": True,
                "child_status": {"returncode": 2 if failing else 0}}  # fmt: skip

    def fake_review(root):
        if failure == "review_error":
            raise ValueError("Child identity differs from authority")
        return {"label": "INDISTINGUISHABLE_NOISE_CONSISTENT", "run_status": "COMPLETE"}

    monkeypatch.setattr(runner, "bounded_process", child)
    monkeypatch.setattr(runner, "read_child", lambda *args: {})
    monkeypatch.setattr(runner, "review_pair", lambda *a, **k: pytest.fail("pair review used"))
    monkeypatch.setattr(
        runner, "review_noise_floor", lambda *a, **k: pytest.fail("noise review used")
    )
    monkeypatch.setattr(module, "review_ref64", fake_review)
    result = runner.run_series(copy.deepcopy(data), Path(data["evidence_root"]))
    expected = list(runner.REF64_PHASES)
    assert launched == (expected if failure in (None, "review_error") else expected[:3])
    assert len(set(outs)) == len(outs)
    if failure is None:
        assert result["status"] == "COMPLETE_CE_REF64_SERIES" and result["label"].startswith(
            "INDISTINGUISHABLE"
        )
        assert (Path(data["evidence_root"]) / "ref64_review.json").exists()
    else:
        assert result["status"] == {"child_killed": "INCOMPLETE", "child_timeout": "INCOMPLETE",
                                    "child_refusal": "VOID", "child_binding": "VOID",
                                    "review_error": "VOID"}[failure]  # fmt: skip


def test_repeat_and_ref64_phases_are_refused_under_other_series(tmp_path, monkeypatch):
    data = ref64_contract(tmp_path)
    for series in (None, "ce_confirmation", "noise_floor"):
        contract = dict(data)
        contract.pop("series")
        if series:
            contract["series"] = series
        for phase in ("ref64_probes", "ref64_perturbed_probe02"):
            monkeypatch.setattr(runner, "_USED", False)
            with pytest.raises(ValueError, match="declared phase"):
                runner.run_child(contract, phase, tmp_path / "x")
    monkeypatch.setattr(runner, "_USED", False)
    with pytest.raises(ValueError, match="declared phase"):
        runner.run_child(data, "baseline_training", tmp_path / "y")
    data["series"] = "ref65"
    with pytest.raises(ValueError, match="Unknown CE series"):
        runner.series_phases(data)


def test_contract_accepts_the_ref64_series(tmp_path):
    data = ref64_contract(tmp_path)
    runner.validate_contract(data)
    assert runner.series_phases(data) == runner.REF64_PHASES


@pytest.mark.parametrize("phase", ["ref64_probes", "ref64_probes_repeat", "ref64_perturbed_probe02",
                                   "ref64_perturbed_probe03"])  # fmt: skip
def test_read_child_accepts_ref64_phases_with_the_right_trace_location(tmp_path, phase):
    from test_h2_ce_controls import good_record

    kind = runner.PHASE_ARMS[phase][1]
    trace = tmp_path / ("pass_0_0" if kind == "perturbed" else "probe_00")
    trace.mkdir()
    (trace / "kernel_trace.json").write_text(
        json.dumps({"traceEvents": [{"cat": "kernel", "name": "gemm"}]})
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "status": "COMPLETE_CE_CAPTURE",
                "patch_after": {
                    "status": "PASS_EXPLICIT_PATCH_BINDINGS",
                    "arm": "ref64",
                    "flags": patches.patch_flags("ref64"),
                    "patch_calls": {"class": 0, "instance": 0},
                },
            }
        )
    )
    controls = good_record()
    controls["phase"] = phase
    (tmp_path / "controls.json").write_text(json.dumps(controls))
    runner.tree_seal(tmp_path)
    assert runner.read_child(tmp_path, phase)["phase"] == phase
    wrong = json.loads((tmp_path / "evidence.json").read_text())
    wrong["patch_after"]["patch_calls"] = {"class": 1, "instance": 1}
    (tmp_path / "evidence.json").write_text(json.dumps(wrong))
    (tmp_path / "files_sha256.json").unlink()
    runner.tree_seal(tmp_path)
    with pytest.raises(ValueError):
        runner.read_child(tmp_path, phase)


def test_default_and_noise_floor_series_are_unchanged():
    assert runner.SERIES["ce_confirmation"] is runner.PHASES
    assert runner.SERIES["noise_floor"] == runner.NOISE_FLOOR_PHASES
    assert runner.series_name({}) == "ce_confirmation"


def test_new_modules_do_not_touch_the_policy_or_the_pinned_runner_files():
    for name in ("h2_ce_ref64.py", "h2_ce_ref64_review.py"):
        text = (ROOT / "benchmarks/harness" / name).read_text()
        mentions = [line for line in text.splitlines() if "numerical_policy" in line]
        assert all("is not used" in line for line in mentions)
        assert "h2_gpu_correctness" not in text
    assert (ROOT / "benchmarks/h2/numerical_policy.json").exists()


def test_addendum_states_the_constants_labels_and_traveling_text():
    text = (ROOT / "benchmarks/h2/CE_FP64_REFERENCE_ADDENDUM.txt").read_text()
    needles = ("k = 3", "Q = 1.5", "Constants fixed after attempts 015-017 were seen",
               "LIGER_CLOSER_BY_FLIPS_NOISE_CONSISTENT", "PYTORCH_CLOSER_BY_FLIPS_NOISE_CONSISTENT",
               "INDISTINGUISHABLE_NOISE_CONSISTENT", "EXCESS_OVER_FLIP_NOISE", "NOT_CALIBRATABLE",
               "INCONCLUSIVE_NO_REPRODUCTION", "REFERENCE_ARM_INVALID", "_LOW_POWER", "_MARGINAL",
               "May NOT claim", "600 s limit",
               "seed = 1000 x level_index + seed_index")  # fmt: skip
    for needle in needles:
        assert needle in text, needle


# torch 2.14.1 CPU randperm/randint realisation of seed 2001
GOLDEN_FLIP = "726cddadbf2c1784904bcdf5d0fcd97f62f4c98086069bb5c3aa03a151306912"


# ------------------------------------------------------------- extra boundaries (review round)
def test_distance_band_edge_is_inclusive_when_the_margin_is_zero():
    fit = manual_fit(alpha=0.0, beta=1.0, sigma=0.0)  # margin 0: |log d - bound| = 0 <= 0.1 * 0
    exact = rv.arm_status(1e-5, 1e-5, fit)
    assert exact["status"] == "NOISE_CONSISTENT" and exact["marginal"] is True
    assert rv.arm_status(1e-5 * (1 - 1e-9), 1e-5, fit)["marginal"] is False


def test_zero_distance_is_never_distance_marginal_but_a_tiny_phi_still_is():
    fit = manual_fit(alpha=-1.0, beta=0.5, sigma=0.5)
    assert rv.arm_status(0.0, 1e-5, fit)["marginal"] is False
    assert rv.arm_status(0.0, 5e-7, fit)["marginal"] is True  # the 0 < phi < 1e-6 rule, not a band


def test_ce64_integrity_check_catches_changes_that_bypass_autograd_versioning():
    cross_entropy, _ = fresh()
    x = logits().requires_grad_(True)
    target = TARGET.clone()
    loss = cross_entropy(x * 1.0, target)
    target.data[0] = 4  # .data bypasses the version counter, so only CE64's own checksum sees it
    with pytest.raises(ce.Ce64RefusedError, match="changed"):
        loss.backward()
    cross_entropy2, _ = fresh()
    source = logits(4)
    leaf = source.clone().requires_grad_(True)
    loss = cross_entropy2(leaf, TARGET)
    leaf.data[0, 0] += 1.0
    with pytest.raises(ce.Ce64RefusedError, match="changed"):
        loss.backward()


def test_ce64_disables_autocast_in_forward_and_backward():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ce._build_function))
    methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("forward", "backward"):
        withs = [n for n in ast.walk(methods[name]) if isinstance(n, ast.With)]
        assert any(
            "autocast" in ast.unparse(w) and "enabled=False" in ast.unparse(w) for w in withs
        ), name
    assert ce.CHUNK_ROWS == 32


def test_flip_never_selects_maximal_or_subnormal_magnitudes_for_any_seed():
    grad = torch.zeros(1, 8, dtype=torch.bfloat16)
    patterns = [0x0000, 0x0001, 0x007F, 0x7F7F, -32768, 0x0080, 0x7F7E, 0x3F80]
    grad.view(torch.int16)[0, :] = torch.tensor(patterns, dtype=torch.int16)
    rows = torch.tensor([0])
    for seed in range(60):
        out, info = ce.flip_one_ulp(grad, rows, 0.3, seed)
        moved = (grad.view(torch.int16) != out.view(torch.int16)).nonzero()[:, 1].tolist()
        assert set(moved) <= {5, 6, 7} and len(moved) == info["flipped"] == 2, (seed, moved)


def test_flip_golden_realisation_is_pinned_for_the_registered_seed_formula():
    torch.manual_seed(5)
    grad = (torch.randn(6, 40) / 30).bfloat16()
    out, info = ce.flip_one_ulp(grad, torch.arange(6), 0.05, ce.perturbation_seed(2, 1))
    assert info["flipped"] == 12 and info["seed"] == 2001
    assert ce.sha256_tensor(out) == GOLDEN_FLIP
    assert "positions_sha256" in info


def test_hook_stores_a_clone_not_the_live_gradient():
    class Head(torch.nn.Module):
        def forward(self, x):
            return x + 0.0  # the gradient reaches x as the very same tensor

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = Head()

    model = Model()
    hooks = ce.LmHeadHooks(model)
    x = torch.randn(1, 3, 4, requires_grad=True)
    hooks.begin("probe_00")
    model.lm_head(x).sum().backward()
    hooks.dlogits["probe_00"].zero_()
    assert torch.equal(x.grad, torch.ones(1, 3, 4))


def test_labels_are_shifted_before_counting_active_rows():
    labels = torch.full((1, 24), -100)
    labels[0, 0] = 5  # the first label predicts nothing after the shift
    labels[0, 1:7] = 5
    assert int(rv.active_rows(labels).sum()) == 6


def test_run_child_ties_the_hook_and_ref64_series_flags_to_the_series_name():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(runner.run_child).lstrip())
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "ref64_series" for t in n.targets)]  # fmt: skip
    assert len(assigns) == 1 and "series_name(contract) == 'ref64'" in ast.unparse(assigns[0].value)


def test_lora_a_gradients_must_be_exactly_zero(tmp_path, monkeypatch):
    clean = review(tmp_path / "a", monkeypatch)
    assert clean["controls"]["failed"] == []
    dirty = review(tmp_path / "b", monkeypatch, lora_a_nonzero=("ce_only_probes", "probe_02"))
    assert dirty["label"] == "REFERENCE_ARM_INVALID"
    assert any(
        "LoRA-A" in reason and "ce_only_probes probe_02" in reason for reason in dirty["reasons"]
    )
    in_pass = review(
        tmp_path / "c", monkeypatch, lora_a_nonzero=("ref64_perturbed_probe03", "pass_4_1")
    )
    assert in_pass["label"] == "REFERENCE_ARM_INVALID"


def test_perturbation_info_records_the_parameters():
    cross_entropy, state = fresh(vocab=64)
    state.perturbation = {"level": 0.01, "seed": 9}
    cross_entropy(logits(3, vocab=64).requires_grad_(True), TARGET).backward()
    info = state.flips[0]
    assert {
        "flipped",
        "level",
        "seed",
        "eligible_total",
        "eligible_usable",
        "positions_sha256",
    } <= set(info)
    assert runner.ref64_record(cross_entropy)["flips"] == state.flips


def test_traveling_text_names_the_fp64_ce_reference_through_the_bf16_backward():
    for text in (rv.TEXT, (ROOT / "benchmarks/h2/CE_FP64_REFERENCE_ADDENDUM.txt").read_text()):
        assert "fp64 CE reference" in text and "bf16 backward" in text
        assert "NOT the true full-model" in text


def test_a_zero_distance_pass_flag_alone_makes_the_result_not_calibratable():
    probes = {2: probe(zero=True), 3: probe()}  # a valid fit is present, the flag still decides
    result = label(probes)
    assert result["label"] == "NOT_CALIBRATABLE" and any("zero" in r for r in result["reasons"])


def test_slope_test_is_strict_at_the_exact_critical_value(monkeypatch):
    levels = [f for f in rv.GRID for _ in range(3)]
    distances = [
        math.exp(-1.0 + 0.5 * math.log(f) + 0.1 * math.sin(i)) for i, f in enumerate(levels)
    ]
    t_beta = rv.fit_loglog(levels, distances)["t_beta"]
    assert t_beta > 1.7  # a real, positive statistic
    monkeypatch.setattr(rv, "t_upper_quantile", lambda alpha, df: t_beta)
    assert rv.fit_loglog(levels, distances)["beta_ok"] is False  # p-value not strictly below 5%
    monkeypatch.setattr(rv, "t_upper_quantile", lambda alpha, df: t_beta * (1 - 1e-12))
    assert rv.fit_loglog(levels, distances)["beta_ok"] is True


def test_stored_dlogits_never_share_memory_with_the_gradient_that_flows_on():
    seen = {}

    class Head(torch.nn.Module):
        def forward(self, x):
            out = x * 2.0
            out.register_hook(lambda grad: seen.update(ptr=grad.data_ptr()))  # runs before ours
            return out

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = Head()

    model = Model()
    hooks = ce.LmHeadHooks(model)
    hooks.begin("probe_00")
    model.lm_head(torch.randn(1, 3, 4, requires_grad=True)).sum().backward()
    assert hooks.dlogits["probe_00"].data_ptr() != seen["ptr"]


def test_perturbation_children_keep_hashes_and_parameters_not_dlogits():
    import inspect

    source = inspect.getsource(runner.run_child)
    assert "keep_gradients=kind == 'probes'" in source
    assert "write_hook_evidence(out / 'hooks', hooks)" in source


# ------------------------------------------------------------------ CB1: ref64 verify success path
sys.path.insert(0, str(ROOT / "tests"))
from test_h2_operator_probe import fake_bindings  # noqa: E402


def ref64_control():
    control, model = fake_bindings("ref64")
    control.counts = {"class": 0, "instance": 0}  # ref64 never calls a Liger entry point

    def wrapper():
        return None

    wrapper.__module__ = "h2_ce_ref64"
    control.reference_ce = wrapper
    control.functional.cross_entropy = wrapper
    return control, model, wrapper


def test_ref64_verify_passes_with_the_ce64_wrapper_bound_and_everything_else_stock():
    control, model, wrapper = ref64_control()
    receipt = control.verify(model)
    assert receipt["status"] == "PASS_EXPLICIT_PATCH_BINDINGS" and receipt["arm"] == "ref64"
    assert receipt["patch_calls"] == {"class": 0, "instance": 0}
    assert receipt["bindings"]["cross_entropy"].startswith("h2_ce_ref64.")


FAULTS = [
    "stock_bound",
    "other_callable",
    "wrapper_not_registered",
    "extra_rope",
    "wrong_forward",
    "counts",
    "missing_module",
]


@pytest.mark.parametrize("fault", FAULTS)
def test_ref64_verify_refuses_every_other_state(fault):
    control, model, wrapper = ref64_control()
    if fault == "stock_bound":
        control.functional.cross_entropy = control.original["cross_entropy"]
    elif fault == "other_callable":
        control.functional.cross_entropy = lambda: None
    elif fault == "wrapper_not_registered":
        control.reference_ce = None
    elif fault == "extra_rope":
        control.modeling.apply_rotary_pos_emb = control.treatment["rope"]
    elif fault == "wrong_forward":
        model.forward = lambda: None
    elif fault == "counts":
        control.counts = {"class": 1, "instance": 1}
    else:
        model.model.layers.pop()
    with pytest.raises(ValueError):
        control.verify(model)


def test_baseline_and_ce_only_verify_are_unchanged_by_the_ref64_branch():
    for arm in ("baseline", "ce_only"):
        control, model = fake_bindings(arm)
        assert control.verify(model)["status"] == "PASS_EXPLICIT_PATCH_BINDINGS"
    control, model = fake_bindings("baseline")
    control.functional.cross_entropy = lambda: None
    with pytest.raises(ValueError):
        control.verify(model)


# ----------------------------------------------------------- non-finite values and run_status
@pytest.mark.parametrize("key", ["phi_p", "phi_l", "d_p", "d_l"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_numbers_are_void_never_excess(key, bad):
    probes = both()
    probes[2][key] = bad
    result = label(probes)
    assert result["row"] == 1 and result["label"] == "VOID"
    assert any("non-finite" in r and key in r for r in result["reasons"])


@pytest.mark.parametrize("poisoned_call", [7, 8, 10])  # d_P, d_L of probe 02, first Rp pass
def test_review_marks_a_non_finite_distance_void(tmp_path, monkeypatch, poisoned_call):
    original = rv.concat_distance
    state = {"calls": 0}

    def poisoned(candidate, reference, names):
        state["calls"] += 1
        value = original(candidate, reference, names)
        return float("nan") if state["calls"] == poisoned_call else value

    monkeypatch.setattr(rv, "concat_distance", poisoned)
    result = review(tmp_path, monkeypatch)
    assert result["label"] == "VOID" and result["run_status"] == "VOID"


def test_run_status_mirrors_the_label_row(tmp_path, monkeypatch):
    assert review(tmp_path / "a", monkeypatch)["run_status"] == "COMPLETE"
    assert review(tmp_path / "b", monkeypatch, refusal="ref64_probes")["run_status"] == "VOID"
    assert (
        review(tmp_path / "c", monkeypatch, drop_pass=(3, "pass_5_2"))["run_status"] == "INCOMPLETE"
    )
    assert review(tmp_path / "d", monkeypatch, names=200)["run_status"] == "VOID"
    assert review(tmp_path / "e", monkeypatch, r2_differs=1)["run_status"] == "COMPLETE"


def test_addendum_pre_registers_the_row_8_reading_the_void_rule_and_revision():
    text = (ROOT / "benchmarks/h2/CE_FP64_REFERENCE_ADDENDUM.txt").read_text()
    needles = (
        "revision 7.3",
        "the flip-ratio band is not applied",
        "other than the deadline",
        "run_status mirrors rows 1-2",
        "two whole-model snapshots",
    )
    for needle in needles:
        assert needle in text, needle
    assert "revision 7 (authoritative)" not in text
