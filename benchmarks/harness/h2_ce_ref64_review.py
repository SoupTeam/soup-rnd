"""Statistics, controls, label table and offline review for the attempt018 fp64-reference arm.

The experiment compares each arm with an fp64 CE reference propagated through the bf16 backward;
it is NOT a comparison with the true full-model gradient.

Pure functions first (CPU-testable, no torch), evidence readers last. Protocol:
benchmarks/h2/CE_FP64_REFERENCE_ADDENDUM.txt. Constants are CONVENTIONS fixed after attempts 015-017
were seen. numerical_policy.json is not used.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

from h2_ce_ref64_evidence import (
    read_hooks,
    validate_dlogits,
    validate_flips,
    validate_probe,
    validate_series_authority,
)

K = 3.0  # excess multiplier
Q = 1.5  # flip-ratio threshold
LOW_POWER_FACTOR = 5.0
PHI_MIN = 1e-6
PHI_MAX = 3e-4
BAND = 0.10
ALPHA_T = 0.05
GRID = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
SEEDS = 3
PASSES = len(GRID) * SEEDS
DECISIVE = (2, 3)
LOSS_SPACINGS = 2
F_RHO = 2.0**-8
TEXT = (
    "Constants fixed after attempts 015-017 were seen. Reference = exact-CE gradient with the rest "
    "of the model in bf16/NF4, not a full-precision gradient: the comparison is against an fp64 CE "
    "reference propagated through the bf16 backward, NOT the true full-model gradient. The "
    "existing "
    "policy comparison and the BLOCKED outcomes of attempts 015/016 are unchanged."
)
POWER_TEXT = (
    "NOISE_CONSISTENT means the arm LoRA-B distance is within detectable_factor = "
    "exp(k * sigma_hat * pf) of the flip-matched random-flip noise (values per probe in the "
    "output); if detectable_factor is large, "
    "NOISE_CONSISTENT is weak."
)
PHASES = (
    "baseline_probes",
    "ce_only_probes",
    "ref64_probes",
    "ref64_probes_repeat",
    "ref64_perturbed_probe02",
    "ref64_perturbed_probe03",
)


# ----------------------------------------------------------------------------- t distribution
def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_upper_quantile(alpha: float, df: int) -> float:
    """t such that P(T > t) = alpha (one-sided critical value)."""
    low, high = 0.0, 50.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if 1.0 - t_cdf(mid, df) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


# --------------------------------------------------------------- calibration regression
def fit_loglog(levels: list[float], distances: list[float]) -> dict:
    """OLS of log G on log f over all passes (18 points, 16 df); a zero distance is undefined."""
    if len(levels) != len(distances) or len(levels) < 3:
        raise ValueError("Regression needs matching lists of at least three points")
    if any(not (g > 0 and math.isfinite(g)) for g in distances):
        raise ValueError("Zero or non-finite perturbation distance: log G undefined")
    x = [math.log(f) for f in levels]
    y = [math.log(g) for g in distances]
    n = len(x)
    xbar, ybar = sum(x) / n, sum(y) / n
    sxx = sum((v - xbar) ** 2 for v in x)
    if sxx <= 0:
        raise ValueError("Regression needs at least two distinct levels")
    beta = sum((a - xbar) * (b - ybar) for a, b in zip(x, y)) / sxx
    alpha = ybar - beta * xbar
    sse = sum((b - (alpha + beta * a)) ** 2 for a, b in zip(x, y))
    df = n - 2
    sigma = math.sqrt(sse / df)
    se_beta = sigma / math.sqrt(sxx)
    t_beta = math.inf if se_beta == 0 and beta > 0 else (0.0 if se_beta == 0 else beta / se_beta)
    critical = t_upper_quantile(ALPHA_T, df)
    return {
        "alpha": alpha,
        "beta": beta,
        "sigma_hat": sigma,
        "n": n,
        "df": df,
        "xbar": xbar,
        "sxx": sxx,
        "t_beta": t_beta,
        "t_critical": critical,
        "beta_ok": beta > 0 and t_beta > critical,
    }


def prediction_factor(fit: dict, phi: float) -> float:
    """pf = sqrt(1 + x0' (X'X)^-1 x0) for the regression prediction at log(phi)."""
    x0 = math.log(phi)
    return math.sqrt(1.0 + 1.0 / fit["n"] + (x0 - fit["xbar"]) ** 2 / fit["sxx"])


def level_summary(levels: list[float], distances: list[float]) -> dict:
    """Report-only: per-level mean and sd of log G and the heteroscedasticity ratio."""
    rows = {}
    for level in sorted(set(levels)):
        values = [math.log(g) for f, g in zip(levels, distances) if f == level and g > 0]
        mean = sum(values) / len(values)
        sd = (
            math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            if len(values) > 1
            else 0.0
        )
        rows[level] = {"mean_log_g": mean, "sd_log_g": sd, "n": len(values)}
    sds = [row["sd_log_g"] for row in rows.values()]
    het = (max(sds) / min(sds)) if sds and min(sds) > 0 else None
    return {"levels": rows, "het": het}


# --------------------------------------------------------------- arm status and labels
def arm_status(d: float, phi: float, fit: dict | None, k: float = K) -> dict:
    """Noise-consistency of one arm in one decisive probe (design section 2.3)."""
    out: dict = {
        "d": d,
        "phi": phi,
        "status": None,
        "marginal": False,
        "tiny_phi": False,
        "detectable_factor": None,
        "bound": None,
        "pf": None,
        "log_d": None,
    }
    if phi == 0:
        out["status"] = "NOISE_CONSISTENT" if d == 0 else "INVALID_PHI0"
        return out
    if phi > PHI_MAX:
        out["status"] = "NOT_CALIBRATABLE"
        return out
    if fit is None:
        out["status"] = "NOT_CALIBRATABLE"
        return out
    evaluated = PHI_MIN if phi < PHI_MIN else phi
    out["tiny_phi"] = phi < PHI_MIN
    pf = prediction_factor(fit, evaluated)
    margin = k * fit["sigma_hat"] * pf
    out["pf"] = pf
    out["detectable_factor"] = math.exp(margin)
    if d == 0:
        out["status"] = "NOISE_CONSISTENT"  # a distance of zero is below any bound; no logarithm
        out["marginal"] = out["tiny_phi"]
        return out
    bound = fit["alpha"] + fit["beta"] * math.log(evaluated) + margin
    log_d = math.log(d)
    out.update(bound=bound, log_d=log_d)
    out["status"] = "NOISE_CONSISTENT" if log_d <= bound else "EXCESS"
    out["marginal"] = out["tiny_phi"] or abs(log_d - bound) <= BAND * margin
    return out


def liger_side(phi_l: float, phi_p: float, q: float = Q) -> bool:
    return phi_l <= phi_p / q


def pytorch_side(phi_p: float, phi_l: float, q: float = Q) -> bool:
    return phi_p <= phi_l / q


def ratio_marginal(phi_l: float, phi_p: float, q: float = Q) -> bool:
    """r = phi_L / phi_P within 10% of 1/Q or of Q (only defined when both are positive)."""
    if phi_l <= 0 or phi_p <= 0:
        return False
    r = phi_l / phi_p
    return abs(r - 1.0 / q) <= BAND / q or abs(r - q) <= BAND * q


def nonfinite_reasons(probes: dict) -> list[str]:
    """Row 1 (non-finite values): a NaN or infinite flip fraction or distance is never labelled."""
    reasons = []
    for probe, row in sorted(probes.items()):
        for key in ("phi_p", "phi_l", "d_p", "d_l"):
            if not math.isfinite(row[key]):
                reasons.append(f"probe {probe:02d}: non-finite {key}")
    return reasons


def classify(
    *,
    void_reasons: list[str],
    incomplete_reasons: list[str],
    failed_controls: list[str],
    probes: dict,
) -> dict:
    """Ordered label table. probes: {2: {...}, 3: {...}} with phi_p, phi_l, d_p, d_l, pl_identical,
    zero_pass (bool), fit (dict or None)."""
    base = {"text": TEXT, "power_text": POWER_TEXT}
    void_reasons = list(void_reasons) + nonfinite_reasons(probes)
    if void_reasons:
        return {**base, "row": 1, "label": "VOID", "reasons": void_reasons}
    if incomplete_reasons:
        return {**base, "row": 2, "label": "INCOMPLETE", "reasons": incomplete_reasons}
    arms: dict = {}
    control7 = []
    for probe in DECISIVE:
        p = probes[probe]
        arms[probe] = {
            "PYTORCH": arm_status(p["d_p"], p["phi_p"], p["fit"]),
            "LIGER": arm_status(p["d_l"], p["phi_l"], p["fit"]),
        }
        for name, row in arms[probe].items():
            if row["status"] == "INVALID_PHI0":
                control7.append(f"probe {probe:02d} {name}: phi=0 but distance>0")
    controls = list(failed_controls) + control7
    if controls:
        return {
            **base,
            "row": 3,
            "label": "REFERENCE_ARM_INVALID",
            "reasons": controls,
            "arms": arms,
        }
    if any(probes[p]["pl_identical"] for p in DECISIVE):
        return {**base, "row": 4, "label": "INCONCLUSIVE_NO_REPRODUCTION", "arms": arms}
    uncalibratable = []
    for probe in DECISIVE:
        p = probes[probe]
        if max(p["phi_p"], p["phi_l"]) > PHI_MAX:
            uncalibratable.append(f"probe {probe:02d}: flip fraction above {PHI_MAX}")
        if p["zero_pass"] or p["fit"] is None:
            uncalibratable.append(
                f"probe {probe:02d}: perturbation distance of zero (log undefined)"
            )
        elif not p["fit"]["beta_ok"]:
            uncalibratable.append(f"probe {probe:02d}: regression slope not positive at 5%")
    if uncalibratable:
        return {
            **base,
            "row": 5,
            "label": "NOT_CALIBRATABLE",
            "reasons": uncalibratable,
            "arms": arms,
        }
    excess = sorted(
        {
            name
            for probe in DECISIVE
            for name, row in arms[probe].items()
            if row["status"] == "EXCESS"
        }
    )
    rows = [row for probe in DECISIVE for row in arms[probe].values()]
    suffix = []
    if any(
        row["detectable_factor"] is not None and row["detectable_factor"] > LOW_POWER_FACTOR
        for row in rows
    ):
        suffix.append("_LOW_POWER")
    arm_marginal = any(row["marginal"] for row in rows)
    ratio_marg = any(ratio_marginal(probes[p]["phi_l"], probes[p]["phi_p"]) for p in DECISIVE)
    liger = all(liger_side(probes[p]["phi_l"], probes[p]["phi_p"]) for p in DECISIVE)
    torch_side = all(pytorch_side(probes[p]["phi_p"], probes[p]["phi_l"]) for p in DECISIVE)
    if excess:
        name = "BOTH" if len(excess) == 2 else excess[0]
        row, label = 8, "EXCESS_OVER_FLIP_NOISE_" + name
        suffix = ["_MARGINAL"] if arm_marginal else []
    elif liger:
        row, label = 6, "LIGER_CLOSER_BY_FLIPS_NOISE_CONSISTENT"
    elif torch_side:
        row, label = 6, "PYTORCH_CLOSER_BY_FLIPS_NOISE_CONSISTENT"
    else:
        row, label = 7, "INDISTINGUISHABLE_NOISE_CONSISTENT"
    if row in (6, 7) and (arm_marginal or ratio_marg):
        suffix.append("_MARGINAL")
    return {
        **base,
        "row": row,
        "label": label + "".join(suffix),
        "suffixes": suffix,
        "arms": arms,
        "excess_arms": excess,
    }


# ----------------------------------------------------------------------------- controls (pure)
def fp32_spacing(value: float) -> float:
    """Gap to the next larger fp32 value (np.spacing for float32 without numpy)."""
    bits = struct.unpack("<I", struct.pack("<f", abs(value)))[0]
    return (
        struct.unpack("<f", struct.pack("<I", bits + 1))[0]
        - struct.unpack("<f", struct.pack("<I", bits))[0]
    )


def loss_agrees(loss_ref: float, loss_other: float) -> bool:
    return abs(loss_ref - loss_other) <= LOSS_SPACINGS * fp32_spacing(loss_ref)


def evaluate_controls(data: dict) -> dict:
    """Controls 1-6 from plain data; returns {'failed': [...], 'detail': {...}}.

    data keys: repeat_identical {probe: bool}, losses {phase: {probe: float}},
    forward_hashes {probe: {key: sha}}, ce64_calls {phase: int}, ce64_expected {phase: int},
    integrity_ok {phase: bool}, liger_events {phase: int},
    receipts_equal bool, identity_equal bool.
    """
    failed, detail = [], {}
    for probe, same in sorted(data["repeat_identical"].items()):
        detail[f"1.repeat_identical.probe{probe}"] = same
        if not same:
            failed.append(f"control 1: R and R2 differ in probe {probe:02d}")
    ref = data["losses"]["ref64_probes"]
    for phase in ("baseline_probes", "ce_only_probes"):
        for probe, value in sorted(data["losses"][phase].items()):
            ok = loss_agrees(ref[probe], value)
            detail[f"2.loss.{phase}.probe{probe}"] = ok
            if not ok:
                failed.append(
                    f"control 2: loss of {phase} differs from R by over {LOSS_SPACINGS} spacings "
                    f"in probe {probe:02d}"
                )
    for probe, hashes in sorted(data["forward_hashes"].items()):
        same = bool(hashes) and all(hashes.values()) and len(set(hashes.values())) == 1
        detail[f"3.same_forward.probe{probe}"] = same
        if not same:
            failed.append(
                f"control 3: lm_head output differs between arms/passes in probe {probe:02d}"
            )
    if not data["identity_equal"]:
        failed.append("control 3: initial identity differs between children")
    for phase, calls in sorted(data["ce64_calls"].items()):
        ok = calls == data["ce64_expected"][phase]
        detail[f"5.calls.{phase}"] = ok
        if not ok:
            failed.append(
                f"control 5: CE64 call count {calls} in {phase}, "
                f"expected {data['ce64_expected'][phase]}"
            )
    for phase, ok in sorted(data["integrity_ok"].items()):
        detail[f"4.integrity.{phase}"] = ok
        if not ok:
            failed.append(f"control 4: CE64 integrity failure in {phase}")
    for phase, events in sorted(data["liger_events"].items()):
        want_events = phase == "ce_only_probes"
        ok = (events > 0) == want_events
        detail[f"5.kernels.{phase}"] = ok
        if not ok:
            failed.append(f"control 5: unexpected liger kernel events ({events}) in {phase}")
    if not data["receipts_equal"]:
        failed.append("control 6: environment receipts differ between children")
    return {"failed": failed, "detail": detail}


def describe_history(history: dict | None, current: dict) -> dict:
    """Control 9 (descriptive only): P and L gradients versus the attempt016 ones, when provided."""
    if history is None:
        return {"status": "NOT_PROVIDED"}
    return {
        "status": "COMPARED",
        "equal": {key: history.get(key) == value for key, value in current.items()},
    }


# ----------------------------------------------------------------------------- evidence readers
def lora_a_nonzero(values: dict) -> list[str]:
    """Integrity: every LoRA-A gradient is exactly zero in every arm and probe."""
    return sorted(
        k
        for k, v in values.items()
        if k.startswith("gradient.") and "lora_A" in k and bool(v.any())
    )


def lora_b_names(values: dict) -> list[str]:
    return sorted(k for k in values if k.startswith("gradient.") and "lora_B" in k)


def concat_distance(candidate: dict, reference: dict, names: list[str]) -> float:
    """Global relative L2 over the concatenated LoRA-B tensors in float64."""

    delta = norm = 0.0
    for name in names:
        ref = reference[name].double().reshape(-1)
        cand = candidate[name].double().reshape(-1)
        norm += float(ref.square().sum())
        delta += float((cand - ref).square().sum())
    if norm <= 0:
        raise ValueError("Reference LoRA-B gradient is zero")
    return math.sqrt(delta / norm)


def per_tensor_report(candidate: dict, reference: dict, names: list[str]) -> dict:
    """Report-only per-tensor relative L2, cosine and the policy-style max-element ratio."""
    rows = {}
    for name in names:
        ref = reference[name].double().reshape(-1)
        cand = candidate[name].double().reshape(-1)
        ref_norm, cand_norm = float(ref.norm()), float(cand.norm())
        denominator = 1e-4 + 0.02 * float(ref.abs().max())
        rows[name] = {
            "relative_l2": float((cand - ref).norm()) / ref_norm if ref_norm else None,
            "cosine": float((cand * ref).sum()) / (cand_norm * ref_norm)
            if cand_norm and ref_norm
            else None,
            "max_ratio": float((cand - ref).abs().max()) / denominator,
        }
    return rows


def active_rows(labels: Any) -> Any:
    """Boolean mask of non-ignored rows after the ForCausalLMLoss pad-and-shift."""
    import torch

    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1)
    return shifted != -100


def flip_count(candidate: Any, reference: Any) -> int:
    return int((candidate.reshape(-1) != reference.reshape(-1)).sum())


def load_phase(root: Path, phase: str) -> dict:
    return json.loads((root / phase / "runner" / "evidence.json").read_text())


def read_probe_tree(path: Path) -> tuple[dict, dict]:
    from h2_numerics import read_probe

    data, values = read_probe(path)
    validate_probe(path, data, values)
    return data, values


def review_ref64(root: Path, history: dict | None = None) -> dict:
    """Fail closed on invalid evidence in both live and offline review."""
    try:
        return _review_ref64(root, history)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {
            "series": "ref64",
            "run_status": "VOID",
            **classify(
                void_reasons=[type(error).__name__ + ": " + str(error)],
                incomplete_reasons=[],
                failed_controls=[],
                probes={},
            ),
        }


def _review_ref64(root: Path, history: dict | None = None) -> dict:
    """Recompute everything from the stored evidence of a ref64 series (offline or on the box)."""
    import torch
    from h2_ce_confirmation import ref64_failure_status
    from safetensors.torch import load_file

    contract = json.loads((root / "contract.json").read_text())
    void: list[str] = []
    incomplete: list[str] = []
    evidence: dict = {}
    for phase in PHASES:
        path = root / phase / "runner" / "evidence.json"
        if not path.exists():
            incomplete.append(f"{phase}: child missing")
            continue
        try:
            evidence[phase] = json.loads(path.read_text())
        except (OSError, ValueError):
            incomplete.append(f"{phase}: child evidence unreadable")
            continue
        if not isinstance(evidence[phase], dict):
            raise ValueError(f"{phase}: child evidence must be an object")
        if evidence[phase].get("status") != "COMPLETE_CE_CAPTURE":
            reasons = incomplete if ref64_failure_status(path.parent) == "INCOMPLETE" else void
            reasons.append(f"{phase}: {evidence[phase].get('error', 'child did not complete')}")
        keys = (
            [f"pass_{level}_{seed}" for level in range(len(GRID)) for seed in range(SEEDS)]
            if "perturbed" in phase
            else [f"probe_{i:02d}" for i in range(4)]
        )
        for key in keys:
            if not (path.parent / key / "model_probe").is_dir():
                incomplete.append(f"{phase}: {key} missing")
    if void or incomplete:
        label = classify(
            void_reasons=void, incomplete_reasons=incomplete, failed_controls=[], probes={}
        )
        return {
            "run_status": label["label"],
            **label,
            "series": "ref64",
        }
    validate_series_authority(root, contract, evidence)
    runs: dict = {}
    names: list[str] = []
    lora_a_bad: list[str] = []
    for phase in ("baseline_probes", "ce_only_probes", "ref64_probes", "ref64_probes_repeat"):
        runs[phase] = {}
        hooks = root / phase / "runner" / "hooks"
        shas = read_hooks(root, phase)
        for probe in range(4):
            key = f"probe_{probe:02d}"
            data, values = read_probe_tree(root / phase / "runner" / key / "model_probe")
            names = names or lora_b_names(values)
            if lora_a_nonzero(values):
                lora_a_bad.append(f"{phase} {key}")
            dlogits = load_file(str(hooks / f"{key}.safetensors"))["dlogits"]
            validate_dlogits(dlogits, values["input.labels"])
            runs[phase][probe] = {
                "loss": data["loss"],
                "grads": {n: values[n] for n in names},
                "labels": values["input.labels"],
                "dlogits": dlogits,
                "sha": shas.get(key),
            }
    if len(names) != 224:
        void.append(f"expected 224 LoRA-B gradient tensors, found {len(names)}")
    rp: dict = {}
    for probe in DECISIVE:
        phase = f"ref64_perturbed_probe{probe:02d}"
        shas = read_hooks(root, phase)
        rp[probe] = []
        for level_index, level in enumerate(GRID):
            for seed_index in range(SEEDS):
                key = f"pass_{level_index}_{seed_index}"
                pdir = root / phase / "runner" / key / "model_probe"
                if not pdir.exists():
                    incomplete.append(f"{phase}: {key} missing")
                    continue
                data, values = read_probe_tree(pdir)
                if lora_a_nonzero(values):
                    lora_a_bad.append(f"{phase} {key}")
                rp[probe].append(
                    {"level": level, "grads": {n: values[n] for n in names}, "sha": shas.get(key)}
                )
    if incomplete:
        return {
            "run_status": "INCOMPLETE",
            **classify(
                void_reasons=[], incomplete_reasons=incomplete, failed_controls=[], probes={}
            ),
            "series": "ref64",
        }
    # numbers
    summary: dict = {"per_probe": {}}
    decisive: dict = {}
    for probe in range(4):
        ref = runs["ref64_probes"][probe]
        active = active_rows(ref["labels"])
        vocab = ref["dlogits"].shape[-1]
        active_elements = int(active.sum()) * vocab
        if probe in DECISIVE:
            validate_flips(evidence[f"ref64_perturbed_probe{probe:02d}"]["ref64"], active_elements)
        flips = {
            x: flip_count(runs[x][probe]["dlogits"], ref["dlogits"])
            for x in ("baseline_probes", "ce_only_probes", "ref64_probes_repeat")
        }
        phi_p = flips["baseline_probes"] / active_elements if active_elements else 0.0
        phi_l = flips["ce_only_probes"] / active_elements if active_elements else 0.0
        d_p = concat_distance(runs["baseline_probes"][probe]["grads"], ref["grads"], names)
        d_l = concat_distance(runs["ce_only_probes"][probe]["grads"], ref["grads"], names)
        pl_identical = all(
            torch.equal(
                runs["baseline_probes"][probe]["grads"][n],
                runs["ce_only_probes"][probe]["grads"][n],
            )
            for n in names
        )
        summary["per_probe"][probe] = {
            "active_elements": active_elements,
            "flips": flips,
            "phi_p": phi_p,
            "phi_l": phi_l,
            "r_f": (phi_l / phi_p) if phi_p else None,
            "d_p": d_p,
            "d_l": d_l,
            "d_pl": concat_distance(
                runs["baseline_probes"][probe]["grads"],
                runs["ce_only_probes"][probe]["grads"],
                names,
            ),
            "rho": max(d_l, F_RHO) / max(d_p, F_RHO),
            "pl_identical": pl_identical,
            "per_tensor_p": per_tensor_report(
                runs["baseline_probes"][probe]["grads"], ref["grads"], names
            ),
            "per_tensor_l": per_tensor_report(
                runs["ce_only_probes"][probe]["grads"], ref["grads"], names
            ),
        }
        if probe in DECISIVE:
            passes = rp[probe]
            distances = [concat_distance(item["grads"], ref["grads"], names) for item in passes]
            zero_pass = any(g == 0 for g in distances)
            if not all(math.isfinite(g) for g in distances):
                void.append(f"probe {probe:02d}: non-finite perturbation distance")
                zero_pass = True  # no regression is attempted; the VOID row decides
            fit = None if zero_pass else fit_loglog([item["level"] for item in passes], distances)
            summary["per_probe"][probe].update(
                perturbation_distances=distances,
                zero_pass=zero_pass,
                fit=fit,
                level_summary=None
                if zero_pass
                else level_summary([item["level"] for item in passes], distances),
            )
            decisive[probe] = {
                "phi_p": phi_p,
                "phi_l": phi_l,
                "d_p": d_p,
                "d_l": d_l,
                "pl_identical": pl_identical,
                "zero_pass": zero_pass,
                "fit": fit,
            }
    # controls
    controls_data = {
        "repeat_identical": {
            probe: all(
                torch.equal(
                    runs["ref64_probes"][probe]["grads"][n],
                    runs["ref64_probes_repeat"][probe]["grads"][n],
                )
                for n in names
            )
            and torch.equal(
                runs["ref64_probes"][probe]["dlogits"],
                runs["ref64_probes_repeat"][probe]["dlogits"],
            )
            and runs["ref64_probes"][probe]["loss"] == runs["ref64_probes_repeat"][probe]["loss"]
            for probe in range(4)
        },
        "losses": {
            phase: {probe: runs[phase][probe]["loss"] for probe in range(4)}
            for phase in ("baseline_probes", "ce_only_probes", "ref64_probes")
        },
        "forward_hashes": {
            probe: {
                **{phase: runs[phase][probe]["sha"] for phase in runs},
                **(
                    {f"rp_{i}": item["sha"] for i, item in enumerate(rp[probe])}
                    if probe in rp
                    else {}
                ),
            }
            for probe in range(4)
        },
        "ce64_calls": {phase: evidence[phase]["ref64"]["ce64_calls"] for phase in PHASES},
        "ce64_expected": {
            "baseline_probes": 0,
            "ce_only_probes": 0,
            "ref64_probes": 4,
            "ref64_probes_repeat": 4,
            "ref64_perturbed_probe02": PASSES,
            "ref64_perturbed_probe03": PASSES,
        },
        "integrity_ok": {
            phase: bool(evidence[phase]["ref64"]["integrity_ok"])
            for phase in PHASES
            if phase.startswith("ref64")
        },
        "liger_events": {
            phase: int(evidence[phase]["kernels"]["family_events"]["cross_entropy"])
            for phase in PHASES
        },
        "receipts_equal": len(
            {
                json.dumps(evidence[phase].get("determinism_receipt"), sort_keys=True)
                for phase in PHASES
            }
        )
        == 1,
        "identity_equal": len({evidence[phase]["initial_identity"] for phase in PHASES}) == 1,
    }
    controls = evaluate_controls(controls_data)
    controls["failed"] += [
        f"integrity: non-zero LoRA-A gradient in {item}" for item in lora_a_bad
    ]  # REFERENCE_ARM_INVALID, no new control number
    for phase in PHASES:
        if evidence[phase].get("ref64", {}).get("refusals"):
            void.append(f"{phase}: CE64 refusal {evidence[phase]['ref64']['refusals'][0]}")
    label = classify(
        void_reasons=void,
        incomplete_reasons=incomplete,
        failed_controls=controls["failed"],
        probes=decisive,
    )
    summary["peak_memory_bytes"] = {
        phase: evidence[phase].get("peak_memory_bytes") for phase in PHASES
    }
    summary["history"] = describe_history(history, {})
    return {
        "series": "ref64",
        "run_status": {1: "VOID", 2: "INCOMPLETE"}.get(label["row"], "COMPLETE"),
        **label,
        "controls": controls,
        "summary": summary,
        "contract_series": contract.get("series"),
    }


def sanitize(value: Any) -> Any:
    """JSON-safe copy: non-finite floats become strings (a t statistic can be infinite)."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def compare_reviews(box: dict, offline: dict) -> dict:
    """Offline recomputation must equal the box's label and every decisive number."""
    keys = ("label", "row", "run_status")
    same = {key: box.get(key) == offline.get(key) for key in keys}
    same["controls"] = box.get("controls", {}).get("failed") == offline.get("controls", {}).get(
        "failed"
    )
    return {"equal": all(same.values()), "detail": same}


def main(argv: list[str]) -> int:
    root = Path(argv[0])
    review = review_ref64(root)
    stored = root / "ref64_review.json"
    result = {"label": review["label"], "run_status": review["run_status"]}
    if stored.exists():
        result["matches_box"] = compare_reviews(json.loads(stored.read_text()), review)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
