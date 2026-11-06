"""Explanatory CE-function micro-test (attempt017): PyTorch vs Liger vs an fp64 reference.

Standalone. It does not import or change the CE runner, numerical_policy.json, the workset or any
pinned runner file; it only READS benchmarks/h2/ce_confirmation_workset.json for shapes and masks.
See benchmarks/h2/CE_MICROTEST_PROTOCOL.txt for the fixed design. CE-function accuracy on synthetic
logits only; it cannot turn the attempt015/016 outcomes into PASS.

One fresh process, two phases: every PyTorch case first (stock F.cross_entropy), then ONE rebind
through the real apply_liger_kernel_to_llama call (never restored), then every Liger case.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPEC = Path(__file__).resolve().parents[2] / "benchmarks/h2"
U = 2.0**-24
A = 17 * U  # CONVENTION: 17 = ceil(log2 128256), a tree-reduction depth assumption (not derived)
RHO = 2.0  # CONVENTION (arbitrary, decisive)
BAND = 0.10  # marginal band
REFERENCE_TOLERANCE = 1e-13
PINS = {"torch": "2.14.1", "liger-kernel": "0.8.5", "transformers": "5.17.0", "triton": "3.8.0"}
SEEDS = (0, 1, 2, 3, 4)
PATHS = ("mean", "sum24")
REGIMES = (("random", 1.0), ("random", 2.0), ("random", 3.0), ("selfcons", 1.5), ("selfcons", 2.0))
EXPECTED_CASES = 4 * len(REGIMES) * len(SEEDS) * len(PATHS)
LIGER_FLAGS = dict(
    rope=False, cross_entropy=True, fused_linear_cross_entropy=False, rms_norm=False, swiglu=False
)
SCOPE = (
    "CE-function accuracy only, synthetic logits. This says nothing about which LoRA-B gradient is "
    "nearer the truth or why probes 02 and 03 differ. The attempt015/016 BLOCKED outcomes and the "
    "policy are unchanged. fp32 accuracy parity does not imply the same bf16-propagated gradients. "
    "Explanatory; it cannot turn attempt015/016 into PASS."
)
_USED = {"value": False}


@dataclass(frozen=True)
class Config:
    vocab: int = 128256
    hidden: int = 4096
    label_mod: int | None = None  # test-only: remap workset token ids below a small vocab
    device: str = "cuda"

    @property
    def real(self) -> bool:
        return (self.vocab, self.hidden, self.label_mod) == (128256, 4096, None)


@dataclass(frozen=True)
class Case:
    shape: int
    regime: int
    seed: int
    path: str

    @property
    def kind(self) -> str:
        return REGIMES[self.regime][0]

    @property
    def scale(self) -> float:
        return REGIMES[self.regime][1]

    @property
    def id(self) -> str:
        return f"{self.kind}{self.scale}-shape{self.shape}-seed{self.seed}-{self.path}"


def all_cases() -> list[Case]:
    """All 200 cases, seed outermost (a stable order shared by both phases)."""
    return [
        Case(shape, regime, seed, path)
        for seed in SEEDS
        for shape in range(4)
        for regime in range(len(REGIMES))
        for path in PATHS
    ]


def load_workset() -> dict:
    return json.loads((SPEC / "ce_confirmation_workset.json").read_text())


# ----------------------------- pure math (CPU-testable) -----------------------------
def denominator(path: str, count: int) -> int:
    """'mean' divides by the non-ignored count; 'sum24' by 24 (the real window denominator)."""
    if path not in PATHS:
        raise ValueError("Unknown path")
    return count if path == "mean" else 24


def reference_gradients(z32: Any, shifted: Any, denom: int) -> tuple[Any, dict]:
    """fp64 CE gradient two independent ways; returns the first plus self-check receipts."""
    import torch

    x = z32.double()
    mask = shifted != -100
    index = shifted.clamp(min=0)
    onehot = torch.zeros_like(x)
    onehot.scatter_(1, index[:, None], 1.0)
    onehot *= mask[:, None]
    weight = mask[:, None].double() / float(denom)
    shifted_exp = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    first = (shifted_exp / shifted_exp.sum(dim=-1, keepdim=True) - onehot) * weight
    del shifted_exp
    lse = torch.logsumexp(x, dim=-1, keepdim=True)
    second = (torch.exp(x - lse) - onehot) * weight
    scale = first.abs().max()
    if not scale > 0 or not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("Reference gradient empty or non-finite")
    spread = float((first - second).abs().max() / scale)
    loss = float(((lse[:, 0] - x.gather(1, index[:, None])[:, 0]) * mask).sum() / denom)
    return first, {"spread": spread, "loss": loss, "ok": spread <= REFERENCE_TOLERANCE}


def alt_gradient(z32: Any, shifted: Any, denom: int) -> Any:
    """Independent PyTorch fp32 route: log-sum-exp with autograd (never calls F.cross_entropy)."""
    import torch

    leaf = z32.clone().requires_grad_(True)
    mask = shifted != -100
    index = shifted.clamp(min=0)
    picked = leaf.gather(1, index[:, None])[:, 0]
    loss = ((torch.logsumexp(leaf, dim=-1) - picked) * mask).sum() / denom
    loss.backward()
    return leaf.grad.detach()


def metrics(candidate: Any, reference: Any) -> dict:
    """Global relative L2, max-abs ratio and cosine; `reference` is the first-named run."""
    cand = candidate.double()
    delta = cand - reference
    ref_norm = float(reference.norm())
    cand_norm = float(cand.norm())
    ref_max = float(reference.abs().max())
    if not (ref_norm > 0 and ref_max > 0 and math.isfinite(cand_norm)):
        raise ValueError("Degenerate reference or non-finite candidate")
    return {
        "e": float(delta.norm()) / ref_norm,
        "m": float(delta.abs().max()) / ref_max,
        "c": float((cand * reference).sum()) / (cand_norm * ref_norm) if cand_norm else 0.0,
    }


def relative_l2(candidate: Any, reference: Any) -> float:
    return float((candidate.double() - reference.double()).norm() / reference.double().norm())


def sha256_tensor(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.int16)
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


# ----------------------------- inputs -----------------------------
WEIGHT_SEED = 20261029  # the synthetic lm_head W is drawn once, on the CPU, from its own seed


def build_weight(cfg: Config) -> Any:
    """Synthetic bf16 W [V, hidden] for the descriptive dh step only (not used to make logits)."""
    import torch

    gen = torch.Generator().manual_seed(WEIGHT_SEED)
    chunks = [
        torch.randn(min(8192, cfg.vocab - start), cfg.hidden, generator=gen).bfloat16()
        for start in range(0, cfg.vocab, 8192)
    ]
    return torch.cat(chunks).to(cfg.device)


def make_inputs(case: Case, cfg: Config, work: dict) -> dict:
    """Deterministic from the case: i.i.d. Gaussian bf16 logits drawn on the CPU, and labels."""
    import torch

    probe = work["probes"][case.shape]
    raw = torch.tensor(probe["labels"], dtype=torch.long)
    if cfg.label_mod:
        raw = torch.where(raw == -100, raw, raw % cfg.label_mod)
    batch, length = raw.shape
    rows = batch * length
    key = case.seed * 1_000_003 + case.shape * 1009 + case.regime * 101
    gen = torch.Generator().manual_seed(key)
    # z = bf16(s * randn(N, V)): drawn and rounded on the CPU, then copied to the device, so
    # regeneration cannot depend on which cuBLAS algorithm is chosen.
    z_cpu = (torch.randn(rows, cfg.vocab, generator=gen) * case.scale).bfloat16()
    z = z_cpu.to(cfg.device).contiguous()
    shifted = torch.nn.functional.pad(raw, (0, 1), value=-100)[:, 1:].reshape(-1)
    if case.kind == "selfcons":
        sample = torch.Generator().manual_seed(key + 7)
        active = (shifted != -100).nonzero()[:, 0]
        probs = torch.softmax(z[active.to(cfg.device)].float(), dim=-1).cpu()
        drawn = torch.multinomial(probs, 1, generator=sample)[:, 0]
        shifted = shifted.clone()
        shifted[active] = drawn
    labels = torch.nn.functional.pad(shifted.view(batch, length)[:, :-1], (1, 0), value=-100)
    return {
        "z": z.view(batch, length, cfg.vocab),
        "labels": labels.to(cfg.device).contiguous(),
        "shifted": shifted.to(cfg.device),
        "rows": rows,
        "targets": int((shifted != -100).sum()),
        "hashes": {"logits": sha256_tensor(z), "labels": sha256_tensor(labels)},
    }


# ----------------------------- implementations under test -----------------------------
def run_loss(
    master: Any, labels: Any, cfg: Config, nib: int | None, *, fp32: bool, counter: Any = None
) -> tuple[float, Any, bool]:
    """One real ForCausalLMLoss evaluation on a FRESH clone: (loss, gradient, input_modified)."""
    import torch
    from transformers.loss.loss_utils import ForCausalLMLoss

    reference = master.float() if fp32 else master
    leaf = reference.clone().requires_grad_(True)
    before = counter.calls if counter is not None else None
    loss = ForCausalLMLoss(leaf, labels, cfg.vocab, num_items_in_batch=nib)
    loss.backward()
    if counter is not None and counter.calls - before != 1:
        raise ValueError("Liger delegate call count differs from one per loss evaluation")
    modified = not torch.equal(leaf.detach(), reference)
    return float(loss.detach()), leaf.grad.detach().clone(), modified


def run_implementation(
    master: Any,
    labels: Any,
    cfg: Config,
    nib: int | None,
    counter: Any = None,
    profiler: Any = None,
) -> dict:
    """Two repeats of both variants; the first run is compared, repeats only test determinism."""
    import torch

    out: dict = {"modified": False}
    for variant in ("fp32", "bf16"):
        runs = []
        for repeat in range(2):
            capture = profiler if (profiler and variant == "fp32" and repeat == 0) else None
            if capture:
                with capture:
                    runs.append(run_loss(master, labels, cfg, nib, fp32=True, counter=counter))
            else:
                runs.append(
                    run_loss(master, labels, cfg, nib, fp32=variant == "fp32", counter=counter)
                )
        out[variant] = runs[0]
        out[variant + "_repeat_equal"] = bool(
            torch.equal(runs[0][1], runs[1][1]) and runs[0][0] == runs[1][0]
        )
        out["modified"] = out["modified"] or any(r[2] for r in runs)
    out["bf16_consistent"] = bool(torch.equal(out["fp32"][1].bfloat16(), out["bf16"][1]))
    return out


class CountingDelegate:
    """Pass-through wrapper around the rebound liger_cross_entropy that counts calls."""

    def __init__(self, wrapped: Any) -> None:
        self.__wrapped__ = wrapped
        self.__module__ = getattr(wrapped, "__module__", "liger_kernel")
        self.__name__ = getattr(wrapped, "__name__", "liger_cross_entropy")
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.__wrapped__(*args, **kwargs)


# ----------------------------- classification (pure; CPU-testable) -----------------------------
def kernel_names(trace_events: list[dict]) -> list[str]:
    return [str(e.get("name", "")).lower() for e in trace_events if e.get("cat") == "kernel"]


def liger_kernel_seen(trace_events: list[dict]) -> bool:
    return any("liger_cross_entropy" in name for name in kernel_names(trace_events))


def run_status(records: list[dict], void_reasons: list[str], expected: int = EXPECTED_CASES) -> str:
    if void_reasons:
        return "VOID"
    return (
        "COMPLETE"
        if len(records) == expected and len({r["id"] for r in records}) == expected
        else "INCOMPLETE"
    )


def _near(value: float, threshold: float) -> bool:
    return abs(value - threshold) <= BAND * threshold


def ce_label(records: list[dict]) -> dict:
    """Precedence-ordered CE-level label (first match wins) with a MARGINAL flag."""
    count_both = sum(1 for r in records if r["e_p"] > A and r["e_l"] > A)
    if count_both * 2 >= len(records):
        marginal = abs(count_both * 2 - len(records)) <= BAND * len(records)
        return {
            "label": "BOTH_OUTSIDE_A_AT_CE_LEVEL_SYNTHETIC",
            "marginal": marginal,
            "cases_both_outside_a": count_both,
        }
    worse = [r for r in records if r["e_l"] > A and r["rho"] > RHO]
    if worse:
        marginal = any(_near(r["e_l"], A) or _near(r["rho"], RHO) for r in worse)
        return {
            "label": "LIGER_WORSE_AT_CE_LEVEL_SYNTHETIC",
            "marginal": marginal,
            "cases_worse": len(worse),
        }
    if all(r["e_l"] <= A and r["rho"] <= RHO for r in records):
        marginal = any(_near(r["e_l"], A) or _near(r["rho"], RHO) for r in records)
        return {"label": "LIGER_NOT_WORSE_AT_CE_LEVEL_SYNTHETIC", "marginal": marginal}
    return {
        "label": "INCONCLUSIVE_AT_CE_LEVEL",
        "marginal": any(_near(r["e_l"], A) or _near(r["rho"], RHO) for r in records),
    }


def representativeness_gate(records: list[dict]) -> dict:
    """Pairs regimes with shapes by loss (N1). HIGH: random s=2 at 24/48 rows -> zero P-vs-L flips.
    LOW: self-consistent s=1.5 at 96/384 rows -> a flip in >=4 of 5 seeds, per shape and path."""
    by = {(r["kind"], r["scale"], r["shape"], r["path"], r["seed"]): r for r in records}
    high_cases = [
        by.get(("random", 2.0, shape, path, seed))
        for shape in (0, 1)
        for path in PATHS
        for seed in SEEDS
    ]
    low_cells = {}
    for shape in (2, 3):
        for path in PATHS:
            cells = [by.get(("selfcons", 1.5, shape, path, seed)) for seed in SEEDS]
            low_cells[f"shape{shape}-{path}"] = (
                None if None in cells else sum(1 for c in cells if c["flips_pl"] > 0)
            )
    if None in high_cases or any(v is None for v in low_cells.values()):
        return {"status": "NOT_EVALUATED_INCOMPLETE"}
    high_ok = all(c["flips_pl"] == 0 for c in high_cases)
    low_ok = all(v >= 4 for v in low_cells.values())
    return {
        "status": "REPRESENTATIVE" if high_ok and low_ok else "NOT_REPRESENTATIVE",
        "high_loss_cell_zero_flips": high_ok,
        "low_loss_seeds_with_flips": low_cells,
        "low_loss_cell_ok": low_ok,
    }


def _spread(values: list[float]) -> dict:
    ordered = sorted(values)
    return (
        {"min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1]}
        if ordered
        else {}
    )


def anchor_summary(records: list[dict]) -> dict:
    """Empirical anchor: two accurate fp32 implementations differ by about D_P,alt (descriptive)."""
    return {
        "e_alt": _spread([r["e_alt"] for r in records]),
        "d_pl": _spread([r["d_pl"] for r in records]),
        "d_p_alt": _spread([r["d_p_alt"] for r in records]),
        "d_l_alt": _spread([r["d_l_alt"] for r in records]),
        "e_l_over_e_p": _spread([r["e_l"] / max(r["e_p"], 1e-30) for r in records]),
        "d_pl_over_d_p_alt": _spread([r["d_pl"] / max(r["d_p_alt"], 1e-30) for r in records]),
    }


def summarize(records: list[dict], void_reasons: list[str], expected: int = EXPECTED_CASES) -> dict:
    status = run_status(records, void_reasons, expected)
    result: dict = {
        "run_status": status,
        "cases_run": len(records),
        "cases_expected": expected,
        "void_reasons": void_reasons,
        "scope": SCOPE,
        "constants": {
            "A": A,
            "u": U,
            "rho_limit": RHO,
            "band": BAND,
            "note": "A and Rho are conventions (arbitrary, decisive, not derived)",
        },
    }
    if status != "COMPLETE":
        result["label"] = "NO_LABEL_" + status
        return result
    result["ce_level"] = ce_label(records)
    result["label"] = result["ce_level"]["label"]
    gate = representativeness_gate(records)
    result["representativeness"] = gate
    result["explanatory_reading"] = (
        "allowed" if gate["status"] == "REPRESENTATIVE" else "WITHHELD_NOT_REPRESENTATIVE"
    )
    result["anchor"] = anchor_summary(records)
    result["bf16_primary"] = {
        "p_vs_l_flip_total": sum(r["flips_pl"] for r in records),
        "cases_with_flips": sum(1 for r in records if r["flips_pl"] > 0),
        "dh_rel_pl": _spread([r["dh_pl"] for r in records]),
        "statement": "fp32 accuracy parity does not imply the same bf16-propagated gradients",
    }
    result["determinism"] = {
        key: sum(1 for r in records if not r[key]) for key in ("p_repeat_equal", "l_repeat_equal")
    }
    return result


# ----------------------------- environment / receipts -----------------------------
def versions() -> dict:
    out = {}
    for name in PINS:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def pin_problems(found: dict) -> list[str]:
    return [
        f"{name} {found.get(name)} != {want}"
        for name, want in PINS.items()
        if found.get(name) != want
    ]


def receipt(cfg: Config) -> dict:
    import torch

    data: dict = {
        "unix_s": time.time(),
        "versions": versions(),
        "device": cfg.device,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        data["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        data["boot_id"] = None
    if cfg.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        data.update(gpu_name=props.name, gpu_uuid=str(props.uuid))
        query = (
            "name,uuid,driver_version,clocks.sm,clocks.max.sm,"
            "clocks_throttle_reasons.active,temperature.gpu,power.draw"
        )
        for key, argv in (
            ("nvidia_smi", ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]),
            (
                "gpu_processes",
                ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            ),
        ):
            try:
                data[key] = subprocess.run(
                    argv, capture_output=True, text=True, timeout=20
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError) as error:
                data[key] = "unavailable: " + type(error).__name__
    return data


# ----------------------------- the two-phase run -----------------------------
class GradStore:
    """Phase-1 PyTorch fp32 gradients, spilled to disk (200 cases would not fit in memory)."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=False)
        self.root = root

    def put(self, case: Case, tensor: Any) -> None:
        import torch

        torch.save(tensor.cpu(), self.root / (case.id + ".pt"))

    def get(self, case: Case, device: str) -> Any:
        import torch

        return torch.load(self.root / (case.id + ".pt")).to(device)


def trace_has_liger(profiler: Any, directory: Path, name: str) -> bool:
    path = directory / (name + ".json")
    profiler.export_chrome_trace(str(path))
    return liger_kernel_seen(json.loads(path.read_text()).get("traceEvents", []))


def make_profiler(cfg: Config) -> Any:
    if cfg.device != "cuda":
        return None
    from torch.profiler import ProfilerActivity, profile

    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])


def rho_value(e_l: float, e_p: float) -> float:
    """rho = E_L / max(E_P, u): the u floor guards a near-exact PyTorch error."""
    return e_l / max(e_p, U)


def expected_profile_ids(cases: list[Case]) -> tuple[list[str], list[str]]:
    """Fixed profiler subset: first case of each shape on both paths (8 Liger) and, for PyTorch,
    the first case of the 24-row and 384-position shapes on the mean path (2 PyTorch)."""
    liger = [c.id for c in cases if first_case_per_shape(c)]
    pytorch = [
        c.id for c in cases if first_case_per_shape(c) and c.path == "mean" and c.shape in (0, 3)
    ]
    return liger, pytorch


def profiler_void_reasons(
    profiles: dict, cases: list[Case], ran_pytorch: set[str], ran_liger: set[str]
) -> list[str]:
    """VOID reasons when the captured subset differs from the fixed one or a capture is wrong."""
    expected_liger, expected_pytorch = expected_profile_ids(cases)
    want_liger = [i for i in expected_liger if i in ran_liger]
    want_pytorch = [i for i in expected_pytorch if i in ran_pytorch]
    got_liger, got_pytorch = profiles.get("liger", {}), profiles.get("pytorch", {})
    reasons = []
    if set(got_liger) != set(want_liger):
        reasons.append(
            f"Liger profiler captures {sorted(got_liger)} != expected {sorted(want_liger)}"
        )
    if set(got_pytorch) != set(want_pytorch):
        reasons.append(
            f"PyTorch profiler captures {sorted(got_pytorch)} != expected {sorted(want_pytorch)}"
        )
    reasons += [
        f"{i}: no liger_cross_entropy kernel in the Liger capture"
        for i, v in got_liger.items()
        if v is not True
    ]
    reasons += [
        f"{i}: liger kernel seen in a PyTorch capture"
        for i, v in got_pytorch.items()
        if v is not False
    ]
    return reasons


def preflight_void_reasons(
    start_receipt: dict, cfg: Config, expect_gpu: str | None, limit: int | None
) -> list[str]:
    """VOID decisions taken before any case runs (pure, so every branch is CPU-testable)."""
    reasons = pin_problems(start_receipt["versions"]) if cfg.real else []
    if cfg.real and (cfg.device != "cuda" or limit is not None):
        reasons.append("Real configuration requires CUDA and all cases")
    if expect_gpu and start_receipt.get("gpu_name") != expect_gpu:
        reasons.append(f"GPU class {start_receipt.get('gpu_name')} != {expect_gpu}")
    if cfg.device == "cuda" and start_receipt.get("gpu_processes"):
        reasons.append("Foreign GPU processes present at start")
    return reasons


def first_case_per_shape(case: Case) -> bool:
    return case.seed == SEEDS[0] and case.regime == 0


def phase_one(
    cases: list[Case],
    cfg: Config,
    work: dict,
    store: GradStore,
    tmp: Path,
    budget: float,
    started: float,
    void: list[str],
) -> tuple[dict, dict]:
    import torch

    stock = torch.nn.functional.cross_entropy
    if getattr(stock, "__module__", "") != "torch.nn.functional":
        void.append("F.cross_entropy is not stock before phase 1")
        return {}, {}
    results: dict = {}
    profiles: dict = {}
    for case in cases:
        if time.monotonic() - started > budget:
            return results, profiles
        data = make_inputs(case, cfg, work)
        z, labels, shifted = data["z"], data["labels"], data["shifted"]
        master32 = z.float()
        denom = denominator(case.path, data["targets"])
        nib = None if case.path == "mean" else 24
        reference, check = reference_gradients(master32.view(-1, cfg.vocab), shifted, denom)
        if not check["ok"]:
            void.append(f"{case.id}: reference self-check {check['spread']:.3e}")
            return results, profiles
        profiler = (
            make_profiler(cfg)
            if (first_case_per_shape(case) and case.path == "mean" and case.shape in (0, 3))
            else None
        )
        impl = run_implementation(z, labels, cfg, nib, profiler=profiler)
        if profiler is not None:
            profiles[case.id] = trace_has_liger(profiler, tmp, "p_" + case.id)
            if profiles[case.id]:
                void.append(f"{case.id}: liger kernel seen in a PyTorch capture")
        if impl["modified"]:
            void.append(f"{case.id}: PyTorch modified its input")
        if sha256_tensor(z) != data["hashes"]["logits"]:
            void.append(f"{case.id}: master logits changed during phase 1")
        alt = alt_gradient(master32.view(-1, cfg.vocab), shifted, denom)
        g_p = impl["fp32"][1].view(-1, cfg.vocab)
        results[case.id] = {
            "hashes": data["hashes"],
            "targets": data["targets"],
            "rows": data["rows"],
            "loss_p": impl["fp32"][0],
            "ref_loss": check["loss"],
            "ref_spread": check["spread"],
            "p": metrics(g_p, reference),
            "alt": metrics(alt, reference),
            "d_p_alt": relative_l2(g_p, alt),
            "flips_p": int((g_p.bfloat16() != reference.bfloat16()).sum()),
            "p_repeat_equal": impl["fp32_repeat_equal"] and impl["bf16_repeat_equal"],
            "p_bf16_consistent": impl["bf16_consistent"],
        }
        if not impl["bf16_consistent"]:
            void.append(f"{case.id}: bf16 variant differs from bf16(fp32 gradient) for PyTorch")
        store.put(case, g_p)
        del data, master32, reference, alt, g_p, impl
        if void:
            return results, profiles
    return results, profiles


def rebind(cfg: Config, void: list[str]) -> tuple[Any, Any, CountingDelegate | None]:
    """ONE rebind through the real apply_liger_kernel_to_llama call; wrapped by a call counter."""
    import torch

    stock = torch.nn.functional.cross_entropy
    from liger_kernel.transformers import monkey_patch

    monkey_patch.apply_liger_kernel_to_llama(**LIGER_FLAGS)
    bound = torch.nn.functional.cross_entropy
    if bound is stock or not str(getattr(bound, "__module__", "")).startswith("liger_kernel"):
        void.append("F.cross_entropy was not rebound to the Liger function")
        return stock, bound, None
    delegate = CountingDelegate(bound)
    torch.nn.functional.cross_entropy = delegate
    return stock, bound, delegate


def phase_two(
    cases: list[Case],
    cfg: Config,
    work: dict,
    store: GradStore,
    tmp: Path,
    phase1: dict,
    budget: float,
    started: float,
    void: list[str],
    delegate: CountingDelegate,
    bound: Any,
) -> tuple[list[dict], dict]:
    import torch

    records: list[dict] = []
    profiles: dict = {}
    weight = build_weight(cfg)
    for case in cases:
        if case.id not in phase1:
            break
        if time.monotonic() - started > budget:
            break
        if torch.nn.functional.cross_entropy is not delegate or delegate.__wrapped__ is not bound:
            void.append(f"{case.id}: F.cross_entropy binding wrong before the case")
            break
        data = make_inputs(case, cfg, work)
        first = phase1[case.id]
        if data["hashes"] != first["hashes"]:
            void.append(f"{case.id}: input hash differs between phase 1 and phase 2")
            break
        z, labels, shifted = data["z"], data["labels"], data["shifted"]
        master32 = z.float()
        denom = denominator(case.path, data["targets"])
        nib = None if case.path == "mean" else 24
        reference, check = reference_gradients(master32.view(-1, cfg.vocab), shifted, denom)
        if not check["ok"] or abs(check["loss"] - first["ref_loss"]) > 1e-9 * max(
            1.0, abs(check["loss"])
        ):
            void.append(f"{case.id}: reference self-check failed in phase 2")
            break
        profiler = make_profiler(cfg) if first_case_per_shape(case) else None
        calls_before = delegate.calls
        impl = run_implementation(z, labels, cfg, nib, counter=delegate, profiler=profiler)
        if delegate.calls - calls_before != 4:
            void.append(
                f"{case.id}: Liger delegate saw {delegate.calls - calls_before} calls, expected 4"
            )
            break
        if profiler is not None:
            profiles[case.id] = trace_has_liger(profiler, tmp, "l_" + case.id)
            if not profiles[case.id]:
                void.append(f"{case.id}: no liger_cross_entropy kernel event in the capture")
                break
        if sha256_tensor(z) != data["hashes"]["logits"]:
            void.append(f"{case.id}: master logits changed during phase 2")
            break
        alt = alt_gradient(master32.view(-1, cfg.vocab), shifted, denom)
        g_l = impl["fp32"][1].view(-1, cfg.vocab)
        g_p = store.get(case, cfg.device)
        bf_ref = reference.bfloat16()
        bf_p, bf_l = g_p.bfloat16(), g_l.bfloat16()
        dh_p, dh_l, dh_r = (m @ weight for m in (bf_p, bf_l, bf_ref))
        l_metrics = metrics(g_l, reference)
        record = {
            "id": case.id,
            "kind": case.kind,
            "scale": case.scale,
            "shape": case.shape,
            "seed": case.seed,
            "path": case.path,
            "rows": first["rows"],
            "targets": first["targets"],
            "e_p": first["p"]["e"],
            "e_l": l_metrics["e"],
            "m_p": first["p"]["m"],
            "m_l": l_metrics["m"],
            "c_p": first["p"]["c"],
            "c_l": l_metrics["c"],
            "rho": rho_value(l_metrics["e"], first["p"]["e"]),
            "e_alt": first["alt"]["e"],
            "d_p_alt": first["d_p_alt"],
            "d_l_alt": relative_l2(g_l, alt),
            "d_pl": relative_l2(g_l, g_p),
            "flips_p": first["flips_p"],
            "flips_l": int((bf_l != bf_ref).sum()),
            "flips_pl": int((bf_p != bf_l).sum()),
            "dh_pl": relative_l2(dh_l, dh_p),
            "dh_p_ref": relative_l2(dh_p, dh_r),
            "dh_l_ref": relative_l2(dh_l, dh_r),
            "loss_p": first["loss_p"],
            "loss_l": impl["fp32"][0],
            "p_repeat_equal": first["p_repeat_equal"],
            "l_repeat_equal": impl["fp32_repeat_equal"] and impl["bf16_repeat_equal"],
            "l_input_modified_in_place": impl["modified"],
            "l_bf16_consistent": impl["bf16_consistent"],
        }
        if not all(math.isfinite(v) for v in record.values() if isinstance(v, float)):
            void.append(f"{case.id}: non-finite value in the case record")
            break
        if not impl["bf16_consistent"]:
            void.append(f"{case.id}: bf16 variant differs from bf16(fp32 gradient) for Liger")
            break
        records.append(record)
        del data, master32, reference, alt, g_l, g_p, impl, dh_p, dh_l, dh_r
    return records, profiles


def run(
    out: Path,
    cfg: Config,
    budget: float,
    store_dir: Path,
    limit: int | None = None,
    expect_gpu: str | None = None,
) -> dict:
    """The whole micro-test in this (fresh) process."""

    if _USED["value"]:
        raise ValueError("Fresh process required")
    _USED["value"] = True
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    void: list[str] = []
    start_receipt = receipt(cfg)
    void += preflight_void_reasons(start_receipt, cfg, expect_gpu, limit)
    cases = all_cases()[:limit] if limit else all_cases()
    result: dict = {}
    profiles: dict = {}
    records: list[dict] = []
    try:
        if not void:
            work = load_workset()
            store = GradStore(store_dir)
            tmp = out / "traces"
            tmp.mkdir()
            phase1, profiles["pytorch"] = phase_one(
                cases, cfg, work, store, tmp, budget, started, void
            )
            if not void:
                stock, bound, delegate = rebind(cfg, void)
                if not void:
                    records, profiles["liger"] = phase_two(
                        cases, cfg, work, store, tmp, phase1, budget, started, void, delegate, bound
                    )
        if not void and cfg.device == "cuda":
            void += profiler_void_reasons(
                profiles, cases, set(phase1), {record["id"] for record in records}
            )
    except Exception as error:  # Any unexpected failure voids the run; nothing is dropped silently.
        void.append("Unexpected failure: " + type(error).__name__ + ": " + str(error))
    result = summarize(records, void, expected=len(cases))
    if limit is not None or not cfg.real:
        result["configuration"] = "TEST_CONFIGURATION_NOT_A_RESULT"
    result.update(
        cases=records,
        profiler_checks=profiles,
        receipt_start=start_receipt,
        receipt_end=receipt(cfg),
        seconds=time.monotonic() - started,
    )
    (out / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False, default=str) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--store-dir", type=Path, required=True, help="scratch dir for phase-1 gradients"
    )
    parser.add_argument("--budget-seconds", type=float, default=2400.0)
    parser.add_argument("--expect-gpu", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab", type=int, default=128256)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--label-mod", type=int, default=None)
    parser.add_argument("--limit-cases", type=int, default=None, help="test only")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = Config(args.vocab, args.hidden, args.label_mod, args.device)
    result = run(
        args.out, cfg, args.budget_seconds, args.store_dir, args.limit_cases, args.expect_gpu
    )
    print(result["run_status"], result["label"])
    if result["run_status"] != "COMPLETE":
        sys.exit(2)


if __name__ == "__main__":
    main()
