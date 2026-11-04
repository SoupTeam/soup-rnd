#!/usr/bin/env python3
"""Ticket A6-08: where the L2L and gradient-accumulation gap of DPO on CUDA comes from.

The rule is ``benchmarks/l2l-preference-diagnosis-rule-a6-ticket08.md``, committed before the first
GPU run. This module holds the judges (pure functions over recorded arms) and the driver that
records the arms through the real DPO wrapper. Benchmark code, not shipped. torch is imported
inside functions.

An arm is ``{"grads": [{name: tensor}, ...], "losses": [float, ...]}``, as in ``l2l_pref_check``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_pref_check  # noqa: E402

F32 = "torch.float32"
#: Fields of a dtype record and the value each must have for the float32 patch to count as taken.
CONFIRMED = {
    "stream_dtype": "float32",
    "train_bf16": False,
    "train_fp16": False,
    "mixed_precision": "no",
    "base_weight": F32,
    "lora_weight": F32,
    "layer_in": F32,
    "layer_out": F32,
}


def _type_problems(arm: str, record: Optional[Dict[str, Any]]) -> List[str]:
    """Why the dtype record of ``arm`` does not confirm float32 with autocast off."""
    if record is None:
        return [f"{arm}: no type record"]
    out = [
        f"{arm}: {field} is {record.get(field)!r}, float32 needs {wanted!r}"
        for field, wanted in CONFIRMED.items()
        if record.get(field) != wanted
    ]
    grads = record.get("grads")
    if not grads or any(kind != F32 for kind in grads):
        out.append(f"{arm}: grads is {grads!r}, float32 needs only {F32!r}")
    return out


def judge_float32(
    ga1: Optional[Dict[str, Any]],
    ga2: Optional[Dict[str, Any]],
    l2l: Optional[Dict[str, Any]],
    dtypes: Dict[str, Any],
    prefix: str = "f32",
) -> Dict[str, Any]:
    """Control 1: float32 L2L against float32 accumulation, bit for bit (rule, control 1).

    ``unavailable`` when an arm has no valid record, a dtype record does not confirm float32
    with autocast off, or the accumulation does not repeat itself bit for bit. ``prefix`` names the
    arms (``f32`` pins the SDPA backend, ``f32u`` leaves it to the default; the addendum).
    """
    arms = {f"{prefix}_ga1": ga1, f"{prefix}_ga2": ga2, f"{prefix}_l2l": l2l}
    unavailable = [f"{name}: no valid arm" for name, arm in arms.items() if arm is None]
    for name in arms:
        unavailable += _type_problems(name, dtypes.get(name))
    if not unavailable:
        repeat = l2l_pref_check._differences(ga1, ga2)
        if repeat:
            unavailable.append(
                f"{prefix}_ga1 against {prefix}_ga2 is not equal: {repeat[0]}"
            )
    if unavailable:
        return {"state": "unavailable", "problems": unavailable}
    problems = l2l_pref_check._differences(ga1, l2l)
    return {"state": "not_equal" if problems else "equal", "problems": problems}


def judge_kernels(
    ga: Optional[List[Dict[str, Any]]], l2l: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Do accumulation and L2L get the same default SDPA kernel? (addendum)

    A probe lists, per attention signature, the backends whose result is bit equal to the
    default call. ``same`` when both arms picked the same sets of backends (how many signatures
    each saw does not matter), ``differ`` when not, ``unknown``
    when a probe is missing or a signature matched no backend. ``common`` lists the backends that
    match the default in every signature of both arms: equal results of two backends can be
    chance, so ``differ`` with a non-empty ``common`` does not show that the arms ran different
    kernels (found in the second run, after the rule was written).
    """
    def lists(probe: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        if not probe or any(not item["equal_to"] for item in probe):
            return None
        return [list(names) for names in sorted({tuple(sorted(i["equal_to"])) for i in probe})]

    ga_kernels, l2l_kernels = lists(ga), lists(l2l)
    if ga_kernels is None or l2l_kernels is None:
        return {"state": "unknown", "ga": ga_kernels, "l2l": l2l_kernels}
    state = "same" if ga_kernels == l2l_kernels else "differ"
    every = [set(names) for names in ga_kernels + l2l_kernels]
    common = sorted(set.intersection(*every))
    return {"state": state, "ga": ga_kernels, "l2l": l2l_kernels, "common": common}


def _first_step(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {"grads": arm["grads"][:1], "losses": arm["losses"][:1]}


def judge_floor(
    ga1: Optional[Dict[str, Any]], floor: Optional[Dict[str, Any]], l2l: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Control 2: L2L against accumulation, with the second SDPA backend as the noise floor.

    The A3-07 rule (``l2l_pref_check._within_floor``): per step and tensor the gap of L2L may be
    1.5 x the gap of ``floor``, and a floor of 0 leaves no room. ``state`` is ``within`` when that
    holds on every step; ``step0_within`` is the same test on step 0 alone, where all arms start
    from the same adapters. ``no_data`` when an arm is missing.
    """
    arms = {"native_ga1": ga1, "native_floor": floor, "native_l2l": l2l}
    missing = [f"{name}: no valid arm" for name, arm in arms.items() if arm is None]
    if missing:
        return {"state": "no_data", "step0_within": None, "problems": missing}
    problems = l2l_pref_check._within_floor(ga1, floor, l2l)
    step0 = l2l_pref_check._within_floor(_first_step(ga1), _first_step(floor), _first_step(l2l))
    return {
        "state": "outside" if problems else "within",
        "step0_within": not step0,
        "problems": problems,
        "max_floor": l2l_pref_check.max_abs_diff(ga1, floor)["grad"],
        "max_gap": l2l_pref_check.max_abs_diff(ga1, l2l)["grad"],
    }


#: Row of the outcome table -> what it means (rule, "Таблица исходов"). The report quotes the rule.
CONCLUSIONS = {
    1: "rounding, no bug; the readiness criterion of the preference methods moves to the A3-07 "
       "tolerance by a new rule before a new run (not in this ticket)",
    2: "rounding larger than the accumulation noise; the tolerance is a user decides matter",
    3: "a bug in the shared L2L path is likely; control 3 names the layer; "
       "fixed in a separate ticket",
    4: "the gap is of the order of the backend noise; precision not separated, a stated limit",
    5: "cause not established",
}


def judge_outcome(float32: str, floor: str) -> Dict[str, Any]:
    """The row of the ticket's outcome table from the states of controls 1 and 2.

    ``trace_needed`` and ``trace_dtype`` say whether control 3 runs and in which type: float32
    when control 1 could speak and found a gap, the native type when it was unavailable.
    """
    trace = {"equal": (False, None), "not_equal": (True, "float32")}.get(float32, (True, "native"))
    out: Dict[str, Any] = {"trace_needed": trace[0], "trace_dtype": trace[1]}
    if floor == "no_data":
        return {**out, "row": None, "conclusion": "undefined: control 2 has no valid data"}
    if float32 == "not_equal":
        row = 3
    elif float32 == "equal":
        row = 1 if floor == "within" else 2
    else:
        row = 4 if floor == "within" else 5
    return {**out, "row": row, "conclusion": CONCLUSIONS[row]}


def first_diverging_layer(
    ga: Optional[Dict[str, Any]], l2l: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Control 3: the first output, in order embedding, layer 0, layer 1, ..., that differs bitwise.

    A trace is ``{"embed": tensor, "layers": [tensor, ...], "mask": bool tensor [batch, seq]}``
    recorded at step 0. Accumulation runs with the padding mask and L2L without it, so outputs at
    padded positions differ by design and never reach the loss: only positions with ``mask`` set
    are compared, and the two masks must be equal. ``first`` is ``"embed"`` or a layer index;
    ``differing`` lists every layer index that differs.
    """
    import torch

    if ga is None or l2l is None:
        return {"state": "no_data", "first": None, "reason": "a trace is missing"}
    mask = ga.get("mask")
    if (mask is None) != (l2l.get("mask") is None) or (
        mask is not None and not torch.equal(mask, l2l["mask"])
    ):
        return {"state": "no_data", "first": None, "reason": "the padding masks differ"}
    pairs = [("embed", ga["embed"], l2l["embed"])] + [
        (i, a, b) for i, (a, b) in enumerate(zip(ga["layers"], l2l["layers"]))
    ]
    if len(ga["layers"]) != len(l2l["layers"]) or any(a.shape != b.shape for _, a, b in pairs):
        reason = "recorded shapes or layer counts differ"
        return {"state": "no_data", "first": None, "reason": reason}
    if mask is not None:
        pairs = [(name, a[mask], b[mask]) for name, a, b in pairs]
    differing = [name for name, a, b in pairs if not torch.equal(a, b)]
    if not differing:
        return {"state": "equal", "first": None, "differing": []}
    first = differing[0]
    a, b = next((a, b) for name, a, b in pairs if name == first)
    return {
        "state": "differs",
        "first": first,
        "differing": [name for name in differing if name != "embed"],
        "embed_differs": "embed" in differing,
        "max_abs_at_first": float((a.double() - b.double()).abs().max()),
    }


# --------------------------------------------------------------------------
# The driver: one process per arm, an ArmWatch around each, judges at the end
# --------------------------------------------------------------------------
MAX_ATTEMPTS = 3
METHOD = "dpo"
K = 1
#: name -> (path, dtype, sdpa). ``dtype`` is ``native`` (what the code picks) or ``f32`` (patched).
ARMS = {
    "native_ga1": ("ga", "native", "MATH"),
    "native_ga2": ("ga", "native", "MATH"),
    "native_l2l": ("l2l", "native", "MATH"),
    "native_floor": ("ga", "native", "EFFICIENT_ATTENTION"),
    "f32_ga1": ("ga", "f32", "MATH"),
    "f32_ga2": ("ga", "f32", "MATH"),
    "f32_l2l": ("l2l", "f32", "MATH"),
    "f32u_ga1": ("ga", "f32", None),
    "f32u_ga2": ("ga", "f32", None),
    "f32u_l2l": ("l2l", "f32", None),
    "unpinned_ga": ("ga", "native", None),
    "unpinned_l2l": ("l2l", "native", None),
}


def apply_dtype_patch(mode: str) -> Dict[str, Any]:
    """Record what the stream dtype resolver returns; in ``f32`` mode force float32 and no autocast.

    The two substitutions of the rule (control 1): ``resolve_stream_dtype`` and the
    ``bf16_fp16_flags`` the DPO wrapper reads. Both are looked up at call time, so setting them
    before ``setup()`` works.
    """
    layer_stream = importlib.import_module("soup_cli.utils.layer_stream")
    dpo = importlib.import_module("soup_cli.trainer.dpo")
    seen: Dict[str, Any] = {}
    real = layer_stream.resolve_stream_dtype

    def resolve(device: str = "cuda") -> str:
        value = "float32" if mode == "f32" else real(device)
        seen["stream_dtype"] = value
        return value

    layer_stream.resolve_stream_dtype = resolve
    if mode == "f32":
        dpo.bf16_fp16_flags = lambda device, **_kw: (False, False)
    return seen


def sdpa_context(backend: Optional[str]) -> Any:
    if backend is None:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(getattr(SDPBackend, backend))


def _first(module: Any, store: Dict[str, Any], key: str, make: Any, *, pre: bool = False) -> Any:
    """Hook that stores ``make(...)`` on the first call only."""

    def hook(*call: Any) -> None:
        if key not in store:
            store[key] = make(*call)

    if pre:
        return module.register_forward_pre_hook(hook, with_kwargs=True)
    return module.register_forward_hook(hook)


def install_probes(model: Any, *, trace: bool) -> Dict[str, Any]:
    """Hooks that read the types seen in the first call of layer 0 and, if asked, every output.

    The types come from inside the call because outside it the streamed weights are placeholders.
    Returns the dict the hooks fill: ``types`` and, with ``trace``, ``embed`` and ``layers``.
    """
    from soup_cli.utils.l2l_step import model_parts

    _causal, owner = model_parts(model)
    seen: Dict[str, Any] = {"types": {}, "layers": {}}
    types = seen["types"]
    layer0 = owner.layers[0]

    def hidden_of(args: Any, kwargs: Any) -> Any:
        return args[0] if args else kwargs["hidden_states"]

    _first(layer0, types, "layer_in", lambda m, a, kw: str(hidden_of(a, kw).dtype), pre=True)
    _first(layer0, types, "layer_out", lambda m, a, out: str(
        (out[0] if isinstance(out, tuple) else out).dtype))
    q_proj = next(
        (m for n, m in layer0.named_modules() if n.endswith("self_attn.q_proj")), None
    )
    if q_proj is not None and hasattr(q_proj, "base_layer"):
        _first(q_proj, types, "base_weight",
               lambda m, a, kw: str(m.base_layer.weight.dtype), pre=True)
        _first(q_proj, types, "lora_weight",
               lambda m, a, kw: str(m.lora_A["default"].weight.dtype), pre=True)
    if trace:
        _first(owner.embed_tokens, seen, "embed", lambda m, a, out: out.detach().cpu().clone())
        for index, layer in enumerate(owner.layers):
            _first(layer, seen["layers"], index, lambda m, a, out: (
                (out[0] if isinstance(out, tuple) else out).detach().cpu().clone()))
    return seen


SDPA_BACKENDS = ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION")
MAX_SIGNATURES = 6


def _snap(value: Any) -> Any:
    return value.detach().clone() if hasattr(value, "detach") else value


def install_sdpa_recorder() -> Dict[str, Any]:
    """Wrap ``scaled_dot_product_attention`` to keep the first call of each distinct signature.

    A signature is the query shape and type, whether a mask is passed and ``is_causal``. Returns
    ``{"seen": {signature: call}, "real": the original function}``; ``probe_sdpa`` reads it and
    ``remove_sdpa_recorder`` puts the original back.
    """
    import torch.nn.functional as functional

    real = functional.scaled_dot_product_attention
    state: Dict[str, Any] = {"seen": {}, "real": real}

    def recorder(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        mask = kwargs.get("attn_mask", args[0] if args else None)
        signature = (
            tuple(query.shape), str(query.dtype), mask is not None,
            bool(kwargs.get("is_causal", False)),
        )  # fmt: skip
        if signature not in state["seen"] and len(state["seen"]) < MAX_SIGNATURES:
            state["seen"][signature] = (
                tuple(_snap(x) for x in (query, key, value)),
                tuple(_snap(x) for x in args),
                {name: _snap(x) for name, x in kwargs.items()},
            )
        return real(query, key, value, *args, **kwargs)

    functional.scaled_dot_product_attention = recorder
    state["functional"] = functional
    return state


def remove_sdpa_recorder(state: Dict[str, Any]) -> None:
    state["functional"].scaled_dot_product_attention = state["real"]


def probe_sdpa(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run each recorded call with no pin and under each backend alone; list the equal ones."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    real = state["real"]
    out: List[Dict[str, Any]] = []
    with torch.no_grad():
        for signature, (qkv, args, kwargs) in state["seen"].items():
            default = real(*qkv, *args, **kwargs)
            matches: Dict[str, str] = {}
            for name in SDPA_BACKENDS:
                try:
                    with sdpa_kernel(getattr(SDPBackend, name)):
                        result = real(*qkv, *args, **kwargs)
                except Exception as error:  # noqa: BLE001 - the reason is the record
                    matches[name] = "unavailable: " + str(error).strip().splitlines()[0][:120]
                    continue
                if torch.equal(result, default):
                    matches[name] = "equal"
                else:
                    gap = float((result.double() - default.double()).abs().max())
                    matches[name] = f"differs by {gap:.3e}"
            out.append({
                "signature": {
                    "shape": list(signature[0]), "dtype": signature[1],
                    "mask": signature[2], "causal": signature[3],
                },
                "matches": matches,
                "equal_to": [name for name, kind in matches.items() if kind == "equal"],
            })  # fmt: skip
    return out


def record_first_mask(trainer: Any, probes: Dict[str, Any]) -> None:
    """Keep the attention mask of the first collated batch, the one step 0 trains on."""
    real = trainer.data_collator

    def collate(features: Any) -> Any:
        batch = real(features)
        if "mask" not in probes:
            probes["mask"] = batch["attention_mask"].detach().cpu().clone().bool()
        return batch

    trainer.data_collator = collate


def run_arm(args: argparse.Namespace) -> None:
    """Train one arm in this process and write gradients, losses, log, type record and trace."""
    import torch
    from transformers import TrainerCallback

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ["SOUP_LAYER_STREAM_CACHE_DIR"] = str(work / "cache")
    if args.device == "cpu":
        os.environ["ACCELERATE_USE_CPU"] = "true"
        l2l_pref_check._pin_training_device_to_cpu()
    seen = apply_dtype_patch(args.dtype)
    rows = l2l_pref_check.read_rows(METHOD)
    cfg = l2l_pref_check.build_config(METHOD, args.base, K, args.path == "l2l", str(work / "out"))
    wrapper_class = getattr(importlib.import_module("soup_cli.trainer.dpo"), "DPOTrainerWrapper")
    os.chdir(work)
    wrapper = wrapper_class(cfg, device=args.device)
    wrapper.setup({"train": rows})
    try:
        l2l_pref_check._start_adapters(wrapper.model, Path(args.start))
        probes = install_probes(wrapper.model, trace=args.trace)
        record_first_mask(wrapper.trainer, probes)
        recorder = install_sdpa_recorder()
        grads: List[Dict[str, Any]] = []

        class Recorder(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
                grads.append(
                    {
                        n: p.grad.detach().cpu().clone()
                        for n, p in model.named_parameters()
                        if "lora_" in n and p.grad is not None
                    }
                )

        wrapper.trainer.add_callback(Recorder())
        try:
            with sdpa_context(args.sdpa):
                wrapper.trainer.train()
        finally:
            remove_sdpa_recorder(recorder)
        sdpa_probe = probe_sdpa(recorder)
        log_rows = [row for row in wrapper.trainer.state.log_history if "loss" in row]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"grads": grads, "losses": [float(r["loss"]) for r in log_rows]},
            out / f"{args.arm}.pt",
        )
        (out / f"{args.arm}.log.json").write_text(json.dumps(log_rows, indent=1))
        targs = wrapper.trainer.args
        l2l_state = getattr(wrapper.trainer, "_soup_l2l", None)
        record = {
            **probes["types"],
            "stream_dtype": seen.get("stream_dtype"),
            "train_bf16": bool(targs.bf16),
            "train_fp16": bool(targs.fp16),
            "mixed_precision": str(wrapper.trainer.accelerator.mixed_precision),
            "grads": sorted({str(t.dtype) for step in grads for t in step.values()}),
            "l2l_state_dtype": str(l2l_state.dtype) if l2l_state is not None else None,
            "attn_impl": str(getattr(wrapper.model.config, "_attn_implementation", None)),
            "sdpa": args.sdpa,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "dtype_mode": args.dtype,
            "path": args.path,
            "sdpa_probe": sdpa_probe,
        }
        (out / f"{args.arm}.types.json").write_text(json.dumps(record, indent=1))
        if args.trace:
            layers = probes["layers"]
            torch.save(
                {
                    "embed": probes["embed"],
                    "layers": [layers[i] for i in sorted(layers)],
                    "mask": probes["mask"],
                },
                out / f"{args.arm}.trace.pt",
            )
    finally:
        close = getattr(wrapper, "_close_stream_runtime", None)
        if close is not None:
            close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_one_arm(
    args: argparse.Namespace, name: str, path: str, dtype: str, sdpa: Optional[str],
    *, trace: bool = False,
) -> Dict[str, Any]:
    """One arm with up to ``MAX_ATTEMPTS`` tries; ``void`` and crashes are rerun, never dropped.

    Returns ``{"attempts": [...], "data": arm or None, "types": record or None, "trace": ...}``.
    ``unknown`` on a required check stops the tries: that arm gives no data.
    """
    import torch
    from arm_validity import ArmWatch

    result: Dict[str, Any] = {"attempts": [], "data": None, "types": None, "trace": None}
    env = {**os.environ, "PYTHONPATH": str(l2l_pref_check.ROOT / "src")}
    for index in range(MAX_ATTEMPTS):
        attempt_dir = Path(args.out) / name / f"attempt{index}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(Path(__file__).resolve()), "arm", "--arm", name, "--path", path,
            "--dtype", dtype, "--base", args.base, "--device", args.device,
            "--work", str(attempt_dir / f"{name}-work"), "--out", str(attempt_dir),
            "--start", str(Path(args.out) / "start.pt"),
        ] + (["--sdpa", sdpa] if sdpa else []) + (["--trace"] if trace else [])  # fmt: skip
        log = attempt_dir / f"{name}.out"
        with ArmWatch(
            arm=name, round=index + 1, run=1, label=args.label,
            required=l2l_pref_check.REQUIRED_CHECKS, model_path=args.model_path,
        ) as watch:  # fmt: skip
            with log.open("w") as handle:
                done = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False
                )  # fmt: skip
        record = watch.record
        (attempt_dir / f"{name}.watch.json").write_text(json.dumps(record, indent=1, default=str))
        entry: Dict[str, Any] = {
            "attempt": index, "outcome": record["outcome"], "returncode": done.returncode,
            "dir": str(attempt_dir.relative_to(args.out)),
        }
        result["attempts"].append(entry)
        if record["outcome"] == "void" or done.returncode != 0:
            entry["excluded"] = (
                "arm is void" if record["outcome"] == "void"
                else "crashed: " + " | ".join(log.read_text().strip().splitlines()[-3:])
            )
            continue
        if record["outcome"] != "ok":
            entry["excluded"] = f"required check {record['outcome']}: no data"
            return result
        result["data"] = torch.load(attempt_dir / f"{name}.pt")
        result["types"] = json.loads((attempt_dir / f"{name}.types.json").read_text())
        entry["sha256"] = _sha256(attempt_dir / f"{name}.pt")
        if trace:
            result["trace"] = torch.load(attempt_dir / f"{name}.trace.pt")
        return result
    return result


def _pair(
    ref: Optional[Dict[str, Any]], other: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Largest gap per step, for the report (not the verdict)."""
    if ref is None or other is None:
        return None
    steps = []
    for step, (g_ref, g_other) in enumerate(zip(ref["grads"], other["grads"])):
        gap = max(
            (float((g_ref[n].double() - g_other[n].double()).abs().max())
             for n in g_ref.keys() & g_other.keys()),
            default=0.0,
        )
        both = step < len(ref["losses"]) and step < len(other["losses"])
        loss = abs(ref["losses"][step] - other["losses"][step]) if both else None
        steps.append({"step": step, "grad": gap, "loss": loss})
    return {"steps": steps, "overall": l2l_pref_check.max_abs_diff(ref, other)}


def run_campaign(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = l2l_pref_check.environment(args.base)
    (out / "environment.json").write_text(json.dumps(env, indent=1))
    runs: Dict[str, Any] = {}
    for name, (path, dtype, sdpa) in ARMS.items():
        runs[name] = run_one_arm(args, name, path, dtype, sdpa)
        print(f"{name}: {[a['outcome'] for a in runs[name]['attempts']]}", flush=True)
    data = {name: run["data"] for name, run in runs.items()}
    types = {name: run["types"] for name, run in runs.items()}
    control1 = judge_float32(data["f32_ga1"], data["f32_ga2"], data["f32_l2l"], types)
    control1u = judge_float32(
        data["f32u_ga1"], data["f32u_ga2"], data["f32u_l2l"], types, prefix="f32u"
    )
    kernels = judge_kernels(
        (types["unpinned_ga"] or {}).get("sdpa_probe"),
        (types["unpinned_l2l"] or {}).get("sdpa_probe"),
    )
    control2 = judge_floor(data["native_ga1"], data["native_floor"], data["native_l2l"])
    outcome = judge_outcome(control1["state"], control2["state"])
    control3: Dict[str, Any] = {"state": "not_needed", "reason": "control 1 is bit equal"}
    if outcome["trace_needed"]:
        mode = "f32" if outcome["trace_dtype"] == "float32" else "native"
        traced = {
            name: run_one_arm(args, name, path, mode, "MATH", trace=True)
            for name, path in (("trace_ga", "ga"), ("trace_l2l", "l2l"))
        }
        runs.update(traced)
        types.update({name: run["types"] for name, run in traced.items()})
        control3 = first_diverging_layer(traced["trace_ga"]["trace"], traced["trace_l2l"]["trace"])
        control3["dtype"] = outcome["trace_dtype"]
        control3["shapes"] = {
            name: list(run["trace"]["embed"].shape) if run["trace"] else None
            for name, run in traced.items()
        }
    pairs = {
        "native_l2l_vs_ga1": _pair(data["native_ga1"], data["native_l2l"]),
        "native_floor_vs_ga1": _pair(data["native_ga1"], data["native_floor"]),
        "native_ga2_vs_ga1": _pair(data["native_ga1"], data["native_ga2"]),
        "f32_l2l_vs_ga1": _pair(data["f32_ga1"], data["f32_l2l"]),
        "f32_ga2_vs_ga1": _pair(data["f32_ga1"], data["f32_ga2"]),
        "f32u_l2l_vs_ga1": _pair(data["f32u_ga1"], data["f32u_l2l"]),
        "f32u_ga2_vs_ga1": _pair(data["f32u_ga1"], data["f32u_ga2"]),
        "unpinned_l2l_vs_ga": _pair(data["unpinned_ga"], data["unpinned_l2l"]),
    }
    losses = {name: arm["losses"] for name, arm in data.items() if arm is not None}
    summary = {
        "control1": control1, "control1u": control1u, "kernels": kernels,
        "control2": control2, "outcome": outcome, "control3": control3,
        "pairs": pairs, "losses": losses, "types": types,
        "attempts": {name: run["attempts"] for name, run in runs.items()},
    }
    (out / "diagnosis.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"control1u={control1u['state']} kernels={kernels['state']}", flush=True)
    print(f"control1={control1['state']} control2={control2['state']} "
          f"row={outcome['row']} control3={control3['state']}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    arm = sub.add_parser("arm", help="train one arm in this process")
    arm.add_argument("--arm", required=True)
    arm.add_argument("--path", choices=["ga", "l2l"], required=True)
    arm.add_argument("--dtype", choices=["native", "f32"], required=True)
    arm.add_argument("--sdpa", choices=["MATH", "EFFICIENT_ATTENTION"], default=None)
    arm.add_argument("--trace", action="store_true")
    arm.add_argument("--base", default=l2l_pref_check.DEFAULT_BASE)
    arm.add_argument("--device", default="cuda")
    arm.add_argument("--work", required=True)
    arm.add_argument("--out", required=True)
    arm.add_argument("--start", required=True)
    run = sub.add_parser("run", help="every arm, the judges and control 3 when needed")
    run.add_argument("--base", default=l2l_pref_check.DEFAULT_BASE)
    run.add_argument("--device", default="cuda")
    run.add_argument("--out", required=True)
    run.add_argument("--label", default="synthetic data, SmolLM2-135M")
    run.add_argument(
        "--model-path",
        default=os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface"),
    )
    args = parser.parse_args(argv)
    if args.command == "arm":
        run_arm(args)
    else:
        run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
