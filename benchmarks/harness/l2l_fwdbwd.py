#!/usr/bin/env python3
"""Ticket A5-03: one forward and backward pass, L2L against gradient accumulation (GA).

Not a training run. For each k in ``--ks`` (1, 2, 4) the same rows are cut into windows of k
rows, the last window incomplete. Every window gets one forward and backward in each arm,
from the same adapters, and the LoRA gradients and the summed loss are compared per window:

* ``ga1`` and ``ga2``: GA twice, A against A, bitwise.
* ``l2l``: the shipped ``L2LStep`` over the window, padded the way the trainer pads it.
* ``ga_floor`` (bf16 only): GA on a second SDPA backend, the noise floor.

The tolerance is the one accepted in A3-07 and is not loosened here: bf16 at most 1.5 x the
floor, fp32 at most 1e-4, A against A bit-equal. The comparison code is the stand's
(``l2l_mistral_stand.decide``); this script adds the single-window seam, the rows (different
lengths, ``-100`` labels, an incomplete last window), and the counter findings: empty
``gemm_4bit`` path counters, layer loads that grow with k, and NF4 restores that stay at zero.

The result carries its card label (``--card``); results from different cards are never
merged. Benchmark code, not shipped. torch is imported inside functions.
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_mistral_stand as stand  # noqa: E402

CARDS = ("8gb", "24gb")
DEFAULT_KS = (1, 2, 4)
FwdBwdRefusedError = stand.StandRefusedError
Row = Tuple[Any, Any]  # (input_ids [1, L], labels [1, L])


# --------------------------------------------------------------------------
# rows and windows
# --------------------------------------------------------------------------
def build_rows(
    lengths: Sequence[int], masked_spans: Sequence[Sequence[tuple]], vocab: int, seed: int = 5
) -> List[Row]:
    """Rows of random ids, one batch row each, with ``-100`` over the masked spans."""
    data = stand.SyntheticRows(lengths, masked_spans, vocab, seed=seed)
    return [
        (data[i]["input_ids"].unsqueeze(0), data[i]["labels"].unsqueeze(0))
        for i in range(len(data))
    ]


def cut_windows(rows: Sequence[Row], k: int) -> List[List[Row]]:
    return [list(rows[i : i + k]) for i in range(0, len(rows), k)]


def check_case_shape(rows: Sequence[Row], k: int) -> None:
    """Refuse rows that would not test what the ticket asks: different lengths, masked labels,
    and (for k above 1) an incomplete last window."""
    if len({int(ids.shape[1]) for ids, _ in rows}) < 2:
        raise FwdBwdRefusedError("the rows must have different lengths")
    if not any(bool((labels == -100).any()) for _, labels in rows):
        raise FwdBwdRefusedError("the rows must carry labels of -100")
    if k > 1 and len(rows) % k == 0:
        raise FwdBwdRefusedError(
            f"{len(rows)} rows fill every window of {k}; use a count that leaves an "
            "incomplete last window"
        )


def n_items(window: Sequence[Row]) -> int:
    """HF's ``num_items_in_batch``: labels after the shift that are not ``-100``."""
    return sum(int((labels[:, 1:] != -100).sum()) for _, labels in window)


# --------------------------------------------------------------------------
# counter findings and verdict
# --------------------------------------------------------------------------
def counter_findings(facts: Dict[str, Any], *, quant: str) -> List[Dict[str, str]]:
    """What the counters say that a pass must not hide. Each finding fails the run."""
    found: List[Dict[str, str]] = []
    if quant == "nf4":
        for arm in ("ga1", "l2l"):
            if not facts["gemm"].get(arm):
                found.append({
                    "kind": "gemm_empty", "arm": arm,
                    "note": f"the gemm_4bit path counter of arm {arm} is empty; "
                    "an empty counter proves nothing about the path",
                })
    loads = facts["loads_first_full_window"]
    ks = sorted(loads)
    if len(ks) < 2 or not loads[ks[0]]:
        found.append({
            "kind": "loads_missing",
            "note": "layer loads of a full window are missing for two values of k "
            f"(or are zero at the smallest); got {loads}",
        })
    elif loads[ks[-1]] >= (ks[-1] / ks[0]) * loads[ks[0]]:
        found.append({
            "kind": "loads_scale_with_k",
            "note": f"layer loads grew with k ({loads}); layer-major order should load a "
            "layer once per window",
        })
    if quant == "nf4" and not sum(facts["restores"].values()):
        found.append({
            "kind": "restores_zero",
            "note": "discrepancy: NF4 restores are 0 on a quantized model "
            f"({facts['restores']}); the restore path was not exercised, this is not a success",
        })
    return found


def overall(cases: Dict[int, Dict[str, Any]], findings: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    failed = [k for k, verdict in sorted(cases.items()) if not verdict["pass"]]
    if failed:
        first = cases[failed[0]]
        return {"pass": False, "status": first["status"],
                "note": f"k={failed[0]}: {first.get('note', '')}"}
    if findings:
        return {"pass": False, "status": "counter_findings",
                "note": "; ".join(f["note"] for f in findings)}
    return {"pass": True, "status": "within_tolerance", "note": ""}


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
@dataclass
class ArmRecord(stand.RunRecord):
    """A run record with what the counters saw, one entry per window."""

    gemm: Optional[Dict[int, Dict[str, int]]] = None
    loads: List[int] = field(default_factory=list)
    restores: List[int] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)


def _autocast(device: str, dtype: Any) -> Any:
    import torch

    if dtype is None or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=str(device).split(":")[0], dtype=dtype)


class _Stepper:
    """L2LStep and activation store per padded shape, as the trainer keeps them."""

    def __init__(self, model: Any, runtime: Any, dtype: Any, device: str):
        self.model, self.runtime, self.device = model, runtime, device
        self.dtype = dtype
        self.steps: Dict[Tuple[int, int], Any] = {}
        self.store: Any = None

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None

    def run(self, window: Sequence[Row]) -> Tuple[List[Any], Any]:
        import torch

        from soup_cli.trainer.l2l_trainer import _window_tensors
        from soup_cli.utils.l2l_activations import ActivationStore
        from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

        ids_list, label_list = _window_tensors(
            [{"input_ids": ids, "labels": labels} for ids, labels in window], self.device
        )
        shape = tuple(int(dim) for dim in ids_list[0].shape)
        step = self.steps.get(shape)
        if step is None:
            step = L2LStep(self.model, self.runtime, capture_layer_call(self.model, ids_list[0]))
            self.steps[shape] = step
        hidden = int(step.causal.config.hidden_size)
        k = len(window)
        if self.store is None or self.store.shape[:2] != shape or self.store.k < k:
            self.close()
            self.store = ActivationStore(
                n_layers=len(step.layers), k=k, chunk_shape=(shape[0], shape[1], hidden),
                dtype=self.dtype or torch.float32, device=self.device,
                pin=str(self.device).startswith("cuda"),
            )
        losses = step.run(
            ids_list, self.store, n_items=n_items(window), labels=label_list
        )
        return losses, step.counters


def run_arm(
    model: Any, runtime: Any, windows: Sequence[Sequence[Row]], *, kind: str, dtype: Any,
    device: str,
) -> ArmRecord:
    """Every window once, forward and backward, from the current adapters.

    ``kind`` is ``ga`` (each row at its own length, loss over the window's label count) or
    ``l2l`` (the window padded to one shape). Gradients are read and cleared per window.
    """
    import l2l_schedule

    from soup_cli.utils.l2l_step import count_loads

    record = ArmRecord(logs=[], rates=[], grads=[], steps=0, l2l=kind == "l2l")
    stepper = _Stepper(model, runtime, dtype, device) if kind == "l2l" else None
    model.train()
    try:
        for window in windows:
            l2l_schedule.zero_grads(model)
            with _autocast(device, dtype):
                if stepper is not None:
                    losses, counters = stepper.run(window)
                    loads, restores = counters.layer_loads, counters.restores
                else:
                    total = n_items(window)
                    losses = []
                    with count_loads(runtime) as counters:
                        for ids, labels in window:
                            out = model(input_ids=ids, labels=labels, num_items_in_batch=total)
                            out.loss.backward()
                            losses.append(out.loss.detach())
                    loads, restores = counters.layer_loads, counters.restores
            record.logs.append({"loss": float(sum(float(loss) for loss in losses))})
            record.rates.append(0.0)
            record.grads.append(l2l_schedule.collect_grads(model))
            record.loads.append(loads)
            record.restores.append(restores)
            record.sizes.append(len(window))
            record.steps += 1
    finally:
        if stepper is not None:
            stepper.close()
    return record


def merge_counts(
    parts: Sequence[Optional[Dict[int, Dict[str, int]]]]
) -> Optional[Dict[int, Dict[str, int]]]:
    """Sum per-M path counts over several runs; None when none of them was counted."""
    counted = [part for part in parts if part is not None]
    if not counted:
        return None
    merged: Dict[int, Dict[str, int]] = {}
    for part in counted:
        for rows, paths in part.items():
            entry = merged.setdefault(rows, {"custom": 0, "dequant": 0})
            for path, n in paths.items():
                entry[path] = entry.get(path, 0) + n
    return merged


@dataclass
class FwdBwdResult:
    cases: Dict[int, Dict[str, Any]]
    facts: Dict[str, Any]
    findings: List[Dict[str, str]]
    overall: Dict[str, Any]
    logs: Dict[str, str]


def run_fwdbwd(
    model: Any, runtime: Any, rows: Sequence[Row], *, ks: Sequence[int], dtype: Any, mode: str,
    quant: str, device: str, seed: int = 1, path_runtime: Any = None,
    sdpa: Callable[[], Any] = stand.pinned_sdpa, floor_sdpa: Optional[Callable[[], Any]] = None,
) -> FwdBwdResult:
    """All cases: for each k the arms, the verdict of ``stand.decide`` and the counters."""
    import l2l_schedule
    import torch

    l2l_schedule.make_non_vacuous(model)
    start = l2l_schedule.snapshot_adapters(model)
    cases: Dict[int, Dict[str, Any]] = {}
    logs: Dict[str, str] = {}
    gemm_parts: Dict[str, List[Any]] = {}
    loads_first: Dict[int, int] = {}
    ga_loads_first: Dict[int, int] = {}
    restores: Dict[int, int] = {}
    for k in ks:
        check_case_shape(rows, k)
        windows = cut_windows(rows, k)
        arms = [("ga1", "ga", sdpa), ("ga2", "ga", sdpa), ("l2l", "l2l", sdpa)]
        if floor_sdpa is not None:
            arms.append(("ga_floor", "ga", floor_sdpa))
        runs: Dict[str, ArmRecord] = {}
        for name, kind, backend in arms:
            l2l_schedule.restore_adapters(model, start)
            torch.manual_seed(seed)
            with backend(), stand.GemmPathCounter(path_runtime) as counter:
                runs[name] = run_arm(
                    model, runtime, windows, kind=kind, dtype=dtype, device=device
                )
            runs[name].gemm = counter.counts
            gemm_parts.setdefault(name, []).append(counter.counts)
            logs[f"k{k}_{name}.log"] = stand._log_text(runs[name])
        cases[k] = stand.decide(
            runs["ga1"], runs["ga2"], runs["l2l"], mode=mode, floor=runs.get("ga_floor")
        )
        full = next(i for i, size in enumerate(runs["l2l"].sizes) if size == k)
        loads_first[k] = runs["l2l"].loads[full]
        ga_loads_first[k] = runs["ga1"].loads[full]
        restores[k] = sum(runs["l2l"].restores)
    facts = {
        "gemm": {name: merge_counts(parts) for name, parts in gemm_parts.items()},
        "loads_first_full_window": loads_first,
        "ga_loads_first_full_window": ga_loads_first,
        "restores": restores,
        "windows": {k: [len(w) for w in cut_windows(rows, k)] for k in ks},
    }
    findings = counter_findings(facts, quant=quant)
    return FwdBwdResult(
        cases=cases, facts=facts, findings=findings, overall=overall(cases, findings), logs=logs
    )


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def render_report(payload: Dict[str, Any]) -> str:
    overall_ = payload["overall"]
    passport = payload.get("passport") or {}
    lines = [
        f"# L2L against gradient accumulation, forward and backward: {payload['model']}",
        "",
        f"Card: **{payload['card']}**. Results from another card are a separate report.",
        "",
        f"**Verdict: {'PASS' if overall_['pass'] else 'FAIL'}** ({overall_['status']})",
    ]
    if overall_.get("note"):
        lines += ["", overall_["note"]]
    lines += ["", "## Command", "", "```", payload.get("command_line", ""), "```", ""]
    lines += ["## Environment", ""]
    for name, version in (passport.get("packages") or {}).items():
        if version:
            lines.append(f"- {name} {version}")
    lines += [
        f"- GPU: {passport.get('gpu')}",
        f"- Container RAM: {passport.get('container_ram')}",
        f"- Commit: {passport.get('commit')}",
        f"- Unavailable: {passport.get('unavailable')}",
        f"- Quantization: {payload['quant']}, dtype mode: {payload['mode']}",
        "",
    ]
    rows = payload.get("rows", {})
    lines += [
        "## Synthetic rows (random tokens, real weights)", "",
        f"- lengths: {rows.get('lengths')}",
        f"- tokens labelled -100: {rows.get('masked')}",
        f"- windows by k (the last one incomplete): {payload['facts'].get('windows')}",
        "- The result is for this model on this card. No number from it carries over to larger "
        "models.",
    ]
    for k, verdict in sorted(payload["cases"].items()):
        lines += ["", f"## k = {k}", "",
                  f"- verdict: {'PASS' if verdict['pass'] else 'FAIL'} ({verdict['status']})"]
        if verdict.get("note"):
            lines.append(f"- {verdict['note']}")
        a_vs_a = verdict.get("a_vs_a") or {}
        lines.append(f"- A against A bit-equal: {a_vs_a.get('equal')} "
                     f"(max |d| {a_vs_a.get('max_abs')})")
        compare = verdict.get("l2l_vs_ga")
        if compare is not None:
            lines.append(f"- L2L against GA bit-equal: {compare.get('equal')} (information, "
                         f"not the criterion); max |d| {compare.get('max_abs')}")
        lines += stand._tolerance_lines(verdict.get("tolerance"))
    facts = payload["facts"]
    lines += ["", "## Counters", ""]
    for arm, counts in facts["gemm"].items():
        lines.append(f"- gemm_4bit path calls, {arm}: {counts if counts else 'empty'}")
    lines += [
        f"- layer loads of the first full window, L2L, by k: {facts['loads_first_full_window']}",
        f"- layer loads of the same window, GA, by k: {facts.get('ga_loads_first_full_window')}",
        f"- NF4 restores in the L2L arm, by k: {facts['restores']}",
        "", "## Findings", "",
    ]
    if payload["findings"]:
        lines += [f"- {f['kind']}: {f['note']}" for f in payload["findings"]]
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, payload: Dict[str, Any], raw_logs: Dict[str, str]) -> None:
    """Write report.md, records.json and the raw logs; an existing directory is never touched."""
    out = Path(out_dir)
    if out.exists():
        raise FwdBwdRefusedError(f"{out} exists; attempts are never overwritten")
    (out / "raw").mkdir(parents=True)
    (out / "report.md").write_text(render_report(payload), encoding="utf-8")
    (out / "records.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8"
    )
    for name, text in raw_logs.items():
        (out / "raw" / name).write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def make_rows(count: int, seq: int, vocab: int) -> List[Row]:
    data = stand.make_rows(count, seq, vocab)
    return [
        (data[i]["input_ids"].unsqueeze(0), data[i]["labels"].unsqueeze(0))
        for i in range(len(data))
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default=stand.DEFAULT_WEIGHTS)
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    parser.add_argument("--rows", type=int, default=7, help="must leave an incomplete window")
    parser.add_argument("--seq", type=int, default=256, help="longest row, in tokens")
    parser.add_argument("--dtype", choices=("auto", "float32"), default="auto")
    parser.add_argument("--rng-seed", type=int, default=1)
    parser.add_argument("--card", required=True, choices=CARDS,
                        help="label of the card this run is signed with")
    parser.add_argument("--out", required=True)
    parsed = parser.parse_args(argv)
    ks = [int(part) for part in parsed.ks.split(",")]

    stand.set_deterministic_env()  # first, before torch and CUDA start
    try:
        if Path(parsed.out).exists():
            raise FwdBwdRefusedError(f"{parsed.out} exists; attempts are never overwritten")
        probe_rows = [(None, None)] * parsed.rows
        for k in ks:
            if k > 1 and len(probe_rows) % k == 0:
                raise FwdBwdRefusedError(
                    f"--rows {parsed.rows} fills every window of {k}; use a count that leaves "
                    "an incomplete last window"
                )
        import stream_probe

        if not stream_probe.cuda_available():
            print("SKIP: CUDA is required for l2l_fwdbwd.py")
            return 0
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(parsed.weights)
        padded = -(-parsed.seq // 64) * 64
        plan = stand.ram_plan(
            n_layers=int(config.num_hidden_layers), hidden=int(config.hidden_size),
            k=max(ks), batch=1, seq=padded,
            weights_bytes=stand.weights_bytes_estimate(config, parsed.quant),
        )
        for key, value in plan.items():
            print(f"ram_plan {key:<22}{value}")
        from soup_cli.utils.layer_stream import free_ram_bytes

        stand.check_ram(plan, free_ram_bytes())
    except FwdBwdRefusedError as exc:
        print(f"REFUSED: {exc}")
        return 2

    import l2l_measure
    import l2l_probe
    import stream_probe
    import torch

    import soup_cli.utils.layer_stream_runtime as path_runtime
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    stand.enable_deterministic()
    device = "cuda"
    dtype = "float32" if parsed.dtype == "float32" else resolve_stream_dtype(device)
    mode = "fp32" if dtype == "float32" else "bf16"
    args = l2l_probe.parse_args([
        "--mode", "correctness", "--weights", parsed.weights, "--quant", parsed.quant,
        "--seq", str(parsed.seq), "--tier", "ram", "--out", str(Path(parsed.out) / "unused.json"),
    ])
    print("RUN: L2L against GA, forward and backward,", parsed.weights, parsed.quant, parsed.card)
    model, runtime, config, *_ = stream_probe.build(args, device, dtype)
    try:
        rows = make_rows(parsed.rows, parsed.seq, int(config.vocab_size))
        result = run_fwdbwd(
            model, runtime, rows, ks=ks, dtype=getattr(torch, dtype), mode=mode,
            quant=parsed.quant, device=device, seed=parsed.rng_seed, path_runtime=path_runtime,
            floor_sdpa=stand.floor_sdpa if mode == "bf16" else None,
        )
    finally:
        runtime.close()
    argv_full = [sys.executable, *sys.argv]
    payload = {
        "model": parsed.weights, "quant": parsed.quant, "card": parsed.card, "mode": mode,
        "dtype": dtype, "data_kind": "synthetic",
        "command_line": " ".join(argv_full),
        "passport": l2l_measure.launch_passport(argv=argv_full),
        "deterministic": {"CUBLAS_WORKSPACE_CONFIG": stand.CUBLAS_WORKSPACE_CONFIG,
                          "sdpa_pinned": stand.PINNED_SDPA_BACKEND},
        "ram_plan": plan,
        "rows": {"lengths": [int(ids.shape[1]) for ids, _ in rows],
                 "masked": sum(int((labels == -100).sum()) for _, labels in rows)},
        "cases": result.cases, "facts": result.facts, "findings": result.findings,
        "overall": result.overall,
    }
    try:
        write_outputs(Path(parsed.out), payload, result.logs)
    except FwdBwdRefusedError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"verdict {result.overall['status']}: {result.overall['note'] or 'within tolerance'}")
    print(f"wrote {Path(parsed.out) / 'report.md'}")
    return 0 if result.overall["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
