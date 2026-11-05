#!/usr/bin/env python3
"""Ticket A6-06: L2L against gradient accumulation for DPO, ORPO, SimPO and KTO.

The rule is ``benchmarks/l2l-preference-check-rule-a6.md``, committed before the first GPU run.
This module holds the judging code (pure functions over recorded arms) and the driver that
records the arms through the real trainer wrappers. Benchmark code, not shipped. torch is
imported inside functions.

An arm is ``{"grads": [{name: tensor}, ...], "losses": [float, ...]}``: the LoRA gradients
before every optimizer step and the logged loss of every step.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_schedule  # noqa: E402


def _differences(ref: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    """What differs bit for bit between two arms, one sentence each; empty when equal."""
    import torch

    out: List[str] = []
    if len(ref["grads"]) != len(cand["grads"]):
        out.append(f"step count {len(ref['grads'])} against {len(cand['grads'])}")
        return out
    for step, (g_ref, g_cand) in enumerate(zip(ref["grads"], cand["grads"])):
        if g_ref.keys() != g_cand.keys():
            out.append(f"step {step}: gradient keys differ")
            continue
        for name in sorted(g_ref):
            if not torch.equal(g_ref[name], g_cand[name]):
                out.append(f"step {step}: {name} differs")
    if len(ref["losses"]) != len(cand["losses"]):
        out.append(f"loss count {len(ref['losses'])} against {len(cand['losses'])}")
    for step, (a, b) in enumerate(zip(ref["losses"], cand["losses"])):
        if a != b:
            out.append(f"step {step}: loss {a!r} against {b!r}")
    return out


#: The A3-07 factor: a candidate may differ from the reference by 1.5 x the noise floor.
FLOOR_FACTOR = 1.5


def _within_floor(ref: Dict[str, Any], floor: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    """The A3-07 rule with A against A as the floor, per tensor and per step, and on the loss.

    ``floor`` is the second accumulation run. A floor of 0 on a tensor leaves no room: the
    candidate has to equal the reference there. Arms of another shape of record never pass.
    """
    out: List[str] = []
    shapes = [len(arm["grads"]) for arm in (ref, floor, cand)]
    if len(set(shapes)) != 1 or any(
        a["grads"][i].keys() != ref["grads"][i].keys()
        for a in (floor, cand)
        for i in range(shapes[0])
    ):
        return ["the arms do not have the same steps and gradient keys"]
    for step in range(shapes[0]):
        for name in sorted(ref["grads"][step]):
            base = ref["grads"][step][name].double()
            noise = float((floor["grads"][step][name].double() - base).abs().max())
            gap = float((cand["grads"][step][name].double() - base).abs().max())
            if not gap <= FLOOR_FACTOR * noise:
                out.append(
                    f"step {step}: {name} differs by {gap:.3e}, limit {FLOOR_FACTOR * noise:.3e}"
                )
    for step, (a, f, c) in enumerate(zip(ref["losses"], floor["losses"], cand["losses"])):
        noise, gap = abs(f - a), abs(c - a)
        if not gap <= FLOOR_FACTOR * noise:
            out.append(f"step {step}: loss differs by {gap:.3e}, limit {FLOOR_FACTOR * noise:.3e}")
    return out


def _not_worth_comparing(arm: Dict[str, Any], windows: int) -> List[str]:
    """Why an arm cannot back a pass even when it equals the reference: it must be non-vacuous."""
    import math

    import torch

    out: List[str] = []
    if len(arm["grads"]) != windows:
        out.append(f"{len(arm['grads'])} optimizer steps, but the data makes {windows} windows")
    if len(arm["losses"]) != windows:
        out.append(f"{len(arm['losses'])} losses logged, but the data makes {windows} windows")
    if not all(math.isfinite(loss) for loss in arm["losses"]):
        out.append("a loss is not finite")
    tensors = [t for step in arm["grads"] for t in step.values()]
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        out.append("a gradient is not finite")
    elif not any(bool((t != 0).any()) for t in tensors):
        out.append("every gradient is zero")
    names = sorted(
        {n for step in arm["grads"] for n in step if ".default." not in n or ".ref." in n}
    )
    if names:
        out.append(f"gradient outside the policy adapter (.default., never .ref.): {names[0]}")
    return out


def judge_cell(
    ga1: Dict[str, Any], ga2: Dict[str, Any], l2l: Dict[str, Any], *, windows: int
) -> Dict[str, Any]:
    """One cell (method, k): ``l2l`` against ``ga1``, with ``ga2`` as the A-against-A control.

    Tier 1 is bit equality. Tier 2 applies only when ``ga2`` differs from ``ga1``: the card
    shows run-to-run noise, and the noise is the floor for the A3-07 factor. Either tier also
    needs the L2L arm to be non-vacuous (``windows`` steps, finite, not all zero, no ``.ref.``).
    """
    a_vs_a_equal = not _differences(ga1, ga2)
    problems = _differences(ga1, l2l)
    tier: Optional[int] = 1
    if problems and not a_vs_a_equal:
        problems = _within_floor(ga1, ga2, l2l)
        tier = 2
    problems = problems + _not_worth_comparing(l2l, windows)
    return {
        "pass": not problems,
        "tier": tier if not problems else None,
        "a_vs_a_equal": a_vs_a_equal,
        "problems": problems,
    }


def _per_step(rows: Sequence[Dict[str, Any]], key: str) -> List[int]:
    """Per-step amounts from the running totals of a training log."""
    totals = [int(row.get(key, 0)) for row in rows]
    return [b - a for a, b in zip([0] + totals, totals)]


def judge_loads(
    method_logs: Sequence[Dict[str, Any]], sft_logs: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """The method's layer loads per step against L2L SFT with the same k (runtime counters).

    Layer loads must be equal at every step, large loads and weight restores not larger.
    Capture loads depend on the micro-batch shape, which differs by method; they are reported.
    """
    problems: List[str] = []
    layer = {"method": _per_step(method_logs, "l2l_layer_loads"),
             "sft": _per_step(sft_logs, "l2l_layer_loads")}
    out: Dict[str, Any] = {
        "layer_loads": layer,
        "capture_layer_loads": {
            "method": int(method_logs[-1].get("l2l_capture_layer_loads", 0)) if method_logs else 0,
            "sft": int(sft_logs[-1].get("l2l_capture_layer_loads", 0)) if sft_logs else 0,
        },
    }
    if len(method_logs) != len(sft_logs):
        problems.append(f"{len(method_logs)} steps against {len(sft_logs)} steps of SFT")
    else:
        for step, (m, s) in enumerate(zip(layer["method"], layer["sft"])):
            if m != s:
                problems.append(f"step {step}: {m} layer loads against {s} of SFT")
        for key, label in (
            ("l2l_large_loads", "large loads"),
            ("l2l_weight_restores", "weight restores"),
        ):
            m_steps, s_steps = _per_step(method_logs, key), _per_step(sft_logs, key)
            out[key] = {"method": m_steps, "sft": s_steps}
            for step, (m, s) in enumerate(zip(m_steps, s_steps)):
                if m > s:
                    problems.append(f"step {step}: {m} {label} against {s} of SFT")
    out.update(problems=problems, **{"pass": not problems})
    return out


KS = (1, 2, 4)
#: Attempts per cell: the first and at most two reruns after a void arm (rule A6).
MAX_ATTEMPTS = 3


def _judge_attempt(attempt: Dict[str, Any]) -> Dict[str, Any]:
    """``excluded`` with a reason, ``no_verdict`` on an unknown required check, or the verdict."""
    if attempt.get("crashed"):
        return {"state": "excluded", "reason": f"crashed: {attempt['crashed']}"}
    for arm, outcome in attempt["arms"].items():
        if outcome == "void":
            return {"state": "excluded", "reason": f"arm {arm} is void"}
    for arm, outcome in attempt["arms"].items():
        if outcome != "ok":
            reason = f"arm {arm} is {outcome}: unknown required check"
            return {"state": "no_verdict", "reason": reason}
    passed = bool(attempt["equality"]["pass"]) and bool(attempt["loads"]["pass"])
    return {"state": "pass" if passed else "fail", "reason": ""}


def _judge_cell_attempts(cell: Dict[str, Any]) -> Dict[str, Any]:
    excluded: List[Dict[str, Any]] = []
    for index, attempt in enumerate(cell["attempts"][:MAX_ATTEMPTS]):
        judged = _judge_attempt(attempt)
        if judged["state"] == "excluded":
            excluded.append({"attempt": index, "reason": judged["reason"]})
            continue
        return {
            "k": cell["k"],
            "verdict": judged["state"],
            "used_attempt": index,
            "excluded": excluded,
            "reason": judged["reason"],
        }
    return {
        "k": cell["k"], "verdict": "no_verdict", "used_attempt": None, "excluded": excluded,
        "reason": f"no valid attempt in {len(cell['attempts'][:MAX_ATTEMPTS])} tries",
    }


def judge_method(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The verdict of one method from its cells: pass, fail or no_verdict.

    A fail on a valid attempt stands, whatever the other cells say. A method with all cells
    passing, valid and present for every k passes; any other case is no verdict.
    """
    judged = [_judge_cell_attempts(cell) for cell in cells]
    missing = [k for k in KS if k not in {cell["k"] for cell in cells}]
    reasons = [f"no cell for k={k}" for k in missing]
    reasons += [
        f"k={cell['k']}: {cell['reason']}" for cell in judged if cell["verdict"] == "no_verdict"
    ]
    if any(cell["verdict"] == "fail" for cell in judged):
        verdict = "fail"
    elif reasons:
        verdict = "no_verdict"
    else:
        verdict = "pass"
    return {"verdict": verdict, "cells": judged, "reason": "; ".join(reasons)}


# --------------------------------------------------------------------------
# The driver: one process per arm, an ArmWatch around each, records after every cell
# --------------------------------------------------------------------------
DEFAULT_BASE = "HuggingFaceTB/SmolLM2-135M"
ARMS = ("ga1", "ga2", "l2l", "sft")
#: Checks that gate an arm (rule A6); ``foreign_reads`` is recorded but not required.
REQUIRED_CHECKS = ("suspend", "power")
BATCH_SIZE = 2
ROOT = Path(__file__).resolve().parents[2]

#: method -> (module, wrapper class, synthetic data file). Betas and gammas stay at the defaults.
METHODS = {
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper", "dpo_pairs_synthetic.jsonl"),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", "orpo_pairs_synthetic.jsonl"),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper", "simpo_pairs_synthetic.jsonl"),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper", "kto_synthetic.jsonl"),
}


def read_rows(method: str) -> List[Dict[str, Any]]:
    """The synthetic rows of one method, as the trainer wrappers take them."""
    path = ROOT / "examples" / "data" / METHODS[method][2]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sft_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The same rows as chat turns: the prompt, then the chosen answer (KTO: the completion)."""
    out = []
    for row in rows:
        answer = row["chosen"] if "chosen" in row else row["completion"]
        out.append(
            {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return out


def windows_for(rows: int, k: int) -> int:
    """Optimizer steps of one epoch: micro-batches of ``BATCH_SIZE`` rows, windows of k."""
    micro = -(-rows // BATCH_SIZE)
    return -(-micro // k)


def build_config(task: str, base: str, k: int, l2l: bool, out_dir: str) -> Any:
    import yaml

    from soup_cli.config.loader import load_config_from_string

    training: Dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": k,
        "quantization": "none",
        "stream_layers": True,
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "seed": 1,
        "lr": 0.01,
        "optimizer": "sgd",
        "scheduler": "constant",
        "warmup_ratio": 0.0,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }
    if l2l:
        training["l2l"] = True
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": task,
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 256, "chat_template": "chatml"},
                "training": training,
                "output": out_dir,
            }
        )
    )


def _lora_state(model: Any) -> Dict[str, Any]:
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def _start_adapters(model: Any, path: Path) -> None:
    """Every arm of a method starts from the same adapters, policy and reference alike.

    The first arm makes the policy's ``lora_B`` non-zero (PEFT starts it at zero, which makes
    every ``lora_A`` gradient zero) and writes the state; the others restore it.
    """
    import torch

    params = _lora_state(model)
    if not path.exists():
        l2l_schedule.make_non_vacuous(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({n: p.detach().cpu().clone() for n, p in params.items()}, path)
        return
    saved = torch.load(path)
    if set(saved) != set(params):
        raise RuntimeError("the saved start state does not match this model's LoRA parameters")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(saved[name].to(param.device, param.dtype))


def _pin_training_device_to_cpu() -> None:
    """A Mac would hand the batch to MPS while the streamed model sits on the CPU."""
    import torch
    from transformers import TrainingArguments

    original = TrainingArguments.device.fget

    def cpu_device(self: Any) -> Any:
        original(self)  # still sets up the distributed state
        return torch.device("cpu")

    TrainingArguments.device = property(cpu_device)


def run_arm(args: argparse.Namespace) -> None:
    """Train one arm in this process and write its gradients, losses and training log."""
    import torch
    from transformers import TrainerCallback

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ["SOUP_LAYER_STREAM_CACHE_DIR"] = str(work / "cache")
    if args.device == "cpu":
        os.environ["ACCELERATE_USE_CPU"] = "true"
        _pin_training_device_to_cpu()
    task = "sft" if args.arm == "sft" else args.method
    rows = read_rows(args.method)
    rows = sft_rows(rows) if args.arm == "sft" else rows
    cfg = build_config(task, args.base, args.k, args.arm in ("l2l", "sft"), str(work / "out"))
    module, name, _data = METHODS[args.method]
    wrapper_class = (
        getattr(importlib.import_module("soup_cli.trainer.sft"), "SFTTrainerWrapper")
        if args.arm == "sft"
        else getattr(importlib.import_module(module), name)
    )
    os.chdir(work)
    wrapper = wrapper_class(cfg, device=args.device)
    wrapper.setup({"train": rows})
    try:
        if args.arm != "sft":
            _start_adapters(wrapper.model, Path(args.start))
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
        wrapper.trainer.train()
        log_rows = [row for row in wrapper.trainer.state.log_history if "loss" in row]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"grads": grads, "losses": [float(r["loss"]) for r in log_rows]},
            out / f"{args.arm}.pt",
        )
        (out / f"{args.arm}.log.json").write_text(json.dumps(log_rows, indent=1))
    finally:
        close = getattr(wrapper, "_close_stream_runtime", None)
        if close is not None:
            close()


def max_abs_diff(ref: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    """Largest gradient and loss difference between two arms, for the record (not the verdict)."""
    grad = 0.0
    for g_ref, g_cand in zip(ref["grads"], cand["grads"]):
        for name in g_ref.keys() & g_cand.keys():
            grad = max(grad, float((g_ref[name].double() - g_cand[name].double()).abs().max()))
    loss = max((abs(a - b) for a, b in zip(ref["losses"], cand["losses"])), default=0.0)
    return {"grad": grad, "loss": loss}


def environment(base: str) -> Dict[str, Any]:
    """Versions, GPU, RAM and commit of this session, for the report."""
    import platform
    import subprocess

    import torch
    import transformers

    def run(*command: str) -> str:
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=30)
            return done.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    import peft
    import trl

    ram = ""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        ram = meminfo.read_text().splitlines()[0]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "base": base,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory if gpu else None,
        "nvidia_smi": run(
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
        ),
        "ram": ram,
        "commit": run("git", "-C", str(ROOT), "rev-parse", "HEAD"),
    }


def _arm_command(args: argparse.Namespace, method: str, k: int, arm: str, attempt_dir: Path):
    return [
        sys.executable, str(Path(__file__).resolve()), "arm",
        "--method", method, "--arm", arm, "--k", str(k), "--base", args.base,
        "--device", args.device, "--work", str(attempt_dir / f"{arm}-work"),
        "--out", str(attempt_dir), "--start", str(Path(args.out) / method / "start.pt"),
    ]  # fmt: skip


def run_attempt(args: argparse.Namespace, method: str, k: int, index: int) -> Dict[str, Any]:
    """One attempt at a cell: four arms, each in its own process under an ArmWatch."""
    import subprocess

    import torch
    from arm_validity import ArmWatch

    attempt_dir = Path(args.out) / method / f"k{k}" / f"attempt{index}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt: Dict[str, Any] = {
        "dir": str(attempt_dir.relative_to(args.out)),
        "arms": {},
        "crashed": None,
        "watch": {},
        "equality": {"pass": False, "tier": None, "a_vs_a_equal": None, "problems": []},
        "loads": {"pass": False, "problems": []},
    }
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for position, arm in enumerate(ARMS):
        log = attempt_dir / f"{arm}.out"
        with ArmWatch(
            arm=arm, round=index + 1, run=position + 1, label=args.label,
            required=REQUIRED_CHECKS, model_path=args.model_path,
        ) as watch:  # fmt: skip
            with log.open("w") as handle:
                done = subprocess.run(
                    _arm_command(args, method, k, arm, attempt_dir),
                    stdout=handle, stderr=subprocess.STDOUT, env=env, check=False,
                )  # fmt: skip
        record = watch.record
        (attempt_dir / f"{arm}.watch.json").write_text(json.dumps(record, indent=1, default=str))
        attempt["arms"][arm] = record["outcome"]
        attempt["watch"][arm] = {
            name: check["outcome"] for name, check in record["checks"].items()
        }
        if done.returncode != 0:
            tail = log.read_text().strip().splitlines()[-3:]
            attempt["crashed"] = f"arm {arm} exited {done.returncode}: " + " | ".join(tail)
            return attempt
    loaded = {arm: torch.load(attempt_dir / f"{arm}.pt") for arm in ARMS}
    rows = len(read_rows(method))
    attempt["equality"] = judge_cell(
        loaded["ga1"], loaded["ga2"], loaded["l2l"], windows=windows_for(rows, k)
    )
    attempt["max_abs"] = {
        "l2l_vs_ga1": max_abs_diff(loaded["ga1"], loaded["l2l"]),
        "ga2_vs_ga1": max_abs_diff(loaded["ga1"], loaded["ga2"]),
    }
    attempt["losses"] = {arm: loaded[arm]["losses"] for arm in ("ga1", "ga2", "l2l")}
    logs = {
        arm: json.loads((attempt_dir / f"{arm}.log.json").read_text()) for arm in ("l2l", "sft")
    }
    attempt["loads"] = judge_loads(logs["l2l"], logs["sft"])
    return attempt


def run_campaign(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = environment(args.base)
    (out / "environment.json").write_text(json.dumps(env, indent=1))
    for method in args.methods:
        cells = []
        for k in args.ks:
            cell: Dict[str, Any] = {"k": k, "attempts": []}
            cells.append(cell)
            for index in range(MAX_ATTEMPTS):
                cell["attempts"].append(run_attempt(args, method, k, index))
                (out / f"records-{method}.json").write_text(
                    json.dumps({"method": method, "environment": env, "cells": cells}, indent=1)
                )
                if _judge_attempt(cell["attempts"][-1])["state"] != "excluded":
                    break
        verdict = judge_method(cells)
        (out / f"records-{method}.json").write_text(
            json.dumps(
                {"method": method, "environment": env, "cells": cells, "verdict": verdict},
                indent=1,
            )
        )
        print(f"{method}: {verdict['verdict']} {verdict['reason']}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    arm = sub.add_parser("arm", help="train one arm in this process")
    arm.add_argument("--method", choices=sorted(METHODS), required=True)
    arm.add_argument("--arm", choices=ARMS, required=True)
    arm.add_argument("--k", type=int, required=True)
    arm.add_argument("--base", default=DEFAULT_BASE)
    arm.add_argument("--device", default="cuda")
    arm.add_argument("--work", required=True)
    arm.add_argument("--out", required=True)
    arm.add_argument("--start", required=True)
    run = sub.add_parser("run", help="every cell of the chosen methods, with validity checks")
    run.add_argument("--methods", nargs="+", default=sorted(METHODS), choices=sorted(METHODS))
    run.add_argument("--ks", nargs="+", type=int, default=list(KS))
    run.add_argument("--base", default=DEFAULT_BASE)
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
