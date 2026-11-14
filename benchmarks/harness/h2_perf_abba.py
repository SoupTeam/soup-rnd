"""H2-close stage P: interleaved throughput and peak VRAM (H2_CLOSE_PROTOCOL.txt section 4).

Three parts:

* ``run_arm``: one run of one arm in the current process. It trains the frozen perf config
  for 20 optimizer steps (4 warm-up) through ``soup bench train``'s real trainer path and
  records per-step token counts, peak memory and, for the Liger CE-only arm, the explicit
  patch bindings. ``child`` wraps it for a fresh process on the GPU box.
* ``Series``: the box-side driver. Every run is a fresh child process inside an F1
  ``ArmWatch``, with GPU exclusivity read before and after. Blocks are A X ... X A, the repeats
  rule of METHOD.md is applied per pair, and the series stops at the cap, the void limit, the
  first failed run or the deadline.
* ``analyze``: pure; recomputes every pair verdict from the run records, so an offline review
  can repeat it without the box.

Kernel profiling is deliberately absent from timed runs: a profiler changes the timing.
Kernel attribution of the Liger CE-only arm comes from stage U's trace and from the patch
bindings verified at the end of each L run.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = REPO / "benchmarks" / "h2"
sys.path.insert(0, str(HERE))

import arm_validity  # noqa: E402

STEPS, WARMUP = 20, 4
RUN_MAX_SECONDS = 600
BLOCK_CAP = 12
VOID_BLOCK_LIMIT = 4
REQUIRED = ("suspend", "power", "foreign_reads")
ALPHA = 0.05
ARMS = {  # arm letter: (PatchController arm or None for Unsloth, perf config)
    "A": ("baseline", "perf_baseline.json"),
    "L": ("ce_only", "perf_liger_ce.json"),
    "U": (None, "perf_unsloth.json"),
}
PAIR_ARMS = ("L", "U")


def block_order(admitted: Sequence[str]) -> tuple[str, ...]:
    """A L U U L A with both arms admitted, A X X A with one (protocol P5)."""
    others = tuple(arm for arm in PAIR_ARMS if arm in admitted)
    if not others or "A" not in admitted or set(admitted) - set(ARMS):
        raise ValueError(f"unsupported admitted arms: {admitted}")
    return ("A", *others, *reversed(others), "A")


# --------------------------------------------------------------------------- one run


class StepTokens:
    """Per-step useful/total tokens, read from the batches training_step actually receives."""

    def __init__(self) -> None:
        self.steps: list[dict] = []
        self._useful = 0
        self._total = 0

    def attach(self, trainer: Any) -> None:
        from transformers import TrainerCallback

        from soup_cli.bench.collector import count_supervised_tokens

        inner = trainer.training_step

        def training_step(model, inputs, *args, **kwargs):
            useful, total = count_supervised_tokens(inputs)
            self._useful += useful
            self._total += total
            return inner(model, inputs, *args, **kwargs)

        trainer.training_step = training_step
        outer = self

        class _Close(TrainerCallback):
            def on_step_end(self, args=None, state=None, control=None, **kwargs):
                outer.steps.append({"useful": outer._useful, "total": outer._total})
                outer._useful = outer._total = 0
                return control

        trainer.add_callback(_Close())


def load_recipe(arm: str, model_path: str, output: Path) -> dict:
    recipe = json.loads((SPEC / "candidate_configs" / ARMS[arm][1]).read_text())
    recipe["base"] = model_path
    recipe["data"]["train"] = str(REPO / recipe["data"]["train"])
    recipe["output"] = str(output)
    return recipe


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_manifest_problems(names: Sequence[str], layers: int) -> list[str]:
    """Stage S3: every layer must carry LoRA A and B on all seven target projections."""
    import re

    found = set()
    for name in names:
        hit = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\.", name)
        if hit:
            found.add((int(hit.group(1)), hit.group(2), hit.group(3)))
    wanted = {(i, t, ab) for i in range(layers) for t in LORA_TARGETS for ab in "AB"}
    missing = sorted(wanted - found)
    return [f"missing LoRA tensors, first: {missing[:3]}"] if missing else []


def _binding(fn: Any) -> str:
    fn = getattr(fn, "__func__", fn)
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', type(fn).__name__)}"


def runtime_record(model: Any) -> dict:
    """Stage S3: what the backend actually runs (dtypes, checkpointing, forward bindings)."""
    from collections import Counter

    from transformers.trainer_utils import unwrap_peft_model

    base = unwrap_peft_model(model)
    layer = base.model.layers[0]
    return {
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
        "trainable_dtypes": dict(
            Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)
        ),
        "gradient_checkpointing": bool(getattr(base, "is_gradient_checkpointing", False)),
        "forwards": {
            "model": _binding(base.forward),
            "attention": _binding(layer.self_attn.forward),
            "mlp": _binding(layer.mlp.forward),
            "input_layernorm": _binding(layer.input_layernorm.forward),
            "final_norm": _binding(base.model.norm.forward),
        },
        "loss_function": _binding(getattr(base, "loss_function", None)),
        "layers": len(base.model.layers),
    }


class UpdateAudit:
    """Stage S rows U2/U3 on one run: h2_update_oracle before/after, per-step gradients."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.before: Optional[dict] = None
        self.gradient_rows: list[dict] = []
        self.runtime: Optional[dict] = None

    def attach(self, trainer: Any) -> None:
        from h2_update_oracle import gradients, snapshot
        from transformers import TrainerCallback

        model = trainer.model
        self.names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        self.runtime = runtime_record(model)
        self.before = snapshot(model, self.names)
        outer = self

        class _Gradients(TrainerCallback):
            def on_pre_optimizer_step(self, args=None, state=None, control=None, **kwargs):
                outer.gradient_rows.append(gradients(trainer.model, outer.names))
                return control

        trainer.add_callback(_Gradients())

    def finish(self, trainer: Any) -> dict:
        from h2_update_oracle import audit_update, snapshot

        after = snapshot(trainer.model, self.names)
        losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
        verdict = audit_update(
            self.before, after, self.gradient_rows, trainer.state.global_step, losses
        )
        problems = list(verdict["reasons"])
        problems += lora_manifest_problems(self.names, self.runtime["layers"])
        return {
            "status": "UPDATES_FAIL" if problems else "UPDATES_PASS",
            "problems": problems,
            "oracle": verdict,
            "trainable_tensors": len(self.names),
            "gradient_observations": len(self.gradient_rows),
            "losses": losses,
            "runtime": self.runtime,
        }


def run_arm(
    arm: str,
    recipe: dict,
    *,
    device: str,
    control_factory: Optional[Callable[[str], Any]] = None,
    audit: bool = False,
) -> dict:
    """Train one arm for STEPS steps and return the bench report plus H2 fields.

    ``audit`` adds the stage S update evidence (adapter/frozen/NF4 hashes before and after,
    per-step gradients); it hashes the whole model, so audited runs are never timed runs.
    """
    from soup_cli.bench.train_run import bench_scope_error, run_bench_train
    from soup_cli.config.schema import SoupConfig
    from soup_cli.data.loader import load_dataset

    control = None
    if control_factory is not None and arm != "U":
        control = control_factory(ARMS[arm][0])
    cfg = SoupConfig.model_validate(recipe)
    if (cfg.backend == "unsloth") != (arm == "U"):
        raise ValueError("config backend does not match the arm")
    # soup bench train refuses the Unsloth backend by name because its own path is untested
    # there. Arm U is run through the same trainer path on purpose; stage S's update audit and
    # the per-step token check stand in for that untested scope (H2_CLOSE_PROTOCOL.txt S2).
    refused = None if arm == "U" else bench_scope_error(cfg)
    if refused:
        raise ValueError(refused)
    tokens, holder = StepTokens(), {}
    auditor = UpdateAudit() if audit else None

    def before_train(wrapper: Any) -> None:
        holder["wrapper"] = wrapper
        if bool(wrapper.trainer.args.use_liger_kernel) != (arm == "L"):
            raise ValueError("use_liger_kernel does not match the arm")
        tokens.attach(wrapper.trainer)
        if auditor is not None:
            auditor.attach(wrapper.trainer)

    report = run_bench_train(
        cfg,
        steps=STEPS,
        warmup=WARMUP,
        device=device,
        load_dataset=lambda c: load_dataset(c.data),
        before_train=before_train,
    )
    trainer = holder["wrapper"].trainer
    report["h2"] = {
        "arm": arm,
        "step_tokens": tokens.steps,
        "binding": control.verify(holder["wrapper"].model) if control is not None else None,
        "bench_scope_override": arm == "U",
        "audit": auditor.finish(trainer) if auditor is not None else None,
    }
    return report


def child(arm: str, model_path: str, out: Path, audit: bool = False) -> int:
    """Fresh-process entry on the GPU box; writes run.json or failure.json into ``out``."""
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    started = time.time()
    try:
        unsloth_version = None
        if arm == "U":
            # Unsloth patches transformers/TRL at import; it must come first in the process.
            if any(m in sys.modules for m in ("torch", "transformers", "trl", "peft")):
                raise ValueError("arm U requires unsloth to be imported before torch/transformers")
            import unsloth

            unsloth_version = getattr(unsloth, "__version__", None)
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("exactly one CUDA GPU required")
        from h2_operator_patch import PatchController

        report = run_arm(
            arm,
            load_recipe(arm, model_path, out / "trained_output"),
            device="cuda",
            control_factory=PatchController,
            audit=audit,
        )
        report["h2"]["unsloth_version"] = unsloth_version
        report["h2"]["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
        report["h2"]["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        report["h2"]["elapsed_seconds"] = time.time() - started
        (out / "run.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        return 0
    except BaseException as error:  # noqa: BLE001 -- every failure is evidence
        import traceback

        (out / "failure.json").write_text(
            json.dumps(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "arm": arm,
                    "traceback": traceback.format_exc()[-6000:],
                }
            )
            + "\n"
        )
        return 1


# --------------------------------------------------------------------------- run validity


def run_problems(report: dict, expected_steps: Sequence[dict]) -> list[str]:
    """Why a finished run is not a valid measurement (protocol P4), empty when valid."""
    problems = []
    if not report.get("valid"):
        problems.append("bench validity: " + json.dumps(report.get("failures")))
    if report.get("steps_measured") != STEPS:
        problems.append(f"steps measured {report.get('steps_measured')} != {STEPS}")
    observed = report.get("h2", {}).get("step_tokens")
    if observed != list(expected_steps):
        problems.append("per-step token counts differ from the frozen receipt")
    rate = report.get("throughput", {}).get("useful_tokens_per_second")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        problems.append("no positive finite useful tokens/s")
    return problems


# --------------------------------------------------------------------------- analysis


@dataclass
class RunRecord:
    arm: str
    block: int
    position: int
    outcome: str  # arm_validity outcome: ok, unknown or void
    value: Optional[float]  # useful tokens/s, None when the run produced no valid report
    memory: dict = field(default_factory=dict)


def _pair_blocks(records: Sequence[RunRecord], order: Sequence[str], arm: str) -> list[list]:
    """Each block as the 4-run A X X A list arm_validity expects; a void block stays void."""
    blocks: dict[int, list[RunRecord]] = {}
    for record in records:
        blocks.setdefault(record.block, []).append(record)
    out = []
    for index in sorted(blocks):
        runs = sorted(blocks[index], key=lambda r: r.position)
        if [r.arm for r in runs] != list(order):
            continue  # incomplete block (deadline or failure): not a block
        void = any(r.outcome == arm_validity.VOID or r.value is None for r in runs)
        picked = [r for r in runs if r.arm in ("A", arm)]
        out.append(
            [
                arm_validity.Run(
                    r.value if r.value is not None else float("nan"),
                    arm_validity.VOID if void else r.outcome,
                )
                for r in picked
            ]
        )
    return out


def blocks_to_add(records: Sequence[RunRecord], order: Sequence[str]) -> int:
    """Largest remaining need over the admitted pairs (protocol P6)."""
    pairs = [a for a in PAIR_ARMS if a in order]
    return max(arm_validity.blocks_to_add(_pair_blocks(records, order, a)) for a in pairs)


def _final_n(blocks: list) -> Optional[int]:
    valid = arm_validity.valid_blocks(blocks)
    if len(valid) < arm_validity.PILOT_BLOCKS:
        return None
    first = max(
        arm_validity.PILOT_BLOCKS, arm_validity.blocks_needed(valid[: arm_validity.PILOT_BLOCKS])
    )
    if len(valid) < first:
        return first
    return max(first, arm_validity.blocks_needed(valid[:first]))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
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


def t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value of Student's t (stdlib only; the box runtime has no scipy)."""
    if not math.isfinite(t):
        return 0.0
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(q: float, df: int) -> float:
    """Quantile q (> 0.5) of Student's t with df degrees of freedom, by bisection."""
    if not 0.5 < q < 1.0 or df < 1:
        raise ValueError("t_quantile needs 0.5 < q < 1 and df >= 1")
    target = 2.0 * (1.0 - q)
    low, high = 0.0, 1.0
    while t_two_sided_p(high, df) > target:
        high *= 2.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if t_two_sided_p(mid, df) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def pair_verdict(records: Sequence[RunRecord], order: Sequence[str], arm: str) -> dict:
    blocks = _pair_blocks(records, order, arm)
    valid = arm_validity.valid_blocks(blocks)
    n = _final_n(blocks)
    base = {
        "pair": f"A-{arm}",
        "valid_blocks": len(valid),
        "void_blocks": len(blocks) - len(valid),
        "blocks_needed": n,
    }
    if n is None or n > BLOCK_CAP or len(valid) < n:
        return {**base, "label": "NO_VERDICT_INSUFFICIENT_REPETITIONS"}
    used = valid[:n]  # each pair's test uses its own first N valid blocks
    a_runs = [r.value for b in used for r in (b[0], b[3])]
    x_runs = [r.value for b in used for r in (b[1], b[2])]
    diffs = [(b[1].value + b[2].value) / 2 - (b[0].value + b[3].value) / 2 for b in used]
    mean_a, d = statistics.fmean(a_runs), statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    df = len(diffs) - 1
    se = sd / math.sqrt(len(diffs))
    half = t_quantile(1 - ALPHA / 2, df) * se
    p = t_two_sided_p(d / se, df) if sd > 0 else (0.0 if d else 1.0)
    low, high = d - half, d + half
    label = "FASTER" if low > 0 else "SLOWER" if high < 0 else "NO_DETECTABLE_DIFFERENCE"
    return {
        **base,
        "blocks_used": n,
        "label": label,
        "mean_a_useful_tok_s": mean_a,
        "mean_x_useful_tok_s": statistics.fmean(x_runs),
        "mean_difference": d,
        "difference_95": [low, high],
        "ratio": 1 + d / mean_a,
        "ratio_95": [1 + low / mean_a, 1 + high / mean_a],
        "p_value": p,
        "block_differences": diffs,
    }


def memory_summary(records: Sequence[RunRecord]) -> dict:
    out = {}
    for arm in sorted({r.arm for r in records}):
        runs = [r for r in records if r.arm == arm and r.value is not None and r.outcome != "void"]
        for key in ("max_memory_allocated_bytes", "max_memory_reserved_bytes"):
            values = [r.memory.get(key) for r in runs if r.memory.get(key) is not None]
            if values:
                out.setdefault(arm, {})[key] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "runs": len(values),
                }
    return out


def analyze(records: Sequence[RunRecord], order: Sequence[str]) -> dict:
    return {
        "order": list(order),
        "pairs": {a: pair_verdict(records, order, a) for a in PAIR_ARMS if a in order},
        "memory": memory_summary(records),
        "performance_scope": "This box, this workload and these configs only (protocol P9)",
    }


# --------------------------------------------------------------------------- series driver


@dataclass
class Series:
    """Run blocks until the repeats rule, the cap, the void limit or the deadline stops them."""

    out: Path
    admitted: Sequence[str]
    expected_steps: Sequence[dict]
    deadline_unix_s: float
    model_path: str
    label: str
    launch: Callable[[str, Path], int] = None  # type: ignore[assignment]
    watch: Callable[..., Any] = arm_validity.ArmWatch
    gpu_processes: Callable[[], Optional[list]] = arm_validity.gpu_processes
    clock: Callable[[], float] = time.time
    records: list = field(default_factory=list)
    run_log: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.order = block_order(self.admitted)
        if self.launch is None:
            self.launch = self._launch_child

    def _launch_child(self, arm: str, out: Path) -> int:
        proc = subprocess.run(
            [sys.executable, "-B", __file__, "child", arm, self.model_path, str(out)],
            # Not the repository: Unsloth writes a compile cache into the working directory, and
            # the box's source checkout must stay clean.
            cwd=self.out,
            timeout=RUN_MAX_SECONDS,
            stdout=(out.parent / f"{out.name}.log").open("wb"),
            stderr=subprocess.STDOUT,
        )
        return proc.returncode

    def _save(self, status: str, reason: str = "") -> dict:
        result = {
            "status": status,
            "reason": reason,
            "protocol": "H2_CLOSE_PROTOCOL.txt section 4",
            "steps": STEPS,
            "warmup": WARMUP,
            "block_cap": BLOCK_CAP,
            "order": list(self.order),
            "runs": self.run_log,
            "analysis": analyze(self.records, self.order) if self.records else None,
            "performance_verdict_scope": "pair labels in analysis.pairs only",
        }
        tmp = self.out / "series.json.tmp"
        tmp.write_text(json.dumps(result, indent=2, default=str) + "\n")
        tmp.replace(self.out / "series.json")
        return result

    def _one(self, block: int, position: int, arm: str, run_index: int) -> RunRecord:
        directory = self.out / f"b{block:02d}_{position}_{arm}"
        before = self.gpu_processes()
        with self.watch(
            arm=arm,
            round=2 * block + 1 + position // (len(self.order) // 2),
            run=run_index,
            label=self.label,
            required=REQUIRED,
            model_path=self.model_path,
        ) as watch:
            try:
                code = self.launch(arm, directory)
            except subprocess.TimeoutExpired:
                code = "timeout"
        after = self.gpu_processes()
        entry = {
            "block": block,
            "position": position,
            "arm": arm,
            "exit": code,
            "watch": watch.record,
            "gpu_processes_before": before,
            "gpu_processes_after": after,
            "problems": [],
        }
        report = None
        if code == 0 and (directory / "run.json").exists():
            report = json.loads((directory / "run.json").read_text())
            entry["problems"] = run_problems(report, self.expected_steps)
        else:
            entry["problems"] = [f"child exit {code}"]
        outcome = watch.record["outcome"]
        if before is None or after is None or before or after:
            # Another process on the GPU is contention, like a foreign read: the run is void.
            entry["exclusivity"] = "void: GPU not exclusive or unreadable before/after the run"
            outcome = arm_validity.VOID
        value = None
        if report is not None and not entry["problems"]:
            value = report["throughput"]["useful_tokens_per_second"]
            entry["useful_tokens_per_second"] = value
            entry["median_step_seconds"] = report["timing"]["median_seconds"]
        entry["memory"] = dict(report["memory"]) if report else {}
        entry["outcome"] = outcome
        self.run_log.append(entry)
        return RunRecord(
            arm, block, position, outcome, value, dict(report["memory"]) if report else {}
        )

    def run(self, block_seconds_estimate: float) -> dict:
        """``block_seconds_estimate`` is raised to the longest block seen so far."""
        self.out.mkdir(parents=True, exist_ok=True)
        block, run_index = 0, 1
        while True:
            need = (
                blocks_to_add(self.records, self.order)
                if self.records
                else arm_validity.PILOT_BLOCKS
            )
            valid = (
                max(
                    len(arm_validity.valid_blocks(_pair_blocks(self.records, self.order, a)))
                    for a in PAIR_ARMS
                    if a in self.order
                )
                if self.records
                else 0
            )
            void = block - valid
            if need <= 0:
                return self._save("COMPLETE")
            if valid >= BLOCK_CAP:
                return self._save("COMPLETE_AT_CAP")
            if void > VOID_BLOCK_LIMIT:
                return self._save("NO_VERDICT_HOST_NOISE", f"{void} void blocks")
            if self.clock() + block_seconds_estimate > self.deadline_unix_s:
                return self._save("INCOMPLETE_DEADLINE", f"{need} block(s) still due")
            started = self.clock()
            for position, arm in enumerate(self.order):
                record = self._one(block, position, arm, run_index)
                run_index += 1
                self.records.append(record)
                entry = self.run_log[-1]
                f1_void = record.outcome == arm_validity.VOID
                if entry["problems"] and not f1_void:
                    # A failed or invalid run that F1 does not explain is a code or setup failure:
                    # replacing its block would only repeat it (protocol P4).
                    return self._save("FAILED_RUN", "; ".join(entry["problems"]))
                self._save("RUNNING")
            block_seconds_estimate = max(block_seconds_estimate, self.clock() - started)
            block += 1


def records_from_runs(runs: Sequence[dict]) -> list[RunRecord]:
    """Rebuild the analysis input from series.json's run log (offline review)."""
    return [
        RunRecord(
            r["arm"],
            r["block"],
            r["position"],
            r["outcome"],
            r.get("useful_tokens_per_second"),
            dict(r.get("memory") or {}),
        )
        for r in runs
    ]


def review_series(path: Path) -> dict:
    """Recompute every pair label from a saved series.json and compare with the box's."""
    saved = json.loads(Path(path).read_text())
    order = tuple(saved["order"])
    records = records_from_runs(saved["runs"])
    fresh = analyze(records, order) if records else None
    return {
        "status": saved["status"],
        "analysis": fresh,
        "matches_box": json.dumps(fresh, sort_keys=True, default=str)
        == json.dumps(saved["analysis"], sort_keys=True, default=str),
    }


def main(argv: list[str]) -> int:
    if len(argv) == 5 and argv[1] in ("child", "audit"):
        return child(argv[2], argv[3], Path(argv[4]), audit=argv[1] == "audit")
    if len(argv) == 8 and argv[1] == "series":
        # series <out> <A,L[,U]> <model_path> <deadline_unix_s> <block_estimate_s> <label>
        receipt = json.loads((SPEC / "perf_workload_receipt.json").read_text())
        series = Series(
            out=Path(argv[2]),
            admitted=tuple(argv[3].split(",")),
            expected_steps=receipt["steps"]["step_tokens"],
            deadline_unix_s=float(argv[5]),
            model_path=argv[4],
            label=argv[7],
        )
        result = series.run(block_seconds_estimate=float(argv[6]))
        print(json.dumps({"status": result["status"], "reason": result["reason"]}))
        return 0
    if len(argv) == 3 and argv[1] == "review":
        print(json.dumps(review_series(Path(argv[2])), indent=2, default=str))
        return 0
    print(
        "usage: h2_perf_abba.py child|audit <A|L|U> <model_path> <out_dir>\n"
        "       h2_perf_abba.py series <out> <A,L[,U]> <model> <deadline> <block_s> <label>\n"
        "       h2_perf_abba.py review <series.json>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
