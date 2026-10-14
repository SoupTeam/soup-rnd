#!/usr/bin/env python3
"""The ``tie_weights=False`` experiment for G7 (ticket 06).

Arm A is the code as shipped. Arm B patches the runtime's ``functional_call`` so it runs with
``tie_weights=False``. Two steps, run on the RTX 3050 laptop on mains after a reboot, with
the fixture from ``visit_profile.py fixture``::

    python benchmarks/harness/tie_experiment.py run --root /var/tmp/g7 \\
        --out-dir /var/tmp/g7/tie

1. The bit-exact gate, in one process. LoRA B matrices get seeded non-zero values so every
   LoRA gradient is non-zero. One step runs as shipped, a second runs untied, and a control
   runs as shipped on an input with one token changed. The loss and every LoRA gradient of the
   two must be equal under ``torch.equal``, and the control must differ. The result goes to
   ``gate.json``.
2. Only after the gate passes: A-B-B-A blocks at 16 tokens by the repeats rule. A pilot of 5
   blocks, then up to N, then one recompute of N (``arm_validity.blocks_to_add``). Each run is
   the bare arm from ``visit_profile`` in a fresh process under the arm watch with all five
   checks. A block with a void run is kept and replaced. ``experiment.json`` holds the gate,
   every block with its arm records and the verdict from
   ``visit_verdicts.evaluate_experiment``.

``checkpoint`` with ``preserve_rng_state=False`` is not part of the experiment.

Benchmark harness code, not shipped. Heavy dependencies are imported inside functions.
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import visit_verdicts
from arm_validity import Run, block_order, blocks_to_add
from rich.console import Console

HERE = Path(__file__).resolve().parent

#: The experiment runs at this many tokens per step.
TOKENS = 16
#: Blocks run before the experiment gives up on void ones.
DEFAULT_MAX_BLOCKS = 40
#: Seed of the gate's input tokens, the step probe's.
INPUT_SEED = 17

console = Console()

#: Runs one arm and returns ``{"value", "outcome", "record"}``.
RunOne = Callable[..., Dict[str, Any]]

#: Which variant is arm A and which is arm B.
VARIANT_OF_ARM = {"A": "shipped", "B": "untied"}


def untied_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """``functional_call`` with ``tie_weights=False``."""
    return functools.partial(original, tie_weights=False)


def untied_patches() -> Dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """The patch point and wrapper that make arm B, for ``visit_timer.install_patches``."""
    return {"functional_call": untied_wrapper}


# --- the bit-exact gate ----------------------------------------------------------------


def compare_passes(
    shipped: Mapping[str, Any], untied: Mapping[str, Any], control: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare two passes under exact equality. Each pass is ``{"loss", "grads"}`` of tensors.

    The gate passes if the loss and every gradient of ``shipped`` and ``untied`` are equal,
    no gradient is all zero, and ``control`` differs from ``shipped`` in the loss or a
    gradient (otherwise equality would prove nothing).
    """
    import torch

    reasons: List[str] = []
    a_grads, b_grads = shipped["grads"], untied["grads"]
    if not a_grads:
        reasons.append("the shipped pass has no LoRA gradients")
    if set(a_grads) != set(b_grads):
        only = sorted(set(a_grads) ^ set(b_grads))
        reasons.append(f"the passes have different gradients: {', '.join(only)}")
    shared = sorted(set(a_grads) & set(b_grads))
    loss_equal = bool(torch.equal(shipped["loss"], untied["loss"]))
    if not loss_equal:
        reasons.append(
            f"the loss differs ({shipped['loss'].item()!r} against {untied['loss'].item()!r})"
        )
    different = [name for name in shared if not torch.equal(a_grads[name], b_grads[name])]
    if different:
        reasons.append(f"LoRA gradient differs: {', '.join(different)}")
    zero = [name for name in shared if not bool(a_grads[name].any())]
    if zero:
        reasons.append(f"LoRA gradient is all zero: {', '.join(zero)}")
    c_grads = control["grads"]
    control_differs = (
        not torch.equal(shipped["loss"], control["loss"])
        or set(a_grads) != set(c_grads)
        or any(not torch.equal(a_grads[name], c_grads[name]) for name in a_grads)
    )
    if not control_differs:
        reasons.append(
            "the control (one input token changed) did not differ, so equality proves nothing"
        )
    grads_equal = not different and set(a_grads) == set(b_grads) and bool(a_grads)
    return {
        "passed": not reasons,
        "loss_equal": loss_equal,
        "grads_equal": grads_equal,
        "grads_compared": len(shared),
        "control_differs": control_differs,
        "reason": "; ".join(reasons)
        or f"loss and {len(shared)} LoRA gradients are bit-equal, the control differs",
    }


def gate_worker(weights: Path, shards: Path, out: Path) -> int:
    """Run the gate in this process on the GPU and write its result as JSON."""
    import bitexact
    import torch
    import visit_profile
    import visit_timer

    from soup_cli.utils.layer_stream_runtime import decoder_owner

    if not torch.cuda.is_available():
        console.print("SKIP: CUDA is required")
        return 0
    model, runtime, config = visit_profile.build_stream_model(weights, shards)
    bitexact.make_non_vacuous_lora(model)
    generator = torch.Generator(device="cuda").manual_seed(INPUT_SEED)
    ids = torch.randint(0, int(config.vocab_size), (1, TOKENS), generator=generator, device="cuda")
    changed = ids.clone()
    changed[0, 0] = (ids[0, 0] + 1) % int(config.vocab_size)
    model.train()

    def one_pass(tokens: Any) -> Dict[str, Any]:
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        return {"loss": loss.detach().clone(), "grads": bitexact.collect_lora_grads(model)}

    shipped = one_pass(ids)
    repeat = one_pass(ids)
    layers = list(decoder_owner(model).layers)
    with visit_timer.install_patches(layers, untied_patches()):
        untied = one_pass(ids)
    control = one_pass(changed)
    result = compare_passes(shipped, untied, control)
    twice = compare_passes(shipped, repeat, control)
    result["shipped_repeat_equal"] = twice["loss_equal"] and twice["grads_equal"]
    result["tokens"] = TOKENS
    result["lora_b_init"] = "seeded randn * 0.02 (bitexact.make_non_vacuous_lora)"
    runtime.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    console.print(f"gate {'passed' if result['passed'] else 'FAILED'}: {result['reason']}")
    return 0


# --- the blocks ------------------------------------------------------------------------


def run_blocks(run_one: RunOne, max_blocks: int = DEFAULT_MAX_BLOCKS) -> List[List[Dict[str, Any]]]:
    """Run A-B-B-A blocks until the repeats rule asks for no more, void blocks replaced.

    ``run_one(variant=, round=, run=)`` runs one arm and returns ``{"value", "outcome",
    "record"}``. A block is two rounds. Raises RuntimeError after ``max_blocks`` blocks
    without finishing.
    """
    blocks: List[List[Dict[str, Any]]] = []

    def runs() -> List[List[Run]]:
        return [[Run(run["value"], run["outcome"]) for run in block] for block in blocks]

    while (missing := blocks_to_add(runs())) > 0:
        if len(blocks) == max_blocks:
            raise RuntimeError(
                f"{len(blocks)} blocks run, still {missing} short; "
                "void blocks are replaced, so the machine may be failing its arm checks"
            )
        arms = block_order(1)[0]
        first_run = len(blocks) * len(arms) + 1
        first_round = 2 * len(blocks) + 1
        block: List[Dict[str, Any]] = []
        for position, arm in enumerate(arms):
            round, run = first_round + position // 2, first_run + position
            result = run_one(variant=VARIANT_OF_ARM[arm], round=round, run=run)
            block.append({"arm": arm, "round": round, "run": run, **result})
        blocks.append(block)
    return blocks


def arm_runner(
    *,
    fixture: Any,
    root: Path,
    out_dir: Path,
    run_arm_process: Optional[Callable[..., Dict[str, Any]]] = None,
) -> RunOne:
    """A ``RunOne`` that starts the bare arm at 16 tokens in a fresh process under the watch."""
    if run_arm_process is None:
        import visit_profile

        run_arm_process = visit_profile.run_arm_process

    def run_one(*, variant: str, round: int, run: int) -> Dict[str, Any]:
        console.print(f"[dim]run {run} (round {round}, {variant})[/]")
        result = run_arm_process(
            arm="bare",
            seq=TOKENS,
            fixture=fixture,
            root=root,
            out_dir=out_dir,
            round=round,
            run=run,
            variant=variant,
            tag=f"run{run:03d}",
        )
        return {
            "value": float(result["run"]["median_step_s"]),
            "outcome": str(result["arm"]["outcome"]),
            "record": result["arm"],
        }

    return run_one


def run_experiment(
    run_gate: Callable[[], Mapping[str, Any]],
    run_one: RunOne,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
) -> Dict[str, Any]:
    """The gate, then the blocks only if the gate passed, then the verdict."""
    gate = dict(run_gate())
    blocks = run_blocks(run_one, max_blocks) if gate.get("passed") else []
    return {
        "gate": gate,
        "blocks": blocks,
        "verdict": visit_verdicts.evaluate_experiment(gate, blocks),
    }


# --- command line ----------------------------------------------------------------------


def gate_in_subprocess(weights: Path, shards: Path, out_dir: Path) -> Dict[str, Any]:
    """Run the gate in a fresh process, so its GPU memory is freed before the blocks."""
    out = out_dir / "gate.json"
    out.unlink(missing_ok=True)
    command = [
        sys.executable, str(HERE / "tie_experiment.py"), "gate",
        "--weights", str(weights), "--shards", str(shards), "--out", str(out),
    ]  # fmt: skip
    subprocess.run(command, check=True)
    if not out.is_file():
        return {"passed": False, "reason": "the gate wrote no result (no CUDA?)"}
    return json.loads(out.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="the gate, then the blocks, then the verdict")
    run.add_argument("--root", type=Path, required=True, help="the fixture root")
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    gate = sub.add_parser("gate", help="the worker: the bit-exact gate in this process")
    gate.add_argument("--weights", type=Path, required=True)
    gate.add_argument("--shards", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "gate":
        return gate_worker(args.weights, args.shards, args.out)
    import visit_profile

    root = args.root.resolve()
    fixture = visit_profile.fixture_paths(root)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_experiment(
        lambda: gate_in_subprocess(fixture.weights, fixture.shards, out_dir),
        arm_runner(fixture=fixture, root=root, out_dir=out_dir),
        args.max_blocks,
    )
    (out_dir / "experiment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    verdict = result["verdict"]
    console.print(f"gate: {result['gate']['reason']}")
    console.print(f"blocks run: {len(result['blocks'])}")
    console.print(f"experiment: {verdict['verdict']}. {verdict['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
