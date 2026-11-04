#!/usr/bin/env python3
"""Restart a killed or crashed L2L run from its newest complete checkpoint.

``soup train --resume auto`` takes the highest ``checkpoint-N`` without looking inside it.
HF writes a checkpoint in place and ``trainer_state.json`` last, so a run killed during a
save leaves a half-written ``checkpoint-N`` and ``auto`` would pick it. This supervisor
picks the checkpoint itself (the highest N whose ``trainer_state.json`` parses and names
step N) and passes ``l2l_ticket04_arm.py --resume-from <path>``, which passes
``soup train --resume <path>``.

One supervised run:

* attempt 1 starts fresh, or from a complete checkpoint already in ``--output``;
* a nonzero exit starts the next attempt from the newest complete checkpoint, or from
  scratch when there is none;
* two failed attempts in a row that advanced the newest checkpoint by nothing stop the
  loop (a bug, not a kill), and so does ``--max-attempts``;
* a restart refuses to continue when the config file or the training data changed since
  attempt 1 (``l2l_resume_identity.json`` in the checkpoint directory). The identity holds
  the sha256 of the config file and of each ``data.train`` file in order (a dataset name
  counts by its text). It does not cover the base model, the eval data, or the contents
  of a Hub dataset;
* checkpoints that already sit in the directory without an identity file came from some
  other run, and the supervisor refuses them unless ``--adopt-checkpoints`` says they
  belong to this one.

The checkpoint directory is ``output`` of the config (default ``./output``, the schema
default), plus ``experiment_name`` when the config sets one: that is where the trainer
writes (``trainer/sft.py``). ``--output`` overrides it.

Each attempt writes ``attempt-N.json`` (the arm record, with ``resumed_from_step``) and
``attempt-N.log`` into ``--results``; ``attempts.json`` lists them all. A second
invocation on the same ``--results`` continues the numbers. The checkpoints stay in one
output directory. A resumed attempt restarts the layer capture and the L2L
counters, so its timings are not comparable with an uninterrupted run: keep the
measurement arms single-shot and use this for long training runs.

``save_steps`` in the config sets how much a kill costs. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

HARNESS = Path(__file__).resolve().parent
IDENTITY_FILE = "l2l_resume_identity.json"
_CHECKPOINT = re.compile(r"checkpoint-([1-9][0-9]*)")


# ==========================================================================
# pure parts (tested on the CPU)
# ==========================================================================
def checkpoint_step(path: Path) -> int:
    """The step in the name of a ``checkpoint-N`` directory."""
    match = _CHECKPOINT.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"not a checkpoint directory: {path}")
    return int(match[1])


def _is_complete(folder: Path, step: int) -> bool:
    """HF writes ``trainer_state.json`` last, so a parsing file that names ``step`` means done.

    That order is read from ``Trainer._save_checkpoint`` in transformers 5.20.2 (model,
    optimizer and scheduler, RNG state, then ``trainer_state.json``). If a later HF writes
    it earlier, this test passes on half-written checkpoints: re-read that method when the
    transformers version moves.
    """
    try:
        state = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and state.get("global_step") == step


def latest_complete_checkpoint(output: Path) -> Optional[Path]:
    """The ``checkpoint-N`` of the highest N that finished saving, or None."""
    output = Path(output)
    if not output.is_dir():
        return None
    found = []
    for entry in output.iterdir():
        match = _CHECKPOINT.fullmatch(entry.name)
        if match and entry.is_dir() and _is_complete(entry, int(match[1])):
            found.append((int(match[1]), entry))
    return max(found)[1] if found else None


def soup_train_command(config: str, name: Optional[str], resume: Optional[str]) -> List[str]:
    """The argument list of ``soup train`` for one attempt (explicit path, never ``auto``)."""
    command = ["train", "--config", config, "--yes"]
    if name:
        command += ["--name", name]
    if resume:
        command += ["--resume", str(resume)]
    return command


def arm_command(
    config: str, out: str, checkpoint: Optional[Path], *, python: str, name: Optional[str] = None,
) -> List[str]:
    """The process of one attempt: the ticket-04 arm, with ``--resume-from`` after a kill."""
    command = [python, str(HARNESS / "l2l_ticket04_arm.py"), "--config", config, "--out", out]
    if name:
        command += ["--name", name]
    if checkpoint is not None:
        command += ["--resume-from", str(checkpoint)]
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_directory(config: Dict[str, Any]) -> Path:
    """Where the trainer writes ``checkpoint-N``: ``output`` and ``experiment_name`` below it."""
    directory = Path(config.get("output") or "./output")
    if config.get("experiment_name"):
        directory = directory / str(config["experiment_name"])
    return directory


def run_identity(config: Path, train: Any) -> Dict[str, Any]:
    """What must stay equal between attempts: the config file and the training data.

    ``train`` is ``data.train``: one path or dataset name, or a list of them. A file is
    identified by its sha256, anything else (a Hub dataset name) by its text. The order of
    a list counts, because it sets the order of the rows.
    """
    sources = list(train) if isinstance(train, (list, tuple)) else [train]
    data = []
    for source in sources:
        text = str(source)
        if Path(text).is_file():
            data.append([text, _sha256(Path(text))])
        else:
            data.append([text, "name:" + hashlib.sha256(text.encode("utf-8")).hexdigest()])
    return {"config_sha256": _sha256(config), "data": data}


def _has_checkpoints(output: Path) -> bool:
    return output.is_dir() and any(
        _CHECKPOINT.fullmatch(entry.name) and entry.is_dir() for entry in output.iterdir()
    )


def check_identity(output: Path, identity: Dict[str, Any], *, adopt: bool = False) -> None:
    """Record the identity on the first attempt; refuse a later one that differs.

    Checkpoints with no identity file came from another run. They are refused unless
    ``adopt`` says they belong to this one.
    """
    output = Path(output)
    path = output / IDENTITY_FILE
    if not path.is_file():
        if _has_checkpoints(output) and not adopt:
            raise ValueError(
                f"{output} holds checkpoints but no {IDENTITY_FILE}, so they may belong to "
                f"another run; pass --adopt-checkpoints if they belong to this one"
            )
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity, indent=1), encoding="utf-8")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved != identity:
        raise ValueError(
            f"the config or data differs from the run that wrote {path}; resume only the "
            f"same run, or use a new output directory"
        )


def supervise(
    run_attempt: Callable[[Any], int],
    find_checkpoint: Callable[[], Any],
    *,
    max_attempts: int,
    step_of: Callable[[Any], int] = checkpoint_step,
    max_stalled: int = 2,
    first_number: int = 1,
) -> Dict[str, Any]:
    """Run attempts until one exits 0, resuming each from the newest complete checkpoint.

    ``run_attempt(checkpoint)`` returns the exit code; ``checkpoint`` is None for a fresh
    start. ``find_checkpoint()`` is called after each failure. ``max_attempts`` counts the
    attempts of this call; they are numbered from ``first_number``.
    """
    attempts: List[Dict[str, Any]] = []
    stalled = 0
    checkpoint = find_checkpoint()
    while True:
        start = step_of(checkpoint) if checkpoint is not None else None
        code = run_attempt(checkpoint)
        attempts.append({
            "attempt": first_number + len(attempts), "resumed_from": start, "returncode": code,
        })
        if code == 0:
            return {"status": "completed", "attempts": attempts}
        checkpoint = find_checkpoint()
        reached = step_of(checkpoint) if checkpoint is not None else 0
        stalled = stalled + 1 if reached <= (start or 0) else 0
        if stalled >= max_stalled:
            return {"status": "gave_up", "reason": "no progress", "attempts": attempts}
        if len(attempts) >= max_attempts:
            return {"status": "gave_up", "reason": "attempt cap", "attempts": attempts}


# ==========================================================================
# the process side
# ==========================================================================
def _read_config(path: Path) -> Dict[str, Any]:
    import yaml

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return loaded


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output", default=None,
        help="the directory that holds checkpoint-N (default: `output` of the config, "
             "plus `experiment_name` when set)")
    parser.add_argument(
        "--adopt-checkpoints", action="store_true",
        help="accept checkpoints in the directory that no earlier supervised run recorded")
    parser.add_argument("--results", required=True, help="where attempt-N.json/.log are written")
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--attempt-timeout", type=int, default=None, help="seconds per attempt")
    parsed = parser.parse_args(argv)
    if parsed.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    import subprocess

    config = _read_config(Path(parsed.config))
    output = Path(parsed.output) if parsed.output else checkpoint_directory(config)
    results = Path(parsed.results)
    results.mkdir(parents=True, exist_ok=True)
    check_identity(
        output, run_identity(Path(parsed.config), config["data"]["train"]),
        adopt=parsed.adopt_checkpoints,
    )
    # An absent key is the schema default (100); only an explicit 0 or less turns saving off.
    if int(config.get("training", {}).get("save_steps", 100)) <= 0:
        print("warning: training.save_steps is 0 or less; a kill restarts from scratch", flush=True)

    earlier = [int(p.stem.split("-")[1]) for p in results.glob("attempt-*.log")
               if p.stem.split("-")[1].isdigit()]
    first_number = max(earlier, default=0) + 1
    numbers = iter(range(first_number, first_number + parsed.max_attempts))

    def run_attempt(checkpoint: Optional[Path]) -> int:
        number = next(numbers)
        command = arm_command(parsed.config, str(results / f"attempt-{number}.json"),
                              checkpoint, python=sys.executable, name=parsed.name)
        print(f"attempt {number}: " + (f"resume from {checkpoint}" if checkpoint else "fresh"),
              flush=True)
        with (results / f"attempt-{number}.log").open("w") as log:
            try:
                return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=parsed.attempt_timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                return 124

    result = supervise(run_attempt, lambda: latest_complete_checkpoint(output),
                       max_attempts=parsed.max_attempts, first_number=first_number)
    if result["status"] != "completed" and not latest_complete_checkpoint(output):
        nested = sorted(str(p.relative_to(output)) for p in output.rglob("checkpoint-*")
                        if p.is_dir())
        if nested:
            print(f"warning: checkpoints exist below {output} but not in it: {nested[:3]}; "
                  f"pass --output with the directory that holds them", flush=True)
    listing = results / "attempts.json"
    if listing.is_file():
        result["attempts"] = json.loads(listing.read_text(encoding="utf-8"))["attempts"] + (
            result["attempts"])
    listing.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
