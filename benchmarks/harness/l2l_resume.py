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
* a restart refuses to continue when the config or the training data changed since
  attempt 1 (``l2l_resume_identity.json`` in ``--output``).

Each attempt writes ``attempt-N.json`` (the arm record, with ``resumed_from_step``) and
``attempt-N.log`` into ``--results``; ``attempts.json`` lists them all. The checkpoints
stay in one output directory. A resumed attempt restarts the layer capture and the L2L
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
    """HF writes ``trainer_state.json`` last, so a parsing file that names ``step`` means done."""
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


def run_identity(config: Path, data: Path) -> Dict[str, str]:
    """What must stay equal between attempts: the config file and the training data."""
    return {"config_sha256": _sha256(config), "data_sha256": _sha256(data)}


def check_identity(output: Path, identity: Dict[str, str]) -> None:
    """Record the identity on the first attempt; refuse a later one that differs."""
    output = Path(output)
    path = output / IDENTITY_FILE
    if not path.is_file():
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
) -> Dict[str, Any]:
    """Run attempts until one exits 0, resuming each from the newest complete checkpoint.

    ``run_attempt(checkpoint)`` returns the exit code; ``checkpoint`` is None for a fresh
    start. ``find_checkpoint()`` is called after each failure.
    """
    attempts: List[Dict[str, Any]] = []
    stalled = 0
    checkpoint = find_checkpoint()
    while True:
        start = step_of(checkpoint) if checkpoint is not None else None
        code = run_attempt(checkpoint)
        attempts.append({"attempt": len(attempts) + 1, "resumed_from": start, "returncode": code})
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
        help="the directory that holds checkpoint-N (default: `output` of the config)")
    parser.add_argument("--results", required=True, help="where attempt-N.json/.log are written")
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--attempt-timeout", type=int, default=None, help="seconds per attempt")
    parsed = parser.parse_args(argv)
    if parsed.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    import subprocess

    config = _read_config(Path(parsed.config))
    output = Path(parsed.output or config["output"])
    results = Path(parsed.results)
    results.mkdir(parents=True, exist_ok=True)
    check_identity(output, run_identity(Path(parsed.config), Path(config["data"]["train"])))
    if int(config.get("training", {}).get("save_steps", 0)) <= 0:
        print("warning: training.save_steps is not set; a kill restarts from scratch", flush=True)

    def run_attempt(checkpoint: Optional[Path]) -> int:
        number = len(list(results.glob("attempt-*.log"))) + 1
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
                       max_attempts=parsed.max_attempts)
    if result["status"] != "completed" and not latest_complete_checkpoint(output):
        nested = sorted(p.name for p in output.rglob("checkpoint-*") if p.is_dir())
        if nested:
            print(f"warning: checkpoints exist below {output} but not in it: {nested[:3]}; "
                  f"pass --output with the directory that holds them", flush=True)
    (results / "attempts.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
