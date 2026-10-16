#!/usr/bin/env python3
"""Run lifecycle for the F3 rental: survive a crash, stay inside the budget, destroy the instance.

``run_lifecycle`` runs arms one by one. A crashing arm is recorded as failed and the
others still run. After every arm the record is uploaded to a private HuggingFace repo.
A restart skips arms whose upload succeeded. Before each arm the script projects the
spend at the hourly rate and stops if the projection passes the ceiling or the time
limit. The last action is the destroy command, and only when no upload is pending, so
a failed upload never costs the data. The laptop timer and a manual check of the
instance list stay as the backstop.

The vast.ai account key is never put on the instance. Each instance carries its own
``CONTAINER_ID`` and ``CONTAINER_API_KEY`` (set by vast.ai), and ``destroy_command``
uses those. This path is not exercised without a rental: confirm it in the rehearsal
of ticket 07 and check the instance list by hand afterwards.

The HuggingFace token comes from the ``HF_TOKEN`` environment variable at upload time.
Every log line and record passes through ``redact`` first.

Benchmark harness code, not shipped. Heavy and network imports are lazy.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

STATE_FILE = "state.json"
UPLOAD_ATTEMPTS = 3


@dataclass(frozen=True)
class ArmPlan:
    name: str
    #: Expected seconds, used to project the spend before the arm starts.
    estimate_seconds: float


@dataclass
class Lifecycle:
    #: Arm name to "done", "failed" or "skipped: <reason>".
    outcomes: Dict[str, str] = field(default_factory=dict)
    spent: float = 0.0
    pending_uploads: List[str] = field(default_factory=list)
    destroyed: bool = False


def redact(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def destroy_command(env: Mapping[str, str]) -> List[str]:
    """The command that destroys this instance with the per-instance key, not the account key."""
    container = env.get("CONTAINER_ID")
    if not container:
        raise RuntimeError("CONTAINER_ID is not set, cannot destroy the instance")
    if not env.get("CONTAINER_API_KEY"):
        raise RuntimeError("CONTAINER_API_KEY is not set, cannot destroy the instance")
    return [
        "vastai",
        "destroy",
        "instance",
        container,
        "--api-key",
        env.get("CONTAINER_API_KEY", ""),
    ]


def default_destroy() -> None:
    subprocess.run(destroy_command(dict(os.environ)), check=True)  # noqa: S603


def default_upload_factory(repo: str) -> Callable[[str, Path], None]:
    """Upload one file to the private repo ``repo``, with the token from ``HF_TOKEN``.

    A name with a slash is the path in the repo (raw evidence keeps its folders); any other
    name uploads under the file's own name. The upload counts only once the repo lists the file.
    """

    def upload(name: str, path: Path) -> None:
        from huggingface_hub import HfApi

        in_repo = name if "/" in name else Path(path).name
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=in_repo, repo_id=repo)
        if not api.file_exists(repo_id=repo, filename=in_repo):
            raise OSError(f"{in_repo} is not in {repo} after the upload")

    return upload


def _read_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"started": None, "uploaded": [], "outcomes": {}}


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_lifecycle(
    arms: Sequence[ArmPlan],
    run_arm: Callable[[str], Mapping[str, object]],
    upload: Callable[[str, Path], None],
    destroy: Callable[[], None],
    workdir: Path,
    rate_per_hour: float,
    ceiling: float,
    time_limit_seconds: float,
    clock: Callable[[], float] = time.time,
    log: Optional[Callable[[str], None]] = None,
    secrets: Sequence[str] = (),
    started: Optional[float] = None,
    evidence: Optional[Callable[[], Sequence[Tuple[str, Path]]]] = None,
) -> Lifecycle:
    """Run the arms, upload every record and the raw evidence, then destroy the instance.

    ``started`` is the Unix time the paid lease began, so spend counts from the rental and not
    from this call. ``evidence`` runs after the last arm and returns ``(name, path)`` pairs, the
    raw reports, watch records, controls and logs. Each must upload before the instance is
    destroyed. A failure to collect them counts as a pending upload.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    state_path = workdir / STATE_FILE
    state = _read_state(state_path)
    state.setdefault("outcomes", {})
    secrets = tuple(secrets) + tuple(
        os.environ[key] for key in ("HF_TOKEN", "CONTAINER_API_KEY") if os.environ.get(key)
    )
    if state["started"] is None or (started is not None and started < state["started"]):
        state["started"] = clock() if started is None else started
        _write(state_path, state)

    def say(line: str) -> None:
        (log or print)(redact(line, secrets))

    def attempt_upload(name: str, path: Path) -> bool:
        for _ in range(UPLOAD_ATTEMPTS):
            try:
                upload(name, path)
            except Exception as error:  # noqa: BLE001 - any upload failure is retried
                say(f"upload of {name} failed: {error}")
                continue
            state["uploaded"].append(name)
            _write(state_path, state)
            return True
        return False

    def cost(seconds: float) -> float:
        return seconds / 3600 * rate_per_hour

    result = Lifecycle()
    stop_reason = ""
    for plan in arms:
        if plan.name in state["uploaded"]:
            result.outcomes[plan.name] = state["outcomes"].get(plan.name, "done")
            continue
        if stop_reason:
            result.outcomes[plan.name] = stop_reason
            continue
        elapsed = clock() - state["started"]
        projected_seconds = elapsed + plan.estimate_seconds
        projected_cost = cost(projected_seconds)
        if projected_seconds > time_limit_seconds:
            stop_reason = "skipped: time limit"
            result.outcomes[plan.name] = stop_reason
            say(f"stop at {plan.name}: projected {projected_seconds / 3600:.2f} h passes the limit")
            continue
        if projected_cost > ceiling:
            stop_reason = "skipped: budget"
            result.outcomes[plan.name] = stop_reason
            say(f"stop at {plan.name}: projected ${projected_cost:.2f} passes ${ceiling:.2f}")
            continue
        try:
            record = dict(run_arm(plan.name))
            record["status"] = "done"
            result.outcomes[plan.name] = "done"
        except Exception as error:  # noqa: BLE001 - one arm crashing must not stop the others
            record = {"arm": plan.name, "status": "failed", "error": redact(str(error), secrets)}
            result.outcomes[plan.name] = "failed"
        path = workdir / f"{plan.name}.json"
        _write(path, record)
        state["outcomes"][plan.name] = result.outcomes[plan.name]
        if not attempt_upload(plan.name, path):
            result.pending_uploads.append(plan.name)
        spent = cost(clock() - state["started"])
        say(f"{plan.name}: {record['status']}, spent ${spent:.2f} of ${ceiling:.2f}")

    try:
        raw_files = list(evidence()) if evidence else []
    except Exception as error:  # noqa: BLE001 - nothing may be destroyed while evidence is missing
        say(f"collecting the raw evidence failed: {error}")
        raw_files = []
        result.pending_uploads.append("evidence")
    for name, path in raw_files:
        if not attempt_upload(name, Path(path)):
            result.pending_uploads.append(name)

    result.spent = cost(clock() - state["started"])
    summary = workdir / "summary.json"
    _write(summary, {"outcomes": result.outcomes, "spent": round(result.spent, 4)})
    if not attempt_upload("summary", summary):
        result.pending_uploads.append("summary")
    if result.pending_uploads:
        say(f"not destroying: pending uploads {result.pending_uploads}; check the instance list")
        return result
    try:
        destroy()
    except Exception as error:  # noqa: BLE001 - report it, the instance list check is the backstop
        say(f"destroy failed: {error}; check the instance list now")
        raise
    result.destroyed = True
    return result
