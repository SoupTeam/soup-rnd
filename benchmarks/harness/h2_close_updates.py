"""H2-close stage U review: actual updates for arms A and L (H2_CLOSE_PROTOCOL.txt section 2).

The `updates` series of h2_ce_confirmation.py runs only the two training children
(baseline_training, ce_only_training) of the existing, tested training phase. This module turns
their sealed evidence into the protocol's per-arm outcomes:

* VOID(check)    the child's F1/ownership/clock controls are not all ok;
* INCOMPLETE     no readable evidence, or the shared deadline (TimeoutError) stopped it;
* UPDATES_FAIL   any of rows U1-U5 fails (the reason names the row);
* UPDATES_PASS   all rows hold.

Gradient and adapter differences between the arms are measured and reported, never gated: the
old numerical policy is not applied here, and the BLOCKED outcomes of 015/016 do not change.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ARMS = {"A": "baseline_training", "L": "ce_only_training"}


def _row(message: str) -> str:
    """Map a reader error to the protocol row it breaks."""
    rows = (
        ("U1", ("recipe/counts", "denominator", "ordered work", "optimizer event")),
        ("U2", ("adapter", "Frozen state", "optimizer", "moments", "Adapter names")),
        ("U3", ("nonfinite", "Nonfinite", "loss", "gradient")),
        ("U4", ("work", "workset", "prepared")),
        ("U5", ("patch", "kernel", "binding", "Kernel")),
    )
    for row, needles in rows:
        if any(n in message for n in needles):
            return row
    return "U?"


def unchanged_adapters(record: dict) -> list[str]:
    """U2: declared trainable tensors whose bytes after the 4 updates equal their initial bytes.

    ``read_training`` has already tied every ``before``/``after`` sha256 to the stored tensor bytes
    (initial.safetensors and the last adapter file) and the names to the declared manifest, but it
    only rejects a step in which no tensor changed. The protocol's U2 needs every tensor to change.
    """
    before, after = record["before"]["trainable"], record["after"]["trainable"]
    if not before or set(before) != set(after):
        raise ValueError("U2 adapter manifest differs between initial and final state")
    return sorted(name for name in before if before[name]["sha256"] == after[name]["sha256"])


def classify_child(root: Path, phase: str) -> dict:
    """Outcome of one child from its sealed evidence, before any pair comparison."""
    from h2_ce_confirmation import read_child
    from h2_ce_controls import validate_controls

    runner = root / phase / "runner"
    try:
        evidence = json.loads((runner / "evidence.json").read_text())
        controls = json.loads((runner / "controls.json").read_text())
    except (OSError, ValueError):
        return {"outcome": "INCOMPLETE", "reason": "no readable child evidence"}
    try:
        validate_controls(controls, phase)
    except ValueError as error:
        return {"outcome": "VOID", "reason": str(error)}
    error = evidence.get("error") or ""
    if error.startswith("TimeoutError"):
        return {"outcome": "INCOMPLETE", "reason": error}
    if error:
        return {"outcome": "UPDATES_FAIL", "row": _row(error), "reason": error}
    try:
        read_child(runner, phase)  # seals, controls, patch counts and kernel attribution (U5)
    except ValueError as error:
        return {"outcome": "UPDATES_FAIL", "row": _row(str(error)), "reason": str(error)}
    return {"outcome": "CAPTURED"}


def _pair_metrics(left: Path, right: Path, initial: dict) -> dict:
    """Descriptive L-vs-A measurements; never a gate."""
    from h2_numerics import gradient_metrics
    from safetensors.torch import load_file

    first = json.loads((left / "training.json").read_text())
    second = json.loads((right / "training.json").read_text())
    loss_diff = [
        b["loss"] - a["loss"] for a, b in zip(first["batches"], second["batches"], strict=True)
    ]
    final_a = load_file(str(left / "adapter_04.safetensors"))
    final_l = load_file(str(right / "adapter_04.safetensors"))
    rows = {}
    for name in sorted(final_a):
        delta_a = final_a[name].double() - initial[name].double()
        delta_l = final_l[name].double() - initial[name].double()
        rows[name] = gradient_metrics(delta_a, delta_l)
    rel = [r["relative_l2"] for r in rows.values() if r["relative_l2"] is not None]
    cos = [r["cosine"] for r in rows.values() if r["cosine"] is not None]
    return {
        "loss_difference_l_minus_a": loss_diff,
        "max_abs_loss_difference": max(abs(v) for v in loss_diff),
        "adapter_delta_relative_l2": {"max": max(rel), "median": sorted(rel)[len(rel) // 2]},
        "adapter_delta_cosine_min": min(cos),
        "identical_final_adapters": all(final_a[n].equal(final_l[n]) for n in final_a),
        "per_tensor": rows,
        "gated": False,
    }


def review_updates(
    root: Path,
    *,
    work: list | None = None,
    expected_initial: str | None = None,
    check_sources: bool = True,
) -> dict:
    """Per-arm outcomes U1-U5 plus descriptive pair metrics for an `updates` series root.

    ``work``, ``expected_initial`` and ``check_sources`` exist for CPU fixtures only; the
    series and the offline review always use the frozen workset, the pinned initial state and
    the source check.
    """
    from h2_ce_confirmation import ARTIFACTS, REPO, SPEC, sha_file
    from h2_ce_training import read_training

    contract = json.loads((root / "contract.json").read_text())
    if check_sources and (
        set(contract.get("artifacts_sha256", {})) != set(ARTIFACTS)
        or any(sha_file(REPO / n) != contract["artifacts_sha256"][n] for n in ARTIFACTS)
    ):
        raise ValueError("Offline reviewer/source differs from captured authority")
    if work is None:
        work = json.loads((SPEC / "ce_confirmation_workset.json").read_text())["training_batches"]
    if expected_initial is None:
        study = json.loads((SPEC / "operator_study.json").read_text())
        expected_initial = study["reference_identity"]["before_sha256"]
    arms, data, initial = {}, {}, {}
    for arm, phase in ARMS.items():
        state = classify_child(root, phase)
        if state["outcome"] == "CAPTURED":
            runner = root / phase / "runner"
            evidence = json.loads((runner / "evidence.json").read_text())
            try:
                if json.loads((runner / "contract.json").read_text()) != contract:
                    raise ValueError("Child contract differs from whole-series authority")
                if any(
                    evidence[k] != contract[k] for k in ("source_commit", "gpu_uuid", "boot_id")
                ):
                    raise ValueError("Child identity differs from authority")
                if evidence["initial_identity"] != expected_initial:
                    raise ValueError("Initial adapter state differs from the pinned state")
                record, start = read_training(runner / "training", work)  # U1, U2, U3, U4
                unchanged = unchanged_adapters(record)
                if unchanged:
                    raise ValueError(
                        f"U2: {len(unchanged)} of {len(record['before']['trainable'])} declared "
                        f"adapter tensors unchanged after the updates: {', '.join(unchanged[:5])}"
                    )
                if record["compute_policy"] != "bf16-autocast-cuda":
                    raise ValueError("GPU training review requires BF16 CUDA evidence")
                losses = [row["loss"] for row in record["batches"]]
                if not all(math.isfinite(v) for v in losses):
                    raise ValueError("Nonfinite loss")
                data[arm], initial[arm] = record, start
                state = {
                    "outcome": "UPDATES_PASS",
                    "optimizer_updates": record["global_step"],
                    "adapter_tensors_changed": f"{len(record['before']['trainable'])}"
                    f"/{len(record['before']['trainable'])}",
                    "microbatches": record["backwards"],
                    "num_items_in_batch": [r["num_items_in_batch"] for r in record["batches"]],
                    "losses": losses,
                    "kernels": evidence["kernels"],
                    "patch_calls": evidence["patch_after"]["patch_calls"],
                }
            except ValueError as error:
                state = {"outcome": "UPDATES_FAIL", "row": _row(str(error)), "reason": str(error)}
        arms[arm] = state
    result = {
        "series": "updates",
        "protocol": "H2_CLOSE_PROTOCOL.txt section 2",
        "arms": arms,
        "performance_verdict": "NO_VERDICT",
    }
    if len(data) == 2:
        # U4 across arms: identical ordered inputs, denominators, initial state and optimizer.
        for key in ("before", "optimizer"):
            if data["A"][key] != data["L"][key]:
                for arm in arms:
                    arms[arm] = {
                        "outcome": "UPDATES_FAIL",
                        "row": "U4",
                        "reason": "pair differs: " + key,
                    }
        # Per-arm equality to the one frozen workset (read_training, U4) already makes the
        # ordered inputs and denominators of A and L identical.
        result["pair_measurements"] = _pair_metrics(
            root / ARMS["A"] / "runner" / "training",
            root / ARMS["L"] / "runner" / "training",
            initial["A"],
        )
    return result
