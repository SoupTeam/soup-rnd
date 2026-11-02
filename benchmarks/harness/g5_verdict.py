"""G5: the verdict of every mixed arm against all-NF4, with C5's rule and C5's code.

Rule: benchmarks/gate-g5-mixed-precision-experts.md. ``c5_verdict.py`` is
imported and not changed. Each mixed arm is scored by ``c5_verdict.verdict``
on the baseline NF4 and bf16 runs, with the arm's runs in the place of Q2_0, so
the ratio, the interval, the gate-suite drops, C5's validity rows and the
one-repeat rule are C5's, with C5's thresholds. G5 adds its own validity rows
and the reproduction control.

    python benchmarks/harness/g5_verdict.py RUNS/mixed \\
        --baselines benchmarks/results/c5-train/runs \\
        --control RUNS/baselines/nf4-s42-a1/run.json \\
        --control-reference benchmarks/results/c5-train/runs/nf4-s42-a1/run.json \\
        --selection SEL/selection.json --bytes SEL/bytes.json \\
        --swap-record benchmarks/results/c5-olmoe-swap/nf4.json \\
        --swap-record benchmarks/results/c5-olmoe-swap/q2_0.json --out OUT/verdict.json

Writes ``verdict.json`` and, beside it, ``quality_vs_bytes.svg`` drawn from the
verdict's ``points``. Exit code 0 means the verdict was computed, whatever it is.

Synthetic records (the CPU pilot) go through the same code, but nothing in their
verdict is a G5 verdict: every arm's ``outcome`` and ``decision.g5`` are
``no_verdict`` (what the code computed stays under ``pipeline_outcome`` and
``decision.pipeline``), ``reportable`` is false, and the chart and the console
say NO VERDICT.

    python benchmarks/harness/g5_verdict.py provenance C5_RUN.json --out FILE

checks, before any arm runs, that C5's code on this checkout hashes as in C5's
record, so the reproduction control can pass at all (exit 1 if not).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from rich.console import Console

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c5_train  # noqa: E402
import c5_verdict as c5  # noqa: E402
import g5_select  # noqa: E402

console = Console()

#: The control is C5's first arm, NF4 seed 42, repeated on the box of the mixed arms.
CONTROL_RUN = ("nf4", 42)
#: |task loss(control) / task loss(C5) - 1| at most this: the seed-to-seed spread of
#: C5's NF4 runs is 0.0009 and the gate's margin is 0.03.
CONTROL_MAX_RELATIVE_DIFF = Fraction("0.002")
#: Correct answers per gate suite may differ by this much from C5's run.
CONTROL_MAX_SUITE_DIFF = 1
BASELINE_ARMS = ("nf4", "q2_0", "bf16")
#: The label of a verdict over synthetic records, in the JSON, the chart and the console.
SYNTHETIC_LABEL = "SYNTHETIC / CPU PILOT: NO VERDICT"
#: C5's harness files and the key under which ``c5_train.py`` records each one's SHA-256.
C5_CODE = (("c5_train.py", "sha256"), ("c5_expert_swap.py", "swap_sha256"),
           ("q2_0_codec.py", "codec_sha256"))
#: The checks ``bytes.json`` must hold for a real verdict: C5's measured caches reproduced.
C5_BYTE_CHECKS = ("all_nf4_layer_files_match_c5", "all_nf4_ssd_bytes_match_c5",
                  "all_q2_0_layer_files_match_c5", "all_q2_0_ssd_bytes_match_c5")


def box_identity(run: dict[str, Any]) -> str:
    """The box a run ran on: its GPU's UUID and its package versions."""
    return json.dumps({"gpu": (run.get("gpu") or {}).get("uuid"),
                       "versions": run.get("versions")}, sort_keys=True)


def _harness(run: dict[str, Any]) -> tuple:
    harness = run["harness"]
    return harness["sha256"], harness["swap_sha256"], harness["codec_sha256"]


#: What a control run must hold before it can be compared with C5's record at all.
CONTROL_FIELDS = ("arm", "seed", "synthetic", "status", "panel", "gate_suites", "config", "data",
                  "batch_order_sha256", "model", "harness", "expert_digest",
                  "non_expert_fingerprints", "train_log", "total_steps")


def _incomplete(run: dict[str, Any]) -> list[str]:
    """Why a run cannot be compared: missing fields, or a status other than complete."""
    why = [f"no {name}" for name in CONTROL_FIELDS if name not in run]
    if run.get("status") != "complete":
        why.insert(0, f"status {run.get('status')!r}")
    return why


def reproduction_control(control: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Does NF4 seed 42 on the new box reproduce C5's NF4 seed 42?

    A control that did not finish (crashed, killed, still running) is a failed
    control with ``control_complete`` false and the reasons, never an exception: the
    rule then re-runs C5's arms on the box."""
    if bool(control.get("synthetic")) != bool(reference.get("synthetic")):
        raise ValueError("synthetic and real records never mix in one control")
    unfinished = {"control": _incomplete(control), "reference": _incomplete(reference)}
    if unfinished["control"] or unfinished["reference"]:
        return {
            "synthetic": bool(control.get("synthetic")),
            "passed": False,
            "checks": {"control_complete": not unfinished["control"],
                       "reference_complete": not unfinished["reference"]},
            "incomplete": unfinished,
            "task_loss": None, "reference_task_loss": None, "relative_task_loss_diff": None,
            "limit": float(CONTROL_MAX_RELATIVE_DIFF), "gate_suite_diff": None,
            "harness": {"control": control.get("harness"),
                        "reference": reference.get("harness")},
            "box": json.loads(box_identity(control)),
            "reference_box": json.loads(box_identity(reference)),
        }
    loss, reference_loss = c5.task_loss(control), c5.task_loss(reference)
    relative = abs(loss / reference_loss - 1)
    suites = {
        name: control["gate_suites"][name]["correct"] - suite["correct"]
        for name, suite in reference["gate_suites"].items()
    }
    checks = {
        "control_complete": True,
        "reference_complete": True,
        "control_is_nf4_seed_42": (control["arm"], control["seed"]) == CONTROL_RUN
        and (reference["arm"], reference["seed"]) == CONTROL_RUN,
        "control_clean": not c5.run_problems(control),
        "reference_clean": not c5.run_problems(reference),
        "same_config": control["config"] == reference["config"],
        "same_data": control["data"]["heldout_sha256"] == reference["data"]["heldout_sha256"]
        and control["data"]["train_sha256"] == reference["data"]["train_sha256"],
        "same_batch_order": control["batch_order_sha256"] == reference["batch_order_sha256"],
        "same_model": (control["model"]["id"], control["model"]["revision"])
        == (reference["model"]["id"], reference["model"]["revision"]),
        "same_c5_code": _harness(control) == _harness(reference),
        "same_expert_weights": control["expert_digest"] == reference["expert_digest"],
        "same_non_expert_weights":
            control["non_expert_fingerprints"] == reference["non_expert_fingerprints"],
        "task_loss_within_tolerance": relative <= CONTROL_MAX_RELATIVE_DIFF,
        "gate_suites_within_tolerance": set(suites) == set(control["gate_suites"])
        and all(abs(diff) <= CONTROL_MAX_SUITE_DIFF for diff in suites.values()),
    }
    return {
        "synthetic": bool(control["synthetic"]),
        "passed": all(checks.values()),
        "checks": checks,
        "task_loss": float(loss),
        "reference_task_loss": float(reference_loss),
        "relative_task_loss_diff": float(relative),
        "limit": float(CONTROL_MAX_RELATIVE_DIFF),
        "gate_suite_diff": suites,
        "harness": {"control": control["harness"], "reference": reference["harness"]},
        "box": json.loads(box_identity(control)),
        "reference_box": json.loads(box_identity(reference)),
    }


def provenance(
    reference: dict[str, Any], digests: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Will C5's code on this checkout hash as in C5's record ``reference`` (a run.json)?

    The control's ``same_c5_code`` compares the bytes ``c5_train.py`` hashes, so
    a checkout with CRLF line endings (``core.autocrlf=true``) or an edited C5
    file fails it. ``eol_only`` says the files differ from the record only in
    their line endings. Package versions are reported, not checked: the rule's
    control measures the number, not the box.
    """
    digests = digests if digests is not None else g5_select.source_digests()
    files = {}
    for name, key in C5_CODE:
        recorded = reference["harness"][key]
        files[name] = {"recorded": recorded, **digests[name],
                       "matches": digests[name]["sha256"] == recorded,
                       "matches_after_lf": digests[name]["sha256_lf"] == recorded}
    matches = all(entry["matches"] for entry in files.values())
    versions, recorded_versions = c5_train._versions(), reference.get("versions") or {}
    return {
        "kind": "g5_provenance",
        "synthetic": bool(reference["synthetic"]),
        "c5_code_matches": matches,
        "eol_only": not matches and all(entry["matches_after_lf"] for entry in files.values()),
        "files": files,
        "crlf_files": sorted(name for name, entry in digests.items()
                             if entry["sha256"] != entry["sha256_lf"]),
        "sources": digests,
        "versions": versions,
        "reference_versions": recorded_versions,
        "version_diff": {
            name: [versions.get(name), recorded_versions.get(name)]
            for name in sorted(set(versions) | set(recorded_versions))
            if versions.get(name) != recorded_versions.get(name)
        },
    }


def map_of(run: dict[str, Any]) -> dict[str, Any] | None:
    """The codec map a ``g5_train.py`` run trained over; None for a C5 arm."""
    return (run.get("swap") or {}).get("codec_map")


def codes_from_run(run: dict[str, Any]) -> list[list[str]]:
    """Per-layer expert codecs as the run swapped them, from its per-tensor record."""
    layers: dict[int, list[str]] = {}
    for name, codes in map_of(run)["per_expert"].items():
        layer = g5_select.layer_of(name)
        if layers.setdefault(layer, codes) != codes:
            raise ValueError(f"layer {layer}: expert tensors were swapped with different codes")
    return [layers[layer] for layer in sorted(layers)]


def _relabel(entries: Sequence[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [{**entry, "arm": arm if entry["arm"] == "q2_0" else entry["arm"]} for entry in entries]


def _counted(runs: Sequence[dict[str, Any]], used: Sequence[dict[str, Any]]
             ) -> list[dict[str, Any]]:
    """Only attempts counted by C5, never the kept spoiled attempts."""
    keys = {(entry["arm"], entry["seed"], entry["attempt"]) for entry in used}
    return [run for run in runs if (run["arm"], run["seed"], run["attempt"]) in keys]


def _c5_verdict(
    runs: Sequence[dict[str, Any]], *, swap_records: Sequence[dict[str, Any]], synthetic: bool,
) -> dict[str, Any]:
    """C5's unchanged selector, with G5 requiring the kept attempt sequence to start at 1."""
    result = c5.verdict(runs, swap_records=swap_records, synthetic=synthetic)
    attempts: dict[tuple[str, int], list[int]] = {}
    for run in runs:
        attempts.setdefault((run["arm"], run["seed"]), []).append(run["attempt"])
    ordered = all(
        all(type(attempt) is int for attempt in tries)
        and sorted(tries) == list(range(1, len(tries) + 1))
        and len(tries) <= c5.MAX_ATTEMPTS
        for tries in attempts.values()
    )
    result["validity"]["attempt_sequence"] = ordered
    if not ordered:
        result.update(outcome="no_verdict", g5="repeat")
    return result


def _same_g5_code(run: dict[str, Any], selection: dict[str, Any]) -> bool:
    """The run and selection must name the same recorded source bytes."""
    harness = (map_of(run) or {}).get("harness") or {}
    selected = selection.get("harness") or {}
    sources = selected.get("sources") or {}
    recorded = harness.get("sources") or {}
    return all(
        sources.get(name, {}).get("sha256")
        and recorded.get(name, {}).get("sha256") == sources[name]["sha256"]
        for name in g5_select.SOURCES
    ) and (
        harness.get("g5_select_sha256") == selected.get("sha256")
        == sources.get("g5_select.py", {}).get("sha256")
        and harness.get("g5_train_sha256")
        == sources.get("g5_train.py", {}).get("sha256")
        and all(sources.get(name, {}).get("sha256") == run["harness"][key]
                for name, key in C5_CODE)
    )


def g5_rows(
    arm: str, runs: Sequence[dict[str, Any]], everything: Sequence[dict[str, Any]],
    selection: dict[str, Any], bytes_record: dict[str, Any], baselines_ok: bool,
) -> dict[str, bool]:
    """G5's validity rows for one mixed arm. ``everything`` is every run the arm's verdict uses."""
    selected = selection["maps"][arm]
    calibration = selection["calibration"]
    _, fraction = g5_select.ARM_SPECS[arm]
    real_hash = (calibration["sha256"] == g5_select.CALIBRATION_SHA256
                 and calibration["train_sha256"] in c5_train.TRAIN_SHA256)
    twins = [other for other, (_, f) in g5_select.ARM_SPECS.items() if f == fraction]
    arm_bytes = bytes_record["arms"]
    size = arm_bytes.get(arm, {}).get("ssd_bytes")
    selected_harness = selection.get("harness") or {}
    selection_c5 = tuple(selected_harness.get(key)
                         for key in ("c5_train_sha256", "swap_sha256", "codec_sha256"))
    by_seed: dict[int, set[str]] = {}
    for run in everything:
        by_seed.setdefault(run["seed"], set()).add(run["batch_order_sha256"])
    return {
        # the calibration rows are the rule's (real) and come from the train split every run
        # trains on; selection.json records no panel hash because the panel is never read
        "selection_data_hash": (selection["synthetic"] or real_hash)
        and "heldout_sha256" not in calibration
        and all(run["data"]["train_sha256"] == calibration["train_sha256"]
                for run in everything),
        "selection_checks": bool(selection["checks"]) and all(selection["checks"].values()),
        "selection_model": all(
            (run["model"]["id"], run["model"]["revision"])
            == (selection["model"]["id"], selection["model"]["revision"]) for run in everything
        ),
        "map_hash": bool(runs) and all(
            map_of(run)["map_sha256"] == selected["map_sha256"]
            and map_of(run)["stats_sha256"] == selection["stats_sha256"]
            and map_of(run)["calibration_sha256"] == calibration["sha256"]
            and g5_select.map_sha256(codes_from_run(run)) == selected["map_sha256"]
            for run in runs
        ),
        "byte_budget_equal": all(
            other in arm_bytes and arm_bytes[other].get("map_sha256")
            == selection["maps"][other]["map_sha256"]
            and isinstance(size, int) and size > 0
            and arm_bytes[other].get("ssd_bytes") == size
            for other in twins
        ) and all(bytes_record["checks"].values())
        # a real verdict needs all-NF4 and all-Q2_0 equal to C5's measured caches
        and (selection["synthetic"]
             or all(bytes_record["checks"].get(name) is True for name in C5_BYTE_CHECKS)),
        "c5_code_unchanged": bool(everything) and all(selection_c5)
        and all(_harness(run) == selection_c5 for run in everything),
        "g5_code_unchanged": bool(runs) and all(_same_g5_code(run, selection) for run in runs),
        "paired_batch_order": all(len(orders) == 1 for orders in by_seed.values()),
        "baselines_reproduced": baselines_ok,
    }


def arm_verdict(
    arm: str, runs: Sequence[dict[str, Any]], baselines: Sequence[dict[str, Any]], *,
    swap_records: Sequence[dict[str, Any]], synthetic: bool, selection: dict[str, Any],
    bytes_record: dict[str, Any], baselines_ok: bool,
) -> dict[str, Any]:
    """C5's verdict with the arm's runs as the candidate, then G5's rows.

    Every attempt goes to C5's one-repeat rule, spoiled ones included (a run that
    crashed before the swap has no codec map); G5's map rows then read only the
    attempts that rule counts, so a kept failed attempt never blocks its repeat."""
    reference = [run for run in baselines if run["arm"] in ("nf4", "bf16")]
    candidate = [{**run, "arm": "q2_0"} for run in runs]
    compared = reference + candidate
    result = _c5_verdict(compared, swap_records=swap_records, synthetic=synthetic)
    result.pop("g5")
    counted = _counted(compared, result["used"])
    counted_mixed = [run for run in counted if run["arm"] == "q2_0"]
    for key in ("used", "missing", "unexpected_attempts", "spoiled", "config_diff"):
        result[key] = _relabel(result[key], arm)
    rows = g5_rows(arm, counted_mixed, counted, selection, bytes_record, baselines_ok)
    result["validity"] = {**result["validity"], **rows}
    if not all(rows.values()):
        result["outcome"] = "no_verdict"
    rule, fraction = g5_select.ARM_SPECS[arm]
    size = bytes_record["arms"].get(arm, {}).get("ssd_bytes")
    return {
        "arm": arm, "rule": rule, "fraction_q2_0": fraction,
        "map_sha256": selection["maps"][arm]["map_sha256"],
        "ssd_bytes": size,
        "ssd_bytes_vs_all_nf4": None if size is None else (
            size / bytes_record["arms"]["all_nf4"]["ssd_bytes"]),
        **result,
    }


def decision(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """What the outcomes mean for G5 (the rule's table)."""
    passing = [arm for arm in g5_select.ARMS if arms[arm]["outcome"] == "pass"]
    if passing:
        best = max(passing, key=lambda arm: (g5_select.ARM_SPECS[arm][1],
                                             g5_select.ARM_SPECS[arm][0] == "traffic"))
        rule, fraction = g5_select.ARM_SPECS[best]
        return {"g5": "rule", "arm": best, "rule": rule, "fraction_q2_0": fraction,
                "unresolved": [arm for arm in g5_select.ARMS
                               if arms[arm]["outcome"] == "no_verdict"]}
    if any(arms[arm]["outcome"] == "no_verdict" for arm in g5_select.ARMS):
        return {"g5": "repeat", "arm": None, "rule": None, "fraction_q2_0": None,
                "unresolved": [arm for arm in g5_select.ARMS
                               if arms[arm]["outcome"] == "no_verdict"]}
    return {"g5": "stop", "arm": None, "rule": None, "fraction_q2_0": None, "unresolved": []}


def _no_verdict(result: dict[str, Any]) -> dict[str, Any]:
    """A synthetic result: what the code computed moves to ``pipeline_*``."""
    out = {**result, "outcome": "no_verdict", "reportable": False,
           "pipeline_outcome": result["outcome"]}
    if "g5" in result:
        out.update(g5="no_verdict", pipeline_g5=result["g5"])
    return out


def synthetic_decision(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The decision of a synthetic verdict: none, with the computed one under ``pipeline``."""
    return {"g5": "no_verdict", "arm": None, "rule": None, "fraction_q2_0": None,
            "unresolved": list(g5_select.ARMS),
            "reason": "synthetic records (CPU pilot): the rule's code ran, nothing is decided",
            "pipeline": decision(arms)}


def _band(intervals: dict[Any, tuple[float, float]] | None) -> list[float] | None:
    if not intervals:
        return None
    return [min(low for low, _ in intervals.values()), max(high for _, high in intervals.values())]


def points(
    bytes_record: dict[str, Any], baseline: dict[str, Any], arms: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Quality (mean task-loss ratio against all-NF4) against full cache bytes, one per arm."""
    out = [
        {"arm": "all_nf4", "label": "all-NF4", "ssd_bytes": bytes_record["arms"]["all_nf4"]
         ["ssd_bytes"], "mean_ratio": 1.0, "band": None, "outcome": "reference"},
        {"arm": "all_q2_0", "label": "all-Q2_0", "ssd_bytes": bytes_record["arms"]["all_q2_0"]
         ["ssd_bytes"], "mean_ratio": baseline["mean_ratio"],
         "band": _band(baseline.get("intervals")), "outcome": baseline["outcome"]},
    ]
    for arm, result in arms.items():
        out.append({"arm": arm, "label": arm, "ssd_bytes": result["ssd_bytes"],
                    "mean_ratio": result["mean_ratio"], "band": _band(result.get("intervals")),
                    "outcome": result["outcome"]})
    return out


def verdict(
    mixed_runs: Sequence[dict[str, Any]], baselines: Sequence[dict[str, Any]], *,
    control: dict[str, Any], control_reference: dict[str, Any], selection: dict[str, Any],
    bytes_record: dict[str, Any], swap_records: Sequence[dict[str, Any]] = (),
    synthetic: bool = False,
) -> dict[str, Any]:
    """The G5 rule: control, baselines (C5's verdict), each mixed arm, then the decision."""
    everything = [*mixed_runs, *baselines, control, control_reference]
    if any(bool(run["synthetic"]) != synthetic for run in everything) or (
        bool(selection["synthetic"]) != synthetic
        # bytes.json written before it carried the label (the 2026-10-21 pilot) is not checked
        or bool(bytes_record.get("synthetic", synthetic)) != synthetic
    ):
        raise ValueError("synthetic and real records never mix in one verdict")
    for run in mixed_runs:
        if map_of(run) is None and not c5.run_problems(run):
            # A finished run trained over no map is a C5 arm in the wrong place, not a
            # spoiled mixed attempt (those are kept and pass through the one-repeat rule).
            raise ValueError("a finished mixed run without a codec map: pass C5 arms with "
                             "--baselines")
    if any(run["arm"] not in BASELINE_ARMS for run in baselines):
        raise ValueError("a baseline that is not a C5 arm (nf4, q2_0, bf16)")
    control_result = reproduction_control(control, control_reference)
    baseline = _c5_verdict(baselines, swap_records=swap_records, synthetic=synthetic)
    counted_baselines = _counted(baselines, baseline["used"])
    mixed_boxes = {box_identity(run) for run in mixed_runs if not c5.run_problems(run)}
    baseline_boxes = {box_identity(run) for run in counted_baselines}
    control_on_box = mixed_boxes == {box_identity(control)}
    # Spoiled attempts are kept, but only counted runs establish the box and reference.
    complete_baselines = baseline["outcome"] != "no_verdict"
    rerun = complete_baselines and len(mixed_boxes) == 1 and baseline_boxes == mixed_boxes
    reused = complete_baselines and control_result["passed"] and control_on_box and (
        baseline_boxes == {box_identity(control_reference)}
        and any(run == control_reference for run in counted_baselines)
    )
    source = "rerun" if rerun else "reused" if reused else "not_reproduced"
    baselines_ok = source != "not_reproduced"
    # by arm name, not by map: two rules may pick the same experts, and each arm trains anyway
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in mixed_runs:
        # g5_train passes the map's arm to c5_train as --arm, so a run that failed
        # before the swap (no codec map yet) still names its arm
        by_arm.setdefault(map_of(run)["arm"] if map_of(run) else run["arm"], []).append(run)
    unknown = sorted(set(by_arm) - set(g5_select.ARMS))
    if unknown:
        raise ValueError(f"runs of arms the rule does not have: {unknown}")
    arms = {
        arm: arm_verdict(
            arm, by_arm.get(arm, []), baselines,
            swap_records=swap_records, synthetic=synthetic, selection=selection,
            bytes_record=bytes_record, baselines_ok=baselines_ok,
        )
        for arm in g5_select.ARMS
    }
    pipeline_decision = synthetic_decision(arms) if synthetic else decision(arms)
    if synthetic:
        arms = {arm: _no_verdict(result) for arm, result in arms.items()}
        baseline = _no_verdict(baseline)
    return {
        "kind": "g5_verdict",
        "synthetic": synthetic,
        # False for synthetic records: nothing in this verdict may be reported as G5's
        "reportable": not synthetic,
        "label": SYNTHETIC_LABEL if synthetic else None,
        "control": {**control_result, "on_mixed_box": control_on_box},
        "baselines": {"source": source, "verdict": baseline},
        "arms": arms,
        "decision": pipeline_decision,
        # how far traffic stood in for sensitivity on the calibration tokens (not a gate row)
        "traffic_vs_sensitivity": {
            key: (selection.get("correlation") or {}).get(key)
            for key in ("mean_spearman_tokens_d", "fractions")
        },
        "points": points(bytes_record, baseline, arms),
    }


def bf16_point(
    bytes_record: dict[str, Any], baselines: Sequence[dict[str, Any]],
    used: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """C5's bf16 reference: task loss over NF4's on seed 42, at C5's measured bf16 bytes.

    ``used`` is the baseline verdict's ``used`` list: the attempts C5's one-repeat rule
    counted. The chart reads those, not attempt 1, so a spoiled first attempt that was
    repeated never reaches the chart while its repeat is in the verdict."""
    counted = {(entry["arm"], entry["seed"]): entry["attempt"] for entry in used}
    runs = {(run["arm"], run["seed"]): run for run in baselines
            if counted.get((run["arm"], run["seed"])) == run["attempt"]}
    if "c5_bf16_ssd_bytes" not in bytes_record or not {("bf16", 42), ("nf4", 42)} <= set(runs):
        return None
    ratio = c5.task_loss(runs["bf16", 42]) / c5.task_loss(runs["nf4", 42])
    return {"arm": "bf16", "label": "bf16 (reference, seed 42)",
            "ssd_bytes": bytes_record["c5_bf16_ssd_bytes"], "mean_ratio": float(ratio),
            "band": None, "outcome": "reference",
            "attempts": {"bf16": counted[("bf16", 42)], "nf4": counted[("nf4", 42)]}}


SVG_SIZE = (720, 460)
SVG_MARGIN = (90, 30, 40, 70)  # left, right, top, bottom
OUTCOME_COLOURS = {"pass": "#1a7f37", "fail": "#cf222e", "no_verdict": "#9a6700",
                   "reference": "#24292f"}


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    step = (high - low) / count
    return [low + i * step for i in range(count + 1)]


def _unit(largest: float) -> tuple[float, str]:
    for scale, name in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
        if largest >= scale:
            return scale, name
    return 1.0, "B"


def _marker_kind(arm: str) -> str:
    """Circle for the traffic rule, square for sensitivity, diamond for the uniform arms."""
    if arm.startswith("mixed_traffic"):
        return "circle"
    if arm.startswith("mixed_sensitivity"):
        return "square"
    return "diamond"


def _marker(kind: str, x: float, y: float, colour: str) -> str:
    if kind == "square":
        return (f'<rect class="point" x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" '
                f'fill="{colour}"/>')
    if kind == "diamond":
        return (f'<path class="point" d="M {x:.1f} {y - 6:.1f} L {x + 6:.1f} {y:.1f} '
                f'L {x:.1f} {y + 6:.1f} L {x - 6:.1f} {y:.1f} Z" fill="{colour}"/>')
    return f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}"/>'


def svg_plot(result: dict[str, Any], title: str) -> str:
    """Quality against full cache bytes, drawn by hand from ``result['points']``.

    x: cache bytes on disk (metadata included); y: mean task-loss ratio against
    all-NF4, with the range of the per-seed 95% intervals as a bar; the dashed
    line is the gate's 1.03. Markers: circle traffic rule, square sensitivity
    rule, diamond all-NF4 / all-Q2_0; colour is the outcome. Both rules of one
    fraction share their bytes, so points are numbered (traffic to the left,
    the rest to the right) and listed in a legend. A point outside the
    quantized range (bf16) is only in the legend. A synthetic result is titled
    NO VERDICT and every point but the references is drawn as ``no_verdict``,
    also from a verdict.json written before synthetic outcomes were relabelled.
    """
    width, plot_height = SVG_SIZE
    left, right, top, bottom = SVG_MARGIN
    synthetic = bool(result.get("synthetic")) or result.get("reportable") is False
    if synthetic:
        title = f"{SYNTHETIC_LABEL}: {title}"
    points = [
        {**p, "outcome": "no_verdict"} if synthetic and p["outcome"] != "reference" else p
        for p in result["points"]
    ]
    quantized = [p for p in points if p["arm"] != "bf16" and p["mean_ratio"] is not None
                 and p["ssd_bytes"] is not None]
    height = plot_height + 16 * len(points) + 10
    scale, unit = _unit(max(p["ssd_bytes"] for p in quantized))
    xs = [p["ssd_bytes"] / scale for p in quantized]
    ys = [v for p in quantized for v in ([p["mean_ratio"]] + (p["band"] or []))]
    ys.append(float(c5.MAX_MEAN_RATIO))
    x0, x1 = min(xs), max(xs)
    pad_x = (x1 - x0) * 0.08 or 0.1
    x0, x1 = x0 - pad_x, x1 + pad_x
    y0, y1 = min(ys + [1.0]), max(ys)
    pad_y = (y1 - y0) * 0.1 or 0.01
    y0, y1 = y0 - pad_y, y1 + pad_y
    axis_y = plot_height - bottom

    def px(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (width - left - right)

    def py(y: float) -> float:
        return axis_y - (y - y0) / (y1 - y0) * (axis_y - top)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14">'
           f'{_escape(title)}</text>']
    for tick in _ticks(x0, x1):
        out.append(f'<line x1="{px(tick):.1f}" y1="{axis_y}" x2="{px(tick):.1f}" '
                   f'y2="{axis_y + 5}" stroke="black"/>')
        out.append(f'<text x="{px(tick):.1f}" y="{axis_y + 18}" '
                   f'text-anchor="middle">{tick:.4g}</text>')
    for tick in _ticks(y0, y1):
        out.append(f'<line x1="{left - 5}" y1="{py(tick):.1f}" x2="{left}" y2="{py(tick):.1f}" '
                   f'stroke="black"/>')
        out.append(f'<text x="{left - 8}" y="{py(tick) + 4:.1f}" text-anchor="end">'
                   f'{tick:.4f}</text>')
    out += [f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" '
            'stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="black"/>',
            f'<text x="{(left + width - right) / 2}" y="{axis_y + 38}" text-anchor="middle">'
            f'full cache on disk, {unit} (1 {unit} = {scale:.0e} B), metadata included</text>',
            f'<text x="18" y="{(top + axis_y) / 2}" text-anchor="middle" '
            f'transform="rotate(-90 18 {(top + axis_y) / 2})">'
            'task loss / all-NF4 task loss, mean of 3 seeds</text>']
    gate = float(c5.MAX_MEAN_RATIO)
    out.append(f'<line x1="{left}" y1="{py(gate):.1f}" x2="{width - right}" y2="{py(gate):.1f}" '
               'stroke="#cf222e" stroke-dasharray="6 4"/>')
    out.append(f'<text x="{width - right - 4}" y="{py(gate) - 5:.1f}" text-anchor="end" '
               f'fill="#cf222e">gate: mean ratio {gate}</text>')
    for number, point in enumerate(points, 1):
        colour = OUTCOME_COLOURS.get(point["outcome"], "#24292f")
        kind = _marker_kind(point["arm"])
        legend_y = plot_height + 16 * (number - 1) + 4
        out.append(_marker(kind, left + 6, legend_y - 4, colour)
                   .replace('class="point"', 'class="legend"'))
        ratio = "n/a" if point["mean_ratio"] is None else f"{point['mean_ratio']:.4f}"
        where = "" if point in quantized else ", off scale"
        size = "n/a" if point["ssd_bytes"] is None else f"{point['ssd_bytes']:,} B"
        out.append(f'<text x="{left + 18}" y="{legend_y:.1f}">{number}. '
                   f'{_escape(point["label"])}: {size}, ratio {ratio}, '
                   f'{point["outcome"]}{where}</text>')
        if point not in quantized:
            continue
        x, y = px(point["ssd_bytes"] / scale), py(point["mean_ratio"])
        if point["band"]:
            out.append(f'<line x1="{x:.1f}" y1="{py(point["band"][0]):.1f}" x2="{x:.1f}" '
                       f'y2="{py(point["band"][1]):.1f}" stroke="{colour}" stroke-width="2"/>')
        out.append(_marker(kind, x, y, colour))
        on_left = kind == "circle"
        out.append(f'<text x="{x - 9 if on_left else x + 9:.1f}" y="{y - 7:.1f}" '
                   f'text-anchor="{"end" if on_left else "start"}" fill="{colour}">'
                   f'{number}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _runs(paths: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    files: list[pathlib.Path] = []
    for path in map(pathlib.Path, paths):
        files += sorted(path.rglob("run.json")) if path.is_dir() else [path]
    return [json.loads(file.read_text(encoding="utf-8")) for file in files], [
        file.as_posix() for file in files]


def _load(path: str) -> dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def control_main(argv: list[str]) -> int:
    """``control CONTROL REFERENCE``: exit 0 if the control reproduces C5, 1 if not."""
    parser = argparse.ArgumentParser(prog="g5_verdict.py control",
                                     description=reproduction_control.__doc__)
    parser.add_argument("control", help="run.json of NF4 seed 42 on the new box")
    parser.add_argument("reference", help="C5's run.json of NF4 seed 42")
    parser.add_argument("--out", required=True, help="where control.json is written")
    args = parser.parse_args(argv)
    result = reproduction_control(_load(args.control), _load(args.reference))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    failed = [name for name, ok in result["checks"].items() if not ok]
    label = "SYNTHETIC " if result["synthetic"] else ""
    if "incomplete" in result:
        console.print(f"{label}control: fail, not comparable: {result['incomplete']} -> {out}")
        return 1
    console.print(f"{label}control: {'pass' if result['passed'] else 'fail'}, task loss "
                  f"{result['task_loss']:.6f} against {result['reference_task_loss']:.6f} "
                  f"(relative diff {result['relative_task_loss_diff']:.6f}, limit "
                  f"{result['limit']}), gate-suite diff {result['gate_suite_diff']}, "
                  f"failed checks {failed or 'none'} -> {out}")
    return 0 if result["passed"] else 1


def provenance_main(argv: list[str]) -> int:
    """``provenance C5_RUN --out FILE``: exit 0 if C5's code here hashes as in C5_RUN."""
    parser = argparse.ArgumentParser(prog="g5_verdict.py provenance",
                                     description=provenance.__doc__)
    parser.add_argument("reference", help="C5's run.json the control will be compared with")
    parser.add_argument("--out", required=True, help="where provenance.json is written")
    args = parser.parse_args(argv)
    result = provenance(_load(args.reference))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for name, entry in result["files"].items():
        console.print(f"  {name}: here {entry['sha256'][:12]} (LF {entry['sha256_lf'][:12]}), "
                      f"C5's record {entry['recorded'][:12]}")
    if result["c5_code_matches"]:
        verdict_text = "C5's code matches C5's record"
    elif result["eol_only"]:
        verdict_text = ("C5's code differs from C5's record in line endings only (CRLF "
                        "checkout); check out LF files, see .gitattributes")
    else:
        verdict_text = "C5's code differs from C5's record"
    console.print(f"provenance: {verdict_text}; package differences from C5's record "
                  f"{result['version_diff'] or 'none'} -> {out}")
    return 0 if result["c5_code_matches"] else 1


def plot_main(argv: list[str]) -> int:
    """``plot VERDICT --out SVG``: the quality-vs-bytes chart from a written verdict.json."""
    parser = argparse.ArgumentParser(prog="g5_verdict.py plot", description=plot_main.__doc__)
    parser.add_argument("verdict", help="verdict.json written by this script")
    parser.add_argument("--out", required=True, help="where the SVG is written")
    args = parser.parse_args(argv)
    result = _load(args.verdict)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg_plot(result, "G5 quality against full cache bytes"), encoding="utf-8")
    label = f"{SYNTHETIC_LABEL}: " if (
        result.get("synthetic") or result.get("reportable") is False) else ""
    console.print(f"{label}{len(result['points'])} points -> {out}")
    return 0


SUBCOMMANDS = {"control": control_main, "plot": plot_main, "provenance": provenance_main}
#: Exit codes: 0 pass (or computed), 1 a DECIDED fail (control or provenance), 2 the check
#: itself could not run (unreadable or malformed input, I/O error, a bug). The callers read
#: 1 as "re-run C5's arms" / "C5's code differs", so a crash must never come back as 1.
EXIT_UNDECIDED = 2


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except SystemExit:
        raise  # argparse usage errors already exit 2
    except Exception as exc:  # noqa: BLE001 - every unexpected failure is "undecided"
        import traceback

        traceback.print_exc()
        console.print(f"[red]g5_verdict.py could not decide[/]: {type(exc).__name__}: {exc} "
                      f"(exit {EXIT_UNDECIDED}, not a failed check)")
        return EXIT_UNDECIDED


def _main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] and argv[0] in SUBCOMMANDS:
        return SUBCOMMANDS[argv[0]](argv[1:])
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="g5_verdict.py control CONTROL REFERENCE --out FILE checks the control alone; "
               "g5_verdict.py plot VERDICT --out SVG redraws the chart from verdict.json; "
               "g5_verdict.py provenance C5_RUN --out FILE checks C5's code before any run.")
    parser.add_argument("paths", nargs="+", help="mixed-arm run.json files or directories")
    parser.add_argument("--baselines", action="append", required=True,
                        help="C5 arm runs (nf4, q2_0, bf16): C5's records or the re-run ones")
    parser.add_argument("--control", required=True,
                        help="run.json of NF4 seed 42 on the box of the mixed arms")
    parser.add_argument("--control-reference", required=True,
                        help="C5's run.json of NF4 seed 42 that the control must match")
    parser.add_argument("--selection", required=True, help="selection.json from g5_select.py")
    parser.add_argument("--bytes", required=True, help="bytes.json from g5_bytes.py")
    parser.add_argument("--swap-record", action="append", default=[],
                        help="a C5 ticket-02 JSON whose non-expert fingerprints must match too")
    parser.add_argument("--synthetic", action="store_true",
                        help="the runs are synthetic: the pipeline runs, the verdict is none")
    parser.add_argument("--out", required=True, help="where verdict.json is written")
    args = parser.parse_args(argv)
    if not args.synthetic and not args.swap_record:
        parser.error("a real verdict checks row 1 against the ticket-02 JSON: pass --swap-record")

    mixed, mixed_files = _runs(args.paths)
    baselines, baseline_files = _runs(args.baselines)
    bytes_record = _load(args.bytes)
    result = verdict(
        mixed, baselines, control=_load(args.control),
        control_reference=_load(args.control_reference), selection=_load(args.selection),
        bytes_record=bytes_record, swap_records=[_load(p) for p in args.swap_record],
        synthetic=args.synthetic,
    )
    bf16 = bf16_point(bytes_record, baselines, result["baselines"]["verdict"]["used"])
    if bf16 is not None:
        result["points"].append(bf16)
    result["inputs"] = {
        "mixed": mixed_files, "baselines": baseline_files, "control": args.control,
        "control_reference": args.control_reference, "selection": args.selection,
        "bytes": args.bytes, "swap_records": list(args.swap_record),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c5._json_ready(result), indent=1) + "\n", encoding="utf-8")
    (out.parent / "quality_vs_bytes.svg").write_text(
        svg_plot(result, "G5 quality against full cache bytes"), encoding="utf-8")
    label = f"{SYNTHETIC_LABEL}: " if args.synthetic else ""
    diff = result["control"]["relative_task_loss_diff"]
    console.print(f"{label}control: {'pass' if result['control']['passed'] else 'fail'} "
                  f"(relative task-loss diff {'n/a' if diff is None else f'{diff:.6f}'}"
                  f"{', incomplete' if 'incomplete' in result['control'] else ''}),"
                  f" baselines {result['baselines']['source']}")
    for arm, entry in result["arms"].items():
        ratio = "n/a" if entry["mean_ratio"] is None else f"{entry['mean_ratio']:.4f}"
        computed = (f" (pipeline computed {entry['pipeline_outcome']})"
                    if "pipeline_outcome" in entry else "")
        size = "n/a" if entry["ssd_bytes"] is None else f"{entry['ssd_bytes']:,} B"
        size_ratio = ("n/a" if entry["ssd_bytes_vs_all_nf4"] is None
                      else f"{entry['ssd_bytes_vs_all_nf4']:.3f}")
        console.print(f"  {arm}: {entry['outcome']}{computed}, mean ratio {ratio}, "
                      f"{size} ({size_ratio} of all-NF4),"
                      f" fail reasons {entry['fail_reasons'] or 'none'}, invalid rows "
                      f"{[k for k, ok in entry['validity'].items() if not ok] or 'none'}")
    console.print(f"{label}G5 decision: {result['decision']['g5']}, reportable "
                  f"{result['reportable']}; {result['decision']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
