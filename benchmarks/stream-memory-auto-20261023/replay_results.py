"""Read-only CPU replay of retained native-residency cohort B results.

No provisioning, GPU work, source changes or raw-result deletion. The exact
archived comparison/trajectory definitions and timing summary are SHA-checked
before execution; heavy tensor dependencies are imported only by those checks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any

from rich.console import Console


def read_bytes(path: Path) -> bytes:
    """Read existing files directly, or contained archival paths from evidence.zip."""
    if path.is_file():
        return path.read_bytes()
    root = os.path.realpath(Path(__file__).parent)
    target = os.path.realpath(path)
    try:
        contained = os.path.commonpath([root, target]) == root and target != root
    except ValueError:
        contained = False
    if not contained:
        raise FileNotFoundError(f"missing path outside the archival report root: {path}")
    member = os.path.relpath(target, root).replace(os.sep, "/")
    import zipfile

    archive = Path(root) / "evidence.zip"
    try:
        with zipfile.ZipFile(archive) as handle:
            info = handle.getinfo(member)
            if info.is_dir():
                raise FileNotFoundError(f"archival member is a directory: {member}")
            return handle.read(info)
    except KeyError as error:
        raise FileNotFoundError(f"missing archival member: {member}") from error


def read_json(path: Path) -> Any:
    return json.loads(read_bytes(path).decode("utf-8"))


def serialized_trace(path: Path) -> dict[str, Any]:
    """Hash the entire file while bounding ordinary read-buffer memory."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 2**20), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("retained trace changed while hashing")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def scan_identical_trace(path: Path, max_slice_elements: int) -> dict[str, Any]:
    """Visit every tensor element in CPU slices, including scalars and the loss."""
    import torch
    from safetensors import safe_open

    if type(max_slice_elements) is not int or max_slice_elements < 1:
        raise ValueError("max_slice_elements must be a positive integer")
    groups: dict[str, Any] = {}
    checked_elements = 0
    maximum_observed_slice = 0
    nonfinite = []
    has_nan = False
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if "loss" not in keys:
            raise ValueError("trace is missing the original comparator's required loss key")
        if len(handle.get_slice("loss").get_shape()) != 1:
            raise ValueError("original comparator requires a one-dimensional loss tensor")
        for name in keys:
            view = handle.get_slice(name)
            shape = tuple(view.get_shape())
            if any(length == 0 for length in shape):
                raise ValueError("original comparator does not define empty-tensor max differences")
            tile = [1] * len(shape)
            remaining = max_slice_elements
            for axis in reversed(range(len(shape))):
                tile[axis] = min(shape[axis], remaining)
                remaining = max(1, remaining // tile[axis])
            starts = itertools.product(*[
                range(0, length, width) for length, width in zip(shape, tile)
            ])
            finite = True
            elements = 0
            for start in starts:
                tensor = (
                    view[tuple(slice(offset, min(offset + width, length))
                               for offset, width, length in zip(start, tile, shape))]
                    if shape else handle.get_tensor(name)
                )
                count = tensor.numel()
                if count > max_slice_elements:
                    raise ValueError("bounded finiteness slice exceeded its element limit")
                elements += count
                maximum_observed_slice = max(maximum_observed_slice, count)
                if not bool(torch.isfinite(tensor).all()):
                    finite = False
                    has_nan = has_nan or bool(torch.isnan(tensor).any())
                del tensor
            if elements != math.prod(shape):
                raise ValueError("bounded finiteness scan did not cover the full tensor")
            checked_elements += elements
            group = name.split("/")[0]
            stats = groups.setdefault(group, {
                "tensors": 0, "failed_tensors": 0, "max_abs": 0.0, "failures": [],
            })
            stats["tensors"] += 1
            if not finite:
                nonfinite.append(name)
                stats["failed_tensors"] += 1
                stats["failures"].append(name)
    return {
        "groups": groups, "tensor_count": len(keys), "checked_elements": checked_elements,
        "finite": not nonfinite, "nonfinite_tensors": nonfinite,
        "contains_nan": has_nan, "maximum_observed_slice_elements": maximum_observed_slice,
        "max_slice_elements": max_slice_elements,
    }


def compare_traces(
    reference: Path, candidate: Path, original_compare: Any, *, max_slice_elements: int = 131072
) -> dict[str, Any]:
    """Exact serialized equality plus bounded finiteness, or the unchanged comparator.

    Equal serialization proves equal keys, shapes, dtypes and all stored values.
    Finiteness is still checked independently: equal NaN/Inf files cannot pass.
    Small unequal traces use the original function and its unchanged tolerances.
    Large unequal files refuse low-memory admission, without a numerical verdict.
    """
    paths = [folder / "trace.safetensors" for folder in (reference, candidate)]
    snapshots = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    identities = [serialized_trace(path) for path in paths]
    exact = identities[0] == identities[1]
    proof: dict[str, Any] = {
        "reference": identities[0], "candidate": identities[1],
        "identical_full_serialized_trace": exact,
    }
    if exact:
        before = paths[0].stat()
        scan = scan_identical_trace(paths[0], max_slice_elements)
        after = paths[0].stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("retained reference trace changed during its bounded scan")
        for path, snapshot in zip(paths, snapshots):
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != snapshot:
                raise ValueError("retained trace changed after serialized equality check")
        return {
            "passed": scan["finite"], "loss_three_decimals": True,
            "groups": scan["groups"], "rtol": 1e-2, "atol": 1e-5,
            # The archived worker calls torch.equal, which rejects equal NaNs.
            "bitwise_equal": not scan["contains_nan"], "tensor_count": scan["tensor_count"],
            "comparison_method": "exact serialized bytes plus independent bounded finiteness",
            "serialized_trace_proof": proof,
            "bounded_finiteness": {
                key: value for key, value in scan.items() if key not in {"groups", "tensor_count"}
            },
            "candidate_finiteness_basis": (
                "identical full file SHA256 and length to scanned reference"
            ),
        }
    from safetensors import safe_open

    # Admission bounds scratch for the unchanged double-matrix comparator on a
    # low-memory laptop. It is not a substitute tolerance or a failed-accuracy
    # verdict; larger unequal traces need replay on a host with adequate RAM.
    if sum(item["bytes"] for item in identities) > 64 * 2**20:
        raise ValueError("unequal traces: unchanged comparator refused low-memory byte admission")
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if any(math.prod(handle.get_slice(key).get_shape()) > 2**20 for key in handle.keys()):
                raise ValueError(
                    "unequal traces: unchanged comparator refused low-memory tensor admission"
                )
    verdict = original_compare(reference, candidate)
    verdict.update(
        comparison_method="unchanged archived comparator after bounded-memory admission",
        serialized_trace_proof=proof,
    )
    return verdict


def archived_functions(archive: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile only the named frozen definitions; never execute the worker body."""
    manifest = read_json(archive / "published-helpers-manifest.json")
    entries = {row["source_name"]: row for row in manifest["files"]}
    checked: dict[str, Any] = {}
    namespace: dict[str, Any] = {"json": json, "random": random, "statistics": statistics}
    for name, definitions in (
        ("worker.py", {"compare", "trajectory"}),
        ("timing_summary.py", {"summarize"}),
    ):
        row = entries[name]
        raw = read_bytes(archive / row["published_name"])
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != row["bytes"] or digest != row["sha256"]:
            raise ValueError(f"archived helper hash/size mismatch: {name}")
        expected = read_json(archive / "launch-manifest.json")[name]
        if digest != expected["sha256"] or len(raw) != expected["bytes"]:
            raise ValueError(f"archived helper differs from frozen B launch: {name}")
        tree = ast.parse(raw.decode("utf-8"), filename=name)
        selected = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in definitions
        ]
        if {node.name for node in selected} != definitions:
            raise ValueError(f"archived helper definitions missing: {name}")
        executable = ast.Module(body=selected, type_ignores=[])
        exec(compile(executable, f"<archived:{name}>", "exec"), namespace)
        checked[name] = {"bytes": len(raw), "sha256": digest}
    for name in ("bench.py", "protocol.json"):
        row = entries[name]
        raw = read_bytes(archive / row["published_name"])
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"archived observation/protocol hash mismatch: {name}")
        checked[name] = {"bytes": len(raw), "sha256": row["sha256"]}
    comparator_path = "benchmarks/harness/liger_streaming_parity.py"
    inventory = read_json(archive / "source-manifest.json")
    expected = next(row for row in inventory["files"] if row["path"] == comparator_path)
    spec = importlib.util.find_spec("benchmarks.harness.liger_streaming_parity")
    if spec is None or spec.origin is None:
        raise ValueError("original comparator module is unavailable")
    raw = Path(spec.origin).read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    normalization = "none"
    if len(raw) != expected["bytes"] or actual_hash != expected["sha256"]:
        # Git can turn the archived comparator's CRLF into LF on Linux. Admit
        # that representation only when newline reconstitution recovers every
        # original byte, including the frozen size and SHA-256. Other edits fail.
        reconstituted = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if (len(reconstituted) != expected["bytes"]
                or hashlib.sha256(reconstituted).hexdigest() != expected["sha256"]):
            raise ValueError("original pretrained comparator source differs from frozen source")
        normalization = "canonical LF then CRLF reconstitution matched frozen bytes exactly"
    checked[comparator_path] = {
        "bytes": len(raw), "sha256": actual_hash,
        "frozen_bytes": expected["bytes"], "frozen_sha256": expected["sha256"],
        "source_representation_normalization": normalization,
    }
    return namespace, checked


def result_root(supplied: Path) -> Path:
    for folder in (supplied, supplied / "results"):
        if (folder / "progress.json").is_file():
            return folder
    raise ValueError("retained results must contain progress.json, directly or under results/")


def expected_adapter_count(model: dict[str, Any], capture: dict[str, Any]) -> int:
    count = int(model["config"]["num_hidden_layers"]) * 14
    if capture.get("adapter_tensors") != count:
        raise ValueError("capture adapter count differs from the frozen model configuration")
    return count


def provenance(
    folder: Path, arm: str, expected_entry_sha: str, expected_layers: int
) -> dict[str, Any]:
    info = read_json(folder / "residency-auto-provenance.json")
    if info.get("status") != "COMPLETED_NOT_AN_INDEPENDENT_VERDICT":
        raise ValueError(f"incomplete process provenance: {folder.name}")
    if info.get("entry_sha256") != expected_entry_sha:
        raise ValueError(f"executed benchmark entry SHA differs: {folder.name}")
    if info.get("arm") != arm or info.get("actual_mode") != {"baseline": 0, "auto": "auto"}[arm]:
        raise ValueError(f"actual arm/mode differs: {folder.name}")
    if info.get("layer_count") != expected_layers:
        raise ValueError(f"decoder count differs: {folder.name}")
    if not 7.5 * 2**30 <= info.get("physical_gpu_bytes", 0) <= 8.1 * 2**30:
        raise ValueError(f"physical GPU capacity differs: {folder.name}")
    indices = info.get("indices")
    if not isinstance(indices, list) or indices != sorted(set(indices)):
        raise ValueError(f"invalid actual resident indices: {folder.name}")
    if any(type(index) is not int or not 0 <= index < expected_layers for index in indices):
        raise ValueError(f"resident index outside decoder: {folder.name}")
    if arm == "baseline" and (indices or info.get("resident_bytes") != 0):
        raise ValueError("baseline unexpectedly has private resident weights")
    if arm == "auto":
        plan = info.get("residency_plan")
        if not isinstance(plan, dict) or plan.get("indices") != indices:
            raise ValueError(f"planner and actual pool selection differ: {folder.name}")
    return info


def replay_model(
    folders: list[Path],
    model: dict[str, Any],
    saved: dict[str, Any],
    functions: dict[str, Any],
    protocol: dict[str, Any],
    entry_sha: str,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"model_id": model["model_id"], "size_b": model["size_b"]}

    def child(name: str) -> Path:
        for folder in reversed(folders):
            candidate = folder / name
            if candidate.exists():
                return candidate
        return folders[0] / name

    primary = [child("capture-baseline"), child("capture-auto")]
    semantics = [child("semantics-baseline"), child("semantics-auto")]
    required = [path / "trace.safetensors" for path in primary + semantics]
    if not all(path.is_file() for path in required):
        report.update(
            status="INCOMPLETE_ACCURACY",
            missing=[str(path) for path in required if not path.is_file()],
        )
        return report
    captures = [read_json(path / "capture.json") for path in primary]
    if any(item.get("status") != "CAPTURED" for item in captures):
        raise ValueError("primary capture process did not complete")
    adapters = expected_adapter_count(model, captures[0])
    expected_adapter_count(model, captures[1])
    semantic_info = [read_json(path / "study-provenance.json") for path in semantics]
    for item in semantic_info:
        if item.get("status") != "CAPTURED" or not (
            item.get("causal_prefix_exact") and item.get("masked_padding_isolation_exact")
        ):
            raise ValueError("causal/padding isolation process did not complete its checks")
    layers = int(model["config"]["num_hidden_layers"])
    report["capture_processes"] = {
        arm: provenance(path, arm, entry_sha, layers)
        for arm, path in zip(("baseline", "auto"), primary)
    }
    report["gates"] = {
        "primary": functions["compare"](*primary),
        "semantics": functions["compare"](*semantics),
    }
    if not all(gate["passed"] for gate in report["gates"].values()):
        report["status"] = "FAILED_ACCURACY"
        return report
    report["accuracy_passed"] = True
    if saved.get("status") == "ACCURACY_PASSED_AUTO_FELL_BACK_NO_RESIDENT_LAYERS":
        if report["capture_processes"]["auto"]["indices"]:
            raise ValueError("saved capacity fallback conflicts with nonempty auto selection")
        if any(path.is_dir() for folder in folders for path in folder.glob("timing-pair-*-*")):
            raise ValueError("saved no-timing capacity fallback has timing directories")
        report.update(
            status="COMPLETE_ACCURACY_AUTO_CAPACITY_FALLBACK_NO_TIMING",
            complete_timing_pairs=0,
            performance=None,
            capacity_verdict="auto selected no resident layers; ordinary streaming retained",
        )
        return report
    order = protocol["pair_order"]
    if order != ["AB", "BA", "BA", "AB"]:
        raise ValueError("frozen protocol pair order differs")
    expected_folders = {
        f"timing-pair-{number}-{arm}" for number in range(4) for arm in ("baseline", "auto")
    }
    observed_folders = {
        path.name for folder in folders
        for path in folder.glob("timing-pair-*-*") if path.is_dir()
    }
    if observed_folders - expected_folders:
        raise ValueError("unexpected extra timing pair directories")
    complete_pairs = []
    for number in range(4):
        paths = [child(f"timing-pair-{number}-{arm}") for arm in ("baseline", "auto")]
        needed = [
            path / name for path in paths
            for name in ("metrics.json", "residency-auto-provenance.json",
                         "adapter/adapter_model.safetensors")
        ]
        if all(path.is_file() for path in needed):
            complete_pairs.append((number, paths))
    report["complete_timing_pairs"] = len(complete_pairs)
    report["timed_pairs"] = []
    report["trajectories"] = {}
    process_ids = []
    for number, paths in complete_pairs:
        checked = functions["trajectory"](*paths, adapters)
        report["trajectories"][str(number)] = checked
        if not checked["passed"]:
            report["status"] = "FAILED_TIMING_TRAJECTORY"
            return report
        pair: dict[str, Any] = {"pair": number, "order": order[number]}
        for arm, path in zip(("baseline", "auto"), paths):
            metrics = read_json(path / "metrics.json")
            if metrics.get("status") != "TIMING_CAPTURED_NOT_A_CORRECTNESS_OR_QUALITY_VERDICT":
                raise ValueError(f"timing process did not complete: {path.name}")
            if not metrics.get("deterministic_algorithms"):
                raise ValueError(f"timing determinism was disabled: {path.name}")
            times = metrics["seconds"]
            if any(not math.isfinite(value) or value <= 0 for value in times):
                raise ValueError("timing array contains non-finite/non-positive values")
            info = provenance(path, arm, entry_sha, layers)
            process_ids.append(info["pid"])
            pair[arm] = times
            pair[arm + "_actual_resident_indices"] = info["indices"]
            pair[arm + "_resident_bytes"] = info["resident_bytes"]
            pair[arm + "_measured_peak_allocated_bytes"] = max(metrics["peak_allocated_bytes"])
            pair[arm + "_measured_peak_reserved_bytes"] = max(metrics["peak_reserved_bytes"])
            pair[arm + "_setup_sampled_peak_allocated_bytes"] = info["setup_peak_allocated_bytes"]
            pair[arm + "_setup_sampled_peak_reserved_bytes"] = info["setup_peak_reserved_bytes"]
        report["timed_pairs"].append(pair)
    if len(complete_pairs) != 4:
        report["status"] = "ACCURACY_ONLY_TIMING_PENDING"
        return report
    if len(set(process_ids)) != 8:
        raise ValueError("timing pairs do not describe eight distinct fresh processes")
    prefix = f"timing-{model['size_b']}-"
    actual_stages = [row for row in stages if row["name"].startswith(prefix)]
    expected_stages = [
        f"{prefix}{number}-{'baseline' if letter == 'A' else 'auto'}"
        for number, pair_order in enumerate(order) for letter in pair_order
    ]
    if [row["name"] for row in actual_stages] != expected_stages:
        raise ValueError("recorded stage history differs from exact protocol timing order")
    if any(row.get("exit_code") != 0 for row in actual_stages):
        raise ValueError("recorded timing stage did not exit successfully")
    report["recorded_stage_order_matches_protocol"] = True
    performance = functions["summarize"](report["timed_pairs"])
    report["performance"] = performance
    report["saved_performance_matches_exactly"] = performance == saved.get("performance")
    if not report["saved_performance_matches_exactly"]:
        report["status"] = "FAILED_SAVED_PERFORMANCE_REPLAY"
        return report
    report["status"] = "COMPLETE_ACCURACY_AND_TIMING_REPLAYED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="+", type=Path,
        help="one or more retained result roots, or their parents containing results/",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive = Path(__file__).parent / "executed"
    report: dict[str, Any] = {
        "status": "STARTED", "results": [str(path) for path in args.results], "models": {},
        "scope": "read-only CPU evidence replay; no model training or GPU allocation",
        "memory_definition": "measured update peaks exclude setup; setup peaks are sampled",
        "throughput_definition": "2048 processed input positions per optimizer update",
    }
    try:
        functions, verified = archived_functions(archive)
        original_compare = functions["compare"]
        functions["compare"] = lambda reference, candidate: compare_traces(
            reference, candidate, original_compare
        )
        report["verified_archived_helpers"] = verified
        protocol = read_json(archive / "protocol.json")
        roots = [result_root(path) for path in args.results]
        snapshots = [(root, read_json(root / "progress.json")) for root in roots]
        # Bundles are incremental snapshots. Prefer the longest recorded stage
        # history; on a tie the later supplied root wins. Raw tensors are read
        # directly from their original bundle, without merging or copying them.
        selected = max(
            enumerate(snapshots),
            key=lambda item: (len(item[1][1].get("stages", [])),
                              len(item[1][1].get("bundles", [])), item[0]),
        )
        progress_root, progress = selected[1]
        report["saved_progress_source"] = str(progress_root / "progress.json")
        report["saved_worker_status"] = progress.get("status")
        models = {int(item["size_b"]): item for item in protocol["models"]}
        for size in (8, 14):
            folders = [root / f"model-{size}" for root in roots
                       if (root / f"model-{size}").is_dir()]
            saved = progress.get("models", {}).get(str(size), {})
            if not folders:
                report["models"][str(size)] = {
                    "status": "NOT_PRESENT_NOT_REPLAYED", "saved_status": saved.get("status")
                }
                continue
            try:
                report["models"][str(size)] = replay_model(
                    folders, models[size], saved, functions, protocol,
                    verified["bench.py"]["sha256"], progress.get("stages", []),
                )
            except Exception as error:
                report["models"][str(size)] = {
                    "status": "FAILED", "error": f"{type(error).__name__}: {error}"
                }
        statuses = [item["status"] for item in report["models"].values()]
        if any(status.startswith("FAILED") for status in statuses):
            report["status"] = "FAILED"
        elif any(status in {
            "COMPLETE_ACCURACY_AND_TIMING_REPLAYED",
            "COMPLETE_ACCURACY_AUTO_CAPACITY_FALLBACK_NO_TIMING",
        } for status in statuses):
            pending = any(status in {"INCOMPLETE_ACCURACY", "ACCURACY_ONLY_TIMING_PENDING"}
                          for status in statuses)
            report["status"] = "PARTIAL_MODELS_PENDING" if pending else "COMPLETE"
        else:
            report["status"] = "INCOMPLETE_NO_COMPLETE_PERFORMANCE_REPLAY"
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Console().print({"status": report["status"], "report": str(args.output)})
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
