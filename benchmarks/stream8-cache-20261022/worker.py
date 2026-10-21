"""One pinned8B experiment; preserve failures and raw evidence, then no further work."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path("/root/stream8-cache")
OUT = ROOT / "results"
REPO = ROOT / "repo"
sys.path[:0] = [str(ROOT), str(REPO), str(REPO / "src")]
POLICY = json.loads((ROOT / "policy.json").read_text())
SUMMARY = {"status": "STARTED", "stages": [], "arms": {}, "comparisons": []}


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def save():
    temporary = OUT / "progress.tmp"
    temporary.write_text(json.dumps(SUMMARY, indent=2))
    temporary.replace(OUT / "progress.json")


def stage(name, command, maximum=1200):
    remaining = POLICY["created_unix"] + POLICY["maximum_seconds"] - time.time()
    if remaining < 14400:
        raise RuntimeError("retain four hours for raw evidence transfer and cleanup")
    SUMMARY["current_stage"] = name
    save()
    start = time.time()
    with (OUT / (name + ".log")).open("w") as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=min(maximum, remaining - 14400))
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    SUMMARY["stages"].append({"name": name, "exit_code": code, "seconds": time.time() - start})
    save()
    return code == 0


def invoke(arm, indices, target, *extra):
    return [sys.executable, str(ROOT / "bench.py"), "--config", str(ROOT / "model-8.yaml"),
            "--arm", arm, "--indices", ",".join(map(str, indices)), "--output", str(target), *extra]


def compare(reference, candidate):
    import torch
    from benchmarks.harness.liger_streaming_parity import compare as original_compare
    from safetensors.torch import load_file

    result = original_compare(reference, candidate)
    first = load_file(str(reference / "trace.safetensors"))
    second = load_file(str(candidate / "trace.safetensors"))
    result["bitwise_equal"] = all(torch.equal(value, second[name]) for name, value in first.items())
    result["tensor_count"] = len(first)
    return result


def trajectory(reference, candidate):
    import torch
    from safetensors.torch import load_file

    first = load_file(str(reference / "adapter/adapter_model.safetensors"))
    second = load_file(str(candidate / "adapter/adapter_model.safetensors"))
    assert set(first) == set(second) and len(first) == SUMMARY["layer_count"] * 14
    metrics = [json.loads((path / "metrics.json").read_text()) for path in (reference, candidate)]
    losses = [[item["loss"] for item in metric["training_log_history"] if "loss" in item]
              for metric in metrics]
    assert all(len(values) == 14 for values in losses)
    for metric in metrics:
        assert metric["amp"] == {"bf16": True, "fp16": False}
        assert len(metric["seconds"]) == 8
        assert all(shape == [1, 1024] for shape in metric["input_shapes"])
    assert metrics[0]["initial_adapter_sha256"] == metrics[1]["initial_adapter_sha256"]
    assert metrics[0]["data_sha256"] == metrics[1]["data_sha256"]
    finite_close = all(torch.isfinite(value).all() and torch.isfinite(second[name]).all()
                       and torch.allclose(value, second[name], rtol=.01, atol=1e-5)
                       for name, value in first.items())
    loss_close = all(f"{a:.3f}" == f"{b:.3f}" for a, b in zip(*losses, strict=True))
    return {"passed": bool(finite_close and loss_close), "loss_three_decimals": loss_close,
            "adapter_tensors": len(first),
            "bitwise_equal": all(torch.equal(value, second[name]) for name, value in first.items())}


def main():
    assert stage("cpu-target-software", [sys.executable, "-m", "pytest",
                                        str(ROOT / "cache_test.py"), "-q", "--tb=short"])
    model = json.loads((ROOT / "protocol.json").read_text())["models"][0]
    SUMMARY["model"] = model
    assert stage("prepare", [sys.executable, str(ROOT / "prepare_model.py"), "8"], 1800)
    baseline = OUT / "capture-baseline"
    if not stage("capture-baseline", invoke("baseline", [], baseline, "--capture")):
        SUMMARY["status"] = "BASELINE_FAILED_NO_PRECISION_OR_SPEED_VERDICT"
        return
    info = json.loads((baseline / "cache-provenance.json").read_text())
    layers, layer_bytes = info["layer_count"], info["per_layer_bytes"]
    SUMMARY.update(layer_count=layers, baseline_capacity=info)
    available = 7 * 2**30 - info["peak_reserved_bytes"] - int(.75 * 2**30)
    # Same cap/reserve, deliberately different distributions. Retention charges
    # 512MiB/layer activation allowance plus immutable weight storage. Allocation
    # cap remains the hard boundary, OOM is recorded without reducing context.
    selections = {}
    for arm in ("resident", "retained", "both"):
        per_layer = layer_bytes + (512 * 2**20 if arm != "resident" else 0)
        count = min(layers // 2, max(0, available // per_layer))
        start = (layers - count) // 2
        selections[arm] = list(range(start, start + count))
    SUMMARY["selection"] = {"indices": selections, "available_bytes": available,
                            "activation_allowance_bytes_per_layer": 512 * 2**20,
                            "reserve_bytes": int(.75 * 2**30)}
    assert stage("semantics-baseline", invoke("baseline", [], OUT / "semantics-baseline",
                                              "--semantics"))
    qualified = []
    for arm, indices in selections.items():
        report = SUMMARY["arms"][arm] = {"indices": indices}
        if not indices:
            report["status"] = "NO_HEADROOM_UNDER_FIXED_CAP"
            continue
        target = OUT / ("capture-" + arm)
        if not stage("capture-" + arm, invoke(arm, indices, target, "--capture")):
            report["status"] = "CAPTURE_FAILED_NO_TIMING"
            continue
        report["capture"] = compare(baseline, target)
        if not report["capture"]["passed"]:
            report["status"] = "ACCURACY_REJECTED_NO_TIMING"
            save()
            continue
        semantic = OUT / ("semantics-" + arm)
        if not stage("semantics-" + arm, invoke(arm, indices, semantic, "--semantics")):
            report["status"] = "SEMANTICS_FAILED_NO_TIMING"
            continue
        report["semantics"] = compare(OUT / "semantics-baseline", semantic)
        if report["semantics"]["passed"]:
            report["status"] = "QUALIFIED_FOR_TIMING"
            qualified.append(arm)
        else:
            report["status"] = "SEMANTICS_REJECTED_NO_TIMING"
        save()
    for arm in qualified:
        for pair, order in enumerate(("AB", "BA", "BA", "AB")):
            destinations = {}
            for letter in order:
                current = "baseline" if letter == "A" else arm
                dest = OUT / f"timing-{arm}-{pair}-{current}"
                destinations[current] = dest
                assert stage(dest.name, invoke(current, [] if current == "baseline"
                                                else selections[current], dest))
            check = trajectory(destinations["baseline"], destinations[arm])
            SUMMARY["comparisons"].append({"arm": arm, "pair": pair, "order": order,
                                           "trajectory": check})
            save()
            if not check["passed"]:
                raise RuntimeError("final trajectory failed; discard performance verdict")
        SUMMARY["arms"][arm]["status"] = "TIMED_REQUIRES_INDEPENDENT_ANALYSIS"
    SUMMARY["status"] = "FINISHED_REQUIRES_INDEPENDENT_ANALYSIS"


def export():
    SUMMARY["current_stage"] = "hash-and-export-raw"
    save()
    manifest = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size,
                                              "sha256": digest(path)}
                for path in sorted(OUT.rglob("*")) if path.is_file()}
    for name in ("policy.json", "protocol.json", "bench.py", "candidate.py", "worker.py",
                 "prepare_model.py", "constraints.txt", "source.tar.gz"):
        path = ROOT / name
        manifest[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with tarfile.open(ROOT / "evidence.tar.gz", "w:gz", compresslevel=1) as archive:
        archive.add(ROOT / "manifest.json", arcname="manifest.json")
        for name in manifest:
            archive.add(ROOT / name, arcname=name)
    path = ROOT / "evidence.tar.gz"
    (ROOT / "export.json").write_text(json.dumps({"bytes": path.stat().st_size,
                                                  "sha256": digest(path)}))
    (ROOT / "worker.exited").touch()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    try:
        main()
    except BaseException as error:
        SUMMARY.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        (OUT / "error.log").write_text(traceback.format_exc())
    finally:
        save()
        export()
