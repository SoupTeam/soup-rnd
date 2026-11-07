"""Continue timing from retained exact-GPU numerical proof, verifying all identities."""

import importlib.metadata
import json
import subprocess
import sys
import traceback

import worker


def main():
    root, out = worker.ROOT, worker.OUT
    reference = json.loads((root / "qualified-reference.json").read_text())
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        text=True).strip()
    uuid, name, driver = [value.strip() for value in gpu.split(",")]
    assert uuid == reference["gpu_uuid"], "different physical GPU: cannot reuse numerical proof"
    assert driver == reference["driver_version"], "GPU driver changed"
    assert worker.POLICY["source_sha256"] == reference["source_sha256"]
    packages = {item.metadata["Name"].lower(): item.version
                for item in importlib.metadata.distributions()}
    for package, version in reference["packages"].items():
        assert packages[package.lower()] == version, "software changed: " + package
    state = reference["qualified_state"]
    assert all(report["capture"]["passed"] and report["semantics"]["passed"]
               for report in state["arms"].values())
    worker.SUMMARY = state
    worker.SUMMARY.update(status="TIMING_REUSES_RETAINED_EXACT_GPU_NUMERICAL_PROOF",
                          stages=[], comparisons=[], bundles=[],
                          verified_acknowledged_bundles=[],
                          qualification_reference={"original_instance": 57170199,
                                                   "all_raw_files_already_verified": 140,
                                                   "archive_sha256": reference["accuracy_sha256"],
                                                   "same_gpu_uuid": uuid,
                                                   "same_driver": driver,
                                                   "same_source_and_packages": True})
    assert worker.stage("target-regressions", [sys.executable, "-m", "pytest",
                                              str(root / "resume_test.py"), "-q", "--tb=short"])
    assert worker.stage("prepare", [sys.executable, str(root / "prepare_model.py"), "8"], 1800)
    assert worker.digest(root / "train.jsonl") == reference["data_sha256"]
    model = json.loads((out / "model-8-manifest.json").read_text())
    assert model == reference["model_manifest"], "pinned snapshot bytes changed"
    (out / "qualification-reference.json").write_text(json.dumps(reference, indent=2))
    (out / "environment.json").write_text(json.dumps({
        "gpu": gpu, "packages": packages, "source_sha256": worker.POLICY["source_sha256"],
        "python": sys.version, "native_regressions_executed": True}, indent=2))
    (out / "timing_worker.py").write_bytes((root / "timing_worker.py").read_bytes())
    worker.SUMMARY["layer_count"] = 36
    for arm in ("resident", "retained", "both"):
        indices = state["selection"]["indices"][arm]
        for pair, order in enumerate(("AB", "BA", "BA", "AB")):
            destinations = {}
            for letter in order:
                current = "baseline" if letter == "A" else arm
                dest = out / f"timing-{arm}-{pair}-{current}"
                destinations[current] = dest
                assert worker.stage(dest.name, worker.invoke(
                    current, [] if current == "baseline" else indices, dest))
                metrics = json.loads((dest / "metrics.json").read_text())
                assert metrics["initial_adapter_sha256"] == reference["initial_adapter_sha256"]
                assert metrics["data_sha256"] == reference["data_sha256"]
            check = worker.trajectory(destinations["baseline"], destinations[arm])
            worker.SUMMARY["comparisons"].append({"arm": arm, "pair": pair, "order": order,
                                                 "trajectory": check})
            worker.save()
            if not check["passed"]:
                raise RuntimeError("final14-update trajectory gate failed")
        worker.SUMMARY["arms"][arm]["status"] = "TIMED_REQUIRES_INDEPENDENT_ANALYSIS"
        worker.bundle("timing-" + arm)
    worker.SUMMARY["status"] = "FINISHED_REQUIRES_INDEPENDENT_ANALYSIS"


if __name__ == "__main__":
    worker.OUT.mkdir(exist_ok=True)
    try:
        main()
    except BaseException as error:
        worker.SUMMARY.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        (worker.OUT / "timing-error.log").write_text(traceback.format_exc())
    finally:
        worker.save()
        try:
            worker.bundle("final")
        finally:
            (worker.ROOT / "worker.exited").touch()
