#!/usr/bin/env python3
"""Unattended Moonlight evaluation driver for one rented host.

One sequential pass under the unchanged frozen protocol. Every scientific command runs
once; a FAILED shard stops the whole run (no retry). Steps are resumable only after an
infrastructure interruption (driver killed / host restart): a step without a completion

marker is re-entered, and an evaluation directory without a manifest is moved aside.
Stdlib only; runs with the system python3.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/p3-moonlight-quality")
AUTO = ROOT / "autorun"
INPUTS = AUTO / "inputs"
MARKERS = AUTO / "markers"
STATUS = AUTO / "status.json"
STOP = AUTO / "STOP"
LOG = AUTO / "driver-steps.log"
RUN_ID = "moonlight-resident-20261111t201548z-whole3"
EVIDENCE = ROOT / "evidence" / RUN_ID
PROTOCOL = EVIDENCE / "frozen-study-protocol.json"
PROTOCOL_SHA256 = "d22871c398d29e25c614129a1a084903fbc0dce09a5bb5d2659c08d9102b2f82"
SCORING_SHA256 = "685fa5bee528969d9b2eef423c90280d179432ba017fbebed12b687f608a8249"
SOURCE_PACKET_SHA256 = "d0ff819e27bdcea710b0cdea6a4e16f3fb8121e3100d7271ecc391ce4013a1df"
PROFILE = "moonlight-16b-a3b"
VENV = Path("/opt/moonlight-py312")
PY = str(VENV / "bin/python")
REPO = ROOT / "review-source"
SOURCE = ROOT / "sources/476b36a473d4467f94469414bef6cee75c9c8172"
CACHE = ROOT / "cache/476b36a473d4467f94469414bef6cee75c9c8172"
TOKENIZER = ROOT / "tokenizer/476b36a473d4467f94469414bef6cee75c9c8172"
# Same absolute paths as every shard on the first host (env sourced before REPO override).
DATA = ROOT / "source/benchmarks/results/p3-stage2-v2/data-build-kaggle-v1/evidence/data"
SCORING = ROOT / "source/benchmarks/results/p3-stage2-v2/p3-stage2-v2-protocol.json"
OBSERVER = AUTO / "moonlight_measured_entry.py"
EVAL_SECONDS = 20746  # frozen loglik.deadline_seconds; one process per shard
# Stop launching new scientific work after this instant (leaves time to export/delete).
LAUNCH_CUTOFF_UTC = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)

ARMS = ("bf16_l2l", "oracle_block64_e4m3")
LENGTHS = (1024, 4096)
ADAPTER_MANIFEST_SHA256 = {
    ("bf16_l2l", 1024): "7f78f7020e4ec3f5e1a5a51c3f722ed664dac197cf727fbdfc51fca012edbd9c",
    ("oracle_block64_e4m3", 1024):
        "aee6c2dfadeeaa93587b2522608b52b48356829f2a8ec99300c1a43b214c2aec",
    ("bf16_l2l", 4096): "968a8b2198bd90c7b90634fe2895b527e27b950d7adb3adc6522c66510a2e187",
    ("oracle_block64_e4m3", 4096):
        "86e07e75df5f2ed8e17fa7fca1e39da0b5154bb85ace8de51a109836633110ed",
}
# Contiguous task-index shards; each process stays far below the ~3510 profiler sessions
# (= forwards) after which the earlier HellaSwag runs refused. ARC runs whole (1381, the
# size that already completed and that the reused BF16 L1024 shard used, so batch
# composition stays identical for that pair). Exact counts are verified by plan_shards.py.
SHARDS = {  # HellaSwag first: it is the endpoint that failed, so a refusal shows up early
    "hellaswag": [(start, start + 600) for start in range(0, 9000, 600)],
    "boolq": [(0, 3270)],
    "lambada": [(0, 5153)],
    "mbpp": [(0, 494)],
    "arc": [(0, 3548)],
    "gsm8k": [(0, 644), (644, 1288)],
}
FORWARD_LIMIT = 1400
# Completed COMPLETE shards from the first host under the same protocol; uploaded, not rerun.
REUSED = {("bf16_l2l", 1024, "arc", 0, 3548), ("bf16_l2l", 1024, "boolq", 0, 3270)}
PROBE_CYCLES = 5000
MAX_SESSIONS_PER_PROCESS = 1400


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"{utc()} {message}\n")


STATE: dict = {}


def status(state: str, step: str, **extra) -> None:
    STATE.update(state=state, step=step, utc=utc(), pid=os.getpid(), **extra)
    write_json(STATUS, STATE)
    log(f"{state} {step} {json.dumps(extra, sort_keys=True)}")


def done(name: str) -> bool:
    return (MARKERS / name).exists()


def mark(name: str, value: dict | None = None) -> None:
    write_json(MARKERS / name, {"utc": utc(), **(value or {})})


class Stopped(Exception):
    pass


class Failed(Exception):
    pass


def env() -> dict:
    value = dict(os.environ)
    value.update({
        "PROFILE": PROFILE, "ROOT": str(ROOT), "REPO": str(REPO), "RUN_ID": RUN_ID,
        "EVIDENCE": str(EVIDENCE), "SOURCE": str(SOURCE), "CACHE": str(CACHE),
        "TOKENIZER": str(TOKENIZER), "DATA": str(DATA), "SCORING_PROTOCOL": str(SCORING),
        "PATH": f"{VENV}/bin:" + value.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": f"{REPO}/src:{REPO}", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "23", "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    })
    return value


def run(name: str, command: list[str], seconds: int) -> int:
    """Run one command in its own session; honour STOP; never retry."""
    output = AUTO / "logs" / f"{name}.log"
    with output.open("ab") as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env(), stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        deadline = time.monotonic() + seconds + 300
        try:
            while True:
                try:
                    return process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
                if STOP.exists() or time.monotonic() > deadline:
                    reason = "STOP requested" if STOP.exists() else "driver deadline"
                    log(f"terminating {name}: {reason}")
                    raise Stopped(f"{name}: {reason}")
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
            raise


def check_stop(cutoff: bool = True) -> None:
    if STOP.exists():
        raise Stopped("STOP requested")
    if cutoff and datetime.now(timezone.utc) > LAUNCH_CUTOFF_UTC:
        raise Stopped("launch cutoff reached")


# --------------------------------------------------------------------------- steps

def step_setup() -> None:
    if run("setup", ["bash", str(AUTO / "setup.sh")], 3600) != 0:
        raise Failed("setup.sh failed")
    protocol = json.loads((INPUTS / "frozen-study-protocol.json").read_bytes())
    for relative, expected in protocol["execution_overlays_sha256"].items():
        actual = sha256(REPO / relative)
        if actual != expected:
            raise Failed(f"overlay hash mismatch {relative}")


def step_inputs() -> None:
    if sha256(INPUTS / "frozen-study-protocol.json") != PROTOCOL_SHA256:
        raise Failed("frozen protocol hash mismatch")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.exists():
        shutil.copy2(INPUTS / "frozen-study-protocol.json", PROTOCOL)
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise Failed("placed protocol hash mismatch")
    for (arm, length), expected in ADAPTER_MANIFEST_SHA256.items():
        source = INPUTS / "adapters" / f"train-{arm}-L{length}" / "final_adapter"
        target = EVIDENCE / f"train-{arm}-L{length}" / "final_adapter"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        manifest = json.loads((target / "manifest.json").read_bytes())
        if sha256(target / "manifest.json") != expected:
            raise Failed(f"adapter manifest hash mismatch {arm} L{length}")
        for relative, digest in manifest["files"].items():
            if relative != "manifest.json" and sha256(target / relative) != digest:
                raise Failed(f"adapter file hash mismatch {arm} L{length} {relative}")
    for arm, length, endpoint, start, stop in REUSED:
        name = f"eval-{arm}-L{length}-{endpoint}-{start}-{stop}"
        target = EVIDENCE / name
        if not target.exists():
            shutil.copytree(INPUTS / "reused" / name, target)
        manifest = json.loads((target / "manifest.json").read_bytes())
        if manifest.get("status") != "COMPLETE" or not manifest.get("selected_shard_complete"):
            raise Failed(f"reused shard not complete {name}")
        for relative, descriptor in manifest["files"].items():
            if sha256(target / relative) != descriptor["sha256"]:
                raise Failed(f"reused shard file hash mismatch {name} {relative}")


def step_probe() -> None:
    result = EVIDENCE / "h2-profiler-probe.json"
    if result.exists():
        result.unlink()
    code = run("probe", [PY, "-B", str(AUTO / "profiler_session_probe.py"), str(result),
                         str(PROBE_CYCLES)], 3600)
    if code != 0 or not result.exists():
        raise Failed("profiler probe did not complete")
    value = json.loads(result.read_text())
    first = value["first_empty_kernels_cycle"]
    # User decision 2026-11-12: a probe that does not reproduce is information, not a gate. The
    # toy loop (no model, one attention call per cycle, ~2 ms per session) does not run the
    # evaluator's real conditions. The first real HellaSwag piece is the test; a refused piece
    # stops the run. A reproduction BELOW twice the piece size would still refuse here.
    if first is None:
        log("PROBE_NOT_REPRODUCED: informational only (user decision); continuing")
        mark("probe", {**value, "gate": "informational"})
        return
    if first < 2 * MAX_SESSIONS_PER_PROCESS:
        raise Failed(f"PROBE_LIMIT_TOO_LOW: first empty cycle {first}")
    mark("probe", value)


def stage_wrapper(name: str, seconds: int, child: list[str]) -> int:
    return run(name, [PY, "-m", "benchmarks.harness.p3_deepseek_quality_stage",
                      "--output", str(EVIDENCE / f"{name}-process"), "--seconds", str(seconds),
                      "--", *child], seconds)


def download_passed(out: Path) -> bool:
    manifest = out / "manifest.json"
    if not manifest.exists():
        return False
    value = json.loads(manifest.read_bytes())
    return value.get("command") == "download" and value.get("status") == "PASS"


def step_download() -> None:
    out = EVIDENCE / "h2-a-download"
    receipt = SOURCE / "checkpoint-identity.json"
    if SOURCE.exists() and receipt.exists() and download_passed(out):
        return  # an earlier pass of this step finished and admitted the source
    # prepare.py:617-618 refuses an existing SOURCE without --resume. --resume re-verifies
    # every complete file, discards .partial files and refuses a source holding a receipt
    # (prepare.py:567-568). A receipt without a PASS manifest is left by a failed admission:
    # move it out; it is regenerated deterministically after the re-verification.
    resume = SOURCE.exists()
    if resume and receipt.exists():
        shutil.move(str(receipt), str(EVIDENCE / f"stale-checkpoint-identity-{int(time.time())}.json"))
    stamp = int(time.time())
    for path in (out, EVIDENCE / "h2-a-download-process"):
        if path.exists():
            shutil.move(str(path), str(path) + f".interrupted-{stamp}")
    child = [PY, "-m", "benchmarks.harness.p3_deepseek_quality_prepare", "download",
             "--profile", PROFILE, "--source", str(SOURCE), "--output", str(out),
             "--approved-pilot"]
    if resume:
        child.append("--resume")
    code = stage_wrapper("h2-a-download", 6600, child)
    if code != 0 or not download_passed(out):
        raise Failed("weight download did not finish with a PASS manifest")


def step_restore_identity() -> None:
    code = run("restore-identity", [PY, "-B", str(AUTO / "restore_mtimes.py"),
                                    str(PROTOCOL), str(SOURCE)], 600)
    if code != 0:
        raise Failed("source mtime restoration failed")


INITIAL_PARAMETER_SHA256 = "cf5b676c42dc0be973668d3adf6087d59a21d958bf6dca7dbc50361bfc5892ac"


def read_safetensors(path: Path) -> tuple[dict, dict]:
    """(metadata, {name: (dtype, shape, raw bytes)}); stdlib only, key order ignored."""
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + size])
    meta = header.pop("__metadata__", {})
    base = 8 + size
    return meta, {name: (item["dtype"], item["shape"],
                         raw[base + item["data_offsets"][0]:base + item["data_offsets"][1]])
                  for name, item in header.items()}


def step_convert() -> None:
    out = EVIDENCE / "h2-b-convert"
    for path in (out, EVIDENCE / "h2-b-convert-process"):
        if path.exists():
            shutil.move(str(path), str(path) + f".interrupted-{int(time.time())}")
    if CACHE.exists():
        shutil.move(str(CACHE), str(CACHE) + f".interrupted-{int(time.time())}")
    code = stage_wrapper("h2-b-convert", 3600, [
        PY, "-m", "benchmarks.harness.p3_deepseek_quality_prepare", "convert",
        "--profile", PROFILE, "--source", str(SOURCE), "--cache", str(CACHE),
        "--output", str(out), "--approved-pilot"])
    if code != 0:
        raise Failed("cache conversion failed")
    # The four adapters were trained from this exact initial adapter. Compare the WEIGHTS, not
    # the file hash: safetensors headers can list __metadata__ keys in a different order on a
    # different host (observed 2026-11-12: all 216 tensors byte-identical, file hash different).
    original = INPUTS / "initial-adapter.safetensors"
    expected = json.loads(PROTOCOL.read_bytes())["initial_adapter_sha256"]
    if sha256(original) != expected:
        raise Failed("bundled original initial adapter does not match the frozen hash")
    rebuilt = out / "initial-adapter.safetensors"
    meta_new, tensors_new = read_safetensors(rebuilt)
    meta_old, tensors_old = read_safetensors(original)
    write_json(EVIDENCE / "h2-initial-adapter-check.json", {
        "file_sha256_rebuilt": sha256(rebuilt), "file_sha256_frozen": expected,
        "parameter_sha256_rebuilt": meta_new.get("parameter_sha256"),
        "parameter_sha256_frozen": meta_old.get("parameter_sha256"),
        "tensor_count": len(tensors_new),
        "tensors_identical": tensors_new == tensors_old,
        "metadata_identical": meta_new == meta_old})
    if tensors_new != tensors_old or meta_new.get("parameter_sha256") != INITIAL_PARAMETER_SHA256:
        raise Failed("INITIAL_ADAPTER_TENSORS_DIFFER: the rebuilt initial weights differ from "
                     "the ones the adapters were trained from; see h2-initial-adapter-check.json")


def step_identity_check() -> None:
    adapters = []
    for (arm, length), digest in ADAPTER_MANIFEST_SHA256.items():
        adapters += ["--adapter", f"{EVIDENCE / f'train-{arm}-L{length}/final_adapter'}:{digest}"]
    code = run("identity-check", [
        PY, "-B", str(AUTO / "dry_admission.py"), "--source", str(SOURCE), "--cache", str(CACHE),
        "--tokenizer-dir", str(TOKENIZER), "--protocol", str(PROTOCOL),
        "--protocol-sha256", PROTOCOL_SHA256, "--scoring-protocol", str(SCORING),
        "--resident", *adapters], 3600)
    if code != 0:
        raise Failed("IDENTITY_MISMATCH: live source/cache/tokenizer/adapter admission differs "
                     "from the frozen protocol; see logs/identity-check.log")


FIRST_NEW = ("oracle_block64_e4m3", 1024, "boolq", 0, 3270)
OLD_BOOLQ = ("bf16_l2l", 1024, "boolq", 0, 3270)


def shard_list() -> list[tuple[str, int, str, int, int]]:
    items = []
    for length in LENGTHS:
        for arm in ARMS:
            for endpoint, intervals in SHARDS.items():
                for start, stop in intervals:
                    items.append((arm, length, endpoint, start, stop))
    items.remove(FIRST_NEW)
    return [FIRST_NEW, *items]


def step_plan() -> None:
    """Exact CPU forward counts for every planned shard; refuse oversized shards."""
    out = EVIDENCE / "h2-shard-plan.json"
    if out.exists():
        out.unlink()
    code = run("plan", [PY, "-B", str(AUTO / "plan_shards.py"), str(out), str(FORWARD_LIMIT),
                        str(TOKENIZER), str(DATA),
                        json.dumps({k: list(v) for k, v in SHARDS.items()})], 3600)
    if not out.exists():
        raise Failed("shard plan did not complete")
    summary = json.loads(out.read_text())["_summary"]
    if code != 0 or summary["max_forwards"] > FORWARD_LIMIT:
        raise Failed(f"SHARD_TOO_LARGE: {json.dumps(summary)}")
    mark("plan", summary)


def shard_dir(arm: str, length: int, endpoint: str, start: int, stop: int) -> Path:
    return EVIDENCE / f"eval-{arm}-L{length}-{endpoint}-{start}-{stop}"


def complete_shard(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    try:
        value = json.loads(manifest.read_bytes())
    except ValueError:  # half-written manifest after a kill
        return False
    return value.get("status") == "COMPLETE" and value.get("selected_shard_complete") is True


def manifest_status(path: Path) -> str:
    """missing | corrupt (unparseable, e.g. killed mid-write) | complete | failed."""
    manifest = path / "manifest.json"
    if not manifest.exists():
        return "missing"
    try:
        value = json.loads(manifest.read_bytes())
    except ValueError:
        return "corrupt"
    if value.get("status") == "COMPLETE" and value.get("selected_shard_complete") is True:
        return "complete"
    return "failed"


def cross_check_once(out: Path) -> None:
    """Old-host and new-host shards must pair; otherwise score the reused cells again here."""
    if done("cross-check"):
        return
    code = run("cross-check", [
        PY, "-B", str(AUTO / "cross_check.py"),
        str(shard_dir(*OLD_BOOLQ)), str(out), str(EVIDENCE / "h2-cross-check.json")], 1800)
    if code == 0:
        mark("cross-check", {"result": "PAIR_OK"})
        return
    if code != 3:
        raise Failed("NEW_SHARD_REFUSED: the new-host shard alone is refused by the pairing "
                     "collector; see logs/cross-check.log and h2-cross-check.json")
    # The new shard is admissible alone but cannot pair with the old-host shards. Protocol and
    # scoring are unchanged: move the two reused shards aside and let the loop score them here.
    rejected = EVIDENCE / "old-host-shards-rejected"
    rejected.mkdir(exist_ok=True)
    for arm, length, endpoint, start, stop in REUSED:
        source = shard_dir(arm, length, endpoint, start, stop)
        if source.exists():
            shutil.move(str(source), str(rejected / source.name))
    mark("cross-check", {"result": "FALLBACK_RERUN_REUSED_SHARDS"})
    log("old-host shards cannot pair with new-host shards; they will be scored again here")


def step_evaluate() -> None:
    items = shard_list()
    total = len(items)
    for index, (arm, length, endpoint, start, stop) in enumerate(items, start=1):
        out = shard_dir(arm, length, endpoint, start, stop)
        name = out.name
        if complete_shard(out):
            if (arm, length, endpoint, start, stop) == FIRST_NEW:
                cross_check_once(out)
            continue
        stamp = int(time.time())
        if manifest_status(out) == "failed":  # a real refusal is a result, never retried
            raise Failed(f"shard {name} has a non-COMPLETE manifest; no retry")
        # A kill before run_endpoint creates `out` still leaves -process/-observation behind,
        # and both would make the wrapper's mkdir(exist_ok=False) fail on re-entry.
        for path in (out, EVIDENCE / f"{name}-process", EVIDENCE / f"{name}-observation"):
            if path.exists():
                shutil.move(str(path), str(path) + f".interrupted-{stamp}")
                log(f"moved aside interrupted {path.name}")
        check_stop()
        status("RUNNING", "evaluate", shard=name, shard_index=index, shard_total=total)
        code = stage_wrapper(name, EVAL_SECONDS, [
            PY, "-B", str(OBSERVER), str(EVIDENCE / f"{name}-observation/observation.json"),
            "benchmarks.harness.p3_deepseek_quality_eval", "run-endpoint",
            "--profile", PROFILE, "--data-root", str(DATA), "--source", str(SOURCE),
            "--cache", str(CACHE), "--tokenizer-dir", str(TOKENIZER),
            "--adapter", str(EVIDENCE / f"train-{arm}-L{length}/final_adapter"),
            "--adapter-sha256", ADAPTER_MANIFEST_SHA256[(arm, length)],
            "--protocol", str(PROTOCOL), "--protocol-sha256", PROTOCOL_SHA256,
            "--scoring-protocol", str(SCORING), "--length", str(length), "--device", "cuda",
            "--endpoint", endpoint, "--start", str(start), "--stop", str(stop),
            "--output", str(out)])
        if not complete_shard(out):
            failure = {}
            if (out / "manifest.json").exists():
                failure = json.loads((out / "manifest.json").read_bytes()).get("failure", {})
            raise Failed(f"shard {name} failed (exit {code}): {json.dumps(failure)}")
        if (arm, length, endpoint, start, stop) == FIRST_NEW:
            cross_check_once(out)


def step_pair() -> None:
    baseline, candidate = [], []
    for arm, length, endpoint, start, stop in shard_list():
        path = shard_dir(arm, length, endpoint, start, stop)
        if not complete_shard(path):
            raise Failed(f"missing complete shard {path.name}")
        (baseline if arm == ARMS[0] else candidate).append(str(path))
    out = EVIDENCE / "paired"
    for path in (out, EVIDENCE / "pair-process"):
        if path.exists():
            shutil.move(str(path), str(path) + f".interrupted-{int(time.time())}")
    code = stage_wrapper("pair", 7200, [
        PY, "-m", "benchmarks.harness.p3_deepseek_quality_eval", "pair",
        "--profile", PROFILE, "--baseline", *baseline, "--candidate", *candidate,
        "--output", str(out), "--data-root", str(DATA), "--protocol", str(PROTOCOL),
        "--protocol-sha256", PROTOCOL_SHA256, "--scoring-protocol", str(SCORING)])
    manifest = out / "manifest.json"
    if code != 0 or not manifest.exists() or \
            json.loads(manifest.read_bytes()).get("status") != "COMPLETE":
        raise Failed("pairing did not complete")


def step_analysis() -> None:
    summary = EVIDENCE / "quality-summary.json"
    if summary.exists():
        summary.rename(summary.with_name(f"quality-summary.interrupted-{int(time.time())}.json"))
    code = run("analysis", [PY, "-B", str(AUTO / "quality_summary.py"), str(EVIDENCE / "paired"),
                            str(SCORING), PROTOCOL_SHA256], 3600)
    if code != 0 or not summary.exists():
        raise Failed("analysis did not complete")


STEPS = [
    ("setup", step_setup), ("inputs", step_inputs), ("probe", step_probe),
    ("plan", step_plan),
    ("download", step_download), ("restore-identity", step_restore_identity),
    ("convert", step_convert), ("identity-check", step_identity_check),
    ("evaluate", step_evaluate), ("pair", step_pair), ("analysis", step_analysis),
]


def package(state: str, detail: str) -> None:
    """Archive all evidence produced here (not weights/cache/venv) with a SHA256 file."""
    exports = ROOT / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    name = f"{RUN_ID}-h2-final-{state.lower()}.tar.gz"
    archive = exports / name
    if archive.exists():
        archive.unlink()
    excluded = ("/train-", "final_adapter")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(str(AUTO), arcname="autorun",
                filter=lambda item: None if "/autorun/inputs" in "/" + item.name else item)
        if EVIDENCE.exists():
            for child in sorted(EVIDENCE.iterdir()):
                if any(part in f"/{child.name}" for part in excluded):
                    continue
                tar.add(str(child), arcname=f"evidence/{child.name}")
    with tarfile.open(archive, "r:gz") as check:  # a truncated archive is never shipped
        members = len(check.getmembers())
    digest = sha256(archive)
    (exports / (name + ".sha256")).write_text(f"{digest}  {name}\n", encoding="utf-8")
    pieces = sum(1 for item in shard_list() if complete_shard(shard_dir(*item))
                 and item not in REUSED)
    status(state, "packaged", detail=detail, archive=str(archive), archive_sha256=digest,
           archive_bytes=archive.stat().st_size, archive_members=members,
           completed_pieces=pieces)


LOCK = AUTO / "driver.lock"
_lock = None


def single_instance() -> None:
    global _lock
    _lock = LOCK.open("w")  # children never inherit it: Popen closes fds by default
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(3)  # another driver is alive; touch nothing


def on_signal(number, _frame) -> None:
    raise Stopped(f"driver received signal {number}")


def wait_for_orphans() -> None:
    """The stage wrapper and entry run in their own sessions and survive a dead driver."""
    me = os.getpid()
    while True:
        busy = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == me:
                continue
            try:
                cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if "benchmarks.harness.p3_deepseek_quality_" in cmd or any(
                    name in cmd for name in ("moonlight_measured_entry.py", "dry_admission.py",
                                             "plan_shards.py", "cross_check.py",
                                             "profiler_session_probe.py", "quality_summary.py")):
                busy.append(entry.name)
        if not busy:
            return
        log(f"waiting for orphaned harness processes {busy}")
        if STOP.exists():
            raise Stopped("STOP requested while orphaned processes were running")
        time.sleep(30)


def main() -> int:
    single_instance()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    for path in (AUTO / "logs", MARKERS):
        path.mkdir(parents=True, exist_ok=True)
    try:
        wait_for_orphans()
        for name, function in STEPS:
            if done(name) and not (name == "setup" and not Path(PY).exists()):
                continue
            check_stop(cutoff=name not in ("pair", "analysis"))
            status("RUNNING", name)
            function()
            mark(name)
        summary = json.loads((EVIDENCE / "quality-summary.json").read_bytes())
        return finish("DONE", f"gated={summary.get('gated_quality_decision')} "
                              f"statistical={summary.get('statistical_quality_decision')}", 0)
    except Stopped as error:
        return finish("STOPPED", str(error), 2)
    except Failed as error:
        return finish("FAILED", str(error), 1)
    except BaseException as error:  # noqa: BLE001 - every exit must leave an archive
        return finish("FAILED", f"{type(error).__name__}: {error}", 1)


def finish(state: str, detail: str, code: int) -> int:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        package(state, detail)
    except BaseException as error:  # noqa: BLE001 - never leave RUNNING without evidence
        status("FAILED", "package-failed", detail=f"{state}: {detail}; packaging failed: "
                                                  f"{type(error).__name__}: {error}")
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
