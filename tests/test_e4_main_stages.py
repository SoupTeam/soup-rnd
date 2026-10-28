"""E4 main series, ticket 1a: per-stage upload to a private HF repo and journal resume.

Seams (the four behaviours the request names, plus the restore and private-repo acceptance
items of ticket 1a): resume skips completed stages; a crash mid-stage resumes that stage without
a new void, and a real void still counts; upload runs after each finished stage, adapters
included; a failed upload retries and never marks the stage incomplete. Children are real
processes that use the real journal; Hugging Face is replaced by an in-memory hub.
"""

# ruff: noqa: I001 (the support import must run before the harness imports)

import shutil
import types
from pathlib import Path

import pytest

from tests.e4_part2_support import REPO_ROOT  # noqa: F401 (puts the harness on sys.path)
import e4_main_stages as stages  # noqa: E402
import e4_part2_calibration as cal  # noqa: E402
import e4_part2_common as common  # noqa: E402
import e4_part2_transfer as transfer  # noqa: E402

HARNESS = REPO_ROOT / "benchmarks" / "harness"

#: A stage child. Its mode comes from ``plan-<run>.txt`` (consumed one word per spawn):
#: ``ok`` writes a record and an adapter, ``void`` writes a real void, ``crash`` dies with
#: the attempt started and not ended, the way a killed process leaves it.
CHILD = """
import os, sys
sys.path.insert(0, "__HARNESS__")
import e4_part2_common as common

out, run_id, spawns = sys.argv[1], sys.argv[2], sys.argv[3]
with open(spawns, "a") as log:
    log.write(run_id + "\\n")
plan_path = os.path.join(out, "plan-" + run_id + ".txt")
if not os.path.exists(plan_path):
    os._exit(3)
with open(plan_path) as handle:
    modes = handle.read().split()
if not modes:
    os._exit(4)
with open(plan_path, "w") as handle:
    handle.write(" ".join(modes[1:]))
mode = modes[0]
journal = common.AttemptsJournal(os.path.join(out, "attempts.jsonl"))
attempt = journal.start(run_id, "F")
if mode == "crash":
    os._exit(137)
runs = os.path.join(out, "runs")
os.makedirs(runs, exist_ok=True)
with open(os.path.join(runs, attempt + ".json"), "w") as handle:
    handle.write("{}")
if mode == "ok":
    adapter = os.path.join(out, "adapters", run_id)
    os.makedirs(adapter, exist_ok=True)
    with open(os.path.join(adapter, "adapter_model.safetensors"), "wb") as handle:
        handle.write(b"weights")
    journal.end(attempt, common.OUTCOME_OK)
else:
    journal.end(attempt, common.OUTCOME_VOID, ["peer: foreign process on the GPU"])
"""


class FakeHub:
    """The private dataset repo as a dict of name -> bytes, with upload failures injected."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.objects: dict = {}
        self.fail_remaining: dict = {}
        self.upload_calls: list = []  # (name, journal lines at the time of the call)
        self.sleeps: list = []

    def upload(self, local: Path, name: str) -> None:
        self.upload_calls.append((name, journal_lines(self.out)))
        if self.fail_remaining.get(name, 0) > 0:
            self.fail_remaining[name] -= 1
            raise OSError(f"network down while sending {name}")
        self.objects[name] = Path(local).read_bytes()

    def download(self, name: str, directory: Path) -> bool:
        if name not in self.objects:
            return False
        target = Path(directory) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.objects[name])
        return True

    def store(self, out: Path) -> stages.StageStore:
        return stages.StageStore(out, self.upload, self.download, sleep=self.sleeps.append)


@pytest.fixture
def ws(tmp_path):
    child = tmp_path / "stage_child.py"
    child.write_text(CHILD.replace("__HARNESS__", str(HARNESS)))
    out = tmp_path / "out"
    out.mkdir()
    return types.SimpleNamespace(child=child, out=out, spawns=tmp_path / "spawns.log",
                                 hub=FakeHub(out))


def journal_lines(out: Path) -> int:
    path = out / "attempts.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def plan(out: Path, run_id: str, *modes: str) -> None:
    (out / f"plan-{run_id}.txt").write_text(" ".join(modes))


def spawned(ws) -> list:
    return ws.spawns.read_text().split() if ws.spawns.exists() else []


def stage(ws, store, run_id: str):
    return stages.run_stage(
        store, run_id, [str(ws.child), str(store.out), run_id, str(ws.spawns)],
        adapters=[f"adapters/{run_id}"],
        stamp=lambda: {"gpu_uuid": "GPU-test"}, identity=lambda stamp: {})


def outcomes(store, run_id: str) -> list:
    return [a["outcome"] for a in store.journal.attempts(run_id)]


# --- resume ------------------------------------------------------------------------------


def test_resume_skips_stages_already_completed_in_the_journal(ws):
    plan(ws.out, "S1", "ok")
    plan(ws.out, "S2", "ok")
    store = ws.hub.store(ws.out)
    stage(ws, store, "S1")
    stage(ws, store, "S2")
    before = journal_lines(ws.out)

    plan(ws.out, "S3", "ok")
    restarted = ws.hub.store(ws.out)  # a new process: fresh objects, same directory
    for run_id in ("S1", "S2", "S3"):
        stage(ws, restarted, run_id)

    assert spawned(ws) == ["S1", "S2", "S3"]  # S1 and S2 were not spawned again
    new_events = common.read_jsonl(ws.out / "attempts.jsonl")[before:]
    assert [e["run_id"] for e in new_events if e["event"] == "start"] == ["S3"]
    assert len(new_events) == 2  # start and end of S3, nothing else


# --- crash and void counting -------------------------------------------------------------


def test_crash_mid_stage_resumes_the_stage_with_one_void(ws):
    plan(ws.out, "S1", "crash", "ok")
    store = ws.hub.store(ws.out)

    stage(ws, store, "S1")

    assert store.journal.void_count("S1") == 1
    assert outcomes(store, "S1") == ["void", "ok"]
    assert store.journal.attempts("S1")[0]["reasons"][0].startswith("interrupted")
    assert cal.completed(store.journal, "S1")
    assert spawned(ws) == ["S1", "S1"]


def test_restart_after_the_parent_died_mid_stage_adds_no_second_void(ws):
    plan(ws.out, "S1", "ok")
    dead = ws.hub.store(ws.out)
    dead.journal.start("S1", "F")  # the process died here: start written, no end, no record

    store = ws.hub.store(ws.out)
    stage(ws, store, "S1")

    assert store.journal.void_count("S1") == 1
    assert outcomes(store, "S1") == ["void", "ok"]
    assert spawned(ws) == ["S1"]


def test_a_real_void_still_counts_and_a_second_one_stops_the_run(ws):
    plan(ws.out, "S1", "void", "void")
    store = ws.hub.store(ws.out)

    with pytest.raises(common.CalibrationStopped):
        stage(ws, store, "S1")
    assert store.journal.void_count("S1") == 2

    plan(ws.out, "S1", "ok")
    restarted = ws.hub.store(ws.out)
    with pytest.raises(common.CalibrationStopped):
        stage(ws, restarted, "S1")  # the count survives the restart; no third child runs
    assert spawned(ws) == ["S1", "S1"]


def test_one_real_void_then_ok_leaves_one_void(ws):
    plan(ws.out, "S1", "void", "ok")
    store = ws.hub.store(ws.out)

    stage(ws, store, "S1")

    assert outcomes(store, "S1") == ["void", "ok"]
    assert store.journal.void_count("S1") == 1


# --- upload ------------------------------------------------------------------------------


def test_upload_runs_after_each_finished_stage_including_adapters(ws):
    plan(ws.out, "S1", "ok")
    plan(ws.out, "S2", "ok")
    store = ws.hub.store(ws.out)

    stage(ws, store, "S1")
    adapter = "adapters/S1/adapter_model.safetensors"
    record = f"runs/{store.journal.attempts('S1')[0]['attempt_id']}.json"
    assert ws.hub.objects[adapter] == b"weights"
    assert record in ws.hub.objects
    assert ws.hub.objects["attempts.jsonl"] == (ws.out / "attempts.jsonl").read_bytes()
    # sent only once the stage's end was journaled (two lines: start and end)
    assert all(lines == 2 for name, lines in ws.hub.upload_calls if name == adapter)

    stage(ws, store, "S2")
    assert "adapters/S2/adapter_model.safetensors" in ws.hub.objects
    assert ws.hub.objects["attempts.jsonl"] == (ws.out / "attempts.jsonl").read_bytes()


def test_failed_upload_retries_and_never_marks_the_stage_incomplete(ws):
    adapter = "adapters/S1/adapter_model.safetensors"
    ws.hub.fail_remaining[adapter] = 2  # first two sends fail, the third succeeds
    plan(ws.out, "S1", "ok")
    store = ws.hub.store(ws.out)

    stage(ws, store, "S1")

    assert [call for call in ws.hub.upload_calls if call[0] == adapter] == [
        (adapter, 2), (adapter, 2), (adapter, 2)]
    assert ws.hub.objects[adapter] == b"weights"
    assert store.pending() == []
    assert len(ws.hub.sleeps) == 2  # backed off between the tries


def test_persistent_upload_failure_keeps_the_stage_complete_and_blocks_new_work(ws):
    adapter = "adapters/S1/adapter_model.safetensors"
    ws.hub.fail_remaining[adapter] = 100
    plan(ws.out, "S1", "ok")
    plan(ws.out, "S2", "ok")
    store = ws.hub.store(ws.out)

    stage(ws, store, "S1")  # returns normally: the stage finished

    assert cal.completed(store.journal, "S1")
    assert store.journal.void_count("S1") == 0
    assert adapter in store.pending()
    with pytest.raises(stages.UploadPendingError):
        stage(ws, store, "S2")  # pending uploads come before any new work
    assert spawned(ws) == ["S1"]

    ws.hub.fail_remaining.clear()
    stage(ws, store, "S2")
    assert store.pending() == []
    assert ws.hub.objects[adapter] == b"weights"
    assert spawned(ws) == ["S1", "S2"]


# --- restore and the private repo ----------------------------------------------------------


def test_restore_pulls_the_journal_and_records_when_the_local_copy_is_missing(ws):
    plan(ws.out, "S1", "void", "ok")
    store = ws.hub.store(ws.out)
    stage(ws, store, "S1")
    shutil.rmtree(ws.out)  # the box's output directory is lost

    fresh = ws.hub.store(ws.out)
    stage(ws, fresh, "S1")  # nothing is spawned: the pulled journal says S1 is complete

    assert spawned(ws) == ["S1", "S1"]
    assert fresh.journal.void_count("S1") == 1
    assert outcomes(fresh, "S1") == ["void", "ok"]
    assert fresh.pending() == []


class FakeApi:
    def __init__(self, private: bool) -> None:
        self.private = private
        self.uploads: list = []

    def create_repo(self, repo_id, **kwargs):
        pass

    def repo_info(self, repo_id, repo_type):
        return types.SimpleNamespace(private=self.private)

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)


def test_hf_store_refuses_a_repo_that_is_not_private(tmp_path):
    api = FakeApi(private=False)

    with pytest.raises(stages.PublicRepoError):
        stages.hf_store(tmp_path / "out", "me/e4-main", api=api)
    assert api.uploads == []


def test_hf_store_uploads_each_file_under_its_out_relative_name(tmp_path):
    api = FakeApi(private=True)
    store = stages.hf_store(tmp_path / "out", "me/e4-main", api=api)
    local = tmp_path / "out" / "adapters" / "S1" / "adapter_model.safetensors"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"weights")

    store.enqueue(["adapters/S1/adapter_model.safetensors"])
    assert store.send_pending() == []

    assert api.uploads == [{
        "path_or_fileobj": str(local), "path_in_repo": "adapters/S1/adapter_model.safetensors",
        "repo_id": "me/e4-main", "repo_type": "dataset"}]


def test_ensure_private_dataset_is_the_check_push_uses():
    assert transfer.ensure_private_dataset(FakeApi(private=True), "me/x") is True
    assert transfer.ensure_private_dataset(FakeApi(private=False), "me/x") is False
