"""E4 main series, ticket 1a: stages that survive a restart and keep their records off the box.

A stage is one journaled attempt process of ``run_id`` (calibration rule section 1: one process,
one journal entry). ``run_stage`` runs it through the calibration driver (``drive_run``), so a
crash is journaled as a void once, the stage is rerun in place, and a second void stops the study
(draft section 8). It never changes the journal to make the upload work.

Off-box copy (nurik fault-tolerance request, ticket 1a): after each stage finishes, its files go to
a private Hugging Face dataset repo: the attempt records, the journal and the stage's adapters.
A file is ``uploaded`` once the repo has it. A failed upload is retried, then kept in
``upload-state.json`` as pending. Pending files are sent before any new stage starts, and a
stage with pending files stays complete.

Layout under ``out`` (the calibration layout): ``attempts.jsonl``, ``runs/<attempt>.json``,
``adapters/<run>/...``, ``upload-state.json``. Upload names are these out-relative paths.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import e4_part2_calibration as cal
import e4_part2_common as common
import e4_part2_transfer as transfer

JOURNAL_NAME = "attempts.jsonl"
RUNS_DIR = "runs"
STATE_NAME = "upload-state.json"


class UploadPendingError(RuntimeError):
    """Files from finished stages are still not in the private repo; no new stage starts."""


class PublicRepoError(RuntimeError):
    """The target dataset repo exists and is not private; nothing is uploaded to it."""


class StageStore:
    """The out directory of one study, mirrored file by file to a remote store.

    ``upload(local, name)`` sends one file; ``download(name, directory)`` writes ``name`` under
    ``directory`` and returns False when the remote has no such file. Both raise on transport
    errors; ``send`` retries them.
    """

    def __init__(self, out: Path | str, upload: Callable[[Path, str], None],
                 download: Callable[[str, Path], bool], *, retries: int = 3,
                 sleep: Callable[[float], Any] = time.sleep) -> None:
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.journal = common.AttemptsJournal(self.out / JOURNAL_NAME)
        self.runs_dir = self.out / RUNS_DIR
        self._upload = upload
        self._download = download
        self._retries = retries
        self._sleep = sleep

    # --- state: uploaded and pending names, persisted with each change ---------------------

    def _state(self) -> Dict[str, List[str]]:
        path = self.out / STATE_NAME
        if not path.exists():
            return {"uploaded": [], "pending": []}
        return json.loads(path.read_text())

    def _save(self, state: Dict[str, List[str]]) -> None:
        (self.out / STATE_NAME).write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")

    def pending(self) -> List[str]:
        return list(self._state()["pending"])

    def enqueue(self, names: Sequence[str]) -> None:
        """Mark names for upload: not yet uploaded, present locally. The journal always goes
        again, because it grows with every stage."""
        state = self._state()
        done, pending = set(state["uploaded"]), set(state["pending"])
        for name in names:
            if not (self.out / name).is_file():
                continue
            if name in done and name != JOURNAL_NAME:
                continue
            pending.add(name)
        state["pending"] = sorted(pending)
        self._save(state)

    def send_pending(self) -> List[str]:
        """Try every pending name once, with retries. Returns the names still not sent.

        The journal goes last and only when every other pending file is in the repo. A journal
        sent ahead of a missing record or adapter would make the remote mark that stage complete.
        """
        still = []
        for name in _journal_last(self.pending()):
            if name == JOURNAL_NAME and still:
                still.append(name)  # withheld: a file of this pass is not in the repo yet
            elif self._send(name):
                self._mark_sent(name)
            else:
                still.append(name)
        return still

    def raise_if_pending(self) -> None:
        """Send what is pending; raise when some of it still cannot be sent."""
        if not self.pending():
            return
        still = self.send_pending()
        if still:
            raise UploadPendingError(
                f"{len(still)} file(s) from finished stages are not in the private repo "
                f"({still[:5]}); no new stage starts until they are")

    def restore(self) -> bool:
        """When the local journal is missing, pull the journal and its attempt records.

        The pulled names are uploaded by definition, so they are not queued again. Returns
        True when anything was restored.
        """
        if self.journal.path.exists():
            return False
        if not self._download(JOURNAL_NAME, self.out):
            return False
        restored = [JOURNAL_NAME]
        for event in self.journal.events():
            if event["event"] == "start":
                name = f"{RUNS_DIR}/{event['attempt_id']}.json"
                if self._download(name, self.out):
                    restored.append(name)
        self._save({"uploaded": sorted(restored), "pending": []})
        return True

    # --- transport --------------------------------------------------------------------------

    def _send(self, name: str) -> bool:
        for attempt in range(1, self._retries + 1):
            try:
                self._upload(self.out / name, name)
                return True
            except Exception as exc:  # any transport error: retry, then keep it pending
                print(f"upload of {name} failed (try {attempt} of {self._retries}): {exc}",
                      file=sys.stderr)
                if attempt < self._retries:
                    self._sleep(2 ** (attempt - 1))
        return False

    def _mark_sent(self, name: str) -> None:
        state = self._state()
        state["uploaded"] = sorted(set(state["uploaded"]) | {name})
        state["pending"] = [n for n in state["pending"] if n != name]
        self._save(state)


def _journal_last(names: Sequence[str]) -> List[str]:
    return sorted(names, key=lambda name: (name == JOURNAL_NAME, name))


def stage_names(store: StageStore, run_id: str, adapters: Sequence[str]) -> List[str]:
    """Out-relative names a finished stage owns: its attempt records, its adapters, the journal."""
    names = [f"{RUNS_DIR}/{attempt['attempt_id']}.json"
             for attempt in store.journal.attempts(run_id)]
    for directory in adapters:
        base = store.out / directory
        if base.is_dir():
            names += [path.relative_to(store.out).as_posix()
                      for path in sorted(base.rglob("*")) if path.is_file()]
    return names + [JOURNAL_NAME]


def run_stage(store: StageStore, run_id: str, argv: List[str], *,
              adapters: Sequence[str] = (), stop_on_failed: bool = False,
              **child: Any) -> Optional[Dict[str, Any]]:
    """Run one stage to completion, or skip it when the journal already completed it.

    ``argv`` is the child command (``drive_run`` prefixes the interpreter); ``child`` is
    passed on to ``drive_run`` (``stamp``, ``identity``, ``dry_run``). Returns the stage's last
    ok or failed record. Raises ``UploadPendingError`` before new work if earlier uploads are
    pending, and ``CalibrationStopped`` when the run is stopped by the driver.
    """
    store.restore()
    done = cal.completed(store.journal, run_id)
    if done:
        store.enqueue(stage_names(store, run_id, adapters))  # sends lost with the box
    store.raise_if_pending()
    if not done:
        try:
            cal.drive_run(store.journal, run_id, argv, stop_on_failed=stop_on_failed,
                          runs_dir=store.runs_dir, **child)
        finally:
            store.enqueue(stage_names(store, run_id, adapters))
            store.send_pending()  # failures stay pending; the journal is not touched
    return cal.find_record(store.out, run_id)


def hf_store(out: Path | str, repo_id: str, api: Optional[Any] = None) -> StageStore:
    """A ``StageStore`` backed by a private Hugging Face dataset repo (``transfer`` conventions)."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    api = api or HfApi()
    if not transfer.ensure_private_dataset(api, repo_id):
        raise PublicRepoError(f"{repo_id} exists and is not private; nothing is uploaded")

    def upload(local: Path, name: str) -> None:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=name, repo_id=repo_id,
                        repo_type="dataset")

    def download(name: str, directory: Path) -> bool:
        try:
            hf_hub_download(repo_id=repo_id, filename=name, repo_type="dataset",
                            local_dir=str(directory))
        except EntryNotFoundError:
            return False
        return True

    return StageStore(out, upload, download)
