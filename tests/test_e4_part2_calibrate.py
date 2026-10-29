"""E4 part 2: the one-command calibration (`calibrate`) and its dry run.

The dry run goes through the same driver, journal, records, decide and report as a paid run, on
a tiny random OLMoE on CPU with a fake box stamp, so the record schema is reviewed before any
paid compute. These tests run it end to end in fresh processes (about 30 s per full run), and
check the refusal and failure paths: wrong card, void, OOM, killed, second void.
"""

import json
import os
import subprocess
import sys

import pytest

import tests.e4_part2_support  # noqa: F401  (puts the harness on sys.path)

pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")

import e4_part2_calibration as cal  # noqa: E402
import e4_part2_common as common  # noqa: E402
import e4_part2_dryrun as dryrun  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(common.library_versions()["mismatches"]),
    reason="the E4 part 2 harness is pinned to torch/transformers/bitsandbytes/peft of the rule",
)

SCRIPT = str(common.HARNESS_DIR / "e4_part2_calibration.py")
RUN_IDS = ({f"F_{a}" for a in cal.ARMS} | {"T", "W", "P"}
           | {f"N_{i}" for i in range(1, cal.N_REPEATS + 1)})


def calibrate(out, *extra, instance="dry-1"):
    """`calibrate --dry-run` as a command line; returns (exit code, stderr)."""
    argv = [sys.executable, SCRIPT, "calibrate", "--dry-run", "--out", str(out),
            "--instance-id", instance, *extra]
    done = subprocess.run(argv, capture_output=True, text=True, check=False,
                          env={**os.environ, "PYTHONHASHSEED": "0"})
    return done.returncode, done.stderr


def records(out):
    return [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]


def by_run(out):
    found = {}
    for rec in records(out):
        found.setdefault(rec["run_id"], []).append(rec)
    return {k: sorted(v, key=lambda r: r["attempt_number"]) for k, v in found.items()}


# --- refusals: nothing is journaled -------------------------------------------------------


class TestRefusesTheWrongCard:
    @pytest.mark.parametrize("card,word", [("t4", "A100"), ("a100-odd", "45056")])
    def test_a_card_outside_both_classes_is_refused_before_any_attempt(self, tmp_path, card,
                                                                       word):
        code, err = calibrate(tmp_path / "out", "--fake-card", card)
        assert code == 6 and "refusing to run" in err and word in err
        assert not (tmp_path / "out" / "attempts.jsonl").exists()
        assert not (tmp_path / "out" / "runs").exists()
        verdict = json.loads(next((tmp_path / "out").glob("preflight-*.json")).read_text())
        assert verdict["card_class"] is None and verdict["dry_run"] is True

    def test_no_instance_id_is_refused(self, tmp_path, monkeypatch):
        for name in common.VAST_ID_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        argv = [sys.executable, SCRIPT, "calibrate", "--dry-run", "--out", str(tmp_path / "o")]
        env = {k: v for k, v in os.environ.items() if k not in common.VAST_ID_VARIABLES}
        done = subprocess.run(argv, capture_output=True, text=True, check=False, env=env)
        assert done.returncode == 6 and "no instance id" in done.stderr

    def test_the_80_gb_class_is_refused_until_the_40_gb_records_say_to_repeat(self, tmp_path):
        code, err = calibrate(tmp_path / "out", "--fake-card", "a100-80")
        assert code == 6 and "40 GB F records" in err
        assert not (tmp_path / "out" / "attempts.jsonl").exists()

    def test_the_class_bounds_are_the_rules(self):
        """The fake cards sit where the rule's classes say: 40,960 MiB is F, 81,920 is F80."""
        stamp = dryrun.fake_stamp("a100-40", "x")
        assert common.card_class_problems(stamp, "F") == []
        assert common.card_class_problems(stamp, "F80")
        stamp = dryrun.fake_stamp("a100-80", "x")
        assert common.card_class_problems(stamp, "F80") == []
        assert common.card_class_problems(stamp, "F")

    def test_dry_run_and_real_records_do_not_share_an_out_directory(self, tmp_path):
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "a.json").write_text(json.dumps({"dry_run": {"fake_card": "x"}}))
        assert "dry-run record" in common.dry_run_mix_problem(tmp_path, dry_run=False)
        assert common.dry_run_mix_problem(tmp_path, dry_run=True) is None
        with pytest.raises(common.CalibrationStopped):
            cal.guard_out(tmp_path, dry_run=False)


# --- the full path, with the bad endings a paid run can have -----------------------------

INJECT = "F_A_on=kill,T=void,W=oom"


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("cal") / "out"
    code, err = calibrate(out, "--inject", INJECT)
    return {"out": out, "code": code, "err": err, "runs": by_run(out)}


class TestFullDryRun:
    def test_it_completes_and_every_run_has_a_final_ok_record(self, full_run):
        assert full_run["code"] == 0, full_run["err"][-2000:]
        assert set(full_run["runs"]) == RUN_IDS
        assert all(v[-1]["outcome"] == "ok" for v in full_run["runs"].values())
        out = full_run["out"]
        decision = json.loads((out / "decision.json").read_text())
        assert decision["decision"] == "FINAL_A100_40GB" and decision["dry_run"] is True
        report = json.loads((out / "report.json").read_text())
        assert report["dry_run"] is True and report["provenance_gaps"] == []
        assert (out / "SHA256SUMS").exists()

    def test_every_record_carries_the_identity_the_review_asked_for(self, full_run):
        every = [r for v in full_run["runs"].values() for r in v]
        assert len(every) == len(RUN_IDS) + 3  # the three injected voids are kept
        for rec in every:
            ident = rec["identity"]
            assert not common.identity_problems(ident), (rec["run_id"], ident)
            assert ident["gpu_name"].startswith("NVIDIA A100")
            assert ident["compute_capability"] == "8.0"
            assert ident["memory_total_mib"] == "40960"
            assert ident["instance_id"] == "dry-1"
            assert ident["dataset_revision"] == dryrun.DRY_DATASET_REVISION
            assert ident["soup_commit"] and len(ident["soup_commit"]) == 40
            assert ident["torch"] == "2.14.1" and ident["bitsandbytes"] == "0.50.2"
            assert ident["transformers"] == "5.19.0"
            assert ident["torch_cuda"] and ident["driver_version"]
            assert rec["dry_run"]["fake_card"] == "a100-40"

    def test_an_attempt_that_the_process_did_not_survive_gets_a_record_from_the_driver(
            self, full_run):
        first, second = full_run["runs"]["F_A_on"]
        assert first["record_source"] == "driver" and first["outcome"] == "void"
        assert first["killed"]["signal"] == "SIGKILL" and first["failure_kind"] == "killed"
        assert first["box_stamp_before"] is None and first["box_stamp_after"]["gpu_uuid"]
        assert not common.identity_problems(first["identity"])
        assert second["record_source"] == "run" and second["outcome"] == "ok"

    def test_a_void_and_an_oom_crash_are_recorded_and_rerun_once_not_silently(self, full_run):
        t_first, t_second = full_run["runs"]["T"]
        assert t_first["outcome"] == "void" and "injected void" in t_first["void_reasons"][0]
        w_first, w_second = full_run["runs"]["W"]
        assert w_first["failure_kind"] == "oom" and "OutOfMemoryError" in w_first["crash"]
        assert w_first["outcome"] == "void" and w_second["outcome"] == "ok"
        assert [r["attempt_number"] for r in (t_first, t_second)] == [1, 2]
        assert "T: void" in full_run["err"] and "W: void" in full_run["err"]
        assert "F_A_on: attempt" in full_run["err"] and "journaled as a void" in full_run["err"]
        report = json.loads((full_run["out"] / "report.json").read_text())
        assert {v["run_id"] for v in report["voids"]} == {"F_A_on", "T", "W"}

    def test_the_journal_has_a_start_and_an_end_for_every_attempt(self, full_run):
        events = common.read_jsonl(full_run["out"] / "attempts.jsonl")
        starts = [e["attempt_id"] for e in events if e["event"] == "start"]
        ends = [e["attempt_id"] for e in events if e["event"] == "end"]
        assert sorted(starts) == sorted(ends) and len(starts) == len(RUN_IDS) + 3
        on_disk = {r["attempt_id"] for v in full_run["runs"].values() for r in v}
        assert on_disk == set(starts)

    def test_a_record_of_a_run_that_finished_has_the_stage_result(self, full_run):
        runs = full_run["runs"]
        assert runs["F_A_off"][-1]["result"]["trainable"]["matches"]
        assert runs["T"][-1]["result"]["tau"] and runs["W"][-1]["result"]["tol_W"]
        assert set(runs["N_1"][-1]["result"]["scores"]) == set(
            runs["N_5"][-1]["result"]["scores"])
        assert runs["P"][-1]["result"]["comparison"]["adapters_bit_identical"] in (True, False)
        assert runs["F_A_on"][-1]["result"]["dry_run_fake_fields"] == dryrun.DRY_FAKE_FIELDS

    def test_running_it_again_does_not_redo_completed_runs(self, full_run):
        before = len(common.read_jsonl(full_run["out"] / "attempts.jsonl"))
        code, _ = calibrate(full_run["out"], "--inject", INJECT)
        assert code == 0
        assert len(common.read_jsonl(full_run["out"] / "attempts.jsonl")) == before

    def test_the_fake_stamp_has_the_keys_of_a_real_one(self):
        sensors = {name: (lambda: None) for name in (
            "ram_available_bytes", "commit_charge", "gpu_processes", "gpu_clock",
            "gpu_memory_used_mib")}
        real = common.box_stamp(0, "x", readers=sensors)
        assert set(dryrun.fake_stamp("a100-40", "x")) == set(real)


class TestTheRepeatOn80Gb:
    def test_an_oom_on_40_gb_sends_the_run_to_an_80_gb_instance_and_it_finishes_there(
            self, tmp_path):
        out = tmp_path / "out"
        code, _ = calibrate(out, "--inject", "F_A_off=oom", instance="dry-40")
        assert code == 4
        decision = json.loads((out / "decision.json").read_text())
        assert decision["decision"] == "REPEAT_F_ON_A100_80GB"
        assert by_run(out)["F_A_off"][-1]["result"]["oom"] is True
        assert not (out / "report.json").exists()
        code, err = calibrate(out, "--fake-card", "a100-80", instance="dry-80")
        assert code == 0, err[-2000:]
        decision = json.loads((out / "decision.json").read_text())
        assert decision["decision"] == "FINAL_A100_80GB"
        assert decision["final_instance"]["instance_id"] == "dry-80"
        runs = by_run(out)
        assert {"F80_A_off", "F80_A_on", "F80_A_bf16", "T", "P"} <= set(runs)
        assert runs["T"][-1]["identity"]["memory_total_mib"] == "81920"
        assert runs["T"][-1]["identity"]["instance_id"] == "dry-80"

    def test_the_40_gb_instance_cannot_continue_once_the_final_card_is_the_80(self, tmp_path):
        out = tmp_path / "out"
        calibrate(out, "--inject", "F_A_off=oom", instance="dry-40")
        calibrate(out, "--fake-card", "a100-80", "--inject", "T=fail", instance="dry-80")
        code, err = calibrate(out, instance="dry-40")
        assert code == 6 and "not this one" in err


class TestAKilledRunTwiceStopsCalibration:
    def test_a_second_void_stops_with_both_attempts_recorded(self, tmp_path):
        out = tmp_path / "out"
        code, err = calibrate(out, "--inject", "F_A_bf16=kill@all")
        assert code == 6 and "second void" in err
        attempts = by_run(out)["F_A_bf16"]
        assert [a["record_source"] for a in attempts] == ["driver", "driver"]
        assert not (out / "decision.json").exists()

    def test_a_failed_stage_stops_and_is_not_rerun(self, tmp_path):
        out = tmp_path / "out"
        code, err = calibrate(out, "--inject", "W=fail")
        assert code == 6 and "calibration fails" in err
        assert len(by_run(out)["W"]) == 1 and by_run(out)["W"][0]["outcome"] == "failed"
        assert "N_1" not in by_run(out)


# --- pieces, in process -------------------------------------------------------------------


class TestExecuteAttemptIdentityAndCard:
    STAMP = dryrun.fake_stamp("a100-40", "i-1")

    def run(self, tmp_path, stamp, **kw):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        prov = {"soup_commit": "a" * 40, "harness_commit": "b" * 40, "harness_tree": "c" * 40,
                "harness_dirty": False, "dirty_paths": []}
        ran = []
        record = cal.execute_attempt(
            journal, "F_A_on", "F", tmp_path / "runs", lambda: ran.append(1) or {},
            stamp=lambda _g, _i: dict(stamp), prov=lambda: prov,
            identify_own=dryrun.fake_identify_own, **kw)
        return record, ran

    def test_a_missing_identity_field_is_a_void_that_keeps_its_record(self, tmp_path):
        stamp = dict(self.STAMP, gpu_memory_total_mib=None)
        record, ran = self.run(tmp_path, stamp, dataset_revision="rev", require_identity=True)
        assert record["outcome"] == "void" and not ran
        assert any("identity.memory_total_mib" in r for r in record["void_reasons"])
        assert (tmp_path / "runs" / f"{record['attempt_id']}.json").exists()

    def test_no_dataset_revision_is_a_void_when_the_identity_is_required(self, tmp_path):
        record, _ = self.run(tmp_path, self.STAMP, require_identity=True)
        assert any("identity.dataset_revision" in r for r in record["void_reasons"])

    def test_a_card_outside_the_class_is_a_card_void_before_the_body_runs(self, tmp_path):
        record, ran = self.run(tmp_path, dryrun.fake_stamp("a100-80", "i-1"), card_prefix="F",
                               dataset_revision="rev", require_identity=True)
        assert record["outcome"] == "void" and not ran
        assert all(r.startswith("card:") for r in record["void_reasons"])

    def test_the_card_check_also_reads_the_stamp_after_the_run(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        stamps = iter([self.STAMP, dryrun.fake_stamp("t4", "i-1")])
        record = cal.execute_attempt(
            journal, "F_A_on", "F", tmp_path / "runs", lambda: {},
            stamp=lambda _g, _i: next(stamps), identify_own=dryrun.fake_identify_own,
            prov=lambda: {"soup_commit": "a", "harness_commit": "b", "harness_tree": "c",
                          "harness_dirty": False, "dirty_paths": []},
            card_prefix="F", dataset_revision="rev", require_identity=True)
        assert any(r.startswith("card:") and "after the run" in r
                   for r in record["void_reasons"])

    def test_a_card_void_is_never_rerun(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        a = journal.start("F_A_on", "F")
        journal.end(a, common.OUTCOME_VOID, ["card: gpu_name=Tesla T4 does not name an A100"])
        original = cal.drive_child
        cal.drive_child = lambda *args, **kw: 0
        try:
            with pytest.raises(common.CalibrationStopped, match="card is not the rule's"):
                cal.drive_run(journal, "F_A_on", [])
        finally:
            cal.drive_child = original

    def test_a_crash_record_says_whether_it_was_an_oom(self, tmp_path):
        import torch

        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")

        def boom():
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(torch.cuda.OutOfMemoryError):
            cal.execute_attempt(journal, "T", "T", tmp_path / "runs", boom,
                                stamp=lambda _g, _i: dict(self.STAMP),
                                identify_own=dryrun.fake_identify_own,
                                prov=lambda: {"soup_commit": "a", "harness_commit": "b",
                                              "harness_tree": "c", "harness_dirty": False,
                                              "dirty_paths": []})
        (record,) = records(tmp_path)
        assert record["failure_kind"] == "oom" and record["outcome"] == "void"
        assert record["void_reasons"] and record["identity"]["gpu_uuid"]


class TestDriverRecord:
    def test_a_child_killed_by_a_signal_gets_a_driver_record_with_the_signal(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        code = (f"import sys; sys.path.insert(0, {str(common.HARNESS_DIR)!r});"
                "import e4_part2_common as c, os, signal;"
                f"c.AttemptsJournal({str(tmp_path / 'attempts.jsonl')!r}).start('W', 'W');"
                "os.kill(os.getpid(), signal.SIGKILL)")
        stamp = dryrun.fake_stamp("a100-40", "i-1")
        cal.drive_child(journal, "W", ["-c", code], runs_dir=tmp_path / "runs",
                        stamp=lambda: stamp,
                        identity=lambda now: common.run_identity(
                            now, common.library_versions(), {"soup_commit": "s"}, "rev"))
        (record,) = records(tmp_path)
        assert record["record_source"] == "driver" and record["kind"] == "W"
        assert record["killed"] == {
            "returncode": -9, "signal": "SIGKILL",
            "note": "the child process ended without closing its attempt"}
        assert record["identity"]["dataset_revision"] == "rev"
        assert journal.attempts("W")[0]["outcome"] == "void"

    def test_a_child_that_wrote_its_own_record_is_not_overwritten(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        attempt = journal.start("W", "W")
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / f"{attempt}.json").write_text('{"mine": true}')
        original = cal.subprocess.run
        cal.subprocess.run = lambda *a, **k: type("R", (), {"returncode": 1})()
        try:
            cal.drive_child(journal, "W", [], runs_dir=tmp_path / "runs")
        finally:
            cal.subprocess.run = original
        assert json.loads((tmp_path / "runs" / f"{attempt}.json").read_text()) == {"mine": True}


class TestInjectSpec:
    def test_parses_kinds_and_the_all_suffix(self):
        assert dryrun.parse_inject("F_A_on=oom,N_2=kill@all") == {
            "F_A_on": ("oom", False), "N_2": ("kill", True)}
        assert dryrun.parse_inject(None) == {}

    @pytest.mark.parametrize("bad", ["T", "T=boom", "=oom", "T=oom@sometimes"])
    def test_rejects_what_it_does_not_know(self, bad):
        with pytest.raises(ValueError):
            dryrun.parse_inject(bad)

    def test_a_kind_hits_the_first_attempt_only_unless_all(self):
        spec = dryrun.parse_inject("T=void,W=void@all")
        assert [dryrun.injected_kind(spec, "T", n) for n in (1, 2)] == ["void", None]
        assert [dryrun.injected_kind(spec, "W", n) for n in (1, 2)] == ["void", "void"]


class TestTheRealPathRequiresTheIdentity:
    """The default of `require_identity` is off; the non-dry `run` (what `calibrate` drives)
    turns it on, so a record missing an identity field is a void."""

    def _run(self, tmp_path, monkeypatch, stamp):
        import types

        import transformers

        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        (tmp_path / "split").mkdir()
        (tmp_path / "split" / "split-record.json").write_text('{"dataset_revision": "rev1"}')
        seen = {}
        real = cal.execute_attempt

        def spy(*args, **kw):
            seen.update(kw)
            kw.update(stamp=lambda _g, _i: dict(stamp), identify_own=dryrun.fake_identify_own,
                      input_check=lambda: [], prov=lambda: {
                          "soup_commit": "a" * 40, "harness_commit": "b" * 40,
                          "harness_tree": "c" * 40, "harness_dirty": False, "dirty_paths": []})
            return real(*args, **kw)

        monkeypatch.setattr(cal, "execute_attempt", spy)
        monkeypatch.setattr(common, "require_single_gpu", lambda _i: None)
        tests.e4_part2_support.stub_pynvml(monkeypatch)
        monkeypatch.setattr(cal, "resolve_device", lambda _d, _i: "cuda:0")
        import torch

        monkeypatch.setattr(torch.cuda, "set_device", lambda _i: None)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            lambda *_a, **_k: object())
        monkeypatch.setattr(cal, "_body", lambda *_a, **_k: {})
        args = types.SimpleNamespace(
            out=tmp_path / "out", stage="F", arm="A_on", f_prefix="F", repeat=1, run_id=None,
            data_root=tmp_path, split_dir=tmp_path / "split", manifest=manifest,
            device="cuda", gpu_index=0, instance_id="i-1", recipe_seed=42)
        code = cal._cmd_run(args)
        return code, seen, records(tmp_path / "out")

    def test_the_non_dry_run_passes_require_identity_and_the_dataset_revision(
            self, tmp_path, monkeypatch):
        code, seen, recs = self._run(tmp_path, monkeypatch, dryrun.fake_stamp("a100-40", "i-1"))
        assert seen["require_identity"] is True and seen["dataset_revision"] == "rev1"
        assert seen["card_prefix"] == "F"
        assert code == 0 and recs[0]["outcome"] == "ok"  # card and identity both pass

    def test_a_missing_identity_field_voids_the_record_on_that_path(self, tmp_path, monkeypatch):
        stamp = dict(dryrun.fake_stamp("a100-40", "i-1"), gpu_memory_total_mib=None)
        code, _, recs = self._run(tmp_path, monkeypatch, stamp)
        assert code == 3 and recs[0]["outcome"] == "void"
        assert any("identity.memory_total_mib" in r for r in recs[0]["void_reasons"])

    def test_calibrate_drives_children_without_the_dry_run_flag(self, tmp_path):
        import types

        args = types.SimpleNamespace(
            out=tmp_path, f_prefix="F", gpu_index=0, instance_id="i-1", data_root=tmp_path,
            split_dir=tmp_path, manifest=tmp_path / "m", device="cuda", dry_run=False)
        argv = cal._child_argv(args, "F", "A_on")
        assert "--dry-run" not in argv and "--manifest" in argv and "--data-root" in argv
