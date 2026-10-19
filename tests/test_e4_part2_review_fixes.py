"""E4 part 2: tests for the code-review fixes (written before the fixes).

1 final-card completeness, 2 non-finite values, 3 repo-wide provenance, 4 push file set,
5 hashing once per instance, 6 non-contributing units, 7 train mode, 8 peers in a container,
9 nits.
"""
# ruff: noqa: I001 (the support import must run before the harness imports)

import json
import math
import os
import subprocess
import types

import pytest

from tests.e4_part2_support import tiny_model, write_legacy_checkpoint

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")

import e4_part2_calibration as cal  # noqa: E402
import e4_part2_common as common  # noqa: E402
import e4_part2_quant as quant  # noqa: E402
import e4_part2_transfer as tr  # noqa: E402
import e4_part2_wref as wref  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(common.library_versions()["mismatches"]),
    reason="the E4 part 2 harness is pinned to the rule's library versions",
)

E = 4
GOOD_STAMP = {"unix_s": 1.0, "gpu_uuid": "GPU-1", "instance_id": "123", "gpu_processes": [],
              "unknown": {}}
GOOD_PROV = {"soup_commit": "a" * 40, "harness_commit": "b" * 40, "harness_tree": "c" * 40,
             "harness_dirty": False}


def stamps(before, after):
    it = iter([before, after])
    return lambda *_: next(it)


def own_stub(pids=None):
    return lambda: {"own_pids": pids if pids is not None else [os.getpid()], "method": "stub"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w")
    model = tiny_model(seed=6, dtype=torch.bfloat16)
    source = write_legacy_checkpoint(model, root / "source")
    quant.run(source, root / "data", root / "m.json", None, 2, E, log=lambda *_: None)
    return {"source": source, "data": root / "data"}


# --- 1 ------------------------------------------------------------------------------------


class TestFinalCardNeedsAllArms:
    def ok(self, reserved=20e9):
        return {"oom": False, "failed": False, "max_memory_reserved_bytes": reserved}

    @pytest.mark.parametrize("have", [[], ["A_off"], ["A_off", "A_on"], ["A_on", "A_bf16"]])
    def test_a_missing_arm_is_incomplete_never_a_card(self, have):
        res = cal.decide_final_card({a: self.ok() for a in have})
        assert res["decision"] == "INCOMPLETE" and res["final_card"] is None
        assert res["missing"] == [a for a in cal.ARMS if a not in have]

    def test_incomplete_80_gb_repeat_is_incomplete(self):
        f40 = {"A_off": self.ok(40e9), "A_on": self.ok(), "A_bf16": self.ok()}
        f80 = {"A_off": self.ok(45e9)}
        res = cal.decide_final_card(f40, f80)
        assert res["decision"] == "INCOMPLETE" and res["final_card"] is None

    def test_all_three_arms_still_decide(self):
        assert cal.decide_final_card({a: self.ok() for a in cal.ARMS})["decision"] == (
            "FINAL_A100_40GB")

    def test_cmd_decide_ignores_void_records_and_reports_incomplete(self, tmp_path):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        for arm in ("A_off", "A_on"):
            cal.execute_attempt(journal, f"F_{arm}", "F", tmp_path / "runs",
                                lambda: {"oom": False, "max_memory_reserved_bytes": 1},
                                stamp=stamps(GOOD_STAMP, GOOD_STAMP), prov=lambda: GOOD_PROV,
                                identify_own=own_stub())
        cal.execute_attempt(journal, "F_A_bf16", "F", tmp_path / "runs", lambda: {},
                            stamp=stamps(GOOD_STAMP, GOOD_STAMP),
                            prov=lambda: dict(GOOD_PROV, harness_dirty=True),
                            identify_own=own_stub())  # void
        assert cal.main(["decide", "--out", str(tmp_path)]) == 4
        assert json.loads((tmp_path / "decision.json").read_text())["decision"] == "INCOMPLETE"


# --- 2 ------------------------------------------------------------------------------------


class TestNonFinite:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("position", ["e_F", "e_R"])
    def test_compute_tau_raises_on_any_non_finite_error_in_either_order(self, bad, position):
        rec = {"cell": "c", "quantity": "output", "e_F": 0.1, "e_R": 0.2}
        rec[position] = bad
        for ordering in ([rec, {"cell": "c", "quantity": "output", "e_F": 1.0, "e_R": 1.0}],
                         [{"cell": "c", "quantity": "output", "e_F": 1.0, "e_R": 1.0}, rec]):
            with pytest.raises(ValueError, match="not finite"):
                cal.compute_tau(ordering)

    def test_tol_w_raises_on_a_non_finite_call(self):
        calls = [{q: 0.1 for q in wref.W_QUANTITIES},
                 dict({q: 0.1 for q in wref.W_QUANTITIES}, output=float("nan"))]
        for ordering in (calls, calls[::-1]):
            with pytest.raises(ValueError, match="not finite"):
                wref.tol_w(ordering)

    def test_finite_max_helper_never_drops_a_nan(self):
        assert cal.finite_max([0.1, 0.4, 0.2]) == 0.4
        for values in ([float("nan"), 1.0], [1.0, float("nan")], [1.0, float("inf")]):
            with pytest.raises(ValueError):
                cal.finite_max(values)

    def test_void_reasons_name_every_non_finite_error_and_reference(self):
        recs = [{"unit": "u1", "dq": "off", "quantity": "output", "e_F": float("nan"),
                 "e_R": 0.1, "ref_finite": True},
                {"unit": "u2", "dq": "on", "quantity": "input_grad", "e_F": 0.1,
                 "e_R": float("inf"), "ref_finite": True},
                {"unit": "u3", "dq": "on", "quantity": "output", "e_F": 0.1, "e_R": 0.1,
                 "ref_finite": False}]
        reasons = cal.t_nonfinite_reasons(recs)
        assert len(reasons) == 3
        assert any("u1" in r and "e_F" in r for r in reasons)
        assert any("u2" in r and "e_R" in r for r in reasons)
        assert any("u3" in r and "reference" in r for r in reasons)
        assert cal.t_nonfinite_reasons([dict(recs[0], e_F=0.0)]) == []

    def _run_t(self, world, **kw):
        bf16 = cal.load_base(world["source"], "cpu", "eager")
        rows = [list(range(3, 12)) * 3 + [i % 100 + 3] for i in range(40)]
        common.set_fp32_reference_mode()
        return cal.run_t(quant.enumerate_units(2, E), cal.UnitFiles(world["data"]), bf16, rows,
                         cal.Recipe(), "cpu", None, per_kind=2, calls_per_unit=2, **kw)

    def test_run_t_is_void_and_has_no_tau_when_an_error_is_nan(self, world, monkeypatch):
        real = cal.max_abs
        count = {"n": 0}

        def poisoned(a, b):
            count["n"] += 1
            return float("nan") if count["n"] == 7 else real(a, b)

        monkeypatch.setattr(cal, "max_abs", poisoned)
        out = self._run_t(world, min_units=1)
        assert out["void_reasons"] and all("not finite" in r for r in out["void_reasons"])
        assert out["tau"] == {}

    def test_run_t_is_void_when_a_reference_is_non_finite(self, world, monkeypatch):
        real = cal.reference_linear32

        def bad_reference(x, w):
            ref, flags = real(x, w)
            ref["output"] = ref["output"] * float("nan")
            return ref, flags

        monkeypatch.setattr(cal, "reference_linear32", bad_reference)
        out = self._run_t(world, min_units=1)
        assert any("reference" in r for r in out["void_reasons"])
        assert out["tau"] == {}

    def test_run_w_is_void_with_no_tolerances_on_a_non_finite_difference(self, world,
                                                                          monkeypatch):
        real = wref.compare_references
        monkeypatch.setattr(wref, "compare_references",
                            lambda a, b: dict(real(a, b), A_grad=float("inf")))
        out = cal.run_w({"A_off": world["data"] / "writeback-dqoff"},
                        [list(range(3, 12)) * 2] * 3, cal.Recipe(), "cpu", layers=(0,),
                        rows=3, experts_implementation="eager")
        assert out["void_reasons"] and out["tol_W"] == {}


# --- 3 ------------------------------------------------------------------------------------


def git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=env).stdout


@pytest.fixture
def mini_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "benchmarks" / "harness").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "benchmarks" / "harness" / "h.py").write_text("x = 1\n")
    (repo / "tests" / "t.py").write_text("y = 1\n")
    (repo / "e4-part2-calibration-rule.md").write_text("rule\n")
    (repo / "e4-rule-part2-draft.md").write_text("main\n")
    (repo / ".gitignore").write_text("scratch/\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


class TestProvenanceCoversTheWholeTree:
    def test_clean_tree_is_clean_with_all_three_ids(self, mini_repo):
        rec = common.provenance(mini_repo)
        assert rec["harness_dirty"] is False and rec["dirty_paths"] == []
        assert len(rec["soup_commit"]) == 40 and len(rec["harness_tree"]) == 40

    @pytest.mark.parametrize("path", [
        "tests/new_test.py", "e4-rule-part2-draft.md", "e4-part2-calibration-rule.md",
        "benchmarks/results/e4-part2-cal/split/smoke.manifest.jsonl", "README.md",
    ])
    def test_any_untracked_or_modified_file_makes_it_dirty(self, mini_repo, path):
        target = mini_repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a") as handle:
            handle.write("change\n")
        rec = common.provenance(mini_repo)
        assert rec["harness_dirty"] is True
        assert any(path in p for p in rec["dirty_paths"])
        reasons = common.validate_provenance(rec)
        assert reasons and path in reasons[0]

    def test_a_gitignored_path_does_not_make_it_dirty(self, mini_repo):
        (mini_repo / "scratch").mkdir()
        (mini_repo / "scratch" / "big.bin").write_bytes(b"x")
        assert common.provenance(mini_repo)["harness_dirty"] is False

    def test_a_staged_but_uncommitted_file_is_dirty(self, mini_repo):
        (mini_repo / "tests" / "staged.py").write_text("z\n")
        git(mini_repo, "add", "tests/staged.py")
        assert common.provenance(mini_repo)["harness_dirty"] is True


# --- 4 ------------------------------------------------------------------------------------


class PushApi:
    def __init__(self, listing=None):
        self.calls, self.uploads, self.listing = [], [], listing

    def whoami(self):
        return {"name": "me"}

    def create_repo(self, repo_id, **kw):
        self.calls.append("create_repo")

    def repo_info(self, repo_id, repo_type):
        return types.SimpleNamespace(private=True)

    def upload_file(self, **kw):
        self.calls.append("upload_file")
        self.uploads.append(kw["path_in_repo"])

    def upload_folder(self, **kw):
        self.calls.append("upload_folder")
        self.uploads.append(kw)

    def list_repo_files(self, repo_id, repo_type):
        return self.listing(self) if callable(self.listing) else self.listing


def make_out(tmp_path):
    out = tmp_path / "out"
    files = {}
    for name, data in (("codes-dqoff/a.bin", b"a"), ("codes-dqon/b.bin", b"bb"),
                       ("writeback-dqoff/c.json", b"{}")):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[name] = {"sha256": common.sha256_bytes(data), "bytes": len(data)}
    (out / ".cache" / "huggingface").mkdir(parents=True)
    (out / ".cache" / "huggingface" / "x.metadata").write_text("m")
    (out / "stray.txt").write_text("not in the manifest")
    (out / "SHA256SUMS").write_text("sums")
    manifest = tmp_path / "quant-manifest.json"
    manifest.write_text(json.dumps({"files": files, "source_files": {}}))
    return out, manifest, files


class TestPushFileSet:
    def test_uploads_only_manifest_files_never_cache_or_strays(self, tmp_path):
        out, manifest, files = make_out(tmp_path)
        api = PushApi(listing=lambda a: sorted(list(files) + [tr.MANIFEST_NAME, ".gitattributes"]))
        assert tr.push(out, manifest, "r", False, api, log=lambda *_: None) == 0
        sent = [u for u in api.uploads if isinstance(u, dict)]
        assert sent, "folders go through upload_folder"
        uploaded = set()
        for kw in sent:
            assert kw["repo_type"] == "dataset"
            patterns = kw["allow_patterns"]
            assert ".cache" not in " ".join(patterns)
            uploaded |= {f"{kw['path_in_repo']}/{p}" for p in patterns}
        assert uploaded == set(files)

    def test_fails_when_the_repo_listing_has_an_extra_file(self, tmp_path):
        out, manifest, files = make_out(tmp_path)
        api = PushApi(listing=lambda a: sorted(list(files) + [tr.MANIFEST_NAME, "stray.txt"]))
        logged = []
        assert tr.push(out, manifest, "r", False, api, log=logged.append) == 1
        assert any("stray.txt" in str(line) for line in logged)

    def test_fails_when_a_manifest_file_is_missing_from_the_repo(self, tmp_path):
        out, manifest, files = make_out(tmp_path)
        api = PushApi(listing=lambda a: [tr.MANIFEST_NAME] + list(files)[:-1])
        assert tr.push(out, manifest, "r", False, api, log=lambda *_: None) == 1

    def test_the_manifest_is_uploaded_and_the_dry_run_uploads_nothing(self, tmp_path):
        out, manifest, files = make_out(tmp_path)
        api = PushApi()
        assert tr.push(out, manifest, "r", True, api, log=lambda *_: None) == 0
        assert api.calls == []
        api = PushApi(listing=lambda a: sorted(list(files) + [tr.MANIFEST_NAME]))
        tr.push(out, manifest, "r", False, api, log=lambda *_: None)
        assert tr.MANIFEST_NAME in api.uploads

    def test_the_real_hub_client_has_the_calls_push_makes(self):
        import huggingface_hub

        for name in ("create_repo", "repo_info", "upload_file", "upload_folder",
                     "list_repo_files", "whoami"):
            assert hasattr(huggingface_hub.HfApi, name), name


# --- 5 ------------------------------------------------------------------------------------


def synthetic_instance(tmp_path):
    root = tmp_path / "data"
    names = {
        "writeback-dqoff/model-layer-00.safetensors": b"w0",
        "writeback-dqoff/model-other.safetensors": b"w1",
        "writeback-dqoff/model.safetensors.index.json": b"i0", "writeback-dqoff/config.json": b"c0",
        "writeback-dqoff/generation_config.json": b"g0", "writeback-dqoff/tokenizer.json": b"t0",
        "writeback-dqon/model-layer-00.safetensors": b"v0",
        "writeback-dqon/model.safetensors.index.json": b"j0", "writeback-dqon/config.json": b"d0",
        "writeback-dqon/generation_config.json": b"h0",
        "codes-dqoff/layer-00.safetensors": b"k0", "codes-dqon/layer-00.safetensors": b"k1",
    }
    source = {
        "model-00001-of-00003.safetensors": b"s1", "model.safetensors.index.json": b"si",
        "config.json": b"sc", "generation_config.json": b"sg", "tokenizer.json": b"st",
        "tokenizer_config.json": b"stc", "special_tokens_map.json": b"ssm",
    }
    files, src = {}, {}
    for name, data in names.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
        files[name] = {"sha256": common.sha256_bytes(data), "bytes": len(data)}
    for name, data in source.items():
        (root / "source").mkdir(exist_ok=True)
        (root / "source" / name).write_bytes(data)
        src[name] = {"sha256": common.sha256_bytes(data), "bytes": len(data)}
    manifest = {"files": files, "source_files": src,
                "model": {"repo": common.MODEL_REPO, "revision": common.MODEL_REV}}
    (root / tr.MANIFEST_NAME).write_text(json.dumps(manifest))
    return root, manifest


def fake_download(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: None)


class TestHashOncePerInstance:
    def test_pull_verifies_everything_once_and_records_it(self, tmp_path, monkeypatch):
        root, manifest = synthetic_instance(tmp_path)
        fake_download(monkeypatch)
        api = types.SimpleNamespace(whoami=lambda: {"name": "me"})
        assert tr.pull(root, "r", api, log=lambda *_: None) == 0
        rec = json.loads((root / tr.VERIFIED_NAME).read_text())
        assert rec["manifest_sha256"] == common.sha256_file(root / tr.MANIFEST_NAME)
        assert len(rec["files"]) == len(manifest["files"]) + len(manifest["source_files"])
        assert rec["files"]["source/config.json"]["sha256"] == common.sha256_bytes(b"sc")

    def test_pull_records_nothing_when_a_file_differs(self, tmp_path, monkeypatch):
        root, _ = synthetic_instance(tmp_path)
        (root / "codes-dqon" / "layer-00.safetensors").write_bytes(b"XX")
        fake_download(monkeypatch)
        api = types.SimpleNamespace(whoami=lambda: {"name": "me"})
        assert tr.pull(root, "r", api, log=lambda *_: None) == 1
        assert not (root / tr.VERIFIED_NAME).exists()

    def test_a_run_loads_only_the_files_its_stage_reads(self, tmp_path):
        root, manifest = synthetic_instance(tmp_path)
        a_off = {n for _, n in cal.loaded_files("F", "A_off", manifest)}
        assert "writeback-dqoff/model-layer-00.safetensors" in a_off
        assert "writeback-dqoff/config.json" in a_off
        assert "source/tokenizer.json" in a_off
        assert not any(n.startswith(("writeback-dqon", "codes-")) for n in a_off)
        assert not any(n.startswith("source/model") for n in a_off)
        bf16 = {n for _, n in cal.loaded_files("F", "A_bf16", manifest)}
        assert "source/model-00001-of-00003.safetensors" in bf16
        assert not any(n.startswith("writeback") for n in bf16)
        t = {n for _, n in cal.loaded_files("T", None, manifest)}
        assert "codes-dqon/layer-00.safetensors" in t and "writeback-dqoff/config.json" in t
        assert "source/model-00001-of-00003.safetensors" in t
        w = {n for _, n in cal.loaded_files("W", None, manifest)}
        assert not any(n.startswith("codes-") for n in w)

    def test_run_check_needs_the_instance_record_and_hashes_only_loaded_files(self, tmp_path,
                                                                              monkeypatch):
        root, manifest = synthetic_instance(tmp_path)
        assert any("verified" in r for r in cal.run_input_check(root, manifest, "F", "A_off"))
        fake_download(monkeypatch)
        api = types.SimpleNamespace(whoami=lambda: {"name": "me"})
        tr.pull(root, "r", api, log=lambda *_: None)
        assert cal.run_input_check(root, manifest, "F", "A_off") == []
        hashed = []
        real = common.sha256_file
        monkeypatch.setattr(common, "sha256_file", lambda p, *a: hashed.append(str(p)) or real(p))
        cal.run_input_check(root, manifest, "F", "A_off")
        assert hashed and not any("writeback-dqon" in h or "codes-" in h for h in hashed)
        # a file the run does not load changes: not noticed here (pull verified it once)
        (root / "codes-dqon" / "layer-00.safetensors").write_bytes(b"zz")
        assert cal.run_input_check(root, manifest, "F", "A_off") == []
        # a file it does load changes: void
        (root / "writeback-dqoff" / "config.json").write_bytes(b"zz")
        reasons = cal.run_input_check(root, manifest, "F", "A_off")
        assert any("writeback-dqoff/config.json" in r for r in reasons)

    def test_run_check_rejects_a_record_for_another_manifest(self, tmp_path, monkeypatch):
        root, manifest = synthetic_instance(tmp_path)
        fake_download(monkeypatch)
        tr.pull(root, "r", types.SimpleNamespace(whoami=lambda: {"name": "me"}),
                log=lambda *_: None)
        rec = json.loads((root / tr.VERIFIED_NAME).read_text())
        rec["manifest_sha256"] = "0" * 64
        (root / tr.VERIFIED_NAME).write_text(json.dumps(rec))
        assert any("another manifest" in r
                   for r in cal.run_input_check(root, manifest, "F", "A_off"))


# --- 6 ------------------------------------------------------------------------------------


class TestNonContributingUnits:
    def units(self):
        return {"q_proj": [quant.Unit(f"q{i}", i, "q_proj", None) for i in range(8)],
                "gate_proj": [quant.Unit(f"g{i}", i, "gate_proj", i) for i in range(8)]}

    def test_lists_units_without_calls_counts_per_kind_and_flags_kinds_below_four(self):
        chosen = self.units()
        inputs = {u.key: [1] for u in chosen["q_proj"]}
        inputs.update({u.key: ([1] if i < 3 else []) for i, u in enumerate(chosen["gate_proj"])})
        out = cal.summarize_contribution(chosen, inputs, min_units=4)
        assert out["contributing_per_kind"] == {"q_proj": 8, "gate_proj": 3}
        assert out["non_contributing"] == [f"g{i}" for i in range(3, 8)]
        assert out["kinds_without_tau"] == ["gate_proj"]

    def test_exactly_four_contributing_units_keep_the_kind(self):
        chosen = self.units()
        inputs = {u.key: ([1] if i < 4 else []) for i, u in enumerate(chosen["gate_proj"])}
        inputs.update({u.key: [1] for u in chosen["q_proj"]})
        out = cal.summarize_contribution(chosen, inputs, min_units=4)
        assert out["kinds_without_tau"] == []

    def test_a_unit_with_one_call_below_calls_per_unit_still_contributes(self):
        chosen = self.units()
        inputs = {u.key: [1] for u in chosen["q_proj"] + chosen["gate_proj"]}
        assert cal.summarize_contribution(chosen, inputs, 4)["non_contributing"] == []

    def test_tau_skips_the_kinds_without_enough_units(self):
        recs = [{"kind": "q_proj", "cell": "q_proj/backward", "quantity": "input_grad",
                 "e_F": 0.1, "e_R": 0.2},
                {"kind": "gate_proj", "cell": "gate_proj/backward", "quantity": "input_grad",
                 "e_F": 0.1, "e_R": 0.2}]
        tau = cal.compute_tau(recs, exclude_kinds=["gate_proj"])
        assert list(tau) == ["q_proj/backward"]

    def run_t(self, world, **kw):
        bf16 = cal.load_base(world["source"], "cpu", "eager")
        rows = [list(range(3, 12)) * 3 + [i % 100 + 3] for i in range(40)]
        common.set_fp32_reference_mode()
        return cal.run_t(quant.enumerate_units(2, E), cal.UnitFiles(world["data"]), bf16, rows,
                         cal.Recipe(), "cpu", None, per_kind=2, calls_per_unit=2, **kw)

    def test_with_the_rule_threshold_of_four_a_two_unit_kind_gets_no_tau(self, world):
        out = self.run_t(world)  # min_units defaults to the rule's 4
        assert set(out["kinds_without_tau"]) == set(cal.KINDS)
        assert out["tau"] == {}
        assert out["contributing_per_kind"] == {k: 2 for k in cal.KINDS} or all(
            v <= 2 for v in out["contributing_per_kind"].values())
        assert "non_contributing" in out and "records" in out

    def test_with_a_lower_threshold_taus_come_back(self, world):
        out = self.run_t(world, min_units=1)
        assert out["tau"] and out["kinds_without_tau"] == []


# --- 7 ------------------------------------------------------------------------------------

SMALL = cal.Recipe(accum=2, f_updates=3, warmup_updates=2, total_updates=8)


def examples(n=12):
    gen = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        prompt = torch.randint(3, 128, (6,), generator=gen).tolist()
        resp = torch.randint(3, 128, (4,), generator=gen).tolist() + [2]
        out.append({"prompt_ids": prompt, "response_ids": resp, "input_ids": prompt + resp,
                    "labels": [-100] * 6 + resp})
    return out


class TestTrainMode:
    @pytest.mark.parametrize("checkpointing", [True, False])
    def test_the_lora_model_is_in_train_mode_either_way(self, checkpointing):
        model = tiny_model(seed=4, dtype=torch.bfloat16).eval()
        pm = cal.build_lora_model(model, SMALL, gradient_checkpointing=checkpointing)
        assert pm.training and pm.base_model.model.training

    def test_default_build_uses_gradient_checkpointing_like_the_real_run(self):
        pm = cal.build_lora_model(tiny_model(seed=4, dtype=torch.bfloat16), SMALL)
        assert pm.base_model.model.is_gradient_checkpointing and pm.training

    def test_train_updates_forces_train_mode_and_trains_with_checkpointing_on_cpu(self):
        pm = cal.build_lora_model(tiny_model(seed=4, dtype=torch.bfloat16), SMALL)
        pm.eval()
        res = cal.train_updates(pm, examples(), SMALL, 2, 42, "cpu", 1)
        assert pm.training and res["all_finite"]

    def test_checkpointed_and_plain_training_agree_on_the_first_loss(self):
        losses = []
        for checkpointing in (True, False):
            pm = cal.build_lora_model(tiny_model(seed=4, dtype=torch.bfloat16), SMALL,
                                      gradient_checkpointing=checkpointing)
            torch.manual_seed(0)
            losses.append(cal.train_updates(pm, examples(), SMALL, 1, 42, "cpu", 1)[
                "updates"][0]["loss"])
        assert losses[0] == pytest.approx(losses[1], rel=1e-5)


# --- 8 ------------------------------------------------------------------------------------


class FakeGpu:
    """A GPU with processes the way a container sees them: host PIDs, used MiB."""

    def __init__(self, procs, own_pid):
        self.procs, self.own = dict(procs), own_pid

    def read(self):
        return [{"pid": p, "type": "C", "name": "python", "used_mib": m}
                for p, m in sorted(self.procs.items())]

    def allocate(self, mib):
        self.procs[self.own] = self.procs.get(self.own, 0) + mib
        return mib

    def release(self, handle):
        self.procs[self.own] = self.procs.get(self.own, 0) - handle


class TestPeersInAContainer:
    def run(self, gpu, **kw):
        return common.identify_own_gpu_pids(gpu.read, gpu.allocate, gpu.release, **kw)

    def test_finds_the_own_host_pid_among_foreign_and_idle_ones(self):
        gpu = FakeGpu({4242: 700, 9: 31000}, own_pid=777)
        res = self.run(gpu)
        assert res["own_pids"] == [777]
        assert "allocation" in res["method"] and res["probes"]
        assert gpu.procs[777] == 0, "the probe memory is released"

    def test_a_process_that_happens_to_grow_during_one_probe_is_not_mistaken_for_us(self):
        gpu = FakeGpu({4242: 700}, own_pid=777)
        calls = {"n": 0}
        real_alloc = gpu.allocate

        def noisy(mib):
            calls["n"] += 1
            if calls["n"] == 1:
                gpu.procs[4242] += mib  # a peer grows by exactly the probe size, once
            return real_alloc(mib)

        res = common.identify_own_gpu_pids(gpu.read, noisy, gpu.release)
        assert res["own_pids"] == [777]

    def test_unidentifiable_own_process_raises_instead_of_guessing(self):
        gpu = FakeGpu({4242: 700}, own_pid=777)
        gpu.allocate = lambda mib: mib  # our allocation never shows up in the process list
        with pytest.raises(common.PeersUnidentifiableError, match="no process"):
            self.run(gpu)

    def test_ambiguity_raises(self):
        gpu = FakeGpu({1: 10, 2: 10}, own_pid=1)

        def both(mib):
            gpu.procs[1] += mib
            gpu.procs[2] += mib
            return mib

        gpu.allocate = both
        gpu.release = lambda h: [gpu.procs.__setitem__(p, gpu.procs[p] - h) for p in (1, 2)]
        with pytest.raises(common.PeersUnidentifiableError, match="more than one"):
            self.run(gpu)

    def test_a_driver_that_hides_per_process_memory_raises(self):
        gpu = FakeGpu({1: 10}, own_pid=1)
        gpu.read = lambda: [{"pid": 1, "type": "C", "name": "python", "used_mib": None}]
        with pytest.raises(common.PeersUnidentifiableError, match="per-process"):
            self.run(gpu)

    def stamp_with(self, pids):
        return dict(GOOD_STAMP, gpu_processes=[{"pid": p, "type": "C", "name": "python"}
                                               for p in pids])

    def attempt(self, tmp_path, before, after, identify):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        return cal.execute_attempt(journal, "T", "T", tmp_path / "runs", lambda: {},
                                   stamp=stamps(before, after), prov=lambda: GOOD_PROV,
                                   identify_own=identify)

    def test_own_host_pid_is_excluded_and_the_method_is_recorded(self, tmp_path):
        s = self.stamp_with([777])
        rec = self.attempt(tmp_path, s, s, lambda: {"own_pids": [777], "method": "m", "probes": []})
        assert rec["outcome"] == "ok"
        assert rec["peers"]["own_pids"] == [777] and rec["peers"]["method"] == "m"

    def test_a_different_host_pid_is_a_foreign_process_and_voids(self, tmp_path):
        s = self.stamp_with([777, 888])
        rec = self.attempt(tmp_path, s, s, lambda: {"own_pids": [777], "method": "m"})
        assert rec["outcome"] == "void"
        assert any("888" in r for r in rec["void_reasons"])

    def test_failure_to_identify_the_own_context_voids_with_the_reason(self, tmp_path):
        def fail():
            raise common.PeersUnidentifiableError("no process matched")

        s = self.stamp_with([777])
        rec = self.attempt(tmp_path, s, s, fail)
        assert rec["outcome"] == "void"
        assert any("peers:" in r and "no process matched" in r for r in rec["void_reasons"])


# --- 9 ------------------------------------------------------------------------------------


class TestNits:
    def test_no_dead_fields_in_the_w_record(self):
        src = open(cal.__file__).read()
        assert "bf16_bitwise_equal_fp32_cast" not in src
        assert "del expected" not in src and "expected = x.requires_grad" not in src
        assert not math.isnan(0.0)
