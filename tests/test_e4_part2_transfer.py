"""E4 part 2 transfer: push refuses bad local files, never creates a public repo; pull verifies."""
# ruff: noqa: I001 (the support import must run before the harness imports)

import json
import types

from tests.e4_part2_support import REPO_ROOT  # noqa: F401 (puts the harness on sys.path)
import e4_part2_common as common  # noqa: E402
import e4_part2_transfer as tr  # noqa: E402



def make_tree(tmp_path):
    out = tmp_path / "out"
    (out / "codes-dqoff").mkdir(parents=True)
    files = {}
    for name, data in (("codes-dqoff/a.bin", b"aaa"), ("b.json", b"{}")):
        (out / name).write_bytes(data)
        files[name] = {"sha256": common.sha256_bytes(data), "bytes": len(data)}
    manifest = tmp_path / "quant-manifest.json"
    manifest.write_text(json.dumps({"files": files, "source_files": {},
                                    "model": {"repo": common.MODEL_REPO,
                                              "revision": common.MODEL_REV}}))
    return out, manifest, files


class FakeApi:
    def __init__(self, private=True):
        self.calls, self.private = [], private

    def whoami(self):
        return {"name": "me"}

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def repo_info(self, repo_id, repo_type):
        return types.SimpleNamespace(private=self.private)

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw["path_in_repo"]))

    def upload_folder(self, **kw):
        self.ignore = kw.get("ignore_patterns")
        self.calls.append(("upload_folder", kw["folder_path"]))

    def list_repo_files(self, repo_id, repo_type):
        return ["codes-dqoff/a.bin", "b.json", "quant-manifest.json"]


def test_verify_tree_reports_missing_wrong_size_and_wrong_hash(tmp_path):
    out, _, files = make_tree(tmp_path)
    assert tr.verify_tree(out, files, "out", log=lambda *_: None) == []
    (out / "b.json").write_bytes(b"{ }")
    (out / "codes-dqoff" / "a.bin").write_bytes(b"aab")
    files["c.bin"] = {"sha256": "0" * 64, "bytes": 1}
    problems = tr.verify_tree(out, files, "out", log=lambda *_: None)
    assert any("b.json" in p and "bytes" in p for p in problems)
    assert any("a.bin" in p and "differs" in p for p in problems)
    assert any("c.bin" in p and "missing" in p for p in problems)


def test_push_dry_run_uploads_nothing(tmp_path):
    out, manifest, _ = make_tree(tmp_path)
    api = FakeApi()
    assert tr.push(out, manifest, "e4-part2-quant", True, api, log=lambda *_: None) == 0
    assert api.calls == []


def test_push_creates_a_private_dataset_under_the_account_then_uploads(tmp_path):
    out, manifest, _ = make_tree(tmp_path)
    api = FakeApi()
    assert tr.push(out, manifest, "e4-part2-quant", False, api, log=lambda *_: None) == 0
    kind, repo_id, kw = api.calls[0]
    assert (kind, repo_id, kw["private"], kw["repo_type"]) == (
        "create_repo", "me/e4-part2-quant", True, "dataset")
    kinds = [c[0] for c in api.calls[1:]]
    assert kinds[0] == "upload_file" and "upload_folder" in kinds
    assert ("upload_folder", str(out / "codes-dqoff")) in api.calls
    assert ".cache/**" in api.ignore  # the resume metadata is not part of the dataset


def test_push_refuses_a_public_repo_and_corrupted_local_files(tmp_path):
    out, manifest, _ = make_tree(tmp_path)
    api = FakeApi(private=False)
    assert tr.push(out, manifest, "e4-part2-quant", False, api, log=lambda *_: None) == 1
    assert [c[0] for c in api.calls] == ["create_repo"]
    (out / "b.json").write_bytes(b"x")
    api2 = FakeApi()
    assert tr.push(out, manifest, "e4-part2-quant", False, api2, log=lambda *_: None) == 1
    assert api2.calls == []


def test_unlisted_files_are_reported(tmp_path):
    out, _, files = make_tree(tmp_path)
    (out / "stray.txt").write_text("x")
    assert tr.unlisted_files(out, files) == ["stray.txt"]
