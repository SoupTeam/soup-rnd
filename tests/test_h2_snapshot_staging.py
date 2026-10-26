"""Transport corruption and immutable destination guards, without network/GPU."""

import hashlib
import io
import json

import pytest

from benchmarks.harness.h2_snapshot_staging import stage, validate_manifest


def manifest():
    return {"repo": "test/model", "revision": "a" * 40, "total_bytes": 4,
            "files_bytes": {"model.bin": 4},
            "files_sha256": {"model.bin": hashlib.sha256(b"data").hexdigest()}}


class Response(io.BytesIO):
    status = 200

    def __init__(self, body=b"data", **headers):
        super().__init__(body)
        self.headers = {"Content-Length": "4", **headers}


def test_single_request_and_readback(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        assert not request.has_header("Range")
        return Response()

    result = stage(manifest(), tmp_path / "snapshot", tmp_path / "proof", opener=opener)
    assert result["status"] == "PASS_REMOTE_SNAPSHOT_SHA"
    assert len(calls) == 1 and ("a" * 40) in calls[0][0]
    assert (tmp_path / "snapshot/model.bin").read_bytes() == b"data"
    assert result["structural_audit"] == "PENDING"
    assert result["performance_verdict"] == "NO_VERDICT"
    assert json.loads((tmp_path / "proof/receipt.json").read_text()) == result


@pytest.mark.parametrize("body,headers,error", [
    (b"dat", {}, "Truncated"),
    (b"datas", {}, "Oversized"),
    (b"oops", {}, "SHA256"),
    (b"data", {"Content-Length": "3"}, "Content-Length"),
    (b"data", {"Content-Encoding": "gzip"}, "encoding"),
])
def test_corruption_fails_without_retry(tmp_path, body, headers, error):
    calls = []

    def opener(*args, **kwargs):
        calls.append(1)
        return Response(body, **headers)

    result = stage(manifest(), tmp_path / "snapshot", tmp_path / "proof", opener=opener)
    assert result["status"] == "FAILED_REMOTE_SNAPSHOT_STAGING"
    assert error in result["error"]
    assert len(calls) == 1 and result["retries"] == 0
    assert not (tmp_path / "snapshot/model.bin").exists()


def test_rejects_partial_http_response(tmp_path):
    response = Response()
    response.status = 206
    result = stage(manifest(), tmp_path / "snapshot", tmp_path / "proof",
                   opener=lambda *args, **kwargs: response)
    assert "200 required" in result["error"]


def test_existing_destination_not_resumed(tmp_path):
    target = tmp_path / "snapshot"
    target.mkdir()
    partial = target / "model.bin.partial"
    partial.write_bytes(b"da")
    with pytest.raises(FileExistsError):
        stage(manifest(), target, tmp_path / "proof")
    assert partial.read_bytes() == b"da"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "sub/file", "..", "x\\y"])
def test_unsafe_paths_rejected(name):
    value = manifest()
    value["files_bytes"] = {name: 4}
    value["files_sha256"] = {name: "a" * 64}
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_mutable_revision_rejected():
    value = manifest()
    value["revision"] = "main"
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_deadline_fails_before_publishing(tmp_path, monkeypatch):
    import benchmarks.harness.h2_snapshot_staging as transport

    values = iter([0, 10])
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(values, 10))
    result = stage(manifest(), tmp_path / "snapshot", tmp_path / "proof", 1,
                   opener=lambda *args, **kwargs: pytest.fail("Must not request after deadline"))
    assert "deadline" in result["error"]
    assert not (tmp_path / "snapshot/model.bin").exists()
