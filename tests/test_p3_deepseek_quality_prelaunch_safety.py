"""Remote CPU checks for provenance, bounded transport and failure receipts."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.request
import urllib.response
from email.message import Message

import pytest

from benchmarks.harness import p3_deepseek_quality_model as model
from benchmarks.harness import p3_deepseek_quality_prepare as prepare
from benchmarks.harness import p3_deepseek_quality_stage as stage


@pytest.mark.parametrize("changed", ["hub_commit", "hub_etag", "hub_sha256", "sha256_origin",
                                     "sha256", "bytes", "tokenizer.json"])
def test_source_provenance_refuses_missing_fields_or_unreviewed_inventory(changed):
    records = {}
    for name, row in model.source_inventory().items():
        weight = name.endswith(".safetensors")
        records[name] = {**row, "hub_commit": model.REVISION,
                         "sha256": row["hub_sha256"] if weight else "a" * 64,
                         "sha256_origin": "hub_lfs_content_identity_not_locally_hashed" if weight
                         else "locally_measured_metadata"}
    if changed == "tokenizer.json":
        records.pop(changed)
    else:
        records["model-00001-of-000163.safetensors"].pop(changed)
    with pytest.raises(ValueError, match="inventory|provenance"):
        model._require_source_provenance(records)


@pytest.mark.parametrize("etag", ['"opaque-cdn-object"', 'W/"weak-transport-id"', None])
def test_chunked_payload_uses_verified_size_not_transport_etag(tmp_path, monkeypatch, etag):
    class Response(io.BytesIO):
        status = 200
        url = "https://cdn.example/file"
        headers = {} if etag is None else {"ETag": etag}

    monkeypatch.setattr(prepare, "_open_https", lambda request: Response(b"data"))
    target = tmp_path / "payload"
    prepare._stream_download("https://cdn.example/file", target,
                             {"bytes": 4, "hub_etag": "e" * 64})
    assert target.read_bytes() == b"data"


@pytest.mark.parametrize("destination", ["https://cdn.example/payload", "http://unsafe.example/"])
def test_redirect_opener_never_drains_body_or_downgrades_https(tmp_path, monkeypatch, destination):
    requested, bodies = [], []

    class Body(io.BytesIO):
        def read(self, size=-1):
            assert size >= 0, "redirect body was drained without a bound"
            return super().read(size)

    class Transport(urllib.request.HTTPSHandler):
        def https_open(self, request):
            requested.append(request.full_url)
            headers = Message()
            if len(requested) == 1:
                headers["Location"] = destination
                body = Body(b"redirect body must not be read")
                bodies.append(body)
                response = urllib.response.addinfourl(body, headers, request.full_url, 302)
            else:
                headers["Content-Length"] = "4"
                response = urllib.response.addinfourl(io.BytesIO(b"data"), headers,
                                                     request.full_url, 200)
            response.msg = "transport fixture"
            return response

    opener = urllib.request.build_opener(prepare._HttpsRedirects(), Transport())
    monkeypatch.setattr(prepare.urllib.request, "build_opener", lambda *args: opener)
    target = tmp_path / "payload"
    if destination.startswith("http:"):
        with pytest.raises(ValueError, match="HTTPS"):
            prepare._stream_download("https://hub.example/start", target, {"bytes": 4})
        assert requested == ["https://hub.example/start"]
    else:
        prepare._stream_download("https://hub.example/start", target, {"bytes": 4})
        assert target.read_bytes() == b"data"
        assert requested == ["https://hub.example/start", destination]
    assert bodies[0].closed


def test_hybrid_cgroup_uses_v1_memory_and_mount_aware_peaks(tmp_path):
    proc, group, unified = tmp_path / "proc", tmp_path / "memory", tmp_path / "unified"
    (proc / "self").mkdir(parents=True)
    group.mkdir()
    unified.mkdir()
    (proc / "meminfo").write_text("MemTotal: 134217728 kB\n")
    (proc / "self/cgroup").write_text("0::/\n4:memory:/docker/worker\n")
    (proc / "self/mountinfo").write_text(
        f"1 0 0:10 / {unified} rw - cgroup2 cgroup rw\n"
        f"2 0 0:11 /docker/worker {group} rw - cgroup cgroup rw,memory\n")
    (group / "memory.limit_in_bytes").write_text(str(96 * 2**30))
    (group / "memory.usage_in_bytes").write_text("1000")
    (group / "memory.stat").write_text("total_inactive_file 0\n")
    (group / "memory.max_usage_in_bytes").write_text("12345")
    assert prepare.inspect_ram(proc=proc)["effective_bytes"] == 96 * 2**30
    measured = prepare.cgroup_observation(proc=proc)
    assert measured["observed_hierarchy_peaks"] == [
        {"path": str(group / "memory.max_usage_in_bytes"), "bytes": 12345}]
    (group / "memory.max_usage_in_bytes").unlink()
    assert prepare.cgroup_observation(proc=proc)["status"] == "UNAVAILABLE"
    (group / "memory.max_usage_in_bytes").write_text("not a counter")
    with pytest.raises(ValueError, match="malformed"):
        prepare.cgroup_observation(proc=proc)


@pytest.mark.parametrize("field,value", [("process_rss_sampled_peak_bytes", None),
                                         ("cuda_reserved_peak_bytes", 0),
                                         ("sampling_errors", ["unreadable RSS"])])
def test_incomplete_memory_evidence_cannot_admit_conversion(field, value):
    measured = {"process_rss_sampled_peak_bytes": 100, "cuda_allocated_peak_bytes": 20,
                "cuda_reserved_peak_bytes": 30, "sampling_errors": []}
    measured[field] = value
    with pytest.raises(ValueError, match="incomplete"):
        prepare.require_memory_evidence(measured)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group")
def test_cleanup_exception_still_publishes_failed_stage_receipt(tmp_path, monkeypatch):
    original = subprocess.Popen

    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        wait = process.wait
        calls = 0

        def injected_wait(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise subprocess.TimeoutExpired("injected reap failure", 10)
            return wait(*args, **kwargs)

        process.wait = injected_wait
        return process

    monkeypatch.setattr(stage.subprocess, "Popen", start)
    result = stage.run_stage([sys.executable, "-c", "import sys;sys.stdout.write('kept')"],
                             output=tmp_path / "stage", seconds=10)
    assert result["status"] == "FAILED"
    assert result["cleanup"]["errors"][0]["operation"] == "wait"
    assert result["evidence_snapshot_only"] is True
    assert json.loads((tmp_path / "stage/result.json").read_text()) == result
    assert (tmp_path / "stage/manifest.json").is_file()
    assert (tmp_path / "stage/stdout.log").read_text() == "kept"


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM")
def test_stage_sigterm_runs_preparation_failure_finalizer(tmp_path):
    child_output = tmp_path / "prepare"
    source = tmp_path / model.REVISION
    code = (
        "import sys,time;from benchmarks.harness import p3_deepseek_quality_prepare as p;"
        "p.admit_machine=lambda *a,**k:time.sleep(60);"
        f"sys.argv=['prepare','download','--approved-pilot','--source',{str(source)!r},"
        f"'--output',{str(child_output)!r}];raise SystemExit(p.main())"
    )
    result = stage.run_stage([sys.executable, "-c", code], output=tmp_path / "stage", seconds=5)
    assert result["status"] == "TIMED_OUT"
    child = json.loads((child_output / "manifest.json").read_text())
    assert child["status"] == "FAILED"
    assert child["error"]["type"] == "InterruptedError"
    assert child["source_path"] == str(source)
    assert not source.exists()
