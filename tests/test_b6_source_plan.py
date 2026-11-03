"""A tighter disk plan must bind to the actual source and writer, before writes."""

import copy
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from benchmarks.harness import b6_source_plan as planning


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = b'{"model_type":"deepseek_v3"}'
    (source / "config.json").write_bytes(config)
    header = {"model.layers.0.mlp.experts.0.gate_proj.weight":
              {"shape": [8, 8], "dtype": "F32", "data_offsets": [0, 256]}}
    raw = json.dumps(header).encode()
    name = "model-00001-of-00001.safetensors"
    payload = len(raw).to_bytes(8, "little") + raw + bytes(256)
    (source / name).write_bytes(payload)
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget

    bound = new_expert_cache_budget([str(source / name)])
    plan = dict(version=1, source_revision="pinned", dtype="float32", n_layers=1,
                quant_suffixes=[], cache_budget_bytes=bound, generic_cache_budget_bytes=bound,
                config_sha256=hashlib.sha256(config).hexdigest(), source_bytes=len(payload),
                files=[dict(name=name, size=len(payload),
                            header_sha256=hashlib.sha256(raw).hexdigest())])
    return source, plan, payload


@pytest.mark.parametrize("defect", ["none", "revision", "config", "header", "size", "bound"])
def test_plan_binds_to_source_headers_sizes_config_and_revision(tmp_path, defect):
    source, plan, _ = _source(tmp_path)
    if defect == "revision":
        plan["source_revision"] = "other"
    elif defect == "config":
        (source / "config.json").write_text("{}")
    elif defect == "header":
        plan["files"][0]["header_sha256"] = "0" * 64
    elif defect == "size":
        plan["files"][0]["size"] += 1
    elif defect == "bound":
        plan["cache_budget_bytes"] -= 1
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    if defect == "none":
        assert planning.validate_plan(source, path, "pinned") == plan
    else:
        with pytest.raises(ValueError, match="mismatch|differs"):
            planning.validate_plan(source, path, "pinned")


@pytest.mark.parametrize("defect", ["none", "suffixes", "layers", "dtype"])
def test_plan_must_match_the_actual_writer_selection(defect):
    plan = dict(quant_suffixes=["self_attn.q_proj.weight"], n_layers=61, dtype="float32")
    actual = copy.deepcopy(plan)
    if defect == "suffixes":
        actual["quant_suffixes"] = []
    elif defect == "layers":
        actual["n_layers"] = 62
    elif defect == "dtype":
        actual["dtype"] = "bfloat16"
    if defect == "none":
        planning.validate_writer_plan(plan, actual["quant_suffixes"], actual["n_layers"],
                                      actual["dtype"])
    else:
        with pytest.raises(ValueError, match="actual writer"):
            planning.validate_writer_plan(plan, actual["quant_suffixes"], actual["n_layers"],
                                          actual["dtype"])


def test_remote_header_reads_only_the_two_exact_ranges(tmp_path, monkeypatch):
    _, _, payload = _source(tmp_path)
    requests = []

    def open_range(request, timeout):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"])
        start, end = map(int, match.groups())
        requests.append((start, end))
        raw = payload[start:end + 1]
        response = SimpleNamespace(status=206,
                                   headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
                                   read=lambda maximum: raw[:maximum])
        from contextlib import nullcontext

        return nullcontext(response)

    monkeypatch.setattr(planning.urllib.request, "urlopen", open_range)
    raw, header = planning.read_remote_header("https://example.test/model", len(payload))
    assert requests == [(0, 7), (8, 7 + len(raw))]
    assert header["model.layers.0.mlp.experts.0.gate_proj.weight"]["shape"] == [8, 8]


@pytest.mark.parametrize("status,content_range", [(200, None), (206, "bytes 0-7/99999")])
def test_ignored_or_wrong_range_is_refused_before_reading_body(monkeypatch, status, content_range):
    from contextlib import nullcontext

    response = SimpleNamespace(status=status, headers={"Content-Range": content_range},
                               read=lambda count: pytest.fail("unbounded weight body read"))
    monkeypatch.setattr(planning.urllib.request, "urlopen",
                        lambda *args, **kwargs: nullcontext(response))
    with pytest.raises(ValueError, match="bounded header range"):
        planning.read_remote_header("https://example.test/model", 1000)


@pytest.mark.parametrize("name", ["../outside.safetensors", "a/model.safetensors",
                                 "a\\model.safetensors", "config.json"])
def test_plan_cannot_read_or_create_paths_outside_its_shards(tmp_path, name):
    with pytest.raises(ValueError, match="top-level"):
        planning.source_file(tmp_path, name)
