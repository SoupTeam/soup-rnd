"""Prospective budgets, source boundary and synthetic-versus-trained safeguards."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.harness import d3_fused_trainer_protocol as protocol
from benchmarks.harness import d3_fused_trainer_series as series
from benchmarks.harness.d3_fused_trainer_analysis import analyze, audit_record


def test_fused_finite_grid_and_historical_rule_remain_distinct():
    from benchmarks.harness import d3_trainer_protocol as old

    rule = protocol.read_rule()
    pilot = protocol.cell_plan(rule)
    assert len(pilot) == 112
    assert len({c["name"] for c in pilot}) == 112
    assert [sum(c["kind"] == k for c in pilot) for k in ("correctness", "timing", "profile")] == (
        [24, 80, 8]
    )
    assert len(protocol.cell_plan(rule, blocks=10)) == 192
    assert len(old.cell_plan(old.read_rule())) == 234
    with pytest.raises(ValueError):
        old.validate_rule(rule)
    with pytest.raises(ValueError):
        protocol.validate_rule(old.read_rule())


@pytest.mark.parametrize(
    "key,value",
    [
        ("allow_allclose", True),
        ("allow_value_dependent_fallback", True),
        ("replacements", 1),
        ("max_blocks", 11),
        ("fused_native_rows_max", 16),
        ("native_rows_from", None),
        ("compute_capability", [8, 0]),
    ],
)
def test_rule_refuses_relaxed_or_changed_contract(key, value):
    rule = protocol.read_rule()
    rule[key] = value
    with pytest.raises(ValueError):
        protocol.validate_rule(rule)


def blocks(count=5):
    return [
        [
            dict(mode=m, seconds_per_input_token=1 if m == "grouped" else 0.95, outcome="ok")
            for m in protocol.ORDER
        ]
        for _ in range(count)
    ]


def test_repeat_rule_requires_complete_ordered_valid_blocks():
    assert protocol.repeat_target(blocks()) == {"status": "complete", "target": 5}
    for variant in (blocks(4), blocks(11), [*blocks()[:-1], blocks()[0][:-1]]):
        with pytest.raises(ValueError):
            protocol.repeat_target(variant)
    for field, value in (
        ("outcome", "void"),
        ("outcome", "unknown"),
        ("seconds_per_input_token", float("nan")),
        ("seconds_per_input_token", True),
        ("mode", "sequential"),
    ):
        rows = blocks()
        rows[0][0][field] = value
        with pytest.raises(ValueError):
            protocol.repeat_target(rows)


def test_repeat_rule_refuses_excessive_variance_without_new_attempts():
    rows = blocks()
    for index, block in enumerate(rows):
        for row in block:
            row["seconds_per_input_token"] = 0.0001 if index % 2 else 100
    assert protocol.repeat_target(rows)["status"] == "no_verdict_precision_budget"


def test_disabled_or_missing_preparation_cannot_be_exported(tmp_path):
    rule = protocol.read_rule()
    rule["execution_enabled"] = False
    target = tmp_path / protocol.PROTOCOL
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(rule))
    assert "formal execution disabled" in protocol.launch_blockers(rule, tmp_path, check_git=False)
    with pytest.raises(ValueError):
        protocol.export_packet(tmp_path / "forbidden.zip", root=tmp_path)
    assert not (tmp_path / "forbidden.zip").exists()


def test_finite_parent_budget_stops_before_spawning_or_appending_cell(tmp_path, monkeypatch):
    rule = protocol.read_rule()
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    monkeypatch.setattr(series, "source_receipt", lambda **kw: {"git_head": "a" * 40})
    monkeypatch.setattr(series, "prepared_inputs", lambda *a, **kw: {})
    monkeypatch.setattr(series, "verify_host", lambda *a: None)
    times = iter([0, rule["session_timeout_seconds"] + 1])
    monkeypatch.setattr(series.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(series.subprocess, "run", lambda *a, **kw: pytest.fail("child was spawned"))
    args = SimpleNamespace(prepared=prepared, out=tmp_path / "session", synthetic=False)
    assert series.parent(args) == 1
    result = json.loads((args.out / "session.json").read_text())
    assert result["status"] == "incomplete" and not result["cells"]


def test_synthetic_switch_cannot_execute_unpinned_actual_model_bytes(tmp_path):
    rule = protocol.read_rule()
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"not the pinned random-weight fixture")
    fixture = rule["required_launcher_fixture"]
    data = {
        "model_geometry": {
            k: fixture[k] for k in ("layers", "experts", "vocab", "tie_word_embeddings")
        },
        "checkpoint_files": {"model.safetensors": fixture["weight_sha256"]},
        "codecs": {"nf4": {"checkpoint": str(tmp_path)}},
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(data))
    source = {
        "prepared": str(path),
        "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="actual nonfixture"):
        series.prepared_inputs(path, rule, source, synthetic=True)


def test_synthetic_checkpoint_checks_do_not_replace_the_prepared_manifest_path(
    tmp_path, monkeypatch
):
    rule = protocol.read_rule()
    fixture = rule["required_launcher_fixture"]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 4,
                "num_experts": 4,
                "vocab_size": 64,
                "tie_word_embeddings": False,
            }
        )
    )
    data = {
        "model_geometry": {
            k: fixture[k] for k in ("layers", "experts", "vocab", "tie_word_embeddings")
        },
        "checkpoint_files": {"model.safetensors": fixture["weight_sha256"]},
        "codecs": {"nf4": {"checkpoint": str(checkpoint)}},
        "boot_id": "same",
    }
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps(data))
    checksum = hashlib.sha256(prepared.read_bytes()).hexdigest()
    source = {"prepared": str(prepared), "prepared_sha256": checksum}
    monkeypatch.setattr(
        series.historical,
        "file_hash",
        lambda p: checksum if p == prepared else fixture["weight_sha256"],
    )

    def verify(path, checked):
        assert path == prepared and checked["hardware"]["boot_id"] == "same"
        return data

    monkeypatch.setattr(series, "verify_prepared", verify)
    assert series.prepared_inputs(prepared, rule, source, synthetic=True) is data


def test_incomplete_synthetic_series_is_never_pretrained_qualification(tmp_path):
    rule = protocol.read_rule()
    payload = json.dumps(rule)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    source = {
        "kind": "synthetic packed-fused launcher only",
        "protocol_utf8": payload,
        "git_blob_sha256": {protocol.PROTOCOL: digest},
    }
    session = dict(
        formal_execution=False, source=source, protocol_sha256=digest, status="incomplete", cells=[]
    )
    (tmp_path / "session.json").write_text(json.dumps(session))
    (tmp_path / "protocol.json").write_text(payload)
    result = analyze(tmp_path)
    assert result["status"] == "incomplete_no_verdict" and not result["complete_pretrained_series"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "fail"),
        ("F1", {"outcome": "unknown"}),
        ("source", {}),
        ("formal_execution", False),
    ],
)
def test_offline_audit_refuses_false_pass_or_changed_identity(field, value):
    cell = protocol.cell_plan(protocol.read_rule())[0]
    session = dict(
        source={"git_head": "a" * 40},
        session_id="one",
        protocol_sha256="same",
        formal_execution=True,
    )
    record = dict(session, cell=cell, status="ok", F1={"outcome": "ok"})
    row = dict(returncode=0, status="ok")
    audit_record(session, cell, row, record)
    changed = copy.deepcopy(record)
    changed[field] = value
    with pytest.raises(ValueError):
        audit_record(session, cell, row, changed)


def test_source_gate_refuses_modified_executed_code(tmp_path, monkeypatch):
    rule = protocol.read_rule()
    names = [*protocol.ENGINES, "src/checked.py", protocol.PROTOCOL]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rule) if name == protocol.PROTOCOL else "# checked source\n")
    receipt = {
        "kind": "synthetic packed-fused launcher only",
        "git_blob_sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names
        },
    }
    (tmp_path / "D3-FUSED-SYNTHETIC-SOURCE.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    monkeypatch.setattr(protocol, "read_rule", lambda: rule)
    assert series.source_receipt(synthetic=True) == receipt
    (tmp_path / "src/checked.py").write_text("# changed source\n")
    with pytest.raises(ValueError, match="executed source changed"):
        series.source_receipt(synthetic=True)


def test_actual_child_command_is_accepted_by_its_cli_without_gpu_work(tmp_path, monkeypatch):
    cell = protocol.cell_plan(protocol.read_rule())[0]
    command = series.child_command(
        tmp_path / "prepared.json", tmp_path / "session", cell, synthetic=True
    )

    def checked(args):
        assert args.synthetic and args.cell == cell["name"]
        assert args.session == tmp_path / "session"
        assert args.out == tmp_path / "session" / cell["name"]
        return 0

    monkeypatch.setattr(series, "child", checked)
    assert series.main(command[command.index("child") :]) == 0


@pytest.mark.parametrize("corrupt", ["adapter", "A-A", "negative"])
def test_offline_audit_rechecks_payloads_even_with_true_control_flags(tmp_path, corrupt):
    import torch
    from safetensors.torch import save_file

    from benchmarks.harness.d3_fused_trainer_analysis import audit_correctness
    from benchmarks.harness.d3_trainer_gate import NumericalFailureError, tensor_record

    value = torch.tensor([1.0])
    final = {"parameter": tensor_record(value)}
    positive = dict(
        steps=[{"parameter_gradients": final}],
        windows=[],
        final=final,
        evaluation={"sha256": "positive"},
        attention="eager",
    )
    for name in ("positive", "A-A", "restored", "negative"):
        path = tmp_path / name
        path.mkdir()
        data = copy.deepcopy(positive)
        if name == "negative":
            data["evaluation"] = {"sha256": "changed"}
            data["steps"][0]["parameter_gradients"] = {"changed": True}
        (path / "record.json").write_text(json.dumps(data))
    (tmp_path / "positive/adapter").mkdir()
    artifact = tmp_path / "positive/adapter/adapter_model.safetensors"
    save_file({"parameter": value}, str(artifact))
    positive["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "positive/record.json").write_text(json.dumps(positive))
    result = dict(
        numerical=positive, reload={"factors": final, "evaluation": positive["evaluation"]}
    )
    audit_correctness(tmp_path, result)
    if corrupt == "adapter":
        save_file({"parameter": value + 1}, str(artifact))
    else:
        changed = copy.deepcopy(positive)
        if corrupt == "A-A":
            changed["evaluation"] = {"sha256": "wrong"}
        (tmp_path / corrupt / "record.json").write_text(json.dumps(changed))
    with pytest.raises((ValueError, NumericalFailureError)):
        audit_correctness(tmp_path, result)


@pytest.mark.parametrize("nested", [False, True])
def test_packed_resident_control_preserves_bytes_and_per_lease_guards(tmp_path, nested):
    import torch

    from benchmarks.harness.d3_fused_trainer_gate import PackedResidentPool
    from soup_cli.utils.expert_group import GroupedExpertPool
    from soup_cli.utils.expert_stream import _validate_routing
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY
    from tests.test_d3_grouped_expert import _fixture

    _module, reader, baseline, old_pool = _fixture(
        tmp_path, nf4=True, nested=nested, dtype="bfloat16"
    )
    codes = {NF4_CODE_KEY: baseline._buffers["code"]}
    if nested:
        codes[NF4_NESTED_CODE_KEY] = baseline._buffers["nested_code"]
    source = GroupedExpertPool(
        reader,
        device="cpu",
        dtype="bfloat16",
        group_width=4,
        weight_budget_bytes=64 * 2**20,
        codes=codes,
        packed_only=True,
    )
    try:
        pool = PackedResidentPool(reader, source)
        _validate_routing(
            torch.zeros(2, reader.shapes["gate_up"][1], dtype=torch.bfloat16),
            torch.zeros(2, 1, dtype=torch.int64),
            torch.ones(2, 1, dtype=torch.bfloat16),
            pool,
        )
        with pool.lease(reader, [0]) as rows, source.lease(reader, [0]) as streamed:
            first = rows[0][0]
            assert torch.equal(first.packed, streamed[0][0].packed)
            assert first.packed.data_ptr() != streamed[0][0].packed.data_ptr()
            first.packed._soup_stream_owner_check()
            with pytest.raises(RuntimeError, match="active"):
                pool.close()
        with pool.lease(reader, [0]):
            with pytest.raises(RuntimeError, match="expired"):
                first.packed._soup_stream_owner_check()
        pool.close()
    finally:
        source.close()
        old_pool.close()
        baseline.close()
        reader.close()
