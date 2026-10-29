"""Synthetic CPU checks of the actual qualification harness and failure guards."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness.d3_trainer_analysis import analyze, replay_native_tensors
from benchmarks.harness.d3_trainer_gate import (
    NumericalFailureError,
    assemble,
    capture_class,
    correctness,
    make_config,
    numerical_view,
    one_replay,
    public_workspace,
    require_exact,
    save,
    tensor_record,
)
from benchmarks.harness.d3_trainer_prepare import metadata
from benchmarks.harness.d3_trainer_protocol import cell_plan, read_rule
from benchmarks.harness.d3_trainer_series import child, child_command, oracle_path, parent


def _prepare_tiny(tmp_path, *, tied=True, layers=2):
    import torch
    from datasets import Dataset
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.layer_shard import resolve_shard_dir

    rule = copy.deepcopy(read_rule())
    rule.update(tokens=64, rows=3, optimizer_steps=2)
    rule["optimizer"]["gradient_windows"] = [2, 1]
    torch.manual_seed(71)
    checkpoint = tmp_path / "checkpoint"
    config = OlmoeConfig(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=128,
        output_router_logits=False,
        attention_dropout=0.0,
        tie_word_embeddings=tied,
        pad_token_id=0,
        eos_token_id=1,
    )
    OlmoeForCausalLM(config).save_pretrained(checkpoint)
    raw = Tokenizer(WordLevel({str(index): index for index in range(64)}, unk_token="0"))
    PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="0", eos_token="1").save_pretrained(
        checkpoint,
    )
    rows = []
    for index in range(3):
        ids = [3 + (index + offset) % 60 for offset in range(64)]
        labels = [value if offset % 11 else -100 for offset, value in enumerate(ids)]
        rows.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * 64})
    dataset = tmp_path / "dataset"
    Dataset.from_list(rows).save_to_disk(str(dataset))
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("synthetic CPU rows only")
    row = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "initial_adapters": str(tmp_path / "initial.safetensors"),
        "shards": resolve_shard_dir(str(checkpoint)),
    }
    prepared = {
        "workspace": str(tmp_path),
        "codecs": {"dense": row},
        "rows_path": str(rows_path),
        "library": "unused-on-cpu",
    }
    config = make_config(rule, prepared, tmp_path / "setup", "dense", "sequential", "ordinary")
    save(dataset / "metadata.json", metadata(config))
    wrapper = SFTTrainerWrapper(config, device="cpu")
    try:
        with public_workspace(prepared, tmp_path / "setup"):
            wrapper.setup({"train": [], "val": []})
        save_file(
            {
                key: value.detach().cpu().contiguous().clone()
                for key, value in get_peft_model_state_dict(wrapper.model).items()
            },
            row["initial_adapters"],
        )
    finally:
        wrapper._close_stream_runtime()
    return rule, prepared


@pytest.fixture
def tiny_prepared(tmp_path):
    return _prepare_tiny(tmp_path)


def test_control_evaluation_uses_execution_device_not_meta_embedding(tmp_path):

    rule, prepared = _prepare_tiny(tmp_path, tied=False)
    wrapper = assemble(
        rule,
        prepared,
        tmp_path / "untied-eval",
        "dense",
        "native-resident",
        "ordinary",
        device="cpu",
    )
    capture = capture_class()(wrapper)
    embedding = wrapper.model.get_input_embeddings()
    assert embedding.weight.is_meta
    observed = []

    def check(_module, args):
        observed.append(args[0].device.type)
        assert not args[0].is_meta, "control allocated token IDs on the meta placeholder device"
        assert args[0].device.type == "cpu"

    hook = embedding.register_forward_pre_hook(check)
    try:
        capture.on_train_end(None, None, None, model=wrapper.model)
        assert observed == ["cpu"]
        assert capture.eval_logits["shape"] == [1, rule["tokens"], 64]
    finally:
        hook.remove()
        capture.close()
        wrapper._close_stream_runtime()


@pytest.mark.parametrize("training", ["ordinary", "l2l"])
def test_untied_streamed_boundaries_train_control_and_reload_exactly(tmp_path, training):
    rule, prepared = _prepare_tiny(tmp_path, tied=False, layers=4)
    native = one_replay(
        rule,
        prepared,
        tmp_path / "untied-native",
        "dense",
        "native-resident",
        "ordinary",
        raw_oracle=True,
        device="cpu",
    )
    output = tmp_path / "untied-controls"
    output.mkdir()
    result = correctness(rule, prepared, output, "dense", "grouped", training, native, device="cpu")
    assert result["A_A"] and result["negative_effective"] and result["restored"]
    assert result["save_reload"]
    require_exact(
        result["reload"]["evaluation"],
        result["numerical"]["evaluation"],
        "untied four-layer reload",
    )


def test_tensor_comparison_keeps_signed_zero_shape_dtype_and_scalar_cases():
    import torch

    positive = tensor_record(torch.tensor([0.0]))
    negative = tensor_record(torch.tensor([-0.0]))
    assert positive != negative
    assert tensor_record(torch.tensor(1.0))["shape"] == []
    with pytest.raises(NumericalFailureError):
        require_exact(positive, negative, "signed-zero")
    with pytest.raises(NumericalFailureError, match="nonfinite"):
        tensor_record(torch.tensor([float("nan")]))


def test_public_workspace_does_not_relax_the_schema_to_filesystem_root(tmp_path):
    data = {"workspace": str(tmp_path), "rows_path": str(tmp_path / "rows"), "codecs": {}}
    previous = Path.cwd()
    with pytest.raises(ValueError, match="escapes"):
        with public_workspace(data, tmp_path.parent / "outside"):
            pytest.fail("uncontained context entered")
    assert Path.cwd() == previous
    with pytest.raises(RuntimeError, match="injected"):
        with public_workspace(data, tmp_path / "inside"):
            assert Path.cwd() == tmp_path
            raise RuntimeError("injected")
    assert Path.cwd() == previous


def test_draft_parent_and_child_refuse_before_any_model_allocation(tmp_path, monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("allocated model/tensor before the source-freeze check")

    monkeypatch.setattr(torch, "empty", forbidden)
    from benchmarks.harness import d3_trainer_series

    draft = copy.deepcopy(read_rule())
    draft["execution_enabled"] = False
    monkeypatch.setattr(d3_trainer_series, "read_rule", lambda: draft)
    args = SimpleNamespace(prepared=tmp_path / "missing", out=tmp_path / "session")
    with pytest.raises(ValueError, match="disabled"):
        parent(args)
    with pytest.raises(ValueError, match="disabled"):
        child(args)
    assert not args.out.exists()


def test_child_argv_and_canonical_oracle_match_the_prospective_plan(tmp_path):
    cells = cell_plan(read_rule())
    for cell in cells:
        command = child_command(tmp_path / "prepared.json", tmp_path, cell)
        assert command[command.index("--cell") + 1] == cell["name"]
        native = cell["name"] == f"correctness-{cell['codec']}-ordinary-native-resident"
        if native:
            assert "--oracle" not in command
        else:
            assert Path(command[command.index("--oracle") + 1]) == oracle_path(
                tmp_path, cell["codec"]
            )


def test_real_cpu_harness_records_all_layer_derivatives_and_canonical_oracles(
    tiny_prepared, tmp_path
):
    rule, prepared = tiny_prepared
    native = one_replay(
        rule,
        prepared,
        tmp_path / "native",
        "dense",
        "native-resident",
        "ordinary",
        raw_oracle=True,
        device="cpu",
    )
    candidate = one_replay(
        rule, prepared, tmp_path / "grouped", "dense", "grouped", "ordinary", device="cpu"
    )
    require_exact(numerical_view(candidate), numerical_view(native), "actual wrapper oracle")
    layerwise = one_replay(
        rule, prepared, tmp_path / "layerwise", "dense", "grouped", "l2l", device="cpu"
    )
    require_exact(numerical_view(layerwise), numerical_view(native), "ordinary versus L2L")
    assert len(native["steps"]) == 2
    assert (tmp_path / "native/oracle/step-1.safetensors").is_file()
    assert all(frame["raw_parameter_gradients"] for frame in native["steps"])
    container = tmp_path / "native-container"
    container.mkdir()
    # The offline layout is the same layout used by real native correctness cells.
    import shutil

    shutil.copytree(tmp_path / "native", container / "positive")
    assert replay_native_tensors(container, native) > 20


@pytest.mark.parametrize("training", ["ordinary", "l2l"])
def test_same_process_positive_negative_restored_and_reload_use_the_public_wrapper(
    tiny_prepared,
    tmp_path,
    training,
):
    rule, prepared = tiny_prepared
    output = tmp_path / f"controls-{training}"
    output.mkdir()
    result = correctness(rule, prepared, output, "dense", "grouped", training, None, device="cpu")
    assert result["A_A"] and result["negative_effective"] and result["restored"]
    assert result["save_reload"]
    require_exact(result["reload"]["evaluation"], result["numerical"]["evaluation"], "reload")


def test_offline_analysis_never_labels_an_incomplete_series_complete(tmp_path):
    from benchmarks.harness.d3_trainer_protocol import file_hash

    save(tmp_path / "protocol.json", read_rule())
    save(
        tmp_path / "session.json",
        {
            "status": "incomplete",
            "session_id": "synthetic-no-run",
            "source": {"git_head": "x"},
            "protocol_sha256": file_hash(tmp_path / "protocol.json"),
            "cells": [],
            "traceback": "synthetic timeout record",
        },
    )
    result = analyze(tmp_path)
    assert not result["d3_accepted"]
    assert not result["complete_public_trainer_series"]
    assert result["status"] == "incomplete_no_task_verdict"
    data = json.loads((tmp_path / "session.json").read_text())
    data["protocol_sha256"] = "wrong"
    save(tmp_path / "session.json", data)
    with pytest.raises(ValueError, match="identities"):
        analyze(tmp_path)


@pytest.mark.parametrize("failure", ["timeout", "child_exit", "foreign_session"])
def test_parent_retains_failed_child_and_refuses_a_second_output_name(
    tmp_path, monkeypatch, failure
):
    import subprocess

    from benchmarks.harness import d3_trainer_series as series

    source = {"git_head": "synthetic-parent-control-not-a-measurement"}
    monkeypatch.setattr(series, "frozen_source", lambda: source)
    monkeypatch.setattr(series, "verify_prepared", lambda *args: {})
    monkeypatch.setattr(series, "verify_host", lambda *args: None)
    calls = []

    def injected(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        folder = Path(command[command.index("--out") + 1])
        session = json.loads((folder.parent / "session.json").read_text())
        folder.mkdir()
        save(
            folder / "result.json",
            dict(
                status="ok",
                source=source,
                F1={"outcome": "ok"},
                session_id="other-process-session",
                protocol_sha256=session["protocol_sha256"],
                formal_execution=True,
                cell=cell_plan(read_rule())[0],
            ),
        )
        return SimpleNamespace(returncode=1 if failure == "child_exit" else 0)

    monkeypatch.setattr(series.subprocess, "run", injected)
    args = SimpleNamespace(prepared=tmp_path / "prepared.json", out=tmp_path / "first")
    assert series.parent(args) == 1
    session = json.loads((args.out / "session.json").read_text())
    assert session["status"] == "incomplete"
    assert len(session["cells"]) == len(calls) == 1
    assert session["cells"][0]["status"] == "failed_or_no_verdict"
    assert session["cells"][0]["command"] == calls[0]
    assert session["cells"][0]["traceback"]
    args.out = tmp_path / "second-name"
    with pytest.raises(FileExistsError):
        series.parent(args)
    assert len(calls) == 1
    assert not args.out.exists()


def test_parent_time_budget_stops_before_spawning_and_preserves_session(tmp_path, monkeypatch):
    from benchmarks.harness import d3_trainer_series as series

    monkeypatch.setattr(series, "frozen_source", lambda: {"git_head": "synthetic-budget"})
    monkeypatch.setattr(series, "verify_prepared", lambda *args: {})
    monkeypatch.setattr(series, "verify_host", lambda *args: None)
    ticks = iter([0.0, 10801.0])
    monkeypatch.setattr(series.time, "monotonic", lambda: next(ticks))

    def forbidden(*args, **kwargs):
        pytest.fail("spawned after the complete-session budget was exhausted")

    monkeypatch.setattr(series.subprocess, "run", forbidden)
    args = SimpleNamespace(prepared=tmp_path / "prepared.json", out=tmp_path / "session")
    assert series.parent(args) == 1
    session = json.loads((args.out / "session.json").read_text())
    assert session["status"] == "incomplete" and not session["cells"]
    assert "budget exhausted" in session["traceback"]
