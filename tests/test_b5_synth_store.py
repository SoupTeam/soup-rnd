"""B5 ticket 05: the synthetic MoE store writer, on the small shape, on CPU.

The writer is ``benchmarks/harness/b5_synth_store.py``. It writes random NF4 experts in the
format of the real ``per_expert`` cache. The weights are random and say nothing about the
quality of a real model, so everything it writes is signed ``synthetic``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import b5_synth_store as synth
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

SEED = 5
STEPS = 3
RUN_CONFIG = """\
base: synthetic-moe-small
task: sft
backend: transformers
data:
  train: unused.json
  format: plaintext
  max_length: 64
training:
  quantization: 4bit
  epochs: 1
  batch_size: 1
  lr: 1.0e-3
  stream_layers: true
  stream_source: disk
  stream_expert_layout: per_expert
  lora:
    r: 2
    alpha: 4
    dropout: 0.0
    target_modules: [q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj]
"""


@pytest.fixture(scope="module")
def small_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("b5-synth") / "synthetic-moe-small-seed5"
    synth.write_store(out, synth.SMALL, seed=SEED)
    return out


def test_the_real_loader_reads_every_expert_of_every_moe_layer_independently(
    small_store: Path,
) -> None:
    import torch

    from soup_cli.utils.expert_shard import ExpertShardReader, aligned_staging
    from soup_cli.utils.layer_shard import expert_paths, read_shard_index

    index = read_shard_index(str(small_store))
    shape = synth.SMALL

    assert index.expert_layout == "per_expert"
    assert index.expert_count == shape.experts
    assert index.expert_layers == tuple(range(shape.layers - shape.moe_layers, shape.layers))
    assert index.quant == "nf4"
    paths = expert_paths(str(small_store), index)
    assert sorted(paths) == list(index.expert_layers)
    for layer, path in paths.items():
        reader = ExpertShardReader(
            path, count=index.expert_count, shapes=index.expert_shapes,
            quant_specs=index.expert_quant_specs, dtype=index.dtype, direct=False,
        )
        try:
            owner, staging = aligned_staging(reader.max_span)
            # Every expert, read on its own and in reverse order: no read depends on another.
            for expert in reversed(range(index.expert_count)):
                host: dict[str, Any] = reader.read_host(expert, staging)
                assert f"{expert}.gate_up" in host and f"{expert}.down" in host, (layer, expert)
                assert host[f"{expert}.gate_up"].dtype == torch.uint8
            assert reader.fetches == index.expert_count
            assert owner.numel() > 0
        finally:
            reader.close()


def _headers(folder: Path) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    from safetensors import safe_open

    result = {}
    for path in sorted(folder.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            result[path.name] = {
                key: (handle.get_slice(key).get_dtype(), tuple(handle.get_slice(key).get_shape()))
                for key in handle.keys()
            }
    return result


@pytest.mark.parametrize("double_quant", [False, True])
def test_files_and_index_match_what_the_real_sharder_writes_for_the_same_shape(
    tmp_path: Path, double_quant: bool
) -> None:
    import json

    from benchmarks.harness import b14_synthetic_checkpoint as checkpoint
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
        shard_layout_of,
    )

    shape = synth.SMALL
    weights = checkpoint.write_checkpoint(tmp_path / "src", shape.config(), dtype="float32")
    skeleton = build_meta_skeleton(str(weights), dtype="float32", quant="nf4",
                                   double_quant=double_quant)
    real = tmp_path / "real"
    shard_checkpoint(
        str(weights), str(real), dtype="float32", arch="deepseek_v3",
        layout=shard_layout_of(skeleton), quant="nf4",
        quant_suffixes=quantised_layer_suffixes(skeleton), quant_device="cpu",
        double_quant=double_quant, expert_layout="per_expert")
    ours = tmp_path / "synthetic-moe-small"
    synth.write_store(ours, shape, seed=SEED, double_quant=double_quant)

    assert len(_headers(real)) == 8  # three layers, two experts files, extras and two large weights
    assert _headers(ours) == _headers(real)
    mine, theirs = (json.loads((folder / "index.json").read_text()) for folder in (ours, real))
    # What a synthetic store cannot share: it has no source checkpoint, and no device quantised it.
    for field in ("source_fingerprint", "source_files", "quant_device"):
        assert mine.pop(field) != theirs.pop(field)
    assert mine == theirs


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("double_quant", [False, True])
def test_the_planned_size_matches_the_bytes_on_disk_within_the_recorded_tolerance(
    tmp_path: Path, dtype: str, double_quant: bool
) -> None:
    out = tmp_path / "synthetic-moe-small"
    plan = synth.write_store(out, synth.SMALL, seed=SEED, dtype=dtype, double_quant=double_quant)

    actual = synth.store_bytes(out)
    assert abs(plan.total_bytes - actual) <= synth.SIZE_TOLERANCE * actual
    for name, planned in plan.files.items():
        size = (out / name).stat().st_size
        assert abs(planned - size) <= synth.SIZE_TOLERANCE * size, name
    assert plan.largest_file_bytes == max(plan.files.values())


def test_the_recorded_tolerance_is_one_percent_or_tighter() -> None:
    assert synth.SIZE_TOLERANCE <= 0.01


def test_the_writer_refuses_when_free_space_is_below_the_plan_and_writes_nothing(
    tmp_path: Path,
) -> None:
    out = tmp_path / "parent" / "synthetic-moe-small"
    plan = synth.plan_store(synth.SMALL)

    with pytest.raises(synth.StoreRefusedError, match="free bytes"):
        synth.write_store(out, synth.SMALL, free_bytes=plan.required_free_bytes() - 1)

    assert not out.exists()
    assert not out.parent.exists()


def test_the_writer_writes_when_free_space_equals_the_requirement(tmp_path: Path) -> None:
    out = tmp_path / "synthetic-moe-small"
    plan = synth.plan_store(synth.SMALL)

    synth.write_store(out, synth.SMALL, free_bytes=plan.required_free_bytes())

    assert (out / "index.json").is_file()


def test_the_deepseek_v3_shape_is_planned_and_refused_on_a_200_gb_disk(tmp_path: Path) -> None:
    plan = synth.plan_store(synth.DEEPSEEK_V3, dtype="bfloat16")

    assert 300e9 < plan.total_bytes < 450e9  # 671B weights at about half a byte each
    assert (plan.shape.layers, plan.shape.moe_layers, plan.shape.experts) == (61, 58, 256)
    assert plan.parameters > 600e9
    with pytest.raises(synth.StoreRefusedError, match="free bytes"):
        synth.write_store(tmp_path / "synthetic-v3", synth.DEEPSEEK_V3, dtype="bfloat16",
                          free_bytes=200 * 10**9)
    assert not (tmp_path / "synthetic-v3").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_a_failed_write_leaves_no_index_and_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    from soup_cli.utils import layer_shard

    real_save = layer_shard._atomic_save
    calls: list[str] = []

    def failing(blob: Any, path: str) -> None:
        calls.append(path)
        if len(calls) == 4:
            raise OSError("disk full")
        real_save(blob, path)

    monkeypatch.setattr(layer_shard, "_atomic_save", failing)
    out = tmp_path / "synthetic-moe-small"
    if existing:
        out.mkdir()

    with pytest.raises(OSError, match="disk full"):
        synth.write_store(out, synth.SMALL, seed=SEED)

    assert len(calls) == 4  # three files were written before the failure
    assert not out.exists() or list(out.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        layer_shard.read_shard_index(str(out))


def test_the_writer_refuses_a_destination_that_is_not_empty_or_not_named_synthetic(
    tmp_path: Path,
) -> None:
    full = tmp_path / "synthetic-moe-full"
    full.mkdir()
    (full / "keep.txt").write_text("x")

    with pytest.raises(synth.StoreRefusedError, match="empty"):
        synth.write_store(full, synth.SMALL)
    with pytest.raises(synth.StoreRefusedError, match="synthetic"):
        synth.write_store(tmp_path / "moe-small", synth.SMALL)

    assert (full / "keep.txt").read_text() == "x"
    assert not (tmp_path / "moe-small").exists()


def _digests(folder: Path) -> dict[str, str]:
    import hashlib

    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.iterdir()) if path.is_file()
    }


@pytest.mark.parametrize("double_quant", [False, True])
def test_one_seed_writes_bit_identical_files_and_another_seed_does_not(
    tmp_path: Path, double_quant: bool
) -> None:
    import torch

    first, second, other = (tmp_path / f"synthetic-moe-{name}" for name in ("a", "b", "c"))

    synth.write_store(first, synth.SMALL, seed=SEED, double_quant=double_quant)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # the bytes must not depend on how many threads torch uses
    try:
        synth.write_store(second, synth.SMALL, seed=SEED, double_quant=double_quant)
    finally:
        torch.set_num_threads(threads)
    synth.write_store(other, synth.SMALL, seed=SEED + 1, double_quant=double_quant)

    one, two, three = _digests(first), _digests(second), _digests(other)
    assert one == two
    assert len(one) == 11  # eight safetensors files, index, config, marker
    # Every weight file differs for another seed (the index and config only name the seed).
    for name in one:
        if name.endswith(".safetensors") and name != "extras.safetensors":
            assert one[name] != three[name], name


def _run_args(root: Path, store: Path, **overrides: Any) -> Any:
    import argparse

    (root / "soup.yaml").write_text(RUN_CONFIG, encoding="utf-8")
    values = dict(
        config=str(root / "soup.yaml"), store=str(store), output=str(root / "out"),
        source=None, cache=None, data=None, kind="synthetic", model=None, revision=None,
        generator=None, steps=STEPS, device="cpu", tier="ram", shard_threads=1,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture(scope="module")
def trained(small_store: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from benchmarks.harness import b5_judge, b5_real_moe

    root = tmp_path_factory.mktemp("b5-synth-run")
    before = _digests(small_store)
    b5_real_moe.run(_run_args(root, small_store))
    return {
        "rows": b5_judge.read_journal(root / "out" / "journal.jsonl"),
        "before": before, "after": _digests(small_store),
    }


def test_cpu_training_on_the_synthetic_store_passes_the_judge(trained: dict[str, Any]) -> None:
    from benchmarks.harness import b5_judge

    result = b5_judge.judge([trained["rows"]])

    assert result["passed"] is True, result["failures"]
    assert result["kind"] == "synthetic"
    assert result["steps"] == STEPS


def test_the_run_reads_experts_from_the_store_and_leaves_it_unchanged(
    trained: dict[str, Any],
) -> None:
    steps = [row for row in trained["rows"] if row["event"] == "step"]
    (environment,) = [row for row in trained["rows"] if row["event"] == "environment"]

    assert all(row["expert_fetches"] > 0 and row["expert_bytes_read"] > 0 for row in steps)
    assert environment["quant"] == "nf4" and environment["expert_layout"] == "per_expert"
    assert trained["after"] == trained["before"]


def test_the_journal_signature_and_the_names_say_synthetic(
    trained: dict[str, Any], small_store: Path
) -> None:
    (signature,) = [row for row in trained["rows"] if row["event"] == "signature"]

    assert signature["kind"] == "synthetic"
    assert "synthetic" in signature["model"].lower()
    assert "b5_synth_store" in signature["generator"] and f"seed={SEED}" in signature["generator"]
    assert "synthetic" in small_store.name


def test_every_table_the_writer_prints_is_signed_synthetic() -> None:
    table = synth._plan_table(synth.plan_store(synth.SMALL))

    assert "synthetic" in str(table.title)
    assert "synthetic" in str(table.columns[0].header)


def test_a_store_of_another_dtype_than_the_run_is_refused(tmp_path: Path) -> None:
    from benchmarks.harness import b5_real_moe

    store = tmp_path / "synthetic-moe-bf16"
    synth.write_store(store, synth.SMALL, seed=SEED, dtype="bfloat16")

    with pytest.raises(ValueError, match="dtype"):
        b5_real_moe.run(_run_args(tmp_path, store))
    assert not (tmp_path / "out").exists()


def test_a_folder_without_the_synthetic_marker_is_refused_as_a_store(tmp_path: Path) -> None:
    from benchmarks.harness import b5_real_moe

    store = tmp_path / "synthetic-moe-small"
    synth.write_store(store, synth.SMALL, seed=SEED)
    (store / synth.MARKER_NAME).unlink()

    with pytest.raises(ValueError, match="marker"):
        b5_real_moe.run(_run_args(tmp_path, store))


def test_the_deepseek_v3_dimensions_give_the_tensor_shapes_of_the_real_model(
    tmp_path: Path,
) -> None:
    """Ticket 06 writes this shape. Its layer shapes must be those transformers builds, so the
    comparison runs on the model's meta skeleton with the V3 dimensions and four layers."""
    import dataclasses

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, decoder_owner

    shape = dataclasses.replace(synth.DEEPSEEK_V3, layers=4, moe_layers=2)
    shape.config().save_pretrained(str(tmp_path))
    model = build_meta_skeleton(str(tmp_path), dtype="bfloat16")

    for number, layer in enumerate(decoder_owner(model).layers):
        built = {name: tuple(tensor.shape) for name, tensor in
                 [*layer.named_parameters(), *layer.named_buffers()]}
        fused = {name: built.pop(name) for name in list(built) if name.startswith("mlp.experts.")}
        mine = {name: dims for name, dims, _role in synth._layer_parts(shape, number)}
        assert mine == built, number
        if shape.is_moe(number):
            gate_up, down = synth._expert_dims(shape)["gate_up"], synth._expert_dims(shape)["down"]
            assert fused == {"mlp.experts.gate_up_proj": (shape.experts, *gate_up),
                             "mlp.experts.down_proj": (shape.experts, *down)}
        else:
            assert fused == {}
    assert tuple(model.get_input_embeddings().weight.shape) == (shape.vocab, shape.hidden)


@pytest.fixture(scope="module")
def cold_run(small_store: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from benchmarks.harness import b5_judge, b5_real_moe

    root = tmp_path_factory.mktemp("b5-synth-cold-run")
    b5_real_moe.run(_run_args(root, small_store, cold=True))
    return {"rows": b5_judge.read_journal(root / "out" / "journal.jsonl")}


def _event(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    (row,) = [row for row in rows if row["event"] == name]
    return row


def test_a_cold_run_states_the_shape_of_the_store(
    cold_run: dict[str, Any], small_store: Path
) -> None:
    shape = _event(cold_run["rows"], "shape")

    assert (shape["layers"], shape["moe_layers"], shape["experts"]) == (3, 2, 8)
    assert (shape["hidden"], shape["expert_size"]) == (64, 64)
    assert shape["parameters"] == synth.total_params(synth.SMALL)
    assert shape["store_bytes"] == synth.store_bytes(small_store)


def test_a_cold_run_drops_the_cache_right_before_training_and_says_how(
    cold_run: dict[str, Any],
) -> None:
    rows = cold_run["rows"]
    drop = _event(rows, "cold_cache")
    start = _event(rows, "training_start")

    assert drop["method"] in ("drop_caches", "fadvise", "none")
    assert drop["resident_after_bytes"] <= drop["resident_before_bytes"]
    assert drop["store_bytes"] > 0
    assert 0 <= start["unix"] - drop["unix"] < 600


def test_every_step_carries_the_disk_read_counter_field(cold_run: dict[str, Any]) -> None:
    steps = [row for row in cold_run["rows"] if row["event"] == "step"]

    assert steps and all("disk_read_bytes" in row for row in steps)
    assert all(row["disk_read_bytes"] is None or row["disk_read_bytes"] >= 0 for row in steps)


def test_a_run_without_cold_writes_no_cache_drop(trained: dict[str, Any]) -> None:
    assert not [row for row in trained["rows"] if row["event"] == "cold_cache"]


def test_the_write_command_records_plan_actual_size_free_space_and_time(tmp_path: Path) -> None:
    import json

    record = tmp_path / "record.json"
    out = tmp_path / "synthetic-store"

    code = synth.main(["write", "--shape", "small", "--dtype", "float32", "--seed", "3",
                       "--out", str(out), "--record", str(record)])
    data = json.loads(record.read_text(encoding="utf-8"))

    assert code == 0
    assert data["planned_bytes"] == synth.plan_store(synth.SMALL, dtype="float32",
                                                     seed=3).total_bytes
    assert data["actual_bytes"] == synth.store_bytes(out)
    assert data["required_free_bytes"] > data["planned_bytes"]
    assert data["free_before_bytes"] >= data["required_free_bytes"]
    assert data["seconds"] > 0 and data["signature"] == "synthetic"
    assert data["generator"].startswith("b5_synth_store/1 seed=3")


def test_a_refused_write_records_the_refusal_and_writes_nothing(tmp_path: Path) -> None:
    import json

    record = tmp_path / "record.json"
    out = tmp_path / "synthetic-store"

    code = synth.main(["write", "--shape", "small", "--dtype", "float32", "--out", str(out),
                       "--record", str(record), "--assume-free-bytes", "1000"])
    data = json.loads(record.read_text(encoding="utf-8"))

    assert code == 3
    assert data["refused"] is True and data["free_before_bytes"] == 1000
    assert data["free_before_bytes"] < data["required_free_bytes"]
    assert not out.exists()


def test_the_environment_event_records_the_host_ram_limit_reading(cold_run: dict[str, Any]) -> None:
    environment = _event(cold_run["rows"], "environment")

    assert set(environment["ram_limit"]) >= {"limit_bytes", "swap_max_bytes", "source"}


# ---- resume: a write that was killed half way is finished, not started again -------------------
def _killed_halfway(folder: Path, files_kept: int) -> list[str]:
    """Leave only the first ``files_kept`` safetensors files, as a killed write does."""
    names = sorted(item.name for item in folder.glob("*.safetensors"))
    (folder / "index.json").unlink()
    for name in names[files_kept:]:
        (folder / name).unlink()
    return names[:files_kept]


def test_a_resumed_write_equals_a_fresh_write_of_the_same_seed(tmp_path: Path) -> None:
    fresh = tmp_path / "synthetic-fresh"
    synth.write_store(fresh, synth.SMALL, seed=SEED)
    resumed = tmp_path / "synthetic-resumed"
    synth.write_store(resumed, synth.SMALL, seed=SEED)
    _killed_halfway(resumed, 4)

    synth.write_store(resumed, synth.SMALL, seed=SEED, resume=True)

    assert _digests(resumed) == _digests(fresh)


def test_a_resumed_write_does_not_touch_the_files_that_are_already_complete(
    tmp_path: Path,
) -> None:
    out = tmp_path / "synthetic-resumed"
    synth.write_store(out, synth.SMALL, seed=SEED)
    kept = _killed_halfway(out, 4)
    before = {name: (out / name).stat().st_mtime_ns for name in kept}

    plan = synth.write_store(out, synth.SMALL, seed=SEED, resume=True)

    assert {name: (out / name).stat().st_mtime_ns for name in kept} == before
    assert plan.skipped_files == 4


def test_a_resumed_write_rewrites_a_file_of_the_wrong_size(tmp_path: Path) -> None:
    fresh = tmp_path / "synthetic-fresh"
    synth.write_store(fresh, synth.SMALL, seed=SEED)
    out = tmp_path / "synthetic-resumed"
    synth.write_store(out, synth.SMALL, seed=SEED)
    kept = _killed_halfway(out, 4)
    broken = out / kept[-1]
    broken.write_bytes(broken.read_bytes()[:-8])

    plan = synth.write_store(out, synth.SMALL, seed=SEED, resume=True)

    assert _digests(out) == _digests(fresh)
    assert plan.skipped_files == 3


def test_a_resumed_write_refuses_a_folder_of_another_seed(tmp_path: Path) -> None:
    out = tmp_path / "synthetic-resumed"
    synth.write_store(out, synth.SMALL, seed=SEED)
    _killed_halfway(out, 4)

    with pytest.raises(synth.StoreRefusedError, match="seed"):
        synth.write_store(out, synth.SMALL, seed=SEED + 1, resume=True)


def test_a_resumed_write_refuses_a_folder_that_is_not_a_synthetic_store(tmp_path: Path) -> None:
    out = tmp_path / "synthetic-other"
    out.mkdir()
    (out / "something.txt").write_text("x")

    with pytest.raises(synth.StoreRefusedError, match="marker"):
        synth.write_store(out, synth.SMALL, seed=SEED, resume=True)


def test_a_failed_resume_keeps_the_files_it_found(tmp_path: Path) -> None:
    out = tmp_path / "synthetic-resumed"
    synth.write_store(out, synth.SMALL, seed=SEED)
    kept = _killed_halfway(out, 4)

    def stop(name: str, done: int, total: int) -> None:
        if done > 5:
            raise RuntimeError("stopped")

    with pytest.raises(RuntimeError):
        synth.write_store(out, synth.SMALL, seed=SEED, resume=True, on_file=stop)

    assert all((out / name).is_file() for name in kept)
    assert not (out / "index.json").exists()


def test_a_resume_needs_free_space_only_for_the_files_still_to_write(tmp_path: Path) -> None:
    out = tmp_path / "synthetic-resumed"
    plan = synth.write_store(out, synth.SMALL, seed=SEED)
    _killed_halfway(out, len(plan.files) - 1)
    last = max(plan.files.values())

    synth.write_store(out, synth.SMALL, seed=SEED, resume=True,
                      free_bytes=int(plan.files[sorted(plan.files)[-1]] * synth.DISK_MARGIN)
                      + last + 1)

    assert (out / "index.json").is_file()


def test_the_resume_flag_records_what_was_skipped_and_the_free_space_it_derives(
    tmp_path: Path,
) -> None:
    import json

    out = tmp_path / "synthetic-resumed"
    synth.write_store(out, synth.SMALL, seed=3, dtype="float32")
    kept = _killed_halfway(out, 4)
    record = tmp_path / "record.json"

    code = synth.main(["write", "--shape", "small", "--dtype", "float32", "--seed", "3",
                       "--out", str(out), "--record", str(record), "--resume"])
    data = json.loads(record.read_text(encoding="utf-8"))

    assert code == 0
    assert data["resumed"] is True and data["skipped_files"] == len(kept)
    assert data["actual_bytes"] == synth.store_bytes(out)
    # The free space before the first segment: what is free now plus what that segment wrote.
    assert data["free_before_bytes"] == data["free_now_bytes"] + data["existing_bytes"]
    assert data["free_before_derived"] is True
