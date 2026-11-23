from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
    from torch.nn.attention import SDPBackend

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks/mbr-moonlight-nondet-20261123"
TARGETS = ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")


def projection_name(layer: int, target: str) -> str:
    return f"layers.{layer}.self_attn.{target}"


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(BENCHMARK.parents[1] / "src"))
    monkeypatch.syspath_prepend(str(BENCHMARK.parents[1]))
    monkeypatch.syspath_prepend(str(BENCHMARK))
    return BENCHMARK


@pytest.mark.parametrize(
    ("left", "right", "equal", "n_diff", "numel", "max_abs_diff", "max_abs", "sum_diff",
     "rel_max_diff", "rel_sum_diff"),
    [
        ([1, 2], [1, 3], False, 1, 2, 1, 2, -1, 1 / 2, 1 / 3),
        ([1, 2], [1, 2], True, 0, 2, 0, 2, 0, 0, 0),
        (0, 0, True, 0, 1, 0, 0, 0, 0, 0),
        ([], [], True, 0, 0, 0, 0, 0, 0, 0),
    ],
)
def test_tensor_pair_diff(
    benchmark: Path, left: int | list[int], right: int | list[int], equal: bool, n_diff: int,
    numel: int, max_abs_diff: float, max_abs: float, sum_diff: float, rel_max_diff: float,
    rel_sum_diff: float,
) -> None:
    import torch
    from nondet_analysis import tensor_pair_diff

    assert tensor_pair_diff(
        torch.tensor(left, dtype=torch.bfloat16), torch.tensor(right, dtype=torch.bfloat16)
    ) == pytest.approx({
        "equal": equal,
        "bitwise_equal": equal,
        "n_diff": n_diff,
        "numel": numel,
        "max_abs_diff": max_abs_diff,
        "max_abs": max_abs,
        "sum_diff": sum_diff,
        "rel_max_diff": rel_max_diff,
        "rel_sum_diff": rel_sum_diff,
    })


@pytest.mark.parametrize(
    ("case", "value_equal", "bitwise_equal"),
    [("signed_zero", True, False), ("nan", False, True), ("dtype", True, False)],
)
def test_tensor_pair_diff_separates_value_and_byte_equality(
    benchmark: Path, case: str, value_equal: bool, bitwise_equal: bool,
) -> None:
    import torch
    from nondet_analysis import tensor_pair_diff

    if case == "signed_zero":
        left, right = torch.tensor([0.0]), torch.tensor([-0.0])
    elif case == "nan":
        left = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
        right = left.clone()
    else:
        left, right = torch.tensor([1.5], dtype=torch.bfloat16), torch.tensor([1.5])
    diff = tensor_pair_diff(left, right)
    assert diff["equal"] is value_equal
    assert diff["bitwise_equal"] is bitwise_equal
    if value_equal:
        assert diff["n_diff"] == diff["max_abs_diff"] == 0


def test_grad_breakdown_matches_moonlight_pattern(benchmark: Path) -> None:
    from nondet_analysis import grad_breakdown

    flags = {
        f"base_model.model.model.layers.{layer}.self_attn.{proj}.lora_{ab}.default.weight":
        ab == "B" and (layer < 26 or proj == "q_proj")
        for layer in range(27)
        for proj in ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
        for ab in ("A", "B")
    }
    result = grad_breakdown(flags, flags)
    assert result["totals"] == {"differing": 105, "total": 216}
    assert result["per_ab"] == {
        "A": {"differing": 0, "total": 108},
        "B": {"differing": 105, "total": 108},
    }
    assert result["per_layer"][25] == {"differing": 4, "total": 8}
    assert result["per_layer"][26] == {"differing": 1, "total": 8}
    assert result["per_projection"]["q_proj"] == {"differing": 27, "total": 54}
    assert result["first_differing_layer_in_backward_order"] == 26
    assert result["first_differing_names"] == [
        "base_model.model.model.layers.26.self_attn.q_proj.lora_B.default.weight"
    ]


def test_grad_breakdown_keeps_unparsed_separate(benchmark: Path) -> None:
    from nondet_analysis import grad_breakdown

    name = "base_model.model.model.layers.2.self_attn.o_proj.lora_A.default.weight"
    result = grad_breakdown([name, "unknown"], {"unknown": True})
    assert result["totals"] == {"differing": 0, "total": 1}
    assert result["unparsed"] == {"differing": 1, "total": 1}
    assert result["first_differing_layer_in_backward_order"] is None
    assert result["first_differing_names"] == []


def test_row_record_preserves_metadata(benchmark: Path) -> None:
    from nondet_analysis import row_record, window_label

    metadata = {
        "model": "Moonlight-16B-A3B",
        "model_revision": "2a9a177f",
        "card": "RTX 3050 Laptop",
        "window": window_label(1, 512),
        "arm": "A0",
        "mode": "bits",
        "torch_version": "2.14.1+cu130",
        "attention_backend": "default",
        "tensor_file": "A0-k1-s512.pt",
    }
    assert metadata["window"] == "k=1,S=512"
    assert row_record(**metadata, extra={"power_ac_online": 1}) == {
        **metadata, "power_ac_online": 1,
    }
    for key in metadata:
        with pytest.raises(ValueError, match=key):
            row_record(**metadata, extra={key: "collision"})


@pytest.mark.parametrize("unusable_seq", [False, True])
def test_probe_reports_two_backward_passes_and_backend_errors(
    benchmark: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unusable_seq: bool,
) -> None:
    import probe_sdpa
    import torch
    from torch.nn import attention, functional

    real_randn = torch.randn
    real_generator = torch.Generator
    randn_devices = []

    def cpu_randn(*args: Any, device: str, **kwargs: Any) -> torch.Tensor:
        randn_devices.append(device)
        return real_randn(*args, device="cpu", **kwargs)

    active_backend = None
    seen = []
    backwards = []

    @contextmanager
    def backend_scope(backend: SDPBackend) -> Iterator[None]:
        nonlocal active_backend
        active_backend = backend
        try:
            yield
        finally:
            active_backend = None

    def fake_sdpa(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, is_causal: bool,
    ) -> torch.Tensor:
        assert is_causal
        seq = query.shape[-2]
        assert tuple(query.shape) == tuple(key.shape) == (1, 16, seq, 192)
        assert tuple(value.shape) == (1, 16, seq, 128)
        assert query.dtype == torch.bfloat16
        warnings.warn("sample attention warning", UserWarning)
        if active_backend == attention.SDPBackend.FLASH_ATTENTION:
            raise RuntimeError("No available kernel")
        assert active_backend == attention.SDPBackend.MATH
        if seq == 512:
            raise RuntimeError("Math unavailable for this sequence")
        seen.append((query, key, value))
        output = (query[..., :128] + key[..., :128]) + value

        def note_backward(_grad: torch.Tensor) -> None:
            backwards.append(active_backend)

        output.register_hook(note_backward)
        return output

    monkeypatch.setattr(torch, "Generator", lambda *, device: real_generator(device="cpu"))
    monkeypatch.setattr(torch, "randn", cpu_randn)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    workspace_configs = []

    def get_device_name() -> str:
        workspace_configs.append(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        return "test card"

    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)
    deterministic = []
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms",
        lambda enabled, *, warn_only: deterministic.append((enabled, warn_only)),
    )
    monkeypatch.setattr(attention, "sdpa_kernel", backend_scope)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", fake_sdpa)

    assert probe_sdpa.main([
        "--seq", "64,512" if unusable_seq else "64", "--backends", "math,flash",
        "--out", str(tmp_path),
    ]) == int(unusable_seq)
    assert workspace_configs == [":4096:8"]
    assert deterministic == [(True, True)]
    assert randn_devices and set(randn_devices) == {"cuda"}
    assert backwards == [attention.SDPBackend.MATH] * 2
    assert len(seen) == 2
    for left, right in zip(*seen):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr()
    results = json.loads((tmp_path / "probe.json").read_text())
    assert results["64"]["flash"] == {"error": "RuntimeError: No available kernel"}
    assert set(results["64"]["math"]) == {"out", "dq", "dk", "dv"}
    assert all(row["equal"] for row in results["64"]["math"].values())
    assert all(row["bitwise_equal"] for row in results["64"]["math"].values())
    assert "| 64 | math | dq | True | 0 |" in (tmp_path / "probe.md").read_text()
    warning_text = (tmp_path / "warnings.txt").read_text()
    assert "seq=64 backend=math: sample attention warning" in warning_text
    assert "seq=64 backend=flash: sample attention warning" in warning_text
    if unusable_seq:
        assert results["512"] == {
            "math": {"error": "RuntimeError: Math unavailable for this sequence"},
            "flash": {"error": "RuntimeError: No available kernel"},
        }
        assert "seq=512 backend=math: sample attention warning" in warning_text


@pytest.mark.parametrize("case", [
    "shared", "other_consumer", "strict", "missing_projection", "missing_call", "missing_grad",
    "missing_file", "oom",
])
def test_run_capture_records_instrumented_run_and_strict_failure(
    benchmark: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    import run_capture
    import torch
    from torch import nn
    from torch.nn import functional

    from benchmarks.harness import mbr_measure
    from soup_cli.utils import l2l_step, layer_stream_runtime

    strict = case == "strict"
    expected_error = {
        "missing_projection": "108 projection",
        "missing_call": "multiple of 27",
        "missing_grad": "sha_dv",
        "missing_file": "sdpa_call0_dq.pt",
    }.get(case)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({} if strict else {
        "_commit_hash": "test-revision",
    }))
    out, warnings_file = tmp_path / "out", tmp_path / "logs/warnings.txt"
    events = []
    model = nn.Module()
    model.layers = nn.ModuleList([
        nn.ModuleDict({"self_attn": nn.ModuleDict({target: nn.Identity() for target in TARGETS})})
        for _ in range(27)
    ])

    def fake_build() -> tuple[torch.nn.Module, object]:
        events.append("build")
        return model, object()

    def fake_sdpa(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        return query + 2 * key + 4 * value

    def backend_flags() -> tuple[bool, bool, bool, bool]:
        cuda = torch.backends.cuda
        return (
            cuda.math_sdp_enabled(), cuda.flash_sdp_enabled(),
            cuda.mem_efficient_sdp_enabled(), cuda.cudnn_sdp_enabled(),
        )

    original_flags = backend_flags()

    def fake_step(_stepper: Any, step: int) -> None:
        dtype = torch.bfloat16 if case == "shared" else torch.float32
        query = torch.zeros((1, 1, 1, 2), dtype=dtype, requires_grad=True)
        key, value = (query.detach().clone().requires_grad_() for _ in range(2))
        outputs = [
            functional.scaled_dot_product_attention(
                query, key, value.detach() if case == "missing_grad" and index == 0 else value,
            )
            for index in range(26 if case == "missing_call" and step == 1 else 27)
        ]
        projections = [
            module(query.clone().float())
            for layer_idx, layer in enumerate(model.layers)
            for target, module in layer["self_attn"].items()
            if not (case == "missing_projection" and layer_idx == 26 and target == "q_proj")
        ]
        other = 10 * query if case == "other_consumer" else 0
        (sum(outputs) + (step + 1) * sum(projections) + other).sum().backward()

    def fake_run(args: argparse.Namespace) -> dict[str, Any]:
        events.append("run entered")
        assert backend_flags() == (original_flags if strict else (True, False, False, False))
        assert (args.kind, args.deterministic, args.math_attention) == ("bits", True, False)
        assert (args.tokenizer, args.profile, args.steps, args.lora_r) == (None, False, 3, 16)
        assert tuple(args.lora_targets) == TARGETS
        torch.use_deterministic_algorithms(True, warn_only=True)
        layer_stream_runtime.build_streamed_model()
        for _ in range(2):
            warnings.warn("sample warning", UserWarning)
        if strict:
            raise RuntimeError("sample_nondeterministic_op has no deterministic implementation")
        if case == "oom":
            return {"status": "OOM", "mode": "deterministic", "error": "out of memory"}
        with torch.no_grad():
            functional.scaled_dot_product_attention(*(torch.zeros(1) for _ in range(3)))
        for step in range(args.steps):
            l2l_step.L2LStep.run(None, step)
        return {"status": "ok", "mode": "deterministic", "tensor_file": "A0-k1-s512.pt"}

    monkeypatch.setattr(mbr_measure, "run_one", fake_run)
    monkeypatch.setattr(layer_stream_runtime, "build_streamed_model", fake_build)
    monkeypatch.setattr(l2l_step.L2LStep, "run", fake_step)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms",
        lambda enabled, *, warn_only=False: events.append((enabled, warn_only)),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test card")
    if case == "missing_file":
        real_save = torch.save

        def save_without_dq(tensor: torch.Tensor, path: Path) -> None:
            if path.name != "sdpa_call0_dq.pt":
                real_save(tensor, path)

        monkeypatch.setattr(torch, "save", save_without_dq)
    real_read_text = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if str(path) == "/sys/class/power_supply/ACAD/online":
            return "1\n"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    argv = [
        "--source", str(source), "--cache", str(tmp_path / "cache"), "--out", str(out),
        "--arm", "A0", "--k", "1", "--seq", "512", "--lora-targets", ",".join(TARGETS),
        "--backend", "default" if strict else "math", "--deterministic",
        "--warnings-file", str(warnings_file), "--tag", "test",
    ]
    if case != "oom":
        argv.append("--capture")
    if strict:
        with pytest.raises(RuntimeError, match="sample_nondeterministic_op"):
            run_capture.main([*argv, "--strict-determinism"])
        assert events == ["run entered", (True, True), (True, False), "build"]
    else:
        assert run_capture.main(argv) == (1 if case == "oom" or expected_error else 0)
        assert events == ["run entered", (True, True), "build"]

    assert backend_flags() == original_flags
    assert functional.scaled_dot_product_attention is fake_sdpa
    assert layer_stream_runtime.build_streamed_model is fake_build
    assert l2l_step.L2LStep.run is fake_step
    assert warnings_file.read_text() == "sample warning\n"
    harness = json.loads((out / "rows.jsonl").read_text())
    row = json.loads((out / "nondet_rows.jsonl").read_text())
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == row
    assert row["attention_backend"] == ("default" if strict else "math")
    assert row["window"] == "k=1,S=512"
    assert row["model_revision"] == ("unknown" if strict else "test-revision")
    assert row["card"] == "test card"
    assert row["power_ac_online"] == harness["power_ac_online"] == 1
    status = {"strict": "CRASH", "oom": "OOM"}.get(case, "incomplete" if expected_error else "ok")
    assert row["status"] == harness["status"] == status
    if strict:
        assert "sample_nondeterministic_op" in row["error"]
        return
    if expected_error is not None:
        assert expected_error in row["error"]
        return
    if case == "oom":
        assert row["error"] == "out of memory"
        return

    capture = out / "capture/test"
    output_grads = json.loads((capture / "output_grads.json").read_text())
    assert set(output_grads) == {
        projection_name(i, target) for i in range(27) for target in TARGETS
    }
    assert output_grads[projection_name(26, "q_proj")] == [
        {
            "sha256": hashlib.sha256(struct.pack("=2f", value, value)).hexdigest(),
            "max_abs": value, "numel": 2, "step": step,
        }
        for step, value in enumerate((1.0, 2.0, 3.0))
    ]
    digests = {}
    for name, value, bits in (("dq", 1.0, 0x3F80), ("dk", 2.0, 0x4000), ("dv", 4.0, 0x4080)):
        tensor = torch.load(
            capture / f"sdpa_call0_{name}.pt", map_location="cpu", weights_only=True,
        )
        if case == "shared":
            assert tensor.dtype == torch.bfloat16
            expected = struct.pack("=2H", bits, bits)
        else:
            assert tensor.dtype == torch.float32
            expected = struct.pack("=2f", value, value)
        assert tensor.view(torch.uint8).flatten().tolist() == list(expected)
        digests[f"sha_{name}"] = hashlib.sha256(expected).hexdigest()
    assert json.loads((capture / "sdpa_calls.json").read_text()) == [
        {"call": index, "q_shape": [1, 1, 1, 2], **digests} for index in range(81)
    ]


@pytest.mark.parametrize(("config", "revision", "expected"), [
    ({"_commit_hash": "local-hash"}, "pinned-hash", "pinned-hash"),
    (None, None, "unknown"),
])
def test_run_capture_model_revision(
    benchmark: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    config: dict[str, str] | None, revision: str | None, expected: str,
) -> None:
    import run_capture

    from benchmarks.harness import mbr_measure

    source = tmp_path / "source"
    source.mkdir()
    if config is not None:
        (source / "config.json").write_text(json.dumps(config))
    out = tmp_path / "out"

    def fake_run(_args: argparse.Namespace) -> dict[str, str]:
        return {"status": "ok", "mode": "deterministic"}

    monkeypatch.setattr(mbr_measure, "run_one", fake_run)
    argv = [
        "--source", str(source), "--cache", str(tmp_path / "cache"), "--out", str(out),
        "--arm", "A0", "--k", "1", "--seq", "512", "--deterministic", "--device", "cpu",
    ]
    if revision is not None:
        argv.extend(("--model-revision", revision))
    assert run_capture.main(argv) == 0
    assert json.loads((out / "nondet_rows.jsonl").read_text())["model_revision"] == expected


@pytest.mark.parametrize("existing", ["nondet_rows.jsonl", "capture/test/sdpa_call0_dq.pt"])
def test_run_capture_rejects_existing_outputs(
    benchmark: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], existing: str,
) -> None:
    import run_capture

    from benchmarks.harness import mbr_measure

    def forbidden_run(_args: argparse.Namespace) -> dict[str, str]:
        pytest.fail("the benchmark must not launch with existing outputs")

    monkeypatch.setattr(mbr_measure, "run_one", forbidden_run)
    existing_path = tmp_path / existing
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_bytes(b"prior evidence")
    with pytest.raises(SystemExit) as error:
        run_capture.main([
            "--source", str(tmp_path / "source"), "--cache", str(tmp_path / "cache"),
            "--out", str(tmp_path), "--arm", "A0", "--k", "1", "--seq", "512",
            "--deterministic", "--tag", "test", "--device", "cpu",
        ])
    assert error.value.code != 0
    assert "already exists" in capsys.readouterr().err
    assert existing_path.read_bytes() == b"prior evidence"
    assert not (tmp_path / "warnings.txt").exists()


def write_capture(path: Path, *, right: bool, legacy: bool = False) -> None:
    import torch

    path.mkdir()
    projections: dict[str, Any] = {}
    for layer in range(27):
        for target in TARGETS:
            digest = "same"
            if layer == 26 and target == "q_proj":
                digest = "digest-b" if right else "digest-a"
            final = {"sha256": digest, "max_abs": 1.0, "numel": 2}
            projections[projection_name(layer, target)] = final if legacy else [
                {"sha256": "same", "max_abs": 1.0, "numel": 2, "step": 0},
                {
                    "sha256": "earlier-b" if right and digest != "same" else "same",
                    "max_abs": 1.0, "numel": 2, "step": 1,
                },
                {**final, "step": 2},
            ]
    (path / "output_grads.json").write_text(json.dumps(projections))
    calls = [
        {
            "call": index, "q_shape": [1, 1, 1, 2], "sha_dq": "digest-b" if right else "digest-a",
            "sha_dk": "same", "sha_dv": "same",
        }
        for index in range(81)
    ]
    (path / "sdpa_calls.json").write_text(json.dumps(calls))
    for grad in ("dq", "dk", "dv"):
        torch.save(torch.zeros(1, 1, 1, 2), path / f"sdpa_call0_{grad}.pt")


@pytest.mark.parametrize("legacy", [False, True])
def test_compare_runs_reports_every_pair_and_capture_key(
    benchmark: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], legacy: bool,
) -> None:
    import compare_runs
    import torch

    first = "base_model.model.model.layers.26.self_attn.q_proj.lora_B.default.weight"
    second = "base_model.model.model.layers.25.self_attn.o_proj.lora_A.default.weight"
    paths = [tmp_path / f"repeat{index}.pt" for index in range(1, 4)]
    for path, value in zip(paths, (1, 2, 2)):
        torch.save({
            "loss": torch.tensor([3.0], dtype=torch.float64),
            "grads": {first: torch.tensor([value]), second: torch.tensor([0])},
        }, path)
    left, right = tmp_path / "left", tmp_path / "right"
    write_capture(left, right=False, legacy=legacy)
    write_capture(right, right=True, legacy=legacy)

    out = tmp_path / "out"
    assert compare_runs.main([
        *map(str, paths), "--captures", str(left), str(right), "--out", str(out),
    ]) == 0
    report = (out / "compare.md").read_text()
    assert "| repeat1.pt | repeat2.pt | True | 1 | 2 | 26 |" in report
    assert "| repeat1.pt | repeat3.pt | True | 1 | 2 | 26 |" in report
    assert "| repeat2.pt | repeat3.pt | True | 0 | 2 | none |" in report
    assert "| 26 | 1 | 1 |" in report
    assert "| q_proj | 1 | 1 |" in report
    assert first in report
    assert (
        f"| layers.26.self_attn.q_proj | digest-a | digest-b | different | "
        f"{'1/1' if legacy else '2/3'} |" in report
    )
    assert "| 0 | sha_dq | digest-a | digest-b | different |" in report
    assert "| 0 | sha_dk | same | same | equal |" in report
    assert "1/2" in capsys.readouterr().out
    assert compare_runs.main(list(map(str, paths))) == 0
    assert "| repeat1.pt | repeat3.pt | True | 1 | 2 | 26 |" in capsys.readouterr().out


@pytest.mark.parametrize(("case", "reason"), [
    ("missing_projection", "108 projection"),
    ("empty_occurrences", "projection gradient occurrence"),
    ("different_occurrences", "occurrence counts"),
    ("missing_digest", "sha_dk"),
    ("different_counts", "call counts"),
    ("different_shape", "q_shape"),
    ("missing_tensor", "sdpa_call0_dv.pt"),
    ("missing_json", "output_grads.json"),
])
def test_compare_runs_refuses_incomplete_captures(
    benchmark: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str, reason: str,
) -> None:
    import compare_runs
    import torch

    left, right = tmp_path / "left", tmp_path / "right"
    write_capture(left, right=False)
    write_capture(right, right=True)
    projection_path = right / "output_grads.json"
    call_path = right / "sdpa_calls.json"
    projections = json.loads(projection_path.read_text())
    calls = json.loads(call_path.read_text())
    name = projection_name(26, "q_proj")
    if case == "missing_projection":
        del projections[name]
    elif case == "empty_occurrences":
        projections[name] = []
    elif case == "different_occurrences":
        projections[name].insert(0, projections[name][0])
    elif case == "missing_digest":
        calls[13]["sha_dk"] = None
    elif case == "different_counts":
        del calls[54:]
    elif case == "different_shape":
        calls[52]["q_shape"][-1] += 1
    projection_path.write_text(json.dumps(projections))
    call_path.write_text(json.dumps(calls))
    if case == "missing_tensor":
        (right / "sdpa_call0_dv.pt").unlink()
    elif case == "missing_json":
        projection_path.unlink()
    tensors = [tmp_path / f"run{i}.pt" for i in range(2)]
    for tensor in tensors:
        torch.save({"loss": torch.tensor([1.0]), "grads": {}}, tensor)
    out = tmp_path / "out"
    assert compare_runs.main([
        *map(str, tensors), "--captures", str(left), str(right), "--out", str(out),
    ]) != 0
    report = (out / "compare.md").read_text()
    assert "incomplete" in report and reason in report
    assert "incomplete" in capsys.readouterr().out
    assert "| Projection |" not in report


def test_compare_runs_counts_bitwise_lora_differences(
    benchmark: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import compare_runs
    import torch

    files = [tmp_path / "left.pt", tmp_path / "right.pt"]
    name = f"{projection_name(26, 'q_proj')}.lora_B.default.weight"
    for path, zero in zip(files, (0.0, -0.0)):
        torch.save({"loss": torch.tensor([1.0]), "grads": {name: torch.tensor([zero])}}, path)
    assert compare_runs.main(list(map(str, files))) == 0
    report = capsys.readouterr().out
    assert "| left.pt | right.pt | True | 1 | 1 | 26 |" in report
    assert f"| {name} | True | False | 0 | 1 |" in report


@pytest.mark.parametrize("module", ["probe_sdpa", "run_capture", "compare_runs"])
def test_python_cli_help(
    benchmark: Path, capsys: pytest.CaptureFixture[str], module: str,
) -> None:
    with pytest.raises(SystemExit) as error:
        importlib.import_module(module).main(["--help"])
    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "linux", reason="the power and sleep controls are Linux-only")
@pytest.mark.parametrize("case", ["unplugged", "collision", "run", "strict_crash"])
def test_matrix_checks_power_and_destinations_then_logs_complete_plan(
    benchmark: Path, tmp_path: Path, case: str,
) -> None:
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    out, trace = tmp_path / "out", tmp_path / "calls"
    if case == "collision":
        (out / "A0-strict-k1-rep1-s512").mkdir(parents=True)

    commands = {
        "cat": """#!/bin/bash
if [[ "$1" == /sys/class/power_supply/ACAD/online ]]; then
  printf '%s\\n' "$TEST_AC_STATE"
else
  exit 10
fi
""",
        "systemd-inhibit": """#!/bin/bash
[[ "$1" == --what=sleep && "$2" == '--why=mbr nondet ticket 12' ]] || exit 11
shift 2
TEST_INHIBITED=1 exec "$@"
""",
        "python": """#!/bin/bash
[[ "$TEST_INHIBITED" == 1 && "$PYTHONWARNINGS" == always ]] || exit 12
case "$1" in
  -c) printf '%s\\n' '/fake/nvidia/cu13' ;;
  -u)
    shift
    [[ "$CUDA_HOME" == /fake/nvidia/cu13 ]] || exit 13
    printf '%s\\n' "$*" >> "$TEST_TRACE"
    if [[ " $* " == *" --strict-determinism "* ]]; then
      if [[ "$TEST_STRICT_LOG" == op ]]; then
        message="RuntimeError: index_add_cuda_ does not have a deterministic implementation,"
        message+=" but you set 'torch.use_deterministic_algorithms(True)'."
        printf '%s\\n' "$message"
      else
        printf '%s\\n' "RuntimeError: CUDA error: an illegal memory access was encountered"
      fi
      exit 23
    fi
    ;;
  *) exit 14 ;;
esac
exit 0
""",
    }
    for name, text in commands.items():
        path = bin_dir / name
        path.write_text(text)
        path.chmod(0o755)

    help_result = subprocess.run(
        ["bash", str(benchmark / "run_matrix.sh"), "--help"],
        check=True, capture_output=True, text=True,
    )
    assert "Usage:" in help_result.stdout
    result = subprocess.run(
        ["bash", str(benchmark / "run_matrix.sh")], check=False, capture_output=True, text=True,
        env={
            **os.environ, "VENV": str(venv), "OUT": str(out),
            "MOONLIGHT_ROOT": str(tmp_path / "moonlight"), "TEST_TRACE": str(trace),
            "TEST_AC_STATE": "0" if case == "unplugged" else "1",
            "TEST_STRICT_LOG": "crash" if case == "strict_crash" else "op",
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )
    if case != "run" and case != "strict_crash":
        assert result.returncode == 1, result.stderr
        assert ("AC" if case == "unplugged" else "exists") in result.stderr
        assert not trace.exists()
        return

    planned = [
        "probe", "A0-default-k1-rep1-s512", "A0-default-k1-rep2-s512", "A0-default-k1-rep3-s512",
        "A0-math-k1-rep1-s512", "A0-math-k1-rep2-s512",
        "A0-default-k1-rep1-s64", "A0-default-k1-rep2-s64", "A0-strict-k1-rep1-s512",
    ]
    calls = trace.read_text().splitlines()
    assert len(calls) == len(planned)
    for index, (call, label) in enumerate(zip(calls, planned)):
        assert f"--out {out / label}" in call
        assert f"--warnings-file {out / label / 'warnings.txt'}" in call
        assert (" --capture" in call) == (index in (1, 2))
        if label.startswith("A0-math"):
            assert "--backend math" in call
    assert "probe_sdpa.py" in calls[0]
    assert "--backend default" in calls[-1] and "--strict-determinism" in calls[-1]
    log = (out / "matrix.log").read_text()
    for label in planned[:-1]:
        assert f"{label} exit=0\n" in log
    assert f"{planned[-1]} exit=23\n" in log
    if case == "strict_crash":
        assert result.returncode == 1, result.stdout
        assert "A0-strict-k1-rep1-s512" in result.stderr
        assert "without a deterministic-op error" in result.stderr
        assert "deterministic-op-unsupported" not in log
    else:
        assert result.returncode == 0, result.stderr
        assert "A0-strict-k1-rep1-s512 deterministic-op-unsupported index_add_cuda_" in log
