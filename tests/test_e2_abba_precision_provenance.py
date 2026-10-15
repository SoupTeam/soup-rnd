"""Check E2 ABBA precision validation independently of GPU."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks/e2/run_e2_abba_verified.py"
TRAIN_RUN = ROOT / "src/soup_cli/bench/train_run.py"


def load_validator():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_precision_provenance"
    ]
    assert len(matches) == 1

    namespace = {}
    module = ast.Module(body=matches, type_ignores=[])
    exec(compile(module, str(RUNNER), "exec"), namespace)

    return namespace["validate_precision_provenance"]


@pytest.mark.parametrize(
    ("mode", "dtype", "bf16", "fp16"),
    [
        ("bf16", "torch.bfloat16", True, False),
        ("fp16", "torch.float16", False, True),
    ],
)
def test_valid_precision(mode, dtype, bf16, fp16):
    validate = load_validator()
    report = {
        "provenance": {
            "dtype": dtype,
            "training_precision": {"bf16": bf16, "fp16": fp16},
        }
    }

    result = validate(report, mode)

    assert result["mode"] == mode
    assert result["reported_dtype"] == dtype
    assert result["training_precision"] == {
        "bf16": bf16,
        "fp16": fp16,
    }


@pytest.mark.parametrize(
    ("mode", "dtype", "bf16", "fp16"),
    [
        ("bf16", "torch.float16", True, False),
        ("bf16", "torch.bfloat16", False, True),
        ("bf16", "torch.bfloat16", True, True),
        ("bf16", "torch.bfloat16", False, False),
        ("fp16", "torch.float16", True, False),
        ("fp16", "torch.float16", True, True),
        ("bf16", "torch.bfloat16", "true", False),
    ],
)
def test_rejects_invalid_precision(mode, dtype, bf16, fp16):
    validate = load_validator()
    report = {
        "provenance": {
            "dtype": dtype,
            "training_precision": {"bf16": bf16, "fp16": fp16},
        }
    }

    with pytest.raises(ValueError):
        validate(report, mode)


def test_rejects_missing_precision_evidence():
    validate = load_validator()

    with pytest.raises(ValueError, match="Missing"):
        validate({"provenance": {"dtype": "torch.bfloat16"}}, "bf16")


def test_rejects_unknown_precision_mode():
    validate = load_validator()

    with pytest.raises(ValueError, match="Unsupported"):
        validate({}, "fp32")


def test_training_report_records_actual_flags():
    source = TRAIN_RUN.read_text(encoding="utf-8")

    assert '"training_precision": {' in source
    assert '"bf16": trainer.args.bf16' in source
    assert '"fp16": trainer.args.fp16' in source


def test_runner_enforces_precision_provenance():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'record["precision"] = validate_precision_provenance(' in source
