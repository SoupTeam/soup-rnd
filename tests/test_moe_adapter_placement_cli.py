"""CLI and record-status checks that do not require torch or transformers."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "moe_adapter_placement.py"
RUN_ARGS = [
    "run", "--model", "olmoe", "--data", "data.json", "--split", "split.json",
    "--out", "run.json", "--arm", "base",
]


@pytest.fixture(scope="module")
def harness():
    original_path = sys.path[:]
    spec = importlib.util.spec_from_file_location("_moe_adapter_placement_cli", HARNESS)
    module = importlib.util.module_from_spec(spec)
    original_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        if original_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = original_module


@pytest.mark.parametrize("mode", ["import", "help"])
def test_import_and_help_without_torch_or_transformers(mode):
    code = """
import sys
sys.modules["torch"] = None
sys.modules["transformers"] = None
import importlib.util
import runpy

path, mode = sys.argv[1:]
if mode == "import":
    spec = importlib.util.spec_from_file_location("_torch_free_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    sys.argv = [path, "--help"]
    runpy.run_path(path, run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(HARNESS), mode],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == "help":
        for command in ("prepare", "profile", "run"):
            assert command in result.stdout


@pytest.mark.parametrize("flag", ["--micro", "--gen-batch"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_run_rejects_nonpositive_batch_sizes(harness, flag, value, capsys):
    with pytest.raises(SystemExit) as exc:
        harness.parse_args([*RUN_ARGS, flag, value])
    assert exc.value.code == 2
    assert flag in capsys.readouterr().err


def test_run_accepts_single_item_batches(harness):
    args = harness.parse_args([*RUN_ARGS, "--micro", "1", "--gen-batch", "1"])
    assert args.micro == 1
    assert args.gen_batch == 1


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"train": {"log": [{"loss": 1.0}]}, "eval": {"nll": float("nan")}},
         "void_nonfinite"),
        ({"train": {"log": [{"loss": 1.0}]}, "eval": {"nll": float("inf")}},
         "void_nonfinite"),
        ({"train": {"log": [{"loss": 1.0}, {"loss": float("nan")}]},
          "eval": {"nll": 1.5}}, "void_nonfinite"),
        ({"train": {"log": [{"loss": float("inf")}]}, "eval": {"nll": 1.5}},
         "void_nonfinite"),
        ({"train": {"log": [{"loss": 1.0}, {"loss": 2.0}]}, "eval": {"nll": 1.5}},
         "ok"),
        ({"eval": {"nll": 1.5}}, "ok"),
        ({"eval": {"nll": float("nan")}}, "void_nonfinite"),
        ({"eval": {"nll": float("inf")}}, "void_nonfinite"),
    ],
    ids=[
        "nan-eval", "inf-eval", "nan-train", "inf-train", "finite-train",
        "finite-base", "nan-base", "inf-base",
    ],
)
def test_final_status_rejects_nonfinite_losses(harness, record, expected):
    assert harness.final_status(record) == expected
