"""Local contract checks for the D2 Kaggle notebook (no remote execution)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "d2-fast-lora-kaggle.ipynb"


def _sources() -> dict[str, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def test_cpu_regression_gate_explicitly_excludes_platform_only_mps_cases() -> None:
    calls = [
        node
        for source in _sources().values()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_pytest"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "cpu-regressions"
    ]
    assert len(calls) == 1
    keywords = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
    assert keywords["expression"] == "not gpu and not smoke"
    assert keywords.get("keyword") == "not mps", (
        "Kaggle Linux cannot pass a no-skips CPU gate that selects the existing Apple MPS tests"
    )
    configuration = _sources()["configuration"]
    assert "UNVERIFIED_PLATFORM_BLOCK" in configuration
    assert "APPLE_MPS" in configuration


@pytest.mark.parametrize(
    "requirement",
    ["transformers==5.17.0", "accelerate==1.14.0", "bitsandbytes==0.50.1"],
)
def test_notebook_uses_the_locally_validated_training_stack(requirement: str) -> None:
    assignments = [
        node
        for source in _sources().values()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PINS" for target in node.targets)
    ]
    assert len(assignments) == 1
    pins = ast.literal_eval(assignments[0].value)
    assert requirement in pins
    assert not any(pin.startswith("torch==") for pin in pins)


def _assignment(name: str) -> ast.AST:
    matches = [
        node.value
        for source in _sources().values()
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert len(matches) == 1, f"Missing or ambiguous notebook setting: {name}"
    return matches[0]


def test_private_frozen_zip_is_the_default_without_a_git_fallback() -> None:
    assert ast.literal_eval(_assignment("SOURCE_MODE")) == "zip"
    assert ast.literal_eval(_assignment("SOURCE_ZIP")) == ""
    assert ast.literal_eval(_assignment("EXPECTED_ZIP_SHA256")) == ""
    stage = _sources()["checkout-install"]
    assert "extract_source_zip" in stage
    tree = ast.parse(stage)
    git_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "SOURCE_MODE" in ast.unparse(node.test)
    )
    assert "SOURCE_MODE == 'zip'" in ast.unparse(git_branch.test)
    assert "ls-remote" not in ast.unparse(ast.Module(body=git_branch.body, type_ignores=[]))
    assert "SOURCE_MODE == 'git'" in ast.unparse(
        ast.Module(body=git_branch.orelse, type_ignores=[])
    )


def test_cpu_regressions_require_both_precision_regression_files() -> None:
    files = ast.literal_eval(_assignment("D2_TESTS"))
    assert "tests/test_d2_mixed_precision_backward.py" in files
    assert "tests/test_d2_reference_precision.py" in files
    source = _sources()["cpu-gates"]
    assert "expected_count=50" in source
    assert "expected_count=178" in source


def test_notebook_junit_keeps_precision_record_properties() -> None:
    assert "junit_family=legacy" in _sources()["evidence-helpers"]


def test_every_code_cell_and_child_script_compiles_without_execution() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = _sources()
    assert len(sources) == 10
    children = []
    for name, source in sources.items():
        compile(source, name, "exec")
        tree = ast.parse(source, feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if any(isinstance(t, ast.Name) and t.id.endswith("_PROBE") for t in node.targets):
                    children.append(node.value.value)
    assert len(children) == 3
    for source in children:
        compile(source, "embedded child", "exec")
        ast.parse(source, feature_version=(3, 10))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code_cells)
