"""Regression tests for E2 gate response normalization."""

import ast
from pathlib import Path

import pytest

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/e2/gates/run_saved_gate.py"
)


def load_normalizer():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalize_response_text"
    ]

    assert len(functions) == 1

    namespace = {}
    module = ast.Module(body=functions, type_ignores=[])

    exec(compile(module, str(RUNNER), "exec"), namespace)

    return namespace["normalize_response_text"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("B<|im_end|>\n", "B"),
        ("C<|im_end|>", "C"),
        ("A<|im_end|>   \n", "A"),
        ("A", "A"),
        ("A\n", "A\n"),
        ("", ""),
        ("A<|im_end|> content", "A<|im_end|> content"),
        ("A<|im_end|><|im_end|>", "A<|im_end|>"),
        ("<|im_end|>", ""),
    ],
)
def test_response_normalization(raw, expected):
    normalize = load_normalizer()
    assert normalize(raw) == expected


def test_rejects_non_string():
    normalize = load_normalizer()

    with pytest.raises(TypeError, match="string"):
        normalize(None)


def test_runner_applies_normalization_and_records_audit():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'RESPONSE_NORMALIZATION_REVISION = "chatml-terminal-im-end-v1"' in source
    assert "answer = normalize_response_text(raw_answer)" in source

    for field in (
        "raw_output_text",
        "scored_output_text",
        "raw_output_sha256",
        "response_normalized",
        "response_normalization_revision",
        "runner_source_sha256",
    ):
        assert f'"{field}"' in source
