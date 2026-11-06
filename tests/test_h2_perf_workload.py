"""H2-close stage P workload: deterministic generation and frozen token accounting."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "harness"))

import h2_perf_receipt  # noqa: E402
import h2_perf_workload  # noqa: E402

H2 = ROOT / "benchmarks" / "h2"
RECEIPT = json.loads((H2 / "perf_workload_receipt.json").read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_is_deterministic_and_equals_the_frozen_file():
    data = h2_perf_workload.render(h2_perf_workload.generate_rows())
    assert data == h2_perf_workload.render(h2_perf_workload.generate_rows())
    assert data == (H2 / "perf_workload.jsonl").read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines()]
    assert len(rows) == 96
    assert all(set(r) == {"instruction", "input", "output"} and r["input"] == "" for r in rows)


def test_another_seed_gives_another_workload():
    other = h2_perf_workload.render(h2_perf_workload.generate_rows(seed=1))
    assert other != (H2 / "perf_workload.jsonl").read_bytes()


def test_receipt_pins_the_current_workload_generator_and_configs():
    assert RECEIPT["status"] == "PERF_WORKLOAD_TOKEN_ACCOUNTING_FROZEN_CPU"
    assert RECEIPT["workload"]["sha256"] == _sha(H2 / "perf_workload.jsonl")
    assert RECEIPT["generator_sha256"] == _sha(ROOT / "benchmarks/harness/h2_perf_workload.py")
    for arm, entry in RECEIPT["configs"].items():
        assert entry["sha256"] == _sha(ROOT / entry["path"]), arm
    assert RECEIPT["performance_verdict"] == "NO_VERDICT"


def test_receipt_totals_are_consistent_and_within_bounds():
    rows = RECEIPT["rows"]
    assert len(rows) == 96 and [r["row"] for r in rows] == list(range(96))
    tokens = [r["tokens"] for r in rows]
    useful = [r["useful_targets"] for r in rows]
    assert RECEIPT["tokens"]["total"] == sum(tokens)
    assert RECEIPT["useful_targets"]["total"] == sum(useful)
    assert max(tokens) < RECEIPT["max_length"] == 1024 and min(tokens) >= 256
    assert all(0 < u < t for u, t in zip(useful, tokens))
    assert RECEIPT["use_liger_kernel"] == {"baseline": False, "liger_ce": True}


def test_perf_configs_differ_only_in_backend_liger_flag_and_output():
    configs = {k: json.loads((ROOT / v["path"]).read_text()) for k, v in RECEIPT["configs"].items()}
    base = configs["baseline"]
    assert base["data"]["train"] == "benchmarks/h2/perf_workload.jsonl"
    assert base["data"]["max_length"] == 1024
    for arm, allowed in (
        ("liger_ce", {"training.use_liger", "output"}),
        ("unsloth", {"backend", "output"}),
    ):
        other, changed = configs[arm], set()
        for key in base:
            if isinstance(base[key], dict):
                changed |= {f"{key}.{k}" for k in base[key] if base[key][k] != other[key].get(k)}
            elif base[key] != other[key]:
                changed.add(key)
        assert changed == allowed, arm


@pytest.mark.skipif(
    not os.environ.get("H2_TOKENIZER_DIR"),
    reason="Set H2_TOKENIZER_DIR to the verified snapshot tokenizer files",
)
def test_receipt_rebuilds_identically_with_the_real_tokenizer(tmp_path):
    fresh = h2_perf_receipt.build(Path(os.environ["H2_TOKENIZER_DIR"]), tmp_path / "run")
    assert fresh == RECEIPT
