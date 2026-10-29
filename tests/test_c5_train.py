"""C5 ticket 04: one LoRA run over a MoE with experts in NF4, Q2_0 or bf16
(benchmarks/harness/c5_train.py).

Spec: .scratch/c5-q2-0-experts/issues/04-lora-nf4-vs-q2-0.md. Rule:
benchmarks/gate-c5-q2-0-experts.md. The tokenizer is byte-level and built in
memory, and the model is a tiny random-init granitemoe, so nothing is
downloaded.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def harness() -> Iterator[ModuleType]:
    original_path = sys.path[:]
    names = ("_c5_train_test_module", "q2_0_codec", "c5_expert_swap", "c5_verdict",
             "l2l_box", "arm_validity")
    original_modules = {name: sys.modules.get(name) for name in names}
    spec = importlib.util.spec_from_file_location(names[0], HARNESS / "c5_train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[names[0]] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        for name, previous in original_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class TestBatchOrder:
    def test_every_example_is_seen_once_in_steps_of_the_effective_batch(self, harness):
        order = harness.batch_order(10, seed=42, micro_batch=2, accumulation=2)
        assert [[len(micro) for micro in step] for step in order] == [[2, 2], [2, 2], [2]]
        seen = [i for step in order for micro in step for i in micro]
        assert sorted(seen) == list(range(10))

    def test_the_order_depends_on_the_seed_only(self, harness):
        assert harness.batch_order(64, 42, 4, 4) == harness.batch_order(64, 42, 4, 4)
        assert harness.batch_order(64, 42, 4, 4) != harness.batch_order(64, 7, 4, 4)

    def test_splitting_a_step_into_smaller_micro_batches_keeps_its_examples(self, harness):
        # A micro-batch size picked to fit memory must not change what a step trains on.
        coarse = harness.batch_order(64, 123, micro_batch=4, accumulation=4)
        fine = harness.batch_order(64, 123, micro_batch=2, accumulation=8)
        assert [sum(step, []) for step in coarse] == [sum(step, []) for step in fine]


@pytest.fixture(scope="module")
def tokenizer(harness: ModuleType) -> Any:
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    return harness.byte_tokenizer()


def _row(
    instruction: str = "Name a colour.", context: str = "", response: str = "Blue.", row: int = 7,
) -> dict:
    return {"row": row, "instruction": instruction, "context": context,
            "response": response, "category": "open_qa"}


def _decode(tokenizer: Any, ids: list[int], mask: list[bool] | None = None) -> str:
    kept = ids if mask is None else [i for i, keep in zip(ids, mask) if keep]
    return tokenizer.decode(kept)


class TestEncodeExample:
    def test_the_prompt_is_soup_alpaca_rendered_as_chatml(self, harness, tokenizer):
        example = harness.encode_example(_row(context="It is the sky."), tokenizer, 1024)
        assert _decode(tokenizer, example["input_ids"]) == (
            "<|im_start|>user\nName a colour.\nIt is the sky.<|im_end|>\n"
            "<|im_start|>assistant\nBlue.<|im_end|>\n"
        )

    def test_training_labels_are_soups_assistant_turn(self, harness, tokenizer):
        example = harness.encode_example(_row(), tokenizer, 1024)
        trained = [label != -100 for label in example["labels"]]
        assert _decode(tokenizer, example["input_ids"], trained) == (
            "<|im_start|>assistant\nBlue.<|im_end|>\n"
        )

    def test_task_loss_tokens_are_the_answer_only(self, harness, tokenizer):
        # Neither the assistant header nor the <|im_end|> marker after the answer:
        # on OLMoE-0924 the marker is 7 plain tokens of near-zero loss.
        example = harness.encode_example(_row(), tokenizer, 1024)
        assert _decode(tokenizer, example["input_ids"], example["response_mask"]) == "Blue."

    def test_truncation_cuts_the_answer_and_its_tokens(self, harness, tokenizer):
        full = harness.encode_example(_row(response="x" * 50), tokenizer, 1024)
        cut = harness.encode_example(_row(response="x" * 50), tokenizer, 70)
        assert len(cut["input_ids"]) == 70
        assert cut["input_ids"] == full["input_ids"][:70]
        assert sum(cut["response_mask"]) == sum(full["response_mask"][:70])

    def test_a_prompt_longer_than_the_limit_leaves_no_answer_tokens(self, harness, tokenizer):
        example = harness.encode_example(_row(instruction="q" * 100), tokenizer, 64)
        assert sum(example["response_mask"]) == 0
        assert all(label == -100 for label in example["labels"])


SUITE_SIZES = {
    "mini_mmlu": 26, "mini_common_sense": 24, "mini_instruction": 24, "mini_arithmetic": 40,
}


def _write_split(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("c5data")
    rows = [
        _row(instruction=f"Question {i}?", context="Some context." if i % 3 else "",
             response=f"Answer number {i}.", row=i)
        for i in range(40)
    ]
    _write_split(directory / "heldout.jsonl", rows[:8])
    _write_split(directory / "train.jsonl", rows[8:])
    return directory


def _train(harness: ModuleType, data_dir: Path, out_dir: Path, arm: str, *extra: str) -> int:
    return harness.main([
        "--arm", arm, "--seed", "42", "--fixture", "granite", "--data-dir", str(data_dir),
        "--out-dir", str(out_dir), "--micro-batch", "2", "--gate-max-new-tokens", "4", *extra,
    ])


@pytest.fixture(scope="module")
def records(
    harness: ModuleType, data_dir: Path, tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict]:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    out = tmp_path_factory.mktemp("runs")
    result = {}
    for arm in ("nf4", "q2_0", "bf16"):
        assert _train(harness, data_dir, out / arm, arm) == 0
        result[arm] = json.loads((out / arm / "run.json").read_text(encoding="utf-8"))
    return result


@pytest.fixture(scope="module")
def verdict_module() -> Iterator[ModuleType]:
    name = "_c5_verdict_from_train_test"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "c5_verdict.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


class TestFixtureRun:
    def test_the_record_is_labelled_synthetic(self, records):
        for record in records.values():
            assert record["synthetic"] is True
            assert "SYNTHETIC" in record["model"]["note"]

    def test_every_run_finishes_all_steps_with_a_positive_first_grad_norm(
        self, records, verdict_module
    ):
        # 32 train rows in steps of 16 examples.
        for record in records.values():
            assert record["total_steps"] == 2
            assert [step["step"] for step in record["train_log"]] == [1, 2]
            assert verdict_module.run_problems(record) == []

    def test_arm_configs_differ_only_in_the_expert_codec(self, records, verdict_module):
        runs = list(records.values())
        assert verdict_module.config_diff(runs) == []
        assert [r["config"]["codec"] for r in runs] == ["nf4", "q2_0", None]

    def test_every_arm_trains_on_the_same_batches_and_data(self, records):
        nf4, q2_0, bf16 = records["nf4"], records["q2_0"], records["bf16"]
        assert nf4["batch_order_sha256"] == q2_0["batch_order_sha256"] == bf16["batch_order_sha256"]
        assert nf4["data"] == q2_0["data"] == bf16["data"]

    def test_non_expert_weights_match_in_every_arm(self, records):
        nf4, q2_0, bf16 = records["nf4"], records["q2_0"], records["bf16"]
        assert nf4["non_expert_fingerprints"]
        assert nf4["non_expert_fingerprints"] == q2_0["non_expert_fingerprints"]
        assert nf4["non_expert_fingerprints"] == bf16["non_expert_fingerprints"]
        assert nf4["swap"]["checks"] and all(nf4["swap"]["checks"].values())
        assert nf4["expert_digest"] != q2_0["expert_digest"] != bf16["expert_digest"]

    def test_only_attention_lora_weights_train(self, records):
        trainable = records["nf4"]["trainable_parameters"]
        assert trainable
        assert all("lora_" in name and any(p in name for p in ("q_proj", "k_proj", "v_proj",
                                                                "o_proj")) for name in trainable)

    def test_the_panel_loss_is_per_example_and_token_weighted(self, records):
        panel = records["nf4"]["panel"]
        assert [e["row"] for e in panel["per_example"]] == list(range(8))
        tokens = sum(e["tokens"] for e in panel["per_example"])
        nll = sum(e["nll"] for e in panel["per_example"])
        assert panel["task_loss"] == pytest.approx(nll / tokens)
        assert [e["tokens"] for e in panel["per_example"]] == [
            e["tokens"] for e in records["q2_0"]["panel"]["per_example"]
        ]

    def test_every_gate_suite_item_is_recorded(self, records):
        suites = records["q2_0"]["gate_suites"]
        assert {name: s["n"] for name, s in suites.items()} == SUITE_SIZES
        for suite in suites.values():
            assert len(suite["items"]) == suite["n"]
            assert suite["correct"] == sum(item["correct"] for item in suite["items"])

    def test_the_record_names_its_box_code_and_versions(self, records):
        record = records["nf4"]
        for package in ("python", "torch", "transformers", "peft", "bitsandbytes", "datasets"):
            assert package in record["versions"]
        assert set(record["harness"]) >= {"sha256", "codec_sha256", "swap_sha256"}
        assert record["commit"]["sha"]
        assert record["box_stamp_before"] and record["box_stamp_after"]
        assert record["started"] <= record["finished"]


class TestCommandLine:
    def test_a_real_model_needs_a_revision(self, harness, tmp_path):
        with pytest.raises(SystemExit):
            harness.main(["--arm", "nf4", "--seed", "42", "--model", "m",
                          "--data-dir", str(tmp_path), "--out-dir", str(tmp_path)])

    def test_row_limits_are_for_the_fixture_only(self, harness, tmp_path):
        with pytest.raises(SystemExit):
            harness.main(["--arm", "nf4", "--seed", "42", "--model", "m", "--revision", "r",
                          "--data-dir", str(tmp_path), "--out-dir", str(tmp_path),
                          "--train-limit", "8"])

    def test_a_run_that_crashes_keeps_its_record(self, harness, data_dir, tmp_path, monkeypatch):
        pytest.importorskip("peft")

        def broken(*args, **kwargs):
            raise RuntimeError("out of memory (simulated)")

        monkeypatch.setattr(harness, "evaluate_panel", broken)
        assert _train(harness, data_dir, tmp_path, "nf4") == 1
        record = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        assert record["status"] == "failed"
        assert "out of memory (simulated)" in record["error"]
        assert len(record["train_log"]) == 2
