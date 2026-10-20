"""CPU fake-generator integration; no inference7B, GPU or quality claims."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "benchmarks/g6"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "g6_paired_probe_tests", ROOT / "paired_generation_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
sys.path.pop(0)


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [8, 9]

    def decode(self, tokens, **kwargs):
        pieces = {1: "", 2: "", 4: "426", 5: "\nu", 6: "ser:", 7: "extra",
                  8: "prompt", 9: "\nuser:"}
        return "".join(pieces[token] for token in tokens)


class FakeModel:
    training = True

    def __init__(self, sequence):
        self.sequence = sequence

    def eval(self):
        self.training = False

    def train(self, state):
        self.training = state

    def generate(self, input_ids, **options):
        import torch

        assert not self.training
        assert options["do_sample"] is False and options["num_beams"] == 1
        assert options["max_new_tokens"] == 128
        result = input_ids
        for token in self.sequence:
            result = torch.cat((result, torch.tensor([[token]])), dim=1)
            if token == options["eos_token_id"] or options["stopping_criteria"](result, None).all():
                break
        return result


def suite():
    return {"generation": {"max_new_tokens": 128}, "cases": [
        {"id": "test-task", "kind": "task", "prompt": "Return426", "scorer": "strip_exact",
         "expected": "426"},
    ]}


def test_new_stop_is_in_generation_and_original_response_stays_untrimmed(tmp_path):
    model = FakeModel([4, 5, 6] + [7] * 125)
    for candidate, name in ((False, "original"), (True, "candidate")):
        calls = []
        result = probe.generate_pass(model, Tokenizer(), suite(), tmp_path / name,
                                     "cpu", lambda: calls.append(1), candidate)
        row = json.loads((tmp_path / name).read_text())
        assert row["prompt_ids"] == [1, 8, 9]
        assert row["task_pass"] is candidate
        assert len(row["raw_completion_ids"]) == (3 if candidate else 128)
        assert row["stop_reason"] == ("role_boundary" if candidate else "max_new_tokens")
        assert row["scored_completion"] == ("426" if candidate else row["raw_completion"])
        assert len(calls) >= len(row["raw_completion_ids"]) + 2
        assert result["quality_verdict"] == "NO_VERDICT" and model.training


def test_eos_is_preserved_and_safety_not_automatically_scored(tmp_path):
    data = suite()
    data["cases"][0]["kind"] = "safety"
    model = FakeModel([4, 2])
    result = probe.generate_pass(model, Tokenizer(), data, tmp_path / "eos",
                                 "cpu", lambda: None, True)
    row = json.loads((tmp_path / "eos").read_text())
    assert row["raw_completion_ids"] == [4, 2] and row["stop_reason"] == "eos"
    assert row["task_pass"] is None and row["safety_review"] == "PENDING"
    assert result["task_total"] == 0


def test_unexplained_stop_or_existing_output_refuses_and_restores_training(tmp_path):
    model = FakeModel([4])
    with pytest.raises(ValueError, match="declared reason"):
        probe.generate_pass(model, Tokenizer(), suite(), tmp_path / "failed",
                            "cpu", lambda: None, True)
    assert model.training
    with pytest.raises(FileExistsError):
        probe.generate_pass(model, Tokenizer(), suite(), tmp_path / "failed",
                            "cpu", lambda: None, True)
    assert model.training


def admission():
    spec = {"total_s": 2400, "budget": {"opening_balance_usd": 38.3, "total_cap_usd": 12},
            "instance_id": 56900314, "gpu_uuid": "GPU-test"}
    receipt = {"account": "soup-rnd", "mode": "generation-probe", "contract_sha256": "hash",
               "observed_unix_s": 1000, "balance_usd": 35.9,
               "total_account_rate_usd_h": 1.38, "instance_id": 56900314,
               "gpu_uuid": "GPU-test", "authorized_for_this_stage": True}
    return spec, receipt


@pytest.mark.parametrize("field,value", [
    ("balance_usd", 26.4), ("balance_usd", float("nan")), ("balance_usd", 45),
    ("total_account_rate_usd_h", 0), ("observed_unix_s", 699),
    ("observed_unix_s", 1001), ("mode", "full"), ("account", "personal"),
    ("authorized_for_this_stage", False), ("instance_id", 123), ("gpu_uuid", "other"),
])
def test_new_budget_authority_is_fresh_bounded_and_only_for_named_stage(field, value):
    spec, receipt = admission()
    probe.admit_probe(spec, receipt, "hash", 1000)
    changed = copy.deepcopy(receipt)
    changed[field] = value
    with pytest.raises(ValueError):
        probe.admit_probe(spec, changed, "hash", 1000)


def test_total_cap_and_time_bound_cannot_silently_expand():
    spec, receipt = admission()
    spec["budget"]["total_cap_usd"] = 20
    with pytest.raises(ValueError):
        probe.admit_probe(spec, receipt, "hash", 1000)
    spec, receipt = admission()
    spec["total_s"] = -1
    with pytest.raises(ValueError):
        probe.admit_probe(spec, receipt, "hash", 1000)


def test_actual_transformers_cpu_generation_obeys_split_marker_criterion():
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, MistralConfig, MistralForCausalLM

    class Force(LogitsProcessor):
        def __call__(self, input_ids, scores):
            position = input_ids.shape[1] - 3
            forced = [4, 5, 6, 2][position]
            result = torch.full_like(scores, -float("inf"))
            result[:, forced] = 0
            return result

    torch.set_num_threads(1)
    model = MistralForCausalLM(MistralConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, sliding_window=None,
    )).eval()
    inputs = torch.tensor([[1, 8, 9]])
    for candidate, expected in ((False, [4, 5, 6, 2]), (True, [4, 5, 6])):
        result = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
            max_new_tokens=128, logits_processor=LogitsProcessorList([Force()]),
            stopping_criteria=probe.stopping(Tokenizer(), 3, lambda: None, candidate),
        )
        assert result[0, 3:].tolist() == expected
