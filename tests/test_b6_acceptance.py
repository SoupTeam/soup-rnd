"""Negative controls for the prospective RAM and paired-evaluation protocol."""

import copy
import math

import pytest


def test_evaluation_matches_direct_causal_loss_without_gradients():
    from types import SimpleNamespace

    import torch

    from benchmarks.harness.b6_evaluation import evaluate_examples

    class TinyCausal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16)

        def forward(self, input_ids, labels):
            assert not self.training and not torch.is_grad_enabled()
            logits = self.head(self.embedding(input_ids))
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1),
            )
            return SimpleNamespace(loss=loss)

    model = TinyCausal()

    def make_batch(text):
        ids = torch.tensor([[int(value) for value in text.split()]])
        return {"input_ids": ids, "labels": ids}

    examples = [{"id": "short", "text": "1 2 3 4"},
                {"id": "long", "text": "1 2 3 4 5 6"}]
    events = []
    actual = evaluate_examples(model, examples, make_batch,
                               lambda kind, **row: events.append((kind, row)), "before")
    assert [row["predicted_tokens"] for row in actual] == [3, 5]
    with torch.no_grad():
        expected = [float(model(**make_batch(row["text"])).loss) for row in examples]
    assert [row["loss"] for row in actual] == expected
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [event[0] for event in events] == [
        "evaluation_before_start", "evaluation_before", "evaluation_before",
    ]


def test_quality_uses_token_weighting_and_accepts_degradation():
    from benchmarks.harness.b6_evaluation import paired_summary

    before = [dict(id="a", loss=1.0, predicted_tokens=2, token_sha256="a" * 64),
              dict(id="b", loss=3.0, predicted_tokens=6, token_sha256="b" * 64)]
    after = copy.deepcopy(before)
    after[0]["loss"] = 2.0
    result = paired_summary(before, after, expected_ids=["a", "b"])
    assert result["before_loss"] == 2.5
    assert result["after_loss"] == 2.75
    assert result["loss_delta"] == 0.25
    assert result["before_perplexity"] == pytest.approx(math.exp(2.5))
    assert result["predicted_tokens"] == 8
    assert result["quality_gain_claimed"] is False


@pytest.mark.parametrize("defect", ["missing", "reordered", "tokens", "digest", "nan"])
def test_paired_comparison_fails_closed(defect):
    from benchmarks.harness.b6_evaluation import paired_summary

    before = [dict(id=str(i), loss=2.0, predicted_tokens=63, token_sha256=str(i) * 64)
              for i in range(8)]
    after = copy.deepcopy(before)
    if defect == "missing":
        after.pop()
    elif defect == "reordered":
        after.reverse()
    elif defect == "tokens":
        after[0]["predicted_tokens"] = 62
    elif defect == "digest":
        after[0]["token_sha256"] = "f" * 64
    else:
        after[0]["loss"] = float("nan")
    with pytest.raises(ValueError):
        paired_summary(before, after, expected_ids=[str(i) for i in range(8)])


def test_real_frozen_protocol_is_integrity_checked(tmp_path):
    from pathlib import Path

    from benchmarks.harness.b6_evaluation import load_evaluation

    root = Path(__file__).parents[1] / "benchmarks/protocols/b6-acceptance-20261029"
    protocol, examples = load_evaluation(root / "protocol.json", root / "evaluation-data.json")
    assert len(examples) == 8
    assert protocol["max_length"] == 64
    damaged = tmp_path / "damaged.json"
    damaged.write_bytes((root / "evaluation-data.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        load_evaluation(root / "protocol.json", damaged)


def test_ram_rejects_large_host_and_unbounded_or_large_cgroup():
    from benchmarks.harness.b6_ram import ram_constraint

    for maximum in (None, 64 * 2**30):
        with pytest.raises(ValueError, match="32 GiB"):
            ram_constraint({"memory_max": maximum, "host_total": 128 * 2**30})


def test_ram_accepts_recorded_hard_cgroup_on_larger_host():
    from benchmarks.harness.b6_ram import ram_constraint

    assert ram_constraint({"memory_max": 32 * 2**30, "host_total": 128 * 2**30}) == {
        "kind": "cgroup_v2", "hard_limit_bytes": 32 * 2**30,
    }


def test_physical_small_host_requires_zero_host_swap():
    from benchmarks.harness.b6_ram import ram_constraint

    assert ram_constraint({"memory_max": None, "host_total": 32 * 2**30,
                           "host_swap_total": 0})["kind"] == "physical_host"
    with pytest.raises(ValueError):
        ram_constraint({"memory_max": None, "host_total": 32 * 2**30,
                        "host_swap_total": 1})


@pytest.mark.parametrize("defect", [
    "oom", "swap", "missing_phase", "failure", "limit_changed", "none",
])
def test_ram_fit_is_not_claimed_after_failed_or_incomplete_run(defect):
    from benchmarks.harness.b6_ram import summarize_ram

    phases = ["download", "preparation", "training", "evaluation_before",
              "evaluation_after", "reload"]
    samples = [dict(phase=phase, memory_current=10, memory_peak=20, swap_current=0,
                    swap_peak=0, events={"oom_kill": 0}, process_rss_bytes=8, memory_max=32 * 2**30)
               for phase in phases]
    code = 0
    if defect == "oom":
        samples[-1]["events"]["oom_kill"] = 1
    elif defect == "swap":
        samples[-1]["swap_peak"] = 1
    elif defect == "missing_phase":
        samples.pop()
    elif defect == "failure":
        code = 1
    elif defect == "limit_changed":
        samples[-1]["memory_max"] = 64 * 2**30
    result = summarize_ram(samples, {"kind": "cgroup_v2", "hard_limit_bytes": 32 * 2**30},
                           code)
    assert result["ram_32gib_constrained_pilot_passed"] is (defect == "none")
