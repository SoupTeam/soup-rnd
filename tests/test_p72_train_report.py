"""The PR tables of the training run: time, expert fetches and bytes, adapter equality, NLL."""

from __future__ import annotations

from benchmarks.harness import p72_train_report as report


def run(arm: str, digests: list[str], seconds: float, fetches: int = 100, nll: float = 0.3) -> dict:
    out = {
        "kind": "l2l" if arm == "A" else "mbr",
        "training_seconds": seconds,
        "steps": [{"step": i + 1, "adapter_sha256": d, "fetches": fetches,
                   "logical_bytes": fetches * 4_866_048, "loss": 1.0}
                  for i, d in enumerate(digests)],
        "total_fetches": fetches * len(digests),
        "total_logical_bytes": fetches * len(digests) * 4_866_048,
    }
    if nll:
        out["heldout"] = {
            "per_example": [
                {"group_sha256": str(i), "type": "GSM_x", "nll_sum": nll * 10 + i * 1e-3,
                 "target_tokens": 10, "nll_token_mean": nll + i * 1e-4}
                for i in range(6)],
            "mean_per_example_nll": nll, "token_weighted_nll": nll,
        }
    return out


def test_the_performance_table_has_one_column_per_run_and_gigabytes_like_the_pr() -> None:
    text = report.performance_table({"L2L A0": run("A", ["a", "b"], 90.5, 441),
                                     "Exact MBR": run("C", ["a", "b"], 17.8, 70)})
    lines = text.splitlines()
    assert lines[0] == "| Metric | L2L A0 | Exact MBR |"
    assert "| Training time | 90.50 s | 17.80 s |" in lines
    assert "| Logical expert fetches | 882 | 140 |" in lines
    assert any(line.startswith("| Logged logical expert bytes | 4.3 GB |") for line in lines)


def test_the_equality_paragraph_says_whether_both_runs_were_deterministic() -> None:
    a, c = run("A", ["a", "b"], 1), run("C", ["a", "b"], 1)
    a["deterministic"] = c["deterministic"] = True
    assert "in torch deterministic mode" in report.equality_text(a, c)
    a["deterministic"] = False
    text = report.equality_text(a, c)
    assert "not in deterministic mode" in text and "does not support" in text
    del a["deterministic"], c["deterministic"]
    assert "determinism was not recorded" in report.equality_text(a, c)


def test_the_equality_paragraph_states_the_step_count_and_the_first_difference() -> None:
    same = report.equality_text(run("A", ["a", "b", "c"], 1), run("C", ["a", "b", "c"], 1))
    assert "3 of 3 steps" in same and "bit-for-bit" in same
    diff = report.equality_text(run("A", ["a", "b", "c"], 1), run("C", ["a", "x", "c"], 1))
    assert "first differ at step 2" in diff


def test_the_quality_table_lists_the_two_nll_values_and_the_paired_interval() -> None:
    text = report.quality_table(run("A", ["a"], 1, nll=0.3), run("C", ["a"], 1, nll=0.3), seed=1)
    assert "| Mean NLL per example | 0.300250 | 0.300250 |" in text
    assert "paired bootstrap" in text
