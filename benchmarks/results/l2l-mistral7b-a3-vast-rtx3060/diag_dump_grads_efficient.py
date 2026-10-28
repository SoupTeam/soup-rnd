# ruff: noqa
# One-off ticket A3-05 diagnostic, run on the Vast instance; kept as evidence, not linted.
"""Diagnostic attempt 05 (SDPA EFFICIENT_ATTENTION pinned in both arms; noise floor) (not part of the shipped stand): the stand unchanged, plus the
per-step LoRA gradients of GA and L2L written to disk before they are compared."""
import sys
sys.path.insert(0, "/root/soup/benchmarks/harness")
import torch
import l2l_mistral_stand as stand

stand.set_deterministic_env()
real = stand.compare_runs
calls = []


def dumping(left, right):
    calls.append(1)
    if len(calls) == 1:
        torch.save(left.grads, "/root/runs/grads_ga_eff.pt")
    if len(calls) == 2:
        torch.save(right.grads, "/root/runs/grads_l2l_eff.pt")
        torch.save({"ga": left.logs, "l2l": right.logs}, "/root/runs/logs_eff.pt")
    return real(left, right)


stand.compare_runs = dumping
stand.PINNED_SDPA_BACKEND = "EFFICIENT_ATTENTION"
print("DIAG: gradients dumped to /root/runs")
sys.exit(stand.main(sys.argv[1:]))
