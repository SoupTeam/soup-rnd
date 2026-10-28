# ruff: noqa
# One-off ticket A3-05 diagnostic, run on the Vast instance; kept as evidence, not linted.
"""Diagnostic attempt 03 (not part of the shipped stand): every row has the same length (a
multiple of 64), so GA and the L2L window run the same M. Masked labels and the incomplete
last window stay. Default bitsandbytes kernels."""
import sys
sys.path.insert(0, "/root/soup/benchmarks/harness")
import l2l_mistral_stand as stand

stand.set_deterministic_env()


def equal_rows(count, seq, vocab):
    spans = []
    for i in range(count):
        length = seq
        row = []
        if i % 2:
            row.append((0, length // 4))
        if i % 3 == 2:
            row.append((length // 2, length // 2 + max(1, length // 8)))
        spans.append(row)
    return stand.SyntheticRows([seq] * count, spans, vocab)


print("DIAG: all rows the same length")
stand.make_rows = equal_rows
sys.exit(stand.main(sys.argv[1:]))
