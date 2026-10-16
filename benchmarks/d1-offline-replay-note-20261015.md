# Offline QA replay note, 2026-10-15

The first Windows Python 3.10 offline QA replay stopped when rescored F1 was
54.46428571428572 versus the GPU worker's Python 3.12 value
54.46428571428571: a 1.4210854715202004e-14 percentage-point difference.
The failure and auditor hash are retained as `qa-replay-attempt1.json`.
All individual predictions and gold remain unchanged. This is metric
aggregation rounding, not a CUDA output, gradient or adapter comparison.
[Python's documentation](https://docs.python.org/3.12/library/functions.html#sum)
records the floating-point summation algorithm change in version 3.12.

Before replay attempt 2, permit at most 1e-12 absolute difference (zero
relative tolerance) only for floating-point aggregate QA metrics. Counts,
keys, gold, IDs, predictions, every loss and adapter names remain exact;
all adapter tensors still require finite values and torch.equal. Report
the actual maximum metric discrepancy. A changed answer or a material
score change still fails. No GPU timing, worker data, source cohort or
earlier verdict is reclassified, and no new CUDA arm is run for this note.
