# D1 amendment I: resume on a fresh physical A100

Dated 2026-10-15, to be committed before any I numerical arm. User explicitly
resumed D1 after the requested pause. All earlier rentals were deleted and their
raw results exported. H's full cohort was never run. This amendment supersedes
H's requirement to use G's former physical GPU, without pooling prior timings.

New rental: 56106516, machine 151295, A100 SXM4. The assigned physical UUID,
original visibility, device count, torch identity and scoped clock-lock result
are recorded in the public GPU context and lease before numerical work.
Exactly one assigned CUDA device is required. Only that UUID is selected for
numerical work, NVML queries and clock reset; other team resources are untouched.
The inherited provider credential remains inside its rental and is used only
to delete that same rental at its immutable deadline. No master key is uploaded.

The initial large PyTorch Docker image stalled during layer download, before
any numerical arm or assigned-device probe. Recreate the same instance with
the official `ubuntu:24.04` image and install `torch==2.11.0` from the official
`https://download.pytorch.org/whl/cu126` wheel index in a dedicated Python 3.12
virtual environment. The pinned numerical stack and measurement plan stay the
same; retain bootstrap, image-change, package and driver evidence separately.

Attempt `nvidia-smi -i <assigned UUID> -lgc 1200,1200` once before measuring.
If accepted, retain that policy through diagnostics and full measurements;
if denied, retain the unchanged provider clock policy and denial log.
Actual timed-work clocks, not the requested setting, determine acceptance.
The <=5% spread guard, every validity check and all failed/void records remain.

Keep the pinned torch 2.11.0 CUDA 12.6, bitsandbytes 0.50.2,
transformers 5.17.0, peft 0.20.0 and psutil 7.2.2 stack.
First repeat G's two correctness fixtures, existing GPU regressions and two
one-block BF16 diagnostic comparisons under a separate `diagnostic-i` output.
Both clock diagnostics must have one accepted block. They have NO_VERDICT for
performance and do not count toward full-cohort repeats.

On diagnostic success, run H's complete unchanged 98-case plan in a fresh
`full-i` output: target-card ablation, all 96 FP16/BF16 projection comparisons,
then five valid ABBA blocks of full QA training. Every projection retains
>=20 steps AND 5 seconds warm-up followed by 200 timed steps. QA retains
20 warm-up + 256 timed optimizer updates and the same fixed data/model/seed.
F's extra thermal prelude is not used. Repeat correctness and GPU tests at
the start of the full cohort, retaining exact source and environment identity.
No validity failure is repaired by filtering clocks or relaxing exactness.

## Budget and cleanup

Observed previous charges: $1.267, subject to final settlement. Reserve $1.60
for prior costs, $0.80 for transfer and uncertainty. New price including
100 GB disk is $0.4903703704/hour. Fifteen hours cost $7.355555556;
total conservative ceiling is $9.755555556, within cumulative $10.
Initial diagnostic lease is two hours; extend only after passing the committed
diagnostic to the immutable 15-hour absolute deadline. Reserve two hours for
export and stop numerical work before it. Local and remote guards are active;
local guard also stops the exact rental at $7.60 estimated/observed new cost.

Export every raw record, verify archive and restored file hashes, then delete
and verify absence of the exact rental. Full grid, numerical replays, report
and independent acceptance remain required. Deadline or budget exhaustion
means incomplete evidence, never a completed D1 claim.
