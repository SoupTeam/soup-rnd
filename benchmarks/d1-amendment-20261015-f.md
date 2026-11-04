# D1 amendment F: independent thermal prelude for the complete cohort

Dated 2026-10-15, before any timing under this amendment. All E attempts remain
under source 241b6bff and separate output paths. Its ablation achieved five
valid ABBA blocks, replacing one disk-read-void block. The QA series drifted
from an early maximum of 1965 MHz to a late minimum of 1860 MHz: ratio
1.0564516 exceeds the fixed 1.05 guard. Do not accept or pool that series'
timings. Its complete training records remain useful numerical evidence only.
No samples, arms or early blocks are removed to rehabilitate its timing.

The provider denied clock locking. Before a NEW complete cohort, fix a thermal
prelude for every QA arm: in a SEPARATE fresh process, run the existing ordinary
RAM-tier TinyLlama synthetic baseline at sequence 1024, BF16, rank 16, seed 842,
20 warm-up plus 200 timed updates. Use the same pinned pretrained weights and
unchanged provider power/clock policy. Discard that process/model; it transfers
no weights, optimizer, gradients, RNG state or CUDA allocator to the subsequent
QA process. Preserve its command, result, elapsed time and prelude GPU snapshot
under the primary arm's `thermal-prelude` directory. It is conditioning, never
an additional quality study or part of the QA update budget.

Then launch the normal fresh QA arm, independently initialized with seed 842,
exactly 20 warm-up + 256 timed updates and 64 held-out predictions. Do not change
its data, optimizer, initialization, numerical equality, primary full-step
timer or allocator peak definition. This gives the card sustained comparable
work before the first QA timer without adding optimizer updates to the measured
training run. Record conditioning and entire fresh-arm wall time separately;
steady-state step timings exclude conditioning, setup and evaluation. They
must not be described as total job turnaround or time-to-quality measurements.

The rationale is a thermal-transient hypothesis, not an established cause of
the E drift. Verify with the unchanged <=5% guard across ALL actual timed-work
samples. If it still fails, stop for review rather than relax the guard.
The prelude itself has no timing acceptance claim. Its process must exit before
the primary GPU lifecycle/F1 watch starts; a failed prelude prevents that arm.

Repeat correctness and existing GPU tests, ablation, QA and the full 96-case
projection grid under the new committed source and new cohort directory.
Do not reuse E ablation or QA times. Projections retain >=20 steps AND 15 seconds
warm-up and 200 timed steps. F1 block sizing/replacements, all other validity
checks, independent acceptance, cumulative $10 budget and rental guards remain.

The first-offer HTTP 400 had no saved response body. E's description of it as
an expired offer is an inference; the confirmed fact is rejection without a
reported allocation. An authenticated instance listing later showed only the
one exact D1 rental, 56073519. The container's PEP-668 system-pip refusal was
also retained; the successful environment uses a dedicated virtualenv sharing
only the image's pinned torch installation. No package changes occur mid-cohort.
