# C2 host-E diagnostic protocol — frozen before diagnostic arms

The strict v1 gate in `c2-linux-nvme-20261018.md` is unchanged. Host E's first
idle check failed on reads. Subsequent idle-only observations show a write
burst of approximately 13 MB once a minute; three bursts occurred 12 five-second
windows apart. No diagnostic depth-rate or training-step arm exists at creation.
The following separate study is selected from idle counters, not a depth gain.
No old void result is reclassified, pooled, or used as a pilot.

This diagnostic records a repeated comparison under bounded background writes.
It does **not** satisfy v1's exclusive-write requirement and can NEVER authorize
production integration, regardless of the effect. The shipping decision remains
keep depth 1 until the original quiet-hardware gate is met independently.

All v1 hardware, full-byte correctness, O_DIRECT, >physical-RAM assets, exact
offsets, four workers/ranges, equal pinned staging, source/seed/model/optimizer
settings, GPU/memory/telemetry checks and the canonical F1 required sensors
remain mandatory. Same fresh-process ABBA five-block pilot and adaptive count;
same all-pairs effect thresholds for descriptive comparison, max 20 blocks.
Use the explicit `--diagnostic-writes` driver flag in both studies; the flag is
forwarded to children and recorded in every result. Defaults remain strict v1.

Only these write validity rows differ prospectively:

- Pre-arm five-second NVMe reads must still be <=1 MiB/s. Write bytes/rates are
  retained but do not alone reject the idle check.
- During every timed interval, physical writes must be BOTH <=0.2% of submitted
  read bytes AND <=8 MiB/s averaged over that interval. This admits the observed
  ~13 MB burst on a 10.07 GB primitive arm; it does not assert that the writes
  have no latency effect. Report actual write exposure beside each arm/pair.
- All other validity rules are unchanged, including physical read accounting,
  required F1 checks, errors/unknowns, immutable files and full sequence retention.
  Any invalid arm stops and voids the entire diagnostic sequence. No replacements
  or deletion of unfavorable observations; a new quiet window gets a new name.

Record primitive GB/s and each step time/loss independently. Report spread,
the F1 repeat calculation, paired ratios and background-write exposure. Any
observed effect remains conditional on this host and its background activity;
no claim of isolated SSD capacity, training quality or general acceleration.
Even an effect passing every threshold is labeled DIAGNOSTIC ONLY by the driver.
The original $1 total rental cap and host-E 55-minute limit remain in force.

## Prospective step wrapper correction (after step-e1, before step-e2)

The first step-e1 child stopped before warmup/timing: its idle check ran after
model construction had started the benchmark's own asynchronous prefetch
(41.8 MB/s reads). The entire failed sequence is retained and excluded. For the
new separately named step-e2 sequence only, check the same five-second idle gate
once before model construction/prefetch; record its timestamp and phase. No
threshold, model, timed interval, F1 order or effect rule changes. The physical
counter sampler still encloses every timed step and F1 watches the entire child.
The primitive series already completed on the earlier source revision and is
not rerun or pooled with replacement data.

## Prospective final retry and operational cap amendment (before step-e3)

Step-e2 stopped after its twelfth child. Its own timed-step checks passed, but
canonical F1 found at least 1,725,698,560 bytes of unaccounted physical reads
(limit 1,000,000,000) during b2-p3-d1. The entire sequence is void, including
its first eleven runs; no effect estimate or pilot from it is reused.

The user authorized completing the paid hardware task and explicitly resumed
it. The earlier $1/55-minute cap was the operator's self-imposed bound. This
prospective final retry explicitly changes ONLY that operational cap to $1.30
total and 75 minutes for E from its original provider start (deadline
17:52:36 UTC). No effect/validity threshold is changed. Before a new step-e3
sequence, require twelve consecutive five-second diagnostic idle windows
(reads <=1 MiB/s, writes retained), record temperatures, and recheck cache
size/mtime identities against the previous sequence. If admission fails or any
step-e3 arm is void/unknown, stop this attempt without another replacement.
Otherwise run the complete fresh five-block pilot and unchanged F1 adaptive
rule, with no pooling from step-e1/e2. The sole accepted primitive sequence
remains unchanged. Step-e3 is still DIAGNOSTIC ONLY and cannot authorize
production integration or close the original exclusive-window gate.
