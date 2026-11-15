# Frozen decision rule: two-read under L2L, ticket 08

The commit adding this rule freezes it before the first run of the new Kanana series.
The user approved the relative foreign-read threshold on 2026-11-15. GPU rental still needs
separate explicit authorization. No Kanana run has been made under this rule.

This is a new series, not a revision of ticket 06. Its saved evidence and verdicts remain
unchanged. The statistical method is `benchmarks/METHOD.md`; accuracy, recipe and outcome
criteria retain `streaming-ordinary-vs-complete-l2l-rule-20261115.md` except as stated below.

## Arms and order

- A: `l2l_complete`, grouping, packed NF4 transfer, 8 GiB prepacked host cache,
  two-read disabled: three reads per hit.
- B: `l2l_two_read`, the same complete levers with two-read enabled: two reads per hit.
- Ordinary: `l2l_ordinary`, used for the new C2 k=2 comparison and accuracy captures.

On a newly qualified host, capture accuracy first at k=2 and k=8. B must pass A3 against
both A and ordinary, and read exactly two experts per hit. A must read exactly three.
The C2 accuracy pairs from the previous rule must also pass. A3 uses the GA MATH versus
EFFICIENT_ATTENTION noise-floor captures from the same session; no tolerance is relaxed.

Then run these comparisons in order:

1. C2 k=2: ordinary A, complete-without-two-read B, ABBA.
2. A versus B at k=1: complete-without-two-read A, complete-with-two-read B, ABBA.
3. A versus B at k=2: the same arms and order.

C1 and C2 k=1 timings are not repeated. Each comparison starts a new five-clean-block pilot;
no ticket-06 blocks or sigma are reused. Void blocks remain saved and are replaced.
`blocks_to_add` supplies the required count with one adjustment. The metric remains the median
of six measured optimizer steps after two warm-up updates, with the prior NF4/BF16/fp32-LoRA,
512-token, batch-one recipe. Each ABBA arm runs in a fresh process.

## Validity

`suspend`, `power` and `foreign_reads` are mandatory. Unknown remains unknown and yields
NO VERDICT; void blocks are excluded and replaced. Foreign-read disk bounds are unchanged.
Let own be ArmWatch's `own_read_bytes` and limit = max(1,000,000,000 bytes, 0.05 * own).

- OK if the maximum possible foreign reads is no greater than limit.
- VOID if the minimum possible foreign reads is greater than limit.
- UNKNOWN otherwise, including missing sensor evidence.

Stop the series after two recorded runs whose minimum foreign reads exceeds 5% of own reads.
Keep those runs and discuss the noisy host with the user before continuing.

Every comparison is reported twice: first under this rule, and again using only runs that
pass the original ADR 0002 limit (strictly below 1,000,000,000 bytes). Apply the same required
checks and block-count rule to both views. If outcomes differ, retain both and label the
comparison sensitive to foreign reads. Never change a saved row or raw watch record to
produce the original-threshold view.

## Outcomes and reporting

Use a two-sided paired t-test on ABBA block mean-A minus mean-B, alpha 0.05. The effect is
(mean A - mean B) / mean A. WIN requires significance and at least 10% reduction; SMALL is a
significant positive reduction below 10%; LOSS is a significant negative reduction;
NO DIFFERENCE is not significant; failed accuracy, unknown validity or insufficient blocks
means NO VERDICT. The original rule's pilot/count formula and single adjustment are retained.

Report C2 k=1 as the prior WIN and C2 k=2 from this new series, plus the two-read A/B results.
The prior C2 k=2 remains NO VERDICT (three clean blocks). k=4 and k=8 performance points are
linear extrapolations, not verdicts. Report all voids, actual physical GPU memory, and the
29.756 GiB emulated RAM limit with its implementation. CPU read counts alone imply no speed
verdict.

## Host qualification and invocation

Before the series, observe `diskwatch.py` for 30 idle minutes. More than two foreign-read
bursts above max(1 GB, 5% of own reads) fail the host. Own reads are near zero when idle, so
the limit there is 1 GB. The user changed this host check from a fixed 1 GB to the relative
limit on 2026-11-15, before the series starts. Require a local block-device volume, direct reads at least
2.5 GB/s, and the ticket's RAM/swap conditions. No measurement starts before host qualification
and explicit rental authorization. Use a fresh result root; never copy an earlier host's runs.

The driver profile is `oc_stage1_drive.py stage1 --two-read` with the source, cache, fresh root,
revision and approved budget arguments. It selects fraction 0.05 and the order above.
`verdict --two-read --ks 1 2` writes both validity views. Legacy commands without `--two-read`
retain the original threshold. Do not override the fraction for this frozen series.
