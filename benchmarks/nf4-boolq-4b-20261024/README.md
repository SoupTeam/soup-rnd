# Qwen3-4B BoolQ: streamed BF16 versus NF4

A matched task-quality pilot, followed by four fresh-process timing pairs.
This measures Soup's streamed precision tradeoff with controlled FP32 adapters.
No acceptable quality-loss
margin was agreed, so equal point estimates or an interval including zero are
not presented as proof of equivalent quality.

## Configuration

Qwen/Qwen3-4B, pinned revision `1cfa9a7208912126459214e8b04321603b3df60c`.
Physical NVIDIA GeForce RTX 3060 Ti 8 GB; RAM-backed streaming, two rotating
buffers, pinned host weights, checkpointed layers, no persistent decoder layers.
BF16 arithmetic, ordinary PEFT LoRA rank/alpha 4 on seven decoder projections,
504 FP32 trainable tensors. FastLoRA, Liger, explicit FlashAttention and compile
were not enabled. Native attention backend and exact resolved Trainer arguments
are retained. NF4 uses double quantization; actual packed bytes/specs were checked.

Training: one seed (792), the same seeded FP32 A / zero B initialization,
256 examples, batch 1, accumulation 2, 128 optimizer updates per format.
Each row is right-padded to 512 positions with only its answer letter and EOS
supervised. The native mask and exact minibatch identity sequence are checked.

The native TRL collator reconstructs an all-ones attention mask for the already
padded training rows. Actual input ids and two-target label masks are audited
against every frozen row. Padding occurs only after both supervised positions;
causal attention prevents it from affecting those positions. The actual mask
is disclosed rather than described as byte-identical to the cached mask.

BoolQ uses the official google/boolq mirror at revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. The 256 held-out rows have passages
disjoint from selected training passages. Entire prompts fit without truncation.
Yes/no option order is determined independently of the gold answer. Thinking is
disabled. Accuracy is forced choice A/B; NLL is the correct letter's probability
over the complete vocabulary, not full-text perplexity. Unrestricted next-token
accuracy and A/B probability mass are also retained to expose format failures.

## Task quality

| Metric | BF16 | NF4 |
| --- | ---: | ---: |
| Before accuracy, % | 85.938 | 85.156 |
| Before answer-token NLL | 2.417417 | 1.656451 |
| Before unrestricted accuracy, % | 85.938 | 85.156 |
| After accuracy, % | 86.719 | 85.547 |
| After answer-token NLL | 1.799177 | 1.101349 |
| After unrestricted accuracy, % | 86.719 | 85.547 |

After-training NF4 minus BF16 accuracy: **-1.172 percentage points**;
paired-example bootstrap 95% interval **[-3.906, +1.172] pp**.
Answer-token NLL delta: **-0.697828** (lower is better).
The majority-label baseline on these rows is **58.984%**.
Bootstrap: 20,000 resamples, seed 792. This interval reflects held-out-example
variation; it does not estimate training-seed or task-to-task uncertainty.

## Speed and memory after quality replay

| Metric | BF16 | NF4 |
| --- | ---: | ---: |
| Median optimizer step, s | 2.3385 | 1.9432 |
| Median complete fresh six-update entry, s | 41.8527 | 38.0145 |
| Peak allocated VRAM, GiB | 2.2515 | 1.9718 |
| Peak reserved VRAM, GiB | 2.5410 | 2.2871 |
| Peak process RSS, GiB | 9.0448 | 3.6256 |

Median paired NF4 optimizer-step time reduction: **+17.230%**;
whole-pair bootstrap 95% interval **[+14.657%, +18.455%]**.
Negative values mean NF4 was slower. These are pointwise descriptive intervals
from four pairs, not a universal or quality-equivalent speedup claim.

Eight distinct timing PIDs; balanced BF16/NF4 order AB/BA/BA/AB, two warmup
plus four measured updates per process (48 updates in total). Each format
starts from its own trained adapter with a fresh optimizer and identical batch
ordering. Startup is reported separately; no time-to-convergence was measured.
The fixed padded workload is identical across formats; no useful-token throughput
claim is inferred from padding. RAM here is peak process RSS, not complete host
memory. Actual physical/cgroup RAM limits are in the hardware record; this pilot
does not establish the requested 32B/32 GiB RAM capacity scenario.

## Correctness, retention and cost

Task logits, all named initial/final adapters and actual adapter changes were
independently replayed from retained bytes before timing. Both formats used the
same initialization, exact minibatch sequence and resolved training arguments.
Training rejects missing/nonfinite adapter gradients. Same-format timing repeats
compare all 504 saved adapter tensors under the original reference-first
`rtol=.01, atol=1e-5`, finite and three-decimal loss gates. No cross-precision
tensor allclose or exact gradient-parity claim is made. FP64 metric replay allows
only the prospectively recorded 1e-10 cross-platform arithmetic roundoff.

- local_study_controls: 10 passed, no failures/errors/skips.
- retention_controls: 4 passed, no failures/errors/skips.
- gpu_host_controls: 144 passed, no failures/errors/skips.

No full-repository or hosted-CI success is inferred from these controls.
Three earlier pilot attempts are retained: a donor-catalogue schema error, an
observer API error, then an adapter-configuration refusal after BF16 calibration.
They supply no accepted quality/timing comparison; all costs are included.
LoRA parameters are explicitly normalized to FP32 before identical initialization
and optimizer creation in both formats; native pre-normalization dtypes are retained.
Intermediate optimizer checkpoints were disabled equally; native final adapters and
all primary quality logits/initial/final states remain fully retained.

Aggregate conservative pilot accounting: **$1.842130/$13.34**;
combined authorized remainder: **$11.501476**.
This includes precautionary allowances and is not an invoice. Earlier unrelated
studies were already deducted from the starting remainder and are not counted twice.
All pilot instances and persistent volumes are deleted; fresh independent
account listings confirm zero instances and zero volumes.

## Reproduction and scope

[Machine-readable results](result-summary.json), [exact source/config/replay
evidence manifest](evidence-manifest.json), [full raw-file index](raw-artifact-index.json.gz).
Concatenate the numbered evidence.zip.part files in manifest order to obtain
the public review ZIP; all selected bytes were reassembled and hash-verified.
Full original logits, adapters, logs and canonical archives remain locally at
`C:/Users/esenb/Documents/SoupResearch/nf4-boolq-4b-20261024-d`.

Dataset attribution: Clark et al., BoolQ, NAACL 2019; original data CC BY-SA 3.0.
The data license applies to data only. Model weights and credentials are not committed.

This is one short answer-only 4B LoRA pilot. Pretraining contamination was not
assessed; generation, full convergence, multiple training seeds and other tasks
were not tested. It does not establish quality/capacity for 32B, 70B or 120B.
The 32B hardware-bound residency study and SSD/cache-placement work remain
separate next stages in the [research plan](../streaming-research-plan.md).
