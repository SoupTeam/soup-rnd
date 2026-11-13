# BoolQ BF16/NF4 study: blocked before model download

**No training, task-quality, throughput or peak training-memory result was obtained.**
This record preserves two failed environment checks and verified cleanup. It does
not show that Qwen3-32B fits, fails to fit, or loses quality under NF4.

The requested study was Qwen3-32B LoRA on a physical 8 GB GPU with a real RAM upper
bound of 32 GiB: original numerical gates first, matched BF16/NF4 BoolQ quality,
then fresh-process NF4 timing with and without resident decoder layers. The
production snapshot was unchanged PR head `1fc95affd9b25922f032f8403c5739ad84b782ea`.

## Actual attempts

| Attempt | Instance / persistent volume | Outcome |
| --- | --- | --- |
| A, listed RTX 4060 Ti 8 GB | 57511194 / 57511193 | Preflight imported Rich before dependencies were installed; stopped before any model download. |
| B, listed RTX 5060 Ti 8 GB | 57512280 / 57512279 | GPU pre-existing-work/memory-pressure predicate failed; stopped before any model download. |

For B, a separate read-only SSH check also established `memory.max=45650804736`
bytes (**42.515625 GiB**), exceeding the requested 32 GiB, and a read-only cgroup
filesystem. No writable delegated limit was established. The search advertised
32068 MiB of CPU RAM; the created contract reported 64136 MiB. Neither an observed
RSS below 32 GiB nor the search listing was treated as an enforced 32 GiB bound.
The GPU predicate's raw process attribution was not independently retained, so
this record makes no specific utilization or peer-process claim.

The import was fixed in the prospectively frozen B helper and tested with Python
`-S`, without Rich installed. B also recorded a revised swap policy before any
model output: host swap could exist, but any container swap use observed by the
0.25-second monitor would invalidate a stage. This was not represented as
kernel-disabled swap. The B disk mount was SATA; A was loopback-backed. Neither
was called verified NVMe. Numerical `rtol=.01`, `atol=1e-5`, finite checks and
three-decimal loss predicates were unchanged; **they were not executed on a model**.

## Dataset and controls prepared locally

The user selected [BoolQ](https://github.com/google-research-datasets/boolean-questions).
The official `google/boolq` mirror was pinned at
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. The prepared fixture has 128 training
rows and 256 validation rows, with disjoint selected passages and a fixed seed.
Entire prompts plus the answer and EOS fit within 512 tokens; passages were not
truncated. A/B option order is independent of the correct label. Training labels
supervise only the answer letter and EOS. Original data and exact selected token
arrays are retained locally; data attribution/license is recorded in the manifest
(Clark et al., BoolQ, NAACL 2019; CC BY-SA 3.0, applying to data only).

**10 local study controls and 4 retention controls passed.** They check the native
pre-tokenized path and mask, cache mismatch rejection, resource-bound refusals,
metric mutations, timing gate mutations and retention. They are not CUDA/model
accuracy tests. No production code was changed and no full-suite or hosted-CI
success is claimed for this report.

The prepared protocol keeps task quality separate from tensor correctness.
Accuracy and full-vocabulary answer-token NLL are descriptive paired comparisons;
no acceptable NF4 quality-loss threshold was invented. A future 4B pilot can answer
this question more cheaply but cannot establish the corresponding result for 32B.
SSD capacity/performance is a separate study. No 70B/120B hardware claim follows.

## Cost, cleanup and evidence

Aggregate conservative accounting: **$0.546395 / $3.89**, leaving **$3.343605**.
This includes precautionary allowances and is **not an invoice**. Both instances
and both persistent volumes were deleted. A fresh independent account listing
confirmed **zero instances and zero volumes**. No further rental is active.

- [Machine-readable outcome](result-summary.json)
- [Preflight evidence archive](evidence.zip): logs, RAM/cgroup evidence, dataset
  manifest, frozen protocols/helper identities, local-control XML and deletion receipts.
- [Archive and payload SHA256 manifest](evidence-manifest.json)
- [Full local retained-file index](raw-artifact-index.json.gz): 141 files, including
  the complete prospective source/helper archives. Nothing in the index is itself
  evidence of model execution.

Full original bytes are preserved locally in
`C:/Users/esenb/Documents/SoupResearch/nf4-quality-20261024-b`.
The public archive round-tripped byte-for-byte against its SHA256 manifest and was
scanned for the account API credential before publication. No model weights or
credentials are committed.
