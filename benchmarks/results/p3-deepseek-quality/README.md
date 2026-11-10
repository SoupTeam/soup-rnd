# DeepSeek diagnostic: inputs and review evidence

Start with the [prelaunch review](../../p3-deepseek-quality-prelaunch-review.md) and
[operator runbook](../../p3-deepseek-quality-pilot-runbook.md). This checkout does
not authorize rental, model-weight download, a pilot or phase3 execution.

## Reproducible inputs

| Input | SHA256 |
| --- | --- |
| `attention-protocol-v1.json` | `fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6` |
| `source-inventory-v1.json` | `5677f32a218d96414334a1536d66a93d0bb87ac98786585faef41329f7ae7e1c` |
| `../p3-stage2-quality/frozen-study-protocol-v1.json` | `170c0b700c137c1f6e0c6788376927e4aaa4e99866b990bc83d0afc42a62bc3c` |
| `../p3-stage2-v2/p3-stage2-v2-protocol.json` | `685fa5bee528969d9b2eef423c90280d179432ba017fbebed12b687f608a8249` |
| `../p3-stage2-quality/admission-kaggle-v3/evidence/dataset/training-sources.jsonl` | `8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7` |
| `../p3-stage2-v2/data-build-kaggle-v1/evidence/data/manifest.json` | `07f3c65cce76d0b209ce20aff7b096857077ac906156ac4ecf2c7955141b2b81` |

The complete frozen task directory and 7801-row training source are included;
no private Kaggle dataset is required to reconstruct the training pack. Tokenizer
metadata comes from the pinned Hub revision after runtime approval, not another
checkpoint. The runbook checks the rebuilt pack's fixed manifest identity.

The 17 copied Granite prerequisite modules are unchanged. Every hash in
`prerequisites-v1.json` matches its v1/v2 frozen protocol entry. Scoped Git
attributes preserve those LF sources and exact JSON/JSONL receipt bytes. Generated
data is folded in GitHub's diff so reviewers can focus on implementation first.

`source-inventory-v1.json` was derived from the public
[Hugging Face revision metadata](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V3/revision/e815299b0bcbac849fa540c768ef21845365c9eb?blobs=true).
It contains the 163 LFS weight identities and four selected metadata Git blobs.
The revision has no `generation_config.json`. This metadata read did not download
weight payloads. Source admission checks the reviewed inventory, metadata content,
weight sizes and cache/source fingerprint; it does not claim a new local SHA256
pass over all 688.6GB of weights.

The frozen rows retain their upstream benchmark provenance and usage conditions;
they are not newly authored Soup data. Source repositories, exact revisions,
source-file hashes, split roles, exclusions and overlap audit are in the included
v2 data manifest and audit files. In particular, training rows are GSM8K-train and
MBPP-train; the pilot timing uses only the frozen HellaSwag/ARC/BoolQ development
inputs. No new model quality outcomes are included.

## Evidence boundaries

- `prelaunch-review-v1/` records new remote CPU verification and direct CLI checks.
  It is not native BF16 qualification or a 671B result.
- `efficient-vast-v1/` is immutable historical Ampere evidence. Its tiny E3 remains
  **FAIL: 0.05028628354277727 >0.05**. The complete small raw tiny-gate directory,
  primitive receipts, selected test logs and standalone step records are included
  unchanged. No gate or E3 was rerun to prepare this PR.
- Historical source ZIPs, the large full evidence tarball, repeated fixture
  checkpoints, provider account responses and private orchestration files are not
  committed. Their historical references do not mean those archives are bundled.
  Selected standalone manifests describe larger original runs; they are not a
  self-contained replay bundle. The complete tiny-gate raw inventory is bundled.
- `pilot-cost-proposal-v1/README.md` and `estimate.json` are the quote-time proposal,
  not spending permission or a current availability guarantee. Raw provider rows
  and private research notes referenced there remain outside Git. The proposal's
  code-gap section predates the fixes in this PR; use the current runbook.

No weights or NF4 cache are included. The full-model pilot, its actual bill and
its deletion confirmation do not yet exist. After that separately approved run,
Task2.2 must present a protocol hash, phase3 cost forecast and scope-cut options;
phase3 remains blocked until the user approves both the hash and budget.
