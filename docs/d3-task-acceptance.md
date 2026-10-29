# D3 acceptance audit, 2026-10-29

**D3 is not signed off.** The live Trello card shows all five items checked and
is in **Обработка Фидбека**. The latest PR review remains **CHANGES_REQUESTED**;
the earlier forward-only approval was dismissed. The board's 100% is an observed
UI state, not independent evidence that the assigned task is complete.

The initial integration/main merge was committed and pushed as
`cd6515ac503b15a111050b35db5c33768d506a9b`, with parents `98706ed0` and
`9d8b1d7b`. Its source freeze and all prepared data/host checks passed. The
first formal public-trainer cell then failed its evaluation control after four
finite training steps; no cell or timing comparison was accepted. The current
corrective working code is not relabeled as that frozen source. See the
[retained failure](../benchmarks/results/d3-public-trainer-attempt-01-cd6515ac/report.md).

## Sources and precedence

- [Assigned Trello card](https://trello.com/c/G6t5xoIj), read directly on
  2026-10-29: design before B1.4; implementation after B1.2 plus F4 baseline;
  exact sequential-loop comparison, matching per-expert resident/streamed kernel,
  and separate speed, memory and actual launch-count evidence.
- Operator-supplied `rndTASK_agent_context(1).md`, a transcription of
  `rndTASK.pdf` dated 2026-10-05, source pages 6, 10 and house rules 17–19.
  Its SHA-256 is retained in
  [requirements.json](../benchmarks/protocols/d3-public-trainer/requirements.json).
  The original PDF was not separately reopened in this audit.
- [Owner-directed full-delivery review](https://github.com/SoupTeam/soup-rnd/pull/28#pullrequestreview-5605458361),
  followed by the current [PR description](https://github.com/SoupTeam/soup-rnd/pull/28).
  A narrower PR description cannot authorize a reduction of the assignment.
- Existing [design/history](grouped-expert-design.md),
  [public trainer integration](d3-trainer-integration.md) and the preserved
  source-pinned reports linked below.

The Trello description explicitly calls itself a decomposition of the PDF.
Differences between its wording and the source transcription remain visible;
this audit does not silently choose the smaller interpretation.

## Requirement-to-delivery mapping

| Requirement | Delivered code/evidence | Remaining qualification or decision |
| --- | --- | --- |
| Use existing prerequisites | B1.1/#24 merged as `4b90b8d3`; B1.2/#25 as `fdf067fc`; F4/#11 as `84042a12`; A3/#34 as `da1fc473`. Main/integration are merged in `cd6515ac`. The integration reuses A3's accumulation/activation/RNG path. | The corrective evaluation passed and is committed at `76d2c42`; its formal run stopped incomplete at the undersized three-hour cap. Revision 3's 36-hour cap needs a new source/rule commit and fresh session. B1.3/B1.4 acceptance is not inferred. |
| Coordinate the B1 interface before final B1 measurements | The operator identified themselves as B1.1/B1.2 owner and delegated the technical decision based on evidence/requirements on 2026-10-29. The current contract below is accepted on that basis. | Pretrained public-trainer qualification remains pending. No retrospective coordination before historical measurements is certified. |
| Group weight restoration and expert matrix work | `expert_group.py`, `expert_group_decode.py`, `expert_group_gemm.py` and `expert_group_compute.py` implement the bounded grouped pipeline; `expert_group_autograd.py` supplies first-order reload/recompute; `expert_training.py` installs it through public SFT. | New public attention-LoRA trainer integration has synthetic qualification, but no pretrained trainer verdict yet. Backward GEMMs remain native one-expert recomputation; do not call them grouped backward GEMMs. |
| Source phrase: grouped **fused** dequant-GEMM | Grouped NF4 restoration and GEMM are separate stages. Optional Triton SiLU/multiply is activation fusion. Main's checkpoint-visible bitsandbytes NF4 linear path already exists and is reused for supported non-expert linears. | There is no new single packed-weight NF4-dequant/GEMM kernel over routed experts. Maintainer acceptance must explicitly resolve whether the staged grouped implementation satisfies the assigned wording or whether literal fusion is still required. The trainer gate cannot itself settle this. |
| Source page 10: do not build SwiGLU; wrap-only | The custom Triton activation study is retained as the operator-requested exploratory supplement, separate from core grouping. | It is not required D3 dequant/GEMM fusion and does not automatically satisfy the source's wrap-only constraint. Keep the distinction visible to the owner/reviewer; no production default selects it. |
| Never materialize a full dense MoE layer in the streamed candidate | One bounded shared arena, separate host staging, lease completion, pre-read/pre-allocation refusals and failure/poisoning tests. Backward saves gathered activations, not reusable weight views. | Verify actual budgets, lifetimes and peaks in the new pretrained trainer. Full resident expert stacks are allowed only in explicitly separate correctness-oracle processes, never in streamed timing processes. |
| Empty experts, variable rows, tails and partial groups | Existing CPU/CUDA routing, empty-token, shape/budget and derivative tests. The historical 69-case CUDA regression packet and later derivative/activation packets are preserved. | New trained runs must retain actual all-layer route/dispatch evidence. Do not modify routes after observing low grouped coverage or count native fallbacks as grouped execution. |
| Exact sequential loop and resident/streamed parity | Historical whole-forward 12/12 cells and complete trained-step 84/84 cells retain exact native oracles and effective controls. The corrective public-wrapper preflight passes 36 four-layer untied synthetic cases and exact sequential comparisons/reloads. | Qualify forward **and** backward, attention-LoRA gradients/updates, A/A, effective negative/restored controls and save/reload through the actual wrapper on pinned trained OLMoE, for ordinary SFT and L2L. The old post-embedding adapter fixture is not that gate. |
| Speed, memory and actual GPU launches caused by grouping | RTX3060 layer study: 60 arms, 1,200 samples, six profiles. Historical RTX4090 trained-step study: native versus Triton activation with the same grouped pool. | Measure sequential/grouped and grouped/Triton contrasts separately in the new public trainer, with the same card, fresh processes, F1 and device traces. Historical percentages are not new-host/full-trainer effects. A positive speedup is not a new acceptance condition. |
| Instrument validity and prospective decision rule | Reused F1 instruments pass actual-shard OK/OK/VOID controls on all 48 prepared expert shards; the complete bound input manifest contains 157 files. The earlier synthetic fixture remains separate. | Preserve the unchanged guest boot, storage identities and controls through source freeze/run; physical-host isolation/cache remain uncertified. |
| Reproducible publication | Existing archive/source hashes, exclusions and byte-identical analysis are preserved; current trainer CPU/CUDA preflight packet has normalized Git-byte checksum verification. | Retain every new cell, failure, tensor/oracle, profile, source blob and command; regenerate analysis and verify hashes from a fresh checkout. Keep old studies and failed attempts unchanged. |
| Repository task/issue and final review | PR #28 and this acceptance map retain the work. On 2026-10-29 the repository Issues UI shows disabled **New issue** and **Issue creation is restricted in this repository**. The prior HTTP 410 attempt remains recorded. | A maintainer must enable/authorize issue tracking or explicitly accept PR/Trello tracking in its place. No issue was created in a different repository. Required CI and full-task reviewer acceptance remain open. |

## Evidence already retained

- [Prospective layer comparison](../benchmarks/results/d3-grouped-layer-20261021/report.md):
  trained layers, cutoff-16 RTX3060 recipe; not public whole-trainer throughput.
- [Whole-model forward gate](../benchmarks/results/d3-full-forward-c94214d3/report.md):
  twelve trained-weight cells; forward correctness only.
- [Complete trained-step study](../benchmarks/results/d3-trained-step-0611f46b/report.md):
  84 cells, frozen base and post-embedding residual adapter, four SGD steps.
  It compares native/Triton activation under the same grouped policy. Whole-step
  peaks did not decrease; mixed timing differences are retained as measured.
- [Earlier synthetic public trainer preflight](../benchmarks/results/d3-trainer-integration-cuda-preflight-20261029/report.md):
  eighteen CUDA cases, 933 CPU passes, eight skips and 116 GPU deselections at that
  preparation source. Later checks are recorded separately from those old counts.
- [Public-trainer protocol and retained sessions](../benchmarks/protocols/d3-public-trainer/README.md):
  four-layer untied random fixture, 36 numerical cases, 18 CUDA profiles and 74
  corrective CPU passes. The earlier broad 999-case runtime selection was not
  rerun after the callback correction; all production source files are unchanged.
  The first frozen pretrained session remains failed with zero accepted cells.
  The second session at `76d2c42` reached the three-hour cap after 39 cells and
  has no complete timing block or throughput verdict. Revision 3 proposes a
  36-hour cap; it must be committed and frozen before another run.

No result above proves quality/convergence, 8 GiB model fit, a full platform
matrix, physical-host isolation or a general performance improvement. Quality
studies and giant-model runs are not invented as new D3 acceptance requirements.

## Current B1 interface decision

On 2026-10-29 the operator, identified as the B1 owner, explicitly delegated this
technical decision to the implementation agent based on results and conformity.
The following current contract is accepted: the existing storage format remains
compatible, the runtime owns lifetime/budget enforcement, unsupported operations
are refused, and selected CPU/synthetic CUDA checks support the shared boundary.
This records the present design decision, not a claim of historical coordination,
a pretrained-model verdict or a maintainer's final D3 acceptance.

1. Reuse the existing `per_expert` index, expert IDs, projection shapes and
   NF4/nested-code metadata. Accept separate and fused source projections in the
   sharder; do not silently reuse an incompatible cache. Preserve interrupted
   conversion invalidation and the existing B2 refusal order.
2. `training.stream_experts` is an explicit SFT opt-in. Unsupported architecture,
   routed-expert LoRA, nonzero router auxiliary loss and unsupported dtype/device
   are refused early. Existing forward-only B1 APIs keep their gradient guards.
3. The new runtime owns readers, the bounded width-1/2/4 pool, optional cuBLAS
   handle and process math policy. Weight views stay inside leases through device
   completion. Saved autograd state must not retain reusable weight slots.
   Validate all expert replacements before installing any and roll back on error.
4. Backward reloads/recomputes one frozen expert at a time and preserves native
   routing/route-weight/scatter derivatives. Only first derivatives of stateless
   SiLU experts are supported. This is the current implementation boundary, not
   approval for routed-expert optimizer states or general higher derivatives.
5. The public SFT wrapper remains the actual entry point; the A3 L2L trainer
   reuses its accumulation windows, ignored-label denominator and RNG replay.
   Ordinary and L2L remain separately qualified. The A3 Mistral noise-floor
   tolerance criterion is **not** imported into D3's byte-exact gate.
6. Arena, host staging, activation budget, BLAS call/workspace budget and total
   measured memory remain separate. `stream_source` selects the non-expert store;
   current routed-expert reads remain buffered disk reads and are reported as such.

The operator's delegation and this decision are retained in requirements.json;
they are not represented as a personally authored reviewer approval.

## Prospective completion sequence

1. **Preparation complete:** new public-trainer helpers pass the final random-weight
   CUDA launcher check (36 numerical cells, 18 profiles), broad selected CPU tests
   (999 passed, eight skipped) and separate planner/replay/fault controls. See the
   [protocol](../benchmarks/protocols/d3-public-trainer/README.md). The existing
   `d3_training_gate.py` runs a different adapter fixture and cannot substitute.
2. **Preparation complete:** checkpoint, public expert caches, exact inputs/labels,
   dtype-correct initial factors and actual-shard F1 mapping/controls are bound.
   No pretrained model forward or training was executed during initial preparation.
   The subsequent first formal attempt is retained separately as failed.
3. Promote the rule, commit the exact implementation, executable helpers and
   evidence, and export canonical Git blobs from a clean checkout. The operator
   performs commit/push. No formal training arm is authorized by a draft.
   Revision 1 completed that freeze but failed its first cell's evaluation.
   Revision 2 fixes the meta-input instrumentation bug and adds four-layer
   untied controls; no new pretrained session runs until that revision is committed.
4. Run the finite correctness/timing/profile series; preserve any failures and
   stop under its declared budget. Review the results without weakening exactness
   or promising a positive speedup/VRAM reduction.
5. Publish/replay the evidence, preserve the current B1 decision, resolve literal-fusion scope
   and repository issue tracking with the maintainer, pass current-head checks,
   and obtain full-task review. Do not infer task acceptance from green tests.
