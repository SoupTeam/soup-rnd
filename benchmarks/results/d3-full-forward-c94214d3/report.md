# D3 whole-model forward gate: 12/12 passed

The operator committed source/protocol `c94214d397ffbe609916221e43fb2efd8c43ad4d` before the session.
A clean actual GitHub checkout verified its HEAD, source blobs, protocol,
immutable prepared weights, instrument receipts, GPU UUID and software before
execution. One session ran with zero replacements. All twelve fresh-process
cells passed; no results from the prior failed session were reused.

## Model and hardware

Pinned trained model: `allenai/OLMoE-1B-7B-0924`, revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`. Batch 1, 512 deterministic
synthetic token IDs, seed 1234. Every native embedding, attention, normalization,
router, head and all sixteen decoder layers execute. Only routed-expert compute
is substituted. BF16, NF4 and nested NF4 each use their own native resident oracle.

One RTX 3090 24 GB, UUID `GPU-5452dd2b-cfd5-1349-10e2-17d8ae60a428`; driver 580.159.03,
PyTorch 2.14.1+cu130, Transformers 5.19.0, bitsandbytes 0.50.2, Triton 3.8.0,
safetensors 0.8.0. One SSD suffices. Same cuBLASLt/no-reduced-BF16/no-split-K/
no-TF32 policy and eager attention, eval/no-grad, no KV cache in every cell.

## Results

| Stored format | Expert mode | Exact/control gate | F1 | Peak allocated GiB | Peak reserved GiB | Peak process RSS GiB | First-forward grouped matrices |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| dense | native | PASS | OK | 12.947 | 13.143 | 2.448 | — |
| dense | sequential | PASS | OK | 0.959 | 1.154 | 1.688 | — |
| dense | grouped-resident | PASS | OK | 12.951 | 13.145 | 2.444 | 332 |
| dense | grouped | PASS | OK | 0.998 | 1.191 | 1.692 | 332 |
| nf4 | native | PASS | OK | 12.947 | 13.143 | 1.943 | — |
| nf4 | sequential | PASS | OK | 0.963 | 1.158 | 1.689 | — |
| nf4 | grouped-resident | PASS | OK | 12.951 | 13.143 | 1.944 | 326 |
| nf4 | grouped | PASS | OK | 1.012 | 1.203 | 1.692 | 326 |
| nf4-dq | native | PASS | OK | 12.947 | 13.143 | 1.950 | — |
| nf4-dq | sequential | PASS | OK | 0.963 | 1.158 | 1.693 | — |
| nf4-dq | grouped-resident | PASS | OK | 12.951 | 13.143 | 1.950 | 322 |
| nf4-dq | grouped | PASS | OK | 1.012 | 1.203 | 1.694 | 322 |

Every cell retains finite full logits, exact native A/A and cross-mode equality,
the effective same-process active-expert negative, exact restored positive,
sixteen-layer sorted dispatch traces, and required F1 OK. The negative zeros
the entire down projection of one actually routed expert; NF4 uses its exact
zero code in every packed nibble. Immutable files and gate/up matrices stay
unchanged. Every streamed first positive loads each active expert once.

Grouped resident and streamed modes use the same `GroupedComputeExperts`
backend and execute actual grouped GEMM. The prospective policy dispatches
OLMoE-shaped rows through 128 to native mm, larger groups to grouped GEMM;
tiny inner dimensions retain cutoff 16. This is explicit shape dispatch,
not comparison-driven fallback. The shared streamed weight arena stays within
64 MiB, with a separate 4 MiB GEMM workspace and 256 MiB logical activation
bound. These bounds are not total VRAM; observed total allocator and RSS peaks
are recorded separately. Peaks cover validation/control execution and setup,
not a steady-state training step or complete system RAM.

## Retained evidence and replay

`raw-records.zip` contains all twelve result/log records, terminal parent
session, launch/attempt records, frozen source bytes, setup archive,
instrument/bundle identities, and an independent audit. Its internal manifest
also binds the complete raw archive and the three saved native logit tensors.

The complete raw archive `d3-full-retained-c94214d3.zip` is
130,567,268 bytes, SHA-256
`226f6c86e447ad70c2faecbc43df1a7dacc49b804b1e9f011cbaa01bd66501f2`. It includes all three full
native logit tensors and is retained on the instance and downloaded outside
Git to `C:/ml/soup/work/`. Both ZIP CRCs and every inner hash were verified
locally. The records ZIP omits only those three tensors; its manifest and
receipt retain their exact identities. Do not destroy the local full backup.

From a checkout containing the pinned commit, extract the complete archive
into a new directory. With CPU PyTorch and safetensors installed:

```bash
python verify.py --session EXTRACTED/session --checkout /path/to/soup-rnd --out replay.json
```

`replay.json` must match `analysis.json` byte-for-byte. The local replay checked
all twenty frozen blobs, twelve distinct process IDs, prepared/oracle identities,
sixteen-layer routes, exact output hashes, changed negative hashes, weight
budgets and positive grouped coverage. This replay audits evidence; it does
not perform another CUDA model execution.

## Previous failure and limits

The [first series at c4900bc9](../d3-full-forward-c4900bc9/report.md) stopped
after five passes because its NF4 sequential one-byte negative was ineffective.
It remains FAIL/incomplete, with zero replacements and all original records
preserved. Revision 2 was developed after that observation, validated on CPU
and tiny CUDA models, then committed prospectively before this entirely new
twelve-cell session. It cannot retrospectively repair the original run.

This completes the pinned whole-model **forward correctness/capacity gate**.
It does not establish backward, LoRA updates, trainer/CLI support, convergence,
real-text quality, throughput, 8 GB whole-model fit, or general bit exactness
under other math policies. The earlier RTX3060 layer study used cutoff 16;
its speed measurements cannot be assigned to this cutoff-128 recipe.
Exposed storage counters and visible GPU peers do not certify physical-host
isolation. Interface coordination with B1 and implementation/evidence review
remain separate; `d3_accepted` stays false pending sign-off.
