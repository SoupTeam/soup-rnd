# D3 public trainer qualification: attempt 03

## Scope

This is a completed exploratory CUDA series on the pretrained
`allenai/OLMoE-1B-7B-0924` checkpoint at revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`, measured from code commit
`f781e92726c9e6b524d75e7e7f3e1058350ba2c6`. The host used one RTX 4090 with
24 GiB VRAM. The recipe uses 512-token inputs, frozen routed experts, attention
LoRA, and ordinary or L2L training. It compares sequential, grouped, and Triton
expert execution for dense, NF4, and nested-NF4 weights.

This is a short fixed-recipe comparison, not a quality, convergence, full-model
training, or final D3 task-acceptance result.

## Completion and validation

- All **234/234** declared cells exited successfully: 36 correctness, 180 timing,
  and 18 profiler cells. Every recorded F1 outcome is `ok`.
- The six codec/training combinations each completed the prospective five-block
  symmetric repeat plan. The repetition calculation requested no added blocks.
- Offline replay verified **4,362 native tensor payloads**, cell result hashes,
  timing inputs and updates, adapter reload values, profile trace identities,
  and actual Triton kernel events. The analysis status is
  `qualification_complete_pending_independent_review`.
- The offline analyzer was corrected after the run to compare the session's
  serialized rule with the rule in the measured Git commit. The runner hashes
  the committed rule bytes, while its retained JSON copy is reserialized and
  therefore has a different byte hash. No measured source or result was changed.
  The analyzer regression suite passes **9 tests**.

## Results

Throughput is the median of the five paired-block summaries for each mode.
Percent changes are the protocol's paired median change in elapsed seconds per
input token; negative grouping values mean grouped was faster, while positive
Triton values mean Triton was slower than grouped.

| Weights / training | Sequential tok/s | Grouped tok/s | Triton tok/s | Grouped vs sequential time | Triton vs grouped time | Peak allocated MiB: seq / grouped / Triton |
|---|---:|---:|---:|---:|---:|---:|
| Dense / ordinary | 51.44 | 55.82 | 54.75 | -7.85% | +1.59% | 672.1 / 712.1 / 712.1 |
| Dense / L2L | 51.28 | 54.79 | 53.87 | -6.53% | +2.78% | 695.1 / 735.1 / 735.1 |
| NF4 / ordinary | 104.17 | 108.10 | 106.67 | -3.42% | +1.03% | 628.1 / 678.1 / 678.1 |
| NF4 / L2L | 112.23 | 112.64 | 113.76 | -1.01% | +1.27% | 651.1 / 701.1 / 701.1 |
| Nested NF4 / ordinary | 100.45 | 103.47 | 102.35 | -3.05% | +0.88% | 626.6 / 676.6 / 676.6 |
| Nested NF4 / L2L | 106.95 | 110.45 | 109.07 | -3.34% | +1.03% | 649.6 / 699.6 / 699.6 |

Grouped execution reduced paired median time by **1.01–7.85%** relative to
sequential execution in this recipe. Triton activation was **0.88–2.78% slower**
than grouped execution by the paired comparison. Its measured PyTorch peak
allocated memory matched grouped execution in all six combinations; it did not
provide a VRAM reduction here. These allocated/reserved figures are PyTorch
allocator measurements, not total device VRAM certification.

## Limits and next action

These results describe one pinned checkpoint, recipe, seed/session, GPU host,
and software stack. They do not establish general speedups or quality, backward
convergence, an 8 GiB fit, or production defaults. The run is complete and no
additional repetition is requested by the frozen rule. Independent review and
the task owner's separate acceptance remain pending. The earlier failed formal
launch is retained separately and is not counted among these 234 cells.

## Raw archive

The full session ZIP contains 6,120 entries (6,119 payloads plus its manifest) and passed both host-side and local ZIP integrity checks. Its SHA-256 is
`73383ef1cc1cdd9a2182dbdf9f7ed639eb7c0238074a4c46f2a68af9bf285b0a`; the
payload hashes are listed in `raw-archive-manifest.json`. The archive is retained on Vast at `/root/d3-public-trainer-results-f781e927-01.zip` and locally outside Git at `C:\\ml\\soup\\work\\d3-public-trainer-f781e927\\d3-public-trainer-results-f781e927-01.zip`. Local byte length and SHA-256 match the server copy; `zipfile.testzip()` passed all 6,120 entries. The payload manifest is `raw-archive-manifest.json`; the local retention receipt is `raw-archive-retention.json`.
