# B3 gate: block-scaled FP8 source checkpoints

Branch `b3-fp8-source` on `SoupTeam/soup-rnd`, tree `71d4cd4` (base `9aa43bd`).
Harness: [`harness/fp8_source_probe.py`](harness/fp8_source_probe.py).
Card: B3 "Читать FP8 checkpoint и готовить NF4 веса" (rndTASK.pdf p. 5).

## 1. Rule, committed before the first real-shard run (2026-10-07)

Results are labelled **SYNTHETIC** (the CI fixtures in
`tests/test_fp8_source_checkpoints.py`) or **REAL** (DeepSeek-V3 source shards).
The two are never pooled.

**G1, bit-exact FP8 -> bf16.** For every float8 weight in the source, the new
path's output (`dequantize_fp8_blockwise`) must be `torch.equal` to an
independent decode: e4m3 read from its bit fields through a 256-entry table (no
`float8` cast), fp32 multiply by the block scale, one rounding to bf16. PASS iff
every tensor is equal. Any unequal tensor is FAIL, named.

**G3, NF4 reconstruction of what the sharder wrote.** For every float8 weight
the sharder quantised to NF4: read the packed nibbles and absmax back from the
layer shard, dequantise with bitsandbytes, and compare with the G1 reference.
Per NF4 codec block of 64 consecutive elements (not per FP8 128 x 128 block):
`e_b = max |n - r|`, `a_b = max |r|`, in fp32.
PASS iff `e_b <= 0.17 x a_b` in every block and every output is finite. A block
with `a_b = 0` passes only if every output in it is exactly zero.
Reported separately for double-quant on (Soup's default) and off.
**0.17 is fixed.** An exceedance is a finding, reported with tensor, block and
absmax spread, and the threshold is not raised. Rationale for the number:
design note section 3a (0.1519 NF4 half-gap + 0.0039 bf16 rounding, analytic
without double-quant; the double-quant remainder is an allowance, not a proof).

**G5, memory.** Peak RSS of the sharding process is recorded with the box. No
threshold: the per-chunk bound is asserted structurally in the tests; this row
records what the real run used.

**G6, sharding time.** Wall seconds for `shard_checkpoint(quant="nf4")` over
the source, divided by source GB (10^9 bytes), with the time split into FP8
dequantisation, NF4 quantisation and the rest. No threshold: a measurement,
valid only for the box it was taken on. Brief's bf16 baseline for scale only,
different box and source format: 138 GB -> 36.4 GB in 217 s.

**Validity.**
- V1: the box is stamped (CPU model, cores, RAM, disk, library versions, tree).
- V2: this run is on a shared cloud sandbox (2 vCPU, 7 GB RAM), not the team
  dev box. Its G6 seconds are not comparable to any dev-box number and say
  nothing about NVMe read rates. G1 and G3 are numerical and do not depend on
  the box.
- V3: a source file that splits a decoder layer across files yields a PARTIAL
  layer here; the record names it and does not count it as a full MoE layer.
- V4: a run that dies (OOM, killed) is kept and reported as void, not retried
  silently.

## 2. Box

_Filled from the harness JSON after the run._

## 3. Results

_Filled after the run._

## 4. Limits recorded for B2 (checked 2026-10-07, before the run)

- `soup train` refuses `model_type='deepseek_v3'` at the arch check, before
  sharding. End-to-end DeepSeek-V3 / K2 streaming needs B2.
- `shard_checkpoint` itself does not consult the arch; this record calls it
  directly.
- The MTP module (`model.layers.61.*`, `num_nextn_predict_layers: 1`) is sharded
  as an ordinary decoder layer (index `n_layers` 62 vs `num_hidden_layers` 61),
  carrying its own `embed_tokens` and `shared_head.head` into that layer shard.
- Dense layers (0 to 2) and MoE layers (3 to 60) produce different key sets.
- The pre-flight's shard-size estimate assumes a 16-bit source; for an FP8
  source it under-estimates the bf16 store by about 2x (planner, not B3).
