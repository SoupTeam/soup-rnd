<!--
Working measurement record, published verbatim. The decision rule in §3 is fixed before the
GPU run and is committed in the branch's pre-GPU commit, before the first GPU arm, which is the
point of it: a rule written after the numbers are in is not a rule.

Measured on research/g4-l2l-activation-compression (cut from origin/main fdf067fc) at
b195da29; this pull request carries the harness, the rule with its post-run V15 fix and the
compact GPU record on current origin/main. src/ is unchanged; everything is under benchmarks/
and tests/.
Harness: benchmarks/harness/g4_activation_codec.py (codecs, error metrics, the store),
g4_data.py (pinned Dolly, C5's split, packing), g4_probe.py (error / train / suites / timing
modes), g4_timing.py (A-B-B-A driver), g4_rule.py (the verdict from the JSON alone),
benchmarks/g4-run.sh (the GPU runbook).
-->

# Probe: G4, compressed L2L boundary activations (int8 / FP8 e4m3 with per-row scales)

**Status: GPU run done 2026-10-24 (commit `b195da29`, one RTX 3090), verdict `PASS: fp8_e4m3`,
int8 FAIL (§8).** The rule in §3 was fixed and committed before the first GPU arm. The
2026-10-25 analyzer fix (§3.4 V15) adds completeness and finiteness checks and leaves the
result unchanged. The CPU pilot in §9 is labelled SYNTHETIC/CPU PILOT and gives no verdict.

---

## 0. The question

L2L (Track A; `probe-rtx5070-l2l-step0.md`) parks the input of every decoder layer for every
micro-batch in host RAM between the forward walk and the backward walk: `n_layers x k` chunks
of `(1, seq, hidden)` in bf16. At a fixed host-RAM budget, those bytes cap the tokens per step.
On the giants this is 61 x 7168 x 2 B = **874,496 B per token** (Kimi K2, DeepSeek-V3).

G4 stores each chunk compressed instead: one row is one token's hidden vector; each row gets
an absmax scale, and the values are stored as int8 (symmetric, 127 levels a side) or as
`torch.float8_e4m3fn` (the absmax scaled to 448). The backward restores the chunk and
recomputes the layer from it. The forward is untouched, so the losses of the step are exact.
The error enters once per layer, through the recompute.

This probe asks:

1. How large is the error after restore, per layer?
2. How much does one recompute from the restored boundary amplify it?
3. Do the LoRA gradients of one L2L step at k = 4 still agree with exact L2L?
4. Does a 394-step Mistral-7B NF4 LoRA run with the compressed store reach the exact store's
   held-out task loss and gate-suite scores, within the run-to-run noise?
5. What do the compressed bytes, the compress and restore time, and the step time cost?

It is **not bit-exact by construction**, so it is quality-gated, like G1 (house rule 2).
A smaller store is never read as a faster step: tokens per step at fixed RAM is arithmetic
(§6), and step time is a separate, informational measurement (§3.5).

**Card step 1 (a working L2L boundary store) is satisfied by the existing
`l2l_activations.ActivationStore`.** Evidence, on CPU:

- `tests/test_g4_codec.py::test_identity_codec_is_a_null_control_through_l2l`: on the tiny
  streamed Llama of `test_l2l_probe_harness.py` (3 layers, k = 4), L2L through the plain store
  gives LoRA gradients and per-micro-batch losses `torch.equal` to gradient accumulation, the
  gradients are non-zero, and the identity codec through `CodecActivationStore` gives
  gradients and losses `torch.equal` to the plain store.
- The CPU pilot (§9) repeats it on SmolLM2-135M (30 layers, k = 4, seq 256, real weights):
  GA == L2L, the recording store == the plain store and identity == plain, in both sets.
- On CUDA, A0's G1 already passed GA == L2L on Mistral-7B at k = 4
  (`probe-vast-rtx3070-l2l-step0-a0.md`). The error mode re-checks it as validity row V2.

## 1. The codecs and the store

`benchmarks/harness/g4_activation_codec.py`:

| codec | payload | scale | element error bound (over the row absmax) | role |
|---|---|---|---|---|
| `bf16` | the activation's own dtype, byte for byte | none | 0 | exact arm; null control |
| `int8` | `round(x / absmax * 127)`, int8 | float32 absmax per row | 1/254 + 2^-8 | candidate |
| `fp8_e4m3` | `(x / absmax * 448).to(float8_e4m3fn)` | float32 absmax per row | 2^-4 + 2^-8 | candidate |
| `int4` | `round(x / absmax * 7)`, two nibbles per byte | float32 absmax per row | 1/14 + 2^-8 | positive control, never trains |

The `2^-8` in each bound covers the bf16 rounding of the restored value.

- **Scale dtype: float32**, storing the row absmax itself; the level count is a constant of the
  codec. float32 holds every bf16 value exactly, so the absmax round-trips exactly and the
  largest element of every row is restored exactly, a massive activation included. Dividing
  first (`x / absmax * levels`) keeps the intermediate in [-1, 1], so a bf16 extreme
  (3.39e38) or a denormal absmax (9.2e-41) neither overflows nor underflows. The cost is 4 B
  per row: 0.1% of a 4096-wide int8 row.
- **Zero row:** absmax 0 is stored as 0, the payload is 0, and the row restores to exact
  zeros.
- **NaN / Inf: refused, never encoded.** A row with a NaN or an Inf has no absmax to scale
  by. `encode` raises `NonFiniteActivationError` (naming how many rows). Inside the store
  the check stays on the device during the walk (`isfinite` of each scale, AND-ed) and is
  read once in `end_step`, which raises `NonFiniteActivationError` naming the first
  (layer, micro-batch). `L2LStep.run` calls `end_step` before it returns, so a step with a
  non-finite boundary never reaches the optimizer. The identity codec stores whatever the
  forward produced, as the plain store does.
- **Where it runs:** compression on the compute device before the device-to-host copy, and
  restore on the compute device after the host-to-device copy. Timed with CUDA events on CUDA
  (resolved after the step's own synchronize) and `perf_counter` on CPU. The only host read
  the store adds is the finite check at `end_step`, after the backward.
- **The store:** `CodecActivationStore` has the `ActivationStore` interface (`put`, `get`,
  `begin_backward`, `end_step`, `close`, `stats`, `k`, `n_layers`, `chunk_bytes`), so
  `l2l_schedule.L2LStep.run` drives it unchanged. It also exposes `payload_chunk_bytes`,
  `scale_chunk_bytes`, `codec_stats` and `host_bytes()`. The payload goes through an
  unmodified inner `ActivationStore` of the payload dtype and shape. The scales sit in ONE
  float32 slab of shape `(n_layers, k, rows)` (pinned when the store pins, zero-initialised)
  and never spill. If allocating the slab fails, the constructor closes the inner store
  (joins its spill threads, removes its spill file) before re-raising.
- **Spill:** only the payload spills, through the inner store and its real backends
  (`DirectSpillFile`, unbuffered; `BufferedSpillFile` for CPU tests). **A spilled payload
  must be a multiple of 4096 B; anything else is refused at construction with a
  `ValueError` (before any spill file exists), never padded.** So the spill file holds
  exactly `spill_layers x k x payload_chunk_bytes`, and no chunk carries unaccounted
  padding. A shape that is refused for spill still runs all-resident. The CPU tests
  round-trip every codec through both backends, two steps, and check that `close`
  (including after a failed write) joins the threads and removes the file. No spill arm
  runs on the GPU (§4).
- **Byte accounting:** `payload_chunk_bytes(shape, codec, *, source_element_bytes=2)` =
  `prod(shape[:-1]) x ceil(hidden / 2)` for int4 (an odd hidden size carries one real pad
  nibble), `prod(shape)` for int8/fp8, `prod(shape) x source_element_bytes` for the identity;
  `scale_chunk_bytes(shape, codec)` = `4 x prod(shape[:-1])` (0 for the identity);
  `compressed_chunk_bytes(shape, codec, *, source_element_bytes=2)` = their sum, which is
  `store.chunk_bytes`. On CPU float32 the identity needs `source_element_bytes=4`. CPU tests
  measure these from `untyped_storage().nbytes()` of the store's real tensors (odd int4
  widths, bf16 and float32 sources). `host_bytes()` reports the resident payload, the scale
  slab, the spill file size (`os.path.getsize`), and the staging rings (measured from the
  ring buffers' storages: `2 x ring x (payload + 4096)`), apart from the per-token figure.
  These are requested bytes; pinned power-of-two rounding (#901) is A4's to add.
- **A4 contract:** A4's `plan_budget(budget_bytes, *, n_layers, k, chunk_bytes,
  pinned_floor_bytes=0, ring=8, pin=True, other_pinned_bytes=0, spill_free_bytes=None)` takes
  `chunk_bytes = store.payload_chunk_bytes` and `pinned_floor_bytes =
  store.scale_chunk_bytes`, both unrounded. The scales are ONE allocation of `n_layers x k x
  scale_chunk_bytes` (rounded once by the pinned allocator), not `n_layers x k` small ones;
  A4Review agreed on 2026-10-22 to model the floor that way. Neither module imports the
  other. The end-to-end composition smoke lives on the A4 branch (`a4_g4_compose.py`): real
  G4 sizes into A4's planner, a G4 store on the real unbuffered `DirectSpillFile`, 8 shapes x
  4 budgets, every allocation, spill file size, bytes moved and payload round trip equal to
  the plan; the unaligned int4 shape is refused by both (CPU only, 2026-10-22, against this
  branch's `g4_activation_codec.py` sha256 `2403acfb...`).

On the 7B fixture shape (1, 512, 4096): bf16 4,194,304 B per chunk; int8 and fp8
2,097,152 + 2,048 = 2,099,200 B; int4 1,048,576 + 2,048 = 1,050,624 B. The int8/fp8 payload is
exactly 2 MiB (a sector multiple, so it may spill) and the k = 4 scale slab
32 x 4 x 512 x 4 B = 256 KiB, both powers of two, so the pinned allocator's rounding (#901)
adds nothing at this shape. On the CPU pilot's SmolLM2 shape (1, 256, 576) the int8 payload
is 147,456 B = 36 sectors and the 30 x 4 scale slab 122,880 B, which the pinned allocator
would round to 131,072 B.

## 2. Fixtures, arms and order

**Model.** `unsloth/mistral-7b-instruct-v0.3` at revision
`60b99dcbc7f853b7cd18472a607e8d045b96f03e` (A0's F1), loaded through the shipped streamed
runtime (`stream_probe.build`): NF4 with double quant, RAM tier, 2 layer buffers, read-ahead 2.
32 layers, hidden 4096. File SHA-256, checked by `g4-run.sh` before any arm:

| file | SHA-256 |
|---|---|
| `model-00001-of-00003.safetensors` | `ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19` |
| `model-00002-of-00003.safetensors` | `8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0` |
| `model-00003-of-00003.safetensors` | `905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca` |
| `config.json` | `9262d15f830e511429ccd175a71680e51c656ef3a4aa7460f3e9596508d32487` |
| `tokenizer.json` | `60b945759e27a63c3c5c0ca675881f5a73b4aa38b5d1d6818570308d4f1a3c59` |
| `tokenizer_config.json` | `b0c776216a54c6d031866d1dff0b31715bd73f5ba87f8a30eb35e8c603dff95d` |

**Data.** `databricks/databricks-dolly-15k` at revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, file `databricks-dolly-15k.jsonl`, SHA-256
`2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`; `g4_data.py` refuses any
other file. C5's split (`gate-c5-q2-0-experts.md`): rows sorted by SHA-256 of
`f"{i}\n{instruction}"`, the first 512 are the **held-out panel**, the next 4000 the **train
split**. The split digests must equal C5's (heldout `b4b2ab93…2d2a37`, train `a915bbbf…fa0b7`),
so G4 reads exactly the rows C5 read, and no panel row is ever trained on.

- Each row becomes an alpaca conversation via Soup's `format_to_messages` and is rendered by
  the model's chat template.
- **Train:** the rendered conversations in split order, tokenized into one stream and cut into
  rows of exactly 512 tokens (the remainder is dropped; no padding, no attention mask). The loss
  covers every token (`labels == input_ids`, as `L2LStep` computes it). With the pinned
  tokenizer: **1579 rows**, so 394 steps at k = 4 (806,912 tokens).
- **Panel:** one conversation per example, truncated to 1024 tokens. Task loss = token-weighted
  NLL of the response tokens (everything after the tokenized prompt), batch 1 through the
  shipped streamed forward. 512 examples, 41,990 response tokens. The first 128 examples
  (20,329 response tokens) form the **curve panel** at checkpoints.

**Training config (every arm).** LoRA r 16, alpha 32, dropout 0, targets q/k/v/o, adapter
seed = the arm's seed. AdamW, lr 2e-4, weight decay 0.01, linear warm-up over 3% of steps then
half cosine to 0 (HF semantics), grad-norm clip 1.0. k = 4 micro-batches of 1 x 512 per step,
one optimizer step per L2L step. `--deterministic` on (cuBLAS workspace `:4096:8`).
394 steps, the train split once. Curve-panel checkpoints at steps 100, 200 and 300. The data
order is fixed and does not depend on the seed.

**Why 394 steps.** One pass over the C5 train split gives a loss curve long enough to show
drift: it covers warm-up, the bulk of the cosine and its tail. Expected time per arm on an RTX
3090-class card **[estimate, not measured]**: the RTX 5070 Laptop ran a Mistral-7B NF4 RAM-tier
step of 1 x 512 in 1.053 s (`probe-rtx5070-what-bounds-streaming.md`), so k = 4 is about 4.2 s
at that rate. 394 x ~4.5 s is about 30 min, plus about 5 min of panel passes and the
build: about 40 min per arm, inside the hour. Time never enters the verdict; a slower box
just runs longer.

**Arms** (all on the same box, same commit, same packages):

| # | file | what | process |
|---|---|---|---|
| 1 | `error.json` | the error experiment (§3.2) on 2 sets of k = 4 train rows (rows 0-7) | one, deterministic |
| 2 | `train-bf16-seed1.json` | exact arm: plain bf16 store through `CodecActivationStore` | fresh |
| 3 | `train-int8-seed1.json` | int8 store | fresh |
| 4 | `train-fp8_e4m3-seed1.json` | fp8 store | fresh |
| 5 | `train-bf16-seed2.json` | **noise-floor arm**: exact, second adapter seed | fresh |
| 6-9 | `suites-<arm>.json` | Soup's four mini gate suites on each arm's saved adapter | fresh each |
| 10 | `timing.json` + `runs/` | A-B-B-A timing, bf16 against int8, then against fp8 | fresh per run |

Arms 2-5 differ only in `--codec` (2-4) or `--seed` (5 against 2); V9 checks this from the
recorded config.

**The error experiment, per set** (`g4_probe.py --mode error`), all on the same micro-batches:
GA over the k micro-batches; exact L2L through the plain `ActivationStore`; the same through a
*recording store*, which at every `put` also restores each codec's round trip and runs the
layer about to be called on both the exact and the restored input (one recompute, under
`no_grad`; the layer is already loaded, so this loads nothing); the identity codec through
`CodecActivationStore` (null control); then int8, fp8 and int4 through `CodecActivationStore`.
Then a per-chunk micro-benchmark of compress, restore and the D2H copy on one real chunk
(layer 16, micro-batch 0), 50 reps after 5 warm-ups, CUDA events.

**Gate suites** (`--mode suites`): `mini_mmlu` (26), `mini_common_sense` (24),
`mini_instruction` (24), `mini_arithmetic` (40), scored by Soup's `score_bundled_suite` through
`make_generator` (greedy, `--max-new-tokens`, 64 by default), every output kept, as C5 scores
them. The streamed model has no KV cache, so generation runs on the **resident** base loaded
NF4 (double quant, bf16 compute) with the arm's saved adapter. The adapter's file hashes must
match the hashes recorded in its training JSON, or the mode refuses.

**Base parity (V13), the main review risk.** The arm trained on the *streamed* base (sharded
by Soup's sharder, NF4 per layer through `layer_shard`), and the suites score the *resident*
base (`from_pretrained` + `BitsAndBytesConfig`, NF4 with double quant). That these two are the
same model is an assumption until it is measured, so before scoring, the suites mode:

1. rebuilds the arm's panel with the resident tokenizer and refuses unless its id digest
   equals the training record's `panel_ids_sha256`;
2. checks the live LoRA tensors of the loaded PeftModel against the saved
   `adapter_model.safetensors`, tensor by tensor (`torch.equal` after undoing a widening
   cast);
3. scores the response NLL of the first `--parity-rows` (16) panel examples on the resident
   base with the adapter, and again with the adapter disabled (informational: how large an
   adapter-sized change is on those rows);
4. records all three in `base_parity`; `g4_rule.py` recomputes both task losses from the
   per-example data and compares them with the NLL the streamed model recorded for the same
   rows at the end of training (`final.per_example`).

The comparison is end of training against end of training, the same adapter tensors, the
same token ids. On the tiny float32 fixture (CPU test) the two agree exactly and both controls
(adapter off, one perturbed adapter tensor) move. On Mistral-7B NF4 on CUDA, the GPU run (§8)
measured it: the resident base reproduced the streamed NLL exactly on every arm's 16 parity rows.

**Timing** (`g4_timing.py`): one comparison per candidate; arm A = bf16 store, arm B = the
codec's store. Each run is `g4_probe.py --mode timing` in a fresh process under
`arm_validity.ArmWatch` (`suspend` and `power` required). Each run: 2 untimed warm-up steps
and 8 timed steps on fixed rows (step = `L2LStep.run` + `optimizer.step` + `zero_grad` +
synchronize; codec timers off). The run's value is the median of those 8 steps. Then 2 more
steps run with the codec's CUDA-event timers on, for compress and restore time per chunk and
per step. Blocks A-B-B-A follow the METHOD.md repeats rule (5-block pilot, then up to N, one
top-up), capped at 12 blocks per comparison. Void blocks are kept and replaced.

## 3. The decision rule, fixed before the GPU run

Implemented by `benchmarks/harness/g4_rule.py`, a pure function of the raw JSON in one results
directory, with a CLI (`python benchmarks/harness/g4_rule.py <dir> --json verdict.json`). The
thresholds of E1-E3, Q1 and Q2 were fixed in `g4_rule.py` before the CPU pilot's error numbers
existed; Q3 and V12 were added before any suite was scored; V13 (base parity) and its bound were
added in the 2026-10-22 review, before any parity number existed. None was changed after
seeing data. The rule is committed in the branch's pre-GPU commit, before any GPU arm.
A missing, unreadable or malformed input file fails its validity rows and never raises: the
CLI then prints `NO VERDICT`.

### 3.1 Values

- Per-token relative L2: `||x_hat_t - x_t|| / ||x_t||` for one token's hidden vector, in
  float64. **rms_row_rel** = the root mean square of that over every token of every
  micro-batch and set at one layer (each token weighs the same; a few massive-activation
  tokens would dominate the whole-tensor norm). **worst_row_rel** = its maximum.
  The whole-tensor relative L2 and max abs error are recorded too.
- **Branch** = layer output minus layer input, the residual update the layer computes. Its error
  is `branch(x_hat) - branch(x)`, measured the same way. The layer output itself contains
  `x_hat`, so its error mostly repeats the restore error and would hide any amplification.
  The output error is recorded, but it is not gated.
- **Gradient agreement**: over all LoRA gradient tensors of one step, compressed L2L against
  exact L2L on the same micro-batches. global_rel_l2 = `||g_c - g||` over all tensors
  concatenated, divided by `||g||`; per-tensor cosine and relative L2.
- **Task loss** TL(arm) = token-weighted response NLL over the 512-example panel after the
  last step.

### 3.2 Error gates (per candidate codec, from `error.json`)

| gate | passes when | bound | why this bound |
|---|---|---|---|
| E1 restore | on **every** layer: rms_row_rel ≤ 0.05 **and** worst_row_rel ≤ 0.25 | 0.05 / 0.25 | On seeded Gaussian rows of 4096, int8 gives 0.009 and fp8 0.027 (§9). 0.05 leaves room for heavy-tailed rows. 0.25 lets one massive-activation token keep its large channels while losing its small ones, but not lose the row. |
| E2 one recompute | on **every** layer: the branch's rms_row_rel ≤ 0.10 | 0.10 | Twice E1: a layer may amplify the restore error at most two-fold. More means the recompute's error grows with each layer that recomputes. |
| E3 gradients | in **every** set: global_rel_l2 ≤ 0.05 **and** min per-tensor cosine ≥ 0.99 | 0.05 / 0.99 | 5% is far below the difference between the exact gradients of two different sets of micro-batches (`between_sets`, recorded beside it), so the codec moves a step less than the choice of batch does. A cosine of 0.99 (about 8°) keeps every tensor's direction. |

### 3.3 Quality gates (per candidate codec, from the training and suites JSON)

Noise floor, from the two exact arms: noise = |TL(bf16 seed 2) − TL(bf16 seed 1)|.

| gate | passes when | margin |
|---|---|---|
| Q1 held-out task loss | TL(codec) − TL(bf16 seed 1) ≤ margin (one-sided: lower is fine) | max(noise, 0.005 x TL(bf16 seed 1)) |
| Q2 loss curve | mean over the last 25% of steps of (train loss codec − exact) ≤ margin | max(mean \|exact seed 1 − exact seed 2\| over those steps, 0.005 x their mean exact loss) |
| Q3 gate suites | in every gating suite: correct(exact) − correct(codec) ≤ margin | max(2, \|correct(exact seed 2) − correct(exact seed 1)\|) per suite |

- A suite the exact arm answers all right or all wrong is reported and not gated (C5, #1192).
- The 0.005 floor stops a lucky pair of seeds from shrinking the margin to near zero.
- The seed changes only the LoRA initialisation (the data order is fixed). The noise
  floor therefore misses the data-order part of training noise, and the gate is stricter
  than a full noise estimate would be: the conservative direction.
- The paired bootstrap of TL(codec) − TL(exact) over panel examples (2000 resamples,
  seed 17) is reported with Q1. It describes panel sampling, not training noise, and does
  not gate.

### 3.4 Validity rows (no verdict when any fails)

| row | condition |
|---|---|
| V1 error eligible | `error.json` is a CUDA run, `verdict_eligible`, not labelled SYNTHETIC/CPU PILOT |
| V2 harness exact | in every set: GA == exact L2L, recording store == plain store, identity codec == plain store; grads `torch.equal`, none all-zero, losses `torch.equal` |
| V3 forward exact | every codec's (int8, fp8, int4) per-micro-batch losses `torch.equal` to exact L2L (a store leaking into the forward fails here) |
| V4 codec conforms | on the real activations, every codec's worst element error over its row absmax is finite and within its bound in §1 |
| V5 control discriminates | int4 FAILS E1, E2 **and** E3 (otherwise that metric cannot tell a coarse codec from a good one) |
| V6 training eligible | the four training arms exist, each CUDA, `verdict_eligible`, not CPU-labelled |
| V7 arms complete | every arm ran all its steps with no `stopped`, every loss and grad norm finite, the first-step grad norm > 0, the final task loss finite |
| V8 same inputs | the same Dolly hash, split digests, train-row and panel id digests, harness hashes, git commit, package versions, GPU and model-file hashes in every arm; a model revision recorded |
| V9 configs | every arm file ran the (codec, seed) its name claims (`train-int8-seed1.json` must record codec int8, seed 1); the main-seed arms differ from the exact arm only in `codec`; the noise arm only in `seed`, and its seed differs |
| V10 first step exact | the first step's per-micro-batch losses are equal across the three main-seed arms (same init and data; the forward is exact before the first update) |
| V11 noise floor usable | the noise is ≤ 0.03 x TL(bf16 seed 1); a wider seed-to-seed spread is too coarse to gate a codec on |
| V12 suites records | one CUDA suites record per training arm, each scoring the adapter whose hashes its training arm recorded, all four suites with integer counts `0 ≤ correct ≤ n`, the same generation settings |
| V13 base parity | for every suites record: the live adapter equals its saved file (0 mismatched of > 0 tensors); its parity rows exist in the training arm's `final.per_example` with equal token counts; and \|TL_resident − TL_streamed\| / TL_streamed ≤ 0.0025 (= 0.005 / 2, half the Q1 floor) over those rows, both recomputed by the rule from per-example NLL |
| V14 error linked | `error.json` records its experiment config, and it and every training arm share the harness hashes, git commit, package versions, GPU, model-file hashes, Dolly file and split digests, and the streamed setup (`k`, `seq`, LoRA rank/alpha/dropout/targets, `quant`, `tier`, `dtype`, `deterministic`, model revision, device); E1-E3 and Q1-Q3 are then about one experiment |
| V15 error complete | (added 2026-10-25, after the GPU run, see below) the expected counts come from the run's own recorded protocol (`args.error_sets`, `n_layers`, `chunk_shape`, `config.k`), never from the arrays that survived: set ids exactly 0..sets-1; per codec (int8, fp8, int4) the layers exactly 0..n_layers-1, each merged over sets x k x tokens-per-chunk rows for restore, branch and output; per set, the three exact controls compare one tensor count and every codec's gradient agreement covers that many tensors (count and per-tensor table), with the same tensor names everywhere and a `min_cosine` equal to the smallest per-tensor cosine; every metric E1-E3 and V4 read is a finite number. Any failure: E1-E3 read no verdict with the reason, never pass |

V15 closes a hole the post-run audit found: E1-E3 aggregated with `max`/`min`, which skip a NaN
that is not the first value, so a NaN in the second error set could pass E3; a dropped set, a
missing or duplicated layer or a partial gradient table also passed. No threshold changed.
Replayed on the unchanged raw files of the GPU run, V15 holds and every gate number, gate
status and the verdict are identical (§8).

### 3.5 Informational (never changes a verdict)

- Host bytes per token per layer per codec, measured from the store's storages, next to the
  arithmetic; host bytes of the whole store; peak allocated and reserved VRAM; RSS and
  available RAM at the end of training (the available RAM is recorded for the comparison).
- Compress and restore time per chunk and per step (CUDA events), D2H copy per chunk.
- Step time and tok/s per arm. `timing.json`: valid blocks, the blocks the repeats rule needs,
  mean step time of A and B, and the ratio B / A. This is a time ratio and is never reported
  as a speed-up from the memory saving.
- The per-token arithmetic of §6.

### 3.6 Outcomes

Per candidate codec:

- **FAIL** if V1-V5 and V15 all hold and E1, E2 or E3 fails. A codec that fails its error
  bound fails whatever the training arms show.
- Otherwise **NO VERDICT** if any validity row fails. The affected gates read "no verdict",
  never pass.
- Otherwise **FAIL** if any of E1-E3 or Q1-Q3 fails, and **PASS** if all six pass.

Overall: `PASS: <codecs>` if at least one candidate passes, `FAIL` if both fail, else
`NO VERDICT`.

**What each outcome means for Track A.**

- **PASS for a codec:** Track A may store L2L boundaries in that codec, and A4 budgets host RAM
  with `compressed_chunk_bytes(shape, codec)`. That halves the boundary-activation store at a
  fixed shape (at 7B, k = 4, seq 512: 512 MiB to 256.25 MiB; not the whole host RAM or VRAM).
  Holding that store's RAM fixed, it is about 2x the tokens per step (§6) as arithmetic.
  Whether those tokens hide the read on the giants is Track A's measurement, not this one.
  The integration (A3/A5/A6, and A4's spill) re-gates on its own path with its own controls.
  Fewer bytes per step is not a faster step. If both codecs pass, the int8 / fp8 choice
  follows the measured compress and restore cost, and is reported, not gated.
- **FAIL for both:** Track A keeps the bf16 store. Per-row absmax at this granularity is
  closed for L2L boundaries. A finer scheme (per-block scales, or splitting off outlier
  channels) would be a new rule, not a retry of this one.
- **NO VERDICT:** nothing changes for Track A. Re-run the arms whose validity row failed;
  §7 keeps each step resumable.

## 4. What this does NOT measure

- **Speed from compression.** The timing arms measure what compression costs in step time.
  No claim is made that the halved bytes speed up a step, and the tokens-per-step factor in §6
  is arithmetic.
- **The giants.** K2 and DeepSeek-V3 appear only as labelled per-token arithmetic. Their
  activation statistics (massive activations, MLA) are not measured.
- **Spill with compression on the GPU.** The store keeps sector alignment and is CPU-tested
  through both real spill backends (`BufferedSpillFile` and the unbuffered `DirectSpillFile`
  on this box's NTFS), but no spill arm runs on CUDA. That is A4's track.
- **Pinned-allocator rounding.** Host bytes are the requested bytes. At the 7B shape both
  pieces are powers of two (§1), so rounding is zero there; on other shapes it is A4's to add.
- **A fused kernel.** Compression is a few torch ops per chunk; a fused kernel would cost less
  and is not built.
- **Breadth.** One model, one task, SFT with LoRA only, one pair of seeds for the noise floor,
  k = 4, seq 512. No DPO-family loss, no other k.
- **Long-run drift beyond 394 steps.**

## 5. Box state required

- Linux, one CUDA GPU with ≥ 8 GB free (A0 ran this 7B NF4 L2L on 8 GB; the shared 24 GB
  server is enough). Python 3.11 or 3.12, `torch==2.14.1` (cu130 wheel) and
  `benchmarks/a0-requirements.txt`, installed by `a0-bootstrap.sh`.
- RAM: about 4.2 GB pinned for the NF4 store plus 0.54 GB (bf16) or 0.27 GB (int8/fp8) for the
  activation store at k = 4, and the sharding pass once. **32 GB recommended.** Available RAM
  and RSS are recorded in every training and timing JSON.
- Disk: about 35 GB free (model 14.5 GB, NF4 shard store about 4.2 GB, venv about 8 GB, four
  adapters at about 55 MB each, results under 50 MB). No large SSD is needed.
- Book a measurement window on the shared server. Training arms do not need an idle GPU for
  validity (their numerics do not depend on peers), but every step stamps the GPU peers
  (`logs/<step>.box`). Timing runs need no other compute process on the GPU; a run with peers
  is kept and marked in the record, not silently dropped.
- Power: a server without a battery reads "no battery, on mains" (METHOD.md, Power). If
  `suspend` reads unknown inside a container, timing runs come back unknown, not void, and
  timing is informational anyway.
- Nobody else's process is killed to make room (house rule).

## 6. Per-token arithmetic (labelled estimates, not measured)

Host bytes per token over all layers = layers x (payload + scale) per row. Tokens per step at
a fixed budget = budget // that, before allocator rounding.

| shape | layers x hidden | bf16 B/token | int8 / fp8 B/token | factor at fixed RAM |
|---|---|---|---|---|
| Mistral-7B v0.3 (fixture) | 32 x 4096 | 262,144 | 131,200 | 1.998 |
| Kimi K2 | 61 x 7168 | **874,496** ("874 KB per token today") | 437,492 | 1.9989 |
| DeepSeek-V3 | 61 x 7168 | 874,496 | 437,492 | 1.9989 |

At a 64 GB (64 x 10^9 B) boundary budget: K2 / DeepSeek-V3 hold 73,185 tokens in bf16 and 146,288
in int8 or fp8. Mistral-7B holds 244,140 and 487,804. These are tokens per step that fit, not
tokens per second. On K2's (1, 512, 7168) chunk, the bf16 payload (7,340,032 B) and the int8
payload (3,670,016 B) both round up to the next power of two in the pinned allocator (+14%
each). The factor between them holds, and their absolute sizes are A4's to plan.

## 7. GPU runbook

`benchmarks/g4-run.sh`, from the repo root on a fresh Linux CUDA box:

```bash
git clone https://github.com/SoupTeam/soup-rnd && cd soup-rnd
git checkout research/g4-l2l-activation-compression   # the commit that holds this rule
bash benchmarks/g4-run.sh setup        # venv, A0 pins, CPU tests, CUDA codec round trip  ~15 min
bash benchmarks/g4-run.sh fixtures     # Mistral @60b99dc + Dolly @bdd27f4, SHA-256 checked ~10-20 min
bash benchmarks/g4-run.sh arms benchmarks/results/g4-<gpu>-<yyyymmdd>   # offline from here
```

`setup` and `arms` refuse to continue when torch sees no CUDA device, and `g4_probe.py` itself
exits 3 (not 0) when `--device cuda` has no CUDA, so no step can be marked done without its
JSON. `arms` runs each step logged and box-stamped, resumable: a finished step leaves
`logs/<step>.done`; a step whose JSON exists without that marker refuses to start until the
JSON is renamed; a step that exits 0 without writing its JSON fails the script.

| step | command (inside the script) | expected duration [estimate] |
|---|---|---|
| error | `g4_probe.py --mode error <COMMON>` (the first build also writes the NF4 shard store) | 15-25 min |
| train ×4 | `g4_probe.py --mode train <COMMON> --steps 394 --codec {bf16,int8,fp8_e4m3} --seed 1`, then `bf16 --seed 2` | ~40 min each, ~2.7 h |
| suites ×4 | `g4_probe.py --mode suites <COMMON> --arm train-<arm>.json` (base parity V13 on 16 panel rows, then the suites) | ~8 min each, ~30 min |
| timing | `g4_timing.py --codecs int8,fp8_e4m3` (≥ 5 blocks x 4 runs per codec, ~2 min a run) | 1.5-3.5 h |
| verdict | `g4_rule.py <dir> --json verdict.json`, then `sha256.txt` over the directory | seconds |

`<COMMON>` = `--weights $MODEL_DIR --model-revision 60b99dcbc7f853b7cd18472a607e8d045b96f03e
--shards $SHARDS --dolly $DOLLY --label g4` (the defaults supply `--quant nf4 --tier ram
--seq 512 --k 4 --lora-r 16`). Every flag the script passes exists in `--help` of the script
it calls; the review ran `--help` for each.

Total about 5.5-7.5 h of box time **[estimate]**. Peak VRAM is expected near A0's 4.1 GB for
training (A0 measured it at k = 16) and about 5 GB for the resident suites model **[estimate]**.

**Files that come back** (in the results directory): `error.json`, `train-{bf16,int8,fp8_e4m3}-seed1.json`,
`train-bf16-seed2.json`, `suites-*.json` (×4, each with its `base_parity` record), `timing.json`
with `runs/attempt-NN/run*.json` and `runs/attempt-NN/run*.log` (one `attempt-NN` per timing
attempt; an earlier attempt is never overwritten), `verdict.json`, `logs/` (`<step>.log`,
`<step>.box`, `git-head.txt`, `git-status.txt`, `pip-freeze.txt`, `verdict.log`) and
`sha256.txt`. The adapters are written outside the results directory, to
`.soup/g4/adapters/<dir>/`; their file hashes are inside each training JSON, and they are a
required export of the rental (V12 re-hashes them), not something to leave on the box. A
failed step leaves its partial JSON; rename it before re-running so the record keeps every
attempt.

## 8. Results

GPU run 2026-10-24, **measured at commit `b195da29`**: `g4-run.sh setup`, `fixtures`, `arms`,
unchanged (394 steps per arm, every threshold and control as in §3). One RTX 3090 24 GB (driver
595.58.03, CUDA 13.2), Linux container, Python 3.12.3, torch 2.14.1+cu130. One attempt per
step, none void or failed. The run's `verdict.json` came from `g4_rule.py` at `b195da29`; the
harness hashes inside the raw JSON are those of `b195da29` and stay as recorded.

**Verdict: `PASS: fp8_e4m3`.** int8 FAILS on E2 and E3. V1-V15 all hold.

**Reanalysis 2026-10-25** with the fixed analyzer (V15, `g4_rule.py` at commit `4dd530da`), on
the unchanged raw files (`sha256.txt` re-verified, 125 files), written to a new
file beside the original, which is kept: V15 holds (2 error sets of 4 micro-batches, all 32
decoder layers for int8, fp8 and int4, 4096 rows per layer, 256 gradient tensors in every
comparison, every metric finite). Every validity row, every gate status, every gate number, the
noise floor, memory and timing sections and the overall verdict are identical to the original.

| codec | E1 restore | E2 one recompute | E3 gradients | Q1 task loss | Q2 curve | Q3 suites | result |
|---|---|---|---|---|---|---|---|
| int8 | pass: rms 0.0480, worst 0.142 | **fail**: branch rms 0.174 (layers 2-7 over 0.10) | **fail**: global rel L2 0.455, min cos -0.056 | pass: +0.000858 (margin 0.007088) | pass: +0.000506 (margin 0.008539) | pass | **FAIL** |
| fp8_e4m3 | pass: rms 0.0264, worst 0.0285 | pass: 0.0322 | pass: 0.0356, min cos 0.9973 | pass: +0.000861 (margin 0.007088) | pass: -0.000725 (margin 0.008539) | pass | **PASS** |

- **Q1 for fp8_e4m3:** task-loss delta +0.000861 against a margin of 0.007088 (the 0.005 floor
  of the exact loss 1.417643; the seed-to-seed spread, 0.000553, is smaller). The paired
  bootstrap over panel examples (informational, §3.3) gives a 95% interval of about
  [+0.000075, +0.00163]: the fp8 arm's panel loss is slightly higher, and that interval
  excludes zero. PASS means the delta is inside the margin fixed before the run. It is not
  evidence of no loss, and the delta is not inside the seed noise (it is above the 0.000553
  spread of one seed pair).
- Q3, correct answers exact / int8 / fp8 (bundled mini suites): mmlu 25 / 24 / 25 of 26,
  common sense 22 / 22 / 22 of 24, instruction 22 / 23 / 24 of 24, arithmetic 27 / 27 / 28 of 40.
- int8: its error per restore is within E1, but the one-recompute branch error on layers 2-7
  and the gradient direction are not. Its training-quality gates passing does not rescue it
  (§3.6: a codec that fails its error bound fails).
- V13: on the 16 parity rows of every arm, the resident NF4 base with the adapter reproduces the
  streamed model's NLL exactly (max per-token difference 0.0). 256 of 256 live adapter tensors
  equal their files. The adapter-off control moves the loss (1.185 to 2.176 on arm 1).
- V10: the first-step per-micro-batch losses of bf16, int8 and fp8 are identical.
- Memory: host bytes per token per layer, measured from the store, bf16 8192 and int8/fp8 4100,
  equal to the arithmetic. The **boundary-activation store** drops from 512 MiB to 256.25 MiB
  at k = 4, seq 512. That is the only memory the codec changes: the NF4 weight store and the
  rest of host RAM are untouched, and peak allocated VRAM is 1.43 GB in every arm.
- Timing (informational, never a verdict): 5 valid A-B-B-A blocks per codec, more than the 2
  the repeats rule needed. Observed mean step-time ratio codec / bf16: int8 1.049, fp8 1.023.
  The repeats rule sizes the comparison to detect a 10% effect, so these two ratios are
  observations from this box, not a measured general cost of compression; no speed-up is
  claimed.
- Validity of the timing runs: `suspend` ok and `power` ok in all 40 runs. `power` here only
  says the box was on mains (no battery). It says nothing about GPU throttling: the
  `gpu_clock` record of the same runs shows NVML clock-event reasons `sw_power_cap` (37
  samples) and `sw_thermal_slowdown` (2), spread over both arms, with the SM clock at
  1800-1950 MHz for the median sample. The A-B-B-A order cancels a linear drift between arms;
  it does not remove these events.
- Step time in the training arms (`step_s` in each training log, 394 steps): median 3.33 s
  (bf16 seed 1), 3.44 s (int8), 3.24 s (fp8), 3.22 s (bf16 seed 2). Wall time per arm 1408-1488 s;
  the whole `arms` stage took 2 h 33 min.

What this establishes is limited to one model (Mistral-7B v0.3 NF4), one task (Dolly SFT with
LoRA), one GPU type, k = 4, seq 512 and 394 steps; §4 lists what it does not measure. In Track A
terms (§3.6): Track A may store L2L boundaries in fp8_e4m3 with per-row scales; int8 at this
granularity stays closed. No spill with compression ran on the GPU; the A4 integration needs
its own controls.

## 9. CPU pilot (SYNTHETIC/CPU PILOT, excluded from any verdict)

Run 2026-10-22 on the development laptop (Windows 11, 12 logical CPUs, 15 GB RAM; local GPU
not used), commit `fdf067fc` plus this branch's uncommitted harness, Python 3.12.10, torch
2.14.1+cpu, transformers 5.19.0, peft 0.21.2. Model `HuggingFaceTB/SmolLM2-135M-Instruct` at
`12fd25f77366fa6b3b4b768ec3050bf629380bac` (30 layers, hidden 576), streamed, `--quant none`,
float32, seq 256, k = 4, LoRA r 8, the pinned Dolly and C5's split. The exact command of each
run is in its JSON's `meta.args`; the sequence is below. Its output (`g4-cpu-pilot/`, every
JSON labelled `SYNTHETIC/CPU PILOT smoke` with `verdict_eligible: false`) is kept on the
research branch `research/g4-l2l-activation-compression`, not in this pull request: it holds
the laptop's local paths and gives no verdict. Total wall time 1665 s.

```bash
export HF_HUB_OFFLINE=1 PYTHONPATH=src
SNAP=~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac
OUT=benchmarks/results/g4-cpu-pilot
C="--device cpu --weights $SNAP --model-revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
   --shards .soup/g4-smollm2-shards --quant none --seq 256 --k 4 --lora-r 8 --label smoke"
python benchmarks/harness/g4_probe.py --mode error $C --error-sets 2 --bench-reps 20 --out $OUT/error.json
for arm in bf16:1 int8:1 fp8_e4m3:1 bf16:2; do c=${arm%%:*}; s=${arm##*:}
  python benchmarks/harness/g4_probe.py --mode train $C --codec $c --seed $s --steps 6 \
    --eval-every 3 --curve-panel 8 --panel-limit 16 \
    --adapter-out .soup/g4-cpu-pilot-adapters/train-$c-seed$s --out $OUT/train-$c-seed$s.json
  python benchmarks/harness/g4_probe.py --mode suites --device cpu --weights $SNAP --quant none \
    --label smoke --arm $OUT/train-$c-seed$s.json --out $OUT/suites-$c-seed$s.json
done
python benchmarks/harness/g4_probe.py --mode timing $C --codec fp8_e4m3 --steps 2 --warmup 1 \
  --out $OUT/timing-fp8_e4m3-run.json
python benchmarks/harness/g4_rule.py $OUT --json $OUT/verdict.json
```

As run on 2026-10-22, before the review. With the reviewed harness the suites command
additionally rebuilds the panel for V13; it reads the Dolly file from `--dolly` or, when that
is absent, from the training arm's recorded `--dolly` and otherwise from the HF cache at the
pinned revision (`g4_data.source_path`).

**What it shows: the pipeline runs end to end, and the rule refuses it.**
`g4_rule.py` gives `Verdict: NO VERDICT` (V1, V6 and V12: "device 'cpu' is not a verdict
run"). Every other validity row of the time was ok: V2 (GA == L2L, recording == plain,
identity == plain, in both sets), V3 (every codec's losses `torch.equal` to exact), V4, V5
(int4 fails E1, E2 and E3), V7-V11. Replayed with the reviewed rule (§11) the pilot also
fails V13, because these suites records predate the base-parity check; the verdict is the
same NO VERDICT.

The numbers below are float32 activations of a 135M model over 8 rows. They do not predict
Mistral-7B in bf16, and they were seen after the bounds in §3 were fixed.

Error experiment (2 sets x k = 4, max over 30 layers):

| codec | restore rms row rel | worst row rel | element / absmax | branch rms row rel | branch / restore, median (max) | grad global rel L2, set 0 / 1 | min cosine, set 0 / 1 |
|---|---|---|---|---|---|---|---|
| int8 | 0.0404 | 0.0547 | 0.00394 | 0.1262 | 2.65 (4.6) | 0.1016 / 0.1089 | 0.9421 / 0.9490 |
| fp8_e4m3 | 0.0261 | 0.0332 | 0.0357 | 0.0447 | 1.67 (2.0) | 0.0520 / 0.0573 | 0.9917 / 0.9912 |
| int4 | 0.5444 | 0.6278 | 0.0714 | 0.9678 | 1.95 (2.37) | 0.8598 / 0.8713 | −0.4485 / −0.4087 |

- On this model int8 restores less accurately than fp8 per token (0.040 against 0.026),
  although its element error is 9x smaller. On i.i.d. Gaussian rows of 4096 the order is the
  reverse: int8 0.0090, fp8 0.0265, int4 0.159. With one outlier channel per row at 60 (about
  15σ), int8 rises to 0.099 and fp8 to 0.019. SmolLM2's boundaries have such outlier
  channels. A uniform grid scaled to the outlier loses the small channels; fp8's relative
  grid keeps them. Mistral-7B's boundaries decide which effect wins, and E1-E3 measure that.
- The recompute amplifies the restore error 1.7x (fp8) to 2.7x (int8) at the median layer.
  This is the effect E2 bounds.
- For scale, the two sets' exact gradients differ by global rel L2 0.65, cosine −0.05
  (`between_sets`): different micro-batches, so this bounds nothing.

Training (6 steps, warm-up 1, curve panel 8 examples at step 3, final panel 16 examples,
982 response tokens; adapters saved to `.soup/g4-cpu-pilot-adapters/`, not committed):

| arm | final task loss | step losses (sum of 4 micro-batches) | measured B / token / layer | store host bytes | compress / restore s (720 chunks each) |
|---|---|---|---|---|---|
| bf16 seed 1 | 2.157059 | 3.24348 2.65361 3.04680 3.22245 2.94419 2.47546 | 2304.0 (float32 x 576) | 70,778,880 | — |
| int8 seed 1 | 2.157263 | 3.24348 2.65361 3.04678 3.22258 2.94412 2.47508 | 580.0 | 17,817,600 | 0.416 / 0.161 |
| fp8 seed 1 | 2.157002 | 3.24348 2.65361 3.04681 3.22246 2.94425 2.47548 | 580.0 | 17,817,600 | 0.415 / 0.258 |
| bf16 seed 2 | 2.156920 | 3.24348 2.65361 3.04684 3.22194 2.94399 2.47498 | 2304.0 | 70,778,880 | — |

- Measured bytes per token per layer equal the arithmetic in every arm: 576 + 4 = 580 B for
  int8 and fp8, against 2304 B for the float32 identity (4 B per element on this CPU model; on
  CUDA the identity is bf16, 2 B).
- The first step's per-micro-batch losses are equal across all main-seed arms (V10).
- All four arms score the same on the gate suites: `mini_mmlu` 7/26, `mini_common_sense` 2/24,
  `mini_instruction` 13/24, `mini_arithmetic` 14/40. Six steps barely move a 135M model.
- Noise floor from the two exact arms: 0.000139, so the task-loss margin is the 0.005 floor
  (0.010785).

Timing smoke (one process, fp8 store, 1 warm-up + 2 timed steps, perf_counter): median 7.42 s
per step. Instrumented steps: compress 0.51 ms and restore 0.33 ms per chunk, 120 chunks per
step. CPU timings do not transfer to CUDA, and this single run is not an A-B-B-A block.

Per-chunk micro-benchmark (one chunk of 256 x 576, 20 reps, perf_counter, medians): int8
compress 0.24 ms, restore 0.14 ms; fp8 0.31 / 0.26 ms; int4 0.48 / 0.36 ms. Payload
147,456 B (int8 / fp8) and 73,728 B (int4) against 589,824 B float32, plus 1,024 B of scales.

## 10. Files

- `benchmarks/harness/g4_activation_codec.py`: codecs, error metrics, byte arithmetic,
  `CodecActivationStore`.
- `benchmarks/harness/g4_data.py`: pinned Dolly, C5's split, packing, panel encoding.
- `benchmarks/harness/g4_probe.py`: `--mode error | train | suites | timing`.
- `benchmarks/harness/g4_timing.py`: A-B-B-A driver under `ArmWatch`.
- `benchmarks/harness/g4_rule.py`: the verdict.
- `benchmarks/g4-run.sh`: the GPU runbook.
- `tests/test_g4_codec.py`, `tests/test_g4_harness.py`, `tests/test_g4_rule.py`: CPU tests.
- `benchmarks/results/g4-rtx3090-20261024/`: the compact record of the GPU run (§8): the
  inputs `g4_rule.py` reads (absolute host paths replaced by `<work>` / `<hf-cache>`;
  `provenance.json` lists each file's original and published SHA-256, and the published
  inputs reproduce the replay exactly), the run's own `verdict-b195da29.json` and log, the V15
  replay and the comparison of the two. Timing run logs, step logs and the adapters stay with
  the raw run.
- The CPU pilot and review evidence of §9 and §11 stay on the research branch.

## 11. Acceptance review, 2026-10-22 (CPU only, SYNTHETIC/CPU PILOT)

An independent review of the harness before any GPU arm. No threshold of E1-E3, Q1-Q3 or
V1-V12 was changed. What changed, each with a CPU test:

- **NaN / Inf fail loudly** (§1). Before, a non-finite row stored a NaN scale and the
  recompute ran on garbage; the old test asserted that behaviour.
- **Spill policy decided and tested on the real backends** (§1): a payload that is not a
  4096 B multiple is refused before any spill file exists; both `BufferedSpillFile` and the
  unbuffered `DirectSpillFile` round-trip every codec for two steps; `close` joins the spill
  threads and removes the file, also after a failed write; a failed scale-slab allocation
  now closes the already-spilling inner store (before, it left the threads and the file).
- **Bytes measured, not assumed**: `compressed_chunk_bytes` equals the store's real storage
  bytes for every codec, odd int4 widths (1, 7, 97, 4097: one real pad nibble) and bf16 /
  float32 sources; the staging bytes are read from the ring buffers and the spill file size
  from disk.
- **Exactness checks have controls**: identity == GA as well as identity == plain L2L; a
  store that leaks its int4 restore into the forward changes the losses (so the loss check
  can fail); int4 moves the gradients more than either candidate.
- **V13 base parity** added (§2, §3.4), with its bound fixed before any parity number.
- **The rule fails closed**: an unreadable, truncated or malformed input file (missing
  fields, wrong types) gives NO VERDICT and never raises; V12 checks the suite counts.
- **The runbook cannot mark a step done without its JSON**: `g4_probe.py --device cuda`
  without CUDA now exits 3 (it printed SKIP and exited 0, so `g4-run.sh` would have marked
  every step done), `setup` / `arms` check CUDA, and a step that leaves no JSON fails.
- **Wording**: "committed" replaced by "fixed before the GPU run; to be committed before
  the first GPU arm" everywhere in the G4 files.

Second review, 2026-10-23 (audit reproductions, CPU only). No threshold changed; two rows
added and one tightened before any GPU arm:

- **V9 binds each file to its arm.** Before, a bf16 record saved as `train-int8-seed1.json`
  passed V9 (codec was an allowed difference); now each file's recorded (codec, seed) must
  equal the one its name claims.
- **V14 links `error.json` to the training arms.** Before, V1-V5 and V6-V10 checked their
  inputs independently, so an error experiment from another commit, model or `k` could gate
  arms it never measured. `--mode error` now records its config.
- **Timing attempts are kept.** `g4_timing.py` refuses an existing `--out`, writes each
  attempt's runs into a new `runs/attempt-NN/`, and never opens a run's report or log over an
  existing file (before, a restarted step reopened `runs/run001-*` with mode `w`).

Smokes, raw logs, commands, versions and `git rev-parse HEAD` are kept on the research branch
under `benchmarks/results/g4-cpu-pilot/acceptance-20261022/` (HEAD fdf067fc plus that branch's
uncommitted harness; Python 3.12.10, torch 2.14.1+cpu, transformers 5.19.0, peft 0.21.2):

- `error-smollm2-seq64-k4.*`: `g4_probe.py --mode error` on SmolLM2-135M, float32, seq 64,
  k = 4, one set: GA == L2L, recording == plain and identity == plain (240 LoRA tensors,
  none all-zero, losses `torch.equal`); int8, fp8 and int4 losses `torch.equal` to exact and
  grads not equal. An earlier attempt whose `--weights` was a mangled POSIX path
  (`/.cache/huggingface/...`: `$HOME` expanded to nothing under the Windows Python) is kept
  as `invalid-posix-path-error-smollm2.*`; it is the production finding below.
- `train-int8-seed1.*` (2 steps, seq 64, 4 panel rows) and `suites-int8-seed1.*` (V13 on 4
  rows, 8 new tokens): resident and streamed task loss both 1.481268 on 246 response tokens,
  0 of 240 adapter tensors mismatched, 1.483368 with the adapter off. This is the float32
  `--quant none` path on CPU, **not** NF4: it does not show that resident NF4 equals
  streamed NF4.
- `store_direct_spill_smoke.*`: the store through the real unbuffered `DirectSpillFile` at
  (1, 512, 4096) for bf16 / int8 / fp8 / int4 and at (1, 256, 576) float32: every restore
  `torch.equal` to the codec round trip, every byte check equal, file removed, threads joined.
- `rule-replay-cpu-pilot.*`: `g4_rule.py` on the §9 pilot: `Verdict: NO VERDICT` (V1, V6,
  V12 and V13 fail).
- `pytest.log` (266 passed), `ruff.log`, `bash-n.log`, `runbook-usage.log`, `help-*.txt` and
  `runbook-argv-check.log` (every argument vector `g4-run.sh` builds, parsed by the real
  `g4_probe.py` / `g4_timing.py` parsers).
- `superseded/`: the same three model smokes run once before the last harness edit (train
  mode now records `stopped` on a refused non-finite step); same numbers, kept as run.
- `finding4_missing_local_path_repro.*`: a missing local model path ends in a Hub repo-id
  error (`spectrum_scan.py:922` → `hubs.py:443/476`) that never says the path is missing.
  Reported, not fixed: `src/` is out of scope.

Post-run analyzer fix, 2026-10-25 (CPU only; no GPU re-run, the computation path is
unchanged). An audit of `g4_rule.py` after the GPU run, checked here as a hypothesis and
confirmed on mutated copies of the run's own `error.json`: E3 took `max`/`min` over the sets,
which skip a NaN that is not first, so a NaN `global_rel_l2` or `min_cosine` in the second set
left fp8 PASS; a dropped set, a missing or duplicated layer, a layer merged over too few rows,
a partial gradient table and an unrecorded protocol also left PASS, and a missing metric
raised. V15 (§3.4) now checks completeness against the recorded protocol and the finiteness
of every metric before any gate aggregates; V4 refuses a non-finite element error the same
way. No threshold changed. `tests/test_g4_rule.py` covers each case (NaN in the first and a
later set, +/-Inf, a missing metric or gradient record, a removed or repeated set, a missing,
duplicated or under-merged layer, an incomplete or miscounted gradient table, a `min_cosine`
not from its table, an unrecorded protocol), that counts follow the recorded protocol rather
than fixed 32 / 256, and that complete finite data keeps its decision. On the analyzer of
`b195da29` the same cases gave `PASS` (fp8 or both), `fail` or raised. The replay of the GPU
run is in §8.
