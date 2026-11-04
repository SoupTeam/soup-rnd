# E4 part 2, calibration rule: SMOKE feasibility and tolerances

Card: [E4]. Precedes the E4 part 2 main rule (`e4-rule-part2`, draft v4.2, not committed).
Status: DRAFT v4, 2026-10-16. Not committed. Nothing in part 2 has been run or measured.
"House rule N" means rule N of the research house rules (shared/RULES.md, snapshot 2026-10-05; not in this repository), checked against current `benchmarks/METHOD.md`, AGENTS and CONTRIBUTING.
Committed before the first calibration run, with the harness code, input hashes and this rule's `.gitattributes` line `<this file> text eol=lf`, and not edited after (house rule 1; `benchmarks/METHOD.md:275-277`). Every `[CAL]` value the main rule uses comes from this record, on the final card, before the main rule is committed. Full training arms are not run under this rule.

## 0. Purpose and limits

| Output | Used by the main rule for |
|---|---|
| F: memory, wall time per update, finiteness per arm | choosing the final card (§3) |
| T: tau per cell and quantity | main §3a checks and the strong-control criterion |
| W: tol_W per quantity | main §3b expert wrapper checks |
| N: suite noise floor per suite | epsilon = max(2 x floor, 1/n_items) |
| P: one repeated short run | reported repeatability, not a decision input |

It produces **no verdict on double-quant**. DECISION rows are read only to normalize, hash and tokenize them for eligibility and the manifest; no DECISION row passes through any model here. No DQ-flipped comparison and no control is computed here, so no tolerance can be tuned to them.

## 1. Fixed inputs

| Item | Value |
|---|---|
| Model | `allenai/OLMoE-1B-7B-0924` @ `6d84c485` |
| Libraries | torch 2.14.1, transformers 5.19.0, bitsandbytes 0.50.2, peft 0.21.2, Python 3.12 |
| Card | F starts on one A100 40 GB vast.ai instance; final card chosen by §3 |
| Experts implementation | `experts_implementation="grouped_mm"` passed by the harness, resolved value recorded |
| Attention implementation | `attn_implementation="sdpa"` passed by the harness in every arm and every stage, resolved value recorded |
| Compute dtype | bf16; adapter factors fp32 |
| Commits | Soup commit, harness commit, harness git tree hash in every record |
| Process | every run in a fresh process; every attempt journaled (`attempts.jsonl`); voids kept |
| Quantized base files | made on part 1's Mac by main rule §2 at harness commit `a13992ef7180539f9317b48f09073fb5215fc7b2` (harness tree `451b75a0f94bd2606b73ea6394d3c3668ee67b38`, clean tree); manifest `benchmarks/results/e4-part2-quant/quant-manifest.json`, SHA-256 `1f3b550d707aa18c098851a1718f1712cdb1162adc28552c5fbbcce8611f0ccb` (3,136 units, 78 output files, 9 source files, U1-U3 pass for both settings); stored in the private HF dataset `RedBullet12/e4-part2-quant` at commit `2e09044127c7f8fd97ac474d89c2d98144ec7550`; every file SHA-256 verified on the instance against it before any run |

**Retired quantization output sets.** Two earlier sets are retired and no result uses them. The first (manifest SHA-256 `fb3edd4a73eff529bb353405cf11fccbe780d5cb97a2515fbcac874327f4602b`) was made from a tree with uncommitted changes (`harness_dirty` true), which `validate_provenance` voids. A rerun from a clean tree then wrote 17 of 78 codes files with different bytes: safetensors 0.8.0 writes the keys of `__metadata__` in an order that changes from process to process, so a header with two keys is not reproducible. The codes files now carry one key. Two full runs from the final harness commit wrote byte-identical outputs (the first stopped on a stray Finder file before writing its manifest). The 46 write-back files are byte-identical to the first set; the 32 codes files differ from it only in that metadata, with tensor data equal.

Per run, recorded: GPU name and UUID, vast instance id, driver and CUDA versions, SM clock at start and end, free host RAM and commit charge, GPU peers and foreign processes, `box_stamp_before.unix_s`. AC/battery is N/A on a hosted instance and recorded as such.

**fp32 references** (T and W) run with autocast disabled, `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False` and `torch.set_float32_matmul_precision("highest")`; the harness asserts the output dtype is fp32 and records these flags.

## 2. Data split (fixed here, used unchanged by the main rule)

Dataset `databricks/databricks-dolly-15k`, full revision hash recorded. Tokenizer: OLMoE's, same model revision, full hash recorded.

1. Model text: each field Unicode NFC, leading and trailing whitespace stripped; internal whitespace and newlines kept. This text is what the model sees and is scored on.
2. Identity key, used only for dedup and ordering: each model-text field with internal whitespace runs collapsed to one space; identity = SHA-256 of the UTF-8 bytes of `json.dumps([instruction, context, response], ensure_ascii=False, separators=(",", ":"))` over those keys.
3. Exclude rows with an empty instruction or response.
4. Dedup on identity, keeping the first occurrence in dataset order.
5. Build the prompt with the wrapper in §2a. Exclude rows whose prompt is over 384 tokens or whose response plus EOS is over 128 tokens. No truncation.
6. Sort eligible rows by identity (hex string, ascending).
7. DECISION = rows [0, 2048), SMOKE = [2048, 2304), TRAIN = [2304, 6400). Remaining rows unused.

A pre-commit count (review script, 2026-10-15, Dolly `bdd27f4d`) found 11,952 eligible rows; the harness recomputes and records it. Fewer than 6,400: stop, report, no calibration run; the question goes back to the owner.

Manifests per split (row id, identity, token counts) are written, hashed and committed with the calibration record.

### 2a. Prompt and targets (agreed 2026-10-16)

- Context non-empty: `### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n`
- Context empty: `### Instruction:\n{instruction}\n\n### Response:\n` (also the gate-suite wrapper)
- Target: the response tokens as tokenized directly after the wrapper (no added leading space), then EOS. Prompt and padding carry label -100; response and EOS are targets.
- Recorded: exact wrapper text and SHA-256, token ids of both wrapper forms, and one worked example each (with and without Context) showing tokens and label mask.

## 3. Feasibility (F) and the final card

Recipe (the main rule's, in full):

| Parameter | Value |
|---|---|
| LoRA | r 16, alpha 32, dropout 0, bias none, standard peft init, no extra trainable modules; placement as main rule §4 |
| Adapter dtype | fp32 trainable factors; ParamWrapper casts them to the base parameter's dtype (bf16) before folding (`peft/tuners/lora/layer.py:2525-2542`) |
| Batch | microbatch 1 sequence, gradient accumulation 16, no packing, max length 512 |
| Loss normalization | sum of response-target losses over the effective batch, divided by that batch's response-target count |
| Optimizer | AdamW, lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0; frozen weights excluded |
| Schedule | warmup 16 updates, linear decay to 0 at update 256; grad clipping norm 1.0 |
| Other | gradient checkpointing on, `use_cache=False`, same attention and experts implementations in every arm |
| Data order | seed-dependent shuffle of TRAIN, identical across arms within a seed |

Run A_off, A_on, A_bf16 once each, seed 42, **16 updates**, in that order, each in a fresh process. Recorded per run: peak `torch.cuda.max_memory_allocated` and `max_memory_reserved` in bytes, whole-device used and free memory in bytes from NVML sampled every second, wall time per update, finiteness of every loss. Projected wall time for the main plan (9 runs + 1 rerun) is reported. Pre-commit estimate, not measured: about 20 GB reserved, range 18-25 GB.

**Learning rate in the 16-update runs (F and P).** The schedule is the Trainer's linear warmup, indexed by the 0-based update about to run: update i runs at `1e-4 x i / 16` for i < 16 and at `1e-4 x (256 - i) / 240` from i = 16. Update 0 therefore runs at lr 0, update 15 at 9.375e-5, and a 16-update run never reaches the full rate. F measures memory, time and finiteness at these rates, not at the peak rate of the main plan; the seed-42 adapter that N scores and P repeats is the product of this schedule. The one AdamW step in T's LoRA quantities (§4) is not scheduled: it uses the full `1e-4`.

**Final-card rule:** if every arm completes with peak reserved memory <= **36,000,000,000 bytes**, the final card is this A100 40 GB instance. If any arm OOMs or exceeds 36,000,000,000 bytes reserved, F is repeated once on an A100 80 GB instance under this rule, and that instance is the final card. If the 80 GB run also OOMs, calibration stops and the recipe goes back to the owner. A non-finite loss fails F; the main rule is then not committed as planned. F also fails on a recipe mismatch (trainable parameter count, non-fp32 trainable factors, or an adapter placement set that differs from main rule §4) or on a changed frozen base. These failures are reported through the final-card decision as F_FAILS, and F is not rerun.

T, W, N and P below run on the final card's instance only.

## 4. Base-weight tolerances (T)

**Two independent per-call attributes**, recorded for every call:
- `needs_grad`: whether input or bias requires grad. In bnb 0.50.2 `matmul_4bit` calls `gemm_4bit` directly without grad, and `MatMul4Bit.apply` with grad, whose forward also calls `gemm_4bit` (`bitsandbytes/autograd/_functions.py:324-334, :453-491`).
- `kernel`: which forward the CUDA backend actually ran inside `gemm_4bit`, fused or dequant + linear fallback, chosen by shape, dtype and card (`backends/cuda/ops.py:951-975`), captured by instrumenting that dispatch.

A **cell** is (kind, needs_grad, kernel), with kind one of q, k, v, o, gate, up, down. Input-gradient backward is its own operation (dequantize, then `torch.matmul`, `_functions.py:384`) and gets its own cell per kind, independent of the forward kernel.

For each kind and each DQ setting, on 8 units of each kind chosen by `random.Random(20261015)` before any is read, in both grad modes:
- Inputs: activations captured at that unit while running SMOKE prompts through the bf16 model, token counts matching training microbatches (attention: full sequence; expert: the tokens routed to that expert).
- ref32: fp32 reference (§1 settings) of `x @ W_dq.T`, with W_dq the written-back weight.
- F: plain `nn.Linear` holding the written-back bf16 weight, bf16.
- R: `Linear4bit` from `Params4bit.from_prequantized` on the transferred codes and QuantState, compute bf16.
- Fixed upstream gradient G ~ N(0, 1), `torch.Generator` seed 20261015, objective L = sum(y * G).
- Quantities: output, input gradient, and the scalar L (reported only).
- e_F, e_R = max abs difference from ref32, per call and quantity.

LoRA quantities, on 8 attention units of each kind with per-module LoRA (r 16; A from peft init; B ~ N(0, 0.02), seed 20261015; fp32 factors) on F and on R: lora_A gradient, lora_B gradient, and factors after one AdamW step (main recipe values, same initial optimizer state), each against an fp32 reference of the same objective.

**Calls per unit.** Up to **4** SMOKE rows per unit: the first four rows, in SMOKE order, in which the unit receives any tokens (an attention unit always does). Each of the four is one call, run in both grad modes. The LoRA quantities use only the **first** of those calls, in both grad modes.

A selected unit that receives no SMOKE tokens (an expert the router never picks) contributes no calls; it is listed, not replaced. The number of contributing units per kind is reported. A kind with fewer than 4 contributing units gets no tau, and the main rule makes no equivalence claim on it.

Any NaN or infinite value in e_F, e_R or a reference voids that T run; it is never dropped from a maximum.

**Rule:** tau(cell, quantity) = 2 x max over calls and both DQ settings of max(e_F, e_R). Quantities each get their own tau: output, input grad, A grad, B grad, update. A cell that never occurs on SMOKE gets no tau; the main rule makes no equivalence claim on it.

Also recorded: E0 on these units (CUDA `dequantize_4bit` of R's codes vs the transferred CPU weight, byte comparison). A difference is reported as a finding, not a failure of T.

## 5. Expert wrapper tolerances (W)

Calibrates the rounding of the **intended** expert operation against an fp32 version of it. The peft wrapper under test is **not** run here, so no wrapper error can enter tol_W.

- Units: fused `gate_up_proj` and `down_proj` of layers 0 and 8, A_off and A_on weights.
- Inputs: hidden states at those layers' MLP input from the first 32 SMOKE rows through each arm's base; one call per row.
- Adapter state: rank 16 per expert, one adapter spanning fused gate and up on `gate_up_proj`, one on `down_proj`. A from peft's default init shape and scheme with `torch.Generator` seed 20261015; B ~ N(0, 0.02), same generator. Factors fp32. Same objective L = sum(out * G) and same initial AdamW state for both implementations below.
- **bf16 reference (the intended operation)**, plain torch, no peft: cast A and B to bf16 first, then per expert `W_eff = torch.baddbmm(W_bf16, B_bf16, A_bf16, alpha=s)` with autocast off, in peft's factor layout (`layer.py:2319-2323, :2525-2542`) reproduced by hand; then the expert MLP with the pinned experts implementation; gradients by autograd; one AdamW step.
- **fp32 reference:** the same computation with fp32 factors, fp32 W_dq and fp32 MLP under the §1 fp32 settings.
- Quantities: output, input gradient, A gradient, B gradient, factors after the step.

**Rule:** tol_W(quantity) = 2 x max over calls, layers and both arms of |bf16 reference - fp32 reference|. SHA-256 of the frozen base tensors before and after the step must be unchanged; any change fails W. Any failure of the fp32 reference to run with the pinned experts implementation voids W; there is no fallback to another implementation. A changed frozen base fails W, and W is not rerun.

## 6. Suite noise floor (N)

`DEFAULT_GENERAL_SUITE` through `score_bundled_suite`, scorer revision 4, greedy, max new tokens 256, the gate-suite wrapper of §2a (exact text and token ids recorded), MCQ cue kept. Fixture manifests and item counts per suite recorded.

Model: A_on base plus the seed-42 adapter from §3's 16-update run on the final card. Scored **five times**, each in a fresh process.

**Rule:** floor(suite) = max - min of the five scores. epsilon(suite) = max(2 x floor, 1/n_items), compared with a strict `>`. One lost answer at epsilon = 1/n_items is not a flag. The floor measures scorer repeatability on one adapter, not seed-to-seed training variance.

## 7. Repeatability (P)

A_on, seed 42, the §3 16-update run repeated once on the final card in a fresh process, after N. Reported: SMOKE NLL of both runs, their difference, and whether adapter weights are bit-identical. One paired observation, not an estimate of the distribution of GPU nondeterminism.

## 8. Validity

| Row | Check | Required | If it fails |
|---|---|---|---|
| suspend, power, foreign_reads (METHOD) | timing-only rows for a local box | N/A: time here is only a feasibility estimate on a hosted GPU | n/a |
| Provenance | commits, tree hash, stack, GPU UUID, instance id | yes | no result from that run |
| Inputs | hashes of quantized base files and manifests | yes | void |
| Peers | no foreign GPU process during a run | yes | void |
| fp32 settings | autocast off, TF32 off, output dtype fp32 | yes | T or W void |
| Finiteness | every loss and output finite | yes | F fails (§3); T/W/N/P run void |
| Frozen base | base hashes unchanged after any step | yes | calibration fails, reported |
| Attempts | every attempt journaled | yes | missing journal reported as provenance gap |

A void run keeps its record and is rerun once; a second void of the same run stops calibration and is reported. A failed attempt of T, W, N or P stops calibration (`CalibrationStopped`) and is reported; it is not rerun. A non-finite P loss or SMOKE NLL is a void with one rerun; only a changed frozen base fails P.

**Card changes.** All `[CAL]` values come from the final card's instance and are copied into the main rule before it is committed; the main series runs on that instance. If the instance changes after the main rule is committed, the old rule and every attempt are kept unchanged, and a new calibration record and a new preregistration version are committed before any new series. No tolerance in a committed rule is replaced at run time.

## 9. Record

`benchmarks/results/e4-part2-calibration-<date>/`: this rule's commit hash, split manifests, F/T/W/N/P raw JSON and logs, per-call `needs_grad` and `kernel`, attempts journal, per-run box stamps, SHA256SUMS, and a report with the values the main rule copies: final card and UUID, tau table by cell and quantity, tol_W table, floor and epsilon per suite.

## Clarification 2026-10-20 (card classes)

The rule's card names, "A100 40 GB" and "A100 80 GB" (sections 1 and 3), are checked from the box stamps of each F and F80 record, before and after the run:

- the GPU name contains `A100` (SXM4 or PCIe);
- `nvidia-smi` `compute_cap` is exactly 8.0;
- `nvidia-smi` `memory.total` is within 40,000 to 42,000 MiB inclusive under F, and within 80,000 to 82,000 MiB inclusive under F80.

A missing or mismatching field stops calibration (`CalibrationStopped`), and `decide` writes no `decision.json`. The 36,000,000,000-byte reserved-memory threshold and all other text of this rule are unchanged. Implemented in `e16b90df` (constants in `e4_part2_common.py`). Requested by the reviewer on PR #21.
