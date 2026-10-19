# Streamed Liger: strict pretrained 7B follow-up

**Verdict: blocked by 7B numerical parity.** The unchanged 24-case native CUDA
matrix passes, but RMSNorm, SwiGLU and their union fail the original elementwise
tolerances on a pretrained 7B model. Finite training and exact adapter export are
insufficient to qualify these arms. No Liger acceleration or memory saving is claimed.

## Frozen scope

- Source: `e7b8682a9526bdb2938a13620c45f188a2d4caa4`, branch
  `koko/d2-liger-streaming`; separate from the NoRA worktrees and PR #5.
- A40, native BF16, driver 595.91.07; Python 3.12.3, Torch 2.11.0+cu126,
  Triton 3.6.0, Liger 0.8.1, Transformers 5.17.0, PEFT 0.20.0, TRL 0.29.1.
- `openlm-research/open_llama_7b_v2`, immutable revision
  `e5961def23172a2384543940e773ab676033c963`. Original `.bin` shards converted
  without changing names/dtypes; original and converted hashes retained.
- Actual Soup SFT, dense frozen base, RAM streaming, two slots, pinned RAM,
  per-layer checkpoints, ordinary PEFT LoRA on all seven projections, rank/alpha 4,
  dropout zero, seed 792. Explicitly initialized identical, nonzero LoRA A/B.
- Two optimizer updates, four microbatches, batch 1, accumulation 2; actual
  sequence length 1024. Synthetic repeated arithmetic text, 32 frozen rows;
  data SHA-256 `313d17f90f4b62004a1d8232ba9bc0b19388a65e933c930ddc9aecfe7ffbf0f9`.
  This is a numerical fixture, not a task-quality or hallucination evaluation.
- Identical initial adapters and input/label tensors must match exactly;
  every gradient, final adapter and logit element uses `rtol=1e-2, atol=1e-5`.
  Captured microbatch losses also must match to three decimals. Original optimizer
  log histories are retained separately; microbatch loss includes accumulation scaling.

The original 24-case test function, including decorators, inputs and assertions,
has an identical AST to `f7649c72`. No tolerance or failing assertion was removed.
Independent local replay also passes all 24 retained tiny CUDA tensor pairs.

## Repeatability before attribution

The first cross-arm screen failed, but an ordinary baseline also failed against
its own repeat. An isolated SDPA backward probe on A40 produced different BF16
gradients on identical inputs: maximum difference 0.001953125. With
`torch.use_deterministic_algorithms(True)`, repeated gradients were bitwise equal.

Both correctness and measurement harnesses now explicitly enable the same
deterministic policy, before model setup, in every arm. The complete ordinary
7B control then repeats **all 1,813 saved tensors bit-for-bit**. An additional
ordinary run with operator diagnostic hooks also passes comparison. The original
nondeterministic screen remains archived, but is not used to attribute errors to Liger.

## Native CUDA and deterministic 7B results

The final source repeats **24 passed, zero skipped/failed**, 67.92 seconds:
three component choices x FP16/BF16 x RAM/disk x dense/NF4. A separate captured
FP16 rounding regression also passed. These tiny-model passes do not establish 7B parity.

| 7B arm | Failing gradient tensors / 896 | Failing final tensors / 448 | Failing logits / 4 | Verdict |
|---|---:|---:|---:|---|
| RMSNorm | 118 | 431 | 4 | Fail |
| SwiGLU | 120 | 322 | 2 | Fail |
| RMSNorm + SwiGLU | 119 | 432 | 4 | Fail |

Every initial adapter and batch matches exactly. All captured values are finite;
no NaN was observed. All four ordinary/native arm exports reload **448 adapter
tensors exactly** through public PEFT APIs. Reload checks validate stored weights,
not 7B prediction equivalence on a different base dtype.

Observed maximum final-adapter differences are 5.81e-5 to 5.93e-5; maximum logit
differences are 1.25, 1.5 and 2.0 respectively. Matching rounded loss alone would
hide the SwiGLU failure, so it is not treated as a substitute for tensor parity.

Disabling FMA in the actual upstream SwiGLU backward did not fix the deterministic
7B comparison and was **not** promoted into production. Actual layer-0 operator
inputs and incoming gradients were retained for further reduction. On those
inputs, forwards match bitwise, but backward differs in 58 RMSNorm elements
(maximum 4.77e-7) and 49 SiLU elements (maximum 7.45e-9). Each isolated operator
still passes the elementwise tolerance locally. These low-bit differences are
observations, not a complete causal explanation for every full-model mismatch.

## Baseline speed and memory calibration

Comparative Liger ABBA timing was refused because no Liger arm qualified. One
ordinary deterministic streaming run provides a baseline calibration only:

| Quantity | Ordinary streamed PEFT baseline |
|---|---:|
| Warmup / measured optimizer steps | 3 / 5 |
| Median synchronized seconds / optimizer step | 4.1843 |
| Processed input tokens / second | 489.45 |
| Peak Torch allocated memory | 1.8448 GiB |
| Peak Torch reserved memory | 2.1504 GiB |

Each optimizer step processes two 1024-token sequences. The five raw durations
are 4.18734, 4.18392, 4.18122, 4.18431, 4.19080 seconds. Windows exclude initial
batch fetch, saving and evaluation; CUDA is synchronized at both boundaries.
Allocator peaks exclude CUDA context/driver allocations and are not an NVML
total-VRAM peak. A single baseline run supplies neither a speedup ratio nor
repeat-level uncertainty for an A/B comparison.

## Dependencies and CI

The `liger` extra now pins `liger-kernel==0.8.1`; streaming refuses other versions
before model downloads. Its private RMSNorm backward launch contract must be
requalified before widening this pin. On the GPU environment, `uv pip check`
passes and resolving `.[train,liger]` requires no dependency replacements.

The original branch's full CI already failed on PPO checkpoint saving with
Transformers 5.20.1 and CPU Fast-LoRA tests automatically selecting MPS on macOS.
The fix respects explicit CPU SFT placement and initializes the model-placement
capability TRL's eager PPO constructor omitted. The PPO failure was independently
reproduced with an isolated Transformers 5.20.1 install, then its actual training
and save test passed after the fix; saves/tests were not disabled.

Local compatibility group: **103 passed, 33 hardware skips**. Final focused group
on the original local stack: **82 passed, 25 CUDA skips**. Full-path Ruff passes.
README/changelog validation: **52 passed**. One initial CI attempt completed the
macOS/Python 3.10 suite with **35,317 passed**, but correctly failed our malformed
changelog fragment. The notes were consolidated into the existing issue #5 fragment;
no changelog guard or runtime test was disabled. Superseded dispatches were canceled.

Full CI is dispatched on the final report commit; executable code is unchanged
from the measured source. See the
[CI history for this branch](https://github.com/SoupTeam/soup-rnd/actions/workflows/ci.yml?query=branch%3Akoko%2Fd2-liger-streaming).
The exact final run, commit and job conclusions are retained in `ci-final.json`
alongside the raw results.

## Evidence, spend and cleanup

Raw results are saved outside Git in
`output/liger-7b-qualification-20261019/`: source archives, pinned environment,
all initial/gradient/final/logit safetensors, failed screens and trials, exact
batch hashes, JUnit, public reload receipts, SDPA and operator probes, baseline
timing samples, controller sources, and provider billing/deletion receipts.
The full archive is SHA-256 verified before instance deletion. Independent local
replay uses elementwise arithmetic rather than trusting the controller verdict.
It confirms the exact ordinary control and all three failing 7B comparisons.
Original source bytes match the saved source archive; canonical LF source bytes
also match Git and the local checkout. Windows CRLF differences are recorded
separately in `source-line-ending-audit.json`.

- Complete archive: **5,584,115,823 bytes**, SHA-256
  `f1de07bfc05ae69a7b5f06828205f4257401071a12447c96eae2af9c9cc39971`.
- Full remote tar integrity was checked. After a local packaging wait timed out,
  the original remote compressor finished normally; its archive was reused.
  A preserved sequential prefix and four nonoverlapping SSH ranges reconstruct
  the same complete archive, verified against the original SHA-256.
- Exact instances **56660857, 56662809 and 56667949** are all verified absent
  through Vast's API. The last lease, including transfer and cleanup, lasted
  **65.68 minutes**. No separate persistent volume was created.
- Provider-observed cumulative charges: **USD 1.057 / USD 2 maximum**,
  comprising USD 0.367 previously and USD 0.690 for this follow-up. Settlement
  can lag; the conservative shutdown estimate including reserve is USD 1.232.
  Own-instance charges are filtered by exact IDs, excluding unrelated jobs.
- `gpu-results.sha256.json`, `independent-replay.json`,
  `tiny-independent-replay.json`, `protocol-audit.json`,
  `cumulative-billing.json` and `vast-deletion.json` retain the verification.

The user's reported resident 1.5B/3B VRAM reductions and 7B full-stack NaN are
different experiments. The public Soup maintainer's
[resident H100 record](https://github.com/MakazhanAlpamys/Soup/issues/78#issuecomment-5218436327)
also concerns a different model/configuration. This run does not reproduce that
NaN, validate those percentages for streaming, or enable Liger RoPE/fused loss.
7B FP16, disk/NF4, Fast-LoRA + Liger and final task quality remain unqualified.
