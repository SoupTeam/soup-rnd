# B1.3: streamed MoE backward, L2L and accumulation

## Provenance

- Repository: `https://github.com/SoupTeam/soup-rnd.git`.
- Base: `9d8b1d7b4b3eabc1634a1f6c2318814c640535c3`, fresh `origin/main` at task start;
  contains B1.2 PR #25 and the merged A3 L2L trainer. No draft A3/two-read branch was used as base.
- Integration branch/worktree: `b1.3-moe-backward`,
  `C:/Users/sydyk/Downloads/soup-rnd-b13`. The user's `g8-prefetch` branch and working files
  were not changed.
- **Measured source: `e691ffffc64349e53ce6d65b97ae7ab7ac2f1bd8`.** All eight final smoke files
  report this SHA, `dirty=false`, imported package path and actual harness hashes.
- This evidence publication follows the measured commit and changes no product, harness or tests.
  `summary.json` contains the complete 22-file code/config/docs inventory against the fixed base.
- Local specification and seven dependency-tracked tickets live under
  `.scratch/b1.3-moe-backward/`; portable snapshots are included in `raw-results.zip`.

## Acceptance criteria

| Criterion | Implementation | Exercised evidence |
| --- | --- | --- |
| 1. Reverse backward, actual reloads, bounded lifetime | Per-hit-expert first-order autograd node saves gathered inputs/routing weights, never arena views. It unpacks saved tensors before leasing, then reloads through the existing reader/pool and differentiates native eager arithmetic before releasing. Runtime owns one expert pool, separate from layer storage. | Primitive CPU/CUDA dense/bf16/NF4/nested-NF4 comparisons, arena churn/checkpoint/unpack/error cases; smoke visits layers 2,1,0, records actual reader events and positive backward reloads, exactly three reads per expert visit and one stable arena pointer. |
| 2. Real L2L/trainer accumulation | Per-expert caches are installed before layer weight maps. The existing `L2LStep.run` and HF trainer subclass admit internal DeepSeek-V3; routing replay checks each layer/micro-batch and frees window state. | Native resident GA vs actual streamed step over repeated windows, k=1/2/4; real HF Trainer multiple updates and short last windows. Standalone reader/pool + L2L smoke performs two genuine SGD updates per k. Public launch remains guarded for B1.4. |
| 3. RNG, dtype and supported gradients | Expert backward replays CPU/CUDA RNG and forward autocast; actual backward is outside forward AMP. Decoder-plus-native-head draws follow resident micro-batch order. Trainable embeddings get original-ID VJPs immediately per micro-batch, with original NEFTune RNG. Frozen routed weights remain frozen. | Exact hidden/routing/router/shared gradients at the block seam; native model/GA comparisons cover 30/32/44/46 nonzero trainable tensors, dropout .05, NEFTune .2, outer RNG and CPU/CUDA forward-bf16 AMP. Supported non-expert targets are not excluded or replaced with refusals. |
| 4. k=1 and multiple micro-batches | Accumulation k is independent from router top-k. Native resident eager is the oracle, not another streamed implementation. Only initialization/construction is shared. | k=1/2/4 smokes for all four target sets; permanent routing top-k cases 1/2/4/8, distinct inputs/routes, shared and unused experts, masking and established mixed-length padding cases. All clean smoke logits/loss/gradients/trained states are exact. |
| 5. Real negative control | Test/harness wrapper modifies returned expert down-matrix staging only while autograd is actually executing backward. No product flag or fake verdict. | Every final corrupted window has the same exact forward/loss as the reference but fails the computed gradient comparison; all clean windows pass. A separate permanent nonzero-dropout broken-replay control also detects gradient disagreement. |

## Final numerical results

Each row covers k=1/2/4, two consecutive optimizer updates per k, on both CPU and CUDA.
All configured gradient **tensors** are nonzero in both arms; embedding vocabulary entries not
used by a batch naturally have zero elements. Across 48 clean windows, clean forward, loss,
gradient and trained-state maximum differences are **0**, and outer RNG agrees. All 48 actual
backward-corrupted windows preserve forward/loss and fail by gradients.

| Synthetic target set | Nonzero gradient tensors | CPU corrupted max-gradient-difference range | CUDA range |
| --- | ---: | ---: | ---: |
| MLA | 30 | 0.498075–4.197987 | 0.498075–4.197973 |
| Embedding + MLA | 32 | 0.911452–3.611841 | 0.911449–3.611841 |
| MLA + shared + head, dropout .05 | 44 | 0.829358–3.273990 | 0.684125–2.408230 |
| Combined, dropout .05 | 46 | 2.034620–10.373321 | 1.866372–14.119805 |

Criteria were fixed before results: fp32 uses `torch.equal`/zero tolerance; model-level bf16
uses the existing dense contract `atol=1e-6, rtol=1e-5`; established mixed-length padding uses
`atol=2e-5, rtol=1e-4`. None was relaxed. Recorded final CPU forward-AMP embedding/default
comparisons have zero loss/gradient differences. See JUnit properties for individual consumers.

Each smoke has one **6144-byte weight arena**, a stable pointer and CUDA allocation delta
6144 bytes; host staging is 20480 bytes. Highest measured CUDA peak across the four final
smokes is 67,997,184 allocated / 69,206,016 reserved bytes. This includes the resident oracle
and harness work: **the expert-arena budget is not a whole-VRAM/RSS budget or a model-fit claim**.

## Commands, environment and outcomes

Exact argv, cwd, timestamps, environment overrides, exit codes, JUnit and unabridged logs are
in `raw-results.zip` under `final-e691ffff/`. `cpu-targeted.json` contains the full 19-file
B1/B2/B1.3/L2L/RNG/trainer/license/GPU-marker selection; do not infer a full-suite PASS from it.
The local Windows interpreter's editable install points at the original checkout; the recorder
explicitly sets `PYTHONPATH=<measured-worktree>/src;<measured-worktree>` on Windows (colon on Linux).

| Executed check at e691ffff | Result |
| --- | --- |
| Targeted CPU pytest, `-o addopts= -o junit_family=legacy -q --tb=short -p no:cacheprovider` | 476 passed, 87 skipped (CUDA cases), 0 failed/errors; 7 retained third-party warnings |
| CUDA pytest, same options plus `-rs -m gpu`, six files listed below | 86 passed, 1 skipped (needs two visible cards), 121 deselected, 0 failed/errors; 16 warnings |
| Standalone smoke, CPU and CUDA, each target set below | Eight commands exit 0, all PASS including backward-only negative controls |
| Complete configured `tests/`, default non-smoke marker and coverage gate, documented `.[dev]` | 36,488 passed, 234 skipped, 2 xfailed, 12 deselected, 20 failed, 0 errors; coverage 86.74% exceeds required 77%; all 20 failed nodes also fail on the fixed base |
| `python -m ruff check src/soup_cli/ scripts/ tests/ benchmarks/` | All checks passed |
| `git diff --check 9d8b1d7b4b3eabc1634a1f6c2318814c640535c3...HEAD` | Exit 0 |

The CUDA invocation used these files:

```text
tests/test_b13_expert_backward_cuda.py
tests/test_b13_l2l_moe_cuda.py
tests/test_b1_expert_forward.py
tests/test_b1_expert_shard.py
tests/test_l2l_rng_cuda.py
tests/test_b2_deepseek_v3_stream.py
```

Executed smoke arguments, each with `--device cpu` and `--device cuda` and a distinct
`--output <recorded-evidence.json>`:

```text
python benchmarks/harness/b13_moe_backward_smoke.py
python benchmarks/harness/b13_moe_backward_smoke.py --targets embedding
python benchmarks/harness/b13_moe_backward_smoke.py --targets head-shared --dropout 0.05
python benchmarks/harness/b13_moe_backward_smoke.py --targets all --dropout 0.05
```

These target/dropout options belong only to the synthetic benchmark, not product YAML/CLI.
CUDA used deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, OMP/MKL threads 1.
Final Windows invocations also set OMP/MKL threads 1.

- CPU: Windows 11 build26300, Python3.12.10, torch2.14.1+cpu.
- CUDA: Vast.ai RTX3060 12GB, Linux6.8/glibc2.39, Python3.12.3, torch2.14.1+cu130, CUDA13.0.
- Both: Transformers5.19.0, PEFT0.21.2, accelerate1.15.0, safetensors0.8.0;
  pytest9.1.1 and Ruff0.16.10. No trained checkpoint download or prior GPU PASS is reused.

The complete configured run retains the repository's default marker and coverage options; a
throwaway, outside-repository observation plugin durably records failed node IDs/tracebacks.
It does not change product code, assertions, expected values or selection. No new B1.3 test
failed. Every failed node was then executed on base9d8b with the same complete dev packages,
visible RTX3060, CUDA/NCCL environment and offline flags; all reproduce there. This is **not a
full-repository PASS** and no failure is hidden or marked xfail to obtain one.

Residual baseline categories: a hard-coded torch-version assertion rejects the truthful `+cu130`
build suffix; foreign-read probe outcome; inherited hardware page-lock/drain/alignment assumptions;
CUDA-initialization/CPU-only peak assumptions; ordinary/resume CPU–CUDA device mismatch;
contrastive embedding LoRA update; ambient `NCCL_VERSION` in launcher expectations; an uncached
HF fixture blocked by deliberate offline mode; and sandbox `preexec_fn` permission failure.
Exact cases and both source-specific tracebacks are in `full-dev-configured-e691ffff/` and
`base-configured-residuals/` in the archive.

Earlier Windows full collection stopped on nine E4 errors from the unchanged POSIX `resource`
import. An early Linux coverage run timed out at 1800s. A completed no-coverage Linux diagnostic
before the complete dev install had 80 failures/9 errors, all reproduced on the same base; the
documented constrained `pip install -e .[dev]` plus Pillow/jsonschema and corrected PATH resolved
the missing-dependency/entrypoint issues. Historical495b/34b9 records keep their original SHAs;
none is relabeled as final-source proof.

Task-owned Vast instance58239965 was destroyed; a fresh instance-list query at
2026-10-29T19:36:50Z found neither it nor the earlier failed-boot58148283. Other instances were
not touched. The provider displays **USD0.565 posted task charges** at19:37:08Z, against a
USD15 budget (GPU, storage, download and failed boot included). Delayed final billing is not
invented. Filtered owned-instance invoices, deprovision command and absent-state evidence are
archived; no task rental remains.

## Independent reviews and delegation

After the model preference changed, every new subagent was a GPT6.1 Sol preset, without model
overrides. Earlier completed Sonnet code was retained rather than hidden or rewritten for attribution.

- Earlier Sonnet authors: `ExpertBackward2` (sonnet-max), expert autograd; `RuntimeInstall2`
  (sonnet-xhigh), runtime integration.
- `SolL2L` (sol-max): MoE/L2L, primitive activation RNG/mutable-state corrections and backward AMP.
- `SolTrainer` (sol-xhigh): real Trainer scenarios, standalone smoke and existing feature docs.
- `SolBackwardReview` (sol-max): independent numerical/lifetime/resource review. Its dropout,
  activation-buffer and interrupted-cleanup findings were fixed; it was not the code author.
- `SolFinalFix` (sol-max): complete final-review wave, streamed embedding VJP, native-head RNG,
  broadened consumer proofs and shared lazy fixture construction.
- Independent Standards: `SolStandards` / `SolStandardsFinal` (sol-xhigh), **APPROVE** after all
  three accepted findings were closed, including every added primitive/nested/test signature.
- Independent Spec: `SolSpec` / `SolSpecFinal` (sol-max), **APPROVE** after both P1 findings
  (embedding A/B gradients and combined decoder/head dropout schedule) were fixed. A separate
  focused consumer probe checked original IDs, immediate embedding backward/no refill,
  NEFTune/outer RNG, frozen fast path and combined-dropout parity.

Review reports preserve their exact review/evidence sources. Spec closure reviewed34b9 product
and the e691 annotation-only diff; final e691 CPU/CUDA execution is the separate integrator proof
above. Integrator owned Git/merges, minor annotation closure, final execution and publication.

## Support boundary and remaining risks

- First-order backward only. Routed base expert weights stay frozen; routed expert adapters and
  full expert training remain refused. Supported non-expert MLA/shared/head/embedding LoRA works.
- Internal native DeepSeek-V3 per-expert runtime/L2L path, not B1.4 public enablement. Public L2L
  launch still admits Llama/Mistral only. No planner, new product YAML/CLI field or default changed.
- Existing uneven stochastic-window and NEFTune-plus-dropout refusals remain; no new masked or
  uneven support is implied. Differentiable custom expert activations with registered mutable
  buffers are refused before reads; accepted stateless stochastic activations replay their RNG.
- Synthetic correctness evidence, not trained-model quality, throughput, multi-GPU or whole-model
  capacity validation. The two-visible-CUDA-card RNG case was not executed.
- Independent review reproduced an inherited AsyncDiskSource thread leak after a later setup
  allocation failure on the pre-B1.3 fused baseline as well. It is not fixed or misrepresented as
  a new regression. Normal runtime expert cleanup, including interrupts, is covered and fixed.
- Existing feature docs are updated. [PR #47](https://github.com/SoupTeam/soup-rnd/pull/47)
  publishes the numbered fragment `changelog.d/0.75.1/47.added.md`. The staged text in the
  archive is the historical pre-PR snapshot and is retained unchanged. Branch/PR publication
  followed the user's explicit authorization; the shared main branch has not been merged.
