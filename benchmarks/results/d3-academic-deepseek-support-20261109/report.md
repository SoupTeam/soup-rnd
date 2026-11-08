# Academic DeepSeek expert-training support: final validation

Date: 2026-11-09
Workspace: isolated worktree based on `1b4547d40e26fcc6d68785fac53b11bda6a9ab6d`
GPU: RTX 3090 24 GB, CUDA 13.0
Checkpoint: `ByteDance-Seed/academic-ds-9B`, revision
`2a205727f616995ad4aba57806f8e2d0d2f28743`

This is a bounded compatibility implementation for the academic-ds-9B DeepSeek-V3
recipe. It is separate from formal D3 task acceptance. The grouped CUDA runs below
are historical: they exercised grouped GEMM before the targeted B14 check found
small-row output and route-gradient differences. Current D3 policy disables that
GEMM for this recipe; see the B14 follow-up and trainer-integration note. A new
post-fallback equivalence smoke is recorded in `post-fallback/report.md`.

## Changes

- `expert_training.py` admits DeepSeek-V3 only on Linux CUDA with native BF16,
  sequential/grouped expert backends, and the pinned recipe geometry
  `(hidden=2048, expert intermediate=1408, experts=64)`. Existing router-aux,
  activation, cache-layout and budget checks still apply.
- The streamed model builder initializes DeepSeek RoPE frequencies on the target
  CUDA device using the Transformers model-owned initializer. This preserves the
  pinned checkpoint's YaRN arithmetic and fixes the CPU-created meta-skeleton
  buffers before training.
- Grouped restore handles the academic-ds-9B expert geometry. The separately
  validated grouped-GEMM shape allowlist does not include these projections.

The work does not claim broad DeepSeek-V3 compatibility. It does not validate
other model geometries, operating systems, dtypes, backends, data, convergence,
L2L, throughput, or a production CLI recipe.

## Validation

- Targeted Ruff using the repository `pyproject.toml`: passed on all three
  changed runtime modules, both harnesses and both relevant test modules.
- CPU regressions: **29 passed, 39 deselected**. The 39 deselected cases are
  GPU-marked; 17 non-failing deprecation/marker warnings were emitted by this
  remote scratch-runner invocation.
- Four fresh full-model `trl.SFTTrainer` CUDA runs completed one synthetic
  512-token batch and one SGD update each through the production
  `build_streamed_model` path, without a validator monkeypatch or a manual RoPE
  initialization call. Each changed all 160 attention-LoRA tensors.

| Codec | Backend | Loss | Grouped matrices | Initial adapter SHA-256 | Updated adapter SHA-256 |
|---|---|---:|---:|---|---|
| NF4 | grouped | 0.9558600187301636 | 628 | `66cf16f48ca78034af6d3ed34695eebb6c1a3bcc99b7c975a49868f55509cfa0` | `39aebfd37dc94449e55bd786188799920beaad52a8f235c660038c0dac1b5383` |
| NF4 | sequential | 0.9558600187301636 | 0 | same | same |
| Nested NF4 | grouped | 0.9597928524017334 | 644 | `66cf16f48ca78034af6d3ed34695eebb6c1a3bcc99b7c975a49868f55509cfa0` | `76fc0bdf4782509236ad300e71649a63c635bf440346e81260193733cf87229f` |
| Nested NF4 | sequential | 0.9597928524017334 | 0 | same | same |

For each codec, grouped and sequential runs had exactly equal loss and matching
initial and updated adapter hashes. This establishes a one-step trainer
integration/equivalence smoke for the fixed synthetic batch; it is not a
performance result or a quality/convergence claim. Attention implementation
was forced to eager for this test-only reproduction.

The first attempt using a reused output name correctly refused to overwrite the
prior run; it is not counted as a new run. The four counted runs use fresh
output directories and explicit zero exit codes.

## Reproduction packet

`academic-deepseek-support-final-20261109.tar.gz` contains all four run outputs,
logs and exit codes, the exact staged source/test snapshot, environment details,
and an internal SHA-256 manifest. Archive SHA-256:

`c5a96cf2c6664723f44a11894411e7140ef29bc0cc0e7688cbad0c6400044628`

No throughput benchmark, full repository test suite, or CI run was performed.
The original D3 checklist remains a separate acceptance decision.
