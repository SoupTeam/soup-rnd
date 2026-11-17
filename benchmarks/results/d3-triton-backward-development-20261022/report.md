# D3: Triton activation and frozen-expert backward development

Date: 2026-10-22. **D3 is not complete.** This record contains development tests
of new runtime code, not a prospectively committed trained-model benchmark,
production training support or a positive performance verdict.

## What changed

`expert_group_autograd.py` adapts the original Soup reload/recompute design to
the bounded RND expert pool. It saves gathered inputs, reloads frozen weights
for first derivatives, computes backward inside the lease, and leaves native
route multiplication/scatter in autograd. The pre-existing forward-only APIs
and public trainer guards remain unchanged. Routed-expert LoRA and higher
derivatives are unsupported.

`expert_group_activation.py` fuses BF16 SiLU and multiplication with `up` into
one Triton activation kernel. It uses libdevice `exp`, round-to-nearest division,
and explicit intermediate BF16 rounding before multiplication. The development
training primitive selects it only with `activation_backend="triton"`; native
activation remains the default. It does not fuse NF4 decode with GEMM. Forward
grouped GEMM remains the explicit cuBLAS backend; backward uses native eager
recomputation. The grouped/Triton path refuses FP16 autocast before expert reads.

## Executed checks

| Development snapshot | CUDA cases | Result | Purpose |
| --- | --- | --- | --- |
| 01 | 14 | All pass, no skips | Isolated activation and tiny frozen-expert derivatives |
| 02 | 22 | All pass, no skips | Adds explicit fusion integration, OLMoE-size matrices and full tiny-model steps |
| 03 | 25 | All pass, no skips | Final source with pre-read FP16-autocast refusals |

These overlapping sessions are **25 unique current CUDA cases**, not 61
independent benchmark samples. No GPU test failed or was excluded. The source
was developed between sessions; every session's exact source and logs remain
in its own archive. Snapshot 03's two runtime modules and two test files match
the current local bytes. The source was uncommitted, based on `d48ea507`; these
tests cannot be re-labelled as prospectively registered after a later commit.

The final 25 cases cover:

- Eight activation cases: varied/empty geometry, input preservation, exact BF16
  bits including signed zero, every finite BF16 gate value with three multiplier
  patterns, budget/autograd refusals and allocated-peak measurement.
- Six small derivative cases: group widths 1/4, BF16/NF4/nested NF4, native
  output/input-gradient/router-gradient equality and actual grouped dispatch.
- Six OLMoE-geometry cases: synthetic `(2048, 2048)` gate/up and `(2048, 1024)`
  down matrices, native/Triton activation, all three storage formats, exact
  derivatives and two identical nonzero low-rank adapter updates. Each case
  records 32 grouped matrix executions across the two steps.
- Two complete optimizer-step cases through a synthetic two-layer MoE, with
  native attention/router/norms/head, one shared reusable expert pool and an
  input adapter. Both activation backends match native logits, nonzero adapter
  gradients and updated parameters across two steps. These weights are random;
  this is not the pretrained sixteen-layer OLMoE model.
- Three pre-read FP16-autocast refusals for the CUDA grouped/Triton path.

Local CPU development: 37 cases and 16 GPU-probe structural cases passed together
on the final source (53 passed, 25 GPU cases deselected). Selected new-file Ruff
checks pass. The full repository
suite and current-head CI have not been run for these uncommitted changes.

## Measured activation allocation

On RTX 3090, input shape `1024 x 2048`, BF16, after kernel warm-up:

| Path | Peak allocated bytes above the already-live baseline |
| --- | ---: |
| PyTorch SiLU then multiply | 4,194,304 (4 MiB) |
| Triton fused activation | 2,097,152 (2 MiB) |

The input itself is already live and occupies 4 MiB in both paths. The removed
temporary is 2 MiB. This is **not a 50% reduction in total model VRAM**: weights,
other activations, routing, library workspaces, context and allocator reserves
are outside this local comparison. No training throughput or total-model
memory reduction has been measured here.

## Environment and verification

One RTX 3090, 24 GB, sm_86, native BF16; driver 595.99.02. Linux/Python 3.12.15,
torch 2.14.1+cu130 (CUDA 13.0), Triton 3.8.0, Transformers 5.19.0, bitsandbytes
0.50.2, safetensors 0.8.0, pytest 9.1.1. Dependencies were installed in an isolated
venv. The installer placed a local PyTorch wheel there; the image's original
cu132 environment was not changed. Preflight listings describe that base
environment, not the test interpreter's CUDA build. Executed package metadata
is in each receipt (`torch` metadata reports 2.14.1). The separate
`runtime-inventory.json`, collected after the tests from the unchanged execution
venv, records actual imported module/build identities and cuBLAS SHA-256. This
post-test inventory is not a test replay or prospective environment freeze.

Each raw ZIP contains the executed source snapshot, collection log, pytest log,
JUnit, source hashes, software receipt and GPU state. An independent local
stdlib verifier checks ZIP CRC, every payload/source SHA-256, equality of
collected and executed node IDs, zero failures/skips and the memory properties.
`summary.json` records archive identities and verification counts. The included
`verify_archive.py` takes three positional arguments: archive path, expected
archive SHA-256 and output JSON path. It needs only Python's standard library.

The raw sources retain their Windows-origin line endings as executed; they are
not normalized inside the ZIP. Public JSON/Markdown use LF, and publication
checksums hash those actual bytes. Historical D3 protocols, measurements and
failed sessions are unchanged.

## Remaining work

The trained complete-model training step and its numerical/performance gate
remain unmeasured. They need a separately prepared and committed prospective
protocol before formal runs: identical data/math, independent native reference,
effective changed/restored controls, storage/peer validity, fixed attempt budget,
complete-step memory/time and actual kernel counts. No speedup threshold is
introduced. B1 interface acknowledgement, repository tracking and final D3
acceptance remain separate; see the full-task audit in the design document.
