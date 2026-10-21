# D3 complete-model forward gate: preparation

Status on 2026-10-21: new-host CUDA smoke, large-shape integration controls,
immutable inputs and real instrument controls are verified and pinned.
**No trained full-model CUDA cell has run.** The enabled recipe still requires
the operator's source/protocol commit and a clean checkout before execution.
This is not completed preregistration or D3 acceptance.

## Dependencies checked

The [dependency audit](dependency-audit-20261021.json) records current GitHub
metadata: B1.1 #24, B1.2 #25 and F4 #11 are merged. The five reused B1 runtime
files match `main` at `fdf067fc` byte-for-byte. No merge or source commit was
performed during that check.

F4 was approved as a descriptive baseline with its limitations. Its review did
not require a new paid run or certify the missing original validity conditions.
The new D3 layer study supplies its own matched comparison rather than treating
F4's modeled dense-layer factor as measured evidence. Interface coordination can
be reviewed against this shared-pool contract; no developer was messaged and no
human coordination/sign-off is claimed here.

## What this implements

`d3_full_model.py` assembles native OLMoE embeddings, attention, norms, rotary
embeddings, routers and LM head. Only routed experts are replaced. One weight
pool is shared across every streamed layer; it is not allocated sixteen times.
There are no public trainer flags, backward support or KV-cache claims.

The independent resident oracle loads dense expert bytes directly, or decodes
the exact NF4 bytes through the audited B1.2 bitsandbytes reference. The grouped
resident control uses the same `GroupedComputeExperts` and grouped GEMM backend
as the streamed candidate. Native attention and routers remain unmodified.

The proposed trained recipe runs all sixteen OLMoE layers at 512 synthetic token
IDs, batch one, with full logits. It is a complete forward through trained
weights, **not real-text validation, a quality/convergence study or training**.
BF16, NF4 and nested NF4 each have four fresh-process cells: native resident,
sequential streamed, grouped resident and grouped streamed. Every cell requires
exact positive/A-A output, an effective active-weight negative and a restored
positive. Numeric mismatch is a failure; unavailable execution is no verdict.
The session stops on either, with no replacements.

## New-host dispatch correction, before the formal freeze

On the RTX 3090 (instance 57057176), the original cutoff-16 arithmetic probe
failed 10/32 comparisons even though all three tiny CUDA controls and 45
small-shape/runtime tests passed. The cuBLAS binary and package versions match
the earlier RTX 3060 study. Retained traces show a different native GEMM kernel
choice; this is not evidence of defective hardware or a successful full gate.
The first preparation stopped before downloading weights.

A separate development sweep retained 1,984 comparisons. Failures occurred in
OLMoE shapes at rows 17--64 and 97--128. The prospective whole-model recipe now
explicitly uses native mm through 128 rows for the OLMoE geometries, grouped GEMM
above that threshold, and the original cutoff 16 for tiny controls. This is a
fixed input-shape policy, never an output comparison/recomputation fallback.
The default layer-study policy remains 16. The previous speed estimates do not
measure this new policy, and no speed improvement on this host is claimed.

After the change, three tiny CUDA controls and all 48 grouped-compute tests
passed on this host. The added large-shape regressions cover both OLMoE inner
dimensions, every row count 0--512, two seeds, and both native/grouped dispatch.
Full-model grouped cells must record actual grouped GEMM work on their first
positive forward; an entirely native execution cannot pass as grouped coverage.
The failed preparation and diagnostic sweeps remain retained separately.

## Required machine

- One Linux GPU of the **16 GB+ class with native BF16**, CUDA sm80 or later.
  The protocol requires at least 15 GiB reported usable memory to allow normal
  driver reservations on a nominal 16 GB card. Two smaller GPUs cannot be added
  together to satisfy this single-device requirement.
- At least 16 GB host RAM; about 80 GB free workspace is a practical preparation
  allowance for the pinned checkpoint, three bundles, environment and outputs.
- One ordinary SSD is sufficient. Dedicated access to two physical drives is
  not required. Storage/GPU visibility limitations must remain explicit.

The routed BF16 expert tensors alone occupy a **calculated 12 GiB**:
`16 layers × 64 experts × (2×1024×2048 + 2048×1024) × 2 bytes`.
Attention, embeddings, LM head, activations and CUDA libraries are additional.
Therefore the existing 12 GB RTX 3060 cannot hold the full resident oracle.
This calculation is not a measured peak or a guarantee that any offered host fits.

## Before any trained full-model forward

1. Select the new host and verify native BF16, usable VRAM, free disk, software,
   cuBLAS binary identity and absence of visible competing GPU workloads.
2. Run the existing shape/exactness probes on that backend and the new **tiny**
   CUDA integration smoke. Tiny controls are distinct from the trained full gate:

   ```bash
   D3_CUBLAS_LIBRARY=/absolute/path/to/libcublas.so.13 PYTHONPATH=src \
     python -m pytest tests/test_d3_full_model.py \
     -k tiny_cuda_full_model -o addopts= -q
   ```

3. Supply the pinned OLMoE checkpoint files from the B1.2 protocol. Preparation
   verifies full checkpoint hashes and does not download weights:

   ```bash
   for codec in dense nf4 nf4-dq; do
     PYTHONPATH=src python benchmarks/harness/d3_full_model_gate.py prepare \
       --model-dir /workspace/olmoe-pinned --prepared /workspace/d3-full-prepared \
       --codec "$codec" --device cuda
   done
   ```

4. Verify all bundle files. Calibrate the named immutable files on this host and
   retain real baseline/own/independent-foreign F1 controls. Old-host inode/device
   receipts cannot be reused. Preserve unsuccessful preparation and controls.
5. GPU/software/library, bundle and instrument hashes are filled and retained in
   `setup-evidence.zip`. The rule is enabled. Have the operator commit/push the
   exact source and protocol **before the first trained full-model cell**.
6. Run from the actual clean committed checkout:

   ```bash
   PYTHONPATH=src python benchmarks/harness/d3_full_model_gate.py run \
     --prepared /workspace/d3-full-prepared \
     --library /absolute/path/to/libcublas.so.13 \
     --out /workspace/d3-full-forward-REAL-COMMIT
   ```

Do not run formal modes from the uncommitted development checkout. A session has one
attempt per actual source commit and zero replacements. OOM does not authorize a
silent retry. The full resident oracle deliberately allocates all expert weights;
only the streamed paths claim a bounded shared expert-weight arena.

## Validation recorded now

Original preparation CPU/regression suite: **145 passed, 72 skipped**, including new tiny
whole-model dense/NF4/nested-NF4 comparisons and execution/provenance controls.
Those three CUDA smoke cases subsequently passed on the RTX 3090, as described
above. The latest CPU regression suite reports 152 passed / 74 hardware skips;
the separate on-host CUDA tests supply the actual GPU evidence. No instance was
restarted, stopped, destroyed or rented by the agent.

Full repository Ruff is clean. No full repository pytest or new trained CUDA
result is claimed. The selected log and exact preparation source are retained in
`preparation-evidence.zip` (the original CPU-only preparation snapshot).
`setup-evidence.zip` retains the subsequent real GPU preparation, first failed
arithmetic probe, diagnostics, corrected-shape regression, immutable bundle
identities, all 48 mapped files, and baseline/own/independent-foreign controls.
Its 24 outer entries and the 61 entries in the retained host archive were
verified, including all archive CRCs and manifest hashes. The local
`setup-verification.json` states that zero trained full-model cells have run.
The operator commit, twelve trained cells and implementation/interface review
remain outstanding.
