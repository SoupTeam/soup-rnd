# Streamed experts: B1.2 forward and B1.3 backward

This internal RND primitive adapts the forward loop from original Soup
[`d500ae0d`](https://github.com/MakazhanAlpamys/Soup/tree/d500ae0dca7c4da905f8d59c9866af2f8aaa824b).
It reuses the [B1.1 cache codec](expert-shard-format.md), preserves the native
eager arithmetic and loads one routed expert at a time. Experts without tokens
are not read. Coverage can approach 100%, so reduced disk reads are not promised.

B1.3 adds first-order backward through frozen routed experts and internal
DeepSeek-V3 runtime/L2L integration. Non-expert LoRA adapters can train; routed
base weights and adapters on routed experts remain unsupported. Backward refetches
each frozen expert inside a bounded lease and accepts only the validated stateless
SiLU activation without hooks; higher-order gradients are refused. The routing weight
multiplies the per-expert node's output with ordinary autograd, so its gradient needs no
reload.
The public L2L launch guard admits Llama/Mistral, and DeepSeek-V3 only with
`stream_expert_layout: per_expert`; planner, budgets and defaults are unchanged. The only
public expert-training entry point is the opt-in B6 recipe below.

CPU and CUDA tiny-model tests compare gradients with resident controls and
exercise optimizer updates and adapter reload. The opt-in SFT configuration is
`training.stream_expert_layout: per_expert`, with `stream_source: disk`. Its disk
preflight currently requires a new single-disk NF4 cache without double quant.
The default remains fused storage. Real giant results and limitations belong in
[the B6 record](../benchmarks/b6-giant-moe-20261028.md).

Meta construction keeps native fused expert factories on the meta device,
avoiding a temporary giant CPU reservation before parameter registration.
Normal model buffers keep their constructor values; a changed expert
constructor that introduces buffers is refused instead of losing those values.

## Memory and ownership

`plan_expert_buffer(reader, dtype=...)` computes a single weight slot independent
of expert count. `ExpertBufferPool` allocates one byte arena with aligned views
for dense gate/up and down matrices, packed weights, statistics, shared NF4 code
tables and explicit nested-statistics scratch. The budget includes padding.
An insufficient budget fails before allocating the arena or reading an expert.
On CUDA, the runtime and preflight use `expert_allocation_budget`: logical
storage plus an explicit 32 MiB allocator reserve. Observed allocator growth is
checked against that finite budget before the first expert read. Different
allocator configurations can still exceed the reserve and fail explicitly.

CUDA NF4 decoding uses bitsandbytes `out` kernels directly into this arena.
Missing CUDA kernels fail explicitly; there is no allocating CUDA fallback.
CPU bitsandbytes 0.50.2 has no such `out` kernels, so CPU development checks use
temporary CPU dequantization results before copying into the arena. The CPU
arena bound does not bound those CPU temporaries or the entire process RSS.

Activations, routing masks, GEMM/backend workspaces, context memory and allocator
reservations are separate from the expert-weight budget. Record total CUDA peak
allocated **and** reserved memory separately; do not describe a 64 MiB weight
arena as a 64 MiB total-VRAM requirement. Host staging also has its own byte count.

One aligned CPU staging buffer is reused. Returned expert weights never alias
staging, but they alias the reusable compute arena and are valid only inside
`pool.lease(reader, expert)`. Nested/concurrent leases and closing an active lease
are refused. Copies are blocking; CUDA synchronizes before releasing a lease,
including on errors. This intentionally favors explicit lifetime safety, with
no asynchronous-prefetch or throughput claim.

Readers sharing a pool must agree on shapes, NF4 specs and stored dtype, and fit
its staging span. The pool owns its buffers; the caller owns and closes readers.
`pool.close()` releases buffer references, not PyTorch's global caching allocator.

## Forward and installation

`streamed_experts_class()` is lazy, so importing the module does not import
torch, transformers, bitsandbytes or PEFT. Its module accepts hidden states
`[tokens, H]`, int64 expert indices `[tokens, k]` and finite routing weights of the
same shape/device. The hidden dtype must match the pool. Expert IDs are visited
in ascending order; token gather, gate/up activation, down projection, route
weighting and index-add follow the resident eager implementation.

`last_trace` retains the latest call's layer, expert ID, token-route count,
logical bytes and load status. It is bounded by the number of routed experts;
it is not an unbounded in-memory training log. `pool.stats()` reports its plan,
allocation delta, cumulative fetches and logical bytes. Logical read bytes are
not a measurement of physical-host disk I/O. The gate records complete routing
indices and weights separately.

`install_streamed_experts(layers, readers, pool)` validates every layer before
replacing anything. It requires the native fused `gate_up_proj`/`down_proj`
contract and frozen routed weights; extra parameters/adapters, mismatched
readers/shapes, duplicate destinations and missing expert modules are refused.
Only the routed-experts submodule is replaced. Router, correction bias, shared
experts and dense decoder layers remain with the original DeepSeek/Kimi block.
The full block and shared-expert call count are tested separately.

The internal `build_streamed_model` path installs one reader per MoE layer and
one shared expert pool before taking layer weight maps, so routed weights are
excluded from decoder-layer slots. `StreamRuntime` owns and closes both the
readers and the pool. Its statistics include `expert_fetches`,
`expert_bytes_read` (logical stored bytes) and `expert_buffer_bytes` (the arena).
Device placement is fixed when the pool is created; moving an enclosing module
does not move the pool.

## B1.3 backward and L2L

Each hit expert's autograd node saves only its gathered hidden states and
routing weights, never a view of the reusable arena. Backward first unpacks
those activations (which may trigger checkpoint recompute), then leases the
same expert again through the reader/pool and rebuilds the native eager local
graph under the forward's autocast state. Gradients flow to hidden states,
routing weights and supported non-expert adapters, not routed base weights.
This is first-order autograd, not a higher-order differentiation interface.
Differentiable calls (forward with a grad-bearing input, and again before backward reads
any expert) admit only the unmodified native stateless SiLU: any other activation, a
parameter, buffer or child, or a forward/backward hook is refused with
`expert backward requires an unmodified stateless SiLU activation`, so recompute can
neither update hidden state twice nor draw random numbers the forward did not. A
stochastic or stateful activation still works for no-grad B1.2 forward calls.

The accepted one-slot read cost is three reads per hit expert/micro-batch:
ordinary checkpointed training does forward, checkpoint recompute and expert
backward; L2L does no-grad forward, grad recompute and expert backward. Unused
experts are not read. A decoder-layer load counter is not an expert-read count.

Internally, `L2LStep.run` and `make_l2l_trainer_class` with `attach_l2l_state`
run native `deepseek_v3` in addition to the dense architectures. Supported non-expert
LoRA includes input embeddings, MLA projections, shared experts and the output head.
After decoder backward, trainable embeddings are recomputed and differentiated one
micro-batch at a time through the existing streamed large slot; frozen embeddings keep
the no-grad fast path. Embedding recompute restores its forward RNG for NEFTune noise
without advancing the outer stream. L2L replays decoder and native-head dropout in
micro-batch accumulation order, including head draws between micro-batches, and leaves
CPU/CUDA RNG where ordinary accumulation leaves it. Router choices are checked between
the forward and recompute. Masked labels, short final windows and uneven windows run each
micro-batch at its own length, as in accumulation; uneven windows with dropout are refused
because rows of different lengths draw different masks. MoE trainer logs add
`l2l_expert_fetches` and `l2l_expert_bytes_read`. None of this widens the public
`training.l2l` architecture guard.

The existing NEFTune-plus-dropout refusal remains: their interleaved draws are not a
supported combination. Uneven windows with NEFTune remain refused as well.

The independent synthetic smoke needs no pytest invocation or model download:

```text
python benchmarks/harness/b13_moe_backward_smoke.py --device cpu --output smoke-cpu.json
python benchmarks/harness/b13_moe_backward_smoke.py --device cuda --output smoke-cuda.json
```

The default still checks 30 MLA adapter tensors. Optional synthetic harness arguments
`--targets embedding` (32 tensors), `--targets head-shared --dropout 0.05` (44) or
`--targets all --dropout 0.05` (46) exercise the embedding/head boundaries. These are
benchmark-only choices, not public training YAML/CLI switches.

It compares repeated k=1/2/4 L2L updates against an independent resident eager
model with identical nonzero LoRA, masked labels and ordinary accumulation. JSON
records source SHA/dirty hashes, environment, all trainable gradient/state differences,
CPU/CUDA RNG equality, reloads, reverse layer visits, arena pointer/plan/budget and CUDA
allocated/reserved peaks separately. A test-only control corrupts reader staging
only on backward reloads: forward/loss must remain equal while gradient
comparison fails. A correct-path PASS alone does not validate that control.
These are tiny synthetic correctness checks, not convergence, throughput or
whole-model memory-fit measurements. B1.2 gate results below remain historical
forward evidence; they are not B1.3 backward or CUDA verification.

## Expert reads per window

`tests/test_p3_expert_read_diagnostic.py` counts the reads of one L2L window at k = 1, 4, 16
and 64 micro-batches. It runs on the CPU, on the tiny DeepSeek-V3 fixture (one dense and two
MoE layers, eight experts per MoE layer), measured on 2026-11-05:

| k | layer loads | expert fetches | logical expert bytes | expert hits | unique experts | physical requests | physical bytes |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 39 | 239616 | 13 | 13 | 78 | 547536 |
| 4 | 3 | 162 | 995328 | 54 | 15 | 324 | 2276160 |
| 16 | 3 | 621 | 3815424 | 207 | 16 | 1242 | 8671128 |
| 64 | 3 | 2526 | 15519744 | 842 | 16 | 5052 | 35311200 |

An expert hit is one expert that the router chose for at least one token of one micro-batch
in one MoE layer. Unique experts are the distinct (layer, expert) pairs hit anywhere in the
window.

Layer loads do not depend on k. Each hit expert is fetched three times per micro-batch:
in the forward, in the recompute and in the expert's own backward. Fetches therefore equal
three times the hits and stay near 39 per micro-batch at every k. At k = 64 the window needs
16 distinct experts and reads 2526 times. An expert is read again for every micro-batch that
hits it, so the "one read per window" assumption does not hold for experts as implemented.
The test makes no speed claim.

Physical requests are the calls to `read_range_into` that expert readers make, counted by
wrapping that function. They are sector-aligned requests that the process hands to the OS,
not device reads, and the page cache is not observed. Layer-weight reads are not counted.
Every expert fetch is two requests (`gate_up` and `down`). Logical bytes are 44 percent of
the physical bytes on this fixture because each small tensor span is rounded out to sector
bounds; the ratio is an effect of the tiny tensors and will be much closer to 100 percent
for real experts.

Reproduce with `pytest tests/test_p3_expert_read_diagnostic.py -v` and read the recorded
properties with `--junitxml`.

### Reads with the complete levers on CPU

`tests/test_complete_reads_tiny.py` repeats the window on the same fixture for four arms
(ordinary, `grouped_expert_routes`, `two_read_experts`, and both together, which is every lever
that runs on a CPU), on gradient accumulation (GA) and on L2L, at k = 1, 4, 16, 64. Expert
fetches of a window:

| k | hits | GA ordinary | GA grouped | GA two-read | GA both | L2L ordinary | L2L both |
|---|---|---|---|---|---|---|---|
| 1 | 13 | 39 | 39 | 26 | 26 | 39 | 26 |
| 4 | 54 | 162 | 162 | 108 | 108 | 162 | 108 |
| 16 | 207 | 621 | 621 | 414 | 414 | 621 | 414 |
| 64 | 842 | 2526 | 2526 | 1684 | 1684 | 2526 | 1684 |

Reads follow k in every arm. Two-read cuts GA and L2L reads from three to two per hit
(ratio 2/3 at every k). Grouping changes no read.
Layer loads are 3 per micro-batch on GA (3k per window) and 3 per window on L2L. After one AdamW
step the LoRA gradients, parameters and optimizer state of every arm equal ordinary bit for bit.
`packed` and `expert_host_cache_bytes` need CUDA and are not measured. The test makes no speed
claim.

## Complete levers (experimental, off by default)

Four switches port the "complete" streaming levers of the koko branches onto this code, and a
fifth, `expert_prefetch`, is new here. All are API parameters of `build_streamed_model` and `install_streaming`, with no config field and
no CLI flag. All apply only to a `per_expert` cache of a DeepSeek-V3-family base, and any of
them on another cache raises a `ValueError` instead of being ignored. With none set, the
default path is unchanged: three reads of each hit expert per step (forward, checkpoint
recompute, backward), one copy per stored component.

| Parameter | Effect | Checked |
|---|---|---|
| `grouped_expert_routes=True` | Each streamed MoE layer finds its hit experts and their token rows with one `nonzero` and one host count read, instead of one `where` per expert. Same experts, same order, same reads. | CPU. Output, gradients and read order are bit-equal to the default path. |
| `two_read_experts=True` | The layer checkpoint gets a context in which the router keeps its top-k decisions in the forward and replays them in the recompute, and the recompute reads no expert. A step reads each hit expert twice. | CPU. Loss and every LoRA gradient are bit-equal to the resident model and the default path, in fp32 and under bf16 autocast. Reads are 2 per hit expert instead of 3. |
| `expert_transfer="packed"` | The pool stages each expert's stored NF4 components (packed weights and statistics) in one pinned block and copies it to the arena once, instead of once per component. Needs CUDA and NF4 in both matrices. The dequantised weights are the same bytes. | CPU: the byte layout, the plan and the refusals. The copy itself and bit-equality on a card are in `tests/test_complete_packed_transfer_cuda.py`, which has not been run. |
| `expert_host_cache_bytes=N` | With `packed`, keeps whole packed blocks in pageable RAM up to `N` bytes, first come first kept, no eviction. A hit reads nothing from the experts file and copies one block. | CPU: the byte budget and first-fit behaviour. The hit path on a card is in the CUDA test file, not run. |
| `expert_prefetch=True` | One worker thread reads the next hit expert of a layer into a second host staging buffer while the current expert computes. The forward hints the next expert of its visit order, the backward the previous one, and no hint crosses a layer. A lease for the hinted expert uses the background read; any other lease discards it and reads directly. One read is in flight in the whole pool. | CPU (`tests/test_p72_expert_prefetch.py`): loss, logits and every gradient bit-equal, same experts, order and counters, hints land on the worker, a failed background read raises in the lease that asked for it, no thread outlives the runtime. On a card: [PR #72 diagnostics](../benchmarks/p72-mbr-phase-profile-20261119.md). |

Prefetch changes when a read starts, never which bytes are read. `reader.fetches` and
`reader.bytes_read` also count a read-ahead that was discarded, while the pool's `fetches` and
`logical_bytes_read` count only the loads a lease used. With hints on, a `last_trace` row's
`logical_bytes_read` is the bytes of its own lease for the same reason. A hint is not issued for
an expert already in the host cache. Measured on Moonlight (RTX 3060, MBR group 8) with
`packed`, it takes the step from 18.8 s to 16.6 s, about -11% over three A-B-B-A blocks; the
read of one expert (about 3 ms) is longer than the compute it hides behind, so the disk still
limits the step.

The packed layout puts every stored component in one aligned region at the front of the
arena, so one copy fills it. The pinned block (`plan.host_pinned_bytes`) is host memory outside
`weight_bytes`. The host cache is charged to its own `N` and not to `stream_ram_cache_mb` or
any other RAM budget, and the pool's `fetches` counts loads, hits included, while
`reader.fetches` counts disk reads.

Two reads works through the layer checkpoint. Layers that `training.stream_recompute` switches
off have no recompute to skip and read each expert twice anyway. L2L bypasses the checkpoint,
and supplies a separate forward/recompute context pair for each (layer, micro-batch)
when the API enables `two_read_experts`. Reference passes stay outside these contexts.
The trainer never sets this experimental API option. In the recompute the
routed experts return zeros, so a forward hook on the decoder layer, its MoE block or its routed
experts, and any global forward hook, is refused with a `RuntimeError`. Under L2L this
also includes the streamed layer wrapper, which runs during recompute. A patched `forward` is
not detected and is unsupported. `install_streamed_experts(..., two_reads=True)` admits only
`DeepseekV3DecoderLayer`.

### Exact expert GEMM: `lt-fused` (experimental, off by default)

`soup_cli.utils.expert_gemm.expert_gemm_context(mode)` picks how a route-preserving MBR group
(`L2LStep.run(..., route_preserving_mbr=True)`) runs its expert GEMMs. It is an API, with no
config field and no CLI flag.

| Mode | Effect |
|---|---|
| `native` (default) | Nothing is patched and the environment is not probed. Native cuBLAS runs one call of M = all rows of the group, so the bits are the grouped-MBR bits. |
| `lt-fused` | Each expert call is cut into the micro-batch segments L2L would have run. A cuBLASLt call forced onto the kernel native cuBLAS picks for that segment replaces the native call, so the group keeps the A0 bits (L2L, group size 1). Forward and the direct backward of one expert visit are one extension call each (`a0_exact_lt`, `direct_expert_backward`). |

`lt-fused` builds a C++/CUDA extension (`a0_exact_lt_ext.cpp`, `a0_exact_lt_kernels.cu`) with
`torch.utils.cpp_extension.load` on first use ([ADR 0002](adr/0002-exact-gemm-optional-jit-extension.md)),
so an install without a compiler keeps working. It never falls back to `native`. Before it builds
anything it raises `ExactGemmUnavailableError` (a `RuntimeError`) when one of these fails:

| Requirement | Why |
|---|---|
| CUDA is available | the extension is cuBLASLt |
| compute capability sm_80 or newer | the algo table is checked on sm_80+ cards (tested on sm_89) |
| PyTorch 2.14 | the fused kernels mirror the ATen reduction order of that series; another series may change bits |
| a C++ compiler, `ninja`, `nvcc` and the cuBLASLt headers (`CUDA_HOME`, `/usr/local/cuda` or the `nvidia-cuda-nvcc` wheel) | the first-use build |

The fused kernels write bf16 rows with a 256-byte stride, so `hidden_size`, the expert
intermediate size and the doubled gate/up width must all be multiples of 128. A model
that breaks this gets an explicit `RuntimeError` naming the offending size; it never
falls back to `native`. Kanana (`hidden_size` 2048, intermediate 768) passes.

Platforms for this streamed-expert/MBR work: **Linux** is the supported target for the
training paths and the CUDA extension. **macOS** is supported on CPU paths only; the
CUDA extension does not build or run there, and no MPS bit-equality claim is made.
**Windows is not supported**: the streaming paths and the exact-GEMM extension are
untested there, and no Windows build or run is offered; this is the same scope note as
the mbr-unified PR. The #77 async extension source is vendored under
`src/soup_cli/utils/_native_expert_async/` for the loader arm; it uses
`st_mtimespec` on Apple platforms so its C++ source is at least macOS-clang clean, is wired
into the per-expert pool only behind the opt-in `cpp_async_2slot`, and its CUDA kernels
remain Linux-only.

Checked: CPU (`tests/test_expert_gemm_select.py`) each refusal with the environment probe
replaced, and that `native` patches nothing; CPU fakes of the extension
(`tests/test_a0_exact_lt*.py`, `tests/test_direct_expert_backward*.py`); GPU (`pytest -m gpu`,
`tests/test_a0_exact_lt_kernels_gpu.py`) the kernels against the ATen chains they replace on
sm_89, and that the installed PyTorch is the 2.14 series. The full-step CPU check
(`tests/test_p72_mbr_cpu_bits.py`) compares A0 with route-preserving MBR through the
exact plan on the tiny MoE, `torch.equal` over losses and every trainable gradient at
group sizes 1, 2 and 4. The speed of `lt-fused` and the bit-equality of a full step on
Kanana are measured by the benchmark harness, not by the unit tests; see
[the exact-MBR speed report](../benchmarks/mbr-exact-speed-20261120.md).

Measured on Kanana-2-30B-A3B (RTX 4060 Ti, 8-GiB allocator limit, one block, see
[the report](../benchmarks/mbr-exact-gemm-levers-20261121.md)): `lt-fused` is bit-equal to A0 at
k = 1, 2, 4, 8, which route-preserving MBR with the ordered backward is not from k = 2. It is not
faster than that arm (0%, -2%, -18% in the three windows that fit) and holds more memory (S = 512:
3.14 against 2.03 GiB allocated), so it is worth taking for the bits only. `async_lease` on top
gave +4 to +9% but one of four k = 2 runs differed from A0 (three repeats were equal), so treat it
as not bit-safe; `fast_row_gather` added nothing.

### Host-side levers: `async_lease` and `fast_row_gather` (experimental, off by default)

Two more switches change when the host waits and what it launches, never a number. They are
class attributes, not parameters of `build_streamed_model`, and have no config field and no CLI
flag. Set one on the class (`ExpertBufferPool.async_lease = True`,
`StreamedExperts.fast_row_gather = True`) or on one instance. With neither set the default path
is unchanged.

| Attribute | Effect | Checked |
|---|---|---|
| `ExpertBufferPool.async_lease = True` | With `expert_transfer="packed"` on CUDA, a lease ends without `torch.cuda.synchronize`. By default every lease ends with a device-wide wait, so the host cannot prepare the next expert while the card computes the last one. Under `async_lease` the only host wait left is for the H2D copy that reads the pinned staging region, before the region is written again. The weights, the dequantisation and the GEMMs stay ordered on the stream that loaded the expert, so the lease must be used on that stream. A lease that raises still waits for the whole device. Other transfers and CPU pools keep the old rule. The host-cache key of a reader path is also resolved once per pool instead of on every load. | CPU (`tests/test_expert_async_lease.py`): the default is off and syncs, only `packed` on CUDA skips the sync, the error path still syncs, the path is resolved once only under the option. CUDA (`tests/test_expert_async_lease_cuda.py`, `pytest -m gpu`): output and gradients bit-equal to the default lease, and no device-wide sync per lease. |
| `StreamedExperts.fast_row_gather = True` | A visit gathers its rows and its route weights with `index_select` and a backward that is one `index_add_` into zeros, instead of the advanced-indexing backward (`index_put_` with `accumulate=True`, a sort and about a dozen kernels). A token reaches an expert once, so the indices of a visit are unique and the backward is a plain scatter; `0 + g` gives the bits of the accumulate, the sign of a zero included. Repeated indices are outside the claim: in bf16 they round differently from the indexing backward, and on CUDA the order of atomics is not defined. | CPU (`tests/test_expert_fast_gather.py`): forward and backward bit-equal to indexing in fp32 and bf16, zeros with a sign included, the streamed experts equal the default path, and no `IndexBackward0` node is left in the graph. |

Measured on Moonlight-16B-A3B (RTX 3060, MBR group 8, window 8x64, packed transfer, host cache,
prefetch), as the difference between neighbouring steps of one process with the option switched
on for every second step (16 steps, 15 pairs): `async_lease` took the step from 6.25 s to 5.32 s
(-0.90 +- 0.04 s) on a Ryzen Threadripper PRO host and from 11.43 s to 9.99 s (-1.32 +- 0.25 s)
on a Xeon E5-2697A v4 host; `fast_row_gather` made no difference on the first (-0.01 +- 0.05 s)
and -0.07 +- 0.03 s on the second. These are single-model engineering measurements; the gradient
digests of every step were unchanged. In the same setting the exact MBR of the benchmark harness
with both levers was 5.6% (first host) and 9.1% (second host) faster than stock route-preserving
MBR over five A-B-C-C-B-A blocks; see
[the exact-MBR speed report](../benchmarks/mbr-exact-speed-20261120.md).

Not ported from the same koko commits: the one-expert lookahead (`prefetch`, `prefetch_after`),
the usage profile and the GPU placement plan. They are not among the four complete levers, and
they build on an expert fetcher with cache tiers that this code does not have.

### The loader arm: `stream_expert_loader` (experimental)

`training.stream_expert_loader` chooses which loader feeds the per-expert cache. The default
`async_lease` is the single-slot pool above and changes nothing. `cpp_async_2slot` selects the
C++ two-slot loader vendored from PR #77 in `utils/_native_expert_async`; it is valid only with
`stream_expert_layout: per_expert` and `stream_layers: true`. Selection is never replaced by the
Python lease: without the compiled extension or without CUDA the run stops before the tokenizer
is loaded and prints the `build_ext.py` command. With both present the expert pool reads every
lease through `PackedStreamAsync` (ring 2, two I/O workers) instead of
`ExpertShardReader.read_host`, on CUDA filling the same packed block that
`expert_transfer="packed"` copies to the device once. The extension is built with
`python src/soup_cli/utils/_native_expert_async/build_ext.py build_ext --build-lib <dir>
--with-cuda-nf4`; `build_ext.py` diagnoses the CUDA 13 / glibc 2.43 `rsqrt` conflict before
`nvcc` and prints the documented two-line header patch without editing anything. CPU tests:
`tests/test_expert_loader.py` (selection, the wired pool through a stream-shaped CPU stand, the
same bytes driving `FusedCpuBackend`, and the compiled stream itself when it is on
`PYTHONPATH`), plus `tests/test_native_build_ext.py`. GPU scaffolds that skip until the
extension is built: `tests/test_nf4_decoder_bnb.py` (decoder against bitsandbytes 0.50.2 on
`uint16`) and `tests/test_native_expert_async_race.py` (rings 2/4/8, cancel, read failure).

## Local checks

```text
python -m pytest tests/test_b13_expert_backward.py tests/test_b13_moe_runtime.py --no-cov -v
python -m pytest tests/test_b13_l2l_moe.py tests/test_b13_moe_trainer.py --no-cov -v
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

B1.3 checks compare native eager outputs, losses and nonzero gradients on dense
and NF4 storage, reload/lifetime behavior, autocast and L2L windows. The trainer
contract is bitwise fp32 for equal lengths, `atol=2e-5, rtol=1e-4` for uneven
lengths and `atol=1e-6, rtol=1e-5` for bf16 autocast; tolerances are fixed before
results. The B1 forward tests also cover dense and NF4 exactness, nested
quantization on/off, varied top-k and seeds, empty/partial/all-hit routing, a
same-process different-weight negative control, one-slot reuse, insufficient
budget, exceptions, explicit higher-order/unsupported-activation refusal and full
native DeepSeek/Kimi MoE blocks. CUDA checks must be reported as skipped when CUDA
is unavailable. Historical B1.2 tests/gates below do not establish the B1.3 result.

## Historical B1.2 prospective trained-checkpoint gate

The [protocol](../benchmarks/protocols/b1.2-expert-forward/protocol.json) pins
OLMoE `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, full checkpoint hashes,
package versions, layers 0/7/15, 512 synthetic hidden states, a 64 MiB expert-weight
budget, exact-equality and negative-control rules, and **one attempt per cell
with zero replacements**. Its 12 cells cover CPU fp32, CUDA bf16, CUDA NF4 and
CUDA NF4 with nested statistics. A single cell's PASS is not a complete task PASS.

Commit the source and protocol before the first trained-checkpoint arm. Use a
clean checkout with results and model data outside it. The harness refuses a
dirty checkout, verifies full checkpoint bytes and refuses existing attempt
directories. Do not bypass the attempt budget by changing session directories.
Review any failure/NO VERDICT before separately proposing a new protocol; retain
every attempted cell, traceback, progress file and original decision rule.

Checkpoint preparation is approximately 14 GB of downloads. Allow about 64 GB
of disk space for checkpoint, converted layer files and logs. No complete model
is loaded onto the GPU; the harness loads one selected layer at a time. Use one
CUDA GPU that supports BF16 and has room for its resident reference. Two SSDs,
bare metal and a long performance rental are not requirements of this gate.
These are preparation estimates, not a measured memory-fit claim.

After the user commits the protocol, downloads the pinned snapshot and selects a
CUDA environment matching it, the Linux sequence is:

```bash
for layer in 0 7 15; do
  for cell in cpu-fp32 cuda-bf16 cuda-bf16-nf4 cuda-bf16-nf4-dq; do
    python benchmarks/harness/b1_expert_forward_gate.py \
      --model-dir /workspace/olmoe-pinned \
      --session /workspace/b1.2-forward-session \
      --layer "$layer" --cell "$cell" --power-source server || exit 1
  done
done
```

Each cell runs in a new process. The real checkpoint router supplies top-k
routing. For NF4, the resident reference is independently reconstructed from the
same stored packed bytes and scales. The original resident stack is released
before the streamed memory arm. Results preserve allocated/reserved baselines
and peaks, source identities, package versions, routing, fetch order and exposed
NVIDIA state. Power source is an operator declaration, not sensor certification.

The generic OLMoE expert-layer gate does not add OLMoE to the RND checkpoint
sharder's supported architectures; B1.1 conversion remains DeepSeek-V3-family.
Original Soup's earlier forward/backward gate is cited as prior evidence only.
This path makes no new convergence, throughput, giant-model, physical 8 GB fit,
physical-host isolation or backward-correctness claim.

## Historical B1.2 local verification, 2026-10-20

On the B1.2 working tree based on B1.1 `9510096a`, Windows/Python 3.11.15,
torch 2.14.1+cpu, transformers 5.19.0, bitsandbytes 0.50.2 and safetensors 0.8.0:

- B1.2 unit/instrument checks plus B1.1/B2/Kimi, trust-remote-code and provenance:
  199 passed, 48 skipped, 8 existing PEFT pattern warnings. CUDA was unavailable.
- Full `ruff check src/soup_cli/ scripts/ tests/ benchmarks/` and
  `git diff --check` passed; the new files were checked for trailing whitespace.
- No full-repository pytest run or trained-checkpoint/CUDA gate was executed.

The borrowed CPU environment lacks pytest-cov, so this local run used
`python -m pytest -o addopts='' ... -q --tb=short`, with worktree `src` and a
separate TRL dependency directory on `PYTHONPATH`. Repository test configuration
and the borrowed environment were not changed. The B1.1 interrupted-conversion
fix and its regressions are included in this branch's base.

## Historical B1.2 synthetic CUDA preflight, 2026-10-20

The retained [raw archive](../benchmarks/results/b1.2-cuda-preflight-20261020/raw-results.zip)
contains the exact uncommitted source bytes tested on Linux/Python 3.12.13,
RTX 3060 (12 GB, native BF16), torch 2.14.1+cu130 and the pinned package versions.
All seven selected tests passed without skips in 3.97 seconds: two B1.1 CUDA
NF4 codec cases, two complete native DeepSeek/Kimi MoE cases and three streamed
BF16/NF4/nested-NF4 forward and weight-arena cases. The last three retain their
same-process negative controls, expert traces and allocated/reserved memory.

Measured single-slot arena allocations were 3,072 / 4,608 / 7,168 bytes and
matched the corresponding plans and budgets. These are tiny synthetic fixtures,
not trained-checkpoint memory-fit or total-VRAM results. The separate
[verification](../benchmarks/results/b1.2-cuda-preflight-20261020/verification.json)
confirms ZIP CRC, all 606 source hashes, identity with the sent payload, seven
JUnit passes, negative controls, load order/count and arena allocation bounds.
Source identity is recorded by manifest and base `9510096a`; it is not presented
as a precommitted trained-checkpoint protocol.

## Historical B1.2 trained-checkpoint gate, 2026-10-20

The prospective gate passed all 12 cells at committed source `99664b09` on the
RTX 3060: trained OLMoE layers 0/7/15, each in CPU fp32, CUDA BF16, CUDA NF4 and
CUDA nested NF4. Recorded output hashes match exactly, max absolute difference
is zero, and every different-weight negative control differs. All 64 experts
were routed and loaded once in ascending order in every cell. No replacement,
failed or unknown cell is present in this series.

CUDA arena allocation was 12 MiB for BF16 and 16 MiB for either NF4 variant,
below the frozen 64 MiB budget. Streamed peak allocated memory was approximately
28.11-32.34 MiB for this layer primitive; this is not a whole-model VRAM bound.
The [report, raw archive and replay analyzer](../benchmarks/results/b1.2-olmoe-forward-20261020/report.md)
retain all cell logs, routing, traces, memory, hashes, protocol and source.

## Historical B1.2 acceptance status

CPU regression, synthetic CUDA preflight and the 12-cell prospective trained-layer
forward/arena gate passed. The B1.1 interrupted-layout review correction is
inherited by this branch. B1.1 dependency review, publication, required CI and
B1.2 reviewer acceptance remain; the task board is not marked Done. A new
changelog fragment must use the eventual B1.2 PR number; B1.1's #24 fragment is
not the B1.2 entry.
