# Resume harness and exact-byte retention

This resumes the same USD 5 compilation share, including USD 0.144429 already
spent. The preceding [blocked report](../stream8-compile-20261022.md) remains
historical evidence. Source production files remain frozen at upstream
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`; research overlays are explicit.

`capture_inventory.py` constructs models on the **meta device** and inventories
ordinary rank/alpha 4, seven-projection PEFT parameters without loading base
weights. It computes tensor payload sizes from shapes and observed trainer dtypes.
For Qwen3-8B: 504 FP32 adapter tensors, 43,646,976 bytes per state; four training
microbatches produce 2,664,005,696 payload bytes including FP32 logits. The bare
semantic path uses BF16 logits and has 97,626,632 payload bytes. These figures
exclude safetensors headers, checkpoint optimizer/tokenizer files and logs.
Actual archive indexes and file manifests govern transfer admission.

`bench.py` compiles only resident decoder `inner.forward` functions. It preserves
the previous selection/compilation core and all tensor arithmetic/gates, adds
separate setup, warmup/update and inclusive compiler-time telemetry, and records
whole-run peaks across per-measured-update peak resets. Source SHA is recorded
in each run. Scheduling, ownership and checkpointing remain eager. Fullgraph,
disabled CUDA graphs/error suppression and measured-window recompilation checks
are retained. No activation-retention kernels or production flags are introduced.

The frozen ordinary SFT wrapper does not have the later research-only
`_streaming_liger_patches` report field. A guarded bridge initializes an absent
field to an empty dictionary only for `training.use_liger=false`; existing
metadata is never cleared. The first failed GPU setup that revealed this is
retained separately. Production files and numerical tolerances are unchanged.

`compare_streamed.py` independently replays the original elementwise comparator
in bounded-memory slices for large logits. Primary remote qualification still
uses the original comparator unchanged. Oracle tests verify agreement for finite
captures, exact-input/initial-state failures, late-slice logits changes, NaN and
three-decimal loss failures. This avoids allocating multiple full FP64 copies of
multi-gigabyte captures on the local Windows host.

The shared [`lossless_results.py`](../harness/lossless_results.py) codec preserves
every original byte and path. It uses content-addressed chunks, byte planes,
compression and exact XOR references inside self-contained tar archives. Encoded
objects, reconstructed chunks/files and the canonical archive have SHA-256 checks.
Duplicate reconstructed files may share hard links. It changes neither tensor
values/dtypes nor comparator inputs. A local ACK follows complete archive and
every-file verification. Raw, encoded and compressed sizes are distinguished;
synthetic/historical codec calibrations are not model-performance evidence.

The scoped CPU suite passes 45 tests with
2 CUDA skips. The rented target software separately passes 16 regressions,
including CUDA asynchronous ownership checks. These checks do not substitute for
full pretrained-model qualification or timing results.

```bash
PYTHONPATH=src python -m pytest tests/test_streaming_residency_candidate.py \
  tests/test_streaming_residency_compile_bench.py tests/test_lossless_results.py \
  tests/test_stream8_compile_resume_20261022.py \
  tests/test_stream8_archive_ranges_20261022.py --no-cov -q
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

The [resume report](RESULTS.md) distinguishes successful, failed and unattempted
model/arm gates and retains failed attempts. No universal model-size maximum is inferred.

The three imported legacy runtime modules are included as ordinary source files
under `benchmarks/harness/`: `liger_streaming_bench.py`, `liger_streaming_parity.py`
and `streaming_optimization_bench.py`. They preserve the AST of the executed,
archived versions; one long assertion is only wrapped for lint. Regression tests
check those frozen ASTs and exercise the published original comparator directly.
The executed benchmark and evidence snapshots disable Git newline conversion so
their recorded byte hashes survive checkout on both Windows and Linux.

`archive_ranges.py` and `remote_ranges.py` implement exact archive transport.
They validate encoded objects against the new manifest, reuse identical objects
from earlier complete or partial transfers, and retrieve missing immutable byte
spans in bounded batches. The whole canonical tar is reconstructed byte for byte
and must match the remote archive SHA before the unchanged codec verifies every
original file. Reusing encoded bytes does not qualify a model or combine GPU
performance cohorts.

A retry timer measures the absence of durable byte progress. Continued progress
across multiple connection failures resets that timer; a genuine lack of progress
remains bounded. Regression tests cover that distinction, corrupted prior objects,
resuming across span boundaries, canonical archive identity and all original bytes.
An incomplete response with SSH status zero also resumes under the same bounded
progress timer; it cannot create an ACK without all byte counts and hashes.
The standalone offline oracles also reconstruct an actual retained 1.046 GB server
archive with its original SHA, without network traffic. These are retention tests,
not pretrained-model performance measurements.
