# Frozen streaming-residency experiment

[Report](../../streaming-residency-20261020.md) and
[results](../../streaming-residency-results-20261020.json) describe the completed run.
This archive preserves the exact tested research overlay, model preparation, inputs,
package constraints, environment and GPU regression record. It contains no provider
credentials, rental controller, model weights or captured tensor binaries.

The tested production baseline is upstream commit
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`, not the older research branch containing
this report. Use a separate checkout of that commit and copy the archive's `overlay/`
contents onto it. Do not apply unrelated Liger/compile/attention patches. The archive
includes the shared timing and semantic-capture helpers so reproduction does not
depend on uncommitted files. Only the residency entry point is part of this study;
other modes in the shared helpers were not qualified here.

`manifest.json` records each archived file's SHA-256 and size. Its contents also
appear inside `frozen-harness.zip`; verify all members before using them.
The public residency candidate, entry point and test are byte-identical to their
archived copies. The frozen timing helper omits the obsolete `liger_components`
config key to match the tested upstream schema; its capture/comparator is unchanged.

## Reproduce on an already available Linux CUDA host

No command here rents hardware. The original layout was `/root/liger/repo` for the
separate upstream checkout, `/root/liger/model` for the pinned snapshot, and
`/root/liger/results` for outputs. The archived preparation/affinity helpers and
configs retain those paths. Use a fresh directory with that layout or deliberately
adapt the paths in all inputs. Do not overwrite an existing experiment.

1. Extract the archive into a fresh staging directory. Copy `overlay/` into the
   separate upstream checkout and `inputs/` into `/root/liger/`.
2. Use Python 3.12 and the recorded `inputs/environment.json` package versions.
   Install the upstream checkout's `[train]` extra with `constraints.txt`, plus
   pytest, pytest-cov, sentencepiece, protobuf and einops. Torch is
   `2.8.0+cu128`, available from the PyTorch cu128 wheel index. Run `pip check`.
3. Set the original environment before importing Torch:

   ```bash
   export PYTHONPATH=/root/liger:/root/liger/repo:/root/liger/repo/src
   export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0
   export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_DISABLED=true
   mkdir -p /root/liger/results
   cd /root/liger/repo
   python /root/liger/prepare_7b.py
   ```

   Preparation downloads the pinned model and converts its original `.bin` shards
   with `weights_only=True`, preserving names and dtypes. Compare the generated
   model/data hashes against the archived manifests. The saved `b1.yaml`/`b2.yaml`
   retain the exact configuration. The frozen dataset contains synthetic rows,
   not a downstream task-quality evaluation.
4. Run the candidate tests first. On the original host Torch CUDA initialization
   preceded pytest collection, as preserved in `pytest_gpu_entry.py`:

   ```bash
   python /root/liger/pytest_gpu_entry.py tests/test_streaming_residency_candidate.py --no-cov -q
   ```

5. Invoke each capture in a fresh process, with GPU-local CPU affinity selected
   before Torch imports. `pin_entry.py` selects four distinct physical cores and
   records that affinity. Example for batch 1:

   ```bash
   python /root/liger/pin_entry.py /root/liger/entry.py --config /root/liger/b1.yaml --output /root/liger/results/check-baseline --capture
   python /root/liger/pin_entry.py /root/liger/entry.py --config /root/liger/b1.yaml --indices 12,13,14,15,16,17,18,19 --output /root/liger/results/check-resident8 --capture
   ```

   The default allocator cap is 10 GiB. Compare the captured directories using
   `benchmarks.harness.liger_streaming_parity.compare(reference, candidate)`;
   preserve its original tolerances. Also check every tensor with `torch.equal`
   to reproduce the stronger observed bitwise result. Repeat with `b2.yaml`,
   resident16 indices 8 through 23, and a repeated native batch-1 control.
   Repeat the semantic checks using `--semantics` instead of `--capture`.
6. Only after qualification, replace `--capture` with `--warmup 6 --steps 8`.
   Follow the archived protocol: four fresh-process pairs per arm and batch,
   AB/BA/BA/AB, checking final saved adapters and loss histories after all 14
   updates. Retain allocated/reserved peaks, host memory, environment and peer
   telemetry. Calculate the median of paired time reductions and the specified
   bootstrap interval; never pool a different host or an earlier lease.

The larger local raw archives remain under
`output/residency-vast-20261020/bundles/`; their export receipts and checksums are
recorded in the public results. They are not embedded in Git. The report's local
replay commands require that retained directory. This small archive supports a
fresh experiment and documents the exact original sources; it does not replace
the retained tensor evidence.

This remains a dense-BF16 research installer. It provides neither automatic VRAM
selection nor a production configuration option. The fixed-microbatch results
do not establish superiority over using the same memory for a larger microbatch.
