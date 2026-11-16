# Physical 8 GiB residency + compilation: planned reproduction

**Status: blocked before model work.** This archive contains intended sources,
CPU checks, hardware/transport diagnostics and deletion proofs. It contains no
pretrained-model accuracy, training-speed, training-memory or capacity result.
See the [report](../../streaming-residency-compile-8gb-20261021.md) and
[blocked record](../../streaming-residency-compile-8gb-20261021-results.json).

`planned-harness.zip` and the adjacent `manifest.json` preserve exact per-file
sizes and SHA-256 hashes. The ZIP includes the same manifest. No provider
controller, credentials, private billing/list responses or model weights are included.
The intended upstream source was not GPU-executed during these attempts.

## Run on an already available Linux CUDA host

Use a separate clean upstream checkout at
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`, then copy `overlay/` over that checkout.
All 602 production files remain unchanged. The PR branch's older production code
is not this intended baseline. Preserve the supplied capture/comparator: its only
capture edit generalizes the 448-adapter assertion to 14 times decoder-layer count.

The planned layout is `/root/liger/repo` for that checkout and `/root/liger` for
the contents of `inputs/`. Install Python 3.12 and the recorded package versions
from `constraints.txt`, including Torch 2.8.0+cu128, Transformers 5.17.0,
PEFT 0.20.0 and TRL 0.29.1; run `pip check`. Confirm a physical 8 GiB GPU,
idle/exclusive GPU use, GPU-local CPU affinity, sufficient actual container RAM,
disk, and reliable hash-verified result retention before downloading models.

Choose either planned protocol and copy it to `/root/liger/protocol.json`.
Before importing Torch:

```bash
export PYTHONPATH=/root/liger:/root/liger/repo:/root/liger/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_DISABLED=true
export SOUP_LAYER_STREAM_CACHE_DIR=/root/liger/shards
export TORCHINDUCTOR_CACHE_DIR=/root/liger/compiler-cache
mkdir -p /root/liger/results
cd /root/liger/repo
python /root/liger/prepare_model.py 8
python /root/liger/pin_entry.py /root/liger/entry.py --config /root/liger/model-8.yaml --arm baseline --output /root/liger/results/check-baseline --capture
```

Preparation downloads the pinned public snapshot and writes the finite synthetic
arithmetic dataset/config. It does not rent hardware. Config preparation and the
large-model GPU paths were not reached in the blocked attempts.

Use baseline `residency-compile-provenance.json` peaks and
`select_resident_layers()` to choose a fixed middle block, reserving 0.75 GiB
inside the 7 GiB allocator limit. The allocator limit excludes driver context.
This is research selection, not a shipped automatic VRAM planner.

Run fresh processes with identical selected `--indices` for `--arm resident`
and `--arm resident_compile`, using `--capture` first. Preserve the original
comparator (`rtol=0.01`, `atol=1e-5`, exact inputs/init, finite values and loss
to three decimals), then run `--semantics` for the causal/padding gates.
The separately labeled `resident_compile_precise` retains operation/cast boundaries;
it changes compiler options, never numerical tolerances.

Only qualified arms proceed to `--warmup 6 --steps 8`. Use four fresh-process
AB/BA/BA/AB pairs. Compare resident against baseline, then compiled resident
against eager resident. Check every final adapter tensor and all 14-update losses,
forbid measured-window recompilation, and retain CPU/GPU memory and peer telemetry.
Do not pool models, hardware or the previous 7B experiment.

`worker.py` records the planned orchestration; its export ACK handshake requires
the private retention controller and is not a standalone launcher. The commands
above allow manual execution without that controller. `setup.sh` is preserved as
a planned setup record. No command in this package provisions a GPU instance.

## Retained diagnostics

`diagnostics/` includes sanitized physical GPU/RAM checks, the direct/HTTPS
throughput failures, the SSH timeout classification, individual deletion proofs
and final instance absence. An exact SSH throughput is intentionally absent:
the transfer did not complete. `validation/` includes the 14-pass/2-CUDA-skip CPU
record and the injected-DNS-failure test with TLS verification retained.

Unredacted provider/connection responses remain only in local ignored study output.
There are no new training tensors: no capture or GPU test started. Resume only
after a result-retention path passes qualification; repeat the numerical gates
before drawing any performance or model-capacity conclusion.
