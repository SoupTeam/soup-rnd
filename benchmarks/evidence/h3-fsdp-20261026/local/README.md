# Local evidence only — no distributed CUDA validation

2026-10-26, Windows, Python 3.10.0, torch 2.13.0+cpu, transformers 5.17.0,
accelerate 1.14.0, PEFT 0.20.0, TRL 0.29.1. Full versions, source hashes and
dirty status at fixture time: `environment-rank-0.json` and `pip-freeze.txt`.
The owning checkout's `src` was explicitly placed first on PYTHONPATH to avoid
the original D2 editable installation supplying the trainer.

The fixture is a **randomly initialized offline tiny Qwen2**, not the real
pretrained Qwen2.5 pilot model: 15,552 total parameters, 1,024 trainable parameters,
one layer, hidden size 32, vocabulary 64, FP32, sequence length 64, global batch 2,
accumulation 2, 5 optimizer steps (1 inspection + 1 warmup + 2 timing + 1 inspection).
CPU timing/RAM fields are diagnostic and never enter the scaling table.

- The first update changed only the two LoRA B tensors; the two A tensors staying
  unchanged at initialization is expected. All four changed by the end.
- First and final inspected gradients were finite, some were nonzero, and frozen
  parameters had no gradients. Frozen-base hashes before/after were identical:
  `3269db231954a385667d330f853bd55119d5835a16ea313a383de3aacece820d`.
- Final adapter: 4 tensors. Training PID 10452; root reload PID 3720;
  checkpoint reload PID 29196. Exact key/value checks passed for both saved
  adapters. Fixed eval logits matched with max absolute error **0.0** for both
  reloads (`atol=rtol=1e-5`). This proves the CPU fixture path, not BF16/FSDP.
- The periodic checkpoint contained optimizer/scheduler files; no attempt was
  made to restore them or continue training. Resume remains unmeasured.
- Initial development failures are retained: `cpu-train-1.log` rejected a
  prepared-data path outside cwd under the existing schema containment rule;
  `cpu-train-2.log` found the harness reading a missing `fsdp_plugin` on the
  single-process control. Both were corrected in the harness, with no product
  code change. They are excluded from every result.

Reproduce from a directory that contains both prepared data and run output
(the Soup schema requires tokenized_path to stay under cwd). Use absolute script
and PYTHONPATH paths if that directory is outside the repository:

```bash
export PYTHONPATH=/path/to/soup-rnd/src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
python /path/to/soup-rnd/benchmarks/harness/h3_fsdp_run.py prepare \
  --prepared cpu-fixture --cpu-fixture --rows 32 --sequence-length 64
python /path/to/soup-rnd/benchmarks/harness/h3_fsdp_run.py train \
  --prepared cpu-fixture --output cpu-run --device cpu --global-batch 2 \
  --correctness-steps 1 --warmup-steps 1 --measured-steps 2
python /path/to/soup-rnd/benchmarks/harness/h3_fsdp_run.py reload \
  --prepared cpu-fixture --output cpu-run --device cpu
python /path/to/soup-rnd/benchmarks/harness/h3_fsdp_run.py reload \
  --prepared cpu-fixture --output cpu-run --device cpu --checkpoint
```

Raw training/reload logs, rank record, snapshots, final adapter and saved numeric
reference are retained here. The script generates the tiny base/data reproducibly;
they are not the immutable pretrained base intended for the GPU pilot.

Local pytest selection: the two H3 files plus existing #373, #352, #350,
distributed device-map, #1204 forwarding, and multi-GPU tests. Raw log:
`targeted-tests-final.log`. Full repository pytest was not run. `ruff-all.log`
covers `src/soup_cli/ scripts/ tests/ benchmarks/`. CPU tests validate accounting,
report rejection and process-group timeout cleanup with doubles; they are not
proof of real GPU communication or sharding.

`preservation.json` verifies the original D2 snapshot's 129 files still matched
the user's checkout byte-for-byte at audit end; ZIP CRC check passed. Original
source HEAD remained `b42a5722b75806ab1306ed3074c404e159941fe8`.
