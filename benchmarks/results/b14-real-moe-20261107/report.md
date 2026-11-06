# B1.4 ticket 08: per_expert correctness on a real small MoE

## Result

On the published checkpoint `ByteDance-Seed/academic-ds-9B` (revision
`2a205727f616995ad4aba57806f8e2d0d2f28743`), the per_expert streamed model equals a resident
native eager model of the same checkpoint with the same LoRA adapters. One forward and backward
pass in bfloat16 on CPU, on two independent machines:

| | Kaggle | vast.ai |
| --- | --- | --- |
| CPU | Intel Xeon 2.20 GHz, 4 cores, 31.3 GiB RAM | AMD Threadripper PRO 3945WX, 251 GiB RAM host |
| torch / transformers / peft | 2.11.0+cpu / 5.20.2 / 0.21.2 | 2.7.1+cu126 (CPU used) / 5.20.2 / 0.21.2 |
| harness source | `2509cba2` (dataset, same code) | `0e2f4100` (adds env paths only) |
| verdict | PASS | PASS |
| logits, loss | bitwise equal | bitwise equal |
| LoRA gradients compared / non-zero / mismatching | 160 / 160 / 0 | 160 / 160 / 0 |
| negative control (one resident expert doubled) | detected | detected |
| harness run time | 1389 s | 599 s |
| peak process RSS (`ru_maxrss`) | 28.2 GB | 37.9 GB |

Both machines also passed the same harness on the tiny synthetic checkpoint before the real run
(Kaggle probe, `kaggle-probe/`): float32 and bfloat16, bitwise, control detected.

## Definition used (ticket 08 checkbox 1)

Fixed in `benchmarks/harness/b14_real_moe_correctness.py` (`check_definition`) before any real run,
commit `5ff2dd4a`. The six points and their result for this checkpoint:

1. `model_type == "deepseek_v3"` and `routed_scaling_factor` is a float: met.
2. (Loads without `trust_remote_code` and without editing `config.json`): met, the run loaded it
   with `trust_remote_code=False`.
3. Routed expert keys `model.layers.<n>.mlp.experts.<id>.{gate,up,down}_proj.weight`, ids
   consecutive from 0 in every MoE layer: met (15 MoE layers, 64 experts).
4. Sum of `.safetensors` at most 20 GB: 18 733 381 272 bytes, met.
5. At least two MoE layers, more routed experts than `num_experts_per_tok`, at least one shared
   expert: met.
6. Trained weights: std of expert weights differs from `initializer_range` (0.02) by at least 5%.
   Measured std of three layer-1 experts: 0.0231 (15.6% off). Router std: 0.0571. Met.

The 20 GB and 5% limits are my choices, not from the plan or the card.

## Pass rule (fixed before the run)

B1.3 contract: float32 bitwise; bfloat16 `atol=1e-6, rtol=1e-5`. Both machines met the stricter
bitwise equality in bfloat16.

## What was run

- Checkpoint sharded with `shard_checkpoint(..., dtype="bfloat16", expert_layout="per_expert")`,
  streamed with `build_streamed_model` (one bounded expert slot per MoE layer), device CPU.
- LoRA on `q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj`, `r=2`, B factors set to
  non-zero values, copied to the resident model.
- One batch of two real sentences, 15 tokens each, one row's first label masked. Text in
  `kaggle/result-run.json` under `batch`.
- The streamed pass ran first, its results were kept in host memory, its cache and model were
  released, then the resident model loaded. Resident reference: `AutoModelForCausalLM` with
  `experts_implementation="eager"`.
- The router chose 44 to 62 of 64 experts per MoE layer (`experts_hit_by_layer` in each JSON).

## Limits

- CPU and bfloat16 only. No CUDA, no GPU memory result, no float32 (the float32 resident model
  needs about 35 GiB), no NF4. GPU capacity is ticket 09.
- One forward and backward pass, one batch. Not a training run, no convergence or speed claim.
- LoRA sits on the MLA projections only. Adapters on routed experts are refused by the runtime.
- `e_score_correction_bias` is zero in this checkpoint, so the router bias path is covered by the
  synthetic tests, not by these weights.
- The hit sets differ between the machines by one expert in 5 of 15 layers (`kaggle` against
  `vast` JSON). Different CPUs give different bfloat16 rounding, which moves a near-tie in the
  router. Equality is shown within each machine, not across machines. The JSON records
  `loss_equal` but not the loss values, so the two machines' losses were not compared.
- Peak RSS on Kaggle, 28.2 GB, is close to its 31.3 GiB: the resident model (17.5 GiB) dominates.
  The peak after the streamed pass alone was 10.6 GB (Kaggle).
- The vast.ai host had 251 GiB RAM and no memory limit set, so its 37.9 GB is not a fit result.

## Environment notes

- Kaggle: the preinstalled `torchao 0.10` breaks `peft 0.21` and is uninstalled by the entry
  script. A first run of the probe failed on it (not kept).
- vast.ai: the first run failed in the download with `TypeError: process() takes no keyword
  arguments` (`httpx2` against the image's Brotli 1.1.0). `pip install -U 'Brotli>=1.2'` fixed it
  (`vast/pip2.log`); the weights were already on disk and the rerun is the one reported.
- vast.ai rental: instance 59462997, about 17 minutes at 0.097 USD/h, destroyed after the result
  was copied. Ceiling was 0.3 USD.

## Files

- `kaggle/result-run.json`, `kaggle/soup-b14-run.log`, `kaggle/run.py`, `kaggle/kernel-metadata.json`
- `vast/result-run.json`, `vast/run.log`, `vast/pip2.log`
- `kaggle-probe/result-probe.json`, `kaggle-probe/soup-b14-probe.log`

## Reproduce

```text
python benchmarks/harness/b14_real_moe_kaggle.py     # B14_STAGE=run B14_SRC=<repo> B14_OUT=<dir>
python -m pytest tests/test_b14_real_moe_correctness.py --no-cov -v   # synthetic harness tests
```
