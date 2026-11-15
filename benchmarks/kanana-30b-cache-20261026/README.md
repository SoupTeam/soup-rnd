# Kanana 30B: four-arm expert-streaming qualification

This is preparation for a new checkpoint experiment. No 30B GPU accuracy,
capacity, speed or downstream-quality result is established here.
The completed [16B experiment](https://github.com/SoupTeam/soup-rnd/blob/4bfccb15a3eb646acf0118b353386b2773c83138/benchmarks/expert-cache-policy-20261025/GPU-RESULTS.md)
remains scoped to its own frozen source, model and cache ceilings.

## Checkpoint and structural preflight

Official model: [Kanana-2-30B-A3B-base-2601](https://huggingface.co/kakaocorp/kanana-2-30b-a3b-base-2601),
revision `0d31365acf656e5f9a593cacc4913e9c0d36cb1c`. Public, ungated,
with a custom [upstream license](https://huggingface.co/kakaocorp/kanana-2-30b-a3b-base-2601/blob/0d31365acf656e5f9a593cacc4913e9c0d36cb1c/LICENSE).
No account terms were accepted. The 13 BF16 safetensors files total
61,343,935,088 bytes including headers. Stored tensor elements number
30,670,815,104; 6,016 are router correction-bias buffers, leaving
30,670,809,088 parameter elements.

The native `DeepseekV3ForCausalLM` configuration has 48 layers, one initial
dense layer, 128 routed experts per MoE layer, top-6 routing and two shared
experts. Full query projection, group-one sigmoid/noaux routing and interleaved
RoPE require new controls; this is not a Qwen architecture rename or port.

Anonymous bounded HTTP ranges verified all source headers against the index,
with zero tensor payload downloaded. All 144 attention LoRA projections and
47 expert blocks match the offline native NF4 meta skeleton. These are
structural checks; complete LFS hashes must be verified after the real download.
The byte-verified metadata, headers and preflight programs are archived in
`model-metadata-evidence.zip`, indexed by `metadata-manifest.json`. The original
model card and license remain available at the upstream links.

Local scoped validation: **179 passed, five CUDA skips**, including real CPU
NF4 and BF16 four-arm tiny-Kanana optimizer trajectories with bitwise agreement.
Ruff and applicable staged pre-commit passed. `cpu-controls.zip` retains the
original XML; its byte hashes and scope are in `cpu-controls-manifest.json`.
This is not a full-suite, hosted-CI or actual CUDA result.

## Fixed comparison

All arms use one checkpoint/store/input fixture and the same FP32 attention
LoRA adapters (`q_proj`, `kv_b_proj`, `o_proj`), rank 4 / alpha 8, dropout zero.
Base storage is double-quantized NF4 and compute is BF16. Batch 1, context 512,
gradient accumulation 2; decoder recompute remains enabled.

| Arm | Two-read | Packed expert RAM ceiling | Dense expert GPU ceiling |
| --- | --- | ---: | ---: |
| ordinary | off | 0 | 0 |
| two_read | on | 0 | 0 |
| cache | off | 8 GiB | 2 GiB |
| two_read_cache | on | 8 GiB | 2 GiB |

Both cached arms use `host_cache_policy="static"`, with identical admission
and cold selection rules. Decoder RAM hotsets, resident decoder layers and
retained activations remain disabled. There is no compile/kernel change.
Existing full-ceiling RAM/VRAM admission and headroom are preserved; an admission
failure is retained explicitly, without a silent smaller-budget fallback.

There are 6,016 routed experts. Each owns 2,434,184 packed NF4 bytes or
9,437,184 dense BF16 bytes. Payload arithmetic permits at most 227 experts in
2 GiB GPU and 3,528 experts in 8 GiB RAM. It does not prove whole-model fit.

## Required gates and evidence

1. Before the large download: actual CUDA NF4/BF16 tiny-Kanana four-arm
   trajectories, plus the existing LRU/static CUDA NF4 controls, without skips.
2. Before timing: four real-model captures, each two AdamW updates with two
   microbatches per update. Compare every candidate separately with ordinary
   using the unchanged reference-relative `rtol=.01`, `atol=1e-5`, finite and
   three-decimal loss gates. Report bitwise agreement separately.
3. Four balanced Latin-order rounds, 16 fresh timing processes, two warmup and
   six measured updates per process. Preserve all raw timing arrays and negative
   results. The practical gate is the unchanged paired-bootstrap lower bound
   of at least 2% step-time reduction.
4. Separate step time, setup, complete eight-update entry, allocated/reserved
   VRAM, stage RSS/lifetime RSS and payload. Physical 8 GB GPU, native RAM bound
   at most 32 GiB and no workload swap must be verified, not simulated.
5. Freeze the public source and private run protocol before leasing. Retain all
   originals losslessly through the tested authenticated range transport,
   independently verify them locally, then delete the exact owned instance and
   persistent volume and verify absence. Preserve shared-account peers.

Budget starts from the previously authorized remaining **$5.169487974758529**,
with earlier costs already deducted. The new-stage ledger starts at zero.
This is a numerical/throughput experiment with frozen random token inputs,
not a downstream-quality, convergence or full-fine-tuning result.
