# Probe G2 - store each layer's inner activations instead of recomputing them

**Status: GPU verdict `DO NOT BUILD: recompute is as fast or faster on both tiers`** (session
2, rule v2, 2026-11-01; §9, `results/g2-gpu-20261031/`). The decision rule in §3 was written
before any measured arm; its V7 row was revised once, after session 1 and before session 2
(§9). The store is bit-exact (G1, G2 pass) but slower than recompute in pinned RAM (G3) and on
a virtual spill disk (G4). The 1.5x figure below is a FLOP ratio, never a measured speed-up.

## 1. The question

The L2L step (A0, `l2l_schedule.L2LStep`) keeps one boundary activation per (layer,
micro-batch) and, in the backward, re-runs every layer's forward under grad to rebuild what
autograd saved inside it. With LoRA on a frozen base that recompute is one forward of the
step's three (forward, recompute, backward): roughly 6N FLOP per token with recompute and
4N without, so a **potential compute ceiling of 1.5x**. G2 asks whether storing the inner
saved tensors (attention and MLP intermediates) in host RAM, or on NVMe past it, is
cheaper than recomputing them:

1. inner saved bytes per token per layer on the 7B and 70B shapes (§2);
2. sustained write and read bandwidth of the spill drive, alone and while the weight
   stream reads the same drive (`g2_disk.py`, §6);
3. first, weights in RAM: store vs recompute, where the store competes with nothing;
4. then, weights streamed from disk with the records spilled to the same drive;
5. the break-even: the saved recompute must exceed write + read + contention + the
   store's own work (§5);
6. losses and gradients bit-equal to the recompute step and to GA (the math is unchanged).

## 2. Why identity hooks are wrong here, and what the store does instead

The streamed runtime keeps `stream_buffers` weight slots and refills a slot with another
layer while earlier layers' graphs are still alive. Scratch experiment `exp2` (3 layers, 2
slots, CPU) kept each (layer, micro-batch) graph with no recompute:

| saved-tensor handling | result |
|---|---|
| none | autograd raises: a saved weight was modified in place (version 9, expected 8) |
| identity `saved_tensors_hooks` | runs; **8 of 24 LoRA gradients silently wrong**, max abs 0.12 |
| copy every saved tensor | exact; copies 887,808 weight bytes beside 494,592 activation bytes |

So suppressing the version error, or hiding it behind a pass-through hook, gives wrong
gradients; copying everything is exact but stores the frozen weights, which is not an
activation cache. `benchmarks/harness/g2_inner_activations.py` classifies every tensor
autograd saves by the storage it lives in:

| class | what it is | stored as |
|---|---|---|
| `weight` | a view of a streamed weight slot | a descriptor (slot, key, geometry, owning layer); resolved against the slot on unpack only if `pool.owner[slot]` is still that layer, else `RuntimeError` |
| `derived` | a dense weight `dequantize_4bit` made from a streamed NF4 slot inside the layer | the recipe; re-dequantised from the reloaded slot on unpack. Paid in the backward, so it is not saved compute |
| `param` | a trainable LoRA tensor | by reference, version-checked on unpack |
| `const` | rotary cos/sin, masks, NF4 code tables | by reference, version-checked |
| `activation` | everything else | the whole storage's bytes, once per (storage, version), restored as the same strided view |

`G2Step` runs each layer's forward once under grad, packs into a per-(layer, micro-batch)
record, writes the record through the unchanged A0 `ActivationStore` (uint8 chunks, sector
padded: the same pinned RAM, unbuffered spill, rings, threads and byte counters), and in the
backward calls `pool.wait(idx)` + `prefetcher.advance(idx)` exactly as the recompute path's
`_body` does before replaying the kept graph. The layer input goes through a
gradient-catching autograd Function (scratch `exp3`), so no boundary tensor stays alive on
the device through an AccumulateGrad node. Independent of the classification, every stored
item's address range is checked against every streamed weight buffer
(`weight_bytes_stored`, must be 0).

### Saved bytes per token per layer (CPU shape estimate)

`g2_probe.py --mode shapes`: one decoder layer of the real shape, frozen weights as
zero-stride expanded scalars (no weight memory), LoRA r = 8 on q/k/v/o, bf16, seq 512, CPU
kernels. Raw: `results/g2-cpu-pilot/shapes_seq512.json`.

| shape | attention | saved B/token/layer | largest items |
|---|---|---:|---|
| Mistral-7B | sdpa | 205,064 | 3 x (512, 14336) bf16 MLP, (512, 4096) fp32 norm inputs |
| Mistral-7B | eager | 307,336 | + (32, 512, 512) fp32 softmax + bf16 probs |
| Llama-70B | sdpa | 405,896 | 3 x (512, 28672) bf16 MLP, (512, 8192) fp32 |
| Llama-70B | eager | 614,536 | + (64, 512, 512) fp32 + bf16 |

These are **estimates for the CUDA kernels**: CPU SDPA saves a different set of tensors
from flash/mem-efficient CUDA SDPA, and the eager rows grow with seq (they hold seq x seq
per head). `correctness` and `profile` record the real record bytes on the GPU box. For
scale only: the boundary activation is 8,192 B/token/layer at 7B bf16, so the inner record
is about 25x the boundary at 7B sdpa (an order of magnitude, as the track context said).
A 32-layer 7B step at seq 512, k = 4 would store about 32 x 4 x 512 x 205 KB ~ 13.4 GB:
that is arithmetic from the estimate, not a measurement.

## 3. The decision rule, written before the run

`RULE` in `benchmarks/harness/g2_rule.py` is this table in machine form and goes into
every session's `protocol.json`. G1-G4 thresholds are unchanged; the V7 normalization
changed from v1 to v2 after session 1 and before session 2, as recorded in §9.

| row | rule | if it fails |
|---|---|---|
| G1 exact | for k in 1, 2, 4 and both stores (pinned RAM records, records through the direct spill): per-micro-batch losses (forward) and every non-zero LoRA gradient (backward) `torch.equal` to GA **and** to the recompute L2L step; GA vs GA reproducible, else the predeclared `--deterministic` rerun (which must itself be reproducible and record `deterministic`); controls: identity hooks without a reload must raise or give different grads, a corrupted record must change grads and leave losses equal; 0 streamed-weight bytes in any record | fail (no verdict when GA is not reproducible or a control cannot fail) |
| G2 bytes | read from **the same record per k that G1 selected** (the default run, or its deterministic rerun; never exactness from one and bytes from the other): every record 0 weight bytes by address, store chunk = record capacity, the store wrote and read exactly `spill_layers x k x capacity` per step, direct backend, no file left behind | fail |
| G3 RAM tier | `ram_store_vs_recompute`: weights in RAM, records in pinned RAM. Per-block ratio B (store) / A (recompute) of median step times, two-sided 95% t interval: **pass if upper < 0.95**, **fail if lower >= 0.95**, else no verdict | as stated |
| G4 disk tier | `disk_spill_vs_recompute`: weights streamed from disk, records spilled to the weights' drive. Same ratio and bounds | as stated |
| V1 | device cuda; a CPU or synthetic session never gets a verdict | no verdict |
| V2 | F1 `suspend`, `power`, `foreign_reads` ok on every arm a gate reads (required); a block with a void arm is replaced (METHOD.md); an unknown removes the gate | no verdict |
| V3 | every spilling store and every sustained I/O arm is direct (unbuffered) | no verdict |
| V4 | A and B of a block train on the same token rows (ids SHA-256) | no verdict |
| V5 | finite losses on every timed step | no verdict |
| V6 | every B arm wrote and read exactly its planned record bytes; A arms none | no verdict |
| V7 | F1 repeats rule: `blocks_to_add == 0`. **Rule v2** (2026-11-01, before session 2): N = ceil(7.85 x (s / 10%)^2) with s = sqrt((CV_A^2 + CV_B^2) / 2), each arm's spread over its own mean; pilot, single top-up and void replacement unchanged (`g2_rule.blocks_to_add`) | no verdict |
| V8 | disk-tier arms: the weight source reads with direct I/O | no verdict |
| V10 | disk-tier arms that spill: the arm's own OS placement records (resolved path, mount, whole physical disk from `/sys/dev/block` or `Get-Partition`) show the records' spill directory and the weight shards on **one physical disk**; an unknown disk is not "same", and two direct-I/O flags are no evidence | no verdict |
| V9 | break-even inputs sustained: `g2_disk` arms `candidate`, fixtures larger than RAM, >= 1800 s each | break-even reported as an estimate |

G3/G4 are not read unless G1 passed: a path that is not exact has no speed. `g2_session.py`
also **stops before the first timed arm** unless G1 and G2 both pass on the correctness
children (`session.json`: `timing_allowed`), so no box time goes to arms the rule would
never read; and it refuses, before any child, a disk comparison whose `--spill-dir` and
`--shards` the OS does not place on one physical disk (`protocol.json`: `placement`).
Verdicts: `BUILD: storing beats recompute on the <tier> tier` (G1, G2 pass and G3 or G4
pass), `DO NOT BUILD` (G1 or G2 fail, or G3 and G4 both fail), else `NO VERDICT`. Memory
and timing below the gates are informational. A pass is a step-time ratio on one card at
one k, not a general speed-up.

## 4. Fixtures, arms and order

- **Model**: `unsloth/mistral-7b-instruct-v0.3` at revision
  `60b99dcbc7f853b7cd18472a607e8d045b96f03e`, NF4 streamed (`--quant nf4`), LoRA r = 8
  q/k/v/o, seq 512, timed arms at k = 4. The measured NF4 path dequantised inside the
  layer; the dense weight was `derived` and re-dequantised in the backward, counted by
  `install_derived_tracking`. Current main can select a fused NF4 path instead. These
  records establish no exactness, byte-count, or timing result for that newer dispatch.
- **Token rows**: seeded random ids (`--input-seed 17`), fingerprinted per arm (V4).
  Losses are not a quality measure here; G2 changes no numerics.
- **Order**: `shapes` -> `correctness_k{1,2,4}` (+ `_det` when GA vs GA is not
  reproducible) -> `profile_ram` -> `profile_disk` -> `ram_store_vs_recompute` blocks ->
  `disk_spill_vs_recompute` blocks. A-B-B-A, fresh process per arm, 5-block pilot then the
  F1 repeats rule, `--max-blocks 20`.
- **Sustained I/O**: `g2_disk.py session` separately on the same box and drives
  (write, read, cycle, weights, cycle+weights; C4 method), exported into
  `disk/rates.json` with `g2_session.py --disk-session`.

## 5. The break-even

`g2_rule.break_even` and `analyse_breakeven` (informational, in `verdict.json`):

$$\text{margin}_{\text{serial}} = t_{\text{recompute}} - \frac{B}{r_w} - \frac{B}{r_r}
 - \Big(\frac{W}{r_{w|spill}} - \frac{W}{r_{w}}\Big) - t_{\text{store}}$$

with `B` the spilled record bytes per step, `r_w`/`r_r` the sustained spill write/read
rates (measured under contention when `cycle+weights` exists), `W` the weight bytes read
per step, `r_w`/`r_{w|spill}` the weight read rate alone and while the spill runs, and
`t_store` the store's own packing/copy seconds: `store_put` + `store_get` of the store
minus those of the recompute step, **minus the seconds `put`/`get` spent waiting on the
spill drive** (`StoreStats.put_wait_s`/`get_wait_s`, reported as
`store_io_wait_s_per_step`). The drive is charged once, by `B / r_w + B / r_r` from the
sustained rates; before the 2026-10-23 review the waits were inside `t_store` too. A
spilling profile without `store_io_wait_s_per_step` gets no margin. The break-even
bandwidth is the symmetric write = read rate at which the serial margin is zero:
`2B / (t_recompute - contention - t_store)`. The overlapped margin charges only the write
time the forward compute cannot hide and the read time the backward cannot hide.
`t_recompute`, `t_store` and the compute times come from `profile` (device synchronised at
every phase boundary, so they are per-phase costs, not the timed step). Without sustained
inputs (V9) the break-even is labelled `estimate`.

## 6. Sustained spill I/O: `g2_disk.py`

The C4 method (`origin/c4-ssd-research` runner): unbuffered I/O from sector-aligned
buffers, fixture certificates (size, SHA-256, mtime; exclusive creation; no links),
per-second windows with per-disk OS counters, the first `--warmup` seconds excluded, every
full window kept (no outlier removal), mean / min / P10 / first and last five windows per
direction, F1 `ArmWatch`, AC power, `G2_SPILL_ENOSPC` on a full drive, spill file removed
on exit and on error. Modes: `write`, `read`, `cycle` (the G2 step pattern: layers
0..L-1 written, then read L-1..0), `weights` (the reference weight stream) and
`cycle+weights` (both at once, measured separately). Two paths are not two disks: the
record names the volume and physical-disk mapping it saw.

## 7. What this does NOT measure

- Quality: none is at stake; G1 is bit-exactness.
- A production trainer path: G2 is harness-only. Integration belongs to A3.
- Attention kernels other than the one the box selects; the CPU shape estimate is not the
  CUDA record size.
- Multi-GPU, other k, other seq. A RAM-tier pass does not imply a disk-tier pass.
- Host RAM fit of the records at 7B k = 4 (~13 GB estimated) is the operator's problem:
  `--store spill` with `--spill-layers` is the way to bound pinned RAM.

## 8. GPU runbook

This runbook describes a new measurement using the published harness, not a full replay
of the historical measured source. See the publication limits at the end of §9.
Target: a Linux CUDA box, 24 GB GPU, at least 48 GB host RAM, one local NVMe that also
holds the NF4 shard cache. Resource figures below are estimates, not measured:

| Resource | Amount | Why |
|---|---|---|
| VRAM | estimate < 12 GB | Mistral-7B NF4 streamed + one layer's records in flight; the records themselves live in host RAM / NVMe |
| Host RAM | >= 48 GB | RAM-tier weights (~4.5 GB NF4 pinned) + records at k = 4: 13.4 GB of payload pinned as **17.2 GB** (a 105.0 MB record is pinned as a 134.2 MB block, #901 power-of-two rounding) + rings (~0.54 GB) + process |
| Disk | ~15 GB source + ~4.5 GB NF4 cache + ~14 GB G2 spill, and for the sustained session two fixtures of 1.25 x RAM plus a 20 GiB reserve: ~140 GiB free on a 48 GB box | `g2_disk.py` refuses a fixture or spill without its reserve |
| Time | G2 session: estimate 4-7 h (correctness 3 x ~5 min, profiles 2 x ~5 min, 2 comparisons x 5-20 blocks x 4 arms x ~1.5 min). Sustained I/O session: 10 arms x (600 s warmup + 1800 s) ~ 6.7 h plus the fixture prefill, run separately | |

```bash
# Start in a checkout containing this publication; record the exact new source commit.
git rev-parse HEAD
A0_PYTHON=python3.12 bash benchmarks/a0-bootstrap.sh   # venv, CUDA torch, pinned deps
.venv/bin/python -m pytest tests/test_g2_inner_activations.py tests/test_g2_rule.py \
  tests/test_g2_disk.py tests/test_l2l_probe_harness.py -q -o addopts=''
export PYTHONPATH="$PWD/src:$PWD/benchmarks/harness" HF_HUB_OFFLINE=0
export SOUP_L2L_POWER_SOURCE=server   # a server has no AC sensor; else every arm is invalid
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("unsloth/mistral-7b-instruct-v0.3",
                  revision="60b99dcbc7f853b7cd18472a607e8d045b96f03e",
                  local_dir=".soup/models/mistral-7b", token=False,
                  allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"])
EOF
export HF_HUB_OFFLINE=1
NVME=/mnt/nvme                     # local NVMe, not overlay/tmpfs; holds .soup/f1-nf4 too
SPILL="$NVME/g2-spill"             # the SAME physical disk as the shards (V10 checks it)
# 1. Sustained spill I/O (§8.1), on a quiet box.
# 2. Plan the G2 session (prints every child command, writes nothing):
.venv/bin/python benchmarks/harness/g2_session.py --weights .soup/models/mistral-7b \
  --shards "$NVME/f1-nf4" --spill-dir "$SPILL" --out .soup/results/g2-01 --k 4
# 3. Run it, with the sustained I/O session's directory for the break-even:
.venv/bin/python benchmarks/harness/g2_session.py --weights .soup/models/mistral-7b \
  --shards "$NVME/f1-nf4" --spill-dir "$SPILL" --out .soup/results/g2-01 --k 4 \
  --disk-session .soup/results/g2-disk-01 --execute
# 4. Recalculate the current-rule verdict from selected JSON records, not the hardware:
.venv/bin/python benchmarks/harness/g2_rule.py .soup/results/g2-01 --out verdict.json
```

The first `correctness` child shards the checkpoint into `--shards` if absent. Copy the
whole session directory and the disk session to `benchmarks/results/g2-<box>/`.

### 8.1 Sustained I/O session

Fixtures larger than RAM, 600 s warmup and 1800 s measured per arm, each mode twice in
A-B-B-A order, a fresh process per arm (the C4 method):

```bash
RAM=$(awk '/^MemTotal:/{print $2*1024}' /proc/meminfo)
UNIT=$((32 * 8 * 1024 * 1024))                         # 32 layers x 8 MiB chunk
SIZE=$(( (RAM * 5 / 4 + UNIT - 1) / UNIT * UNIT ))     # 1.25 x RAM, whole layers
df -B1 --output=avail "$NVME"                          # need >= 2 * SIZE + 20 GiB
mkdir -p "$NVME/g2-fixture" "$SPILL"
.venv/bin/python benchmarks/harness/g2_disk.py prepare-weights \
  --path "$NVME/g2-fixture/weights.bin" --bytes "$SIZE" --reserve-bytes 20GiB
.venv/bin/python benchmarks/harness/g2_disk.py session \
  --modes write,read,cycle,weights,cycle+weights --out .soup/results/g2-disk-01 \
  --spill-dir "$SPILL" --spill-bytes "$SIZE" --chunk-bytes 8MiB --layers 32 \
  --weights "$NVME/g2-fixture/weights.bin" --weight-chunk-bytes 8MiB \
  --warmup 600 --duration 1800 --window 1 --verify sample --verify-every 16 \
  --reserve-bytes 20GiB --stop-temperature 70 --arm-timeout 4500 --label g2-disk-gpu
```

`session` writes `manifest.json` and `rates.json`; `g2_session.py --disk-session` exports
them into `disk/rates.json` with `g2_disk.export_session`. Exported rates are busy means
(bytes over seconds inside that direction's I/O calls, full windows after warmup); the
spill rates come from the `cycle+weights` arms when present (spill under contention),
invalid or killed arms are left out of the means and listed with their problem. An
optional second session with `--modes weights,cycle+weights --weight-rate <bytes/s>`
measures contention at the training step's own weight rate instead of the unpaced worst
case. V9 accepts only `candidate` arms with fixtures larger than RAM and >= 1800 s.

## 9. Results

### SYNTHETIC / CPU PILOT (no verdict)

Only `results/g2-cpu-pilot/shapes_seq512.json` is selected for this publication. It
supports §2's CPU shape estimates and cannot establish G3/G4. The original CPU sessions,
smokes, and their logs remain local; their historical results are not replayable from
this PR's selected CPU evidence.

Second review, 2026-10-23 (audit reproductions; no threshold changed, V10 added before any
GPU arm): G1 and G2 read one selected record per k (a deterministic rerun with
`spill_read = 0` had passed G1 on itself and G2 on the default run's bytes); disk arms
record and V10 checks the OS placement of spill and shards; `profile` takes the spill I/O
waits out of `t_store`; the session refuses a disk comparison not placed on one disk and
stops before the timed arms when G1/G2 do not pass. Each has a regression in
`tests/test_g2_rule.py` that targets the previous behavior. The original CPU review
session remains in the private local research archive and is not bundled here.

### Before the GPU session (2026-11-01, committed before the rental)

Rule thresholds, gates G1-G4 and V1-V11 are unchanged. Three changes, each from a local CUDA
run of the whole session on an RTX 4050 Laptop GPU (WSL2, torch 2.14.1+cu130, bitsandbytes
0.50.2, SmolLM2-135M NF4, seq 256, k = 4, 2 blocks per comparison; a smoke, no verdict):

- **A NaN gradient no longer loses the correctness record.** On CUDA NF4 the corrupted-record
  control gives NaN gradients; `l2l_schedule.compare` put NaN into `max_abs` and the strict
  JSON writer (`allow_nan=False`) raised, so `correctness_k*.json` was never written and G1
  would have been NO VERDICT on a passing store. A non-finite difference is now `unequal`, named
  in `nonfinite`, and kept out of `max_abs`. The control still fails the comparison as G1
  requires (`tests/test_l2l_probe_harness.py::test_a_nan_gradient_...`).
- **V12 label (no gate depends on it).** `g2_disk` records a disk's vendor and model and labels
  a hypervisor's disk `virtual` (virtio vendor 0x1af4, which a vast.ai VM reports with no model
  string; QEMU/Hyper-V model strings; `vd*`/`xvd*`). G4's detail and the break-even carry
  `disk_scope`; on a virtual disk they describe that disk at that time, not a physical NVMe.
- The host-RAM estimate above is 17.2 GB pinned records, not about 13 GB.

What the smoke showed on CUDA: G1 PASS at k = 1, 2, 4 (both stores bit-equal to GA and to the
recompute step; the identity-hook control **raised** "found a recycled weight slot" on NF4; the
corrupted record changed the gradients and not the losses), G2 PASS (0 weight bytes, spill
written = read = plan, direct, no file left). Every timed arm was **void**: the laptop ran on
battery (`AC1 online=0`), and F1's power check voids an arm off mains, as designed. A4's
vast.ai VM arms read "no battery ... so on mains". The CUDA laptop smoke records and
operational run plan are omitted from this publication. No paced contention result is
claimed. The sustained I/O results below are the selected machine-1 records.

### GPU session 1 (rule v1), RTX 4090 VM, 2026-10-31

Instance 58578846 (vast.ai offer 57922513, RTX 4090 VM, gpu_frac 1.0, PCIe 4.0 x16, 47.4 GiB
guest RAM, ext4 on virtio `vda`), code `40aea9b9`, session 21:22:07Z-22:21:23Z, exit 0, 109
children. Raw records stay outside Git, archive SHA-256 `6c845a0d…`. The historical
`g2_rule.evaluate` at the pinned rule-v1 source produced the recorded verdict; the current
rule-v2 publication evaluator rejects this older protocol rather than reinterpreting it.

**`NO VERDICT`**: G1 **pass** (on the predeclared `--deterministic` reruns: the default runs'
GA vs GA differed in 250 of 250 LoRA gradients on this box), G2 **pass** (0 weight bytes;
record 104,996,864 B, 205,064 B/token/layer on CUDA SDPA; k = 4 spill wrote = read = plan
13,439,598,592 B, direct), G3 **fail** (ratio 1.860, CI [1.858, 1.862], 5 blocks), G4 **no
verdict**: 20 blocks (the cap), every ratio >= 4.27, mean 5.99, CI [5.29, 6.69], but V7 asked
for 2,021 more blocks.

Why V7 exceeded the session's cap: F1's v1 rule pools the A and B spreads in seconds and
compares them with 10% of A's mean. The disk B arm runs about 5x A, 5 s against 1.17 s,
and alternated between about 5 s and 10 s. Its relative noise of about 25% therefore
counted about 25 times over. The required 2,041 blocks were finite, but above the cap of
20. With each arm's spread taken over its own mean, rule v2 asks for 64 total blocks on
these same records, 44 more than completed. This is a retrospective comparison only.
**Rule v2 was designed after seeing session 1's data** (the owner asked for a pass or fail on
every gate, 2026-11-01). It changes only how V7 normalises the spread; the G3/G4 bounds (95%
CI against 0.95), the pilot, the single top-up and every other row are unchanged. Rule v1's
verdict of session 1 stands as recorded; v2 is committed before session 2 and judges only it.
Session 2 runs on a new VM with `--max-blocks 150`; v1's runbook capped at 20, below what
either rule asked for on session 1. Its time limit was set before the run.

### GPU session 2 (rule v2), RTX 3090 VM, 2026-10-31 to 2026-11-01

Instance 58588456 (offer 58411255, RTX 3090 VM, gpu_frac 0.25, PCIe 4.0 x16, 49.3 GiB guest
RAM, ext4 on virtio `vda`), code `5027d2ad`, `--max-blocks 150`, session 23:20:42Z-02:09:34Z,
exit 0, 49 children (raw archive SHA-256 `27b95bef…`).

**`DO NOT BUILD: recompute is as fast or faster on both tiers`**: G1 **pass** (deterministic
reruns again), G2 **pass** (same record and spill bytes as session 1), G3 **fail** (ratio
1.244, CI [1.243, 1.245], 5 blocks), G4 **fail** (ratio 1.705, CI [1.637, 1.772], 5 blocks;
v2 pilot N = 2). Profile: recompute 0.81 s/step against 1.10 s of store packing and copies on
the RAM tier, 0.85 s against 1.14 s on the disk tier.

### Break-even, measured (machine 1)

The sustained I/O session ran on machine 1 after session 1, same disk (10 arms `candidate`,
600 s + 1800 s each, fixtures 63.6 GB > RAM). Spill under contention: write 2.155, read 3.110
GB/s; weights 5.767 GB/s alone, 1.798 GB/s beside the spill. With session 1's profile (V9 and
V11 hold, `measured`): RAM tier serial margin -0.60 s/step; disk tier -13.75 s/step serial,
-12.71 s overlapped. Holding those measured profile costs fixed, drive bandwidth alone
cannot break even because the store overhead already exceeds the saved recompute. No
claim follows about other hardware or implementations. Evidence, cost, and limits are in
`results/g2-gpu-20261031/README.md`.

### Publication selection and provenance

This benchmark-only PR transfers selected G2 files from research snapshot `b315e8dc`
onto current main. It does not add a product feature. The recorded recommendation remains
`DO NOT BUILD` on the measured fixture. Numerical evidence JSON, historical verdicts,
thresholds, `provenance.json`, and `SHA256SUMS` remain unchanged.

Published evidence includes both GPU sessions' JSON, machine 1's sustained-I/O JSON and
historical break-even export, and the single CPU shape estimate used in §2. Rental guard
and watcher code, rental/API artifacts, logs, per-second samples, operational pilots and
the rest of the CPU sessions stay in local private research archives. No omitted path
is required by the selected-record recalculation. Archive hashes and original path
substitutions remain in `results/g2-gpu-20261031/provenance.json`; they do not make the
private raw files available for independent audit.

The measured source commits were `40aea9b92b6611735a05bec2862fcca0fa31f497` for session 1
and sustained I/O, and `5027d2ada3392593d064bbebf4e442da68c4b1c7` for session 2. Those
objects exist in local private research history, but this PR neither publishes their
ancestry nor establishes that a remote can supply them. Recorded source hashes identify
files; they do not supply the original source. The current-main harness includes
compatibility adaptations and must not be described as the measured runtime.

The current rule-v2 CLI recalculates session 2 from selected records without CUDA:

```bash
python benchmarks/harness/g2_rule.py benchmarks/results/g2-gpu-20261031/session2-rtx3090
```

This is record recalculation, not source, kernel, sensor, or full I/O-window replay. The
publication evaluator rejects a missing or different protocol rule. To reproduce both
sessions' recorded verdicts under the rule each was run with (session 1 v1, session 2 v2),
run `python benchmarks/harness/g2_replay.py`: it rebuilds each measured source tree from
public commit `fdf067fc` plus the frozen harness files in
`benchmarks/results/g2-gpu-20261031/frozen-source/`, checks the protocol's source hashes,
and recomputes the verdicts and the ten sustained-I/O summaries (from the sanitized
per-second samples in `raw-records/`). It does not rerun kernels or F1 sensors.
