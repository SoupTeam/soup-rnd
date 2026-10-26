# A4 v3 GPU session: one RTX 3090 VM, 2026-10-26 (protocol v3, rule v2)

**`SPILL READY FOR A3: automatic fallback past the RAM budget; on the disk tier spill to a
drive the weights are not read from`** by rule v2 (`verdict.json`, unchanged): G1 pass, G2
pass, G3 pass, G4 pass, **G5 fail**. Re-running `a4_rule.py` on the session's raw records
reproduces `verdict.json` byte for byte; on this published copy it gives the same verdict and
gate statuses.

| gate | result | blocks | interval |
|---|---|---|---|
| G1 | pass: k = 2, 3, 4 at spill layers 20-32, forward and backward `torch.equal` to all-pinned and GA under the predeclared `--deterministic` rerun (the default runs were not reproducible GA vs GA, as in v1); the corrupted-read control differs | | |
| G2 | pass: 140 arms, held == plan <= budget; every B arm's excess outside its store <= its block's A excess (4 B in every block) | | |
| G3 | pass: Ctrl-C mid-backward, exception mid-forward, injected ENOSPC, real ENOSPC on a 64 MiB loop volume | | |
| G4 `ram_k2_half` | pass: overhead +2.51% | 5 | [-1.04%, +6.07%] |
| G4 `ram_k4_half` | pass: overhead +1.69% | 5 | [-2.13%, +5.50%] |
| G5 `disk_k4_same_drive` | **fail**: weight read rate with spill writes on the weights' drive 0.830x | 5 | [0.822, 0.837] |

Informational: `ram_k2_min` +3.85% [-0.97%, +8.66%], `ram_k3_min` +3.78% [+1.98%, +5.58%],
`ram_k4_min` +6.18% [+2.42%, +9.93%], `ram_k4_quarter` +3.86% [+0.41%, +7.30%], 5 blocks each;
`disk_k4_same_drive` step +13.3% [+10.3%, +16.4%]. The repeats rule needed no block past the
5-block pilot in any comparison. Hidden share of the spill I/O 0.987-0.996 (medians).

- Measured at `26c54ca8` (protocol v3; `a4_rule.py` is byte-identical to `efdd1a76`, so the
  rule is `a4-ram-budget-spill-v2`). Every source file hashed in `protocol.json` equals that
  commit. torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2.
- Box: vast.ai instance on host 122017 / machine 97012 (Quebec), RTX 3090 24 GB, gpu_frac 0.25
  (vast's ratio of offer GPUs to system GPUs; not isolation, not a separate disk), KVM VM
  `docker.io/vastai/kvm:cuda-12.9.1-auto`, driver 580.95.05, 49.3 GiB RAM, one virtio disk
  `vda` for the spill and the shards. The session and all 149 children (all exit 0) ran in one
  `systemd-run --scope -p MemoryMax=24G -p MemorySwapMax=0`, read back per PID; every arm
  record carries `memory.max` 25769803776 and `memory.swap.max` 0. 140 arms, all F1 `ok`.
- Admission (`admission/`, before the session): v3 protocol settings, ADMITTED: 1,561 of
  1,561 scheduled cycles, largest lateness 0.679 s (limit 5 s), 0 of 199,808 chunks read back
  wrong, no file left; 0.838 TB written and read back at 0.466 GB/s over 30 min.
- Description only (drops no arm, changes no verdict): no spill arm wrote below the
  admission's demand (0.233 GB/s) or its tested rate (0.466 GB/s); the slowest wrote at
  1.746 GB/s (`disk_k4_same_drive`), 7.5x the demand. The largest spill demand of any arm on
  this box was 0.156 GB/s (`ram_k3_min`).
- Contents: every JSON record of the session (`protocol.json`, `preflight.json`,
  `drive.json`, `correctness_k*.json`, `cleanup.json`, 140 arm records, `session.json`,
  `verdict.json`) and the admission records, without run logs. Box paths are replaced by
  `<box>` / `<work>` / `<enospc-volume>`; `provenance.json` lists each file's raw and published
  SHA-256 and the archive's hash (kept outside the repository).
- The step here was slower than in v1 on the same GPU model: all-pinned 0.84-0.88 s per
  micro-batch against v1's 0.58-0.59 s, at a higher SM clock (median 1950 MHz against
  1800 MHz). The cause is not established (this host shares the box with other GPUs; the CPU,
  host memory and PCIe path are not recorded). G4 compares B with A inside each block on the
  same box, so the verdict does not depend on the absolute step.
