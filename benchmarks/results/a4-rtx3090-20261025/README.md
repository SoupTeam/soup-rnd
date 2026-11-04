# A4 v1 GPU session: one RTX 3090 VM, 2026-10-25 (rule v1)

**`SPILL NOT SAFE: A3 must not ship the spill path as built`** by rule v1 (`verdict.json`,
unchanged): G1 pass, **G2 fail**, G3 pass, G4 no verdict, G5 no verdict.

- Measured at `e123b5ff` (the A4 branch before its rebase onto main; the L2L step and store
  then ran from the harness's own copies, hashed in `protocol.json`). The rule that judged it
  is `a4-ram-budget-spill-v1` (`verdict.json["rule"]`).
- Box: vast.ai instance 57643280, RTX 3090 24 GB, KVM VM `docker.io/vastai/kvm:cuda-12.9.1-auto`,
  driver 580.95.05, 98.8 GiB RAM, one virtio disk for the spill and the shards; torch
  2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2. The session and all
  289 children ran in one `systemd-run --scope -p MemoryMax=24G -p MemorySwapMax=0`, read back
  per PID; every arm record carries `memory.max` 25769803776 and `memory.swap.max` 0.
- Contents: every JSON record of the session (`protocol.json`, `preflight.json`, `drive.json`,
  `correctness_k*.json`, `cleanup.json`, 280 arm records, `session.json`, `verdict.json`),
  without run logs. Box paths are replaced by `<box>` / `<work>` / `<enospc-volume>`;
  `provenance.json` lists each file's raw and published SHA-256 and the archive's hash
  (581 files, kept outside the repository). Rule v1 on this published copy gives the same
  verdict and gate statuses.
- After the run, not changing the verdict: `posthoc-G2-4-bytes.md` (the 4 B are torch's
  pinned buffer for `float(loss)`, confirmed locally: `posthoc_probe_hostalloc.py`,
  `posthoc-probe-result.json`) and `v1-under-rule-v2-ILLUSTRATION.json` (rule v2 on these
  records, an illustration, not a verdict).
