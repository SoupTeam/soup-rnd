# A3 stage 2 GPU session 3: RTX 4090 VM, 2026-10-28 (rule v3). SPILL CONFIRMED; 1b FAIL

**`SPILL CONFIRMED: equal, within its RAM, cleaned up; overhead below 10% confirmed on a faster machine; L2L is outside the A3 stand's GA tolerance (criterion 1b, not a spill finding)`**

This is the exact verdict in [`verdict.json`](verdict.json), rule v3's session output, not a
new judgement. Criteria 1, 2, 3 and 4 PASS; **criterion 1b FAILS**. The
[protocol](../../a3-spill-gpu-check.md#37-protocol-v3-fixed-after-session-2-before-session-3)
was fixed at `bc2ca55038a2d85950356cfdd38d556333519ad1` before the rental. The measured
code was `ec795ef175c222b60ff91b9ba44f7f2bfad55dd5`; every one of the 664
`source_sha256` entries in `protocol.json` matches that commit, checked against
`git cat-file` bytes by the integration owner. Sessions
[1](../a3-spill-rtx4090-20261027/README.md) and
[2](../a3-spill-v2-rtx4090-20261027/README.md) keep their original NO VERDICT results.

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | spill == no spill | PASS | `l2l_spill`, 13 of 32 layers spilled, equals `l2l_nospill_1` bit for bit in gradients, losses, learning rates and optimizer-step count (3); `l2l_nospill_2` repeats the no-spill reference bit for bit |
| 1b | L2L vs GA (reported separately) | FAIL | `l2l_out_of_tolerance`, four checks: gradient relative errors 0.0180907, 0.0213343, 0.1772194 at zero-based steps 0/1/2 against limits 0.0164525, 0.0206153, 0.0766768; loss error 0.000221731 at step 0 against 0.0000102543. GA repeats bit for bit; the EFFICIENT_ATTENTION floor arm judged |
| 2 | RAM | PASS | spill `memory.peak` 6,154,133,504 B <= MemoryMax 6,409,945,088 B; no-spill peaks 6,354,161,664 and 6,374,264,832 B at 25,769,803,776 B (24 GiB); no OOM kill; CUDA host-allocator excess outside the planned store 36,877 B in all three arms |
| 3 | cleanup | PASS | no file after normal `train()` or process exit; one SIGINT in the backward-recompute hold of optimizer step 2 raised KeyboardInterrupt, exit 130, file gone; SIGKILL left the file, the next start removed it and trained all 3 steps |
| 4 | cost | PASS | +2.39% per optimizer step, two-sided 95% interval [-1.80%, +6.57%], 5 valid A-B-B-A blocks, no void blocks; B children spilled 12-18 layers |

The four 1b failures and their numbers are the same as session 2's. Their cause remains
unestablished. They are not a spill finding: the spill and no-spill L2L arms are bit-equal.
This session does not establish L2L/GA parity or model quality.

## Signal and cleanup evidence

The tiny 8-layer CPU signal preflight passed on this VM **before downloading the model**,
through the session's background launch. Its launcher read SIG_IGN; children inherited that
and installed Python's `default_int_handler`. All checks passed: SIGINT interrupted the
backward hold, exit 130, no file left; SIGKILL left a file; the next child removed it and
trained every step, in the planned scopes. Only the path-sanitized report is published as
[`signal-preflight/preflight.json`](signal-preflight/preflight.json), not its tiny model,
shards, child files or logs. This preflight is launch evidence, not a CUDA measurement.

In the scored CUDA `sigint` child, the parent sent exactly one SIGINT while the child held
at the backward recompute of **zero-based layer 16**, first micro-batch of optimizer
step 2, with 1 optimizer step complete and a spill file present. The child had 17 spilled
layers for this scenario (the scored equality arm had 13). At the send, SIGINT was caught,
not ignored or blocked. The hold was released by KeyboardInterrupt, the child exited
`interrupted` / return code 130, and no spill file remained. `session.json`,
`cleanup/sigint.json` and `markers/sigint.json` retain those readings.

`kill9` was killed at its backward hold. Its spill file stayed after SIGKILL;
`kill9_next` in the same directory listed it in `stale_removed`, removed it and trained
3 optimizer steps normally. Normal completion also removed the spill file.

Cleanup remains **best-effort**, not crash-proof. SIGKILL, an OOM kill, power loss or an OS
crash cannot run a handler; the stale file stays until a new run uses the same directory
and the original process is no longer alive, or it is removed by hand. This experiment
does not cover a full drive, shared spill directories, resume with spill or other shapes.

## Box, admission and calibration

- vast.ai KVM VM, instance 58144303; Quebec host 655981 / machine 154269, the same host as
  session 1. RTX 4090 24 GB, PCIe gen 4 x16, driver 580.95.05, 49.35 GiB host RAM; kernel
  6.8.0-59-generic, systemd 249, local ext4 `/dev/vda1` for weights and spill. No fitting
  PCIe 4.0 RTX 3090 / 3090 Ti VM was offered, so the 4090 was used as the protocol permits.
  The provider's `gpu_frac` 0.125 is not evidence of isolation or a separate physical disk.
- Offered combined rate: $0.562962962962963/h. Actual invoiced charges are below; this
  advertised rate is not a bill or an elapsed-runtime calculation.
- Python 3.12.16; torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2,
  bitsandbytes 0.50.2. VM checks recorded **323 passed, 1 skipped, 2 deselected**; the two
  separately CPU-run resume checks recorded **2 passed**. These are retained VM records,
  not a claim about current-branch CI. Fixture hashes match A4 v3 and sessions 1 and 2.
- Per-process scope readback: cgroup v2, MemoryMax as planned, MemorySwapMax 0. A real
  64 MiB memcg OOM probe was read back through `journalctl -k`; this is the separate
  preflight probe, not an OOM in a scored training child.
- Admission: **ADMITTED**. No-spill A.value 1.557790986 s; D 172,318,018.5355 B/s,
  period 0.778895493 s. All 180/180 cycles completed in 139.646 s, maximum lateness
  0.034849073 s (limit 5 s), 0/11,520 mismatched chunks, no file left. Full inputs,
  schedule and CSV records are under `step0/` and `admission/`.
- Calibration: working set 5,528,711,168 B at 24 GiB produced tight limit
  6,409,945,088 B. The single calibration check spilled 14 layers, within [12, 20], so
  no second calibration check was needed. Descriptive build margin +1,290,664,024 B,
  training margin +373,282,520 B, no warning. The product's **512 MiB reserve is chosen,
  not measured**; these margins do not validate it for other models or shapes.
- Session start **2026-10-28T20:31:59Z**, finish **20:46:26Z**, from the retained
  `session-start.txt` and `session-exit.txt`. All 32 children ran; session exit 0,
  `state: finished`. Thirty children exited 0, `sigint` 130 and `kill9` -9 as intended.

Descriptive growth after the plan point (not a new criterion): across 27 records limited by
their own scope, `memory.peak - plan.used - plan.host_bytes` was 186,515,456 to
297,218,048 B; the largest is 55.36% of the chosen reserve. Fourteen tight-limit records
ended 255,811,584 to 362,397,696 B below MemoryMax. No spill files remained at session end.

The speed conclusion is exactly the rule's: below 10% confirmed **on a faster machine**.
It covers this admitted 4090 VM, Mistral-7B NF4, LoRA r 8, k 4, seq 512, the RAM tier,
this virtual disk and these step lengths. It does not promise that overhead on a slower
GPU, another drive, another model or another shape.

## Replay and publication provenance

The integration owner extracted the **measured commit** into a temporary checkout and ran
its CLI `compare` on the retained tensors on CPU, followed by its rule on the raw session.
Both `compare.json` and `verdict.json` reproduced **byte for byte**; all six gradient tensor
hash checks passed. [`verification.json`](verification.json) records the replay result.
The original and replayed comparison SHA-256 are both
`02ef68a1fa8fe4ff26257a6b1e7e8116c77c83f670082f2c28537041b2bc71ba`.
The 18 tensor files (492,359,238 B) are retained privately in the raw session's sibling
`a3-spill-03.tensors/` and in the private archive, so this comparison can be recomputed.
They are not in this public folder; public JSON alone cannot redo the tensor comparison.
The rule can be recomputed from this public JSON (box paths are placeholders in that copy).

Published: `protocol.json`, `session.json`, `verdict.json`, both calibration records,
`compare.json`, six arm records, the two cleanup records, 20 cost records, 31 markers,
the step-0 and admission records, the sanitized signal-preflight report and the local
verification report. The publisher uses an explicit filename whitelist, not a recursive
copy. No child/setup/test logs, tensors, operational scripts, credentials, kernel journal,
deleter log, or private API/billing snapshots are published.

Only these path substitutions change evidence bytes, applied longest first:

- `/root/a3/soup-rnd` -> `<work>`;
- `/root/a3` -> `<box>`.

[`provenance.json`](provenance.json) lists every evidence file's **raw SHA-256 and published
SHA-256**, plus hashes and relevant values for retained private evidence. `SHA256SUMS` is
LF and covers the 75 evidence files and `provenance.json`; this authored README is not
part of the evidence manifest. `.gitattributes` marks the whole result folder `-text` to
preserve bytes on checkout.

Private archive `m3-final.tar`: **212 entries**, SHA-256
`19c3ada65e89ad0adaf805dbc3c530e8a5a3de5cd38984f5e7c3d8c9554819c2`.
The archive hash and count were verified by the integration owner. Private `deleter.log`
SHA-256: `547680e52a0748a0bc5f526cd3fcec0c15c311289a42ab1dc28c7a46c1500a69`.

## Invoices and instance status

Invoices for instance 58144303: GPU 0.803 h at $0.5333/h, charged **$0.428**; storage
0.832 h at $0.0296/h, charged **$0.025**; session total **$0.453**. A3's three sessions
total $0.270 + $0.567 + $0.453 = **$1.290**, leaving **$8.710** of the $10 budget.
The private invoice snapshot's hash is in `provenance.json`.

At **2026-10-29T13:19:26.272356+00:00**, the API returned `instances: null` for this
instance, confirming it gone. **Exact deletion time and mechanism were not retained.**
The deleter was armed for 2026-10-29T08:09Z, but its retained log records only arming,
not a deletion action. The deadline is not evidence that the instance self-deleted then.
