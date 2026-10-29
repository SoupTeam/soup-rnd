# A3 stage 2 GPU session 2: one RTX 4090 VM, 2026-10-27 (rule v2). NO VERDICT on cleanup

**`NO VERDICT: criterion 3 has no verdict; L2L is outside the A3 stand's GA tolerance
(criterion 1b, not a spill finding)`** by rule v2 (`verdict.json`, unchanged). Re-running
`a3_spill_rule.py` at `667c4a1b` on the session's raw records reproduces `verdict.json` byte for
byte; on this published copy it reproduces the published file byte for byte (the two differ only
by the path placeholders).

| # | Criterion | Result | Numbers |
|---|---|---|---|
| 1 | spill == no spill | PASS | `l2l_spill` (16 of 32 layers spilled) bit-equal to `l2l_nospill_1` over 3 optimizer steps (gradients, losses, learning rates); `l2l_nospill_2` repeats `l2l_nospill_1` bit for bit |
| 1b | L2L vs GA (reported) | FAIL | A3 stand `decide()`: `l2l_out_of_tolerance` at 4 checks: gradient relative error 0.0181, 0.0213, 0.1772 at steps 0, 1, 2 against limits 0.0165, 0.0206, 0.0767, and loss at step 0 (0.000222 against 0.0000103). `ga_2` repeats `ga_1` bit for bit; the floor arm (`ga_floor`, EFFICIENT_ATTENTION) judged |
| 2 | RAM | PASS | `memory.peak` 6,102,315,008 B for `l2l_spill` at MemoryMax 6,354,370,560 B; 6,300,880,896 and 6,400,360,448 B for the no-spill arms at 24 GiB; no OOM kill; excess (CUDA host-allocator growth over training outside the planned store) 36,877 B with and without spill |
| 3 | cleanup | NO VERDICT | normal end: no file after `train()` nor after the exit (holds); kill9: the file stayed after SIGKILL, `kill9_next` removed it at start and trained 3 steps (holds); sigint: the scenario was not reached |
| 4 | cost | PASS | overhead +0.03% per optimizer step, 95% interval [-0.02%, +0.09%], 5 A-B-B-A blocks; the B children spilled 16 to 22 layers |

1b is reported apart and is not a spill finding: spill equals no spill bit for bit, so the gap
lies between the L2L path and gradient accumulation in this bf16 configuration. Its cause is not
established here. The gradient tensors it was computed from were not brought back (below).

## The sigint part

The parent sent one SIGINT 0.52 s after the step-2 entry marker (1 optimizer step done, inside
`training_step`, the spill file present). The child kept writing markers for 1.31 s after the
signal, finished steps 2 and 3 and exited `normal` with return code 0; its spill file was
removed by the normal end. Rule v2 gives no verdict because no KeyboardInterrupt came.

The session was launched as `setsid nohup bash -c '...' &` from a non-interactive shell. A
background job of such a shell starts with SIGINT ignored, exec keeps an ignored signal
ignored, neither `subprocess.Popen` nor `systemd-run --scope` resets it, and Python leaves an
inherited SIG_IGN in place. Reproduced on Linux after the session with the same launch: a child
started that way stays running after a real SIGINT, and raises KeyboardInterrupt once it
restores Python's handler. Session 2 recorded no SigIgn reading, so for session 2 itself this is
the explanation the reproduction supports, not a reading from the box. Protocol v3 (§3.7 of
`../../a3-spill-gpu-check.md`) changes the stand; it does not re-judge this session.

## Memory after the plan point (description, no verdict)

In each of the 28 records whose plan was limited by the child's own scope, `memory.peak` at the
end minus the plan's `used` minus the plan's `host_bytes` (growth outside the store, over the
whole run) was 183,582,720 to 297,558,016 B; the largest is 55% of the product's 512 MiB
reserve. The 15 children at the tight limit ended 252,055,552 to 366,022,656 B below their
MemoryMax. One box, one model and shape: the reserve stays chosen, not measured.

## Box, admission, calibration

- Measured at `667c4a1b` (protocol v2 fixed at `6bdd0ec8` before the rental). All 663 source
  files hashed in `protocol.json` equal that commit. torch 2.14.1+cu130, transformers 5.19.0,
  peft 0.21.2, bitsandbytes 0.50.2.
- Box: vast.ai KVM VM, RTX 4090 24 GB, PCIe gen 4 x16 in all 33 child records, driver
  580.95.05, 47.4 GiB RAM, one virtio ext4 disk for the spill and the weights, gpu_frac 1.0
  (vast's ratio; not isolation, not a separate disk). The only RTX 3090 VM offered was PCIe 3.0,
  so the PCIe 4.0 4090 was taken as §5 allows. Fixture SHA-256 equal to A4 v3's and session 1's.
- Preflight (`admission/`): per-PID cgroup readback 24 GiB, swap 0; a real 64 MiB memcg OOM kill
  read back through `journalctl -k` by the stand (`kernel-oom-probe.txt`).
- Admission (`admission/`, `step0/`): ADMITTED. A.value 1.1890 s (no-spill A child, median of
  5 timed steps), D 0.226 GB/s, 180 of 180 cycles of 0.595 s, largest lateness -0.043 s (limit
  5 s), 0 of 11,520 chunks wrong, no file left.
- Calibration: working set 5,566,652,416 B at 24 GiB; first tight limit 6,447,693,824 B gave
  10 spilled layers (outside [12, 20]); recomputed once, 6,354,370,560 B gave 16. Margins at that
  limit (description): build +1,031,861,472 B, training +279,468,808 B, no warning.
- Session: 33 children from 14:58:42Z to 15:07:46Z, exit 0, `state: finished`; 32 exited 0,
  `kill9` -9 by its intended SIGKILL (no kernel OOM line). No spill file on the box at the end.
- Cost of the box: GPU 0.994 h x $0.52 = $0.517, storage $0.050, **$0.567** (invoices). The
  instance was destroyed and confirmed gone through the API.

## Contents

`protocol.json`, `session.json`, `verdict.json`, `calibration*.json`, `compare.json`, the arm,
cleanup and cost records, every child's marker, the admission's inputs (`step0/`) and records
(`admission/`). Box paths are replaced by `<box>` / `<work>`; `provenance.json` lists each
file's raw and published SHA-256, the archive's hash, the kernel journal's and the deleter log's.
Not published: child logs, the setup and test logs, the kernel journal, the deleter. The
gradient tensors (470 MB) were left on the box and destroyed with it, so `compare.json` cannot
be recomputed from tensors; each arm record keeps the per-step SHA-256 of its gradients, and
`compare.json` records that every tensor file matched them.
