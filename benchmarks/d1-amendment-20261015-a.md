# D1 amendment A, 2026-10-15, before the second CUDA cohort

The original rule and the first cohort at 5f415bf0 are retained unchanged.
The first fp16/bf16 correctness runs passed; timing remains NO_VERDICT.
This amendment fixes instrument/placement defects observed in that cohort.

1. PyTorch reports this device UUID without `GPU-`, whereas NVML includes it.
   Parse the UUID through uuid.UUID and store the canonical `GPU-<uuid>` form,
   alongside the raw CUDA value. The actual UUID must still match exactly.
   Old records with false identity checks remain rejected; they are not repaired.
2. Place temporary sharded checkpoints under the OS temporary directory, which
   is allowed by Soup's existing containment gate. Do not change that gate,
   HOME, TMPDIR, accepted model weights or the kernel arithmetic.
3. F1 cannot map this local Docker overlay to a block device. Only for the
   recorded `maps to no block device` condition, use a conservative alternate:
   sum the read-byte deltas of EVERY /proc/diskstats entry, partitions and stacked
   devices included, with NO subtraction for own-process I/O. Below 1 GB is ok;
   1 GB or more is void. Missing/changing/nonmonotonic counters are unknown.
   This is a stricter bound on local block I/O; double counting and own reads can
   reject a clean run, but cannot hide foreign local block reads. Record every
   device and both readings. Retain the original F1 unknown row in the record.
   Required power and suspend checks remain; no other unknown is waived.

This alternate concerns the local RAM-tier container study only. It is not a
per-volume measurement, does not attribute a reader and is not a cold/NFS/network
filesystem gate. No disk-tier or isolation claim beyond the recorded sensors
follows. All GPU peer, clock, power-limit, numerical and repeat guards remain.

The five-block protocol is unchanged. Initial one-block screens cannot establish
an accepted speedup. No production implementation is changed in this amendment.
