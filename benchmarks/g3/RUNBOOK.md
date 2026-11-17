# G3 operator runbook

No paid window is authorized by these files. Do not use a G6 instance, old lease
limit, old boot/controls, or another person's running GPU. The macOS inventory
is only a collector smoke, not a negative GDS result.

## Before a GPU window

1. Review `PROTOCOL.md`, source and CPU tests. Have the owner commit/push them
   before transfer arms; do not manufacture an earlier timestamp.
2. Agree a cumulative lease/disk/traffic cap, exact target instance and absolute
   deadline. `authority.example.json` is intentionally disabled and unconfigured.
3. Select a Linux CUDA GPU with a visible local NVMe namespace and ext4/xfs data
   mount. An overlay/virtio/LVM mapping is blocked by the current scope. A GPU
   with unknown storage topology is not ready for a paid measurement window.
4. Need installed CUDA runtime **development** headers, cuFile headers/library,
   a C++17 compiler, Python 3.10–3.12, `rich`, `psutil`, NVIDIA tools, `findmnt`,
   `lsblk`, `lspci`, and `dd`. No weights or model download. Phase-1 discovery
   retains a missing/failed gdscheck; native probing may follow only if selected
   GPU and actual local NVMe topology are established and the native binary
   builds. The gdscheck message alone is never the verdict.
5. Compile on that exact SDK. Missing stats headers/API are an environment gap,
   not unsupported hardware. Do not install a kernel module or alter the host's
   driver/security configuration as an automatic repair. A previously approved
   official userspace SDK installation must be bounded and recorded separately.
   On Linux with the real development headers installed, the regression test
   below checks native declarations without executing the GPU:

   ```sh
   python3 -m pytest --no-cov -q tests/test_g3_probe.py -k installed_cuda_sdk
   ```

   Missing SDKs skip this test; a skipped check is not a compilation pass.
6. Freeze a valid authority, same-boot source/runtime identities and sensor
   controls. Keep package writers and other GPU work out of the measurement
   window. Never stop or inspect unrelated instances to obtain a quiet window.

All paths below are operator-assigned absolute paths on the authorized host:
`REPO` is the committed checkout, `SESSION` a **new** root on the target NVMe,
`EVIDENCE` an archive path outside that root, and `G3_GPU_UUID` the exact selected
GPU UUID. These are placeholders, not ready-to-run credentials or a lease.
No commit/push/PR is performed by the runner.

## Inventory, preparation and build

```sh
cd "$REPO"
python benchmarks/harness/g3_capability.py --data-path "$SESSION" \
  --out "$SESSION/inventory_001"
python benchmarks/harness/g3_probe.py prepare --work "$SESSION/work"
python benchmarks/harness/g3_probe.py build --out "$SESSION/build" \
  --cuda-root /usr/local/cuda --stats on
```

Create `SESSION` once before inventory; it must already exist for its mount to
be attributable. Default fixture is 64 MiB of deterministic, distinct 4 KiB
blocks. It is fsynced and read back with SHA256. All outputs are exclusive.
The native build records its compiler, linked libraries, available header SHA
and binary SHA. No CPU substitute can pass the native-binary guard.

`--stats off` is a prospective older-SDK capability-only option. It cannot
produce a positive direct-route verdict or a performance result. It may retain
an explicit native API rejection, with successful ordinary transfer controls.
Changing SDK/config/source after a transfer arm requires a separately declared
attempt; never overwrite or silently restart the existing attempt.

## Independent disk-sensor controls

Choose `EXPECTED_DEVICE` from the actual path/counter mapping and predeclare it.
This is the F1 sensor disk name, potentially a partition, rather than a guessed
provider disk label. The calibration verifies that direct reads advance its
counter, and tests clean, owned and **independent foreign** reads. It creates
one 1.25 GiB non-sparse control fixture; no cache dropping or privileged commands.

Watcher connection:

```sh
timeout -s TERM -k 10s 910s env CUDA_VISIBLE_DEVICES="$G3_GPU_UUID" \
python benchmarks/harness/g3_controls.py calibrate \
  --out "$SESSION/controls" --expected-device "$EXPECTED_DEVICE" \
  --harness "$REPO/benchmarks/harness/arm_validity.py"
```

After `FOREIGN_READY`, from a **second independent SSH connection**:

```sh
timeout -s TERM -k 10s 160s python "$REPO/benchmarks/harness/g3_controls.py" foreign --out "$SESSION/controls"
```

Starting the foreign reader as the watcher's child is invalid. Preserve the
handshake and ancestor evidence. The required result is
`PASS_DISK_SENSOR_CONTROLS`, same boot and same target sensor device. This is
not proof that an entire rented physical host was isolated.

## Capability, then optional performance

The authority JSON must contain the newly agreed cap, rate, reserved traffic and
cleanup cost, prior spending under this same G3 budget, exact instance/GPU/boot,
lease start/deadline and `authorized:true`. Closed G6 costs are a separate study;
failed setups within this G3 budget may not reset its cumulative cap.
The code accepts only finite, positive estimates within that cap and a lease
no longer than three hours. Each phase needs at least 940 seconds left and is
internally bounded to 900 seconds; each native child to 120 seconds. CPU fixture
and SDK setup are not reasons to reset the lease clock. Preserve timeout output.

`COMMIT` is the exact owner-pushed 40-character source commit. `RULE_SHA` is the
SHA256 of the committed `benchmarks/g3/PROTOCOL.md`. The runner verifies each
listed source file against `git show COMMIT:path` before executing a transfer.
The checkout must contain the Git objects; copied files alone do not pass.

```sh
python benchmarks/harness/g3_probe.py capability --phase capability \
  --work "$SESSION/work" --build "$SESSION/build" \
  --authority "$SESSION/authority.json" --controls "$SESSION/controls/calibration.json" \
  --commit "$COMMIT" --rule-sha "$RULE_SHA"
python benchmarks/harness/g3_probe.py audit --out "$SESSION/work/capability_001"
```

Four fresh processes, control/direct/direct/control, read one complete fixture
per process. The ordinary path uses the same aligned O_DIRECT + pinned staging
+ completed H2D primitive as #974, implemented in C++ rather than importing the
Python tensor wrapper. It never selects the shipped reader's buffered fallback.
The direct path uses cudaMalloc device memory, verifies the driver's effective
compatibility flag, and requests per-process level-3 cuFile statistics. Its
read bytes, selected UUID and exclusive P2P/NVFS counts must cover all reads;
POSIX, unaligned, dynamic-route and error counts must be zero.

Both paths check every transfer against an independent CPU read outside the
timing bracket, then save a complete D2H result. The Python audit independently
recomputes the deterministic fixture SHA and checks actual D2H bytes. Runtime
versions, selected GPU and native process return codes are retained. Two equal
explicit native not-supported errors, bracketed by successful positive controls,
can yield `UNSUPPORTED_ON_TESTED_STACK`. A permission, compiler, driver-install
or integrity failure cannot. If unsupported, report this tested stack and stop;
no performance ratio is produced and no two-drive support is inferred.

Only if the raw local audit yields `DIRECT_PATH_VERIFIED`, on the **same** boot,
GPU, binary and lease:

```sh
python benchmarks/harness/g3_probe.py capability --phase performance \
  --work "$SESSION/work" --build "$SESSION/build" \
  --authority "$SESSION/authority.json" --controls "$SESSION/controls/calibration.json" \
  --commit "$COMMIT" --rule-sha "$RULE_SHA"
python benchmarks/harness/g3_probe.py audit --out "$SESSION/work/performance_001"
```

Each fresh process performs two warm-up plus three timed full-fixture reads and
returns their median. Every read, including warm-up, is byte-checked outside
its timing bracket. Direct route counters cover all five reads. Start with five
A-B-B-A blocks; use METHOD's one-top-up repeats rule. Absolute limit: ten total
blocks / 900 seconds, including replacements for void blocks. If the rule asks
for more, retain the evidence and report NO_VERDICT, not a shortened comparison.
Statistics remain enabled: these are instrumented primitive timings, not a
training speedup. CPU time and **explicit** pinned-host allocation are recorded.
Total internal CUDA/cuFile pinned host allocation is UNKNOWN in this version;
no total-RAM saving claim or completed pinned-RAM checklist follows from it.

The implemented runner covers **one declared NVMe namespace**. If capability is
positive, the source's two-drive question still needs a separately frozen paired
concurrent runner and two distinct observed devices. Two independent single-disk
runs, or two files on one disk, must not be published as aggregate two-drive
bandwidth. This limitation is visible in the report and prevents a full positive
G3 closure until resolved. A stack-specific unsupported result follows the
negative-deliverable branch explicitly allowed by the task.

## Preserve and verify locally before provider cleanup

Quiesce this session's processes; never delete evidence to make a rerun possible.
The session root includes build/compiler metadata, controls, all failed and
successful arms, byte readbacks, authority, source snapshots and raw sensor rows.

```sh
python benchmarks/harness/g3_archive.py pack --root "$SESSION" --archive "$EVIDENCE"
```

Read the archive SHA from the adjacent machine-readable `.receipt.json` file. Download the archive without disabling
TLS or SSH host-key checks. On the local machine, with no server dependency:

```sh
python benchmarks/harness/g3_archive.py verify --archive "$EVIDENCE" \
  --expected-sha "$ARCHIVE_SHA" --out "$LOCAL_NEW_VERIFY_DIRECTORY"
```

Every unique byte blob is exported once, while every original logical path is
reconstructed and SHA-verified. Verification refuses unsafe paths, symlinks,
extra/missing/duplicate members, wrong sizes, modified SHA and oversized exports.
Never execute code extracted from an untrusted archive. Run the reviewed local
auditor on preserved evidence, then prepare an English report with exact scope,
versions, commands and limitations. Stop only the authorized instance; permanent
Destroy needs action-time confirmation. Verify cleanup through the provider.

The original `c611499fa` preparation failed native compilation before any transfer.
Preserve that failure; a corrected source needs a new owner commit and separately
declared attempt, with prior G3 spending included. No accepted transfer result exists.
CPU test substitutes are marked SYNTHETIC_CPU_MOCK and
are rejected from hardware verdicts. No bare "unsupported on all GeForce" or
speedup claim is justified by these preparations.
