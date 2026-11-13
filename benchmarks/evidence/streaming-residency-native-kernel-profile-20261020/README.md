# Additional native kernel profile: blocked before GPU work

**No native 1024-token training kernel profile was obtained.** E failed DNS
lookup during artifact-transfer preflight, before GPU model work or GPU tests.
The instance was deleted and independently verified absent. This package
preserves the corrected planned caller and failure/cleanup metadata; it contains
no observed kernel events. The completed A/D performance matrix is unaffected.

[Report](../../streaming-residency-flashattention-20261020.md) and
[attempt metadata](../../streaming-residency-native-kernel-profile-results-20261020.json)
describe the planned native-only 7B streaming profile. This adds **no speed or VRAM
comparison** and supplies no samples to the A/D timing matrix.

An earlier optional caller on D requested one measured step and was rejected
before model setup by the unchanged minimum-two-step guard. That failed caller,
exception and completed matrix remain retained. E uses warmup 1 plus two measured
steps, all under the profiler: three optimizer updates with accumulation 2.

The planned profile uses the same BF16 model/revision, nonzero initial adapters, data,
microbatch 1, sequence length 1024, 16 resident layers and 10 GiB allocator cap.
It selects `sdpa` explicitly. The external FA package is unused by this route.
E was created after D deletion; its preflight attempt overlaps no matrix timing window.

The frozen GPU source archive remains
`fc843cff5c56bfd875b5f8f3808f9d18972a3583cffdc92c152715722a85e6c7`.
The archive is identical to the model-running leases, whose 602 production files
match upstream `d500ae0dca7c4da905f8d59c9866af2f8aaa824b`. E prepared this source
locally but did not upload or execute the model. No numerical or performance gate changes.

`planned-profile.zip` and its internal/external SHA-256 manifests preserve the
planned native profile entry/config, dependency constraints, idle-GPU preflight,
failure, cost and deletion records. There is no E Chrome trace, model capture,
GPU test record or GPU export archive. Larger original observations from A/D
remain retained at the paths listed by their reports and export receipts.
No credentials, rental controller or model weights are published.

Use the [shared source/dependency setup](../streaming-residency-flashattention-20261020/README.md)
on an already available compatible Linux CUDA host. Copy archived `overlay/`
onto a separate checkout of the pinned upstream, and place `inputs/` under
`/root/liger`. The native-only route does not require the external FA wheel.
Preserve the recorded environment, four GPU-local physical CPU cores, idle-GPU
preflight and >=16 GB/s H2D qualification. Then run:

```bash
python /root/liger/prepare_7b.py
python /root/liger/profile_training_entry.py
```

When run on an already available host, the entry pins CPU affinity before importing Torch
and invokes the unchanged
benchmark at the actual training shape. It saves a complete Chrome trace,
original metrics/provenance and counts of actual CUDA flash forward/backward
events. Verify the input/initial-adapter hashes against D's observations.
The profile's step times and memory values are instrumented and excluded from
every performance/VRAM comparison. No command in this package rents hardware.
