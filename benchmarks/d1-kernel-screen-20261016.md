# D1 kernel screen: correctness first, no grid

This screen tests one new candidate, `d1_fused_nf4.py`, outside production
dispatch. It implements forward and dX with tiled NF4 decode inside `tl.dot`,
including nested scales and weight-dtype rounding. All frozen tensor state
goes through `save_for_backward`. No full dense-weight tensor, atomics or
split-K output reduction is used by the candidate. These are implementation
properties, not evidence of physical-GPU correctness or speed.

## Short decision sequence

1. CPU wrapper tests: saved-state version checks, checkpoint buffer refill,
   owner check, invalid scale extent rejection. CPU interpreter: FP16 decode,
   forward/dX and a small recycled-buffer positive/negative control. BF16 is
   unsupported by the installed interpreter and must run on physical CUDA.
2. Compile the eight FP16/BF16 × plain/nested × forward/dX variants for sm86.
   Compilation is not execution. CPU dot reduction differs from Torch on some
   inputs; never loosen the native physical-CUDA exactness criterion for that.
3. Physical-CUDA screen: seed 842, TF32 off, deterministic algorithms,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, same-process unpatched native bitsandbytes
   reference; independently compare forward, dX and the decoded-weight oracle.
   Shapes (M,N,K): (3,32,128), (17,67,96), FP16/BF16, plain/nested NF4.
   Retain raw CPU tensor witnesses and their hashes. Stop at the first exactness
   failure, before timing, training, model downloads or the larger alias test.
4. Only a passing candidate proceeds to the existing >165 MiB alias and
   streamed decoder/LoRA-gradient gates. Only after those pass can a short
   full-step speed/VRAM test justify a product change. This screen alone never
   closes D1 and never changes Soup's default dispatch.

## Cost boundary

User approved <=$0.25 from the remaining $1.939 for one short CUDA test.
Cumulative D1 cap remains $10. Use one exact-instance-bound rental with a
conservative teardown deadline and enough cost reserve. Do not touch other
account rentals. No full matrix, full-training repetitions or model downloads
are authorized by this screen. Preserve all failed attempts and delete the
test rental after retrieving the small evidence bundle.

## Run

```powershell
python benchmarks/harness/d1_kernel_quick_gate.py --device cpu --output cpu.json
```

```bash
python benchmarks/harness/d1_kernel_quick_gate.py --device cuda --output cuda.json
```

Triton is a prototype dependency, not a new Soup install requirement. Use an
isolated environment and record its exact Torch/bitsandbytes/Triton versions.
