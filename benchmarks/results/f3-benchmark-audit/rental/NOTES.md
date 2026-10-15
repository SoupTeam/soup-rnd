# F3 rental notes (ticket 08)

Label: `"A100-PCIE-80GB vast.ai, real"`. Instance 56167498, offer 45716570 (UK), A100 80GB PCIe,
1.00 $/h, storage 0.20 $/GB/month, traffic 2.67 $/TB down and 4.00 $/TB up. Image
`pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`, disk 60 GB. Created 2026-10-16 about 04:08 (+05),
destroyed about 04:20. Spend: balance 4.06 $ to 3.91 $, so about 0.15 $.

## Results

- `sleep(60)` observation (`f3-results/observe-sleep60.json`): `suspend` and `power` answer,
  `foreign_reads` does not (the model path did not exist yet, no data dir at that point).
- C1 arm A (Unsloth 2026.1.2, Qwen2.5-0.5B, batch 16, seq 512, 20 steps): `pass`, in two fresh
  processes, peak allocated 11.2 GB. Batch 16 fit; no changed condition.
- C1 arm B (Chronicals): out of scope, custom loop, not an HF `Trainer`. No speed blocks.
- C2: not run. `f3_arm.py` has no LoRA path and `unsloth_max_benchmark` is not in the repo.
- Raw logs: `c1.log`, `bootstrap.log`, `f3-results/`. Uploaded copy: private HF repo
  `m1kado/f3-audit-results`.

## What went wrong on the way

1. First run: the control reports were not on the instance, so the audit refused with
   "no control run" before arm A started. Fixed in `f3_rental.py` and the laptop control
   reports (SmolLM2-135M, "RTX 3050 Laptop, control") were copied over. The control was not
   rerun on the A100.
2. Second run: the runtime image has no C compiler, Inductor failed. `build-essential` installed
   by hand; `f3_bootstrap.sh` does not do this yet.
3. `CONTAINER_ID` is not set in the ssh shell, so the on-instance `destroy` raised after the upload.
   The instance was destroyed from the laptop. The per-instance destroy path stays unverified.
4. The lifecycle spend line reads 0.00 $ because the lifecycle clock starts at the run, not at the
   rental. The real figure comes from the account balance.
