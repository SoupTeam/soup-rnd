# D2 SQuAD 2.0 FP16 calibration on Kaggle T4

All six training arms completed 20 optimizer updates on the same physical Tesla T4.
FastLoRA and the NoRA fork control matched ordinary PEFT LoRA bitwise in all 616
saved adapter/gradient tensors and matched all 20 losses to three decimal places.
Liger and Unsloth failed the unchanged numerical gate. The full campaign stopped
before training/evaluation, so there is no EM/F1 or convergence result.

| Arm | Median update | Peak allocated / reserved VRAM | Losses matching PEFT | Tensors beyond 1% relative L2 |
| --- | ---: | ---: | ---: | ---: |
| PEFT LoRA | 2.528 s | 5.524 / 5.936 GiB | Reference | Reference |
| Soup FastLoRA | 2.644 s | 4.477 / 4.861 GiB | 20/20 | 0/616 |
| LoRA on the NoRA fork | 2.621 s | 5.524 / 5.936 GiB | 20/20 | 0/616 |
| NoRA | 2.834 s | 5.526 / 5.938 GiB | Different adapter math | Different adapter math |
| LoRA + Liger | 2.519 s | 4.616 / 4.938 GiB | 13/20 | 56/616 |
| Unsloth | 2.533 s | 6.478 / 6.629 GiB | 8/20 | 148/616 |

FastLoRA reduced allocated VRAM by 18.95% and increased median update time by 4.58%
in this run. NoRA's median update took 8.12% longer than its same-fork control.
These are single-calibration observations without paired ABBA timing repetitions.
The NoRA author-fork audit passed zero-update, gradient, save/load and merge checks.
It did not evaluate task quality.

## Scope and evidence

- TinyLlama 1.1B Chat at fe8a4ea1ffedaf415f4da2f062534de366a451e6, seed 792,
  rank/alpha 16, microbatch 2, accumulation 8. FP16 base and autocast, FP32 LoRA
  parameters and GradScaler initialized at 128. The published D2 protocol's BF16
  run remains open. Two separate Kaggle L4 requests were assigned T4s.
- The fixed SQuAD 2.0 fixture has 13,032 train, 1,024 validation and 2,048 test
  examples. The tokenized split hashes match the historical protocol. Calibration
  used matched train batches. It did not evaluate the validation or test split.
- This is a resident training-stack calibration. Soup FastLoRA uses PR #5 source
  at [abbbff3c](https://github.com/SoupTeam/soup-rnd/commit/abbbff3cd7295a4c6209c6b6a84b56db6ba10f8f).
- [Machine-readable replay](local-replay.json) and [evidence bundle](evidence.zip)
  retain per-update timings/losses, batch IDs, execution proof, the input manifests,
  checksums and the executed Kaggle source. All six snapshot checksums were verified
  after download. The full original tensor files remain in the local archive.
  The notebook may require Kaggle access.
  [Kaggle run, version 2](https://www.kaggle.com/code/koko667/soup-d2-squad-fp16-calibration/versions/2).
