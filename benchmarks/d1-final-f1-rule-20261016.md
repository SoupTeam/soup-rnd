# D1 final bounded F1 check, fixed before execution

Purpose: test the existing scoped BF16 candidate through the actual Soup streaming
runtime on a decoder with 4096-wide projections. This is a synthetic shape fixture,
not pretrained model quality or a universal Llama throughput claim.

Frozen configuration: Llama, seed842, hidden4096, intermediate4096, four layers,
32 attention heads, eight KV heads, vocab64, sequence128, nested NF4, BF16,
two RAM-backed streaming slots, pageable host memory, LoRA q/v rank4 alpha8,
dropout0, SGD lr0.001. Default BF16 reduced precision remains True; TF32 disabled.
A is shipped streaming dispatch. B uses wmma_native64 only for previously covered
descriptors; all other projections follow the shipped fallback.

Before timing: three full updates in separate native/candidate processes must match
logits, loss, every gradient and updated adapter exactly. Otherwise stop without timing.
Existing archived 192 MiB checkpoint and negative-control gates remain applicable.

Timing: 20 warmup and 200 synchronized full SGD steps per fresh-process arm,
five ABBA blocks. The existing F1 ArmWatch, GPU lifecycle monitor, clock spread <=5%,
unchanged power limit and sampled memory utilization apply. Missing sensors mean
NO_VERDICT. Existing conservative all-block-read fallback is allowed for overlayfs.
Stop after the first invalid block; retain every failed attempt.

Record tok/s, allocated/reserved peaks, actual per-step timings, raw telemetry,
and a same-session 128x4096x4096 dense GEMM after primary peak capture (>=200 calls,
>=1 second timed and >=20 calls/1 second warmup). Apply d1_reference validation.
Apply F1 sizing after five blocks. If extra blocks are required, report NO_VERDICT;
no automatic extension. Positive and negative measured outcomes are both useful.

An SM clock lock is optional only on the exact leased GPU UUID, recorded and reset
on exit and by lease cleanup. No reference arithmetic or validity threshold changes.
Maximum new lease cost $0.30 within the existing total $10; maximum lease 45 minutes.
No new long training, old measurement grid, model download or quality campaign.
The result supports only this hardware, stack, shapes and synthetic RAM-tier fixture.
