# MBR batch-invariance: GPU digest series (ticket 06)

Date 2026-11-19. Rule written before the run: `benchmarks/mbr-invariance-f1-rule-20261119.md` (commit 2c02ab14).
Host: vast.ai instance 61340345 (offer 49992718, Utah, Threadripper PRO 3945WX), RTX 3060 12 GB,
torch 2.14.1+cu130, driver 615.71.09. Moonlight-16B-A3B, bf16, window 8 x 64,
`--phase-profile-only --phase-profile-blocks 0 --timed-steps 2`, one bridge call per run
(`remote-series.sh`, recipes in `.scratch/mbr-batch-invariance/recipes/`). Data: `<run>/result_phase_profile_only.json`,
`summary.json`, `environment.txt`, `pip-freeze.txt`.

Configurations: complete = `--expert-transfer packed --expert-host-cache-mib 12288`; B adds
`--expert-prefetch` and `--mbr-group-size 8`; invariant = `--expert-gemm invariant`; O0 = `--expert-transfer sync`.

| Run | grad_sha256 (first 12) | loss | step 2, s | grad_norm |
|---|---|---|---|---|
| O0 | d0a663ace110 | 5.668856 | 56.3 | 0.786238 |
| A0 | d0a663ace110 | 5.668856 | 21.3 | 0.786238 |
| B0 | 23a4369d4773 | 5.667779 | 5.9 | 0.775628 |
| A1 | c9ec9f12c5c5 | 5.671228 | 38.5 | 0.787242 |
| B1 | 4b78e5193233 | 5.671228 | 23.7 | 0.787305 |
| B1 group 1 | c9ec9f12c5c5 | 5.671228 | 37.8 | 0.787242 |
| B1 group 2 | e0d01aaacc02 | 5.671228 | 30.4 | 0.787152 |
| B1 group 4 | a6aa1199c12e | 5.671228 | 26.1 | 0.787376 |
| B1s (sync, no levers) | 4b78e5193233 | 5.671228 | 29.9 | 0.787305 |
| A1 repeat, B1 repeat | same as A1, same as B1 | | | |

Both timed steps of every run have the same digest. Step times mix disk state and are not a speed verdict (F1 is not run).

## Outcome

- O0 = A0: complete is bit-equal to ordinary on Moonlight (safety check passed).
- S0: A0 != B0, loss differs by 1.1e-3 (PR #72 divergence reproduced).
- **S1: A1 != B1 (red).** The loss is bit-equal (5.671227931976318), the LoRA gradients are not (grad_norm 0.787242 vs 0.787305).
- S2: B1 with group 1 = A1 bit for bit; groups 2, 4, 8 are three different digests, none equal to A1.
- S3: A1 and B1 each repeat bit for bit, so the card is deterministic; the S1 difference is not noise.
- B1s = B1: removing packed, prefetch and host cache changes nothing, so the offload levers are not the cause.
- By the outcome table: S1 red, B1s != A1, so the cause is MBR or another operator that depends on the group size. Next step is a diagnosis (`/diagnosing-bugs`).
- Not collected: rel-L2, cosine and route-change counts for the reference pairs. The bridge stores digests, not gradient tensors. Available reference: losses and grad_norm above.

Limits: one model, one card, one host, one torch version.
