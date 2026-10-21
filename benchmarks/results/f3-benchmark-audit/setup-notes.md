# F3 laptop setup notes (ticket 07)

Labels: `"RTX 3050 Laptop, control"` and `"RTX 3050 Laptop, rehearsal"`. No number here enters a verdict.
Machine: RTX 3050 Laptop 4 GB, driver 595.99.02, power limit 80 W (checked with
`f3_setup.py power`, output "power limit ok"). The idle SM clock of 210 MHz is the idle state, not a cap.

## Installed versions

Both venvs: Python 3.12.15 through `uv`, installed with `--no-deps` from `benchmarks/harness/f3_locks/`.
Locks were resolved with `--exclude-newer 2026-01-07`, the day after the preprint.

| Venv | Package | Version |
|---|---|---|
| `~/.venvs/f3-unsloth` | unsloth / unsloth-zoo | 2026.1.2 / 2026.1.2 |
| | torch / triton | 2.9.1 (cu128 wheels) / 3.5.1 |
| | transformers / trl / peft | 4.57.3 / 0.24.0 / 0.18.0 |
| | accelerate / datasets | 1.12.0 / 4.3.0 |
| `~/.venvs/f3-chronicals` | torch / transformers / peft | 2.9.1 / 4.57.3 / 0.18.0 |
| | accelerate / datasets | 1.12.0 / 4.4.2 |

`f3_setup.py verify` on both venvs: no `cut-cross-entropy`, no `xformers`. Unsloth reports
"Xformers = None" and trains. Chronicals code is fetched at
`9b33605177a98bc798387c1742b9d36e53ea9712` into `~/f3-foreign/chronicals` (outside git) and imports
from there. The `chronicals` pip package is not installed.

## Dataset

`yahma/alpaca-cleaned`, revision `12567cabf869d7c92e573c7c783905fc160e9639`, 51,760 examples,
text = instruction, optional input, output, truncated to 512 tokens. sha256 is over the token ids as
compact JSON. The tokens stay in `~/f3-data/`, not in git.

| Tokenizer | sha256 |
|---|---|
| `Qwen/Qwen2.5-0.5B` (real arms) | `f968d82c9244011d9c36d88cad0a00d8c8c1f8e69f35e53c900d89af390afed3` |
| `HuggingFaceTB/SmolLM2-135M` (rehearsal) | `9d2b3e1fc4f1343559c2becc935944fc8a563c4651aa70203abc0ab3338685c1` |

## Control and rehearsal

- Control, 12 steps, `weight_decay` 0, SmolLM2-135M on GPU: rejected in the native trainer and in the
  Unsloth venv. Both `grad_norm` and `parameters_changed` refused in both paths.
  Raw reports: `control/`.
- Rehearsal: arm A (Unsloth, real labels) is `pass` in two fresh processes, arm B (Chronicals) is
  `out of scope` (custom loop). Lifecycle ran with a local folder for upload and a stub for destroy.
  Raw reports: `rehearsal/`. The speed blocks did not run because B has no validity verdict.

## Inferred claims from `research.md`

| Claim | Result |
|---|---|
| Current Unsloth `2026.10.6` does not fit transformers 5.19.0 | Not tested. The 2026.1.2 lock resolved with transformers 4.57.3, as `research.md` predicts for the old release. |
| Unsloth 2026.1.2 and Chronicals need separate venvs | Both resolved and installed separately without conflict. Whether one venv would work was not tried. |
| All-masked labels give `grad_norm` 0.0 | Confirmed on the real model in both paths. Loss logged NaN in the last control run I read (console log, path not recorded), while the tiny CPU test of ticket 04 logged 0.0. The refusal does not depend on loss. |
| `unsloth-zoo` and `unsloth` need Cut Cross-Entropy and xFormers | Both are declared dependencies in the resolution. Unsloth 2026.1.2 imports and trains without them. |
| Memory of batch 16 at 512 | Not tested. The laptop has 4 GB. Full fine-tuning of SmolLM2-135M at batch 4 and length 512 ran out of memory; batch 1 fits. |

## Findings that change the rental

1. The collector hashes trainable parameters with `.numpy()`, which fails on bf16 weights
   (`TypeError: Got unsupported ScalarType BFloat16`). `src/soup_cli/bench/` stays unchanged, so
   `f3_arm.py` loads float32 weights and uses bf16 autocast. If a published arm holds bf16 weights,
   that is a changed condition and goes in the conditions table.
2. The Chronicals arm is out of scope under the spec (custom loop), so the speed row of C1 will not run.
3. The `vastai destroy` path with the per-instance key is still unverified. It needs a rental.
