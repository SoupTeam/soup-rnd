# B1.1 CUDA expert-shard verification, 2026-10-20

PR: https://github.com/SoupTeam/soup-rnd/pull/24
B1.1 source commit: `9510096a657b41585a509a35b2a710301f934bef`.

**Result: 2 passed, 0 failed, 0 errors, 0 skipped on RTX 3060.**

- NF4 expert bytes match independent CUDA quantization, nested statistics off: PASS.
- NF4 expert bytes match independent CUDA quantization, nested statistics on: PASS.

Environment: Linux, Python 3.12.13, torch 2.14.1+cu130,
transformers 5.19.0, bitsandbytes 0.50.2,
safetensors 0.8.0; CUDA runtime 13.0, GPU sm_86.

## Provenance and scope

This is a separate publication of the two B1.1 cases from the existing combined
seven-test B1.1/B1.2 synthetic preflight, not a new execution. The original
unmodified log, JUnit XML, environment report and complete source manifest remain
under `original/`. `b1.1-junit-extract.xml` is explicitly a derived two-case extract.
The five B1.2 passes in the original log are outside this B1.1 verdict.

The original run used an uncommitted B1.2 working tree based on the commit above.
All 598 bundled B1.1 source/test/config files were compared byte-for-byte
with its Git blobs. Every one of the original 606 source hashes
and the original archive CRC were verified. This does not recast the execution
as a clean-checkout run or as full repository / full CUDA / trained-model validation.
The reviewer-requested interrupted-conversion and refusal-order CPU regressions
are separate from these two CUDA codec tests.

Original archive SHA-256:
`8c93ded0a54173c5b5ec3b40cb164249e8506cf6a72aeb8b9c8a834f94457b45`.

## Contents and replay

The paths in this section are inside [`cuda-verification.zip`](cuda-verification.zip).
The manifest next to this report covers the published files; the archive contains
its own complete file manifest.

`summary.json` records the test cases, environment and source identities.
`SHA256SUMS.json` covers every file in this bundle except itself.
`source/` contains the exact B1.1 source bytes matching the commit.

To rerun these two tests with the recorded dependencies in a CUDA environment:

```bash
cd source
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. python -m pytest -o addopts= tests/test_b1_expert_shard.py -m gpu -v
```

No rerun was performed while preparing this publication.
