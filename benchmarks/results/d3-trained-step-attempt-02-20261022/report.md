# D3 trained OLMoE attempt 02 — no verdict

Date: 2026-10-22. This is the retained second formal attempt for the trained-step study, not a model result or timing sample.

- Frozen source commit: `23bad80765ae3a02c19887e0ba5c3bc8550d225e`.
- Frozen protocol SHA-256: `2e7a2709eeaaa564223914a88e8d34ebb97056ebd26567c9818bbb714f6738f0` (revision 2).
- Session: `730ef9ec393745068b3c776dd198ddce`.
- Attempted cell: `correctness-dense-native` (first of 84 planned cells).
- The source packet, runtime, GPU, library contents and all three prepared bundles passed the independent preflight checks.
- The cell stopped in `runtime_identity` before model construction: the operator-created Bash wrapper had CRLF line endings, so its `--library` argument included a trailing carriage return and the cuBLAS file identity check raised `FileNotFoundError`.
- Classification: `no_verdict`, as required for an identity-check failure. No model step, F1 monitor or measurement ran; the GPU remained idle. The session stopped immediately and no replacement was made under protocol revision 2.

The archive retains the session/result/log, exact source receipt, wrapper and supervisor configuration. This failed invocation is separate from attempt 01 and does not erase or relabel it. Any next formal series requires a new frozen protocol and commit; the launch wrapper must be created with LF line endings on the Linux host.
