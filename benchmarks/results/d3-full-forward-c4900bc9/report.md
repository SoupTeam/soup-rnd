# D3 first whole-model forward gate: incomplete

Real source commit: `c4900bc9d82e2d908aaf36bd72c4bcfc05de4ca7`.
The operator committed and published the source/protocol before this session.
The clean server clone verified its actual HEAD and all pinned source blobs.

| Format | Mode | Result | F1 |
|---|---|---|---|
| BF16 | native resident | OK | OK |
| BF16 | sequential streamed | OK | OK |
| BF16 | grouped resident | OK | OK |
| BF16 | grouped streamed | OK | OK |
| NF4 | native resident | OK | OK |
| NF4 | sequential streamed | FAIL: ineffective negative | OK |
| NF4 | grouped resident / streamed | unrun | unrun |
| nested NF4 | all four modes | unrun | unrun |

The NF4 sequential cell reached the positive equality/A-A checks, then changing
one packed byte in an active down projection did not change its whole-model
logits. This control failure stops acceptance. It remains FAIL; no replacement
was attempted and the five earlier passes do not establish a complete gate.

`raw-records.zip` preserves every attempted cell result/log, the incomplete
parent session, attempt marker, launch record, and all frozen source bytes.
Its oracle retention receipt audits both full native tensors against their
reported tensor hashes and retains the complete archive manifest. Both full
tensors are preserved separately in the 84,609,463-byte complete raw archive
`d3-full-failed-c4900bc9.zip`, SHA-256
`dd3a7a13ff02d523e5c40d69a87cc150a80df1c9a818ed44453a61e6f044e0bf`.
That full bundle is retained on the instance and was downloaded and verified outside Git (CRC and all 39 manifest entries).
The records ZIP does not include those two large tensors.

A subsequent development-only control revision zeros the whole down projection
of one actually routed expert. Its revised full series requires a new prospective
commit and cannot reuse these five cells. No training, performance, convergence
or task-completion claim follows from this incomplete correctness session.
