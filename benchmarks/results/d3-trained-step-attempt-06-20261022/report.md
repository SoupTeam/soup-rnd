# D3 trained OLMoE attempt 06 — no verdict

Formal session under protocol revision 6, frozen commit `8e5a7708602620e0f334e5493e8ef7bbabd68865`.

- Session: `52531df71df74c0a93fdac1f2425877d`.
- Frozen protocol SHA-256: `6e49ebbba4700542ceab7e639e07be33f2bcc29c8d18366ac4a14387d8422ee7`.
- All 18 correctness cells completed and passed their frozen numerical checks. The first timing cell ran three steps, then F1 returned `unknown`; the fixed series stopped at 19 of 84 cells. No timing result is accepted and no replacement was permitted under revision 6.
- The no-verdict came from combining machine-global `/proc/diskstats` with container-visible overlay `loop0`: that arm recorded 21,881,462,784 bytes read by its own process tree, a 21,886,324,736-byte `loop0` delta, and 21,066,153,984 bytes of concurrent host-global reads on `nvme0n1`. The resulting foreign-read interval was 4,861,952–21,071,015,936 bytes, so F1 correctly refused to classify it.
- A separate diagnostic on this unchanged container confirmed that 20 GiB of aligned O_DIRECT reads incremented cgroup-v2 `loop0` by exactly 20 GiB while cgroup-v2 `nvme0n1` stayed unchanged. The revised cgroup-scoped baseline/own/independent-foreign controls returned `ok/ok/void`.
- The diagnostic is not a trained-model result and cannot retroactively validate the stopped session. Host-wide I/O outside the container is not certified.

The compact archive retains the complete session and source packet, all result JSON/logs, and hashes/lengths for the 12 omitted native oracle tensor files. The full 809,432,989-byte supplemental archive remains on the Vast instance at `/workspace/d3-attempt-06-8e5a770-full.zip`, SHA-256 `4573b19ec19724bbe542dab0436645afe3959026825c25d76f6a3f076b3c75af`. The compact archive SHA-256 is recorded in `SHA256SUMS.json`.

The prospective revision 7 switches only the Linux F1 disk counters for this D3 harness to the container's cgroup-v2 `io.stat`, retains the same fixed 84-cell budget and zero replacements, refreshes the F1 controls/setup evidence, and records this stopped attempt. A new commit and clean source export are still required before any new formal cell. No commit was created by this preparation.
