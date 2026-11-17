# D3 RTX 4090 VM preparation â€” no trained-model comparison

This record prepares prospective trained-step protocol revision 8. It contains
instrument controls and synthetic regressions, not a complete OLMoE optimizer
step, model throughput, quality evidence, or final D3 acceptance.

## Environment and immutable inputs

- Vast instance 57244283, machine 55267; KVM guest, ext4 on `/dev/vda1`.
- RTX 4090, 24 GB class; GPU UUID
  `GPU-20b47f62-2dce-6790-3477-7a6b76093e15`, guest driver 580.95.05.
- Python 3.12.15, torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0,
  transformers 5.19.0, bitsandbytes 0.50.2, safetensors 0.8.0.
- Pinned OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
  All four checkpoint hashes match the prospective protocol. The dense, NF4
  and NF4-DQ bundle hashes reproduce the previous preparation identities;
  see `prepared-weights.json`. Preparing/reading weights did not execute a
  pretrained model forward or backward.

## F1 controls

The unchanged default F1 sensors observe all guest-visible whole disks through
`/proc/diskstats`; they do not observe or certify the underlying physical host.
The new mapping explicitly selects this counter family and rejects overlay,
loop, missing, reset, noisy or ambiguous readings. There is no silent fallback
from an unavailable sensor and no forced zero for other disks.

| Control | Expected | Observed | Read coverage |
|---|---|---|---|
| Synthetic fixture baseline | OK | OK | No application reads |
| Synthetic fixture own reads | OK | OK | All 16 GiB read twice; exact 32 GiB disk delta, zero other-disk delta |
| Independent fixture reader | VOID | VOID | Separate systemd service reads 4 GiB |
| Actual-shard baseline | OK | OK | All 48 immutable shards mapped beforehand |
| Actual-shard own reads | OK | OK | 19,833,356,288 bytes across all dense/NF4/NF4-DQ expert files |
| Independent actual-shard reader | VOID | VOID | Separate systemd service, not a descendant of the watched process |

The complete-shard control reads each file from beginning to its final aligned
4 KiB extent, omitting only an unaligned tail of less than 4 KiB. It does not
repeat only the first MiB. The mapping separately retains two exact direct-read
counter observations per shard. Raw counters, file identities, PIDs/parents,
and all F1 check records remain in `preparation-evidence.zip` and the extracted
mapping/control JSONs.

Revision 7 at `6f9252a9` never created a formal session. Its full-shard preflight
still duplicated overlay-loop and backing-device reads under cgroup v2. Preserve
that diagnostic, protocol and setup in the three `superseded-revision-07-*`
files. This is not a seventh failed formal model attempt and cannot validate
the earlier no-verdict attempt 06 retrospectively.

## Synthetic CUDA failure and explicit corrected profile

The first synthetic suite produced 326 passes, one root-permission skip, and
two failures in the complete-row GEMM regression. An exhaustive diagnostic
then found native-versus-grouped differences for rows 481â€“512, for both tested
OLMoE projection geometries and seeds 11/29. Its original outputs and sources
are retained; the failed suite is not relabelled as passing.

The optional `native_olmoe_rows_from=481` profile delegates these large OLMoE
matrices to native `torch.mm`, alongside the existing native rows 1â€“128.
Rows 129â€“480 still execute grouped GEMM. Tiny-control dispatch and the default
disabled upper boundary remain unchanged. Allocation planning accounts for
which calls actually need grouped pointer arrays. Both activation timing arms
must use this same explicit profile, and all native-oracle equality checks and
positive grouped-dispatch requirements remain intact.

The corrected preliminary suite passed 335 tests with one root-permission
skip, including every row count 0â€“512 for both geometries/seeds. A separate
24-test CPU check covers guest-counter rejection and the trained-step analyzer.
The final selected suite passed **346 tests**, with the same one root-permission
skip and zero failures/errors. Selected-source Ruff passed. Its receipt/logs
cover the final source snapshot and prospective setup pins and are published
separately in this directory.
No full repository suite or general cross-architecture numerical guarantee is
claimed. The skipped unreaped-child permission test requires non-root Linux;
its skip does not remove any required check from the actual F1 controls.

## Prospective analysis and next execution boundary

The analyzer now averages both A arms and both B arms of every ABBA block.
A regression with equal underlying costs plus linear drift requires a zero
contrast; the earlier last-arm overwrite would not satisfy that control.

Revision 8 retains 18 correctness cells, 60 timed cells and six profiles,
one fresh process per cell, one session per committed source and zero
replacements. Commit the prepared source, protocol and setup together, export
the clean Git blobs, and verify that packet before creating the formal series.
A mismatch, OOM, interruption, identity change or unknown/void F1 must stop it.
No formal revision-8 session was started during this preparation. The six prior
formal attempts remain separate, with their original exclusions and scopes.
