# B6 review follow-up, 2026-10-29

The CI regression is fixed and the user approved a prospective eight-fragment
WikiText-2 evaluation. The new RAM-constrained giant measurement is **blocked**:
no follow-up checkpoint conversion, giant training or giant evaluation ran.
The October 28 result remains a separate experiment with
`ram_32gb_hardware_validated: false`; its measured 0.1456235249 tokens/s is unchanged.
This report does not claim that all review acceptance items are satisfied.

## CI correction and local verification

The Windows TOCTOU simulation patched process-wide `os.fstat`, which also affected
pytest file capture. An unrelated real descriptor reproduced
`AttributeError: MockSwappedFstat has no attribute st_size`. The tests now replace
only `soup_cli.utils.paths.os` with a local namespace; production path checks are
unchanged. The regression exercises capture and stats an unrelated descriptor
while the swap mock remains active.

All 15 tests in the affected file pass on Windows/Python 3.10 and in a clean
Linux/Python 3.12 container with pytest 9.1.1. The Linux output and JUnit XML are
retained in [the evidence directory](evidence/b6-20261029/). Ruff passes across
`src/soup_cli/ scripts/ tests/ benchmarks/`. The earlier broader selected
B1/B6/TOCTOU run passed 169 tests with 11 skips. After legacy-controller tests
were added, the final focused acceptance/verdict/TOCTOU run passed 48 tests on
both Windows/Python 3.10 and Linux/Python 3.12 (PyTorch 2.8 CPU, pytest 9.1.1).
The first core-only Linux attempt lacked PyTorch for one causal-loss test;
its setup failure is retained separately, and the complete-dependency repeat
passed all 48 tests.
These are selected runs with coverage disabled, not the full hosted coverage gate.
Local commands set `PYTHONPATH` to this worktree and its `src` directory, avoiding
an unrelated editable installation. The unchanged E4 Unix `resource` imports
prevent collecting the entire suite on this Windows host.

The required hosted Ubuntu/Python 3.12 result must be checked on the final PR
head. Local targeted success does not substitute for that result.

## Approved protocol and implementation

The [frozen protocol](protocols/b6-acceptance-20261029/protocol.json) and
[eight original texts](protocols/b6-acceptance-20261029/evaluation-data.json)
were selected before any follow-up giant loss measurement. The user explicitly
approved comparison of base and adapter loss/perplexity on eight WikiText-2 test
fragments of at most 64 tokens each, reporting any outcome without requiring an
improvement. Dataset provenance and attribution are in the protocol README.

The harness verifies zero initial LoRA B matrices, records per-example token
digests/counts and losses before and after two updates, and computes token-weighted
NLL and perplexity. Exact parameter/logits reload remains a separate requirement.
Negative tests reject changed/missing pairs, a false unadapted baseline, evaluation
on the wrong side of training, missing phases, widened RAM limits, OOM kills and
swap usage. The RAM monitor resolves the current memory controller using mountinfo
and membership on cgroup v1 and v2. RSS on a larger-RAM host cannot pass the gate.

`b6_acceptance_verdict.py` combines the original giant checks with these new raw
quality and RAM records. Its tests are synthetic validation of the checker;
**no real follow-up acceptance verdict exists**. Reviewer/owner approval is external.

## Hardware attempts, stopped before downloading weights

The following exact-owned attempts are retained under
[hardware receipts](evidence/b6-20261029/hardware-attempts/). No persistent volume
was created. Hardware-only probes must not be presented as real model results.

| Attempt | Instance | Outcome |
| --- | --- | --- |
| A | 58145965 | Nominal 32 GB offer; actual cgroup v2 limit 45,538,607,104 bytes. Rejected and deleted. |
| B | 58146695 | Physical-32 GB candidate could not load the image from Docker Hub. Stopped and deleted. |
| C | none | Offer disappeared before creation. |
| D | 58148472 | Legacy cgroup v1 was not understood by the initial v2 probe. Rejected and deleted; monitor support was then corrected. |
| E | 58149518 | V1-aware repeat observed 129,482,358,784 bytes despite requested Docker memory flags. Rejected and deleted. |
| F | 58150314 | Alternate official image registry also timed out on the physical-32 GB host. Stopped and deleted. |
| G | none | VM account-key registration rejected team context; a subsequent VM creation request returned HTTP 400. No instance ID was issued and three independent inventory checks found the unique label absent. |

The VM plan would put all model phases in an owned systemd scope with
`MemoryMax=32G` and `MemorySwapMax=0`, verified before source download. It was not
executed. No claim is made about its actual memory limit or GPU capability.
The final prospective compute rule is native BF16 where supported, otherwise
FP32; this was fixed before any follow-up giant loss, with identical dtype/cache
for both paired arms. It does not change the original BF16 pilot.

## Cost, cleanup and remaining work

The [cumulative ledger](evidence/b6-20261029/billing-cumulative.json) records
**$3.863 total** at its timestamp: $3.707 original work plus $0.156 follow-up
probes, including compute, storage and network. All nine charged B6 instance IDs
and the rejected VM label were independently absent. No paid B6 resource remains.
Billing may receive later provider adjustments.

The $5 ceiling is conservatively treated as cumulative, leaving $1.137 rather
than assuming another $5 allocation. The selected verified VM quote charged about
$2.77 for 710 GB ingress alone, before runtime/storage. A live search found no
8 GB / >=28 GB RAM / >=1300 GiB disk offer with ingress <=$0.0003/GB. The rejected
VM quote and remaining-budget search are retained. Further paid retries were
stopped instead of consuming the remainder without a viable complete run.

Still required: a usable one-GPU 8 GB environment with an actual <=32 GiB memory
constraint, sufficient disk and an affordable checkpoint transfer; then the
approved eight-pair run and exact reload, raw accounting, and reviewer acceptance.
The already trained adapter and all original negative/positive journals remain
available in [the original report](b6-giant-moe-20261028.md).
