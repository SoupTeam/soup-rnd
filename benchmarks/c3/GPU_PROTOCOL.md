# C3 synthetic CUDA pilot, attempt 001

This is a bounded engineering pilot, not pretrained-model quality evidence or
full C3/A4 acceptance. Freeze the source manifest and this protocol before renting.
No protocol/source repair or retry inside the paid attempt.

- One Linux NVIDIA GPU, initially idle and then attributed through exclusive
  local context creation. Preserve boot ID, GPU UUID, driver, package versions,
  mount/device inventory and all stdout/stderr. Unknown ownership blocks compute.
- Run the two marked C3 CUDA controls first, with deterministic algorithms,
  CUBLAS_WORKSPACE_CONFIG=:4096:8 and TF32 disabled. Require two passes, no skips.
  These compare tiny streamed training and exercise the shared activation budget.
- For the workload generate a random dense Llama: 8 layers, hidden 1024,
  intermediate 4096, heads 8, KV heads 2, vocabulary 256, context limit 256;
  weight seed 7. FP32 storage/training; LoRA q/v, rank4, alpha8, dropout0,
  adapter seed3; AdamW lr0.001, remaining defaults pinned by the runtime lock.
- A: existing disk source. B: partial RAM source, 512MiB envelope, 64MiB fixed
  reserve, 128MiB resident allocation cap. Both read_ahead=1, two device buffers,
  pinned host storage, exactly the same saved source weights and input order.
  B must select at least one and fewer than eight decoder layers. Buffered I/O
  fallback is refused. Physical NVMe identity must still be reviewed separately.
- Five ABBA blocks, each arm a fresh model/optimizer/source. Six optimizer steps
  per arm, first two warmup. Batch1, sequence128, IDs=(arange(128)+step)%256,
  labels equal IDs. No sample or arm replacement; stop on any runtime/numerical
  failure. Each gradient and final adapter is hashed in sorted-name order;
  all arm losses, gradient hashes and final adapter hashes must match exactly.
- Read accounting instruments successful readinto return bytes and calls in the
  async reader. Headers and initial RAM fill are excluded and disclosed. Per-file
  records and per-step deltas are retained. These are application range-read bytes,
  NOT physical device sectors. Do not force the theoretical twice-resident-byte
  saving: boundary reuse, alignment and readahead can change observed totals.
- Record source/setup time separately. Step timing includes forward, backward,
  optimizer and synchronizations; gradient hashing is outside that timing.
  Preserve per-arm peak CUDA allocated/reserved bytes and Linux lifetime peak RSS;
  the latter is not an isolated per-arm host-memory peak.
  F1 ArmWatch defaults retain suspend/power/foreign-read outcomes and GPU clocks.
  Unknown F1, unmapped physical disks, incomplete controls or unresolved clocks
  prohibit a performance conclusion. No automatic conversion of numerical PASS
  to speedup. Apply the existing F1 repeat rule to the five-block pilot; any further
  blocks need a separately reviewed attempt, within remaining authority.
- Local CPU smoke and stub controls are development evidence only, never CUDA
  results. The shared-budget activation bridge is RAM-only and not integrated
  into A4's spill scheduler/public L2L trainer; its acceptance remains pending.

Budget: user authorized at most USD5 for C3 including all setup/storage/traffic.
One attempt: startup600s, setup1800s, CUDA controls600s, pilot900s. Paid time cap2h
including preservation, priced at <=USD0.65/h total plus <=USD10/TB traffic.
Reserve USD1 cleanup plus 50GB traffic. Conservative upper bound USD2.80; never
use the rest to silently restart failed arms. Preserve logs/results/source manifest
locally with SHA before cleanup. Stop paid compute at completion or failure;
Destroy requires the concrete resource confirmation mandated by browser tooling.
No changes to G3/H2/E2/G6, no messages, commits, pushes or PR creation.
