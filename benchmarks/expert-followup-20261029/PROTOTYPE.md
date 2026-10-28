# Streaming follow-up: prepacked RAM, usage placement and later lookahead

Experimental opt-in APIs on top of the qualified `f5670242` packed-transfer
prototype. No CLI/schema/default change, GPU result or automatic full-memory
placement claim follows from this source preparation.

* `configure_transfer("packed")`, then `configure_host_packing(True)` stores a
  static RAM entry as one owned aligned byte arena with typed views. The complete
  arena, including zero padding, counts toward the existing RAM ceiling. Hits
  copy one CPU block into the bounded pinned staging slot rather than assembling
  its components again. The cache itself remains pageable; there is no second
  uncharged cache or whole-RAM pinning. Invalid/LRU/synchronous combinations fail.
* `configure_usage_profile(ExpertUsageProfile())` observes actual logical fetches
  including both cache tiers, never speculative lookahead. Freeze observations
  after setup and disable the observer during accuracy and unprofiled timing.
  `profile.plan(gpu_bytes).keys` ranks frequency times stored-transfer bytes per
  dense-GPU byte. This is a deterministic greedy traffic proxy, not measured
  latency, an optimal knapsack solver or a prediction of future routing.
* `configure_placement(keys)` clears old caches and restricts dense GPU admission
  to frozen `(realpath(shard), expert_index)` keys. `None` retains first-fit;
  an empty plan admits no GPU entries. Whole-expert budgets and source identities
  still apply. Host/GPU entries remain disjoint.
* `configure_transfer("prefetch_after")` enqueues the current MLP and accumulation
  before preparing the next expert. The prior `prefetch` schedule remains an
  explicit control. Neither schedule prefetches during two-read replay/backward.
  Copy-ready/post-consumer fences and immutable decoded-weight ownership remain.

Next qualification: fresh packed baseline, each feature separately, a separate
3-GiB GPU-cache arm, and the combination on physical 8-GB VRAM/native <=32-GiB RAM.
Profile the current baseline first; qualify every arm's full two-update/four-micro
trajectory under original tolerances before speed/memory. Different cache formats
and plans may change hit counts; exact accounting and capacity gates remain.
