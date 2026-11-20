# D2 fresh-process timing follow-up

This extends the existing timing rule without changing any numerical criterion
or reclassifying the original VOID arms. Scope: synthetic dense Llama-3.1-8B
projection shapes, 128 tokens, FP16/BF16, single/QKV/MLP, both dX modes. This is
not full-model training or a Liger/Unsloth comparison.

Before observing the new run, the protocol is:

- Seed 792; identical frozen state, input and upstream tensors in both arms.
  A separate GPU process first checks the actual case against unpatched PEFT
  and float64, including the original changed-adapter control. Save tensors.
- Five ABBA cycles for each case. Every A/B position uses a fresh process with
  only its measured model resident on CUDA. At least twenty warmups and two
  seconds of warming (the initial short-warmup pilot is retained); at least 100 raw
  forward/backward CUDA-event sample pairs and one second per arm. This changes
  the sampling structure from the old same-process run, not its tolerances.
- The optional Linux CPU-prefork launcher preloads Python libraries in the
  supervisor, verifies that CUDA is uninitialized, and forks a new process per
  arm. All CUDA RNG, contexts, models and tensors are created inside that child.
  The same birth/death evidence remains mandatory. This removes repeated CPU
  imports outside the event regions; it does not reuse GPU contexts or models.
- A CPU-only NVML supervisor records process/clock/power state every 20 ms during
  the arm. It requires empty compute and graphics lists before child creation,
  a single persistent NVML compute PID after CUDA initialization, and empty
  lists after child exit. Child PID, GPU UUID and /proc status are retained.
  This observed birth/death establishes the context identity even when the host
  and container PID namespaces differ. Any ambiguity fails closed. Transient
  samples during initialization are retained for independent inspection too.
- Reject missing telemetry, peers, non-performance power states and hardware/
  thermal throttle flags. GPU-idle/software-power-cap flags are recorded. The
  supervisor's lifecycle result alone does not establish timing acceptance.
- For aggregate timing acceptance, require at least three telemetry samples
  inside each arm's recorded timed interval. Require the observed SM-clock
  max/min spread within each arm and between all four positions of a cycle to
  be no more than 5%. Missing/unstable arms invalidate the entire ABBA cycle;
  preserve all excluded samples. No retrospective threshold tuning.
- Verify the final measured output and every requested gradient outside the
  CUDA event regions. Retain the measured tensors and the expected/oracle
  tensors for independent replay. Dropout is zero, weights do not update, and
  the same deterministic input/upstream is used for every sample.
- Reset peak allocation/reservation after warmup and releasing gradients. Peak
  memory includes one model, input/upstream, activations and adapter gradients;
  correctness/oracle snapshots stay on CPU and are loaded after recording peaks.
  CUDA runtime/driver memory is not included in torch allocated/reserved counts.
- Summarize forward/backward separately; report each cycle's PEFT/fast ratio and
  the median ratio across accepted cycles. A valid no-gain/regression result is
  acceptable. Do not call lifecycle validity or a descriptive ratio a speedup.

The source ZIP, source inventory, raw timings, telemetry, failed attempts,
commands, logs, GPU/software metadata and replay/check results belong in the
handoff. Large reconstructible fixture weight files can stay out of the ZIP;
retain their SHA-256 and freeze the source that generates them. The seed and
fixture-generation code must remain unchanged across a case's arms.
