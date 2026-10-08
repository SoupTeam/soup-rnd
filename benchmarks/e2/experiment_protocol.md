
# E2 Experimental Protocol

## Objective

Evaluate frozen-prefix activation caching for Top-K LoRA
fine-tuning against a no-cache baseline.

## Models

- Mistral-7B
- Small MoE (architecture and checkpoint to be selected)

## K/L Ratios

- 0.25
- 0.50
- 0.75
- 1.00 (control: no frozen prefix)

K is the number of trainable upper decoder layers.
L is the total number of decoder layers.

## Benchmark Design

- A = Top-K LoRA without activation caching
- B = Same Top-K LoRA with E2 caching
- Run order: A-B-B-A
- Each run uses a fresh process
- Repeat at least 3 cycles
- Keep model, dataset, seed, batch size, precision,
  optimizer and sequence lengths fixed
- Use the same GPU type for every A/B comparison
- Record GPU, CPU, RAM, software versions and timestamps

## Cache Evaluation

Measure separately:

1. Cold-cache construction cost
2. Warm-cache training performance
3. Second-epoch performance
4. Cache hit/miss counts
5. Total training runtime

## Quality Evaluation

Compare validation loss on the same held-out dataset.

Proposed quality tolerance:
- Relative validation-loss degradation <= 1%

This tolerance is provisional and requires approval
before the full experiments.

## Correctness

- Compare baseline and cached outputs
- Compare LoRA parameters with defined numerical tolerance
- Verify input, mask and position-dependent cache keys
- Verify frozen-weight and configuration invalidation
- Reject nondeterministic frozen prefixes

## Verdict

PASS:
- Correctness and quality checks pass
- Performance improvement is repeatable

FAIL:
- Correctness or quality checks fail
- Or a repeatable slowdown is established for the
  performance claim

NO VERDICT:
- Insufficient repetitions
- Inconsistent machine conditions
- Incomplete measurements
- Unsupported architecture or insufficient GPU memory

Report correctness, quality and speed verdicts separately.
Do not turn an inconclusive speed result into a failure
of correctness.

K/L = 1 is a no-cache control, not an E2 speedup claim.

