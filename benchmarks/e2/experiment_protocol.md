
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

Compare E2 against the no-cache Top-K LoRA baseline
at the same K/L ratio.

Use the same:
- Initial model weights
- Training and validation datasets
- Random seed
- Optimizer and learning-rate schedule
- Precision and batch size
- Number of optimizer steps

Primary metric: held-out validation loss.

Relative quality degradation:

delta_loss = 100 * (loss_E2 - loss_baseline) / loss_baseline

Proposed acceptance threshold:
delta_loss <= 1.0%

This threshold must be approved before full experiments.

Evaluate K/L = 0.25, 0.50 and 0.75 separately.

K/L = 1.00 is the no-frozen-prefix control.
E2 caching must be disabled for this configuration.

Training loss alone is not sufficient to establish
quality equivalence.

## Performance Evaluation

Use three complete A-B-B-A cycles.

Each run must use a fresh process and the same
hardware configuration.

Record:
- GPU name and UUID
- GPU memory and utilization
- GPU clocks and temperature, where available
- CPU and system memory
- PyTorch, CUDA and Transformers versions
- Model revision and dataset identity
- Configurations and random seeds
- Supervised and total token counts
- Cold-cache construction cost
- Warm-cache step timing
- Second-epoch timing
- End-to-end training runtime

Do not compare performance measurements collected
on different GPU models.

Report speedup as:

speedup = baseline_time / cached_time

Evaluate cold-cache, warm-cache and total runtime
separately.

## Verdict Rules

Report three independent verdicts:
correctness, quality and performance.

### Correctness

PASS:
- Required equivalence and invalidation checks pass.

FAIL:
- Incorrect reuse, stale activations or equivalence
  outside the declared numerical tolerance.

NO VERDICT:
- Required checks are missing or incomplete.

### Quality

PASS:
- Validation loss degradation is within the
  approved threshold.

FAIL:
- Validation loss degradation exceeds the
  approved threshold.

NO VERDICT:
- No comparable validation results, or the
  threshold has not been approved.

### Performance

PASS:
- A repeatable speedup is demonstrated under
  comparable conditions and the predeclared
  performance acceptance rule is met.

FAIL:
- A repeatable slowdown is established under
  comparable conditions.

NO VERDICT:
- Results are inconsistent, incomplete, or
  hardware conditions are not comparable.

Do not classify a noisy benchmark as PASS or FAIL.

### Overall

Overall PASS requires all mandatory criteria
to pass.

If any mandatory criterion fails, overall FAIL.

Otherwise, overall NO VERDICT.

K/L = 1.00 is a control, not an E2 speedup claim.

