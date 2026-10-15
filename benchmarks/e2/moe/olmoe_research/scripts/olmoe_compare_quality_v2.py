#!/usr/bin/env python3
"""Compare SAME-K deterministic OLMoE v2 Soup-preprocessed quality reports.

A parity comparison, NOT a quality acceptance decision or a K-vs-K comparison.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--cached', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Refusing to replace an existing comparison')
    a = json.loads(args.baseline.read_text())
    b = json.loads(args.cached.read_text())
    for key in ('kind', 'k', 'frozen_layers', 'model_id', 'revision', 'precision', 'data'):
        if a[key] != b[key]:
            raise RuntimeError(f'Comparison configuration mismatch: {key}')
    if a['kind'] != 'olmoe_pretrained_soup_preprocessing_linear_quality_v2':
        raise RuntimeError('Not an OLMoE v2 report')
    if a['mode'] != 'baseline' or b['mode'] != 'cached':
        raise RuntimeError('Expected baseline then cached report')
    if a['k'] == 16 or a['k'] not in (4, 8, 12):
        raise RuntimeError('K16 has no valid cached comparison')
    if a['adapter']['initial_sha256'] != b['adapter']['initial_sha256']:
        raise RuntimeError('Initial trainable LoRA weights differ')
    if a['adapter']['final_trainable_sha256'] != b['adapter']['final_trainable_sha256']:
        raise RuntimeError('Final trainable LoRA weights differ')
    if a['adapter']['safetensors_sha256'] != b['adapter']['safetensors_sha256']:
        raise RuntimeError('Saved adapter bytes differ')
    for key in ('seed', 'epochs', 'lr', 'scheduler', 'weight_decay',
                'batch_size', 'grad_accumulation', 'optimizer_steps_verified',
                'total_optimizer_steps', 'epoch_training_orders_sha256'):
        if a['training'][key] != b['training'][key]:
            raise RuntimeError(f'Training setup mismatch: {key}')
    steps_a = a['training']['losses']
    steps_b = b['training']['losses']
    if len(steps_a) != len(steps_b):
        raise RuntimeError('Different number of training steps')
    max_diff = 0.0
    for index, (left, right) in enumerate(zip(steps_a, steps_b)):
        for key in ('epoch', 'example_index', 'scaler_before', 'scaler_after', 'lr_used'):
            if left[key] != right[key]:
                raise RuntimeError(f'Training step {index} mismatch: {key}')
        for key in ('total_loss', 'task_loss_estimate', 'aux_loss'):
            delta = abs(left[key] - right[key])
            max_diff = max(delta, max_diff)
            if delta > 2e-3:
                raise RuntimeError(f'Training step {index} loss mismatch: {key}: {delta}')
    if len(a['validation']['per_example']) != len(b['validation']['per_example']):
        raise RuntimeError('Different validation row counts')
    max_val_diff = 0.0
    for index, (left, right) in enumerate(zip(a['validation']['per_example'], b['validation']['per_example'])):
        for key in ('total_loss', 'task_loss_estimate', 'aux_loss'):
            delta = abs(left[key] - right[key])
            max_val_diff = max(max_val_diff, delta)
            if delta > 2e-3:
                raise RuntimeError(f'Validation row {index} mismatch: {key}: {delta}')
    rows = a['data']['training_subset_rows']
    epochs = a['training']['epochs']
    expected_hits = rows * (epochs - 1)
    if b['cache'] != {
        'enabled': True,
        'hits': expected_hits,
        'misses': rows,
        'skipped_frozen_layer_calls': expected_hits * b['frozen_layers'],
        'full_sha256_recomputes': 1,
    }:
        raise RuntimeError(f'Invalid cache counters: {b["cache"]}')
    if a['cache']['enabled'] or a['cache']['hits'] or a['cache']['misses']:
        raise RuntimeError('Baseline unexpectedly used a cache')
    result = {
        'kind': 'olmoe_soup_preprocessed_linear_same_k_comparison_v2',
        'k': a['k'],
        'max_train_loss_abs_diff': max_diff,
        'max_validation_loss_abs_diff': max_val_diff,
        'initial_adapter_exact': True,
        'final_adapter_exact': True,
        'adapter_file_exact': True,
        'cache': b['cache'],
        'verdict': 'TECHNICAL_SMOKE_PARITY_PASS_ONLY',
        'research_acceptance': 'NO VERDICT',
        'warning': 'Soup preprocessing + linear LR, but independent HF Trainer sampler equivalence and acceptance not established',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('OLMOE V2 SAME-K QUALITY PARITY PASSED')


if __name__ == '__main__':
    main()
