"""Audit actual collated H2 work; equality here establishes no numerical/speed verdict."""
from __future__ import annotations

import math
import re
from typing import Any

from h2_update_oracle import tensor_sha, token_counts

HASH = re.compile(r'[0-9a-f]{64}')
COUNT_KEYS = ('processed_positions', 'nonpadding_tokens', 'padding_positions', 'useful_targets')


def batch_evidence(inputs: dict[str, Any]) -> dict:
    """Fingerprint all three collated tensors, outside future timing windows."""
    import torch

    ids, mask, labels = (inputs[key] for key in ('input_ids', 'attention_mask', 'labels'))
    counts = token_counts(mask, labels)
    if ids.shape != labels.shape or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Input IDs must be integer [batch, sequence] matching labels')
    if ids.shape[0] < 1 or ids.shape[1] < 2 or torch.any(ids < 0):
        raise ValueError('Empty/short batch or negative input IDs')
    if torch.any((labels != -100) & (labels != ids)):
        raise ValueError('Supervised causal labels must match input IDs')
    return {'shape': list(ids.shape), 'counts': counts,
            **{key + '_sha256': tensor_sha(inputs[key])
               for key in ('input_ids', 'attention_mask', 'labels')}}


def validate_work(records: dict, *, expected_batches: int, accumulation: int) -> dict:
    """Require complete fixed-accumulation work and observed step boundaries."""
    if (type(expected_batches) is not int or type(accumulation) is not int
            or expected_batches < 1 or accumulation < 1
            or expected_batches % accumulation):
        raise ValueError('Declare positive batches with complete accumulation windows')
    reasons = []
    totals = dict.fromkeys(COUNT_KEYS, 0)
    batches = records.get('batches', [])
    events = records.get('optimizer_events', [])
    gradients = records.get('gradients', [])
    if len(batches) != expected_batches:
        reasons.append('Unexpected number of collated batches')
    if len(gradients) != expected_batches:
        reasons.append('Unexpected number of completed backwards')
    if len(events) != expected_batches // accumulation:
        reasons.append('Unexpected number of optimizer events')
    for index, row in enumerate(batches):
        shape, counts = row.get('shape'), row.get('counts', {})
        valid_shape = (isinstance(shape, list) and len(shape) == 2
                       and all(type(v) is int and v > 0 for v in shape) and shape[1] >= 2)
        valid_counts = (set(counts) == set(COUNT_KEYS)
                        and all(type(v) is int and v >= 0 for v in counts.values()))
        if not valid_shape or not valid_counts:
            reasons.append(f'Malformed batch dimensions/counts at {index}')
            continue
        if (counts['processed_positions'] != math.prod(shape)
                or counts['nonpadding_tokens'] + counts['padding_positions'] != math.prod(shape)
                or counts['useful_targets'] > min(counts['nonpadding_tokens'],
                                                  shape[0] * (shape[1] - 1))
                or counts['useful_targets'] == 0):
            reasons.append(f'Inconsistent or empty supervised work at {index}')
        for key in ('input_ids', 'attention_mask', 'labels'):
            value = row.get(key + '_sha256')
            if not isinstance(value, str) or HASH.fullmatch(value) is None:
                reasons.append(f'Missing {key} fingerprint at {index}')
        loss = row.get('loss')
        if type(loss) not in (int, float) or not math.isfinite(loss):
            reasons.append(f'Missing/nonfinite loss at {index}')
        for key in COUNT_KEYS:
            totals[key] += counts[key]
    for index, event in enumerate(events, 1):
        boundary = event.get('completed_batches')
        if type(boundary) is not int or boundary != index * accumulation:
            reasons.append(f'Incorrect accumulation boundary at optimizer event {index}')
    return {'status': 'FAIL' if reasons else 'PASS_WORK_ACCOUNTING', 'reasons': reasons,
            'totals': totals, 'performance_verdict': 'NO_VERDICT'}


def audit_pair(left: dict, right: dict, *, expected_batches: int, accumulation: int) -> dict:
    """Compare actual ordered batches, ignoring loss values and final adapter differences.

    Does not establish model/quantizer equivalence, numerical tolerance, kernel attribution,
    correct updates or timing validity; those remain separately required gates.
    """
    audits = [validate_work(value, expected_batches=expected_batches, accumulation=accumulation)
              for value in (left, right)]
    reasons = [f'arm {index}: {reason}' for index, audit in enumerate(audits)
               for reason in audit['reasons']]
    for index, (first, second) in enumerate(zip(left.get('batches', []),
                                               right.get('batches', []))):
        for key in ('shape', 'counts', 'input_ids_sha256', 'attention_mask_sha256',
                    'labels_sha256'):
            if first.get(key) != second.get(key):
                reasons.append(f'Ordered batch {index} differs: {key}')
    return {'status': 'FAIL' if reasons else 'PASS_IDENTICAL_COLLATED_WORK',
            'reasons': reasons, 'arms': audits, 'performance_verdict': 'NO_VERDICT'}
