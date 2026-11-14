"""CPU-testable H2 evidence helpers; no model loading, training launch or speed verdict."""
from __future__ import annotations

import hashlib
import math
from typing import Any


def token_counts(attention_mask: Any, labels: Any) -> dict[str, int]:
    """Count decoder-only predicted targets after the causal shift, once per batch."""
    import torch

    if attention_mask.ndim != 2 or labels.shape != attention_mask.shape:
        raise ValueError('Expected matching [batch, sequence] masks and labels')
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError('Labels must be integer token IDs or -100')
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError('Attention mask must be binary')
    if torch.any((attention_mask == 0) & (labels != -100)):
        raise ValueError('Padding labels must be ignored')
    if torch.any((labels < 0) & (labels != -100)):
        raise ValueError('Unexpected negative label')
    processed = labels.numel()
    nonpadding = int(attention_mask.sum().item())
    return {'processed_positions': processed, 'nonpadding_tokens': nonpadding,
            'padding_positions': processed - nonpadding,
            'useful_targets': int((labels[:, 1:] != -100).sum().item())}


def tensor_sha(tensor: Any) -> str:
    """Hash exact dtype/shape/bytes with bounded CPU staging, outside timed regions."""
    import torch

    if not tensor.is_contiguous():
        raise ValueError('Noncontiguous tensor requires a separately declared hashing policy')
    digest = hashlib.sha256(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), 1024 * 1024):
        chunk = flat[start:start + 1024 * 1024].to('cpu').contiguous().view(torch.uint8)
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def snapshot(model: Any, expected_trainable_names: list[str]) -> dict:
    """Require exact adapter names; fingerprint frozen parameters AND registered buffers."""
    import torch
    from h2_quant_state import quantizer_snapshot

    expected = set(expected_trainable_names)
    parameters = dict(model.named_parameters())
    actual = {name for name, param in parameters.items() if param.requires_grad}
    if not expected or len(expected) != len(expected_trainable_names) or actual != expected:
        raise ValueError('Trainable names differ from the declared adapter manifest')
    trainable = {name: {'shape': list(parameters[name].shape),
                        'dtype': str(parameters[name].dtype),
                        'finite': bool(torch.isfinite(parameters[name]).all().item()),
                        'sha256': tensor_sha(parameters[name])} for name in sorted(expected)}
    frozen = {'parameter:' + name: tensor_sha(param) for name, param in parameters.items()
              if name not in expected}
    frozen.update({'buffer:' + name: tensor_sha(value) for name, value in model.named_buffers()})
    return {'trainable': trainable, 'frozen': frozen,
            'quantizer': quantizer_snapshot(model)}


def gradients(model: Any, expected_trainable_names: list[str]) -> dict:
    """Observe before clipping/zero_grad; initially zero LoRA-A gradients are allowed."""
    import torch

    parameters = dict(model.named_parameters())
    actual = {name for name, param in parameters.items() if param.requires_grad}
    if actual != set(expected_trainable_names) or not actual:
        raise ValueError('Gradient manifest differs')
    finite, nonzero = True, []
    for name in sorted(actual):
        value = parameters[name].grad
        if value is None:
            continue
        finite = finite and bool(torch.isfinite(value).all().item())
        if bool(torch.any(value != 0).item()):
            nonzero.append(name)
    return {'finite': finite, 'nonzero_names': nonzero}


def audit_update(before: dict, after: dict, gradient_rows: list[dict],
                 optimizer_steps: int, losses: list[float]) -> dict:
    """Check update evidence; self-reported step counts alone do not establish training."""
    reasons = []
    if type(optimizer_steps) is not int or optimizer_steps < 1:
        reasons.append('No positive integer optimizer-step count')
    if not losses or not all(type(v) in (float, int) and math.isfinite(v) for v in losses):
        reasons.append('Missing or nonfinite losses')
    names = set(before.get('trainable', {}))
    if not names or names != set(after.get('trainable', {})):
        reasons.append('Adapter manifest changed or empty')
    changed = []
    for name in sorted(names & set(after.get('trainable', {}))):
        left, right = before['trainable'][name], after['trainable'][name]
        if left.get('finite') is not True or right.get('finite') is not True:
            reasons.append('Nonfinite adapter state: ' + name)
        if left['shape'] != right['shape'] or left['dtype'] != right['dtype']:
            reasons.append('Adapter shape/dtype changed: ' + name)
        if left['sha256'] != right['sha256']:
            changed.append(name)
    if not changed:
        reasons.append('No adapter bytes changed')
    if before.get('frozen') != after.get('frozen'):
        reasons.append('Frozen parameter/buffer state changed')
    if before.get('quantizer') != after.get('quantizer'):
        reasons.append('NF4 quantizer state changed')
    if (not gradient_rows or any(row.get('finite') is not True for row in gradient_rows)
            or not any(set(row.get('nonzero_names', [])) & names for row in gradient_rows)):
        reasons.append('Missing finite nonzero adapter-gradient evidence')
    return {'status': 'FAIL' if reasons else 'PASS_UPDATE_EVIDENCE',
            'reasons': reasons, 'changed_adapters': changed, 'performance_verdict': 'NO_VERDICT',
            'scope': 'Helper evidence only; requires trusted runner, per-step records and audit.'}
