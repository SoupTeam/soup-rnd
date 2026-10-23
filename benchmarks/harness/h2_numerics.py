"""Prospective paired loss/gradient checks; never a throughput or quality verdict."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from h2_update_oracle import snapshot, tensor_sha
from h2_work_equivalence import batch_evidence
from rich.console import Console

POLICY_KEYS = {'schema', 'loss_atol', 'loss_rtol', 'gradient_atol', 'gradient_rtol',
               'gradient_relative_l2_max', 'gradient_cosine_min'}


def validate_policy(policy: dict) -> None:
    if set(policy) != POLICY_KEYS or policy['schema'] != 'h2-numerics-v1':
        raise ValueError('Unknown numerical policy')
    for key in POLICY_KEYS - {'schema'}:
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Policy limits must be finite nonnegative numbers')
    if not 0 < policy['gradient_cosine_min'] <= 1:
        raise ValueError('Cosine threshold must lie in (0,1]')


def collect_probe(model: Any, batch: dict, expected_names: list[str], out: Path,
                  *, bf16: bool) -> dict:
    """Train-mode model loss/backward, no optimizer; restore RNG and clear probe grads.

    This checks model loss kernels, not TRL's accumulation/loss-normalization policy.
    The caller provides a deterministic collated row before creating its optimizer.
    """
    import torch
    from safetensors.torch import save_file

    if out.exists():
        raise FileExistsError('Probe output must be new')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Probe may only run before training gradients exist')
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1:
        raise ValueError('Single-device probe required')
    device = next(iter(devices))
    if bf16 and device.type != 'cuda':
        raise ValueError('Declared bf16 GPU probe requires CUDA')
    if not bf16 and (device.type != 'cpu' or any(p.is_floating_point()
                     and p.dtype is not torch.float32 for p in model.parameters())):
        raise ValueError('Local reference probe requires actual FP32 CPU parameters')
    out.mkdir(parents=True)
    modes = [(module, module.training) for module in model.modules()]
    before = snapshot(model, expected_names)
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            inputs = {k: v.to(device) for k, v in batch.items()}
            work = batch_evidence(inputs)
            if any(not isinstance(v, torch.Tensor) for v in inputs.values()):
                raise ValueError('Only tensor model inputs supported')
            model.train()
            autocast = (torch.autocast('cuda', dtype=torch.bfloat16) if bf16
                        else contextlib.nullcontext())
            with autocast:
                loss = model(**inputs).loss
            scalar = float(loss.detach().item())
            if not math.isfinite(scalar):
                raise ValueError('Nonfinite probe loss')
            loss.backward()
            parameters = dict(model.named_parameters())
            tensors = {'input.' + k: v.detach().cpu().contiguous().clone()
                       for k, v in inputs.items()}
            for name in expected_names:
                grad = parameters[name].grad
                if grad is None or not torch.isfinite(grad).all().item():
                    raise ValueError('Missing/nonfinite gradient: ' + name)
                tensors['gradient.' + name] = grad.detach().cpu().contiguous().clone()
        model.zero_grad(set_to_none=True)
        after = snapshot(model, expected_names)
        if before != after:
            raise ValueError('Probe changed parameter/buffer/quantizer state without an optimizer')
        save_file(tensors, str(out / 'probe.safetensors'))
        result = {'schema': 'h2-model-probe-v1', 'loss': scalar, 'work': work, 'before': before,
                  'tensor_sha256': {k: tensor_sha(v) for k, v in tensors.items()},
                  'scope': 'One collated row, train-mode model loss/backward, no optimizer',
                  'compute_policy': 'bf16-autocast-cuda' if bf16 else 'fp32-cpu-test',
                  'performance_verdict': 'NO_VERDICT'}
        (out / 'probe.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
        (out / 'files_sha256.json').write_text(json.dumps(pins, indent=2) + '\n')
        return result
    except Exception as error:
        (out / 'failure.json').write_text(json.dumps({'error': type(error).__name__ + ': '
                                                     + str(error), 'verdict': 'NO_VERDICT'}) + '\n')
        raise
    finally:
        model.zero_grad(set_to_none=True)
        for module, training in modes:
            module.training = training


def read_probe(root: Path) -> tuple[dict, dict]:
    import torch
    from safetensors.torch import load_file

    expected = {'probe.json', 'probe.safetensors', 'files_sha256.json'}
    if root.is_symlink() or {p.name for p in root.iterdir()} != expected:
        raise ValueError('Incomplete or extra probe evidence')
    if any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
        raise ValueError('Probe evidence must contain regular files')
    if (root / 'probe.safetensors').stat().st_size > 512 * 1024**2:
        raise ValueError('Probe tensor size exceeds declared limit')
    pins = json.loads((root / 'files_sha256.json').read_text())
    if set(pins) != expected - {'files_sha256.json'}:
        raise ValueError('Invalid probe file manifest')
    for name, digest in pins.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Probe file SHA mismatch')
    result = json.loads((root / 'probe.json').read_text())
    if (result.get('schema') != 'h2-model-probe-v1' or result.get('compute_policy') not in
            {'fp32-cpu-test', 'bf16-autocast-cuda'} or not result['before']['trainable']):
        raise ValueError('Unknown/empty probe declaration')
    tensors = load_file(str(root / 'probe.safetensors'), device='cpu')
    if set(tensors) != set(result['tensor_sha256']):
        raise ValueError('Probe tensor manifest mismatch')
    for name, tensor in tensors.items():
        if tensor_sha(tensor) != result['tensor_sha256'][name] or not torch.isfinite(tensor).all():
            raise ValueError('Probe tensor SHA/finite check failed')
    work = batch_evidence({k.removeprefix('input.'): v for k, v in tensors.items()
                           if k.startswith('input.')})
    if result['work'] != work:
        raise ValueError('Actual probe work differs from metadata')
    for name, state in result['before']['trainable'].items():
        grad = tensors.get('gradient.' + name)
        if grad is None or list(grad.shape) != state['shape'] or str(grad.dtype) != state['dtype']:
            raise ValueError('Gradient differs from initial adapter shape/dtype: ' + name)
    return result, tensors


def gradient_metrics(reference: Any, candidate: Any) -> dict:
    """Accumulate FP64 metrics in bounded CPU chunks, preserving zero-vector semantics."""
    if (reference.shape != candidate.shape or reference.dtype != candidate.dtype
            or reference.numel() == 0):
        raise ValueError('Gradient shape/dtype mismatch')
    left, right = reference.reshape(-1), candidate.reshape(-1)
    norm, other_norm, delta, dot, max_abs, ref_max = 0., 0., 0., 0., 0., 0.
    for start in range(0, left.numel(), 65536):
        first, second = left[start:start+65536].double(), right[start:start+65536].double()
        if not first.isfinite().all() or not second.isfinite().all():
            raise ValueError('Nonfinite gradient values')
        diff = first - second
        norm += first.square().sum().item()
        other_norm += second.square().sum().item()
        delta += diff.square().sum().item()
        dot += (first * second).sum().item()
        max_abs = max(max_abs, diff.abs().max().item())
        ref_max = max(ref_max, first.abs().max().item())
    if not all(math.isfinite(v) for v in (norm, other_norm, delta, dot, max_abs, ref_max)):
        raise ValueError('Gradient metric overflow')
    return {'max_abs': max_abs, 'reference_max_abs': ref_max,
            'relative_l2': math.sqrt(delta / norm) if norm else (0.0 if not delta else None),
            'cosine': dot / math.sqrt(norm * other_norm) if norm and other_norm
            else (1.0 if not norm and not other_norm else 0.0),
            'reference_zero': norm == 0, 'candidate_zero': other_norm == 0}


def compare_probes(reference_root: Path, candidate_root: Path, policy: dict) -> dict:
    validate_policy(policy)
    reference, left = read_probe(reference_root)
    candidate, right = read_probe(candidate_root)
    reasons = []
    for key in ('schema', 'compute_policy', 'before', 'work'):
        if reference[key] != candidate[key]:
            reasons.append('Probe initial state/work/precision differs: ' + key)
    for key in set(left) | set(right):
        if key.startswith('input.') and (key not in left or key not in right
                                         or not left[key].equal(right[key])):
            reasons.append('Actual model input differs: ' + key)
    names = set(reference['before']['trainable'])
    if (set(left) != set(right) or {k.removeprefix('gradient.') for k in left
                                  if k.startswith('gradient.')} != names):
        reasons.append('Gradient target manifest differs')
    losses = [reference['loss'], candidate['loss']]
    if (any(type(v) not in (float, int) or not math.isfinite(v) for v in losses)
            or abs(losses[0] - losses[1]) > policy['loss_atol']
            + policy['loss_rtol'] * abs(losses[0])):
        reasons.append('Loss differs beyond prospective tolerance')
    metrics = {}
    for name in sorted(names):
        key = 'gradient.' + name
        if key not in left or key not in right:
            continue
        try:
            row = gradient_metrics(left[key], right[key])
        except ValueError as error:
            reasons.append(str(error) + ': ' + name)
            continue
        metrics[name] = row
        if (row['relative_l2'] is None
                or row['relative_l2'] > policy['gradient_relative_l2_max']
                or row['cosine'] < policy['gradient_cosine_min']
                or row['max_abs'] > policy['gradient_atol']
                + policy['gradient_rtol'] * row['reference_max_abs']):
            reasons.append('Gradient differs beyond prospective tolerance: ' + name)
    if not any(not row['reference_zero'] for row in metrics.values()):
        reasons.append('No nonzero reference adapter gradients')
    return {'status': 'FAIL' if reasons else 'PASS_MODEL_PROBE_TOLERANCES',
            'reasons': reasons, 'gradient_metrics': metrics, 'policy': policy,
            'performance_verdict': 'NO_VERDICT',
            'scope': 'One-row model loss/gradient parity only; not full-training equivalence'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare_probes(args.reference, args.candidate, json.loads(args.policy.read_text()))
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    Console().print(result['status'] + '; performance=NO_VERDICT')
    if result['status'] != 'PASS_MODEL_PROBE_TOLERANCES':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
