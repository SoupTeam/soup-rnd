"""Offline gradient geometry; fitted scales are diagnostic, never acceptance criteria."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from rich.console import Console


def scale_geometry(reference, candidate) -> dict:
    """Fit candidate = scale * reference in FP64, without altering either tensor."""
    if (reference.shape != candidate.shape or reference.dtype != candidate.dtype
            or not reference.is_floating_point() or reference.numel() == 0):
        raise ValueError('Expected matching nonempty floating gradients')
    left, right = reference.reshape(-1), candidate.reshape(-1)
    rr = cc = rc = 0.0
    for start in range(0, left.numel(), 65536):
        a, b = left[start:start + 65536].double(), right[start:start + 65536].double()
        if not a.isfinite().all() or not b.isfinite().all():
            raise ValueError('Nonfinite gradient')
        rr += a.square().sum().item()
        cc += b.square().sum().item()
        rc += (a * b).sum().item()
    if not all(math.isfinite(x) for x in (rr, cc, rc)):
        raise ValueError('Gradient moment overflow')
    scale = rc / rr if rr else None
    residual = 0.0
    if scale is not None:
        for start in range(0, left.numel(), 65536):
            a = left[start:start + 65536].double()
            b = right[start:start + 65536].double()
            residual += (b - scale * a).square().sum().item()
    return {
        'reference_norm': math.sqrt(rr), 'candidate_norm': math.sqrt(cc),
        'candidate_over_reference_norm': math.sqrt(cc / rr) if rr else None,
        'best_fit_candidate_scale': scale,
        'residual_over_reference_norm': math.sqrt(residual / rr) if rr else None,
        'both_zero': rr == cc == 0,
    }


def diagnose(reference_root: Path, candidate_root: Path) -> dict:
    from h2_numerics import read_probe

    reference, left = read_probe(reference_root)
    candidate, right = read_probe(candidate_root)
    for key in ('schema', 'compute_policy', 'before', 'work'):
        if reference[key] != candidate[key]:
            raise ValueError('Probe metadata mismatch: ' + key)
    if set(left) != set(right):
        raise ValueError('Probe tensor names mismatch')
    if any(not left[k].equal(right[k]) for k in left if k.startswith('input.')):
        raise ValueError('Probe input mismatch')
    rows = {name: scale_geometry(left['gradient.' + name], right['gradient.' + name])
            for name in sorted(reference['before']['trainable'])}
    pins = {
        arm: {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ('probe.json', 'probe.safetensors', 'files_sha256.json')}
        for arm, root in [('reference', reference_root), ('candidate', candidate_root)]
    }
    return {
        'schema': 'h2-offline-gradient-geometry-v1',
        'status': 'DESCRIPTIVE_ONLY', 'performance_verdict': 'NO_VERDICT',
        'scope': 'Saved one-row probe; no causal attribution or tolerance change',
        'source_sha256': pins, 'gradients': rows,
        'nonzero_reference_count': sum(row['reference_norm'] > 0 for row in rows.values()),
        'both_zero_count': sum(row['both_zero'] for row in rows.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Diagnostic output must be new')
    result = diagnose(args.reference, args.candidate)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    Console().print(result['status'], args.output)


if __name__ == '__main__':
    main()
