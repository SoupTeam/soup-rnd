"""One model backward with bounded-scope profiling; no optimizer or training loop."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from h2_gpu_correctness import sha_file
from h2_update_oracle import snapshot, tensor_sha
from h2_work_equivalence import batch_evidence


def identity(before: dict, inputs: dict) -> dict:
    return {'before_sha256': hashlib.sha256(json.dumps(
        before, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'work': batch_evidence(inputs),
        'input_sha256': {'input.' + k: tensor_sha(v) for k, v in inputs.items()}}


def collect(model: Any, batch: dict, names: list[str], out: Path, *, cuda: bool,
            expected_identity: dict | None = None) -> dict:
    import torch
    from safetensors.torch import save_file

    if out.exists() or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Need a new output and clean gradients')
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1:
        raise ValueError('Single device required')
    device = next(iter(devices))
    if cuda and device.type != 'cuda':
        raise ValueError('CUDA capture requires actual CUDA parameters')
    if not cuda and (device.type != 'cpu' or any(
            p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError('CPU test requires FP32 parameters')
    out.mkdir(parents=True)
    probe_dir = out / 'model_probe'
    probe_dir.mkdir()
    modes = [(m, m.training) for m in model.modules()]
    before = snapshot(model, names)
    observed = {}
    handles = []

    def hook(name: str):
        def record(module: Any, inputs: tuple, output: Any) -> None:
            row = observed.setdefault(name, {'calls': 0, 'input_dtypes': set(),
                                            'output_dtypes': set()})
            row['calls'] += 1
            for tensor in inputs:
                if isinstance(tensor, torch.Tensor):
                    row['input_dtypes'].add(str(tensor.dtype))
            if isinstance(output, torch.Tensor):
                row['output_dtypes'].add(str(output.dtype))
        return record

    try:
        for name, module in model.named_modules():
            if name.endswith(('layernorm', '.norm', '.mlp', '.base_layer', '.lm_head')):
                handles.append(module.register_forward_hook(hook(name)))
        with torch.random.fork_rng(devices=[device.index or 0] if cuda else []):
            model.train()
            inputs = {k: v.to(device) for k, v in batch.items()}
            work = batch_evidence(inputs)
            if expected_identity is not None and identity(before, inputs) != expected_identity:
                raise ValueError('Initial model/input identity differs from frozen baseline')
            activities = [torch.profiler.ProfilerActivity.CPU]
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            # Hashing all frozen weights and writing evidence stay OUTSIDE the trace window.
            with torch.profiler.profile(activities=activities, record_shapes=False,
                                        profile_memory=False, with_stack=False) as profile:
                autocast = torch.autocast('cuda', dtype=torch.bfloat16) if cuda else (
                    contextlib.nullcontext())
                with torch.profiler.record_function('h2_operator_probe_forward_backward'):
                    with autocast:
                        loss = model(**inputs).loss
                    scalar = float(loss.detach().item())
                    if not math.isfinite(scalar):
                        raise ValueError('Nonfinite loss')
                    loss.backward()
                if cuda:
                    torch.cuda.synchronize()
            profile.export_chrome_trace(str(out / 'kernel_trace.json'))
            parameters = dict(model.named_parameters())
            tensors = {'input.' + k: v.detach().cpu().contiguous().clone()
                       for k, v in inputs.items()}
            for name in names:
                grad = parameters[name].grad
                if grad is None or not torch.isfinite(grad).all().item():
                    raise ValueError('Missing/nonfinite gradient: ' + name)
                tensors['gradient.' + name] = grad.detach().cpu().contiguous().clone()
        model.zero_grad(set_to_none=True)
        if snapshot(model, names) != before:
            raise ValueError('Probe changed model state without optimizer')
        save_file(tensors, str(probe_dir / 'probe.safetensors'))
        result = {'schema': 'h2-model-probe-v1', 'loss': scalar, 'work': work, 'before': before,
                  'tensor_sha256': {k: tensor_sha(v) for k, v in tensors.items()},
                  'scope': 'One collated row, train-mode model loss/backward, no optimizer',
                  'compute_policy': 'bf16-autocast-cuda' if cuda else 'fp32-cpu-test',
                  'performance_verdict': 'NO_VERDICT'}
        (probe_dir / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
        (probe_dir / 'files_sha256.json').write_text(json.dumps(
            {p.name: sha_file(p) for p in probe_dir.iterdir()}, indent=2) + '\n')
        dtypes = {name: {k: sorted(v) if isinstance(v, set) else v for k, v in row.items()}
                  for name, row in observed.items()}
        (out / 'observed_module_dtypes.json').write_text(json.dumps(dtypes, indent=2) + '\n')
        return {'probe_loss': scalar, 'optimizer_updates': 0, 'backward_calls': 1,
                'probe_sha256': sha_file(probe_dir / 'files_sha256.json'),
                'trace_sha256': sha_file(out / 'kernel_trace.json'),
                'state_unchanged': True, 'performance_verdict': 'NO_VERDICT'}
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for module, training in modes:
            module.training = training
