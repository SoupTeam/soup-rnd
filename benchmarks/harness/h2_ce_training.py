"""Prospective full-tensor CE confirmation; no performance or quality verdict."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from h2_gpu_correctness import sha_file
from h2_numerics import gradient_metrics, validate_policy
from h2_update_oracle import snapshot, tensor_sha
from h2_work_equivalence import batch_evidence


def write_json(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seal(root: Path) -> None:
    """Integrity receipt, not a substitute for the independent numerical reviewer."""
    files = {p.name: sha_file(p) for p in root.iterdir() if p.name != 'files_sha256.json'}
    (root / 'files_sha256.json').write_text(json.dumps(files, indent=2) + '\n')


def copy_tensors(values: dict) -> dict:
    import torch

    if not values:
        raise ValueError('Empty tensor evidence')
    result = {}
    for name, value in values.items():
        if value is None or not torch.isfinite(value).all():
            raise ValueError('Missing/nonfinite tensor: ' + name)
        result[name] = value.detach().cpu().contiguous().clone()
    return result


class Recorder:
    """Capture real Trainer observations; reject reordered work and incomplete updates."""

    def __init__(self, root: Path, model: Any, names: list[str], work: list[dict], *,
                 accumulation: int, compute_policy: str) -> None:
        if (len(work) != 16 or accumulation != 4 or compute_policy not in
                {'fp32-cpu-test', 'bf16-autocast-cuda'}):
            raise ValueError('Exactly sixteen batches/four updates and declared precision required')
        root.mkdir(parents=True, exist_ok=False)
        self.root, self.model, self.names, self.work = root, model, names, work
        self.before = snapshot(model, names)
        self.parameters = dict(model.named_parameters())
        self.batches: list[dict] = []
        self.backwards = 0
        self.events: list[dict] = []
        self.pending = False
        self.settings: dict | None = None
        self.compute_policy = compute_policy
        self.previous = self.adapters()
        self.save('initial', self.previous)

    def adapters(self) -> dict:
        return copy_tensors({name: self.parameters[name] for name in self.names})

    def save(self, name: str, tensors: dict) -> None:
        from safetensors.torch import save_file

        path = self.root / (name + '.safetensors')
        if path.exists():
            raise FileExistsError('No overwriting evidence')
        save_file(tensors, str(path))

    def optimizer_created(self, optimizer: Any, *, max_grad_norm: float) -> None:
        import torch

        if type(optimizer) is not torch.optim.AdamW or self.settings is not None:
            raise ValueError('One plain AdamW optimizer required')
        groups = optimizer.param_groups
        params = [p for group in groups for p in group['params']]
        if (len({id(p) for p in params}) != len(params)
                or {id(p) for p in params} != {id(self.parameters[n]) for n in self.names}):
            raise ValueError('Optimizer must own every and only declared adapter once')
        if type(max_grad_norm) not in (int, float) or max_grad_norm != 1.0:
            raise ValueError('Resolved clipping must be exactly 1.0')
        self.settings = {'class': 'torch.optim.AdamW', 'max_grad_norm': max_grad_norm,
                         'groups': []}
        for group in groups:
            if (group['lr'] != 0.0002 or group['weight_decay'] != 0
                    or group.get('amsgrad') or group.get('maximize')):
                raise ValueError('Unexpected resolved AdamW policy')
            self.settings['groups'].append({
                'names': sorted(n for n in self.names
                                if any(self.parameters[n] is p for p in group['params'])),
                **{k: group[k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad',
                                        'maximize', 'foreach', 'fused', 'capturable',
                                        'differentiable')},
            })

    def batch(self, inputs: dict, loss: Any, denominator: Any) -> None:
        if len(self.batches) != self.backwards or self.pending:
            raise ValueError('Unexpected forward/backward ordering')
        actual = {k: inputs[k].detach().cpu().tolist() for k in
                  ('input_ids', 'attention_mask', 'labels')}
        if self.backwards >= 16 or actual != self.work[self.backwards]:
            raise ValueError('Actual batch differs from fixed ordered work')
        scalar = float(loss.detach().item())
        if not math.isfinite(scalar):
            raise ValueError('Nonfinite training loss')
        if hasattr(denominator, 'item'):
            denominator = denominator.item()
        if denominator is not None and (type(denominator) is not int or denominator <= 0):
            raise ValueError('Unexpected normalization denominator')
        self.batches.append({'inputs': actual, 'loss': scalar,
                             'num_items_in_batch': denominator, **batch_evidence(inputs)})

    def backward(self) -> None:
        if len(self.batches) != self.backwards + 1:
            raise ValueError('Backward without exactly one forward')
        # Every backward must reach all declared trainables, including zero LoRA-A gradients.
        import torch

        if any(self.parameters[n].grad is None or not torch.isfinite(
                self.parameters[n].grad).all() for n in self.names):
            raise ValueError('Missing/nonfinite backward gradients')
        self.backwards += 1

    def preclip(self, *, scaler: Any) -> None:
        if (scaler is not None or self.pending or self.backwards != 4 * (len(self.events) + 1)
                or len(self.batches) != self.backwards):
            raise ValueError('Expected unscaled complete accumulation boundary before clipping')
        values = copy_tensors({n: self.parameters[n].grad for n in self.names})
        if not any(v.count_nonzero().item() for v in values.values()):
            raise ValueError('No nonzero gradients')
        self.save(f'gradient_{len(self.events) + 1:02d}', values)
        self.pending = True

    def poststep(self, optimizer: Any) -> None:
        import torch

        if not self.pending or self.settings is None:
            raise ValueError('Optimizer step without preclip evidence')
        if any(g['lr'] != 0.0002 or g['weight_decay'] != 0 for g in optimizer.param_groups):
            raise ValueError('Optimizer policy drifted')
        current = self.adapters()
        if all(current[n].equal(self.previous[n]) for n in self.names):
            raise ValueError('No adapter changed at optimizer step')
        for name in self.names:
            state = optimizer.state[self.parameters[name]]
            if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                    or state['step'].item() != len(self.events) + 1
                    or any(not torch.isfinite(v).all() for v in state.values())):
                raise ValueError('Missing/nonfinite optimizer state at update')
        self.save(f'adapter_{len(self.events) + 1:02d}', current)
        self.previous = current
        self.events.append({'completed_batches': self.backwards,
                            'adapter_sha256': {n: tensor_sha(v) for n, v in current.items()}})
        self.pending = False

    def finish(self, optimizer: Any, *, global_step: int) -> dict:
        if (self.pending or self.backwards != 16 or len(self.events) != 4
                or global_step != 4 or self.settings is None):
            raise ValueError('Incomplete training evidence')
        moments = {}
        for name in self.names:
            state = optimizer.state[self.parameters[name]]
            if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or state['step'].item() != 4:
                raise ValueError('Incomplete AdamW state')
            moments.update({name + '.' + key: value for key, value in state.items()})
        self.save('optimizer_final', copy_tensors(moments))
        after = snapshot(self.model, self.names)
        if any(after[key] != self.before[key] for key in ('frozen', 'quantizer')):
            raise ValueError('Frozen base/quantizer changed')
        result = {'schema': 'h2-ce-training-v1', 'before': self.before, 'after': after,
                  'compute_policy': self.compute_policy, 'batches': self.batches,
                  'backwards': self.backwards, 'events': self.events,
                  'global_step': global_step, 'optimizer': self.settings,
                  'performance_verdict': 'NO_VERDICT'}
        write_json(self.root / 'training.json', result)
        seal(self.root)
        return result


def recording_trainer(base: type, holder: dict, window: Any = None) -> type:
    """Actual Trainer loop owns normalization, clipping, optimizer and instance patch."""
    class CETrainer(base):
        def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Any:
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)

        def compute_loss(self, model: Any, inputs: dict, *args: Any, **kwargs: Any) -> Any:
            observed = {k: inputs[k] for k in ('input_ids', 'attention_mask', 'labels')}
            result = super().compute_loss(model, inputs, *args, **kwargs)
            if model.training:
                holder['recorder'].batch(observed, result[0] if isinstance(result, tuple)
                                         else result, kwargs.get('num_items_in_batch'))
            return result

        def training_step(self, *args: Any, **kwargs: Any) -> Any:
            with window.step() if window else nullcontext():
                result = super().training_step(*args, **kwargs)
            holder['recorder'].backward()
            return result

        def _clip_grad_norm(self, model: Any) -> Any:
            if self.accelerator.num_processes != 1 or not self.accelerator.sync_gradients:
                raise ValueError('Single-process synchronized accumulation required')
            holder['recorder'].preclip(scaler=self.accelerator.scaler)
            return super()._clip_grad_norm(model)

        def create_optimizer(self, *args: Any, **kwargs: Any) -> Any:
            result = super().create_optimizer(*args, **kwargs)
            holder['recorder'].optimizer_created(self.optimizer,
                                                 max_grad_norm=self.args.max_grad_norm)
            self.optimizer.register_step_post_hook(
                lambda optimizer, *unused: holder['recorder'].poststep(optimizer))
            holder['optimizer'] = self.optimizer
            return result

    return CETrainer


def read_training(root: Path, work: list[dict]) -> tuple[dict, dict]:
    import torch
    from safetensors.torch import load_file

    expected = {'training.json', 'initial.safetensors', 'optimizer_final.safetensors'}
    expected |= {f'{kind}_{step:02d}.safetensors' for step in range(1, 5)
                 for kind in ('gradient', 'adapter')}
    if (root.is_symlink() or not root.is_dir()
            or {p.name for p in root.iterdir()} != expected | {'files_sha256.json'}
            or any(p.is_symlink() or not p.is_file() for p in root.iterdir())):
        raise ValueError('Incomplete/nonregular training evidence')
    pins = json.loads((root / 'files_sha256.json').read_text())
    if set(pins) != expected or any(sha_file(root / n) != sha for n, sha in pins.items()):
        raise ValueError('Training file SHA mismatch')
    if any((root / n).stat().st_size > 512 * 1024**2 for n in expected):
        raise ValueError('Training evidence file exceeds declared size bound')
    data = json.loads((root / 'training.json').read_text())
    if (data['schema'] != 'h2-ce-training-v1' or data['backwards'] != 16
            or len(data['batches']) != 16 or len(work) != 16
            or data['global_step'] != 4 or len(data['events']) != 4
            or data['compute_policy'] not in {'fp32-cpu-test', 'bf16-autocast-cuda'}):
        raise ValueError('Unexpected training recipe/counts')
    for key in ('frozen', 'quantizer'):
        if data['before'][key] != data['after'][key]:
            raise ValueError('Frozen state changed')
    for index, row in enumerate(data['batches']):
        inputs = {k: torch.tensor(v, dtype=torch.long) for k, v in row['inputs'].items()}
        actual = batch_evidence(inputs)
        if (row['inputs'] != work[index] or any(row[k] != v for k, v in actual.items())
                or type(row['loss']) not in (int, float) or not math.isfinite(row['loss'])):
            raise ValueError('Invalid ordered work/loss')
        denominator = row['num_items_in_batch']
        count = sum(sum(v != -100 for v in seq[1:]) for batch in
                    work[index // 4 * 4:index // 4 * 4 + 4] for seq in batch['labels'])
        if denominator is not None and (type(denominator) is not int or denominator != count):
            raise ValueError('Invalid accumulation normalization denominator')
    initial = load_file(str(root / 'initial.safetensors'))
    names = set(data['before']['trainable'])
    if set(initial) != names or not names or names != set(data['after']['trainable']):
        raise ValueError('Adapter names changed')
    optimizer = data['optimizer']
    if (optimizer.get('class') != 'torch.optim.AdamW' or optimizer.get('max_grad_norm') != 1.0
            or not optimizer.get('groups')):
        raise ValueError('Invalid optimizer declaration')
    seen = []
    for group in optimizer['groups']:
        seen.extend(group['names'])
        if (group['lr'] != 0.0002 or group['weight_decay'] != 0
                or group['betas'] != [0.9, 0.999] or group['eps'] != 1e-8
                or group['amsgrad'] or group['maximize'] or group['differentiable']):
            raise ValueError('Invalid resolved optimizer policy')
    if set(seen) != names or len(seen) != len(names):
        raise ValueError('Invalid optimizer parameter ownership')
    for name, tensor in initial.items():
        state = data['before']['trainable'][name]
        final_state = data['after']['trainable'][name]
        if (not torch.isfinite(tensor).all() or tensor_sha(tensor) != state['sha256']
                or list(tensor.shape) != state['shape'] or str(tensor.dtype) != state['dtype']
                or state['finite'] is not True or final_state['finite'] is not True
                or final_state['shape'] != state['shape']
                or final_state['dtype'] != state['dtype']):
            raise ValueError('Initial adapter mismatch')
    previous = initial
    for step, event in enumerate(data['events'], 1):
        if event['completed_batches'] != step * 4 or set(event['adapter_sha256']) != names:
            raise ValueError('Invalid optimizer event')
        values = load_file(str(root / f'adapter_{step:02d}.safetensors'))
        grads = load_file(str(root / f'gradient_{step:02d}.safetensors'))
        for tensors in (values, grads):
            if set(tensors) != names or any(
                    v.shape != initial[n].shape or v.dtype != initial[n].dtype
                    or not torch.isfinite(v).all() for n, v in tensors.items()):
                raise ValueError('Invalid gradient/adapter tensor')
        if not any(v.count_nonzero().item() for v in grads.values()):
            raise ValueError('No nonzero gradient evidence')
        if all(values[n].equal(previous[n]) for n in names):
            raise ValueError('No adapter changed')
        if any(tensor_sha(v) != event['adapter_sha256'][n] for n, v in values.items()):
            raise ValueError('Adapter event tensor mismatch')
        previous = values
    if any(tensor_sha(v) != data['after']['trainable'][n]['sha256']
           for n, v in previous.items()):
        raise ValueError('Final adapter mismatch')
    moments = load_file(str(root / 'optimizer_final.safetensors'))
    if set(moments) != {n + '.' + k for n in names for k in ('step', 'exp_avg', 'exp_avg_sq')}:
        raise ValueError('Incomplete optimizer moments')
    for name in names:
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            value = moments[name + '.' + key]
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite optimizer moments')
            if key == 'step':
                if value.numel() != 1 or value.item() != 4:
                    raise ValueError('Wrong optimizer step')
            elif value.shape != initial[name].shape or value.dtype != initial[name].dtype:
                raise ValueError('Wrong optimizer moment shape/dtype')
            elif key == 'exp_avg_sq' and (value < 0).any():
                raise ValueError('Negative second moments')
    return data, initial


def compare_tensors(left: dict, right: dict, policy: dict) -> dict:
    """Strict prospective gradient/delta gates, including elementwise and zero semantics."""
    import torch

    if set(left) != set(right) or not left:
        raise ValueError('Tensor comparison manifest differs')
    metrics = {}
    for name in left:
        row = gradient_metrics(left[name], right[name])
        lhs, rhs = left[name].reshape(-1), right[name].reshape(-1)
        elementwise = all(torch.allclose(rhs[i:i + 65536].double(), lhs[i:i + 65536].double(),
                                        atol=policy['gradient_atol'],
                                        rtol=policy['gradient_rtol'])
                          for i in range(0, lhs.numel(), 65536))
        if (not elementwise or row['relative_l2'] is None
                or row['relative_l2'] > policy['gradient_relative_l2_max']
                or row['cosine'] < policy['gradient_cosine_min']):
            raise ValueError('Tensor exceeds prospective tolerance: ' + name)
        metrics[name] = row
    return metrics


def review_training(left: Path, right: Path, policy: dict, work: list[dict]) -> dict:
    from safetensors.torch import load_file

    validate_policy(policy)
    first, initial_left = read_training(left, work)
    second, initial_right = read_training(right, work)
    for key in ('before', 'optimizer', 'compute_policy'):
        if first[key] != second[key]:
            raise ValueError('Pair initial state/optimizer/precision mismatch: ' + key)
    for lhs, rhs in zip(first['batches'], second['batches']):
        if lhs['num_items_in_batch'] != rhs['num_items_in_batch']:
            raise ValueError('Pair normalization differs')
        if abs(lhs['loss'] - rhs['loss']) > policy['loss_atol'] + (
                policy['loss_rtol'] * abs(lhs['loss'])):
            raise ValueError('Pair loss exceeds tolerance')
    steps = []
    for step in range(1, 5):
        gradients = [load_file(str(root / f'gradient_{step:02d}.safetensors'))
                     for root in (left, right)]
        metric = compare_tensors(*gradients, policy)
        del gradients
        adapters = [load_file(str(root / f'adapter_{step:02d}.safetensors'))
                    for root in (left, right)]
        # FP64 subtraction exposes small update differences; do not compare large final weights.
        deltas = [{n: v.double() - initial[n].double() for n, v in values.items()}
                  for values, initial in zip(adapters, (initial_left, initial_right))]
        steps.append({'step': step, 'gradients': metric,
                      'adapter_deltas': compare_tensors(*deltas, policy)})
        del adapters, deltas
    return {'status': 'PASS_CE_TRAINING_TENSORS', 'steps': steps,
            'performance_verdict': 'NO_VERDICT',
            'scope': 'Sixteen fixed microbatches; dispatch/runtime/controls required separately'}
