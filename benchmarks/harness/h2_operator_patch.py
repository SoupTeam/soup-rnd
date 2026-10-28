"""Explicit Llama patch isolation in a disposable process; never restore global patches."""
from __future__ import annotations

import inspect
from collections import Counter
from typing import Any

FLAGS = ('rope', 'cross_entropy', 'fused_linear_cross_entropy', 'rms_norm', 'swiglu')
ARMS = {
    'baseline': (), 'ce_only': ('cross_entropy',),
    'fused_lce_only': ('fused_linear_cross_entropy',), 'rms_norm_only': ('rms_norm',),
    'rope_only': ('rope',), 'swiglu_only': ('swiglu',),
    'combined': ('rope', 'fused_linear_cross_entropy', 'rms_norm', 'swiglu'),
}
_INSTALLED = False


def patch_flags(arm: str) -> dict[str, bool]:
    if arm not in ARMS:
        raise ValueError('Unknown operator arm')
    return {name: name in ARMS[arm] for name in FLAGS}


def validate_counts(arm: str, counts: dict[str, int]) -> None:
    patch_flags(arm)
    expected = 0 if arm == 'baseline' else 1
    if counts != {'class': expected, 'instance': expected}:
        raise ValueError('Missing/duplicate/unexpected patch point')


def kernel_audit(events: list[dict], arm: str) -> dict:
    flags = patch_flags(arm)
    kernels = Counter(e.get('name', '').lower() for e in events if e.get('cat') == 'kernel')
    if not kernels:
        raise ValueError('No executed CUDA kernels')
    signatures = {'rope': '_triton_rope', 'rms_norm': '_rms_norm_',
                  'swiglu': '_swiglu_', 'cross_entropy': 'liger_cross_entropy'}
    hits = {key: sum(count for name, count in kernels.items() if pattern in name)
            for key, pattern in signatures.items()}
    for key, count in hits.items():
        expected = flags[key] or (key == 'cross_entropy' and flags['fused_linear_cross_entropy'])
        if bool(count) != expected:
            raise ValueError('Executed kernel attribution mismatch: ' + key)
    return {'status': 'PASS_DECLARED_KERNEL_FAMILIES', 'kernel_events': sum(kernels.values()),
            'family_events': hits, 'scope': 'Family attribution, not numerical parity'}


class PatchController:
    """Intercept both Soup and HF patch entry points, keeping all five flags explicit."""

    def __init__(self, arm: str) -> None:
        global _INSTALLED
        self.flags = patch_flags(arm)
        if _INSTALLED:
            raise ValueError('Fresh process required for every arm')
        _INSTALLED = True
        self.arm = arm
        self.counts = {'class': 0, 'instance': 0}
        self.capture_prepared = False
        import torch.nn.functional as functional
        import transformers.trainer as trainer
        from transformers.models.llama import modeling_llama

        import soup_cli.utils.liger as soup_liger

        self.modeling = modeling_llama
        self.functional = functional
        self.original = self.bindings()
        expected_modules = ('transformers.models.llama.modeling_llama',) * 4 + (
            'torch.nn.functional',)
        for value, expected in zip(self.original.values(), expected_modules):
            if value.__module__ != expected:
                raise ValueError('Process is already patched or has an unsupported Llama layout')
        from liger_kernel.transformers import monkey_patch
        from transformers.trainer_utils import unwrap_peft_model

        self.patch = monkey_patch
        self.unwrap = unwrap_peft_model
        params = inspect.signature(monkey_patch.apply_liger_kernel_to_llama).parameters
        if set(params) != set(FLAGS) | {'model'}:
            raise ValueError('Unsupported pinned patch signature')
        if self.bindings() != self.original:
            raise ValueError('Import changed model bindings before explicit patch')
        self.treatment = {
            'rope': monkey_patch.liger_rotary_pos_emb,
            'rms_norm': monkey_patch.LigerRMSNorm,
            'swiglu': monkey_patch.LigerSwiGLUMLP,
            'fused_linear_cross_entropy': monkey_patch.llama_lce_forward,
            'cross_entropy': monkey_patch.liger_cross_entropy,
        }
        soup_liger.apply_liger_kernel = self.class_patch
        trainer.apply_liger_kernel = self.instance_patch

    def bindings(self) -> dict:
        m = self.modeling
        return {'rope': m.apply_rotary_pos_emb, 'rms_norm': m.LlamaRMSNorm,
                'swiglu': m.LlamaMLP, 'fused_linear_cross_entropy': m.LlamaForCausalLM.forward,
                'cross_entropy': self.functional.cross_entropy}

    def class_patch(self, model_name: str) -> bool:
        if self.arm == 'baseline' or self.counts['class'] or self.counts['instance']:
            raise ValueError('Unexpected class patch invocation')
        self.patch.apply_liger_kernel_to_llama(**self.flags)
        self.counts['class'] += 1
        return True

    def instance_patch(self, model: Any, kernel_config: dict | None) -> None:
        if (self.arm == 'baseline' or self.counts != {'class': 1, 'instance': 0}
                or kernel_config not in (None, {}, self.flags)):
            raise ValueError('Unexpected instance patch invocation/configuration')
        self.patch.apply_liger_kernel_to_llama(model=self.unwrap(model), **self.flags)
        self.counts['instance'] += 1

    def prepare_capture(self, trainer: Any, model: Any) -> dict:
        """Apply the instance stage explicitly for a probe that never calls train()."""
        if self.capture_prepared:
            raise ValueError('Capture lifecycle already prepared; no retry')
        # Consume even a failed attempt: partial patching cannot be retried safely.
        self.capture_prepared = True
        enabled = self.arm != 'baseline'
        if (trainer.model is not model or trainer.model_wrapped is not model
                or trainer.args.use_liger_kernel is not enabled
                or trainer.args.liger_kernel_config not in (None, {}, self.flags)
                or trainer.optimizer is not None or trainer.lr_scheduler is not None
                or trainer.state.global_step != 0 or trainer.is_in_train):
            raise ValueError('Unexpected Trainer state before diagnostic patch')
        if self.counts != {'class': int(enabled), 'instance': 0}:
            raise ValueError('Unexpected patch lifecycle after setup')
        if enabled:
            self.instance_patch(model, trainer.args.liger_kernel_config)
        validate_counts(self.arm, self.counts)
        return {'status': 'PASS_EXPLICIT_CAPTURE_LIFECYCLE',
                'instance_patch_owner': 'diagnostic' if enabled else 'none',
                'patch_calls': dict(self.counts), 'trainer_train_called': False,
                'optimizer_present': False, 'global_step': trainer.state.global_step}

    def verify(self, model: Any) -> dict:
        validate_counts(self.arm, self.counts)
        expected = {key: self.treatment[key] if self.flags[key] else value
                    for key, value in self.original.items()}
        if self.bindings() != expected:
            raise ValueError('Unexpected global patch binding')
        base = self.unwrap(model)
        forward = getattr(base.forward, '__func__', base.forward)
        if forward is not expected['fused_linear_cross_entropy']:
            raise ValueError('Unexpected actual model forward')
        # Attribute and exact type checks detect unchanged, missing and extra module replacements.
        norms = [base.model.norm]
        mlps = []
        for layer in base.model.layers:
            norms += [layer.input_layernorm, layer.post_attention_layernorm]
            mlps.append(layer.mlp)
        if (len(norms) != 65 or len(mlps) != 32
                or any(type(m) is not expected['rms_norm'] for m in norms)
                or any(type(m) is not expected['swiglu'] for m in mlps)):
            raise ValueError('Unexpected loaded RMSNorm/MLP replacement')
        for modules, key, patched_forward in [
            (norms, 'rms_norm', self.patch.LigerRMSNorm.forward),
            (mlps, 'swiglu', self.patch.LigerSwiGLUMLP.forward),
        ]:
            wanted = patched_forward if self.flags[key] else self.original[key].forward
            if any(getattr(m.forward, '__func__', m.forward) is not wanted for m in modules):
                raise ValueError('Unexpected instance module forward: ' + key)
        return {'status': 'PASS_EXPLICIT_PATCH_BINDINGS', 'arm': self.arm,
                'flags': self.flags, 'patch_calls': dict(self.counts),
                'bindings': {k: v.__module__ + '.' + v.__qualname__ for k, v in expected.items()}}
