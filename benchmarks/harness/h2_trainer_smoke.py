"""Offline CPU integration: real Soup SFT setup/train on a tiny random Llama + PEFT."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any

from h2_update_oracle import audit_update, gradients, snapshot, tensor_sha
from h2_work_equivalence import batch_evidence, validate_work
from rich.console import Console


def recording_trainer(base: type, records: dict) -> type:
    """Subclass the real trainer; retain collated work, backward and optimizer events."""
    def journal(kind: str, value: dict) -> None:
        if records.get('journal_path'):
            with Path(records['journal_path']).open('a') as stream:
                stream.write(json.dumps({'kind': kind, 'value': value}, allow_nan=False) + '\n')

    class RecordingTrainer(base):
        def compute_loss(self, model: Any, inputs: dict, *args: Any, **kwargs: Any) -> Any:
            training = model.training
            batch = batch_evidence(inputs) if training else None
            result = super().compute_loss(model, inputs, *args, **kwargs)
            if training:
                loss = result[0] if isinstance(result, tuple) else result
                scalar = float(loss.detach().item())
                records['batches'].append({**batch, 'loss': scalar if math.isfinite(scalar)
                                          else repr(scalar)})
                journal('batch', records['batches'][-1])
            return result

        def training_step(self, model: Any, *args: Any, **kwargs: Any) -> Any:
            result = super().training_step(model, *args, **kwargs)
            records['gradients'].append(gradients(model, records['expected_names']))
            journal('backward', records['gradients'][-1])
            return result

        def create_optimizer(self, *args: Any, **kwargs: Any) -> Any:
            result = super().create_optimizer(*args, **kwargs)
            optimizer = self.optimizer
            if id(optimizer) not in records['observed_optimizers']:
                def completed_step(*unused: Any) -> None:
                    records['optimizer_events'].append({
                        'completed_batches': len(records['batches']),
                        'adapter_sha256': {name: tensor_sha(param)
                                           for name, param in self.model.named_parameters()
                                           if param.requires_grad},
                    })
                    journal('optimizer', records['optimizer_events'][-1])
                optimizer.register_step_post_hook(completed_step)
                records['observed_optimizers'].add(id(optimizer))
            return result

    return RecordingTrainer


def build_local_model(path: Path) -> None:
    """Create local assets from constructors, never from a Hub checkpoint."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(42)
    words = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'hello', 'world', 'answer', 'seven']
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]',
        bos_token='[BOS]', eos_token='[EOS]', model_max_length=64,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] + ' ' + eos_token + ' ' }}"
        "{% endfor %}"
    )
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(words), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=2, eos_token_id=3,
        attention_dropout=0.0, tie_word_embeddings=False,
    ))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def run_smoke(root: Path, mode: str = 'normal') -> dict:
    """Exercise Soup's real wrapper. Negative controls never count as accepted training."""
    if mode not in {'normal', 'skip_optimizer', 'wrong_targets'}:
        raise ValueError('Unknown smoke mode')
    if root.exists():
        raise FileExistsError('Evidence output must be new')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['ACCELERATE_USE_CPU'] = 'true'
    os.environ['ACCELERATE_TORCH_DEVICE'] = 'cpu'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    import torch
    import trl

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper

    root.mkdir(parents=True)
    model_path = root / 'random_model'
    build_local_model(model_path)
    config = SoupConfig.model_validate({
        'base': str(model_path), 'task': 'sft', 'backend': 'transformers',
        'data': {'train': 'local-rows-supplied-in-memory', 'format': 'alpaca', 'max_length': 64,
                 'train_on_responses_only': False},
        'output': str(root / 'trained'),
        'training': {'epochs': 1, 'batch_size': 1, 'gradient_accumulation_steps': 2,
                     'quantization': 'none', 'seed': 42, 'data_seed': 42, 'lr': 0.01,
                     'optimizer': 'adamw_torch', 'scheduler': 'constant', 'warmup_ratio': 0.0,
                     'weight_decay': 0.0, 'gradient_checkpointing': False,
                     'auto_mixed_precision': False, 'logging_steps': 1, 'save_steps': 1000,
                     'lora': {'r': 2, 'alpha': 4, 'dropout': 0.0,
                              'target_modules': ['q_proj'] if mode == 'wrong_targets'
                              else ['q_proj', 'v_proj']}},
    })
    expected = [f'base_model.model.model.layers.0.self_attn.{projection}.lora_{side}.default.weight'
                for projection in ['q_proj', 'v_proj'] for side in ['A', 'B']]
    records = {'expected_names': expected, 'batches': [], 'gradients': [],
               'optimizer_events': [], 'observed_optimizers': set(),
               'journal_path': str(root / 'events.jsonl')}
    original_trainer = trl.SFTTrainer
    trl.SFTTrainer = recording_trainer(original_trainer, records)
    try:
        class CPUWrapper(SFTTrainerWrapper):
            @staticmethod
            def _as_sft_config(training_args: Any, max_length: int,
                               packing: bool = False) -> Any:
                # HF may auto-select local MPS despite Soup's CPU model-load setting.
                # This smoke explicitly fixes the real TrainingArguments device.
                training_args.use_cpu = True
                return SFTTrainerWrapper._as_sft_config(training_args, max_length, packing)

        wrapper = CPUWrapper(config, device='cpu', report_to='none')
        # The public wrapper receives rows already normalized by Soup's data loader.
        wrapper.setup({'train': [
            {'messages': [{'role': 'user', 'content': prompt},
                          {'role': 'assistant', 'content': answer}]}
            for prompt, answer in [('hello world', 'answer seven'), ('hello', 'seven'),
                                   ('world', 'answer'), ('hello world', 'seven seven')]
        ]})
        if any(param.device.type != 'cpu' for param in wrapper.model.parameters()):
            raise ValueError('CPU-only smoke selected devices: '
                             + str({str(p.device) for p in wrapper.model.parameters()}))
        try:
            before = snapshot(wrapper.model, expected)
        except ValueError as error:
            if mode != 'wrong_targets':
                raise
            result = {'status': 'NEGATIVE_CONTROL_REJECTED', 'reason': str(error),
                      'mode': mode, 'performance_verdict': 'NO_VERDICT'}
        else:
            from h2_numerics import collect_probe

            probe_batch = wrapper.trainer.data_collator([wrapper.trainer.train_dataset[0]])
            model_probe = collect_probe(wrapper.model, probe_batch, expected,
                                        root / 'model_probe', bf16=False)
            if mode == 'skip_optimizer':
                class NoOpAdamW(torch.optim.AdamW):
                    def step(self, closure: Any = None) -> None:
                        return None
                wrapper.trainer.optimizer = NoOpAdamW(
                    [p for p in wrapper.model.parameters() if p.requires_grad], lr=0.01,
                )
            # Attribution only: profiler overhead makes this unsuitable for throughput claims.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                wrapper.train()
            after = snapshot(wrapper.model, expected)
            audit = audit_update(before, after, records['gradients'],
                                 len(records['optimizer_events']),
                                 [row['loss'] for row in records['batches']])
            if len(records['batches']) != 4 or len(records['optimizer_events']) != 2:
                audit['status'] = 'FAIL'
                audit['reasons'].append('Expected exactly four batches and two optimizer events')
            result = {'mode': mode, 'audit': audit, 'before': before, 'after': after,
                      'model_probe': model_probe,
                      'work_audit': validate_work(records, expected_batches=4, accumulation=2),
                      'records': {k: v for k, v in records.items() if k != 'observed_optimizers'},
                      'trainer_global_step': wrapper.trainer.state.global_step,
                      'trainer_class': type(wrapper.trainer).__qualname__,
                      'attention_implementation': wrapper.model.config._attn_implementation,
                      'executed_cpu_operations': [
                          {'name': event.key, 'count': event.count} for event in prof.key_averages()
                      ],
                      'model_classes': sorted({type(m).__module__ + '.' + type(m).__name__
                                               for m in wrapper.model.modules()}),
                      'performance_verdict': 'NO_VERDICT'}
    finally:
        trl.SFTTrainer = original_trainer
    result.update({'scope': 'Tiny random CPU Llama, real Soup/TRL SFT and PEFT; not 8B/CUDA',
                   'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'oracle_sha256': hashlib.sha256(
                       Path(__file__).with_name('h2_update_oracle.py').read_bytes()).hexdigest(),
                   'work_oracle_sha256': hashlib.sha256(
                       Path(__file__).with_name('h2_work_equivalence.py').read_bytes()).hexdigest(),
                   'devices': ['cpu'],
                   'versions': {name: importlib.metadata.version(name)
                                for name in ['torch', 'transformers', 'trl', 'peft']}})
    (root / 'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    pins = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}
    (root / 'files_sha256.json').write_text(json.dumps(pins, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['normal', 'skip_optimizer', 'wrong_targets'],
                        default='normal')
    args = parser.parse_args()
    result = run_smoke(args.out, args.mode)
    status = result.get('audit', {}).get('status', result.get('status'))
    Console().print(str(status) + '; performance=NO_VERDICT')
    if args.mode == 'normal' and status != 'PASS_UPDATE_EVIDENCE':
        raise SystemExit(2)
    if args.mode != 'normal' and status not in {'FAIL', 'NEGATIVE_CONTROL_REJECTED'}:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
