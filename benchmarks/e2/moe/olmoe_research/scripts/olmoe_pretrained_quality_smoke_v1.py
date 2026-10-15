#!/usr/bin/env python3
"""Research-only, real Alpaca-data OLMoE quality pipeline smoke.

NOT confirmatory quality evidence and NOT an exact reproduction of Soup SFT:
- Uses explicitly documented user+assistant ChatML formatting with all-token loss.
- Fixed-order subset, constant LR, short repeated examples.
- Includes router auxiliary loss, FP16 autocast + GradScaler, FP32 LoRA AdamW.
- Supports baseline K=4/8/12/16 and research cached K=4/8/12.
- Research-only cache is PINNED by source SHA256. Production Mistral is unchanged.
Use it to validate the evaluation/save/correctness pipeline before agreeing final
quality thresholds and training/validation protocol with the team lead.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, __version__ as peft_version
from transformers import AutoTokenizer, OlmoeForCausalLM, __version__ as transformers_version

from soup_cli.data.chat_templates import resolve_chat_template

MODEL_ID = 'allenai/OLMoE-1B-7B-0924'
REVISION = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
DATA_TRAIN_SHA = '535f364382cca5a5fe82b8c0b6788182db89b7c55439061dab0d47cbaff319a1'
DATA_VAL_SHA = '39c8736ea6b123e01f7d0c56eced3d579e486155fb5773e666375557e42edf55'
RESEARCH_CACHE_SHA = '20c3dbdd1e7515c89d7c22967f4d2e87a1b3d10bcb2134b3d81a0c73edc05b6b'
GUARDED_CACHE_SHA = 'e0852684edbc6fc740506b7212f896ae48d835ab8f5dab5b247afa6201804a26'
LAYER_COUNT = 16
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def named_tensors_sha256(pairs) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(pairs, key=lambda pair: pair[0]):
        x = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def imported(path: Path, name: str, expected_sha: str):
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise RuntimeError(f'Missing or SHA256 mismatch: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path, expected_hash: str, count: int):
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f'Dataset SHA mismatch: {path}')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) != count:
        raise RuntimeError(f'Unexpected dataset size: {path}: {len(rows)}')
    for index, row in enumerate(rows):
        if sorted(row) != ['input', 'instruction', 'output'] or not all(isinstance(row[k], str) for k in row):
            raise RuntimeError(f'Unexpected Alpaca row schema: {path}, row {index}')
        if not row['instruction'].strip() or not row['output'].strip():
            raise RuntimeError(f'Empty required Alpaca text at {path}:{index}')
    return rows


def encode_row(tokenizer, row, max_length: int):
    # Explicit smoke-only mapping; final confirmatory work must audit that this
    # matches the approved Soup SFT formatting and loss-mask semantics.
    user_text = row['instruction'].strip()
    if row['input'].strip():
        user_text += '\n\n' + row['input'].strip()
    messages = [
        {'role': 'user', 'content': user_text},
        {'role': 'assistant', 'content': row['output'].strip()},
    ]
    encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        return_dict=True,
    )
    ids = encoding['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2 or ids.shape[1] > max_length:
        raise RuntimeError(f'Unexpected tokenization shape: {tuple(ids.shape)}')
    attention_mask = encoding.get('attention_mask')
    if attention_mask is None:
        attention_mask = torch.ones_like(ids)
    if attention_mask.shape != ids.shape or not torch.all(attention_mask == 1).item():
        raise RuntimeError('Smoke supports unpadded single-example input only')
    return ids.contiguous()


def forward_loss(model, ids_gpu: torch.Tensor):
    mask = torch.ones_like(ids_gpu)
    positions = torch.arange(ids_gpu.shape[1], device=ids_gpu.device).unsqueeze(0)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = model(
            input_ids=ids_gpu,
            attention_mask=mask,
            position_ids=positions,
            labels=ids_gpu.clone(),  # train_on_responses_only=False: all-token objective
            use_cache=False,
            output_router_logits=True,
        )
    if out.router_logits is None or len(out.router_logits) != LAYER_COUNT or out.aux_loss is None:
        raise RuntimeError('Router auxiliary outputs are missing')
    if not torch.isfinite(out.loss).item() or not torch.isfinite(out.aux_loss).item():
        raise RuntimeError('Nonfinite FP16 loss or router auxiliary loss')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, choices=[4, 8, 12, 16], required=True)
    parser.add_argument('--mode', choices=['baseline', 'cached'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-rows', type=int, default=8)
    parser.add_argument('--eval-rows', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--probe', type=Path, default=Path('/workspace/olmoe_disk_cache_research_v1.py'))
    parser.add_argument('--guard-module', type=Path, default=Path('/workspace/olmoe_guarded_cache_v2.py'))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists; refusing stale cache and report')
    if args.k == 16 and args.mode != 'baseline':
        parser.error('K=16 has no frozen prefix; cached mode is not applicable')
    if args.train_rows < 2 or args.train_rows > 2000 or not 1 <= args.eval_rows <= 200:
        parser.error('Invalid train/eval subset sizes')
    if args.epochs < 1 or args.epochs > 2 or args.max_length != 128:
        parser.error('This smoke accepts 1-2 epochs and max_length=128 only')
    if transformers_version != '5.19.0':
        raise RuntimeError(f'Unexpected Transformers version: {transformers_version}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU: export CUDA_VISIBLE_DEVICES=0')
    if torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError('This smoke is intended only for SM7.5 FP16 hardware')
    cutoff = LAYER_COUNT - args.k
    repo = Path('/workspace/soup-rnd')
    train_path = repo / 'benchmarks/e2/quality/data/train.jsonl'
    validation_path = repo / 'benchmarks/e2/quality/data/validation.jsonl'
    train_rows = load_jsonl(train_path, DATA_TRAIN_SHA, 2000)[:args.train_rows]
    val_rows = load_jsonl(validation_path, DATA_VAL_SHA, 200)[:args.eval_rows]
    snapshot = (Path.home() / '.cache/huggingface/hub'
                / 'models--allenai--OLMoE-1B-7B-0924' / 'snapshots' / REVISION)
    if not (snapshot / 'config.json').is_file() or len(list(snapshot.glob('model-*.safetensors'))) != 3:
        raise RuntimeError('Pinned pretrained OLMoE checkpoint not found locally')

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    template = resolve_chat_template('chatml')
    if not isinstance(template, str) or not template:
        raise RuntimeError('Soup chatml template resolution failed')
    tokenizer.chat_template = template
    tokenized_train = [encode_row(tokenizer, row, args.max_length) for row in train_rows]
    tokenized_val = [encode_row(tokenizer, row, args.max_length) for row in val_rows]
    if len({tuple(item.squeeze(0).tolist()) for item in tokenized_train}) != len(tokenized_train):
        raise RuntimeError('Repeated tokenized train inputs would invalidate cache hit audit')
    tokenized_train_hash = named_tensors_sha256([(str(i), t) for i, t in enumerate(tokenized_train)])
    tokenized_val_hash = named_tensors_sha256([(str(i), t) for i, t in enumerate(tokenized_val)])
    print(f'DATA CHECK: train_rows={len(tokenized_train)} val_rows={len(tokenized_val)} '
          f'train_token_hash={tokenized_train_hash[:16]}', flush=True)

    print(f'Loading pinned pretrained OLMoE in FP16; K={args.k}, mode={args.mode}', flush=True)
    base = OlmoeForCausalLM.from_pretrained(
        str(snapshot), dtype=torch.float16, device_map={'': 0},
        local_files_only=True, trust_remote_code=False,
    )
    if len(base.model.layers) != LAYER_COUNT or base.config.num_experts != 64:
        raise RuntimeError('Unexpected pretrained architecture')
    if float(base.config.attention_dropout) != 0.0:
        raise RuntimeError('Active attention dropout unsupported')
    base.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=['q_proj', 'v_proj'],
        layers_to_transform=list(range(cutoff, LAYER_COUNT)),
        layers_pattern='layers', bias='none',
    )
    model = get_peft_model(base, lora_cfg)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if len(trainable) != 4 * args.k:
        raise RuntimeError(f'Wrong number of trainable LoRA tensors: {len(trainable)}')
    for name, p in trainable:
        match = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
        if not match or int(match.group(1)) < cutoff or 'lora_' not in name or p.dtype != torch.float32:
            raise RuntimeError(f'Wrong LoRA trainable tensor {name}, dtype {p.dtype}')
    if any(p.requires_grad for p in base.model.embed_tokens.parameters()):
        raise RuntimeError('Embedding unexpectedly trainable')
    for layer in base.model.layers[:cutoff]:
        if any(p.requires_grad for p in layer.parameters()):
            raise RuntimeError('Frozen decoder parameter trainable')
    initial_adapter_hash = named_tensors_sha256(trainable)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
    if not scaler.is_enabled():
        raise RuntimeError('FP16 GradScaler disabled')

    cache = None
    if args.mode == 'cached':
        probe = imported(args.probe, 'olmoe_quality_research_cache_v1', RESEARCH_CACHE_SHA)
        if probe.MODEL_REVISION != 'random-olmoe-4layers-seed42-research-only':
            raise RuntimeError('Unexpected cache source identity')
        probe.MODEL_REVISION = f'{MODEL_ID}@{REVISION}'
        guarded = imported(args.guard_module, 'olmoe_quality_guard_v2', GUARDED_CACHE_SHA)
        cache_type = guarded.guarded_cache_type(probe.ResearchOlmoeCache)
        args.output_dir.mkdir(parents=True)
        cache = cache_type(base, args.output_dir / 'cache', cutoff)
        cache.set_model_revision(f'{MODEL_ID}@{REVISION}')
    else:
        args.output_dir.mkdir(parents=True)

    torch.cuda.reset_peak_memory_stats()
    losses = []
    step_count = 0
    start = time.perf_counter()
    model.train()
    with (cache if cache is not None else nullcontext()):
        for epoch in range(1, args.epochs + 1):
            for index, ids_cpu in enumerate(tokenized_train, 1):
                ids = ids_cpu.to('cuda:0')
                model.zero_grad(set_to_none=True)
                before_scale = float(scaler.get_scale())
                out = forward_loss(model, ids)
                scaler.scale(out.loss).backward()
                scaler.unscale_(optimizer)
                for name, p in trainable:
                    if p.grad is None or not torch.isfinite(p.grad).all().item():
                        raise RuntimeError(f'Nonfinite/unset LoRA gradient: {name}')
                scaler.step(optimizer)
                scaler.update()
                after_scale = float(scaler.get_scale())
                if after_scale < before_scale or not optimizer.state:
                    raise RuntimeError(f'FP16 overflow or skipped AdamW step: {before_scale} -> {after_scale}')
                step_count += 1
                coef = float(base.config.router_aux_loss_coef)
                task_loss = float(out.loss.detach().float().item()) - coef * float(out.aux_loss.detach().float().item())
                losses.append({
                    'epoch': epoch,
                    'example_index': index,
                    'total_loss': float(out.loss.detach().float().item()),
                    'task_loss_estimate': task_loss,
                    'aux_loss': float(out.aux_loss.detach().float().item()),
                    'scaler_before': before_scale,
                    'scaler_after': after_scale,
                })
            print(f'EPOCH {epoch}: optimizer_steps={step_count} '
                  f'last_total_loss={losses[-1]["total_loss"]:.5f} '
                  f'cache_hits={cache.hits if cache else 0} '
                  f'cache_misses={cache.misses if cache else 0}', flush=True)
    torch.cuda.synchronize()
    train_wall = time.perf_counter() - start
    if step_count != args.train_rows * args.epochs:
        raise RuntimeError('Optimizer step count mismatch')
    if initial_adapter_hash == named_tensors_sha256(trainable):
        raise RuntimeError('LoRA adapters did not change')
    if cache is not None:
        expected_hits = args.train_rows * (args.epochs - 1)
        if cache.misses != args.train_rows or cache.hits != expected_hits:
            raise RuntimeError(f'Unexpected cache counts: hits={cache.hits}, misses={cache.misses}')
        if cache.skipped_frozen_layer_calls != expected_hits * cutoff:
            raise RuntimeError('Frozen layers not actually skipped')
        if cache.fingerprint_recomputes != 1:
            raise RuntimeError('Guard unexpectedly recomputed fingerprint')

    # Uncached evaluation on a DISJOINT validation subset, after training.
    # This is a smoke check, NOT a final gate suite or K=16 comparison.
    model.eval()
    eval_values = []
    with torch.inference_mode():
        for ids_cpu in tokenized_val:
            out = forward_loss(model, ids_cpu.to('cuda:0'))
            total = float(out.loss.detach().float().item())
            aux = float(out.aux_loss.detach().float().item())
            coef = float(base.config.router_aux_loss_coef)
            eval_values.append({'total_loss': total, 'task_loss_estimate': total - coef * aux, 'aux_loss': aux})
    mean_task = sum(x['task_loss_estimate'] for x in eval_values) / len(eval_values)
    mean_total = sum(x['total_loss'] for x in eval_values) / len(eval_values)
    mean_aux = sum(x['aux_loss'] for x in eval_values) / len(eval_values)
    if not all(torch.isfinite(torch.tensor(x)).item() for x in (mean_task, mean_total, mean_aux)):
        raise RuntimeError('Nonfinite validation metrics')
    adapter_dir = args.output_dir / 'adapter'
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    weights_file = adapter_dir / 'adapter_model.safetensors'
    if not weights_file.is_file():
        raise RuntimeError('Missing saved PEFT adapter safetensors')
    result = {
        'kind': 'olmoe_pretrained_real_alpaca_quality_pipeline_smoke_v1',
        'verdict': 'TECHNICAL_SMOKE_ONLY',
        'quality_acceptance': 'NO VERDICT',
        'model_id': MODEL_ID, 'revision': REVISION,
        'mode': args.mode, 'k': args.k, 'frozen_layers': cutoff,
        'pretrained': True, 'device': torch.cuda.get_device_name(0),
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'precision': {'base_fp16': True, 'lora_fp32': True, 'autocast_fp16': True,
                      'gradscaler_enabled': scaler.is_enabled(), 'final_scale': float(scaler.get_scale())},
        'data': {'train': str(train_path), 'validation': str(validation_path),
                 'train_sha256': DATA_TRAIN_SHA, 'validation_sha256': DATA_VAL_SHA,
                 'training_subset_rows': args.train_rows, 'eval_subset_rows': args.eval_rows,
                 'tokenized_train_sha256': tokenized_train_hash,
                 'tokenized_validation_sha256': tokenized_val_hash,
                 'text_format': 'user: instruction + optional input, assistant: output; Soup chatml override',
                 'train_on_responses_only': False, 'max_length': args.max_length,
                 'chatml_template_sha256': hashlib.sha256(template.encode()).hexdigest()},
        'training': {'seed': SEED, 'epochs': args.epochs, 'optimizer': 'AdamW',
                     'lr': 1e-4, 'weight_decay': 0.0,
                     'scheduler': 'constant (SMOKE ONLY, not Mistral linear schedule)',
                     'batch_size': 1, 'grad_accumulation': 1,
                     'optimizer_steps_verified': step_count,
                     'losses': losses,
                     'seconds_including_cache_build': round(train_wall, 6)},
        'validation': {'mean_task_loss_estimate': mean_task,
                       'mean_total_loss': mean_total, 'mean_aux_loss': mean_aux,
                       'per_example': eval_values,
                       'warning': 'Only a tiny subset; not task accuracy, safety evaluation, or accepted K-comparison'},
        'adapter': {'initial_sha256': initial_adapter_hash,
                    'final_trainable_sha256': named_tensors_sha256(trainable),
                    'safetensors_sha256': sha256_file(weights_file)},
        'cache': {'enabled': cache is not None,
                  'hits': cache.hits if cache else 0,
                  'misses': cache.misses if cache else 0,
                  'skipped_frozen_layer_calls': cache.skipped_frozen_layer_calls if cache else 0,
                  'full_sha256_recomputes': cache.fingerprint_recomputes if cache else 0},
        'provenance': {'script_sha256': sha256_file(Path(__file__)),
                       'verified_cache_sha256': sha256_file(args.probe) if cache else None,
                       'guard_sha256': sha256_file(args.guard_module) if cache else None,
                       'torch': torch.__version__, 'transformers': transformers_version,
                       'peft': peft_version, 'python': sys.version.split()[0],
                       'host': platform.node(), 'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES')},
        'limitations': [
            'Not exact Soup SFT preparation, even though using its registered ChatML Jinja template',
            'Eight-row subsets and constant LR; no real quality or safety acceptance',
            'Version-counter guard requires explicit invalidation for .data/raw-storage mutations',
            'No repeated ABBA, no benchmark speed claim',
            'Does not compare K values or establish K16 reference quality',
        ],
    }
    (args.output_dir / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('VALIDATION:', {'mean_task_loss_estimate': round(mean_task, 6),
                          'mean_total_loss': round(mean_total, 6),
                          'mean_aux_loss': round(mean_aux, 6)}, flush=True)
    print(f'OLMOE K{args.k} {args.mode.upper()} REAL-DATA QUALITY PIPELINE SMOKE PASSED', flush=True)
    print('REPORT:', args.output_dir / 'report.json', flush=True)

if __name__ == '__main__':
    main()
