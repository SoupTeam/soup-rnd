"""Prospective single-process H2 correctness runner. Never a throughput benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from rich.console import Console

TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
PACKAGES = ('torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'accelerate', 'datasets')
SOURCES = ('h2_gpu_correctness.py', 'h2_update_oracle.py', 'h2_quant_state.py',
           'h2_work_equivalence.py', 'h2_trainer_smoke.py', 'h2_watchdog.py', 'h2_numerics.py')


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(root: Path, expected: dict[str, str]) -> None:
    """Require a complete regular-file local snapshot; no Hub/cache or symlink fallback."""
    if not root.is_dir() or root.is_symlink() or not expected:
        raise ValueError('Pinned regular-file local model snapshot required')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Snapshot symlinks are not accepted')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError('Snapshot file manifest differs')
    for name, digest in expected.items():
        candidate = os.path.realpath(root / name)
        if (os.path.commonpath([os.path.realpath(root), candidate]) != os.path.realpath(root)
                or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                or sha_file(Path(candidate)) != digest):
            raise ValueError('Snapshot containment/SHA mismatch: ' + name)
    if not {'config.json', 'tokenizer_config.json', 'tokenizer.json'} <= actual:
        raise ValueError('Model config and tokenizer assets must be pinned together')
    if not any(name.endswith('.safetensors') for name in actual):
        raise ValueError('Pinned safetensors weights required')


def expected_adapters(config: Any, rank: int) -> dict[str, list[int]]:
    """Explicit dense Llama projection shapes, independent of observed trainable names."""
    if config.model_type != 'llama' or type(rank) is not int or rank < 1:
        raise ValueError('Only declared dense Llama LoRA is supported')
    hidden, intermediate = config.hidden_size, config.intermediate_size
    head_dim = getattr(config, 'head_dim', None) or hidden // config.num_attention_heads
    widths = {'q_proj': (hidden, config.num_attention_heads * head_dim),
              'k_proj': (hidden, config.num_key_value_heads * head_dim),
              'v_proj': (hidden, config.num_key_value_heads * head_dim),
              'o_proj': (config.num_attention_heads * head_dim, hidden),
              'gate_proj': (hidden, intermediate), 'up_proj': (hidden, intermediate),
              'down_proj': (intermediate, hidden)}
    expected = {}
    for layer in range(config.num_hidden_layers):
        for target, (inputs, outputs) in widths.items():
            block = 'self_attn' if target in TARGETS[:4] else 'mlp'
            prefix = f'base_model.model.model.layers.{layer}.{block}.{target}'
            expected[prefix + '.lora_A.default.weight'] = [rank, inputs]
            expected[prefix + '.lora_B.default.weight'] = [outputs, rank]
    return expected


def validate_loaded(wrapper: Any, arm: str) -> dict:
    """Inspect real loaded modules. Flags alone cannot establish actual kernel dispatch."""
    import torch
    from bitsandbytes.nn import Linear4bit

    model, trainer = wrapper.model, wrapper.trainer
    cfg = model.config
    if (cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.vocab_size) != (
                'llama', 32, 4096, 14336, 32, 8, 128256):
        raise ValueError('Loaded architecture differs from declared Llama-3.1-8B')
    expected = expected_adapters(cfg, 16)
    trainable = {name: list(value.shape) for name, value in model.named_parameters()
                 if value.requires_grad}
    if trainable != expected:
        raise ValueError('Actual LoRA names/shapes differ from the declared projection manifest')
    devices = {str(p.device) for p in model.parameters()}
    if devices != {'cuda:0'}:
        raise ValueError('CPU/disk/multi-GPU offload is not permitted: ' + str(devices))
    args = trainer.args
    if not args.bf16 or args.fp16:
        raise ValueError('Resolved trainer compute must be bf16, never fp16 fallback')
    quantized = {name: module for name, module in model.named_modules()
                 if isinstance(module, Linear4bit)}
    expected_modules = {name.split('.lora_')[0] + '.base_layer' for name in expected}
    if set(quantized) != expected_modules:
        raise ValueError('Missing/extra NF4 projection modules')
    if any(module.compute_dtype is not torch.bfloat16 for module in quantized.values()):
        raise ValueError('Actual NF4 module compute dtype differs from bf16')
    if not model.is_gradient_checkpointing:
        raise ValueError('Requested gradient checkpointing is inactive')
    classes = sorted({type(m).__module__ + '.' + type(m).__qualname__ for m in model.modules()})
    forwards = sorted({getattr(m.forward, '__module__', '') + '.'
                       + getattr(m.forward, '__qualname__', '') for m in model.modules()})
    if arm == 'liger' and (not getattr(wrapper, '_liger_applied', False)
                          or not args.use_liger_kernel
                          or not any('liger_kernel.' in name for name in classes)):
        raise ValueError('Liger requested but actual patched modules were not found')
    if arm == 'baseline' and any('liger_kernel.' in name or 'unsloth' in name
                                 for name in classes + forwards):
        raise ValueError('Baseline has unexpected treatment classes')
    if arm == 'unsloth' and not any('unsloth' in name for name in classes + forwards):
        raise ValueError('Unsloth attribution unresolved; refuse a fallback comparison')
    return {'trainable_shapes': expected, 'devices': sorted(devices),
            'nf4_modules': sorted(quantized), 'model_classes': classes,
            'forward_implementations': forwards,
            'attention_implementation': getattr(cfg, '_attn_implementation', None),
            'gradient_checkpointing': model.is_gradient_checkpointing,
            'scope': 'Loaded-module evidence; executed-kernel validation still required'}


def validate_authority(contract: dict, arm: str) -> None:
    """Cheap authority checks before the external supervisor launches any child."""
    if contract.get('execution_authorized') is not True:
        raise ValueError('H2 execution authority is not granted')
    if arm not in {'baseline', 'liger', 'unsloth'}:
        raise ValueError('Unknown arm')
    if contract.get('license_and_access_confirmed') is not True:
        raise ValueError('Model license/access is not confirmed')
    if (type(contract.get('instance_id')) is not int or contract['instance_id'] <= 0
            or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', contract.get('gpu_uuid') or '')
            or not re.fullmatch(r'[0-9a-f-]{36}', contract.get('boot_id') or '')):
        raise ValueError('Exact instance, GPU UUID and boot identity required')
    costs = [contract.get(name) for name in ('budget_cap_usd', 'prior_spend_usd',
                                           'hourly_rate_usd', 'cleanup_traffic_reserve_usd')]
    if (any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in costs)
            or costs[0] <= 0 or costs[2] <= 0):
        raise ValueError('Explicit finite H2 cost authority required')
    if re.fullmatch(r'[0-9a-f]{40}', contract.get('model_revision') or '') is None:
        raise ValueError('Exact model/tokenizer snapshot revision required')
    if (type(contract.get('max_seconds')) is not int
            or not 1 <= contract['max_seconds'] <= 900):
        raise ValueError('Bound correctness attempt to at most 900 seconds')
    deadline = contract.get('deadline_unix_s')
    if (type(deadline) not in (float, int) or not math.isfinite(deadline)
            or deadline <= time.time() + 2):
        raise ValueError('Unexpired absolute whole-attempt deadline required')
    if costs[1] + costs[2] * (contract['max_seconds'] + 2) / 3600 + costs[3] > costs[0]:
        raise ValueError('Correctness attempt exceeds cumulative H2 budget')


def preflight(contract: dict, arm: str) -> dict:
    """No allocations/downloads; run artifact/runtime checks inside the watchdog."""
    validate_authority(contract, arm)
    versions = contract.get('versions', {})
    required = set(PACKAGES) | {'liger-kernel', 'unsloth', 'triton'}
    # One common environment across all arms; absence is not a treatment.
    if set(versions) != required or not all(isinstance(v, str) and v for v in versions.values()):
        raise ValueError('Exact common runtime versions are required')
    for name, version in versions.items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Installed runtime differs: ' + name)
    for name in SOURCES:
        if contract.get('runner_sha256', {}).get(name) != sha_file(Path(__file__).with_name(name)):
            raise ValueError('Runner source pin differs: ' + name)
    repo = Path(__file__).parents[2]
    commit = contract.get('source_commit') or ''
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError('Committed Soup source revision required')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    changed = subprocess.run(['git', 'diff', 'HEAD', '--name-only', '--', 'src'], cwd=repo,
                             check=True, capture_output=True, text=True, timeout=10).stdout
    if head != commit or changed.strip():
        raise ValueError('Soup checkout differs from the committed source authority')
    for name in SOURCES:
        committed = subprocess.run(['git', 'show', f'{commit}:benchmarks/harness/{name}'],
                                   cwd=repo, check=True, capture_output=True, timeout=10).stdout
        if hashlib.sha256(committed).hexdigest() != contract['runner_sha256'][name]:
            raise ValueError('Runner is not committed at source revision: ' + name)
    policy_path = repo / 'benchmarks/h2/numerical_policy.json'
    policy_sha = sha_file(policy_path)
    committed_policy = subprocess.run(
        ['git', 'show', f'{commit}:benchmarks/h2/numerical_policy.json'], cwd=repo,
        check=True, capture_output=True, timeout=10,
    ).stdout
    if (contract.get('numerical_policy_sha256') != policy_sha
            or hashlib.sha256(committed_policy).hexdigest() != policy_sha):
        raise ValueError('Numerical policy must match prospective committed pin')
    root = Path(contract['model_path'])
    verify_snapshot(root, contract.get('model_files_sha256', {}))
    return {'versions': versions, 'model_revision': contract['model_revision'],
            'snapshot_files': len(contract['model_files_sha256'])}


def run(contract: dict, arm: str, out: Path) -> dict:
    """Correctness-only instrumentation; never use this window to report tok/s."""
    if out.exists():
        raise FileExistsError('Evidence output must be new; no resume or retry')
    out.mkdir(parents=True)
    result = {'status': 'FAILED', 'arm': arm, 'performance_verdict': 'NO_VERDICT'}
    original = None
    records = None
    alarm_handler = None
    started = time.monotonic()
    try:
        result['preflight'] = preflight(contract, arm)
        if os.name != 'posix' or not hasattr(signal, 'SIGALRM'):
            raise ValueError('Linux bounded single-process runner required')
        import platform
        if platform.system() != 'Linux':
            raise ValueError('Linux CUDA host required')
        def expired(*unused: Any) -> None:
            raise TimeoutError('Correctness attempt deadline reached; do not retry')
        alarm_handler = signal.signal(signal.SIGALRM, expired)
        signal.alarm(contract['max_seconds'])
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false')
        if arm == 'unsloth':
            import unsloth  # noqa: F401 — must precede transformers
        import torch
        import trl
        from h2_numerics import collect_probe
        from h2_trainer_smoke import recording_trainer
        from h2_update_oracle import audit_update, snapshot
        from h2_work_equivalence import validate_work

        from soup_cli.config.schema import SoupConfig
        from soup_cli.data.loader import load_dataset
        from soup_cli.trainer.sft import SFTTrainerWrapper

        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or torch.cuda.get_device_capability()[0] < 8
                or not torch.cuda.is_bf16_supported()
                or torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3):
            raise ValueError('Need one visible Ampere+ CUDA GPU with bf16 and at least16GiB')
        if (Path('/proc/sys/kernel/random/boot_id').read_text().strip() != contract['boot_id']
                or str(torch.cuda.get_device_properties(0).uuid) != contract['gpu_uuid']):
            raise ValueError('Live boot/GPU identity differs from authority')
        repo = Path(__file__).parents[2]
        recipe = repo / 'benchmarks/h2/candidate_configs' / (arm + '.json')
        fixture = repo / 'benchmarks/h2/synthetic_fixture.jsonl'
        for name, path in [('recipe_sha256', recipe), ('fixture_sha256', fixture)]:
            if sha_file(path) != contract.get(name):
                raise ValueError('Prospective recipe/fixture SHA mismatch')
        raw = json.loads(recipe.read_text())
        raw['base'], raw['data']['train'], raw['output'] = (
            contract['model_path'], str(fixture), str(out / 'trained'))
        cfg = SoupConfig.model_validate(raw)
        records = {'expected_names': [], 'batches': [], 'gradients': [],
                   'optimizer_events': [], 'observed_optimizers': set()}
        records['journal_path'] = str(out / 'events.jsonl')
        original = trl.SFTTrainer
        trl.SFTTrainer = recording_trainer(original, records)
        wrapper = SFTTrainerWrapper(cfg, device='cuda', report_to='none')
        dataset = load_dataset(cfg.data)
        if len(dataset['train']) != 128:
            raise ValueError('Fixture rows filtered or changed')
        wrapper.setup(dataset)
        result['loaded'] = validate_loaded(wrapper, arm)
        expected = list(result['loaded']['trainable_shapes'])
        records['expected_names'] = expected
        result['before'] = snapshot(wrapper.model, expected)
        probe_batch = wrapper.trainer.data_collator([wrapper.trainer.train_dataset[0]])
        result['model_probe'] = collect_probe(wrapper.model, probe_batch, expected,
                                               out / 'model_probe', bf16=True)
        wrapper.trainer.create_optimizer()
        if type(wrapper.trainer.optimizer) is not torch.optim.AdamW:
            raise ValueError('Actual optimizer is not the declared torch AdamW')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            wrapper.train()
        torch.cuda.synchronize()
        result['after'] = snapshot(wrapper.model, expected)
        result['work_audit'] = validate_work(records, expected_batches=128, accumulation=4)
        result['update_audit'] = audit_update(result['before'], result['after'],
                                              records['gradients'],
                                              len(records['optimizer_events']),
                                              [row['loss'] for row in records['batches']])
        result['records'] = {k: v for k, v in records.items() if k != 'observed_optimizers'}
        result['executed_operations'] = [{'name': e.key, 'count': e.count}
                                         for e in prof.key_averages()]
        prof.export_chrome_trace(str(out / 'kernel_trace.json'))
        if (result['work_audit']['status'] != 'PASS_WORK_ACCOUNTING'
                or result['update_audit']['status'] != 'PASS_UPDATE_EVIDENCE'
                or wrapper.trainer.state.global_step != 32):
            raise ValueError('Actual update/work accounting failed')
        result['status'] = 'CORRECTNESS_EVIDENCE_REQUIRES_INDEPENDENT_REVIEW'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if original is not None:
            trl.SFTTrainer = original
        if alarm_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, alarm_handler)
        if records is not None:
            result['records'] = {k: v for k, v in records.items() if k != 'observed_optimizers'}
        result['instrumented_elapsed_seconds_not_performance'] = time.monotonic() - started
        (out / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
        (out / 'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        pins = {p.relative_to(out).as_posix(): sha_file(p) for p in out.rglob('*') if p.is_file()}
        (out / 'files_sha256.json').write_text(json.dumps(pins, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--arm', choices=['baseline', 'liger', 'unsloth'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.contract.read_text()), args.arm, args.out)
    Console().print(result['status'] + '; performance=NO_VERDICT')
    if result['status'] == 'FAILED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
