"""Offline, fresh-process integration with real Soup/TRL and a constructor-built Llama."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / 'benchmarks/harness/h2_trainer_smoke.py'


@pytest.mark.parametrize('mode', ['normal', 'skip_optimizer', 'wrong_targets'])
@pytest.mark.parametrize('bounded_profile', [False, True])
def test_real_sft_update_and_failure_controls(tmp_path, mode, bounded_profile):
    for dependency in ['torch', 'transformers', 'trl', 'peft', 'datasets', 'tokenizers']:
        pytest.importorskip(dependency)
    output = tmp_path / mode
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', ACCELERATE_USE_CPU='true',
               ACCELERATE_TORCH_DEVICE='cpu', TOKENIZERS_PARALLELISM='false')
    result = subprocess.run(
        [sys.executable, str(RUNNER), '--out', str(output), '--mode', mode]
        + (['--bounded-profile'] if bounded_profile else []),
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((output / 'evidence.json').read_text())
    assert evidence['performance_verdict'] == 'NO_VERDICT'
    if mode == 'wrong_targets':
        assert evidence['status'] == 'NEGATIVE_CONTROL_REJECTED'
        assert 'Trainable names' in evidence['reason']
        assert not (output / 'trained/adapter_model.safetensors').exists()
    else:
        assert evidence['trainer_global_step'] == 2
        assert len(evidence['records']['optimizer_events']) == 2
        assert len(evidence['records']['gradients']) == 4
        batches = evidence['records']['batches']
        assert len(batches) == 4
        events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
        assert [event['kind'] for event in events].count('batch') == 4
        assert [event['kind'] for event in events].count('backward') == 4
        assert [event['kind'] for event in events].count('optimizer') == 2
        assert evidence['work_audit']['status'] == 'PASS_WORK_ACCOUNTING'
        assert [event['completed_batches'] for event in
                evidence['records']['optimizer_events']] == [2, 4]
        assert all(len(batch['labels_sha256']) == 64 and
                   len(batch['attention_mask_sha256']) == 64 for batch in batches)
        # Four messages pairs have lengths6,4,4,6, including two explicit EOS each.
        assert sum(b['counts']['nonpadding_tokens'] for b in batches) == 20
        assert sum(b['counts']['useful_targets'] for b in batches) == 16
        assert evidence['before']['frozen'] == evidence['after']['frozen']
        executed = {row['name'] for row in evidence['executed_cpu_operations']}
        assert 'aten::mm' in executed
        assert any('Backward' in name for name in executed)
        assert evidence['devices'] == ['cpu']
        assert (output / 'trained/adapter_model.safetensors').is_file()
        if mode == 'normal':
            assert evidence['audit']['status'] == 'PASS_UPDATE_EVIDENCE'
            assert evidence['audit']['changed_adapters']
        else:
            assert evidence['audit']['status'] == 'FAIL'
            assert 'No adapter bytes changed' in evidence['audit']['reasons']
            assert evidence['before']['trainable'] == evidence['after']['trainable']
    pins = json.loads((output / 'files_sha256.json').read_text())
    assert 'evidence.json' in pins
    for name, digest in pins.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_import_does_not_load_training_libraries():
    code = (f'import runpy,sys;runpy.run_path({str(RUNNER)!r});'
            'assert not any(n in sys.modules for n in ["torch","transformers","trl","peft"])')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'benchmarks/harness'))
    subprocess.run([sys.executable, '-c', code], env=env, check=True,
                   capture_output=True, timeout=10)
