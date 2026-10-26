"""Local gates for the prospective GPU runner; no GPU run is accepted here."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / 'benchmarks/harness'
sys.path.insert(0, str(HARNESS))
spec = importlib.util.spec_from_file_location('h2_gpu_correctness', HARNESS/'h2_gpu_correctness.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_declared_projection_shapes_match_real_peft_llama_without_model_download():
    pytest.importorskip('torch')
    pytest.importorskip('peft')
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = get_peft_model(LlamaForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, target_modules=list(runner.TARGETS), lora_dropout=0.0,
    ))
    expected = runner.expected_adapters(config, 2)
    assert {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad} == expected
    assert len(expected) == 28
    key = 'base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight'
    assert expected[key] == [8, 2]


def local_snapshot(tmp_path):
    # File integrity fixture only; these bytes are not claimed as model weights.
    root = tmp_path / 'snapshot'
    root.mkdir()
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors']:
        (root/name).write_bytes(b'file-integrity-fixture')
    return root, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}


def test_complete_local_snapshot_integrity(tmp_path):
    root, pins = local_snapshot(tmp_path)
    runner.verify_snapshot(root, pins)
    (root/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        runner.verify_snapshot(root, pins)


@pytest.mark.parametrize('problem', ['extra', 'missing', 'bad_digest', 'no_weights'])
def test_incomplete_snapshot_cannot_pass(tmp_path, problem):
    root, pins = local_snapshot(tmp_path)
    if problem == 'extra':
        (root/'extra').write_bytes(b'extra')
    elif problem == 'missing':
        (root/'tokenizer.json').unlink()
    elif problem == 'bad_digest':
        pins['model.safetensors'] = 'x' * 64
    else:
        del pins['model.safetensors']
        (root/'model.safetensors').unlink()
    with pytest.raises(ValueError):
        runner.verify_snapshot(root, pins)


@pytest.mark.requires_symlink
def test_snapshot_cannot_follow_external_weight_link(tmp_path):
    root, pins = local_snapshot(tmp_path)
    target = tmp_path/'outside'
    target.write_bytes((root/'model.safetensors').read_bytes())
    (root/'model.safetensors').unlink()
    (root/'model.safetensors').symlink_to(target)
    with pytest.raises(ValueError, match='symlinks'):
        runner.verify_snapshot(root, pins)


def authority():
    return {'execution_authorized': True, 'license_and_access_confirmed': True,
            'instance_id': 1, 'gpu_uuid': 'GPU-' + '0'*8 + '-' + '0'*4 + '-' + '0'*4
            + '-' + '0'*4 + '-' + '0'*12, 'boot_id': '0'*8+'-'+'0'*4+'-'+'0'*4+'-'+'0'*4+'-'+'0'*12,
            'model_revision': 'a'*40, 'max_seconds': 60, 'deadline_unix_s': time.time() + 120,
            'budget_cap_usd': 1,
            'prior_spend_usd': 0, 'hourly_rate_usd': 1, 'cleanup_traffic_reserve_usd': 0.1}


@pytest.mark.parametrize('problem,match', [('disabled', 'not granted'), ('license', 'license'),
                                          ('identity', 'identity'), ('nan', 'cost authority'),
                                          ('over_budget', 'exceeds'), ('revision', 'revision'),
                                          ('deadline', '900'), ('versions', 'versions')])
def test_missing_authority_or_pins_refused_before_model_load(problem, match):
    value = authority()
    if problem == 'disabled':
        value['execution_authorized'] = False
    elif problem == 'license':
        value['license_and_access_confirmed'] = False
    elif problem == 'identity':
        value['gpu_uuid'] = None
    elif problem == 'nan':
        value['budget_cap_usd'] = float('nan')
    elif problem == 'over_budget':
        value['prior_spend_usd'] = 1
    elif problem == 'revision':
        value['model_revision'] = 'main'
    elif problem == 'deadline':
        value['max_seconds'] = 901
    with pytest.raises(ValueError, match=match):
        runner.preflight(value, 'baseline')


def test_disabled_run_preserves_failure_without_importing_gpu_libraries(tmp_path):
    contract = tmp_path/'disabled.json'
    contract.write_text('{"execution_authorized": false}')
    out = tmp_path/'evidence'
    code = (f'import runpy,sys;sys.argv=["runner","--contract",{str(contract)!r},'
            f'"--arm","baseline","--out",{str(out)!r}];'
            f'runpy.run_path({str(HARNESS/"h2_gpu_correctness.py")!r},run_name="__main__")')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(HARNESS)), timeout=10)
    assert result.returncode == 2
    evidence = json.loads((out/'evidence.json').read_text())
    assert evidence['status'] == 'FAILED'
    assert evidence['performance_verdict'] == 'NO_VERDICT'
    assert 'not granted' in evidence['error']
    pins = json.loads((out/'files_sha256.json').read_text())
    for name, sha in pins.items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == sha
    with pytest.raises(FileExistsError):
        runner.run({}, 'baseline', out)


def test_runner_import_keeps_gpu_stack_lazy():
    code = (f'import runpy,sys;runpy.run_path({str(HARNESS/"h2_gpu_correctness.py")!r});'
            'assert not any(n in sys.modules for n in '
            '["torch","trl","transformers","peft","unsloth","bitsandbytes"])')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True,
                   env=dict(os.environ, PYTHONPATH=str(HARNESS)), timeout=10)


@pytest.mark.parametrize('prefix', ['', 'GPU-'])
@pytest.mark.parametrize('uppercase', [False, True])
def test_live_gpu_uuid_formats_preserve_complete_identity(prefix, uppercase):
    raw = 'c78c68bf-6162-1d29-2f0e-00efc65ce2d6'
    contract = {'gpu_uuid': 'GPU-' + raw, 'boot_id': 'boot-fixed'}
    observed = prefix + (raw.upper() if uppercase else raw)
    runner.validate_live_identity(contract, 'boot-fixed', observed)


@pytest.mark.parametrize('observed', [None, '', '0', 'NVIDIA L4',
    'MIG-c78c68bf-6162-1d29-2f0e-00efc65ce2d6',
    'c78c68bf61621d292f0e00efc65ce2d6',
    'c78c68bf-6162-1d29-2f0e-00efc65ce2d6 ', '-' * 36])
def test_malformed_or_nonphysical_gpu_identity_is_rejected(observed):
    with pytest.raises(ValueError, match='UUID'):
        runner.canonical_gpu_uuid(observed)


@pytest.mark.parametrize('changed', ['boot', 'gpu'])
def test_live_identity_still_rejects_different_boot_or_single_uuid_bit(changed):
    raw = 'c78c68bf-6162-1d29-2f0e-00efc65ce2d6'
    contract = {'gpu_uuid': 'GPU-' + raw, 'boot_id': 'boot-fixed'}
    boot = 'boot-other' if changed == 'boot' else 'boot-fixed'
    observed = raw[:-1] + '7' if changed == 'gpu' else raw
    with pytest.raises(ValueError, match='differs'):
        runner.validate_live_identity(contract, boot, observed)
