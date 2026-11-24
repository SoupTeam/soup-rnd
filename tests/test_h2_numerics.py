"""Real tiny PEFT forward/backward plus independent faulty-backend controls."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / 'benchmarks/harness'
sys.path.insert(0, str(HARNESS))
spec = importlib.util.spec_from_file_location('h2_numerics', HARNESS/'h2_numerics.py')
numerics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numerics)
POLICY = json.loads((HARNESS.parent/'h2/numerical_policy.json').read_text())


@pytest.fixture
def pair(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from h2_gpu_correctness import TARGETS
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         attention_dropout=0.0)
    model = get_peft_model(LlamaForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, target_modules=list(TARGETS), lora_dropout=0.0,
    ))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    batch = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
             'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]),
             'labels': torch.tensor([[-100, -100, 3, 4, 5]])}
    model.eval()
    state = torch.random.get_rng_state().clone()
    for arm in ('reference', 'candidate'):
        numerics.collect_probe(model, batch, names, tmp_path/arm, bf16=False)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not model.training
    assert all(p.grad is None for p in model.parameters())
    return tmp_path/'reference', tmp_path/'candidate'


def rewrite(candidate, change):
    """Re-sign local fixture to model a numerically wrong backend, not corrupt transport."""
    from safetensors.torch import load_file, save_file

    metadata = json.loads((candidate/'probe.json').read_text())
    values = load_file(str(candidate/'probe.safetensors'))
    change(metadata, values)
    metadata['tensor_sha256'] = {k: numerics.tensor_sha(v) for k, v in values.items()}
    save_file(values, str(candidate/'probe.safetensors'))
    (candidate/'probe.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    pins = {name: hashlib.sha256((candidate/name).read_bytes()).hexdigest()
            for name in ['probe.json', 'probe.safetensors']}
    (candidate/'files_sha256.json').write_text(json.dumps(pins, indent=2)+'\n')


def test_real_probe_pair_passes_without_optimizer_or_rng_side_effects(pair):
    result = numerics.compare_probes(*pair, POLICY)
    assert result['status'] == 'PASS_MODEL_PROBE_TOLERANCES'
    assert result['performance_verdict'] == 'NO_VERDICT'
    assert any(not row['reference_zero'] for row in result['gradient_metrics'].values())


@pytest.mark.parametrize('problem', ['loss', 'gradient_sign', 'gradient_scale', 'initial_state',
                                    'zero_gradient_leak', 'inputs'])
def test_wrong_backend_rejected_even_with_resigned_transport_and_equal_step_count(pair, problem):
    def change(metadata, values):
        if problem == 'loss':
            metadata['loss'] += 1.0
        elif problem == 'initial_state':
            metadata['before']['frozen']['parameter:unexpected'] = 'f'*64
        elif problem == 'inputs':
            values['input.input_ids'][0, 0] += 1
            metadata['work'] = numerics.batch_evidence({k.removeprefix('input.'): v
                                                       for k, v in values.items()
                                                       if k.startswith('input.')})
        else:
            for name, tensor in values.items():
                if not name.startswith('gradient.'):
                    continue
                if problem == 'zero_gradient_leak' and not tensor.any():
                    tensor.reshape(-1)[0] = 1e-12
                    return
                if problem != 'zero_gradient_leak' and tensor.any():
                    tensor.mul_(-1 if problem == 'gradient_sign' else 2)
                    return
            raise AssertionError('Expected gradient control tensor not found')
    rewrite(pair[1], change)
    assert numerics.compare_probes(*pair, POLICY)['status'] == 'FAIL'


def test_corrupt_payload_fails_before_numerical_comparison(pair):
    path = pair[1]/'probe.safetensors'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='SHA mismatch'):
        numerics.compare_probes(*pair, POLICY)


@pytest.mark.parametrize('value', [True, -1, float('nan'), float('inf')])
def test_invalid_policy_refused(value):
    policy = {**POLICY, 'gradient_rtol': value}
    with pytest.raises(ValueError):
        numerics.validate_policy(policy)


def test_gradient_overflow_cannot_turn_into_a_nan_pass():
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='overflow'):
        numerics.gradient_metrics(torch.tensor([1e300], dtype=torch.float64),
                                  torch.tensor([-1e300], dtype=torch.float64))
