"""Real CPU NF4/double-quant controls, not GPU/kernel-performance evidence."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / 'benchmarks/harness'
sys.path.insert(0, str(HARNESS))
spec = importlib.util.spec_from_file_location('h2_quant_state', HARNESS/'h2_quant_state.py')
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)


def model_and_step():
    torch = pytest.importorskip('torch')
    pytest.importorskip('bitsandbytes')
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit
    from bitsandbytes.nn import Params4bit
    from h2_update_oracle import gradients, snapshot

    try:
        packed, state = quantize_4bit(torch.linspace(-1, 1, 128).reshape(8, 16),
                                     quant_type='nf4', compress_statistics=True)
    except NotImplementedError:
        pytest.skip('Installed bitsandbytes does not implement CPU NF4 quantization')

    class QuantizedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Params4bit(packed, requires_grad=False, quant_state=state,
                                     compress_statistics=True, quant_type='nf4',
                                     quant_storage=torch.uint8, bnb_quantized=True)
            self.adapter = torch.nn.Parameter(torch.zeros(8, 16))

        def forward(self, inputs):
            base = dequantize_4bit(self.weight, self.weight.quant_state)
            return torch.nn.functional.linear(inputs, base + self.adapter)

    model = QuantizedModel()
    before = snapshot(model, ['adapter'])
    optimizer = torch.optim.SGD([model.adapter], lr=0.01)
    loss = model(torch.ones(1, 16)).square().mean()
    loss.backward()
    row = gradients(model, ['adapter'])
    optimizer.step()
    return model, before, row, loss.item()


def test_real_cpu_quantization_and_update_preserve_nested_decode_state(tmp_path):
    from h2_update_oracle import audit_update, snapshot

    model, before, row, loss = model_and_step()
    state = before['quantizer']['parameters']['weight']['state']
    assert state['nested'] and state['state2']['absmax']['sha256']
    assert state['offset']['dtype'] == 'torch.float32'
    after = snapshot(model, ['adapter'])
    assert before['quantizer'] == after['quantizer']
    audit = audit_update(before, after, [row], 1, [loss])
    assert audit['status'] == 'PASS_UPDATE_EVIDENCE'
    (tmp_path / 'nf4_cpu_evidence.json').write_text(json.dumps({
        'scope': 'Real CPU NF4/double-quant 8x16 weight and dense adapter; not LoRA/8B/CUDA',
        'before': before, 'after': after, 'gradient': row, 'loss': loss, 'audit': audit,
        'performance_verdict': 'NO_VERDICT',
    }, indent=2, allow_nan=False) + '\n')


@pytest.mark.parametrize('field', ['absmax', 'code', 'nested_absmax', 'nested_code', 'offset',
                                  'blocksize', 'dtype', 'shape'])
def test_hidden_state_mutation_fails_even_when_registered_weights_buffers_unchanged(field):
    torch = pytest.importorskip('torch')
    from h2_update_oracle import audit_update, snapshot

    model, before, row, loss = model_and_step()
    state = model.weight.quant_state
    if field == 'blocksize':
        state.state2.blocksize *= 2
    elif field == 'dtype':
        state.state2.dtype = torch.float64
    elif field == 'shape':
        state.shape = torch.Size([16, 8])
    else:
        owner = state.state2 if field.startswith('nested_') else state
        key = field.removeprefix('nested_')
        tensor = getattr(owner, key)
        with torch.no_grad():
            tensor.reshape(-1)[0] += 1
    after = snapshot(model, ['adapter'])
    assert before['frozen'] == after['frozen']
    audit = audit_update(before, after, [row], 1, [loss])
    assert audit['status'] == 'FAIL'
    assert 'NF4 quantizer state changed' in audit['reasons']


@pytest.mark.parametrize('problem', ['unknown_field', 'uninitialized', 'fp4', 'nan',
                                    'inconsistent_nested', 'unquantized'])
def test_unknown_or_invalid_quantizer_refused(problem):
    torch = pytest.importorskip('torch')
    model, _, _, _ = model_and_step()
    if problem == 'unknown_field':
        model.weight.quant_state.extra_scale = torch.ones(1)
    elif problem == 'uninitialized':
        model.weight.quant_state = None
    elif problem == 'fp4':
        model.weight.quant_type = 'fp4'
    elif problem == 'nan':
        model.weight.quant_state.offset.fill_(float('nan'))
    elif problem == 'inconsistent_nested':
        model.weight.quant_state.nested = False
    else:
        model.weight.bnb_quantized = False
    with pytest.raises(ValueError):
        quant.quantizer_snapshot(model)
