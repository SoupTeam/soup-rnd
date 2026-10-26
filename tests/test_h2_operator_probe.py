"""Isolation/authority negative controls and a real CPU backward; no CUDA acceptance."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks/harness'))
import h2_operator_patch as patches  # noqa: E402
import h2_operator_probe as runner  # noqa: E402


def contract():
    now = time.time()
    study = json.loads((ROOT / 'benchmarks/h2/operator_study.json').read_text())
    value = {k: study[k] for k in ('model_revision', 'versions', 'model_files_sha256',
                                  'numerical_policy_sha256', 'recipe_sha256', 'fixture_sha256')}
    value.update(schema='h2-operator-authority-v1', execution_authorized=True,
                 license_and_access_confirmed=True, arms=list(patches.ARMS), instance_id=1,
                 gpu_uuid='GPU-00000000-0000-0000-0000-000000000001',
                 boot_id='00000000-0000-0000-0000-000000000001', max_seconds=300,
                 series_started_unix_s=now, deadline_unix_s=now + 2140,
                 budget_cap_usd=5, prior_spend_usd=0, hourly_rate_usd=1,
                 cleanup_traffic_reserve_usd=0.2)
    value['source_files_sha256'] = {n: runner.sha_file(ROOT / n) for n in runner.ARTIFACTS}
    value['runner_sha256'] = {n: runner.sha_file(ROOT / 'benchmarks/harness' / n)
                              for n in runner.SOURCES}
    return value


def test_complete_authority_passes_without_gpu_import():
    assert runner.validate_contract(contract())['arm_order'] == list(patches.ARMS)


@pytest.mark.parametrize('field,value', [
    ('execution_authorized', False), ('arms', ['baseline']), ('max_seconds', 301),
    ('budget_cap_usd', 5.01), ('versions', {}), ('model_revision', 'a' * 40),
    ('fixture_sha256', 'f' * 64), ('recipe_sha256', 'f' * 64), ('model_files_sha256', {}),
    ('numerical_policy_sha256', 'f' * 64), ('runner_sha256', {}),
    ('source_files_sha256', {}), ('series_started_unix_s', float('nan')),
    ('series_started_unix_s', 0), ('deadline_unix_s', time.time() + 5000),
    ('prior_spend_usd', 4.6),  # One arm fits; complete matrix does not.
])
def test_authority_refuses_drift_and_whole_series_overrun(field, value):
    data = contract()
    data[field] = value
    with pytest.raises(ValueError):
        runner.validate_contract(data)


def test_disabled_series_never_creates_child_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'bounded_process', lambda *a, **k: pytest.fail('child started'))
    with pytest.raises(ValueError, match='not granted'):
        runner.run_series({'execution_authorized': False}, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('arm', list(patches.ARMS))
def test_class_and_instance_calls_use_all_explicit_flags(arm):
    control = patches.PatchController.__new__(patches.PatchController)
    control.arm, control.flags = arm, patches.patch_flags(arm)
    control.counts = {'class': 0, 'instance': 0}
    calls = []
    control.patch = SimpleNamespace(apply_liger_kernel_to_llama=lambda **kw: calls.append(kw))
    control.unwrap = lambda model: model
    if arm == 'baseline':
        with pytest.raises(ValueError):
            control.class_patch('unused')
        with pytest.raises(ValueError):
            control.instance_patch(object(), None)
    else:
        model = object()
        assert control.class_patch('unused') is True
        control.instance_patch(model, None)
        assert calls == [control.flags, dict(control.flags, model=model)]
        with pytest.raises(ValueError):
            control.class_patch('unused')
        with pytest.raises(ValueError):
            control.instance_patch(model, {})
    patches.validate_counts(arm, control.counts)


def test_instance_cannot_sneak_in_default_extra_patch():
    control = patches.PatchController.__new__(patches.PatchController)
    control.arm, control.flags = 'ce_only', patches.patch_flags('ce_only')
    control.counts = {'class': 1, 'instance': 0}
    with pytest.raises(ValueError, match='configuration'):
        control.instance_patch(object(), {'fused_linear_cross_entropy': True})


def test_second_controller_refused_before_heavy_imports(monkeypatch):
    monkeypatch.setattr(patches, '_INSTALLED', True)
    with pytest.raises(ValueError, match='Fresh process'):
        patches.PatchController('baseline')


@pytest.mark.parametrize('arm', list(patches.ARMS))
def test_kernel_families_reject_missing_and_undeclared_execution(arm):
    flags = patches.patch_flags(arm)
    names = {'rope': '_triton_rope', 'rms_norm': '_rms_norm_forward_kernel',
             'swiglu': '_swiglu_forward_kernel', 'cross_entropy': 'liger_cross_entropy_kernel'}
    selected = [name for key, name in names.items() if flags[key] or (
        key == 'cross_entropy' and flags['fused_linear_cross_entropy'])]
    events = [{'cat': 'kernel', 'name': name} for name in ['gemm'] + selected]
    assert patches.kernel_audit(events, arm)['status'] == 'PASS_DECLARED_KERNEL_FAMILIES'
    if selected:
        with pytest.raises(ValueError):
            patches.kernel_audit(events[:-1], arm)
    extra = next((n for n in names.values() if n not in selected), None)
    if extra:
        with pytest.raises(ValueError):
            patches.kernel_audit(events + [{'cat': 'kernel', 'name': extra}], arm)
    with pytest.raises(ValueError, match='No executed'):
        patches.kernel_audit([{'cat': 'cpu_op', 'name': 'liger_cross_entropy_kernel'}], arm)


def tiny_model():
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.0,
    )), LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    batch = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
             'attention_mask': torch.ones((1, 5), dtype=torch.long),
             'labels': torch.tensor([[-100, -100, 3, 4, 5]])}
    return model, names, batch


def test_real_cpu_capture_matches_frozen_collector_and_preserves_state(tmp_path):
    import torch
    from h2_numerics import collect_probe, compare_probes
    from h2_operator_capture import collect, identity
    from h2_update_oracle import snapshot

    model, names, batch = tiny_model()
    model.eval()
    state = torch.random.get_rng_state().clone()
    expected = identity(snapshot(model, names), batch)
    collect_probe(model, batch, names, tmp_path / 'old', bf16=False)
    result = collect(model, batch, names, tmp_path / 'new', cuda=False,
                     expected_identity=expected)
    policy = json.loads((ROOT / 'benchmarks/h2/numerical_policy.json').read_text())
    pair = compare_probes(tmp_path / 'old', tmp_path / 'new/model_probe', policy)
    assert pair['status'] == 'PASS_MODEL_PROBE_TOLERANCES'
    assert result['optimizer_updates'] == 0 and result['backward_calls'] == 1
    assert result['state_unchanged'] and not model.training
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    trace = json.loads((tmp_path / 'new/kernel_trace.json').read_text())
    assert sum(e.get('name') == 'h2_operator_probe_forward_backward'
               for e in trace['traceEvents']) == 1
    with pytest.raises(ValueError, match='new output'):
        collect(model, batch, names, tmp_path / 'new', cuda=False)


@pytest.mark.parametrize('field', ['before_sha256', 'work', 'input_sha256'])
def test_wrong_initial_identity_rejected_before_forward(tmp_path, field):
    from h2_operator_capture import collect, identity
    from h2_update_oracle import snapshot

    model, names, batch = tiny_model()
    expected = identity(snapshot(model, names), batch)
    expected[field] = 'bad'
    handle = model.register_forward_pre_hook(lambda *a: pytest.fail('forward called'))
    try:
        with pytest.raises(ValueError, match='identity differs'):
            collect(model, batch, names, tmp_path / 'bad', cuda=False,
                    expected_identity=expected)
    finally:
        handle.remove()
    assert all(p.grad is None for p in model.parameters())


def test_disabled_child_preserves_error_without_model_import(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, '_USED', False)
    result = runner.run_child({'execution_authorized': False}, 'baseline', tmp_path / 'failed')
    assert result['status'] == 'FAILED' and 'not granted' in result['error']
    pins = json.loads((tmp_path / 'failed/files_sha256.json').read_text())
    assert pins and all(runner.sha_file(tmp_path / 'failed' / n) == sha
                        for n, sha in pins.items())
    with pytest.raises(ValueError, match='fresh process'):
        runner.run_child({'execution_authorized': False}, 'baseline', tmp_path / 'retry')
    assert not (tmp_path / 'retry').exists()


@pytest.mark.parametrize('outcome,expected_calls,expected_status', [
    ('treatment_fail', 7, 'COMPLETE_DIAGNOSTIC_MATRIX'),
    ('baseline_fail', 1, 'BLOCKED'), ('timeout', 2, 'BLOCKED'),
    ('log_limit', 2, 'BLOCKED'), ('child_error', 2, 'BLOCKED'),
    ('cleanup_missing', 2, 'BLOCKED'),
])
def test_supervised_matrix_stops_on_infra_but_records_treatment_mismatch(
        monkeypatch, tmp_path, outcome, expected_calls, expected_status):
    import h2_numerics

    historical = tmp_path / 'historical'
    historical.mkdir()
    (historical / 'files_sha256.json').write_text('{}')
    study = {'historical_reference_manifest_sha256': runner.sha_file(
        historical / 'files_sha256.json')}
    monkeypatch.setattr(runner, 'validate_contract', lambda contract: study)
    monkeypatch.setattr(h2_numerics, 'read_probe', lambda path: ({}, {}))
    calls = []
    comparisons = []

    def supervised(argv, out, **kwargs):
        calls.append(argv[argv.index('--arm') + 1])
        assert 0 < kwargs['seconds'] <= 300
        out.mkdir()
        second = len(calls) == 2
        return {'status': ('TIMEOUT' if outcome == 'timeout' else 'LOG_LIMIT')
                if second and outcome in {'timeout', 'log_limit'} else 'CHILD_EXITED',
                'group_cleanup_confirmed': not (second and outcome == 'cleanup_missing'),
                'child_status': {'returncode': int(second and outcome == 'child_error')}}

    def compare(reference, candidate, policy):
        comparisons.append((reference, candidate))
        failure = outcome == 'baseline_fail' or (outcome == 'treatment_fail'
                                                 and len(comparisons) == 2)
        return {'status': 'FAIL' if failure else 'PASS_MODEL_PROBE_TOLERANCES'}

    monkeypatch.setattr(runner, 'bounded_process', supervised)
    monkeypatch.setattr(h2_numerics, 'compare_probes', compare)
    root = tmp_path / 'series'
    value = {'evidence_root': str(root), 'max_seconds': 300,
             'deadline_unix_s': time.time() + 2100, 'historical_reference_probe': str(historical)}
    result = runner.run_series(value, root)
    assert len(calls) == expected_calls
    assert result['status'] == expected_status and result['performance_verdict'] == 'NO_VERDICT'
    assert not result['provider_stop_performed']
    assert comparisons[0][0] == historical
    if len(comparisons) > 1:
        assert comparisons[1][0] == root / 'baseline/runner/capture/model_probe'
    if outcome == 'treatment_fail':
        assert result['arms']['ce_only'] == 'FAIL'
    pins = json.loads((root / 'files_sha256.json').read_text())
    assert all(runner.sha_file(root / n) == sha for n, sha in pins.items())
    with pytest.raises(FileExistsError):
        runner.run_series(value, root)


def fake_bindings(arm):
    """Control-flow doubles only; these classes are never claimed as Liger kernels."""
    class OriginalNorm:
        def forward(self):
            pass

    class OriginalMLP:
        def forward(self):
            pass

    class TreatmentNorm:
        def forward(self):
            pass

    class TreatmentMLP:
        def forward(self):
            pass

    def original_forward(self):
        pass

    def treatment_forward(self):
        pass

    original = {'rope': lambda: None, 'rms_norm': OriginalNorm, 'swiglu': OriginalMLP,
                'fused_linear_cross_entropy': original_forward, 'cross_entropy': lambda: None}
    treatment = {'rope': lambda: None, 'rms_norm': TreatmentNorm, 'swiglu': TreatmentMLP,
                 'fused_linear_cross_entropy': treatment_forward, 'cross_entropy': lambda: None}
    flags = patches.patch_flags(arm)
    expected = {k: treatment[k] if flags[k] else v for k, v in original.items()}
    model_type = type('Model', (), {'forward': expected['fused_linear_cross_entropy']})
    model = model_type()
    model.model = SimpleNamespace(norm=expected['rms_norm'](), layers=[
        SimpleNamespace(input_layernorm=expected['rms_norm'](),
                        post_attention_layernorm=expected['rms_norm'](),
                        mlp=expected['swiglu']()) for _ in range(32)])
    control = patches.PatchController.__new__(patches.PatchController)
    control.arm, control.flags = arm, flags
    control.original, control.treatment = original, treatment
    control.counts = {'class': int(arm != 'baseline'), 'instance': int(arm != 'baseline')}
    control.modeling = SimpleNamespace(
        apply_rotary_pos_emb=expected['rope'], LlamaRMSNorm=expected['rms_norm'],
        LlamaMLP=expected['swiglu'], LlamaForCausalLM=model_type)
    control.functional = SimpleNamespace(cross_entropy=expected['cross_entropy'])
    control.unwrap = lambda model: model
    control.patch = SimpleNamespace(LigerRMSNorm=TreatmentNorm, LigerSwiGLUMLP=TreatmentMLP)
    return control, model


@pytest.mark.parametrize('arm', list(patches.ARMS))
def test_exact_patch_binding_and_instance_audit(arm):
    control, model = fake_bindings(arm)
    assert control.verify(model)['status'] == 'PASS_EXPLICIT_PATCH_BINDINGS'


@pytest.mark.parametrize('fault', ['extra_global', 'wrong_model_forward', 'missing_module',
                                   'extra_module_patch', 'wrong_instance_forward', 'missing_call'])
def test_patch_audit_refuses_silent_treatment_changes(fault):
    control, model = fake_bindings('ce_only')
    if fault == 'extra_global':
        control.modeling.apply_rotary_pos_emb = control.treatment['rope']
    elif fault == 'wrong_model_forward':
        model.forward = lambda: None
    elif fault == 'missing_module':
        model.model.layers.pop()
    elif fault == 'extra_module_patch':
        model.model.norm = control.treatment['rms_norm']()
    elif fault == 'wrong_instance_forward':
        model.model.norm.forward = lambda: None
    else:
        control.counts['instance'] = 0
    with pytest.raises(ValueError):
        control.verify(model)
