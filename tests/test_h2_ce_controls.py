"""Fail closed on missing or contaminated per-process CE controls."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/harness'))
from h2_ce_controls import validate_controls  # noqa: E402


def good_record() -> dict:
    return {
        'phase': 'baseline_probes',
        'ownership': {'method': 'exclusive-local-CUDA-initialization',
                      'before': [], 'after': [123], 'context_active': True,
                      'single_device': True},
        'gpu_pid_samples': [{'at_s': 0.0, 'reading': [123], 'problem': None},
                            {'at_s': 3.0, 'reading': [123], 'problem': None}],
        'f1': {'outcome': 'ok', 'required': ['suspend', 'power', 'foreign_reads'],
               'checks': {key: {'outcome': 'ok'}
                          for key in ('suspend', 'power', 'foreign_reads')},
               'gpu_clock': {'outcome': 'ok', 'samples': 2, 'unknown_samples': 0,
                             'sm_clock_mhz': {'min': 210.0, 'median': 1500.0, 'max': 2500.0},
                             'reasons_seen': {'gpu_idle': 1, 'sw_power_cap': 1}}},
    }


def test_valid_controls_and_recorded_throttling() -> None:
    # Clock reasons are retained, not interpreted as a speed verdict.
    validate_controls(good_record(), 'baseline_probes')


@pytest.mark.parametrize('fault', [
    'phase', 'busy_start', 'unknown_owner', 'multiple_owner', 'unknown_pid',
    'foreign_pid', 'empty_pid', 'one_sample', 'missing_check', 'unknown_f1',
    'void_f1', 'missing_clock', 'unknown_clock', 'zero_clock', 'nan_clock',
])
def test_invalid_controls(fault: str) -> None:
    data = copy.deepcopy(good_record())
    if fault == 'phase':
        data['phase'] = 'ce_only_training'
    elif fault == 'busy_start':
        data['ownership']['before'] = [123]
    elif fault == 'unknown_owner':
        data['ownership']['method'] = 'unresolved'
    elif fault == 'multiple_owner':
        data['ownership']['after'] = [123, 456]
    elif fault == 'unknown_pid':
        data['gpu_pid_samples'][1].update(reading=None, problem='unavailable')
    elif fault in ('foreign_pid', 'empty_pid'):
        data['gpu_pid_samples'][1]['reading'] = [123, 456] if fault == 'foreign_pid' else []
    elif fault == 'one_sample':
        data['gpu_pid_samples'].pop()
    elif fault == 'missing_check':
        del data['f1']['checks']['foreign_reads']
    elif fault in ('unknown_f1', 'void_f1'):
        data['f1']['checks']['foreign_reads']['outcome'] = fault.split('_')[0]
    elif fault == 'missing_clock':
        del data['f1']['gpu_clock']
    elif fault == 'unknown_clock':
        data['f1']['gpu_clock']['unknown_samples'] = 1
    elif fault in ('zero_clock', 'nan_clock'):
        value = 0 if fault == 'zero_clock' else float('nan')
        data['f1']['gpu_clock']['sm_clock_mhz']['min'] = value
    with pytest.raises(ValueError):
        validate_controls(data, 'baseline_probes')


def test_stop_keeps_f1_when_pid_sampler_fails(monkeypatch) -> None:
    from types import SimpleNamespace

    from h2_ce_controls import CaptureControls

    control = CaptureControls('baseline_probes', '/model')
    stopped = []

    def fail() -> None:
        raise RuntimeError('sensor failed')

    control.pids = SimpleNamespace(stop=fail)
    control.watch = SimpleNamespace(stop=lambda: stopped.append(True) or {'outcome': 'unknown'})
    with pytest.raises(RuntimeError, match='sensor failed'):
        control.stop()
    assert stopped == [True]
    assert control.record['f1']['outcome'] == 'unknown'
    assert control.record['status'] == 'FAILED'


@pytest.mark.parametrize('reading', [None, [99]])
def test_busy_or_unknown_gpu_never_initializes_cuda(monkeypatch, reading) -> None:
    import h2_ce_controls as controls

    monkeypatch.setattr(controls, 'gpu_compute_pids', lambda: reading)
    control = controls.CaptureControls('baseline_probes', '/model')
    with pytest.raises(ValueError, match='not observably idle'):
        control.start()
    assert control.watch is None and control.pids is None
