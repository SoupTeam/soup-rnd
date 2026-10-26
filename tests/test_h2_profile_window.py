"""Bounded attribution must preserve training and fail closed on missing evidence."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
HARNESS = ROOT / 'benchmarks/harness'
sys.path.insert(0, str(HARNESS))
from h2_profile_window import ProfileWindow  # noqa: E402
from h2_trainer_smoke import recording_trainer  # noqa: E402


def test_only_selected_backward_is_profiled_checks_and_optimizer_still_run(tmp_path):
    torch = pytest.importorskip('torch')
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    records = {'expected_names': ['weight', 'bias'], 'batches': [], 'gradients': [],
               'optimizer_events': [], 'observed_optimizers': set()}
    window = ProfileWindow(tmp_path, cuda=False)

    class Base:
        def create_optimizer(self):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)

        def training_step(self, model, index):
            with torch.profiler.record_function(f'batch_{index}'):
                loss = model(torch.ones(1, 2)).square().mean()
                (loss / 4).backward()
            records['batches'].append({'loss': loss.item()})
            return loss.detach()

    trainer = recording_trainer(Base, records, window)()
    trainer.model = model
    trainer.create_optimizer()
    for index in range(1, 129):
        trainer.training_step(model, index)
        if index % 4 == 0:
            with torch.profiler.record_function('optimizer_outside_window'):
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
    assert len(records['batches']) == len(records['gradients']) == 128
    assert len(records['optimizer_events']) == 32
    assert all(row['finite'] for row in records['gradients'])
    evidence = window.require_complete()
    assert evidence['microbatch_1based'] == 5
    names = {event['name'] for event in json.loads(
        (tmp_path / 'kernel_trace.json').read_text())['traceEvents']}
    assert {name for name in names if name.startswith('batch_')} == {'batch_5'}
    assert 'optimizer_outside_window' not in names
    assert any('Backward' in name for name in names)
    assert not any(name.startswith('aten::isfinite') for name in names)
    assert evidence['trace_sha256'] == ProfileWindow._sha(tmp_path / 'kernel_trace.json')


def test_short_training_cannot_claim_required_trace(tmp_path):
    window = ProfileWindow(tmp_path, cuda=False)
    for _ in range(4):
        with window.step():
            pass
    with pytest.raises(ValueError, match='not completed'):
        window.require_complete()
    assert not list(tmp_path.iterdir())


def test_selected_step_failure_cannot_be_accepted_or_retried(tmp_path):
    pytest.importorskip('torch')
    window = ProfileWindow(tmp_path, cuda=False, microbatch=1)
    with pytest.raises(RuntimeError, match='training failed'):
        with window.step():
            raise RuntimeError('training failed')
    with window.step():
        pass
    with pytest.raises(ValueError, match='not completed'):
        window.require_complete()
    assert not (tmp_path / 'profile_window.json').exists()


def test_existing_trace_is_never_overwritten(tmp_path):
    pytest.importorskip('torch')
    path = tmp_path / 'kernel_trace.json'
    path.write_text('prior evidence')
    window = ProfileWindow(tmp_path, cuda=False, microbatch=1)
    with pytest.raises(FileExistsError):
        with window.step():
            pytest.fail('must reject before training')
    assert path.read_text() == 'prior evidence'


def test_export_failure_is_fatal_not_silently_unprofiled(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')

    def failed(*args):
        raise OSError('export failed')

    monkeypatch.setattr(torch.profiler.profile, 'export_chrome_trace', failed)
    window = ProfileWindow(tmp_path, cuda=False, microbatch=1)
    with pytest.raises(OSError, match='export failed'):
        with window.step():
            torch.ones(1).add_(1)
    with pytest.raises(ValueError, match='not completed'):
        window.require_complete()


def test_real_soup_training_is_identical_with_bounded_attribution(tmp_path):
    for dependency in ['torch', 'transformers', 'trl', 'peft', 'datasets', 'tokenizers']:
        pytest.importorskip(dependency)
    evidence = []
    for mode in ['original', 'bounded']:
        output = tmp_path / mode
        command = [sys.executable, str(HARNESS / 'h2_trainer_smoke.py'), '--out', str(output)]
        if mode == 'bounded':
            command.append('--bounded-profile')
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        evidence.append(json.loads((output / 'evidence.json').read_text()))
    left, right = evidence
    for key in ['before', 'after', 'records', 'work_audit', 'audit', 'trainer_global_step']:
        if key == 'records':
            for subkey in ['batches', 'gradients', 'optimizer_events']:
                assert left[key][subkey] == right[key][subkey]
        else:
            assert left[key] == right[key]
    assert right['audit']['status'] == 'PASS_UPDATE_EVIDENCE'
    assert right['profile_window']['recorded_microbatches'] == 1
    assert right['performance_verdict'] == 'NO_VERDICT'
