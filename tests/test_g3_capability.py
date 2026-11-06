"""G3 inventory must not confuse discovery, fallback or a broken setup with DMA proof."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

COLLECTOR = Path(__file__).parents[1] / 'benchmarks/harness/g3_capability.py'
spec = importlib.util.spec_from_file_location('g3_capability', COLLECTOR)
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)


def success():
    return {'status': 'COMPLETED', 'returncode': 0, 'stdout': 'diagnostic output',
            'stderr': '', 'truncated': False}


@pytest.mark.parametrize('system', ['Darwin', 'Windows', 'Unknown'])
def test_non_linux_is_outside_scope_without_running_commands(tmp_path, system):
    def forbidden(*args):
        pytest.fail('Non-Linux inventory must not query CUDA or launch commands')
    report = g3.collect_inventory(tmp_path, system=system, runner=forbidden, which=forbidden)
    assert report['classification']['status'] == 'OUT_OF_SCOPE_PLATFORM'
    assert report['classification']['support_verdict'] == 'NO_VERDICT'


@pytest.mark.parametrize('name', ['gpu', 'mount', 'block_devices', 'gdscheck'])
@pytest.mark.parametrize('problem', ['missing', 'timeout', 'error', 'empty', 'truncated'])
def test_incomplete_diagnostics_are_blocked_not_unsupported(name, problem):
    rows = {key: success() for key in ['gpu', 'mount', 'block_devices', 'gdscheck']}
    if problem == 'missing':
        del rows[name]
    elif problem == 'timeout':
        rows[name]['status'] = 'TIMEOUT'
    elif problem == 'error':
        rows[name]['returncode'] = 1
    elif problem == 'empty':
        rows[name]['stdout'] = ''
    else:
        rows[name]['truncated'] = True
    verdict = g3.classify_inventory('Linux', rows)
    assert verdict['status'] == 'ENVIRONMENT_BLOCKED'
    assert verdict['support_verdict'] == 'NO_VERDICT'
    assert not verdict['direct_path_verified']


@pytest.mark.parametrize('diagnostic', [
    'GDS supported', 'GDS unsupported', 'nvidia-fs absent', 'PCI P2PDMA enabled',
    'allow_compat_mode=true', 'cuFileRead succeeded',
])
def test_text_markers_cannot_be_promoted_to_dma_verdict(diagnostic):
    rows = {key: success() for key in ['gpu', 'mount', 'block_devices', 'gdscheck']}
    rows['gdscheck']['stdout'] = diagnostic
    verdict = g3.classify_inventory('Linux', rows)
    assert verdict['status'] == 'CONTROLLED_PROBE_REQUIRED'
    assert verdict['support_verdict'] == 'NO_VERDICT'
    assert not verdict['direct_path_verified'] and not verdict['performance_measured']


def test_linux_uses_target_directory_and_only_expected_diagnostics(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(g3, 'read_small', lambda path: {'path': str(path), 'status': 'UNAVAILABLE'})
    def runner(argv):
        seen.append(argv)
        return success()
    g3.collect_inventory(
        tmp_path, system='Linux', runner=runner,
        which=lambda name: '/tools/gdscheck' if name == 'gdscheck' else None,
    )
    assert len(seen) == 6
    assert any(argv[:3] == ['findmnt', '--json', '--target'] and str(tmp_path) in argv
               for argv in seen)
    assert ['/tools/gdscheck', '-p'] in seen
    assert all(argv[0] in ['nvidia-smi', 'findmnt', 'lsblk', 'lspci', '/tools/gdscheck']
               for argv in seen)


def test_missing_data_path_cannot_probe_parent_filesystem(tmp_path):
    with pytest.raises(ValueError):
        g3.collect_inventory(tmp_path / 'absent')
    data = tmp_path / 'file'
    data.write_bytes(b'not a directory')
    with pytest.raises(ValueError):
        g3.collect_inventory(data)


def test_bundle_hashes_actual_bytes_and_refuses_overwrite(tmp_path):
    out = tmp_path / 'evidence'
    report = {'schema': 'test', 'result': 'NO_VERDICT'}
    g3.write_bundle(out, report)
    pins = json.loads((out / 'files_sha256.json').read_text())
    for name, sha in pins.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    old = (out / 'inventory.json').read_bytes()
    with pytest.raises(FileExistsError):
        g3.write_bundle(out, {'result': 'changed'})
    assert (out / 'inventory.json').read_bytes() == old


def test_raw_command_error_and_timeout_are_preserved():
    error = g3.run_command([sys.executable, '-c', 'import sys;sys.stderr.write("bad");sys.exit(7)'])
    assert error['returncode'] == 7 and error['stderr'] == 'bad'
    timed = g3.run_command([sys.executable, '-c', 'import time;time.sleep(5)'], timeout=0.05)
    assert timed['status'] == 'TIMEOUT' and timed['returncode'] != 0
    missing = g3.run_command(['/nonexistent/g3-program'])
    assert missing['status'] == 'UNAVAILABLE'


def test_excess_output_is_flagged_and_bounded():
    result = g3.run_command([sys.executable, '-c', f'print("x"*{g3.OUTPUT_LIMIT + 10})'])
    assert result['truncated'] is True
    assert len(result['stdout']) == g3.OUTPUT_LIMIT


def test_import_does_not_initialize_training_or_cuda():
    code = (f'import runpy,sys;runpy.run_path({str(COLLECTOR)!r});'
            'assert not any(n in sys.modules for n in ["torch","transformers","kvikio","cupy"])')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=10)
