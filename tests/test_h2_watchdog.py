"""Real owned-process tests for timeout, descendants, isolation and bounded logs."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX sessions and killpg required')
HARNESS = Path(__file__).parents[1] / 'benchmarks/harness'
sys.path.insert(0, str(HARNESS))
spec = importlib.util.spec_from_file_location('h2_watchdog', HARNESS/'h2_watchdog.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


def run(code, out, **kwargs):
    return watchdog.bounded_process([sys.executable, '-c', code], out,
                                    grace_seconds=0.1, **kwargs)


def test_normal_exit_and_nonzero_exit_are_distinguished_and_evidence_retained(tmp_path):
    for status in (0, 7):
        out = tmp_path/str(status)
        result = run(f'import sys;sys.stdout.write("evidence\\n");sys.exit({status})',
                     out, seconds=2)
        assert result['status'] == 'CHILD_EXITED'
        assert result['child_status']['returncode'] == status
        assert result['new_session_verified'] and result['group_cleanup_confirmed']
        assert result['leader_reaped_after_cleanup']
        assert (out/'stdout.log').read_text() == 'evidence\n'
        assert json.loads((out/'watchdog.json').read_text()) == result
        with pytest.raises(FileExistsError):
            run('pass', out, seconds=2)


@pytest.mark.parametrize('parent_exits', [False, True])
def test_cleanup_kills_owned_descendants_and_leaves_separate_process_alive(tmp_path, parent_exits):
    outside = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'],
                               start_new_session=True)
    grandchild = ('import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                  'time.sleep(30)')
    code = ('import subprocess,sys,signal,time,json,os;'
            'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
            f'child=subprocess.Popen([sys.executable,"-c",{grandchild!r}]);'
            'sys.stdout.write(json.dumps([os.getpid(),child.pid])+"\\n");sys.stdout.flush();'
            + ('sys.exit(0)' if parent_exits else 'time.sleep(30)'))
    try:
        result = run(code, tmp_path/'timeout', seconds=0.8)
        assert result['status'] == ('CHILD_EXITED' if parent_exits else 'TIMEOUT')
        assert result['term_sent_to_owned_group'] and result['kill_sent_to_owned_group']
        assert result['elapsed_seconds'] < 5
        pids = json.loads((tmp_path/'timeout/stdout.log').read_text())
        for pid in pids:
            # A reparented killed descendant may remain a zombie until init reaps it.
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                   capture_output=True, text=True, timeout=2).stdout.strip()
            assert not state or state.startswith('Z'), state
        assert outside.poll() is None
    finally:
        outside.terminate()
        outside.wait(timeout=5)


def test_stdout_flood_is_rejected_with_bounded_retained_bytes(tmp_path):
    result = run('import os,time;os.write(1,b"x"*1000000);time.sleep(30)',
                 tmp_path/'flood', seconds=2, max_log_bytes=4096)
    assert result['status'] == 'LOG_LIMIT'
    assert (tmp_path/'flood/stdout.log').stat().st_size == 4096
    assert result['group_cleanup_confirmed']


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf'), 901, True])
def test_invalid_deadline_rejected_without_starting_a_process(tmp_path, seconds):
    with pytest.raises(ValueError, match='Finite bounded'):
        run('pass', tmp_path/'unused', seconds=seconds)
    assert not (tmp_path/'unused').exists()


def test_disabled_authority_cannot_launch_watchdog(tmp_path):
    contract = tmp_path/'disabled.json'
    contract.write_text('{"execution_authorized":false}')
    out = tmp_path/'unused'
    result = subprocess.run([sys.executable, str(HARNESS/'h2_watchdog.py'),
                             '--contract', str(contract), '--arm', 'baseline', '--out', str(out)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert not out.exists()
    assert 'not granted' in result.stderr
