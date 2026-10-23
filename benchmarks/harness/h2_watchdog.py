"""Bound an owned H2 process group. Does not stop provider billing or escaped sessions."""
from __future__ import annotations

import argparse
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

# Keep the session leader alive until the supervisor finishes group cleanup.
# The supervisor never polls/waits/reaps this leader before its final group kill:
# its PID remains reserved, so killpg cannot target a reused group identity.
GUARD = '''
import json,signal,subprocess,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *args: None)
child = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL)
status = child.wait()
with Path(sys.argv[1]).open('x') as stream:
    json.dump({'returncode':status,'child_pid':child.pid},stream)
while True:
    time.sleep(3600)
'''


def bounded_process(argv: list[str], out: Path, *, seconds: float,
                    grace_seconds: float = 2.0, max_log_bytes: int = 16 * 1024**2) -> dict:
    """POSIX supervision of only the newly created session, with bounded raw logs."""
    if os.name != 'posix':
        raise ValueError('POSIX process groups required')
    if (type(seconds) not in (float, int) or not math.isfinite(seconds) or not 0 < seconds <= 900
            or type(grace_seconds) not in (float, int) or not math.isfinite(grace_seconds)
            or not 0 < grace_seconds <= 5 or type(max_log_bytes) is not int
            or not 1 <= max_log_bytes <= 64 * 1024**2):
        raise ValueError('Finite bounded deadline/grace/log limits required')
    if not argv or not all(isinstance(value, str) and value for value in argv):
        raise ValueError('Explicit nonempty argv required; no shell')
    out.mkdir(parents=True, exist_ok=False)
    status_path = out / 'child_status.json'
    result = {'status': 'SUPERVISOR_ERROR', 'argv': argv, 'performance_verdict': 'NO_VERDICT',
              'provider_stop_performed': False, 'group_cleanup_confirmed': False}
    started = time.monotonic()
    child = None
    selector = selectors.DefaultSelector()
    files = {}
    counts = {'stdout': 0, 'stderr': 0}
    discarded = {'stdout': 0, 'stderr': 0}

    def retain(name: str, raw: bytes) -> bool:
        keep = min(len(raw), max_log_bytes - counts[name])
        files[name].write(raw[:keep])
        counts[name] += keep
        discarded[name] += len(raw) - keep
        return keep < len(raw)
    try:
        files = {name: (out / (name + '.log')).open('xb') for name in counts}
        child = subprocess.Popen(
            [sys.executable, '-c', GUARD, str(status_path.absolute()), *argv],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        group = child.pid
        if os.getpgid(group) != group or os.getsid(group) != group:
            raise RuntimeError('New child session ownership could not be established')
        result.update(leader_pid=group, process_group=group, new_session_verified=True)
        for name, stream in [('stdout', child.stdout), ('stderr', child.stderr)]:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        reason = None
        while reason is None:
            for key, _ in selector.select(timeout=0.025):
                raw = os.read(key.fileobj.fileno(), 65536)
                if not raw:
                    selector.unregister(key.fileobj)
                    continue
                if retain(key.data, raw):
                    reason = 'LOG_LIMIT'
            if reason is None and time.monotonic() - started >= seconds:
                reason = 'TIMEOUT'
            if reason is None and status_path.exists():
                # Read only after guard closes its complete small status file.
                try:
                    status = json.loads(status_path.read_text())
                except json.JSONDecodeError:
                    continue
                if type(status.get('returncode')) is not int:
                    raise ValueError('Invalid owned-child status')
                result['child_status'] = status
                reason = 'CHILD_EXITED'
        result['status'] = reason
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if child is not None:
            # The leader has not been reaped; its PID cannot have been reused.
            try:
                os.killpg(child.pid, signal.SIGTERM)
                result['term_sent_to_owned_group'] = True
                until = time.monotonic() + grace_seconds
                while time.monotonic() < until:
                    for key, _ in selector.select(timeout=0.025):
                        raw = os.read(key.fileobj.fileno(), 65536)
                        if not raw:
                            selector.unregister(key.fileobj)
                            continue
                        retain(key.data, raw)
                os.killpg(child.pid, signal.SIGKILL)
                result['kill_sent_to_owned_group'] = True
            except ProcessLookupError:
                result['group_already_absent'] = True
            finally:
                child.wait(timeout=5)
                result['leader_reaped_after_cleanup'] = True
                drain_until = time.monotonic() + 0.25
                while selector.get_map() and time.monotonic() < drain_until:
                    for key, _ in selector.select(timeout=0.025):
                        raw = os.read(key.fileobj.fileno(), 65536)
                        if raw:
                            retain(key.data, raw)
                        else:
                            selector.unregister(key.fileobj)
                result['log_pipes_reached_eof'] = not bool(selector.get_map())
                # killpg can still see zombies; no claim of process-tree containment.
                result['group_cleanup_confirmed'] = bool(
                    result.get('kill_sent_to_owned_group') or result.get('group_already_absent'))
                for stream in (child.stdout, child.stderr):
                    stream.close()
        selector.close()
        for stream in files.values():
            stream.close()
        result['retained_log_bytes'] = counts
        result['discarded_log_bytes'] = discarded
        if any(discarded.values()) and result['status'] == 'CHILD_EXITED':
            result['status'] = 'LOG_LIMIT'
        result['elapsed_seconds'] = time.monotonic() - started
        (out / 'watchdog.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--arm', choices=['baseline', 'liger', 'unsloth'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    from h2_gpu_correctness import sha_file, validate_authority

    # Refuse disabled/unpinned authority before any child/model/GPU process starts.
    validate_authority(contract, args.arm)
    deadline = contract['deadline_unix_s']
    seconds = min(contract['max_seconds'], deadline - time.time() - 2)
    if seconds <= 0:
        raise ValueError('No time left before absolute deadline and cleanup grace')
    runner = Path(__file__).with_name('h2_gpu_correctness.py')
    if contract['runner_sha256'].get(Path(__file__).name) != sha_file(Path(__file__)):
        raise ValueError('Watchdog source differs from authority')
    argv = [sys.executable, str(runner), '--contract', str(args.contract.absolute()),
            '--arm', args.arm, '--out', str((args.out / 'runner').absolute())]
    result = bounded_process(argv, args.out, seconds=seconds)
    pins = {p.relative_to(args.out).as_posix(): sha_file(p)
            for p in args.out.rglob('*') if p.is_file()}
    (args.out / 'files_sha256.json').write_text(json.dumps(pins, indent=2) + '\n')
    Console().print(result['status'] + '; performance=NO_VERDICT; provider stop not performed')
    if result['status'] != 'CHILD_EXITED' or result['child_status']['returncode'] != 0:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
