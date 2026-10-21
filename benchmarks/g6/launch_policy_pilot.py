"""External process-group timeout for one admitted pilot stage; never retries."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from policy_pilot_support import bound, load_contract
from run import write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cuda-gate', 'pilot'))
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args, rest = parser.parse_known_args()
    if any(value in {'--out', '--spec', '--spec-sha', '--artifact-root', '--contract',
                     '--contract-sha'} for value in rest):
        parser.error('Duplicate supervisor-owned arguments are forbidden')
    spec = load_contract(args.spec, args.spec_sha, args.artifact_root)
    seconds = bound(spec, args.mode)
    if args.out.exists():
        parser.error('Existing output forbids retry/resume')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'cuda-gate':
        code = Path(__file__).with_name('policy_cuda_gate.py')
        command = [sys.executable, str(code), '--contract', str(args.spec),
                   '--contract-sha', args.spec_sha]
    else:
        code = Path(__file__).with_name('policy_pilot.py')
        command = [sys.executable, str(code), 'pilot', '--spec', str(args.spec),
                   '--spec-sha', args.spec_sha]
    command += ['--artifact-root', str(args.artifact_root), '--out', str(args.out), *rest]
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       G6_POLICY_EXTERNAL_BOUND_S=str(seconds))
    timed_out = False
    with Path(str(args.out) + '.launcher.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment, start_new_session=True)
        try:
            status = child.wait(timeout=seconds + 30)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            status = child.returncode
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise
    write(Path(str(args.out) + '.launcher.json'), {
        'exitcode': status, 'hard_timeout': timed_out, 'bound_s': seconds,
        'supervisor_grace_s': 40, 'contract_sha256': args.spec_sha, 'no_retry': True,
    })
    if status or timed_out:
        raise SystemExit(status or 1)


if __name__ == '__main__':
    main()
