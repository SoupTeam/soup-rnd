"""Detached orchestration; no retries and no source/measurement parameter changes."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/root/c4-pattern-v1')
SESSION = ROOT / 'pattern-session'
with (ROOT / 'timed-launch.lock').open('x') as lock:
    lock.write('One launch only; retain this marker after interruption.\n')
shutil.copyfile(__file__, SESSION / 'launch_wrapper.py')
codes = {}
with (ROOT / 'timed-launch.log').open('x') as log:
    codes['series'] = subprocess.run([
        'python3', 'runner.py', 'run-series', '--session', str(SESSION),
        '--receipt', str(ROOT / 'committed-receipt.json')],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    codes['analysis'] = subprocess.run([
        'python3', 'analyze.py', '--session', str(SESSION)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    (SESSION / 'launch-result.json').write_text(json.dumps(codes, indent=2)+'\n')
    subprocess.run(['python3', 'runner.py', 'bundle', '--session', str(SESSION)],
                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
