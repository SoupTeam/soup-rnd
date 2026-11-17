"""Read-only early check of the pair/analysis requirements on the shards finished so far.

Usage: python -X utf8 precheck.py
Copies two CPU-only scripts to /tmp on the host, runs them at low priority with CUDA hidden,
removes them. Changes nothing in the run; never starts, stops or deletes anything.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
if state.get("closed") or not state.get("ssh_host"):
    sys.exit("The host is closed; nothing to check.")
options = ["-i", r"C:\Users\sydyk\.ssh\e1_vast_ed25519", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=25", "-o", "IdentitiesOnly=yes",
           "-o", f"UserKnownHostsFile={(HERE / 'known_hosts').as_posix()}"]
target = f"root@{state['ssh_host']}"
port = str(state["ssh_port"])
for name in ("h2_predry.py", "h2_precollect.py"):
    subprocess.run([r"C:\Windows\System32\OpenSSH\scp.exe", *options, "-P", port,
                    str(HERE / name), f"{target}:/tmp/{name}"], check=True,
                   capture_output=True, timeout=120)
remote = (
    "R=/data/p3-moonlight-quality/review-source; cd /data/p3-moonlight-quality; "
    "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
    "TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=23 PYTHONDONTWRITEBYTECODE=1 "
    "PYTHONPATH=$R/src:$R; P=/opt/moonlight-py312/bin/python; "
    "nice -n 19 $P -B /tmp/h2_predry.py --stats 2>&1 | tail -n 40; "
    "nice -n 19 $P -B /tmp/h2_precollect.py 2>&1 | tail -n 40; "
    "rm -f /tmp/h2_predry.py /tmp/h2_precollect.py")
done = subprocess.run([r"C:\Windows\System32\OpenSSH\ssh.exe", *options, "-p", port, target,
                       remote], capture_output=True, text=True, encoding="utf-8",
                      errors="replace", timeout=1800)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("\n".join(line for line in done.stdout.splitlines()
                if line.strip() and "vast.ai" not in line and "Have fun" not in line))
