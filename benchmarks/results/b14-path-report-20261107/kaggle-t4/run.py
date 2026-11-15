import glob
import os
import subprocess
import sys

script = glob.glob("/kaggle/input/**/benchmarks/harness/b14_path_run.py", recursive=True)[0]
for stage in ("oversized", "real"):
    env = dict(os.environ, B14_STAGE=stage, B14_PIP="1", B14_SEQS="16,512,2048",
               B14_TIER="ram", B14_WORK="/tmp/b14")
    subprocess.run(["rm", "-rf", "/tmp/b14"])
    print(f"=== stage {stage}", flush=True)
    subprocess.run([sys.executable, script], env=env)
