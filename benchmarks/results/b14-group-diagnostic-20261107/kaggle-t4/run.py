import glob
import os
import subprocess
import sys

root = glob.glob("/kaggle/input/**/src/soup_cli", recursive=True)[0]
base = os.path.dirname(os.path.dirname(root))
script = os.path.join(base, "benchmarks/harness/b14_group_diag.py")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.2",
                "safetensors", "psutil", "rich", "pydantic", "pyyaml", "typer"])
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
env = dict(os.environ, PYTHONPATH=os.path.join(base, "src"), B14G_OUT="/kaggle/working",
           B14G_WORK="/tmp", B14G_TAG="kaggle-t4")
subprocess.run([sys.executable, script], env=env)
