import glob
import os
import runpy

os.environ["B14_STAGE"] = "run"
script = glob.glob("/kaggle/input/**/benchmarks/harness/b14_real_moe_kaggle.py", recursive=True)[0]
runpy.run_path(script, run_name="__main__")
