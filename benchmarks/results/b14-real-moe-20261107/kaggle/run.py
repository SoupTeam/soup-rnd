import glob
import os
import runpy

os.environ["B14_STAGE"] = "run"
runpy.run_path(glob.glob("/kaggle/input/**/benchmarks/harness/b14_real_moe_kaggle.py", recursive=True)[0], run_name="__main__")
