import glob
import os
import runpy

os.environ["B14_STAGE"] = "tiny"
os.environ["B14_PIP"] = "1"
os.environ["B14_SEQS"] = "16,256"
script = glob.glob("/kaggle/input/**/benchmarks/harness/b14_path_run.py", recursive=True)[0]
runpy.run_path(script, run_name="__main__")
