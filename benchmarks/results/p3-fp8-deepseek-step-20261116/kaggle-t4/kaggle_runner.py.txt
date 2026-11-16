import os, subprocess, sys
C = "e7fc9db04c9bd5a43d73bdd6a88310ea7486a116"
R, V = "/tmp/soup", "/tmp/py312"
DEPS = ("torch==2.11.0 transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 "
        "bitsandbytes==0.50.2 safetensors==0.8.0 rich==13.9.4 pydantic==2.14.1 "
        "numpy==2.1.3 psutil==7.2.2 pyyaml==6.0.3")
LOG = open("/kaggle/working/run.log", "w")
def sh(cmd, **kw):
    LOG.write(f"+ {cmd}\n"); LOG.flush()
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    LOG.write(p.stdout); LOG.flush(); print(p.stdout[-4000:], flush=True)
    if p.returncode: raise SystemExit(f"exit {p.returncode}: {cmd}")
sh(f"git clone -q https://github.com/SoupTeam/soup-rnd.git {R} && git -C {R} checkout -q {C} && git -C {R} rev-parse HEAD")
sh(f"{sys.executable} -m pip install -q uv==0.13.7")
sh(f"{sys.executable} -m uv venv --python 3.12.3 --seed {V}")
sh(f"{sys.executable} -m uv pip install --python {V}/bin/python {DEPS} -e {R}")
sh("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader")
sh(f"{V}/bin/python -c 'import sys,torch,transformers,peft;print(sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__,peft.__version__)'")
env = {**os.environ, "PYTHONPATH": f"{R}/src:{R}", "CUDA_VISIBLE_DEVICES": "0",
       "HF_HUB_OFFLINE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
sh(f"{V}/bin/python benchmarks/harness/p3_fp8_deepseek_step.py --device cuda --output /kaggle/working/out --ks 1 4", cwd=R, env=env)
