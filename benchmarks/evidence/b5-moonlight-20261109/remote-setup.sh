#!/bin/bash
# B5 ticket 03: host setup on a vast.ai container (nvidia/cuda:13.0.0-devel-ubuntu24.04).
# Same pins as the local verification (transformers 5.20.2, peft 0.21.2, torch 2.14.1 cu130).
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip psmisc util-linux > /dev/null
mkdir -p /root/b5 && cd /root/b5
python3 -m venv venv
venv/bin/pip install --no-cache-dir -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
venv/bin/pip install --no-cache-dir -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 \
  bitsandbytes==0.50.2 safetensors==0.8.0 pydantic==2.14.1 psutil==7.2.2 rich==15.0.0 \
  pyyaml==6.0.3 typer==0.20.1 numpy==2.4.6 huggingface-hub==2.2.0 tokenizers==0.23.3 \
  hf-xet==1.7.0 tiktoken==0.14.0 plotext packaging
mkdir -p repo && tar -xzf /root/b5/repo.tgz -C repo
venv/bin/pip freeze > /root/b5/pip-freeze.txt
venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
