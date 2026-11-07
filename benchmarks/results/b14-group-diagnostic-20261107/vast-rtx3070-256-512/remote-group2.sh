#!/bin/bash
# B1.4 grouped-compute diagnostic, buckets 128, 256, 512, on the rented RTX 3070.
cd /root && mkdir -p soup out && tar xzf b14g-src2.tgz -C soup
pip install -q "transformers==5.20.2" safetensors psutil rich pydantic pyyaml typer > /root/out/pip.log 2>&1
pip install -q torch==2.11.0 > /root/out/pip-torch.log 2>&1
pip uninstall -y -q torchvision torchaudio > /root/out/pip-uninstall.log 2>&1
pip list 2>/dev/null | grep -i -E "^(torch|transformers|safetensors|psutil|nvidia-cublas)" > /root/out/pip-versions.txt
export PYTHONPATH=/root/soup/src B14G_OUT=/root/out B14G_WORK=/root B14G_TAG=vast-rtx3070-256-512 B14G_BUCKETS=128,256,512
python3 /root/soup/benchmarks/harness/b14_group_diag.py > /root/out/run.log 2>&1
echo "exit $?" > /root/out/stages.txt
