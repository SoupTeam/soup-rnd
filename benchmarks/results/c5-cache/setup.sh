set -x
export PATH=/venv/main/bin:$PATH
echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd /root/soup-rnd && git log --oneline -1
pip install -q -e ".[train]" "torch==2.14.1" "transformers==5.20.2" "peft==0.21.2" "bitsandbytes==0.50.2" "datasets==5.1.0" "tokenizers==0.23.3"
pip uninstall -y torchvision torchaudio
python -c "import torch,transformers,bitsandbytes;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),transformers.__version__,bitsandbytes.__version__)"
pip freeze > /root/c5/pip-freeze.txt
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,pcie.link.gen.max --format=csv > /root/c5/nvidia-smi.csv
free -b > /root/c5/free.txt
df -B1 /root > /root/c5/df.txt
echo "== setup done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
