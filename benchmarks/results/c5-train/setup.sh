set -x
export PATH=/venv/main/bin:$PATH
echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd /root/soup-rnd && git log --oneline -1
pip install -q -e ".[train]" "torch==2.14.1" "transformers==5.20.2" "peft==0.21.2" "bitsandbytes==0.50.2" "datasets==5.1.0" "tokenizers==0.23.3"
pip uninstall -y torchvision torchaudio
python -c "import torch,transformers,peft,bitsandbytes,datasets;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),transformers.__version__,peft.__version__,bitsandbytes.__version__,datasets.__version__)"
python benchmarks/harness/c5_data.py --out-dir /root/c5/data
echo "== setup done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
