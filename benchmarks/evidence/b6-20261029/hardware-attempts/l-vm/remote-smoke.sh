#!/bin/bash
set -euo pipefail
export PYTHONPATH=/root/b6/repo:/root/b6/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
mkdir -p /root/b6/results /root/b6/repo
trap 'code=$?; if test "$code" != 0; then echo "$code" > /root/b6/results/failure-code.txt; until test -f /root/b6/abort; do sleep 5; done; fi; echo "$code" > /root/b6/results/exit-code.txt' EXIT
cd /root/b6
tar -xzf source.tar.gz -C repo
cp hardware-probe.json results/hardware-probe.json
cp remote-smoke.sh results/remote-smoke.sh
nvidia-smi -q > results/nvidia-smi-q.txt
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv > results/gpu.csv
lscpu --json > results/lscpu.json
cat /proc/meminfo > results/meminfo-start.txt
cat /proc/self/cgroup > results/cgroup-identity.txt
lsblk -o NAME,TYPE,SIZE,ROTA,MODEL > results/disks.txt
df -B1 /root/b6 > results/filesystem-start.txt
curl -LsSf https://astral.sh/uv/install.sh -o /root/b6/install-uv.sh
sh /root/b6/install-uv.sh
/root/.local/bin/uv --version > /root/b6/results/uv-version.txt
/root/.local/bin/uv venv --python 3.12 /root/b6/venv
compute_major=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | cut -d. -f1)
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)
if test "$compute_major" -lt 7 && test "$driver_major" -ge 560; then cuda_index=cu126; torch_version=2.8.0; elif test "$driver_major" -ge 570; then cuda_index=cu128; torch_version=2.8.0; elif test "$driver_major" -ge 560; then cuda_index=cu126; torch_version=2.8.0; elif test "$driver_major" -ge 550; then cuda_index=cu124; torch_version=2.6.0; elif test "$driver_major" -ge 470; then cuda_index=cu118; torch_version=2.6.0; else exit 20; fi
/root/.local/bin/uv pip install --python /root/b6/venv/bin/python \
 --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/$cuda_index \
 "torch==$torch_version" 'transformers==5.16.1' 'peft==0.20.0' 'trl==0.29.1' \
 'accelerate==1.14.0' 'bitsandbytes==0.50.2' 'safetensors==0.8.0' \
 'pytest==9.1.1' 'pydantic>=2' 'rich' 'typer' 'pyyaml' 'packaging' 'psutil'
/root/.local/bin/uv pip freeze --python /root/b6/venv/bin/python > results/pip-freeze.txt
/root/b6/venv/bin/python - <<'SMOKE' | tee /root/b6/results/runtime-smoke.log
import torch,json,tempfile,gc,hashlib
from pathlib import Path
from rich.console import Console
from peft import LoraConfig,TaskType,load_peft_weights,set_peft_model_state_dict
from tests.test_b2_deepseek_v3_stream import _save_fixture,_tiny_deepseek_v3_config
from soup_cli.utils.layer_stream_runtime import build_meta_skeleton,quantised_layer_suffixes,shard_layout_of,build_streamed_model,assert_streamed_adapter_saved
from soup_cli.utils.layer_shard import shard_checkpoint
from soup_cli.utils.peft_wiring import MOE_TEXT_LORA_TARGETS
assert torch.cuda.is_available() and torch.cuda.device_count()==1
props=torch.cuda.get_device_properties(0)
assert 6*2**30<=props.total_memory<=9*2**30
dtype='bfloat16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'float32'
torch.set_num_threads(1);torch.manual_seed(20261029);torch.use_deterministic_algorithms(True)
root=Path('/root/b6/tiny-smoke');source=root/'source';cache=root/'cache';root.mkdir()
config=_tiny_deepseek_v3_config(n_routed_experts=4,n_group=2,topk_group=1);_save_fixture(source,config)
skel=build_meta_skeleton(str(source),dtype=dtype,quant='nf4',double_quant=False)
suffixes=quantised_layer_suffixes(skel);layout=shard_layout_of(skel);del skel
index=shard_checkpoint(str(source),str(cache),dtype=dtype,arch='deepseek_v3',layout=layout,quant='nf4',quant_suffixes=suffixes,quant_device='cuda',double_quant=False,expert_layout='per_expert')
lora=LoraConfig(r=2,lora_alpha=4,lora_dropout=0,bias='none',target_modules=list(MOE_TEXT_LORA_TARGETS['deepseek_v3']),task_type=TaskType.CAUSAL_LM)
def build():
    model,runtime=build_streamed_model(model_id=str(source),shard_dir=str(cache),index=index,lora_config=lora,device='cuda',dtype=dtype,buffers=2,pin=False,quant='nf4',double_quant=False,tier='disk',read_ahead=1)
    if dtype=='float16':
        for param in model.parameters():
            if param.requires_grad:param.data=param.data.float()
    return model,runtime
model,runtime=build();params=[p for p in model.parameters() if p.requires_grad]
ids=torch.randint(0,config.vocab_size,(1,16),device='cuda');batch={'input_ids':ids,'labels':ids}
result=model(**batch);assert torch.isfinite(result.loss);result.loss.backward()
grads=[p.grad for p in params if p.grad is not None];assert len(grads)==len(params) and all(torch.isfinite(g).all() for g in grads)
norm=float(torch.nn.utils.clip_grad_norm_(params,1.0));assert norm>0
before=[p.detach().clone() for p in params];torch.optim.AdamW(params,lr=1e-4).step();assert any(not torch.equal(a,b) for a,b in zip(before,params))
model.eval()
with torch.no_grad():logits=model(**batch).logits.cpu()
adapter=root/'adapter';model.save_pretrained(adapter);assert_streamed_adapter_saved(model,str(adapter));runtime.close();del model,runtime,result,params,grads,before;gc.collect();torch.cuda.empty_cache()
model,runtime=build();set_peft_model_state_dict(model,load_peft_weights(str(adapter),device='cpu'));model.eval()
with torch.no_grad():reloaded=model(**batch).logits.cpu()
assert torch.equal(logits,reloaded)
runtime.close()
r={'scope':'synthetic NF4 streamed-runtime capability check before real checkpoint download','gpu':props.name,'vram_bytes':props.total_memory,'dtype':dtype,'gradient_norm':norm,'reload_equal':True,'torch':torch.__version__,'cuda':torch.version.cuda}
Path('/root/b6/results/runtime-smoke.json').write_text(json.dumps(r,indent=2));Console().print(r)
SMOKE
cat > /root/b6/experiment.sh <<'EXPERIMENT'
#!/bin/bash
set -euo pipefail
cd /root/b6/repo
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
echo download > /root/b6/phase.txt
touch /root/b6/download-started
/root/b6/venv/bin/python - <<'DOWNLOAD' | tee /root/b6/results/download.log
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil,json,time
required=688586727753+519579082752+4*2**30
free=shutil.disk_usage('/root/b6').free
assert free>=required,(free,required)
start=time.time()
snapshot_download('deepseek-ai/DeepSeek-V3',revision='e815299b0bcbac849fa540c768ef21845365c9eb',local_dir='/root/b6/source',allow_patterns=['*.safetensors','*.json','tokenizer.model','*.tiktoken','*.txt'],max_workers=8,token=False)
files=list(Path('/root/b6/source').glob('*.safetensors'))
assert len(files)==163 and sum(p.stat().st_size for p in files)==688586727753
print(json.dumps({'seconds':time.time()-start,'source_bytes':sum(p.stat().st_size for p in files),'files':len(files)}))
DOWNLOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/b6/venv/bin/python benchmarks/harness/b6_real_moe.py \
 --source /root/b6/source --cache /root/b6/cache --output /root/b6/results/real \
 --data examples/b6-pilot-data.json --revision e815299b0bcbac849fa540c768ef21845365c9eb \
 --steps 2 --max-length 64 --shard-threads 8 \
 --evaluation-protocol benchmarks/protocols/b6-acceptance-20261029/protocol.json \
 --evaluation-data benchmarks/protocols/b6-acceptance-20261029/evaluation-data.json \
 --ram-phase-file /root/b6/phase.txt | tee /root/b6/results/real.log
cp /root/b6/cache/index.json /root/b6/results/cache-index.json
EXPERIMENT
chmod +x /root/b6/experiment.sh
cp /root/b6/experiment.sh /root/b6/results/experiment.sh
cd /root/b6/repo
/root/b6/venv/bin/python benchmarks/harness/b6_ram.py --output /root/b6/results \
 --phase-file /root/b6/phase.txt --interval 2 -- bash /root/b6/experiment.sh
