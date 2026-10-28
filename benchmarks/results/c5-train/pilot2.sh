export PATH=/venv/main/bin:$PATH
cd /root/soup-rnd
echo "== pilot 2 (gradient checkpointing) start $(date -u +%Y-%m-%dT%H:%M:%SZ) (not an arm; never enters the verdict)"
python benchmarks/harness/c5_train.py --arm nf4 --seed 42 --model allenai/OLMoE-1B-7B-0924 \
  --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --device cuda \
  --data-dir /root/c5/data --out-dir /root/c5/pilot/nf4-s42-gc --gradient-checkpointing 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date -u +%H:%M:%S)" "$line"; done
echo "== pilot end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
