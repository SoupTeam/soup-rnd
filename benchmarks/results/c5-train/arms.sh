export PATH=/venv/main/bin:$PATH
cd /root/soup-rnd
echo "== arms start $(date -u +%Y-%m-%dT%H:%M:%SZ) commit $(git rev-parse HEAD)"
benchmarks/harness/c5_train_all.sh /root/c5/data /root/c5/runs \
  --model allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --device cuda --gradient-checkpointing
echo "== arms end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
