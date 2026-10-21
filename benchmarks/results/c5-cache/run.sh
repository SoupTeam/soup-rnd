set -eux
export PATH=/venv/main/bin:$PATH
cd /root/soup-rnd
echo "== build start $(date -u +%Y-%m-%dT%H:%M:%SZ) commit $(git rev-parse HEAD)"
python benchmarks/harness/c5_cache.py build --model allenai/OLMoE-1B-7B-0924 \
  --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --device cuda \
  --cache-root /root/c5/caches --out /root/c5/cache.json
echo "== compare start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python benchmarks/harness/c5_cache.py compare --cache-root /root/c5/caches \
  --label "RTX 3090, real" --device cuda --out /root/c5/restore.json
echo "== end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
