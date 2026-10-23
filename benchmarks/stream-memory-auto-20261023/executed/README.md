# Executed benchmark helpers: cohort B

Exact helper sources and metadata are stored at their original paths inside the parent
`evidence.zip`. Extract it into the parent report directory for manual training replay:
`python -m zipfile -e evidence.zip .` from that directory. The retained-results replay
utility can read archived helper bytes directly from the ZIP without extracting them.

The six `*.py.txt` files are the exact bytes of the Python helpers used by
`soup-memory-impact-20261023-b`. Only the archival filename has a `.txt` suffix;
the Python contents were not reformatted or reconstructed. Copy them to their
original `.py` names for replay. `setup.sh`, constraints, protocol and manifests
are likewise exact copies.

`published-helpers-manifest.json` records their sizes and SHA-256 hashes. All ten
published files that have entries in the frozen B launch manifest match those
entries. The launch manifest itself and prospective amendment are also hashed;
they do not have self-entries in the launch manifest.

The production source archive had SHA-256
`adfc15617878f3592f3bd49c37fd69ef7b94f8e4e2d5b177e409a2e42f7e1bb7`.
The full source inventory is in `source-manifest.json`. The separate
`production-newline-normalization.json` maps executed bytes to normalized-LF
bytes for the four production feature files. Its only transformation is
`CRLF -> LF`; it does not describe a different GPU execution.

## Required source and dependencies

Use Linux, one physical 8 GiB CUDA GPU, and the feature checkout identified by
the enclosing report. The original benchmark harness is from
[PR #5 base f8833454](https://github.com/SoupTeam/soup-rnd/tree/f8833454c9a7e60af5c7d4b038600616f8979b25/benchmarks/harness):
`liger_streaming_bench.py`, `liger_streaming_parity.py` and
`streaming_optimization_bench.py` are reused unchanged. That historical base
alone does not contain the new production residency feature. Check the source
manifest, including the newline map, against the feature checkout being used.

The executed environment used Python 3.12, Torch 2.8.0 with CUDA 12.8 wheels,
Transformers 5.17.0, PEFT 0.20.0, TRL 0.29.1, Accelerate 1.14.0 and
bitsandbytes 0.50.2. `constraints.txt` records the full dependency pins.

The following illustrates manual replay on an already provisioned Linux machine
with the feature checkout installed at `/root/liger/repo`. Set `ARCHIVE` to this
directory in that checkout:

```bash
ARCHIVE=/root/liger/repo/benchmarks/stream-memory-auto-20261023/executed
mkdir -p /root/liger/results
for path in "$ARCHIVE"/*.py.txt; do
  cp "$path" "/root/liger/$(basename "${path%.txt}")"
done
cp "$ARCHIVE/constraints.txt" "$ARCHIVE/protocol.json" /root/liger/

python3 -m pip install uv
uv venv --python 3.12 /root/liger/venv
uv pip install --python /root/liger/venv/bin/python \
  --constraint /root/liger/constraints.txt --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.8.0' 'transformers==5.17.0' 'peft==0.20.0' 'trl==0.29.1' \
  'accelerate==1.14.0' 'bitsandbytes==0.50.2' \
  -e '/root/liger/repo[train]' pytest pytest-cov cryptography pip sentencepiece protobuf einops
uv pip check --python /root/liger/venv/bin/python
source /root/liger/venv/bin/activate
export PYTHONPATH=/root/liger:/root/liger/repo:/root/liger/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_DISABLED=true
export SOUP_LAYER_STREAM_CACHE_DIR=/root/liger/shards
cd /root/liger
python bench.py --help
python prepare_model.py 8
```

`prepare_model.py` downloads the protocol's pinned model revision and creates
identical local data/configs for baseline and auto. It requires room for the
original weights and their exact streamed shards. The benchmark enforces a
7 GiB per-process CUDA allocator budget on the physical 8 GiB GPU.

## Accuracy before timing

```bash
python pin_entry.py bench.py --config model-8-baseline.yaml --arm baseline \
  --output results/replay-8-capture-baseline --capture
python pin_entry.py bench.py --config model-8-auto.yaml --arm auto \
  --output results/replay-8-capture-auto --capture
python pin_entry.py bench.py --config model-8-baseline.yaml --arm baseline \
  --output results/replay-8-semantics-baseline --semantics
python pin_entry.py bench.py --config model-8-auto.yaml --arm auto \
  --output results/replay-8-semantics-auto --semantics
python - <<'PY'
from pathlib import Path
from benchmarks.harness.liger_streaming_parity import compare
for scope in ('capture', 'semantics'):
    reference = Path('results') / f'replay-8-{scope}-baseline'
    candidate = Path('results') / f'replay-8-{scope}-auto'
    assert compare(reference, candidate)['passed'], scope
PY
```

The unchanged comparator checks reference-first `rtol=0.01`, `atol=1e-5`, finite
values, exact inputs/initial adapters, and losses rounded to three decimals.
Capture retains two optimizer updates and their pre-update gradients. Semantic
checks cover causal-prefix and masked-padding isolation.

For speed, start fresh processes in protocol order `AB, BA, BA, AB`, with
`A=baseline` and `B=auto`. Each invocation uses a new output directory:

```bash
python pin_entry.py bench.py --config model-8-baseline.yaml --arm baseline \
  --output results/replay-8-pair0-baseline --warmup 6 --steps 8
python pin_entry.py bench.py --config model-8-auto.yaml --arm auto \
  --output results/replay-8-pair0-auto --warmup 6 --steps 8
```

Repeat the other three pairs in their specified order. The archived `worker.py`
shows the exact final-adapter, fourteen-loss, initial-state/data hash, precision,
and 28-microbatch shape checks required before accepting each timing pair.
`timing_summary.py` summarizes whole-pair bootstrap confidence and the unchanged
2% practical speed gate. Auto's actual selected layers are recorded in every
`residency-auto-provenance.json`; do not substitute a requested count.

Reported timed VRAM comes from the eight measured updates after peak resets.
Warmup peaks are cumulative until those resets. Setup peaks are the allocator
statistics observed after setup, which itself contains resetting probes; they
are not a guarantee of the maximum across every setup allocation. Token/s uses
2048 processed input positions per optimizer update, not a separate count of
supervised labels.

## Scope of the archive

This is benchmark-source provenance and a manual benchmark replay recipe, not a
complete replay of the paid cloud orchestration. Cloud authentication, API keys,
connection files, private launch/guard controllers, artifact stores, and their
per-lease policies are excluded. `worker.py` and `setup.sh` are archived for audit
but refer to additional orchestration/retention helpers that are not all included
here; they are not standalone launch scripts. Manual replay requires separate
resource management and complete retention/comparison of the raw outputs.

This packaging alone establishes no result for 14B/27B, NF4 GPU performance,
NVMe throughput, or an optimal speed-based memory selection policy.
