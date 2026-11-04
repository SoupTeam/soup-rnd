#!/usr/bin/env bash
# Run only inside an already-approved Linux instance. This script rents nothing.
set -euo pipefail
H3_REPO="$(pwd)"
H3_EVIDENCE="$H3_REPO/benchmarks/evidence/h3-fsdp-20261026"
mkdir -p "$H3_EVIDENCE/live"
exec > >(tee "$H3_EVIDENCE/live/bootstrap.log") 2>&1
test -z "$(git status --porcelain)" || { echo 'Commit the checkout first.' >&2; exit 1; }
git rev-parse HEAD > "$H3_EVIDENCE/live/commit.txt"
df -h > "$H3_EVIDENCE/live/disk-before.txt"
nvidia-smi -q > "$H3_EVIDENCE/live/nvidia-smi-before.txt"
nvidia-smi topo -m > "$H3_EVIDENCE/live/topology-before.txt"
python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13)'
test "$(df -Pk "$H3_EVIDENCE" | awk 'NR==2 {print $4}')" -gt 15728640
python3 -m venv "$H3_EVIDENCE/venv"
source "$H3_EVIDENCE/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[train]' psutil -c benchmarks/h3-fsdp-constraints.txt
python -m pip check
python -m pip freeze > "$H3_EVIDENCE/live/bootstrap-pip-freeze.txt"
export PYTHONPATH="$H3_REPO/src"
export TOKENIZERS_PARALLELISM=false
python benchmarks/harness/h3_fsdp_run.py prepare --prepared "$H3_EVIDENCE/prepared" \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
test "$(df -Pk "$H3_EVIDENCE" | awk 'NR==2 {print $4}')" -gt 15728640
python benchmarks/harness/h3_fsdp_campaign.py \
  --prepared "$H3_EVIDENCE/prepared" --output "$H3_EVIDENCE/live/campaign" \
  --repeats 3 --timeout-seconds 900 --campaign-seconds 5400
df -h > "$H3_EVIDENCE/live/disk-after.txt"
# Copy the evidence off the instance, then destroy it externally; a stopped
# instance can still incur disk charges. This script never destroys user resources.
