set -eu
cd /workspace/d1probe
gpu_uuid=$(cat clock_uuid)
nvidia-smi -i "$gpu_uuid" -pm 1 > results/persistence-enable.txt 2>&1
nvidia-smi -i "$gpu_uuid" -lgc 1200,1200 > results/clock-reapply.txt 2>&1
cat results/persistence-enable.txt results/clock-reapply.txt
nvidia-smi --query-gpu=uuid,name,driver_version,clocks.current.sm,persistence_mode,power.limit --format=csv,noheader > results/clock-resource-check.txt
cat results/clock-resource-check.txt
