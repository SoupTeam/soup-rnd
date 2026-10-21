# Reproduction commands (paths used on host E)
# Python 3.12.12, driver 595.99.02; see c2-environment-e2.json for full freeze.
cd /workspace/c2
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
/venv/main/bin/python benchmarks/harness/synth_checkpoint.py --shape llama-70b --vocab 32768 --seed 1234 --out /workspace/c2-weights
/venv/main/bin/python benchmarks/harness/c2_linux_campaign.py --prepare-model --weights /workspace/c2-weights --shards /workspace/c2/c2-cache --out /workspace/c2-cache-build.json
/venv/main/bin/python benchmarks/harness/c2_linux_depth_probe.py --fixture /workspace/c2-fixture.bin --prepare --out /workspace/c2-fixture-prepare.json
# Strict admission was not met: preserve c2-sensor-manual-e1.json and c2-prequiet-e1.json.
# Only AFTER the separate diagnostic protocol was committed, the manual sensor
# script in sensor-diagnostic-command.txt admitted the mapping/correctness record.
# e10b7d9aeab0a9ffd86ee4df36a817e0464c0de2:
/venv/main/bin/python benchmarks/harness/c2_f1_campaign.py --kind primitive --fixture /workspace/c2-fixture.bin --admission /workspace/c2-admission-e2.json --device nvme0n1 --diagnostic-writes --out /workspace/c2-diag-primitive-e1
/venv/main/bin/python benchmarks/harness/c2_f1_campaign.py --kind step --fixture /workspace/c2-fixture.bin --weights /workspace/c2-weights --shards /workspace/c2/c2-cache --admission /workspace/c2-admission-e2.json --device nvme0n1 --diagnostic-writes --out /workspace/c2-diag-step-e1
# step-e1 failed before timing and remains excluded; prospective correction committed.
# 193e433c:
/venv/main/bin/python benchmarks/harness/c2_f1_campaign.py --kind step --fixture /workspace/c2-fixture.bin --weights /workspace/c2-weights --shards /workspace/c2/c2-cache --admission /workspace/c2-admission-e2.json --device nvme0n1 --diagnostic-writes --out /workspace/c2-diag-step-e2
# Every out path must be new. Do not rerun over retained observations.
# step-e2 was wholly void after 12 children: 1,725,698,560 unattributed read bytes.
# a3e3fcdd preregisters one final full retry and the explicit operational cap amendment.
/venv/main/bin/python /dev/shm/prequiet_e3.py
/venv/main/bin/python benchmarks/harness/c2_f1_campaign.py --kind step --fixture /workspace/c2-fixture.bin --weights /workspace/c2-weights --shards /workspace/c2/c2-cache --admission /workspace/c2-admission-e2.json --device nvme0n1 --diagnostic-writes --out /workspace/c2-diag-step-e3
# prequiet_e3.py is preserved verbatim as prequiet-e3-command.txt.
