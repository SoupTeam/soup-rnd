#!/bin/bash
set -euo pipefail
cd /data/h2/repo
python3 -m benchmarks.harness.h2_snapshot_staging --manifest benchmarks/h2/snapshot_preparation_result.json --destination /data/h2/model_snapshot --evidence /data/h2/evidence/staging --deadline-seconds 1700
python3 - <<'AUDIT'
import json,importlib.util,pathlib
p=pathlib.Path('/data/h2');receipt=json.loads((p/'evidence/staging/receipt.json').read_text());assert receipt['status']=='PASS_REMOTE_SNAPSHOT_SHA'
m=json.loads((p/'repo/benchmarks/h2/snapshot_preparation_result.json').read_text())
spec=importlib.util.spec_from_file_location('audit',p/'snapshot_auditor.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
progress={'destination':str(p/'model_snapshot'),'repo':m['repo'],'revision':m['revision'],'files':{n:{'status':'VERIFIED','expected_bytes':m['files_bytes'][n],'sha256':m['files_sha256'][n]} for n in m['files_bytes']}}
r=mod.audit(progress);(p/'evidence/snapshot_structural_audit.json').write_text(json.dumps(r,indent=2)+'\n');assert r['status']=='PASS_SNAPSHOT_FILE_AND_STRUCTURE_AUDIT'
AUDIT
