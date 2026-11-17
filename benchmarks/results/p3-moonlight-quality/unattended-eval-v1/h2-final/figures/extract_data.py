"""Stream the evidence archives once and write a compact data.json for plotting."""
import json
import sys
import tarfile
from pathlib import Path

RESULT = Path(r"C:\Users\sydyk\Downloads\moonlight-quality-results-h2"
              r"\moonlight-resident-20261111t201548z-whole3-h2-final-done.tar.gz")
TRAIN = Path(r"C:\Users\sydyk\Downloads\moonlight-quality-evidence-20261111-whole3")
data = {"nll": [], "train": {}}
with tarfile.open(RESULT, "r:gz") as tf:
    for member in tf:
        if member.name == "evidence/quality-summary.json":
            data["summary"] = json.loads(tf.extractfile(member).read())
        elif member.name == "evidence/paired/nll-pairs.jsonl":
            for line in tf.extractfile(member).read().decode().splitlines():
                row = json.loads(line)
                data["nll"].append({k: row[k] for k in (
                    "benchmark", "length", "n_targets", "baseline_nll_mean",
                    "candidate_nll_mean", "nll_mean_difference")})
for arm in ("bf16_l2l", "oracle_block64_e4m3"):
    for length in (1024, 4096):
        want = f"train-{arm}-L{length}/steps.jsonl"
        rows = None
        for name in (f"moonlight-resident-20261111t201548z-whole3-train-{arm}-L{length}.tar.gz",
                     "moonlight-resident-20261111t201548z-whole3-final-remaining-evidence.tar.gz"):
            with tarfile.open(TRAIN / name, "r:gz") as tf:
                for member in tf:
                    if member.name.endswith(want):
                        rows = [json.loads(x) for x in tf.extractfile(member).read().decode()
                                .splitlines()]
                        break
            if rows:
                break
        if f"{arm}|{length}" == "bf16_l2l|1024":
            print("step keys:", sorted(k for k, v in rows[0].items()
                                       if not isinstance(v, (dict, list))), flush=True)
        data["train"][f"{arm}|{length}"] = rows
        print(arm, length, len(rows), flush=True)
Path(sys.argv[1]).write_text(json.dumps(data), encoding="utf-8")
