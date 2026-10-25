# E4 part 2: full calibration on a fresh vast.ai A100 PCIe 40 GB

Scope: the full calibration (F with decide, T, T with `--fallback-gateup`, W, N x5, P, report) into one new dated folder. It stops after `report`. It does not start the main series. The rule text is `e4-part2-calibration-rule.md`; the card-change clause is its line 155; the gate/up targeting is main rule section 4a.

Every command runs from the repo root with the venv active. Set the variables first; nothing secret goes in a file.

```bash
export REPO=$HOME/soup-rnd
export DATA=/workspace/e4-data                       # data-root (source/, writeback-*, codes-*)
export SPLIT=/workspace/e4-split                     # split manifests
export RUN=/workspace/e4-calibration-$(date -u +%Y-%m-%d)   # NEW dated folder, never reused
export IID=<vast instance id>                        # from the vast console or `vastai show instances`
export MAN=$REPO/benchmarks/results/e4-part2-quant/quant-manifest.json
export H=$REPO/benchmarks/harness/e4_part2_calibration.py
```

## 0. Card

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```
Must print `NVIDIA A100-PCIE-40GB`, about 40960 MiB, compute capability 8.0. Otherwise stop (rule section 3 and the 2026-10-20 clarification).

## 1. Clone and install

```bash
git clone https://github.com/SoupTeam/soup-rnd.git $REPO
cd $REPO
git checkout e4-part2-run          # the run branch; its head is the single run commit
git rev-parse HEAD                 # record this hash in the calibration report
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,calibration]" -c .github/constraints/e4-part2-pins.txt
python -c "import sys; sys.path.insert(0,'benchmarks/harness'); import e4_part2_common as c; m=c.library_versions(); assert not m['mismatches'], m"
git status --short                 # must be empty
```
Stop if the pin check fails or `git status` is not empty.

## 2. Data-root: tokenizer, then the transfer pull (this writes verified.json)

`verified.json` is written only by `e4_part2_transfer.py pull`, after every file has matched the manifest. A data root without it stops the run before any attempt (exit 6, `setup_problem`). Do not copy files into `$DATA` by hand.

```bash
mkdir -p $DATA/source
hf download allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --include tokenizer.json tokenizer_config.json special_tokens_map.json --local-dir $DATA/source
```
The pull below downloads and verifies these again; this step only puts the tokenizer where the split step can read it early.

Pull, with a read-only token held only in the environment:
```bash
python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('RedBullet12/e4-part2-quant').sha)"
read -rs -p "HF read token: " HF_TOKEN; echo; export HF_TOKEN
python benchmarks/harness/e4_part2_transfer.py pull --repo RedBullet12/e4-part2-quant --data-root $DATA
unset HF_TOKEN
env | grep -c '^HF_TOKEN=' || true       # must print 0
test -f $DATA/verified.json && echo "verified.json present" || echo "NO verified.json: stop"
shasum -a 256 $MAN                       # must print 1f3b550d707aa18c098851a1718f1712cdb1162adc28552c5fbbcce8611f0ccb
```
- The dataset sha printed first should be `2e09044127c7f8fd97ac474d89c2d98144ec7550`. `pull` takes the head of the dataset repo; the SHA-256 checks make a different head fail. Stop if the printed sha differs and the pull reports a mismatch.
- Stop if `pull` exits non-zero (`VERIFY FAILED` or `manifest model ... is not the pinned one`) or if `verified.json` is absent.
- Revoke the token on huggingface.co (Settings, Access Tokens) after the pull. Unsetting the variable does not revoke it.

## 3. Split (rebuild, then check the section 9.13 hashes)

```bash
python $H split --data-root $DATA --out $SPLIT --dataset-revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a
shasum -a 256 $SPLIT/decision.manifest.jsonl $SPLIT/smoke.manifest.jsonl $SPLIT/train.manifest.jsonl
```
Expected, in this order:
- `90787bbf0557d2194f65a7abe4260c2096f4c0237d534040d4e4d7830723453f` (DECISION, 2048 rows)
- `8e13970ff802550a0367ebb94bc5ae88b27a9ad1b2cb24f0bcb70b4d4a03b55e` (SMOKE, 256 rows)
- `ba33e9dc09a4b37bf85f9d10c05071e5a03ccf412c917bc3ccfd3446d6e7c047` (TRAIN, 4096 rows)

Stop on any difference. `split` exits 1 if the eligible count is below the floor.

## 4. Peers and a fresh output folder

```bash
python $H peers-selftest --gpu-index 0
mkdir -p $RUN && test -z "$(ls -A $RUN)"   # the folder must be new and empty
```

Common flags for every stage below (`C`):
```bash
C="--out $RUN --data-root $DATA --split-dir $SPLIT --manifest $MAN --device cuda --gpu-index 0 --instance-id $IID"
```

## 5. F: three arms, then decide

```bash
python $H drive --stage F --f-prefix F $C
cat $RUN/decision.json
```
`drive --stage F` runs A_off, A_on, A_bf16 (each a fresh process) and then decides. Continue only if `decision.json` says `"decision": "FINAL_A100_40GB"`. Exit code 4 means the card is not final, and exit code 6 means calibration was refused or stopped. Either way, stop.

## 6. T: full selection, then the gate/up fallback targets

```bash
python $H run --stage T $C --run-id T
python $H run --stage T $C --fallback-gateup --run-id T_fallback_gateup
```
Check the second record: `target_misses` must be empty (step 10).

## 7. W, N x5, P

```bash
python $H run --stage W $C
for i in 1 2 3 4 5; do python $H run --stage N --repeat $i $C; done
python $H run --stage P $C
```

## 8. Report and summary

```bash
python $H report --out $RUN
python - "$RUN" <<'PY'
import json, glob, sys
run = sys.argv[1]
rep = json.load(open(f"{run}/report.json"))
print("final_card:", rep["final_card"], rep["final_card_decision"]["decision"])
print("tau keys:", len(rep["tau"]), "tol_W:", rep["tol_W"])
print("noise floors:", {s: (v["floor"], v["epsilon"]) for s, v in rep["noise_floor"].items()})
print("projected total_hours:", rep["projected_main_plan"]["total_hours"])
print("voids:", rep["voids"], "failed_runs:", rep["failed_runs"], "provenance_gaps:", rep["provenance_gaps"])
for f in sorted(glob.glob(f"{run}/runs/*.json")):
    r = json.load(open(f))
    if r.get("run_id") == "T_fallback_gateup" or r.get("result", {}).get("target_misses") is not None:
        res = r.get("result") or {}
        print("T_fallback_gateup target_misses:", res.get("target_misses"))
        print("T_fallback_gateup fallback_sources:", {k: [(m["M"], m["source_units"][:3], m["rows"]) for m in v] for k, v in list(res.get("fallback_sources", {}).items())[:2]})
PY
```

## 9. Stop rules (rule section 8)

- A void keeps its record. Rerun that exact command once, in the same folder, with the same flags. A second void of the same run stops calibration: do not continue, do not start the main series, and report both records.
- Exit 4: the card is not the final card. Stop.
- Exit 6: calibration was refused or stopped (`CalibrationStopped`), including a missing pynvml, a wrong card, a failed T, W, N or P, and a fallback pool that cannot reach M. Stop.
- Exit 6 with `verified.json is missing` or `another manifest`: a setup error, refused before any attempt. No attempt is journaled, so the §8 rerun is not used. Run the pull in section 2, then repeat the same command. Any other exit 6 is a stop.
- Any entry in `target_misses`: the targeted records do not show the fallback kernel. Stop.
- Any changed frozen base hash (W or P): stop.

## 10. Stop here

The runbook ends at the report. Commit nothing from the instance. Copy `$RUN` off the box with its `attempts.jsonl`, `runs/`, `decision.json`, `report.json`, `preflight-*.json` and the driver log. The main series is a separate step, after the report is reviewed and #38 is refilled.
