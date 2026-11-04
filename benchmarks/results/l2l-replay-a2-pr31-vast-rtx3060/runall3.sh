cd /root/soup && export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1
P=.venv/bin/python; D=benchmarks/harness/l2l_smollm2_check.py; O=/root/run3
mkdir -p $O
st() { echo "$(date -Is) $*" >> $O/status; }
st start
$P benchmarks/harness/l2l_fit_check.py --weights /root/smollm2 > $O/fit-estimate.json 2> $O/fit-estimate.err; st "fit-estimate exit $?"
$P $D --weights /root/smollm2 --quant none --dropout 0.05 --rng-seed 1234 --ks 4 --out $O/probe-k4 > $O/probe-k4.log 2>&1; st "probe-k4 exit $?"
$P benchmarks/harness/l2l_fit_check.py --weights /root/smollm2 --records $O/probe-k4/records.json > $O/fit-measured.json 2> $O/fit-measured.err; rc=$?; st "fit-measured exit $rc"
[ $rc -eq 0 ] || { st "ABORT: measured peak does not fit"; exit 1; }
$P $D --weights /root/smollm2 --quant none --dropout 0    --out $O/d0-01 > $O/d0-01.log 2>&1; st "d0-01 exit $?"
$P $D --weights /root/smollm2 --quant none --dropout 0.05 --rng-seed 1234 --out $O/d005-01 > $O/d005-01.log 2>&1; st "d005-01 exit $?"
$P $D --weights /root/smollm2 --quant none --dropout 0.05 --rng-seed 1234 --out $O/d005-02 > $O/d005-02.log 2>&1; st "d005-02 exit $?"
$P $D --weights /root/smollm2 --quant none --dropout 0.05 --rng-seed 1234 --deterministic --out $O/d005-det > $O/d005-det.log 2>&1; st "d005-det exit $?"
st ALLDONE
