#!/bin/bash
# Paired F1 (A-B-B-A blocks) for each mode in turn.  usage: f1-run.sh "mode mode ..."   (BLOCKS env, default 5)
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
for ARMS in $1; do
  echo "f1_$ARMS" > $R/phase.txt
  rm -rf $R/results/f1d-$ARMS
  python repo/benchmarks/harness/p72_f1_blocks.py --root $R --output $R/results/f1d-$ARMS \
    --arms $ARMS --host-cache-mib 12288 --timed-steps 3 --max-blocks ${BLOCKS:-5} \
    --label "RTX 3060 (vast.ai), Moonlight, $ARMS" > $R/results/f1d-$ARMS.log 2>&1
  echo "f1 $ARMS exit $?" >> $R/results/f1d.log
done
echo done > $R/results/f1d-DONE
