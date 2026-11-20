#!/bin/bash
# A-B-C-C-B-A blocks (A0 / B0 / exact MBR).  usage: abc-run.sh   (BLOCKS env, default 5)
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
rm -rf $R/results/abc $R/results/abc-DONE
python repo/benchmarks/harness/p72_abc_blocks.py --root $R --output $R/results/abc \
  --host-cache-mib 12288 --timed-steps 3 --blocks ${BLOCKS:-5} --max-blocks 8 > $R/results/abc.log 2>&1
echo "abc exit $?" >> $R/results/abc.log
echo done > $R/results/abc-DONE
