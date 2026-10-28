# Prospective B6 acceptance follow-up

The user approved this eight-fragment paired evaluation before measurement on
2026-10-29 (Asia/Qyzylorda), with a new maximum spend of $5 and an instruction to
spend only what is needed. Final reviewer/owner acceptance is separate.

Data: Salesforce/WikiText, wikitext-2-raw-v1 test split, pinned dataset commit
b08601e04326c79dfdd32d625aee71d232d685c3. Eight eligible row positions are selected
systematically across the corpus before tokenization; the exact rule, row IDs,
original texts and hashes are in evaluation-data.json. No evaluation text is
used for training. Each sequence is independently truncated to at most 64 model
tokens with the pinned DeepSeek-V3 tokenizer; no cross-fragment context is added.

Compare the unadapted base and final two-update attention-LoRA model on identical
input IDs. Retain per-example causal loss/count/digest, token-weighted mean NLL,
perplexity and paired deltas. Any finite result, including degradation, is
reported. No improvement threshold, convergence or broad generalization claim
will be invented after observing the result. The original October 28 result
and its measured 0.1456235249 tokens/s remain a separate experiment.

All source download/conversion, training, both evaluation arms and exact reload
must run under a recorded hard RAM constraint of at most 32 GiB, with swap and
OOM accounting. RSS alone cannot satisfy that rule. A hardware probe that fails
this check stops before downloading the 688 GB checkpoint.

## Dataset attribution

The eight original text rows are from WikiText by Stephen Merity, Caiming Xiong,
James Bradbury and Richard Socher (2016), derived from Wikipedia contributors.
They retain their dataset licenses, not Soup's code license. The pinned
[dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
lists CC BY-SA 3.0 and GFDL in metadata and links CC BY-SA 4.0 in its licensing
section; consult that source for the license notices. The selection preserves
original row text; only selection and subsequent model-token truncation are new.
Source parquet URL, dataset revision and checksum are retained in the JSON.

## Hardware choice, fixed before the new measurement

The initially observed Turing/32 GB offer disappeared before any lease was
created. Pascal is permitted with PyTorch 2.8/CUDA 12.6 and float32 compute,
subject to actual 8 GB/32 GiB guards and a tiny NF4 train/reload capability test.
Other GPUs use native BF16 when available, otherwise FP16 with FP32 adapter
parameters. Both paired arms always use the same actual dtype/cache. This is
not a cross-hardware/dtype comparison with the original BF16 result. The data,
selection, token limit and decision rule remain unchanged.
