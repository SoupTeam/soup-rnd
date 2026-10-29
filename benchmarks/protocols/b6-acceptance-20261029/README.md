# Prospective B6 acceptance follow-up

The user approved this eight-fragment paired evaluation before measurement on
2026-10-29 (Asia/Qyzylorda), with a $5 spending ceiling and an instruction to
spend only what is needed. Conservatively, the ceiling includes the original
$3.707 pilot as well as follow-up probes. Final reviewer/owner acceptance is separate.

After the initial follow-up ended at $3.863 cumulative, the user authorized up
to another $5 on 2026-10-29 to finish the same protocol. This new spending window
includes every new attempt, compute, storage and traffic, with a cumulative cap
of $8.863. Evaluation selection and acceptance criteria are unchanged.

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
The final compute rule below supersedes the initial FP16 fallback proposal.
Both paired arms always use the same actual dtype/cache. This is
not a cross-hardware/dtype comparison with the original BF16 result. The data,
selection, token limit and decision rule remain unchanged.

## RAM-accounting compatibility

Hardware probes encountered both unified cgroup v2 and legacy v1 hosts before
any new giant evaluation. The monitor resolves the current process's controller
membership against mountinfo, including a mount rooted at the container path.
V1 uses memory.limit_in_bytes, max_usage_in_bytes, memory.stat swap counts and
memory.oom_control; v2 uses memory.max, memory.peak, memory.swap.current/peak and
memory.events. A host-root file is not substituted for the container controller.
Docker memory/swap limits were requested at creation, but the actual probes did
not observe a 32 GiB limit. Source download was refused; the criterion remains
a recorded hard limit of at most 32 GiB.

## Final compute rule before any new giant losses

After hardware-only probes (no full new checkpoint download and no new giant
loss measured), available VM hardware includes Turing. The final rule is native
BF16 when supported, otherwise FP32, avoiding FP16 range/scaling ambiguity.
Both paired arms use the same actual dtype/cache. An owned VM may use a systemd
scope with MemoryMax=32G and MemorySwapMax=0; the monitor must observe the actual
scope and accounting throughout every model phase. Parent VM RAM is disclosed.
The eight texts, token limit and result-reporting decision rule are unchanged.

## Execution status

No new giant evaluation loss has been measured. See
[the follow-up report](../../b6-acceptance-20261029.md) for rejected hardware
probes, the VM creation failure, cumulative billing and remaining acceptance work.
