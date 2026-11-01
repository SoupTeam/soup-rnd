# Short original BF16 versus NF4 quality screen

Frozen before allocation or observing quality results; separate from synthetic F1
and the real7B native/candidate RAM streaming screen. No change to those rules.

- Original `mistralai/Mistral-7B-v0.3`, revision
  `caa1feb0e54d415e2df31207e5f4e273e33509b1`; verify all three original shard SHA256s.
- Dataset: [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
  revision `b08601e04326c79dfdd32d625aee71d232d685c3`,
  `wikitext-2-raw-v1/test-00000-of-00001.parquet` (CC BY-SA 3.0 / GFDL).
- Join all original test rows with two newlines, tokenize without added special tokens.
  Select 32 nonoverlapping 512-token windows evenly over the complete token stream:
  start `floor(i * (token_count - 512) / 31)`, i=0..31. No post-result selection.
  Score all 511 causal next-token targets in each window: 16,352 targets total.
- Separate fresh processes: resident original BF16, then resident bitsandbytes nested
  NF4 with BF16 compute, all 224 decoder linears quantized, embeddings/head BF16.
  Same tokens, software, GPU, SDPA, eval/inference mode, no KV cache, seed842, TF32off.
  No LoRA or training. No experimental D1 kernel. Clocks do not affect the metric;
  no performance verdict, clock lock or VM is required.
- Metrics: float32 per-token cross-entropy, token-weighted mean NLL, exp(mean NLL)
  perplexity, paired delta NLL, relative perplexity change, descriptive top1 agreement.
  Retain all per-token NLLs/predictions, input hash, offsets and model/data hashes.
  Recompute summaries independently on CPU. Nonfinite values/incomplete coverage fail.
- This is an English pretrained quantization screen, not proof of unchanged quality,
  full benchmark perplexity, RAM streaming correctness or quality after training.
  Report the actual difference; no arbitrary equivalence threshold or quality gate.
- One RTX3090 Docker rental, at most45minutes, max $1.50 all-in within D1 $15.
  Latest visible cumulative $8.887; reserve $1 for delayed prior bandwidth billing.
  Select rate <=$0.40/h, download >=400Mbps, reliability >=0.97, 1GPU, >=48GBRAM,
  and bounded networking. Keep failed attempts; stop on correctness/resource failure.
  Export/hash evidence and destroy the exact labelled rental immediately afterward.
