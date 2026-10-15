# D1 local preparation evidence, 2026-10-15

**D1 is not complete. No CUDA measurement, pretrained-model training or cloud
rental has occurred in this preparation.**

- `doctor/result.json`: torch 2.13.0+cpu, no CUDA. Includes the imported source path.
- `cpu-correctness/result.json`: first failed attempt, missing repository fixture
  import path. Retained; no tensor comparison was completed in this attempt.
- `cpu-correctness-attempt2/`: corrected CPU run and replayable tensor files.
- `cpu-replay/result.json`: independent replay of that first successful CPU run.
- `cpu-correctness-final/`: final CPU positive/negative control and streaming run.
- `cpu-replay-final/result.json`: independent tensor replay, all comparisons exact
  and the finite deliberately broken gradient detected.
- `qa-data-audit.json`: all 448 public QA examples tokenized without truncation;
  prompt/padding mask and generation headroom checked with the pinned real tokenizer.

The QA audit initially failed because this transformers version returned a
BatchEncoding from apply_chat_template. The new regression reproduced it before
the fix; prompt() now explicitly requests list output and accepts mapping output
for compatibility. The successful audit is a preparation result, not task quality.

Validation command:

```bash
pytest tests/test_d1_nf4_study.py tests/test_d1_qa.py tests/test_issue842_checkpoint_visible_nf4.py tests/test_issue968_bnb_nf4_path_divergence.py --no-cov -q
```

Result: **50 passed, 13 skipped** (hardware-dependent existing cases).
`ruff check src/soup_cli/ scripts/ tests/ benchmarks/`: clean.
The full repository suite/matrix is not represented by these checks.

Each result's harness hash identifies the executed local bytes. The preparation
runs precede the commit of the rule; no CUDA arm has run before the rule commit.
Tiny CPU fixtures validate the instruments, not CUDA dispatch or the large alias
threshold. The D1 native source remains unchanged in this branch.
