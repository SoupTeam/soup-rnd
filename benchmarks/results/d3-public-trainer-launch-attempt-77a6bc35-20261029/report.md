# D3 public trainer launch attempt — 2026-10-29

Status: **no verdict**. This is a retained failed launch, not a pretrained
training or performance result.

- Frozen source head: `77a6bc357e524974191e6822a748d22953761871`.
- The protocol readiness check passed with 234 minimum and 414 maximum cells.
- The host and prepared-manifest preflight passed for the pinned GPU and all 157
  prepared files.
- The parent launched the first correctness cell, but the child exited before
  model setup with `ModuleNotFoundError: No module named 'soup_cli'`.
- Cause: the execution environment set `PYTHONPATH` to the checkout root but
  omitted its `src` directory.
- No pretrained forward, optimizer step, timing, or CUDA performance result was
  produced. The cell status is retained as `no_verdict`.
- The invocation correction was checked separately by importing
  `soup_cli.trainer.sft` and `benchmarks.harness.d3_trainer_gate` with both the
  checkout root and `checkout_root/src` on `PYTHONPATH`.

Retained files are the exact session record, frozen protocol, failed cell
result, and its empty process log. Do not delete or substitute this attempt.
A subsequent formal series requires a new committed head and must preserve this
record; the attempt budget and numerical criteria are otherwise unchanged.
