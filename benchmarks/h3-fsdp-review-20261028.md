# H3: RAM-efficient FSDP2 loading review follow-up (2026-10-28)

This addresses [PR #42's P2 review finding](https://github.com/SoupTeam/soup-rnd/pull/42#discussion_r4393465170)
against `d5fb39d55e6c7f1398bc06d3a6adc296d8290964`. It changes the
`cpu_ram_efficient_loading=True` preparation path. Earlier hardware campaigns
used `False`; their commands, raw artifacts and measured source SHAs are retained.

## Trigger and correction

Soup previously called joint `fully_shard([embed, head])` during SFT setup.
Accelerate 1.14 captures `original_sd = model.state_dict()` later. Under
RAM-efficient loading, that source then contains DTensors rather than full CPU
weights. Accelerate's loader converts them through `to_local()` and distributes
the resulting local tensor as if it were the full embedding. Global DTensor
shape alone does not establish that the source is complete.

For this loading mode Soup now defers joint placement to Accelerate's
auto-wrap-policy factory boundary, after full-state capture, meta conversion
and re-tying, and before any child wrapping. The exact installed upstream
preparation code runs with an instance-local factory binding. Its caller is
also rebound only on this accelerator. No Accelerate module/class globals are
patched, no upstream preparation implementation is copied into Soup, and the
RAM-efficient option remains enabled. Unknown binding APIs fail before sharding.
Non-efficient loading retains the previously measured eager placement.

The boundary is an Accelerate implementation detail, checked against the
installed 1.14.0 stack. Future changes to its order require rerunning these
integration tests; binding-name checks alone cannot establish that order.

## Local regression coverage

`tests/test_h3_fsdp_tied_embeddings.py` launches two independent CPU/Gloo ranks
with real Torch 2.13 DTensors, public `fully_shard`, real broadcast/distribution,
Accelerate's `_prepare_fsdp2` caller, full-state loader and optimizer remapping.
It exercises both RAM-efficient settings with FP32 and BF16 source parameters.
The fixture supplies lightweight plugin/accelerator state and bypasses unrelated
`_prepare_one` object/device preparation; it is not an end-to-end Trainer run.

Checks include:

- No eager DTensor creation in the RAM-efficient mode; installation is idempotent.
- Full ordinary 8x4 source embeddings reach the loader on both ranks. Only rank 0
  starts with the real weights in the efficient case; rank 1 starts with zeros.
- Actual local shards are 4x4 and reconstruct the original weights exactly on
  both ranks. Embedding/head retain shared weight identity and one FSDP state.
- BF16 trainable parameters are upcast to FP32 before full-state capture, and
  the optimizer refers to the final prepared adapter parameter.
- Non-persistent buffers survive meta loading; global Accelerate functions stay
  unchanged. Existing untied/trainable/FSDP1/non-distributed no-op tests remain.
- An unsupported preparation boundary fails without changing the tied weights.

The two-rank tests have a 90-second deadline per case and terminate their own
workers on failure. They skip only when distributed/Gloo is unavailable.

Reproduce on the pinned Torch 2.13 / Accelerate 1.14 stack:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python -m pytest \
  tests/test_h3_fsdp_tied_embeddings.py --no-cov -q --tb=short
python -m ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

Validation logs and source/environment hashes are recorded in
[`evidence/h3-fsdp-review-20261028/`](evidence/h3-fsdp-review-20261028/).
The expanded local selection passed **337 tests**, including all four real
two-rank cases; full-scope Ruff and `git diff --check` passed. This count is one
selection, not a sum with earlier overlapping results. A controlled mutation
that restores eager sharding fails the efficient-loading regression before
entering the mismatched broadcast; the source was restored byte-for-byte.

## Limits

This follow-up proves the tested preparation/loading and parameter-remapping
contract on CPU. It does not establish GPU forward/backward correctness for
RAM-efficient loading, optimizer/scheduler resume, or new scaling results.
No GPU was rented for this change. The previous GPU results keep their original
scope. Ring attention, AutoTP, pipeline, multi-node, 4xA100 and other unmeasured
directions are not completed by this fix; full H3 closure still needs an agreed
scope and reviewer acceptance.
