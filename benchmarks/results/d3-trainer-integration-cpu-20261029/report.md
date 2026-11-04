# D3 trainer integration: synthetic CPU validation

Validation date: 2026-10-29 (Asia/Qyzylorda). **931 passed,
8 skipped, zero failures**, including
**22 new D3 integration cases**, in one pytest process.
The raw JUnit and terminal log are retained. GPU-marked cases were deselected;
the full repository suite and any CUDA execution are not claimed.
After the combined run, the 22 integration cases were separately rechecked with
stronger dtype/shape/byte comparisons (including signed zero): all passed.
That test-only strengthening is present in the snapshot; runtime code is unchanged.

This is an **uncommitted working-tree** check, based on HEAD `98706ed06319603db802fda1a86b5b7f1f97b33e`
with pending merge parent `9d8b1d7b4b3eabc1634a1f6c2318814c640535c3`. The author/operator
will commit and push. `source-snapshot.zip` and actual working-file SHA-256 values
identify the tested sources; these are not relabeled as the previous measured commit.

The opt-in regular SFT builder installs B1 expert shards and D3 bounded expert
reload/backward. Native OLMoE routing remains intact; routed weights stay frozen
and attention LoRA trains. The merged A3 L2L trainer also accepts this OLMoE path.
See [integration design](../../../docs/d3-trainer-integration.md).

Checks cover exact native dense loss/gradients and repeated updates, empty/failure
and budget regressions, NF4/nested-NF4 backward, L2L k=1/2 with ignored labels and
dropout/RNG, real SFTTrainer steps (including a shorter final accumulation window),
adapter save/restoration, atomic installation and resource cleanup on error.
The broader retained suites cover existing B1/B2, D3, A3 and streaming behavior.

The installed TRL 0.29.1 casts QLoRA adapters to BF16 during Trainer construction.
During development, loading those saved factors into FP32 adapters was rejected
by exact logit comparison. Restoration through the same Trainer/dtype recipe
matches exactly. No comparison tolerance was relaxed; factor values and dtypes
are checked too. This is a same-recipe trainer replay, not an arbitrary inference
loader/dtype compatibility claim.

The tested models use small deterministic **synthetic random weights and tokens**.
They do not provide a new result for trained OLMoE or certify grouped CUDA/Triton.
The new mode explicitly refuses router auxiliary loss (including a nonzero config
coefficient), routed-expert adapters and unsupported architectures/backends.
Only first derivatives are supported. Memory reservations are not total-VRAM claims.

Next qualification needs a separately committed/frozen pretrained CUDA trainer
protocol, matching reference/candidate numerics and F1 controls. No new rental,
commit, push or remote measurement was performed for these CPU checks. The old
84-cell fixture result and every retained failed formal attempt stay unchanged.
Final D3 acceptance remains with the reviewer.
