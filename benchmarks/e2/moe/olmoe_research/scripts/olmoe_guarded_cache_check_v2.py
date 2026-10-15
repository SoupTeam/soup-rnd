#!/usr/bin/env python3
"""Fail-closed toy OLMoE v2 guard checks before timing pretrained K8.

Run on Vast with installed Soup and transformers 5.19.0.
Tests multi-step baseline/cache parity, in-place frozen weight invalidation,
upper-layer parameter changes, and explicitly flagged `.data` changes.
No production E2 modification, pretrained quality or performance verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import torch

PINNED_V1_SHA256 = "20c3dbdd1e7515c89d7c22967f4d2e87a1b3d10bcb2134b3d81a0c73edc05b6b"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--guard", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("Refusing to overwrite an existing report")
    if hashlib.sha256(a.probe.read_bytes()).hexdigest() != PINNED_V1_SHA256:
        raise RuntimeError("Original cache v1 SHA mismatch")
    probe = load_module("olmoe_verified_v1", a.probe)
    guard = load_module("olmoe_guarded_v2", a.guard)
    CacheV2 = guard.guarded_cache_type(probe.ResearchOlmoeCache)

    baseline, cached = probe.make_models()
    baseline.train()
    cached.train()
    torch.manual_seed(probe.SEED + 1)
    ids = torch.randint(3, 256, (1, 12))
    mask = torch.ones_like(ids)
    mask[0, -1] = 0
    pos = torch.arange(12).unsqueeze(0)
    labels = ids.clone()
    labels[mask == 0] = -100
    optim_a = torch.optim.SGD((p for p in baseline.parameters() if p.requires_grad), lr=1e-3)
    optim_b = torch.optim.SGD((p for p in cached.parameters() if p.requires_grad), lr=1e-3)

    with tempfile.TemporaryDirectory(prefix="olmoe-guard-v2-") as td:
        with CacheV2(cached, Path(td), cutoff=probe.CUTOFF) as cache:
            cache.set_model_revision("random-olmoe-4layers@guard-v2")
            for step in range(3):
                x = probe.run_step(baseline, ids, mask, pos, labels)
                y = probe.run_step(cached, ids, mask, pos, labels)
                for key in ("logits", "loss", "aux_loss"):
                    probe.assert_close(key, y[key], x[key])
                assert len(x["routers"]) == len(y["routers"]) == 4
                for xx, yy in zip(x["routers"], y["routers"]):
                    probe.assert_close("router", yy, xx)
                assert x["grads"].keys() == y["grads"].keys()
                for name in x["grads"]:
                    probe.assert_close(name, y["grads"][name], x["grads"][name])
                optim_a.step()
                optim_b.step()
                for (name_a, p_a), (name_b, p_b) in zip(baseline.named_parameters(), cached.named_parameters()):
                    assert name_a == name_b
                    if p_a.requires_grad:
                        probe.assert_close(name_a, p_b.detach(), p_a.detach())
                assert cache.fingerprint_recomputes == 1, "Upper-layer optimizer must not force a full rehash"
                print(f"STEP {step+1}: parity PASS, full SHA256 calls={cache.fingerprint_recomputes}", flush=True)
            assert (cache.hits, cache.misses, cache.skipped_frozen_layer_calls) == (2, 1, 4)

            def check_miss(label, ids_arg, mask_arg, pos_arg):
                previous = cache.misses
                new_labels = ids_arg.clone()
                new_labels[mask_arg == 0] = -100
                probe.run_step(cached, ids_arg, mask_arg, pos_arg, new_labels, backward=False)
                assert cache.misses == previous + 1, f"Expected a miss for {label}"
                print(f"INVALIDATION {label}: PASS", flush=True)

            changed_ids = ids.clone()
            changed_ids[0, 0] = 3 if ids[0, 0].item() != 3 else 4
            check_miss("input_ids", changed_ids, mask, pos)
            changed_mask = mask.clone()
            changed_mask[0, -1] = 1
            check_miss("attention_mask", ids, changed_mask, pos)
            changed_pos = pos.clone()
            changed_pos[0, -1] += 1
            check_miss("position_ids", ids, mask, changed_pos)
            assert cache.fingerprint_recomputes == 1
            with torch.no_grad():
                cached.model.embed_tokens.weight[3, 0] += 0.01
            check_miss("frozen_weight_normal_inplace", ids, mask, pos)
            assert cache.fingerprint_recomputes == 2

            # Deliberate unsafe bypass is made safe by explicit notification.
            # Without mark_prefix_changed(), .data writes may silently reuse stale cache.
            with torch.no_grad():
                cached.model.embed_tokens.weight.data[3, 0] += 0.01
            cache.mark_prefix_changed()
            check_miss("frozen_weight_data_with_explicit_invalidation", ids, mask, pos)
            assert cache.fingerprint_recomputes == 3
            assert cache.explicit_invalidations == 1
            assert (cache.hits, cache.misses, cache.skipped_frozen_layer_calls) == (2, 6, 4)
            report = {
                "kind": "olmoe_guarded_fingerprint_v2_toy_correctness",
                "pretrained": False,
                "research_acceptance": "NO VERDICT",
                "verdict": "EXPLORATORY_CORRECTNESS_PASS_ONLY",
                "cache": {"hits": cache.hits, "misses": cache.misses, "skipped_frozen_layer_calls": cache.skipped_frozen_layer_calls},
                "guard": {"fingerprint_recomputes": cache.fingerprint_recomputes, "metadata_calls": cache.metadata_calls, "explicit_invalidations": cache.explicit_invalidations},
                "v1_sha256": hashlib.sha256(a.probe.read_bytes()).hexdigest(),
                "guard_sha256": hashlib.sha256(a.guard.read_bytes()).hexdigest(),
                "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\nOLMOE GUARDED FINGERPRINT V2 TOY CORRECTNESS PASS", flush=True)
    print("REPORT:", a.output, flush=True)


if __name__ == "__main__":
    main()
