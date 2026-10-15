#!/usr/bin/env python3
"""Research-only OLMoE frozen-prefix disk cache equivalence/invalidation probe.

Tested target: transformers 5.19.0, a SMALL RANDOMLY INITIALIZED OlmoeForCausalLM.
This is NOT a pretrained evaluation, production integration, speed benchmark,
or evidence of final E2 acceptance.

The cache stores the boundary hidden state and EVERY frozen-prefix router's
pre-softmax logits together, in one atomically replaced safetensors file.
It keys entries by actual input IDs, masks, position IDs, model identity,
configuration and the frozen weights. It deliberately supports only explicit,
non-KV-cache, input-ID-based forwards with output_router_logits=True.

Outputs: report.json, separate .safetensors cache files in --output-dir.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import types
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import OlmoeConfig, OlmoeForCausalLM, __version__ as transformers_version

from soup_cli.utils.frozen_prefix_cache import (
    build_cache_metadata,
    cache_key,
    fingerprint_frozen_prefix,
    fingerprint_prefix_configuration,
)

SCHEMA = "olmoe-research-disk-cache-v1"
MODEL_REVISION = "random-olmoe-4layers-seed42-research-only"
SEED = 42
CUTOFF = 2
NUM_STEPS = 3


def assert_close(label: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=label)
    return float((actual.detach() - expected.detach()).abs().max().item())


class ResearchOlmoeCache:
    """Temporary decoder patch; fail closed on unsupported OLMoE forward modes."""

    def __init__(self, model: OlmoeForCausalLM, directory: Path, cutoff: int) -> None:
        self.model = model
        self.decoder = model.model
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cutoff = cutoff
        self.hits = 0
        self.misses = 0
        self.skipped_frozen_layer_calls = 0
        self._original_forward = None
        self._hooks = []
        self._capture_hidden: torch.Tensor | None = None
        self._capture_routers: dict[int, torch.Tensor] = {}

        if not 0 < cutoff < len(self.decoder.layers):
            raise ValueError("A nonempty frozen and trainable stack is required")
        parameters = list(self.decoder.embed_tokens.parameters())
        for layer in self.decoder.layers[:cutoff]:
            parameters.extend(layer.parameters())
        if any(p.requires_grad for p in parameters):
            raise ValueError("All embedding/frozen-prefix parameters must be frozen")
        for module in [self.decoder.embed_tokens, *self.decoder.layers[:cutoff]]:
            for child in module.modules():
                if isinstance(child, torch.nn.modules.dropout._DropoutNd) and child.training and child.p:
                    raise ValueError("Active frozen-prefix dropout is unsupported")
        if getattr(self.decoder.config, "attention_dropout", 0.0):
            raise ValueError("Frozen attention dropout must be zero")

    def _metadata(self, input_ids, attention_mask, position_ids) -> dict[str, Any]:
        return build_cache_metadata(
            model_revision=MODEL_REVISION,
            frozen_prefix_fingerprint=fingerprint_frozen_prefix(self.decoder, self.cutoff),
            config_fingerprint=fingerprint_prefix_configuration(self.decoder, self.cutoff),
            cutoff=self.cutoff,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def _path(self, metadata: dict[str, Any]) -> Path:
        return self.directory / f"{cache_key(metadata)}.safetensors"

    def _save(self, metadata, hidden, routers) -> None:
        if not isinstance(hidden, torch.Tensor) or len(routers) != self.cutoff:
            raise ValueError("Incomplete frozen-prefix cache contents")
        tensors = {"hidden_states": hidden.detach().cpu().contiguous()}
        for i, logits in enumerate(routers):
            if not isinstance(logits, torch.Tensor):
                raise TypeError("Expected a router tensor")
            tensors[f"router_{i:04d}"] = logits.detach().cpu().contiguous()
        destination = self._path(metadata)
        fd, temp = tempfile.mkstemp(prefix=".olmoe-cache-", suffix=".tmp", dir=self.directory)
        try:
            os.close(fd)
            save_file(tensors, temp)
            os.replace(temp, destination)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def _load(self, metadata, embeddings):
        path = self._path(metadata)
        if not path.is_file():
            return None
        tensors = load_file(str(path), device="cpu")
        expected = {"hidden_states", *(f"router_{i:04d}" for i in range(self.cutoff))}
        if set(tensors) != expected:
            raise RuntimeError("Invalid cached OLMoE tensor set")
        hidden = tensors["hidden_states"]
        batch, seq = embeddings.shape[:2]
        if tuple(hidden.shape) != tuple(embeddings.shape) or hidden.dtype != embeddings.dtype:
            raise RuntimeError("Cached hidden-state shape/dtype mismatch")
        count_experts = self.decoder.config.num_experts
        routers = []
        for i in range(self.cutoff):
            tensor = tensors[f"router_{i:04d}"]
            if tuple(tensor.shape) != (batch * seq, count_experts):
                raise RuntimeError("Cached router logits shape mismatch")
            if tensor.dtype != embeddings.dtype:
                raise RuntimeError("Cached router logits dtype mismatch")
            routers.append(tensor.to(device=embeddings.device))
        return hidden.to(device=embeddings.device), tuple(routers)

    def __enter__(self):
        if self._original_forward is not None:
            raise RuntimeError("OLMoE cache already installed")
        self._original_forward = self.decoder.forward

        def boundary_hook(_module, _args, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Unexpected OLMoE boundary output")
            self._capture_hidden = output.detach().clone()

        self._hooks.append(self.decoder.layers[self.cutoff - 1].register_forward_hook(boundary_hook))
        for index, layer in enumerate(self.decoder.layers[:self.cutoff]):
            # OlmoeSparseMoeBlock.gate is an OlmoeTopKRouter. Its first result
            # contains pre-softmax logits used by router auxiliary loss.
            def router_hook(_module, _args, output, i=index):
                if not isinstance(output, (tuple, list)) or not isinstance(output[0], torch.Tensor):
                    raise RuntimeError("Unexpected OLMoE router output")
                self._capture_routers[i] = output[0].detach().clone()

            self._hooks.append(layer.mlp.gate.register_forward_hook(router_hook))

        def forward_override(decoder_self, *args, **kwargs):
            if decoder_self is not self.decoder:
                raise RuntimeError("Model replica/sharding is unsupported in this research probe")
            if args or kwargs.get("input_ids") is None or kwargs.get("inputs_embeds") is not None:
                raise ValueError("Use keyword input_ids only; inputs_embeds/positional args unsupported")
            if kwargs.get("past_key_values") is not None or kwargs.get("use_cache") not in (None, False):
                raise ValueError("KV cache is unsupported")
            if kwargs.get("output_router_logits") is not True:
                raise ValueError("output_router_logits=True is required by this probe")
            ids = kwargs["input_ids"]
            mask = kwargs.get("attention_mask")
            positions = kwargs.get("position_ids")
            if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor) or not isinstance(positions, torch.Tensor):
                raise ValueError("Explicit tensor input_ids, attention_mask and position_ids required")

            with torch.no_grad():
                embeddings = self.decoder.embed_tokens(ids)
            metadata = self._metadata(ids, mask, positions)
            found = self._load(metadata, embeddings)
            if found is None:
                self.misses += 1
                self._capture_hidden = None
                self._capture_routers.clear()
                result = self._original_forward(**kwargs)
                routers = tuple(self._capture_routers.get(i) for i in range(self.cutoff))
                if self._capture_hidden is None or any(x is None for x in routers):
                    raise RuntimeError("Missing hidden/router tensors on cache miss")
                if result.router_logits is None or len(result.router_logits) != len(self.decoder.layers):
                    raise RuntimeError("Original OLMoE router capture is incomplete")
                self._save(metadata, self._capture_hidden, routers)
                return result

            cached_hidden, frozen_routers = found
            self.hits += 1
            original_layers = [layer.forward for layer in self.decoder.layers[:self.cutoff]]
            for i, layer in enumerate(self.decoder.layers[:self.cutoff]):
                def skip_forward(_layer, hidden_states, *other, _index=i, **kw):
                    self.skipped_frozen_layer_calls += 1
                    return cached_hidden if _index == self.cutoff - 1 else hidden_states

                layer.forward = types.MethodType(skip_forward, layer)
            try:
                result = self._original_forward(**kwargs)
            finally:
                for layer, original in zip(self.decoder.layers[:self.cutoff], original_layers):
                    layer.forward = original
            # capture_outputs sees only executed upper layer routers on hit.
            upper = result.router_logits
            if upper is None or len(upper) != len(self.decoder.layers) - self.cutoff:
                raise RuntimeError("Unexpected upper-router count on cache hit")
            result.router_logits = (*frozen_routers, *upper)
            return result

        self.decoder.forward = types.MethodType(forward_override, self.decoder)
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.decoder.forward = self._original_forward
        self._original_forward = None
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        return False


def make_models():
    torch.manual_seed(SEED)
    config = OlmoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        router_aux_loss_coef=0.01,
        output_router_logits=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    original = OlmoeForCausalLM(config)
    for model in (original,):
        for p in model.model.embed_tokens.parameters():
            p.requires_grad_(False)
        for layer in model.model.layers[:CUTOFF]:
            for p in layer.parameters():
                p.requires_grad_(False)
    return copy.deepcopy(original), copy.deepcopy(original)


def run_step(model, ids, mask, positions, labels, backward=True):
    model.zero_grad(set_to_none=True)
    if backward:
        output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                       labels=labels, output_router_logits=True, use_cache=False)
        output.loss.backward()
    else:
        with torch.no_grad():
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           labels=labels, output_router_logits=True, use_cache=False)
    if output.aux_loss is None or output.router_logits is None or len(output.router_logits) != 4:
        raise RuntimeError("Incomplete MoE router/auxiliary result")
    if not torch.isfinite(output.loss).item() or not torch.isfinite(output.aux_loss).item():
        raise RuntimeError("Non-finite MoE loss")
    grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    return {
        "logits": output.logits.detach().clone(),
        "loss": output.loss.detach().clone(),
        "aux_loss": output.aux_loss.detach().clone(),
        "routers": tuple(x.detach().clone() for x in output.router_logits),
        "grads": grads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists():
        parser.error(f"Refusing to overwrite existing output dir: {root}")
    root.mkdir(parents=True)
    cache_dir = root / "cache"
    torch.set_num_threads(min(4, torch.get_num_threads()))

    baseline, cached = make_models()
    baseline.train()
    cached.train()
    torch.manual_seed(SEED + 1)
    ids = torch.randint(3, 256, (1, 12))
    mask = torch.ones_like(ids)
    mask[0, -1] = 0
    positions = torch.arange(ids.shape[1]).unsqueeze(0)
    labels = ids.clone()
    labels[mask == 0] = -100
    optim_a = torch.optim.SGD((p for p in baseline.parameters() if p.requires_grad), lr=1e-3)
    optim_b = torch.optim.SGD((p for p in cached.parameters() if p.requires_grad), lr=1e-3)

    steps = []
    invalidation = {}
    with ResearchOlmoeCache(cached, cache_dir, CUTOFF) as cache:
        for step in range(NUM_STEPS):
            a = run_step(baseline, ids, mask, positions, labels)
            b = run_step(cached, ids, mask, positions, labels)
            difference = {
                "logits": assert_close("logits", b["logits"], a["logits"]),
                "loss": assert_close("loss", b["loss"], a["loss"]),
                "aux_loss": assert_close("auxiliary loss", b["aux_loss"], a["aux_loss"]),
            }
            if set(a["grads"]) != set(b["grads"]) or not a["grads"]:
                raise RuntimeError("Trainable gradient keys differ or are empty")
            max_grad = 0.0
            for name in a["grads"]:
                max_grad = max(max_grad, assert_close(name, b["grads"][name], a["grads"][name]))
            difference["gradients"] = max_grad
            for left, right in zip(a["routers"], b["routers"]):
                assert_close("router logits", right, left)
            optim_a.step()
            optim_b.step()
            for (na, pa), (nb, pb) in zip(baseline.named_parameters(), cached.named_parameters()):
                if na != nb:
                    raise RuntimeError("Parameter ordering differs")
                if pa.requires_grad:
                    assert_close(f"updated parameter {na}", pb, pa)
            steps.append({"step": step + 1, "maximum_absolute_differences": difference})
            print(f"STEP {step + 1}: equivalent; cache hits={cache.hits}, misses={cache.misses}", flush=True)

        if (cache.hits, cache.misses, cache.skipped_frozen_layer_calls) != (2, 1, 4):
            raise RuntimeError("Unexpected cache miss/hit or frozen-layer skipping counts")

        def check_miss(name, changed_ids, changed_mask, changed_positions):
            before = cache.misses
            new_labels = changed_ids.clone()
            new_labels[changed_mask == 0] = -100
            run_step(cached, changed_ids, changed_mask, changed_positions, new_labels, backward=False)
            if cache.misses != before + 1:
                raise RuntimeError(f"Expected an invalidation miss for {name}")
            invalidation[name] = True
            print(f"INVALIDATION {name}: miss confirmed", flush=True)

        ids_changed = ids.clone()
        ids_changed[0, 0] = (int(ids_changed[0, 0]) % 253) + 3
        if torch.equal(ids_changed, ids):
            ids_changed[0, 0] = 3 if ids[0, 0] != 3 else 4
        check_miss("input_ids", ids_changed, mask, positions)

        mask_changed = mask.clone()
        mask_changed[0, -1] = 1
        check_miss("attention_mask", ids, mask_changed, positions)

        positions_changed = positions.clone()
        positions_changed[0, -1] += 1
        check_miss("position_ids", ids, mask, positions_changed)

        with torch.no_grad():
            cached.model.embed_tokens.weight[3, 0] += 0.01
        check_miss("frozen_weights", ids, mask, positions)

        counts = {
            "hits": cache.hits,
            "misses": cache.misses,
            "skipped_frozen_layer_calls": cache.skipped_frozen_layer_calls,
        }
        if counts != {"hits": 2, "misses": 5, "skipped_frozen_layer_calls": 4}:
            raise RuntimeError(f"Unexpected final cache counts: {counts}")

    report = {
        "schema": SCHEMA,
        "kind": "olmoe_disk_cache_multistep_and_invalidation",
        "model": MODEL_REVISION,
        "architecture": "random_4layer_olmoe",
        "pretrained": False,
        "production_e2_modified": False,
        "performance_benchmark": False,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cutoff": CUTOFF,
        "steps": steps,
        "cache": counts,
        "invalidation": invalidation,
        "verdict": "EXPLORATORY_CORRECTNESS_PASS_ONLY",
        "research_acceptance": "NO VERDICT",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nOLMOE DISK CACHE MULTISTEP EQUIVALENCE AND INVALIDATION PASSED", flush=True)
    print("Report:", root / "report.json")


if __name__ == "__main__":
    main()
