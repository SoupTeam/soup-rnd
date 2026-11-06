"""Native decoder boundary diagnostics, outside training timing and acceptance gates.

The caller admits and owns the resident model and an idle, resident activation store.
Only one microbatch's decoder captures are retained at a time. Diagnostic CPU copies,
replays and local VJPs are not training memory or whole-step performance measurements.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _helpers() -> tuple[Any, Any]:
    try:
        from . import g4_activation_codec, p3_fp8_smoke
    except ImportError:
        import g4_activation_codec
        import p3_fp8_smoke
    return g4_activation_codec, p3_fp8_smoke


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def _cpu(value: Any) -> Any:
    return value.detach().to("cpu", copy=True)


def _range(value: Any) -> dict[str, Any]:
    import torch

    values = value.detach().double()
    finite = torch.isfinite(values)
    result: dict[str, Any] = {
        "dtype": str(value.dtype), "shape": list(value.shape),
        "elements": value.numel(), "finite": bool(finite.all()),
        "nonfinite_elements": int((~finite).sum()),
    }
    if not result["finite"] or not value.numel():
        return result
    absolute = values.abs()
    result.update({
        "min": float(values.min()), "max": float(values.max()),
        "absmax": float(absolute.max()), "zero_elements": int((values == 0).sum()),
        "rms": float(values.square().mean().sqrt()),
    })
    if value.ndim:
        rows = absolute.reshape(-1, value.shape[-1]).amax(dim=-1)
        result["row_absmax_min"] = float(rows.min())
        result["row_absmax_max"] = float(rows.max())
    return result


def _router_capture(module: Any, args: Any, output: Any) -> dict[str, Any]:
    """Read native IDs; separate explanation math never supplies a layer's routes."""
    import torch

    result: dict[str, Any] = {"input": _cpu(args[0])}
    if not isinstance(output, tuple) or len(output) != 3:
        result["missing"] = "Expected native router tuple (logits, weights, ordered IDs)"
        return result
    logits, weights, indices = output
    if indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        result["missing"] = "Native output[2] is not a two-dimensional integer ID tensor"
        return result
    result.update({
        "ids": _cpu(indices), "weights": _cpu(weights), "logits": _cpu(logits),
        "parameter_dtype": str(module.weight.dtype),
    })
    required = ("num_group", "topk_group", "top_k", "e_score_correction_bias")
    if any(not hasattr(module, name) for name in required):
        result["explanation_missing"] = "Native router lacks grouped sigmoid routing attributes"
        return result
    groups, keep = int(module.num_group), int(module.topk_group)
    experts = int(logits.shape[-1])
    if groups < 1 or experts % groups or experts // groups < 2 or not 0 < keep <= groups:
        result["explanation_missing"] = "Unsupported grouped top-two score dimensions"
        return result
    # Compute on the router's device, not a CPU recreation of the float32 linear map.
    with torch.no_grad():
        scores = logits.detach().sigmoid()
        bias = module.e_score_correction_bias.detach()
        choice = scores + bias
        group_scores = choice.reshape(-1, groups, experts // groups).topk(2, dim=-1)
        group_scores = group_scores.values.sum(dim=-1)
        selected_groups = group_scores.topk(keep, dim=-1, sorted=False).indices
        mask = torch.zeros_like(group_scores, dtype=torch.bool)
        mask.scatter_(1, selected_groups, True)
        eligible = mask.unsqueeze(-1).expand(-1, groups, experts // groups)
        eligible = eligible.reshape(-1, experts)
        explanatory_ids = choice.masked_fill(~eligible, -math.inf).topk(
            int(module.top_k), dim=-1, sorted=False,
        ).indices
    result.update({
        "scores": _cpu(scores), "choice": _cpu(choice), "bias": _cpu(bias),
        "group_scores": _cpu(group_scores), "selected_groups": _cpu(selected_groups),
        "eligible": _cpu(eligible), "explanatory_ids": _cpu(explanatory_ids),
        "group_size": experts // groups,
    })
    return result


class _Observation:
    """Short-lived read-only hooks, shared by forward captures and layer replays."""

    def __init__(self, layer: Any) -> None:
        self.values: dict[str, Any] = {}
        self.handles: list[Any] = []
        try:
            attention = getattr(layer, "self_attn", None)
            norm = getattr(layer, "post_attention_layernorm", None)
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if attention is not None:
                self.handles.append(attention.register_forward_hook(self._attention))
            if norm is not None:
                self.handles.append(norm.register_forward_pre_hook(self._residual))
            if gate is not None and hasattr(gate, "e_score_correction_bias"):
                self.handles.append(gate.register_forward_hook(self._router))
        except BaseException:
            self.close()
            raise

    def _attention(self, _module: Any, _args: Any, output: Any) -> None:
        self.values["attention_output"] = _cpu(_hidden(output))

    def _residual(self, _module: Any, args: Any) -> None:
        self.values["attention_residual"] = _cpu(args[0])

    def _router(self, module: Any, args: Any, output: Any) -> None:
        self.values["router"] = _router_capture(module, args, output)

    def take(self) -> dict[str, Any]:
        result, self.values = self.values, {}
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()


def _margin(selected: Any, excluded: Any) -> float | None:
    if not selected.numel() or not excluded.numel():
        return None
    return float(selected.min() - excluded.max())


def _token_explanation(capture: dict[str, Any], token: int) -> dict[str, Any]:
    import torch

    if "choice" not in capture:
        return {"status": "MISSING", "reason": capture.get("explanation_missing", "No scores")}
    native = capture["ids"][token]
    choice = capture["choice"][token]
    selected = torch.zeros_like(choice, dtype=torch.bool)
    selected[native.long()] = True
    groups = capture["selected_groups"][token]
    group_scores = capture["group_scores"][token]
    selected_group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    selected_group_mask[groups.long()] = True
    eligible = capture["eligible"][token]
    return {
        "status": "OBSERVED_EXPLANATION", "token_flat": token,
        "native_ordered_ids": native.tolist(),
        "native_weights": capture["weights"][token].tolist(),
        "selected_sigmoid_scores": capture["scores"][token, native.long()].tolist(),
        "selected_bias": capture["bias"][native.long()].tolist(),
        "selected_choice_scores": choice[native.long()].tolist(),
        "expert_margin_within_explanatory_groups": _margin(
            choice[selected], choice[eligible & ~selected],
        ),
        "expert_margin_all_experts": _margin(choice[selected], choice[~selected]),
        "explanatory_selected_groups": groups.tolist(),
        "native_selected_expert_groups": sorted(set((native // capture["group_size"]).tolist())),
        "group_scores": group_scores.tolist(),
        "group_margin": _margin(
            group_scores[selected_group_mask], group_scores[~selected_group_mask],
        ),
        "explanation_ordered_ids": capture["explanatory_ids"][token].tolist(),
        "explanation_order_matches_native": bool(torch.equal(
            capture["explanatory_ids"][token], native,
        )),
        "absent_margin_reason": "No excluded eligible expert/group yields null margin",
    }


def _routing(original: dict[str, Any], restored: dict[str, Any], shape: Any) -> dict[str, Any]:
    left, right = original.get("router"), restored.get("router")
    if left is None and right is None:
        return {"status": "NOT_APPLICABLE", "reason": "Dense decoder layer has no native router"}
    if left is None or right is None or "ids" not in left or "ids" not in right:
        return {
            "status": "MISSING", "reason": "Native router IDs were not captured on both paths",
            "original": None if left is None else left.get("missing"),
            "restored": None if right is None else right.get("missing"),
        }
    metrics, _smoke = _helpers()
    want, got = left["ids"], right["ids"]
    if want.shape != got.shape:
        return {"status": "MISSING", "reason": "Native router ID shapes differ"}
    ordered = (want != got).any(dim=-1)
    set_changed = (want.sort(dim=-1).values != got.sort(dim=-1).values).any(dim=-1)
    changed = ordered.nonzero().flatten().tolist()
    changes = []
    for token in changed:
        before, after = set(want[token].tolist()), set(got[token].tolist())
        changes.append({
            "token_flat": token, "batch_row": token // int(shape[-2]),
            "sequence_position": token % int(shape[-2]),
            "removed_experts": sorted(before - after), "added_experts": sorted(after - before),
            "set_changed": bool(set_changed[token]),
        })
    # Explain the first changed token and the tightest original expert/group margins.
    examples = changed[:1]
    tight: list[tuple[float, int]] = []
    for token in range(want.shape[0]):
        explanation = _token_explanation(left, token)
        margins = [explanation.get(key) for key in (
            "expert_margin_within_explanatory_groups", "group_margin",
        )]
        finite = [value for value in margins if value is not None and math.isfinite(value)]
        if finite:
            tight.append((min(finite), token))
    examples += [token for _margin_value, token in sorted(tight)[:2] if token not in examples]
    if not examples and want.shape[0]:
        examples = [0]
    return {
        "status": "EQUAL" if not changed else "DIVERGED",
        "authority": "Native gate.forward output[2], never explanatory top-k calculations",
        "original_ordered_ids": want.tolist(), "restored_ordered_ids": got.tolist(),
        "ordered_changed_tokens": len(changed), "set_changed_tokens": int(set_changed.sum()),
        "changed_tokens": changes,
        "first_changed_token": changes[0] if changes else None,
        "router_input_original": _range(left["input"]),
        "router_input_restored": _range(right["input"]),
        "router_input_error": metrics.restore_error(left["input"], right["input"]),
        "native_logits_original": _range(left["logits"]),
        "native_logits_restored": _range(right["logits"]),
        "native_logit_error": metrics.restore_error(left["logits"], right["logits"]),
        "native_selected_weight_dtype": str(left["weights"].dtype),
        "native_router_parameter_dtype": left["parameter_dtype"],
        "grouped_explanation": {
            "source": "Native returned logits.sigmoid() plus native correction bias on device",
            "formula": "Group score=sum(top2 biased scores); top groups then top experts",
            "authority": "Explanation only, no acceptance threshold and no forced routes",
            "tie_limit": "Unsorted top-k order/ties are authoritative only in native output",
            "original_missing": left.get("explanation_missing"),
            "native_bias": _range(left["bias"]) if "bias" in left else None,
            "score_dtype": str(left["scores"].dtype) if "scores" in left else None,
            "restored_missing": right.get("explanation_missing"),
            "examples": [{
                "original": _token_explanation(left, token),
                "restored": _token_explanation(right, token),
                "router_input_error": metrics.restore_error(
                    left["input"].reshape(-1, left["input"].shape[-1])[token:token + 1],
                    right["input"].reshape(-1, right["input"].shape[-1])[token:token + 1],
                ),
            } for token in examples],
        },
    }


def _perturbation(original: dict[str, Any], restored: dict[str, Any]) -> dict[str, Any]:
    metrics, _smoke = _helpers()
    result = {}
    for name in ("attention_output", "attention_residual"):
        if name not in original or name not in restored:
            result[name] = {"status": "MISSING", "reason": "Native hook did not observe boundary"}
            continue
        result[name] = {
            "original": _range(original[name]), "restored": _range(restored[name]),
            "error": metrics.restore_error(original[name], restored[name]),
        }
    result["source"] = (
        "Same decoder: boundary -> native self_attn output -> residual addition -> "
        "post_attention_layernorm input -> native router input"
    )
    return result


def _vjp(layer: Any, original: Any, restored: Any, args: Any, kwargs: Any) -> dict[str, Any]:
    """A deterministic local attribution, deliberately not a production gradient gate."""
    import torch

    metrics, _smoke = _helpers()
    if getattr(layer, "gradient_checkpointing", False):
        return {"status": "MISSING", "reason": "Native layer gradient checkpointing is enabled"}
    params = [(name, param) for name, param in layer.named_parameters() if param.requires_grad]
    gradients: list[dict[str, Any]] = []
    routes: list[Any] = []
    output_dtypes: list[str] = []
    missing: list[str] = []
    for boundary in (original, restored):
        observation = _Observation(layer)
        try:
            with torch.enable_grad():
                leaf = boundary.detach().clone().requires_grad_(True)
                output = _hidden(layer(leaf, *args, **kwargs))
                output_dtypes.append(str(output.dtype))
                upstream = torch.arange(output.numel(), device=output.device, dtype=torch.float32)
                upstream = (upstream.remainder(23) - 11) / (11 * math.sqrt(output.numel()))
                upstream = upstream.reshape(output.shape).to(output.dtype)
                values = torch.autograd.grad(
                    output, (leaf, *(param for _name, param in params)),
                    grad_outputs=upstream, allow_unused=True,
                )
            captured = observation.take()
            routes.append(captured.get("router", {}).get("ids"))
            names = ["decoder_input", *(name for name, _param in params)]
            gradients.append({name: _cpu(value) for name, value in zip(names, values)
                              if value is not None})
            missing.extend(name for name, value in zip(names, values) if value is None)
        finally:
            observation.close()
    if missing:
        return {"status": "MISSING", "reason": "Unused diagnostic differentiation targets",
                "targets": sorted(set(missing))}
    left, right = routes
    if (left is None) != (right is None) or (
        left is not None and not torch.equal(left, right)
    ):
        return {"status": "MISSING", "reason": "Grad-enabled replay changed native routes"}
    input_summary = metrics.grad_agreement(
        {"decoder_input": gradients[0]["decoder_input"]},
        {"decoder_input": gradients[1]["decoder_input"]},
    )
    adapter_summary = None
    if params:
        adapter_summary = metrics.grad_agreement(
            {name: gradients[0][name] for name, _param in params},
            {name: gradients[1][name] for name, _param in params},
        )
        adapter_summary.pop("per_tensor", None)
    input_summary.pop("per_tensor", None)
    return {
        "status": "OBSERVED", "gate": False,
        "kind": "Same native layer local VJP, not loss gradient or production gradient gate",
        "upstream": "((flat_index % 23)-11)/(11*sqrt(output.numel())), cast to output dtype",
        "input_gradient": input_summary, "trainable_gradient": adapter_summary,
        "trainable_dtypes": sorted({str(param.dtype) for _name, param in params}),
        "output_dtypes": output_dtypes,
        "native_ordered_routes_stable_in_grad_replays": True,
        "parameters_grad_fields": "Unchanged: autograd.grad does not accumulate .grad",
        "_native_ids": None if left is None else left.tolist(),
    }


def _scales(store: Any, layer: int, mb: int, source: Any) -> dict[str, Any]:
    if not hasattr(store, "_scale_view"):
        return {"status": "NOT_APPLICABLE", "reason": "Uncompressed store has no codec scales"}
    if getattr(store, "is_cuda", False):
        # Wait only for this put before reading the host scale slab.
        event = store._events.get((layer, mb))
        if event is not None:
            event.synchronize()
    rows = math.prod(source.shape[:-1])
    scale = store._scale_view((layer, mb), rows)
    result = _range(scale)
    result.update({
        "status": "OBSERVED", "source": "Production store's per-token absmax scale slab",
        "semantics": "absmax, not absmax/448; codec normalizes then multiplies by 448",
        "payload_dtype": str(store._payload[(layer, mb)].dtype),
    })
    return result


def diagnose_decoder(
    model: Any, store: Any, ids: Sequence[Any], *, device: str, input_set: str,
) -> dict[str, Any]:
    """Compare native decoder inputs with production put/get restores without route overrides.

    The supplied store must be idle and have no spilled layers. Its reservation remains
    caller-owned. Captures are scratch writes, not a checkpoint of a live training step.
    Hooks, RNG, store statistics and idle slot validity are restored on every path.
    """
    import torch

    from soup_cli.utils.l2l_step import model_parts

    metrics, smoke = _helpers()
    bounds = smoke.thresholds()
    result: dict[str, Any] = {
        "status": "MISSING", "input_set": input_set, "device": device,
        "complete": False, "finite": True,
        "source": "Resident native decoder forward inputs/kwargs -> product store.put/get",
        "model_class": type(model).__name__, "store_class": type(store).__name__,
        "microbatches": len(ids), "bounds_source": "p3_fp8_smoke.thresholds -> g4_rule",
        "bounds": bounds, "chunks": [], "missing_diagnostics": [],
        "route_policy": "Native ordered IDs only; no monkeypatch, route forcing or new tolerance",
        "candidate_policy": "None selected",
        "limitations": [
            "Local single-boundary restores, not cumulative compressed-forward propagation",
            "Decoder-local VJPs do not attribute embedding/head adapters or full-step gradients",
            "Diagnostic VJP samples first route-stable microbatch per layer, not a training gate",
            "Diagnostic copies and replays are outside step timing and memory peaks",
            "Store's existing lease covers production slots only, not diagnostic CPU clones",
            "Caller owns admission, deterministic seeds, precision policy and final acceptance",
        ],
    }
    if not ids or getattr(store, "spill_layers", 0):
        result["missing_diagnostics"].append("Requires nonempty ids and resident non-spill store")
        return result
    if getattr(store, "_closed", False) or getattr(store, "_closing", False):
        result["missing_diagnostics"].append("Caller store is closed or closing")
        return result
    if getattr(store, "_shapes", {}):
        result["missing_diagnostics"].append(
            "Store must be idle; live slot validity is not overwritten",
        )
        return result
    if len(ids) > store.k:
        result["missing_diagnostics"].append("Input window exceeds admitted store capacity")
        return result
    dropout = [type(module).__name__ for module in model.modules()
               if module.training and isinstance(module, torch.nn.Dropout) and module.p]
    if dropout:
        result["missing_diagnostics"].append("Active dropout prevents deterministic native replays")
        return result
    _causal, owner = model_parts(model)
    layers = owner.layers
    if len(layers) != store.n_layers:
        result["missing_diagnostics"].append("Decoder layer count differs from admitted store")
        return result
    if getattr(_causal.config, "use_cache", False):
        result["missing_diagnostics"].append("Native fixture must have use_cache=False")
        return result
    target = torch.device(device)
    if torch.is_autocast_enabled(target.type):
        result["missing_diagnostics"].append(
            "Invoke diagnostic outside autocast, including its local VJP backward",
        )
        return result
    store_device = torch.device(store.device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    if store_device.type == "cuda" and store_device.index is None:
        store_device = torch.device("cuda", torch.cuda.current_device())
    if store_device != target:
        result["missing_diagnostics"].append("Diagnostic device differs from admitted store device")
        return result
    result["precision_observed"] = {
        "model_parameter_dtypes": sorted({str(param.dtype) for param in model.parameters()}),
        "trainable_parameter_dtypes": sorted({str(param.dtype) for param in model.parameters()
                                               if param.requires_grad}),
        "store_compute_dtype": str(store.dtype),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "autocast": "No diagnostic autocast context; caller must invoke outside autocast",
    }
    stats_before = dict(vars(store.stats))
    event_state = {
        name: getattr(store, name) for name in ("_last_event", "_last_stream", "_cuda_used")
        if hasattr(store, name)
    }
    events_before = dict(store._events) if hasattr(store, "_events") else None
    observations: list[_Observation] = []
    handles: list[Any] = []
    captures: dict[int, dict[str, Any]] = {}
    e1: dict[int, list[Any]] = {index: [] for index in range(len(layers))}
    e2: dict[int, list[Any]] = {index: [] for index in range(len(layers))}
    vjp_layers: set[int] = set()
    first_divergence = None
    errors: list[str] = []
    cleanup_errors: list[str] = []
    started_store = False
    cuda_devices = []
    if target.type == "cuda":
        cuda_devices = [
            target.index if target.index is not None else torch.cuda.current_device(),
        ]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            for mb, batch in enumerate(ids):
                captures.clear()
                try:
                    for index, layer in enumerate(layers):
                        observation = _Observation(layer)
                        observations.append(observation)

                        def capture_input(
                            _module: Any, args: Any, kwargs: Any, *, layer_index: int = index,
                        ) -> None:
                            boundary = args[0] if args else kwargs["hidden_states"]
                            tail = kwargs.copy()
                            tail.pop("hidden_states", None)
                            captures[layer_index] = {
                                "input": _cpu(boundary), "args": args[1:], "kwargs": tail,
                            }
                            store.put(layer_index, mb, boundary)

                        def capture_output(
                            _module: Any, _args: Any, output: Any, *, layer_index: int = index,
                        ) -> None:
                            captures[layer_index]["output"] = _cpu(_hidden(output))

                        handles.append(layer.register_forward_pre_hook(
                            capture_input, with_kwargs=True,
                        ))
                        handles.append(layer.register_forward_hook(capture_output))
                    started_store = True
                    with torch.no_grad():
                        model(input_ids=batch)
                    for index, observation in enumerate(observations):
                        captures[index]["observation"] = observation.take()
                finally:
                    for handle in handles:
                        handle.remove()
                    handles.clear()
                    for observation in observations:
                        observation.close()
                    observations.clear()
                store.begin_backward()
                for index in reversed(range(len(layers))):
                    capture = captures.pop(index)
                    original = capture["input"].to(target)
                    restored = store.get(index, mb)
                    restored_cpu = _cpu(restored)
                    restore_error = metrics.restore_error(capture["input"], restored_cpu)
                    e1[index].append(restore_error)
                    observation = _Observation(layers[index])
                    try:
                        with torch.no_grad():
                            output = _hidden(layers[index](
                                restored, *capture["args"], **capture["kwargs"],
                            ))
                        replay = observation.take()
                    finally:
                        observation.close()
                    branch_error = metrics.restore_error(
                        capture["output"] - capture["input"], _cpu(output - restored),
                    )
                    e2[index].append(branch_error)
                    routes = _routing(capture["observation"], replay, original.shape)
                    if routes["status"] == "DIVERGED":
                        divergence = {"layer": index, "microbatch": mb,
                                      **routes["first_changed_token"]}

                        def key(item: dict[str, Any]) -> tuple[int, int, int]:
                            return item["layer"], item["microbatch"], item["token_flat"]

                        if first_divergence is None or key(divergence) < key(first_divergence):
                            first_divergence = divergence
                    row = {
                        "layer": index, "microbatch": mb,
                        "boundary_original": _range(capture["input"]),
                        "boundary_restored": _range(restored_cpu), "e1": restore_error,
                        "scales": _scales(store, index, mb, original), "e2": branch_error,
                        "perturbation": _perturbation(capture["observation"], replay),
                        "routing": routes,
                    }
                    if index not in vjp_layers and routes["status"] in ("EQUAL", "NOT_APPLICABLE"):
                        try:
                            attribution = _vjp(
                                layers[index], original, restored,
                                capture["args"], capture["kwargs"],
                            )
                            observed_ids = attribution.pop("_native_ids", None)
                            if routes["status"] == "EQUAL" and attribution["status"] == "OBSERVED":
                                if observed_ids != routes["original_ordered_ids"]:
                                    attribution = {
                                        "status": "MISSING",
                                        "reason": "Grad-enabled native IDs differ from no-grad IDs",
                                    }
                            row["gradient_attribution"] = attribution
                        except Exception as error:
                            row["gradient_attribution"] = {
                                "status": "MISSING", "reason": f"{type(error).__name__}: {error}",
                            }
                        vjp_layers.add(index)
                    else:
                        row["gradient_attribution"] = {
                            "status": "NOT_RUN", "reason": (
                                "Bounded one-microbatch sample already attempted for layer"
                                if index in vjp_layers else "Native routes diverge or are missing"
                            ),
                        }
                    result["chunks"].append(row)
                    capture.clear()
                    original = restored = restored_cpu = output = replay = None
                store.end_step()
                started_store = False
                # The uncompressed store's end_step intentionally retains these shapes.
                store._shapes.clear()
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        for handle in handles:
            handle.remove()
        for observation in observations:
            observation.close()
        captures.clear()
        if started_store:
            try:
                store.end_step()
            except Exception as error:
                cleanup_errors.append(f"{type(error).__name__}: {error}")
        drained = True
        if target.type == "cuda":
            try:
                torch.cuda.synchronize(target)
            except Exception as error:
                drained = False
                cleanup_errors.append(f"Diagnostic drain: {type(error).__name__}: {error}")
        if drained:
            if events_before is not None:
                store._events.clear()
                store._events.update(events_before)
            for name, value in event_state.items():
                setattr(store, name, value)
        store._shapes.clear()
        for name, value in stats_before.items():
            setattr(store.stats, name, value)
    rows = {str(index): metrics.merge_errors(parts) for index, parts in e1.items() if parts}
    branches = {str(index): metrics.merge_errors(parts) for index, parts in e2.items() if parts}
    complete = len(result["chunks"]) == len(layers) * len(ids) and not errors and not cleanup_errors
    e1_ok = complete and smoke._finite_tree(rows) and all(
        row["rms_row_rel"] <= bounds["restore_rms_row_rel_max"]
        and row["worst_row_rel"] <= bounds["restore_worst_row_rel_max"]
        for row in rows.values()
    )
    e2_ok = complete and smoke._finite_tree(branches) and all(
        row["rms_row_rel"] <= bounds["branch_rms_row_rel_max"] for row in branches.values()
    )
    result.update({
        "status": "OBSERVED" if complete else "MISSING", "complete": complete,
        "e1": {"status": "PASS" if e1_ok else "FAIL", "layers": rows},
        "e2": {"status": "PASS" if e2_ok else "FAIL", "layers": branches},
        "first_divergent_native_route": first_divergence,
        "first_divergence_order": "layer, microbatch, flattened token, not replay execution order",
        "errors": errors, "cleanup_errors": cleanup_errors,
        "store_state": {
            "slot_validity": "Idle; diagnostic scratch payload is invalid after return",
            "caller_lease": "Retained; caller closes the store",
            "statistics_restored": True,
            "async_metadata_restored": drained,
            "cleanup_complete": not cleanup_errors,
        },
    })
    for row in result["chunks"]:
        for name in ("routing", "gradient_attribution"):
            finding = row[name]
            if finding["status"] == "MISSING":
                result["missing_diagnostics"].append({
                    "layer": row["layer"], "microbatch": row["microbatch"],
                    "diagnostic": name, "reason": finding.get("reason"),
                })
        grouped = row["routing"].get("grouped_explanation", {})
        for path in ("original_missing", "restored_missing"):
            if grouped.get(path):
                result["missing_diagnostics"].append({
                    "layer": row["layer"], "microbatch": row["microbatch"],
                    "diagnostic": f"grouped_routing.{path}", "reason": grouped[path],
                })
        for name in ("attention_output", "attention_residual"):
            finding = row["perturbation"][name]
            if finding.get("status") == "MISSING":
                result["missing_diagnostics"].append({
                    "layer": row["layer"], "microbatch": row["microbatch"],
                    "diagnostic": name, "reason": finding["reason"],
                })
    nonfinite: list[str] = []
    result = smoke._json_safe(result, nonfinite)
    result["finite"] = not nonfinite
    result["nonfinite_metric_paths"] = nonfinite
    return result
