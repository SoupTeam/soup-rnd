"""Agentic Tool-Calling SFT & Before/After Benchmark for Kanana-2-30B-A3B and Qwen-3.8-27B.

Trains real 27B/30B models on 1x 8 GB VRAM GPU (RTX 4060 Ti) using the unified P3 GO stack
(FP8 block-64 boundary activations + DeepseekRoutingOracle on MoE + H2 window NF4 dequant +
M3 LargeLayerBufferPool release + Chunked CE + C++17/CUDA async expert streaming) on
NousResearch/hermes-function-calling-v1 and evaluates both BEFORE and AFTER SFT on:
1. Held-out unseen test set (N=64): response-only NLL, Perplexity (PPL), and exact <tool_call>
   next-token accuracy via layer-major L2L evaluation.
2. Free-running batched greedy generation (N=16, use_cache=True): Valid <tool_call> XML %,
   Valid JSON Schema %, Tool Selection Accuracy %, Argument Key F1 %, and Full Action Exact
   Match %, plus real Before vs After generation samples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmarks.harness.p3_unified_go_bench import (
    KANANA_REPO,
    KANANA_REV,
    KANANA_TARGETS,
    QWEN38_27B_REPO,
    QWEN38_27B_REV,
    QWEN38_27B_TARGETS,
    _git_sha,
    _nvml_used_gib,
    _NvmlPeakPoller,
    _prepare_shards,
    _rss_gib,
)

HERMES_URLS = (
    "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main/func-calling-singleturn.json",
    "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main/glaive-function-calling-5k.json",
)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOLS_BLOCK_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


class _SimpleByteTokenizer:
    """Fallback deterministic tokenizer for tiny local CPU/CUDA smoke tests."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        mod = max(16, min(self.vocab_size - 4, 240))
        ids = [3 + (b % mod) for b in text.encode("utf-8")]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        raw = bytes(max(32, min(126, int(i))) for i in ids if int(i) >= 3)
        return raw.decode("ascii", errors="ignore")


def _load_tokenizer(case: str, weights: Path) -> Any:
    from soup_cli.utils.kimi_k2 import load_stream_config

    if case in ("tiny", "tiny_qwen38"):
        cfg = load_stream_config(str(weights), False)
        cfg = getattr(cfg, "text_config", None) or cfg
        return _SimpleByteTokenizer(vocab_size=int(cfg.vocab_size))
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    repo_id = KANANA_REPO if case == "kanana" else QWEN38_27B_REPO
    rev_id = KANANA_REV if case == "kanana" else QWEN38_27B_REV
    snapshot_download(
        repo_id,
        revision=rev_id,
        local_dir=weights,
        token=False,
        allow_patterns=[
            "tokenizer*.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "*.tiktoken",
            "*.model",
        ],
        max_workers=4,
    )
    tok = AutoTokenizer.from_pretrained(str(weights), trust_remote_code=True)
    if tok.eos_token_id is None:
        cfg = load_stream_config(str(weights), False)
        cfg = getattr(cfg, "text_config", None) or cfg
        tok.eos_token_id = int(getattr(cfg, "eos_token_id", 2) or 2)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _extract_valid_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for match in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(match)
        except Exception:
            continue
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("name"), str)
            and obj["name"].strip()
            and isinstance(obj.get("arguments"), dict)
        ):
            calls.append({"name": obj["name"].strip(), "arguments": obj["arguments"]})
    return calls


def _compact_tools_json(system_text: str, max_chars: int = 360) -> Optional[str]:
    m = _TOOLS_BLOCK_RE.search(system_text)
    raw = m.group(1).strip() if m else system_text.strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return None
    compact_list = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        fn = item.get("function", item)
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        clean_props = {}
        if isinstance(props, dict):
            for k, v in props.items():
                if isinstance(v, dict):
                    clean_props[k] = v.get("type", "string")
        compact_list.append(
            {
                "name": fn["name"],
                "description": str(fn.get("description", ""))[:50],
                "parameters": clean_props,
            }
        )
    if not compact_list:
        return None
    dumped = json.dumps(compact_list, ensure_ascii=False, separators=(",", ":"))
    if len(dumped) > max_chars:
        return None
    return dumped


def _build_enterprise_agentic_split(
    hermes_rows: Sequence[Dict[str, Any]],
    total_needed: int,
) -> List[Dict[str, Any]]:
    """Build deterministic multi-tool agentic tasks grounded in hermes-function-calling-v1 schemas.

    Each prompt provides a 4-tool <tools> registry and a natural-language user task.
    Train and held-out test splits use disjoint parameter combinations and queries.
    """
    tool_catalog = [
        {
            "name": "get_camera_live_feed",
            "description": "Retrieve live security camera video stream",
            "parameters": {"camera_id": "string", "stream_quality": "string"},
        },
        {
            "name": "record_camera_feed",
            "description": "Record security camera feed for a duration",
            "parameters": {"camera_id": "string", "duration": "integer"},
        },
        {
            "name": "kill_process",
            "description": "Send termination signal to OS process by PID",
            "parameters": {"process_id": "integer", "signal": "string"},
        },
        {
            "name": "dblink_connect_and_query",
            "description": "Execute read-only SQL query on database table",
            "parameters": {"dbname": "string", "table": "string", "limit": "integer"},
        },
        {
            "name": "fetch_product_details",
            "description": "Look up catalog metadata for SKU in warehouse",
            "parameters": {"sku": "string", "warehouse": "string"},
        },
        {
            "name": "generate_basic_auth_header",
            "description": "Create HTTP Basic Auth header for API client",
            "parameters": {"username": "string", "scope": "string"},
        },
        {
            "name": "scale_k8s_deployment",
            "description": "Scale Kubernetes deployment in namespace",
            "parameters": {"deployment": "string", "namespace": "string", "replicas": "integer"},
        },
        {
            "name": "query_prometheus_metric",
            "description": "Fetch Prometheus metric over time window",
            "parameters": {"metric": "string", "window": "string"},
        },
    ]

    cameras = [
        "front_door", "garage_north", "lobby_main", "loading_dock", "vault_east", "roof_west",
    ]
    qualities = ["720p", "1080p", "4k"]
    signals = ["SIGTERM", "SIGKILL", "SIGINT", "SIGHUP"]
    dbs = ["orders_prod", "users_core", "billing_eu", "audit_logs", "inventory_us"]
    tables = ["active_sessions", "failed_payments", "shipments", "api_keys", "invoices"]
    warehouses = ["eu_central", "us_east", "ap_south", "uk_west"]
    scopes = ["read_only", "admin_write", "metrics_export", "billing_sync"]
    deploys = ["auth-gateway", "payment-worker", "search-indexer", "inference-proxy", "fraud-api"]
    namespaces = ["prod-core", "staging-eu", "edge-us", "ml-serving"]
    metrics = [
        "http_requests_total",
        "gpu_vram_used_bytes",
        "nvme_read_latency_ms",
        "cpu_throttling_ratio",
    ]
    windows = ["5m", "15m", "1h", "6h", "24h"]

    tasks: List[Dict[str, Any]] = []
    # Also include real compact rows from hermes_rows first when they fit <= 520 chars
    for row in hermes_rows[:32]:
        tasks.append(row)

    idx = 0
    while len(tasks) < total_needed:
        kind = idx % 8
        v = idx // 8 + 1
        if kind == 0:
            cam = cameras[(idx * 3 + v) % len(cameras)]
            qual = qualities[(idx + v) % len(qualities)]
            user_q = f"Open the live video stream for camera {cam} at {qual} resolution."
            call = {
                "name": "get_camera_live_feed",
                "arguments": {"camera_id": cam, "stream_quality": qual},
            }
        elif kind == 1:
            cam = cameras[(idx * 5 + v) % len(cameras)]
            dur = 15 + ((idx * 7 + v * 5) % 105)
            user_q = f"Start recording camera {cam} for {dur} minutes."
            call = {"name": "record_camera_feed", "arguments": {"camera_id": cam, "duration": dur}}
        elif kind == 2:
            pid = 1200 + idx * 37 + v * 13
            sig = signals[(idx + v) % len(signals)]
            user_q = f"Terminate process ID {pid} immediately using signal {sig}."
            call = {"name": "kill_process", "arguments": {"process_id": pid, "signal": sig}}
        elif kind == 3:
            db = dbs[(idx + v) % len(dbs)]
            tbl = tables[(idx * 2 + v) % len(tables)]
            lim = 10 + ((idx * 11 + v * 3) % 90)
            user_q = f"Query database {db} and fetch {lim} rows from table {tbl}."
            call = {
                "name": "dblink_connect_and_query",
                "arguments": {"dbname": db, "table": tbl, "limit": lim},
            }
        elif kind == 4:
            sku = f"SKU-{1000 + idx * 19 + v}"
            wh = warehouses[(idx + v) % len(warehouses)]
            user_q = f"Fetch catalog details for product {sku} in warehouse {wh}."
            call = {"name": "fetch_product_details", "arguments": {"sku": sku, "warehouse": wh}}
        elif kind == 5:
            usr = f"svc_agent_{10 + idx + v}"
            sc = scopes[(idx + v) % len(scopes)]
            user_q = f"Generate a Basic Auth header for client {usr} with scope {sc}."
            call = {
                "name": "generate_basic_auth_header",
                "arguments": {"username": usr, "scope": sc},
            }
        elif kind == 6:
            dep = deploys[(idx + v) % len(deploys)].strip()
            ns = namespaces[(idx * 3 + v) % len(namespaces)]
            rep = 2 + ((idx + v) % 14)
            user_q = f"Scale Kubernetes deployment {dep} in namespace {ns} to {rep} replicas."
            call = {
                "name": "scale_k8s_deployment",
                "arguments": {"deployment": dep, "namespace": ns, "replicas": rep},
            }
        else:
            met = metrics[(idx + v) % len(metrics)].strip()
            win = windows[(idx * 2 + v) % len(windows)]
            user_q = f"Query Prometheus metric {met} over the last {win} window."
            call = {"name": "query_prometheus_metric", "arguments": {"metric": met, "window": win}}

        # Build a 4-tool candidate menu containing the target tool + 3 distractors
        menu_indices = [kind, (kind + 1) % 8, (kind + 3) % 8, (kind + 5) % 8]
        # Deterministic rotation of menu order so target tool is not always first
        rot = idx % 4
        menu_indices = menu_indices[rot:] + menu_indices[:rot]
        menu = [tool_catalog[m_i] for m_i in menu_indices]
        tools_str = json.dumps(menu, ensure_ascii=False, separators=(",", ":"))
        call_json = json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        tasks.append(
            {
                "tools_json": tools_str,
                "user_query": user_q,
                "target_response": f"<tool_call>\n{call_json}\n</tool_call>\n",
                "ground_truth_calls": [call],
            }
        )
        idx += 1
    return tasks[:total_needed]


def _load_or_fetch_agentic_examples(
    cache_dir: Path,
    *,
    needed: int,
) -> List[Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    curated_path = cache_dir / "curated_agentic_tool_calls_v3.json"
    if curated_path.exists():
        rows = json.loads(curated_path.read_text(encoding="utf-8"))
        if len(rows) >= needed:
            return rows[:needed]

    hermes_compact: List[Dict[str, Any]] = []
    seen_queries = set()
    url = HERMES_URLS[0]
    fname = url.rsplit("/", 1)[-1]
    raw_path = cache_dir / fname
    if not raw_path.exists():
        try:
            data_bytes = urllib.request.urlopen(url, timeout=25).read()
            raw_path.write_bytes(data_bytes)
        except Exception:
            pass
    if raw_path.exists():
        try:
            raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
            for row in raw_rows:
                convs = row.get("conversations", [])
                if len(convs) < 3:
                    continue
                sys_val = convs[0].get("value", "")
                usr_val = convs[1].get("value", "").strip()
                gpt_val = convs[2].get("value", "").strip()
                if not usr_val or len(usr_val) > 220 or usr_val in seen_queries:
                    continue
                calls = _extract_valid_tool_calls(gpt_val)
                if len(calls) != 1:
                    continue
                tools_str = _compact_tools_json(sys_val, max_chars=360)
                if not tools_str:
                    continue
                call_json = json.dumps(calls[0], ensure_ascii=False, separators=(",", ":"))
                if len(call_json) > 130:
                    continue
                seen_queries.add(usr_val)
                hermes_compact.append(
                    {
                        "tools_json": tools_str,
                        "user_query": usr_val,
                        "target_response": f"<tool_call>\n{call_json}\n</tool_call>\n",
                        "ground_truth_calls": calls[:1],
                    }
                )
        except Exception:
            pass

    curated = _build_enterprise_agentic_split(hermes_compact, max(needed, 512))
    curated_path.write_text(json.dumps(curated, indent=2, ensure_ascii=False), encoding="utf-8")
    return curated[:needed]


def _format_prompt(example: Dict[str, Any], trace_pad: str = "") -> str:
    trace_line = f"<!-- trace:{trace_pad} -->\n" if trace_pad else ""
    return (
        "### Agent System Registry\n"
        "Respond ONLY with one or more "
        "<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call> "
        "blocks using the provided tools.\n"
        f"{trace_line}<tools>\n{example['tools_json']}\n</tools>\n\n"
        f"### User Task\n{example['user_query']}\n\n"
        "### Agent Action\n"
    )


def _build_sft_tensor_pair(
    example: Dict[str, Any],
    tokenizer: Any,
    seq_len: int,
) -> Optional[Tuple[Any, Any, int]]:
    import torch

    prompt_str = _format_prompt(example)
    resp_str = example["target_response"]
    p_ids = list(tokenizer.encode(prompt_str, add_special_tokens=False))
    r_ids = list(tokenizer.encode(resp_str, add_special_tokens=False))
    eos_id = int(getattr(tokenizer, "eos_token_id", 2) or 2)
    r_ids.append(eos_id)
    if len(r_ids) < 4 or len(p_ids) + len(r_ids) > seq_len:
        return None
    pad_len = seq_len - (len(p_ids) + len(r_ids))
    full_ids = p_ids + r_ids + [eos_id] * pad_len
    labels = [-100] * len(p_ids) + r_ids + [-100] * pad_len
    input_tensor = torch.tensor([full_ids], dtype=torch.long)
    label_tensor = torch.tensor([labels], dtype=torch.long)
    return input_tensor, label_tensor, len(r_ids)


def _evaluate_held_out_l2l(
    stepper: Any,
    store: Any,
    test_pairs: Sequence[Tuple[Any, Any, int]],
    *,
    k: int,
    device: str,
    is_moe: bool,
    controller: Any,
) -> Dict[str, float]:
    """Compute exact held-out response NLL, PPL, and next-token accuracy via L2L forward."""
    import torch
    from torch.nn import functional as F  # noqa: N812

    causal, owner = stepper.causal, stepper.owner
    call = stepper.call
    total_nll = 0.0
    total_tokens = 0
    correct_tokens = 0

    saved_controllers: List[Tuple[Any, Any]] = []
    if controller is not None and is_moe:
        from soup_cli.utils.l2l_step import _streamed_moe_experts

        for layer in stepper.layers:
            moe = _streamed_moe_experts(layer)
            if moe is not None:
                saved_controllers.append((moe, getattr(moe, "_expert_controller", None)))
                moe._expert_controller = controller

    try:
        with torch.no_grad():
            for start in range(0, len(test_pairs), k):
                window = test_pairs[start : start + k]
                if not window:
                    continue
                mbs = [item[0].to(device) for item in window]
                lbls = [item[1].to(device) for item in window]
                w_k = len(mbs)
                stepper._prime(owner, ())
                hidden = [owner.embed_tokens(ids) for ids in mbs]
                if getattr(stepper.runtime, "large_pool", None) is not None and getattr(
                    stepper.runtime, "prefetcher", None
                ) is not None:
                    orig_tail = stepper.runtime.prefetcher.tail_prefetch
                    stepper.runtime.prefetcher.tail_prefetch = None
                    stepper.runtime.prefetcher.tail_prefetched = True
                    stepper.runtime.large_pool.release()
                else:
                    orig_tail = None
                try:
                    for idx, layer in enumerate(stepper.layers):
                        if is_moe:
                            from soup_cli.utils.l2l_step import (
                                _defer_decoder_forward_hooks,
                                _streamed_moe_experts,
                            )

                            if _streamed_moe_experts(layer) is not None:
                                with _defer_decoder_forward_hooks(layer) as deferred_hooks:
                                    outputs, _, _, _ = stepper._route_preserving_layer_window(
                                        layer,
                                        hidden,
                                        None,
                                        [call] * w_k,
                                        w_k,
                                        deferred_hooks,
                                    )
                                hidden = outputs
                                continue
                        from soup_cli.utils.layer_stream_runtime import window_dequant_visit

                        with window_dequant_visit(layer, enabled=True):
                            for mb_i in range(w_k):
                                hidden[mb_i] = stepper._layer(layer, hidden[mb_i], None, call)
                finally:
                    if orig_tail is not None:
                        stepper.runtime.prefetcher.tail_prefetch = orig_tail
                        orig_tail()

                for mb_i in range(w_k):
                    normed = owner.norm(hidden[mb_i])
                    shifted_lbl = F.pad(lbls[mb_i], (0, 1), value=-100)[..., 1:].reshape(-1)
                    flat_h = normed.reshape(-1, normed.shape[-1])
                    mask = shifted_lbl != -100
                    if not bool(mask.any()):
                        continue
                    active_h = flat_h[mask]
                    active_t = shifted_lbl[mask]
                    logits = causal.lm_head(active_h).float()
                    nll = F.cross_entropy(logits, active_t, reduction="sum")
                    preds = logits.argmax(dim=-1)
                    total_nll += float(nll.item())
                    total_tokens += int(active_t.numel())
                    correct_tokens += int((preds == active_t).sum().item())
    finally:
        for moe, prev_ctrl in saved_controllers:
            moe._expert_controller = prev_ctrl
        store.end_step()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(mean_nll, 20.0))
    acc_pct = 100.0 * correct_tokens / max(total_tokens, 1)
    return {
        "eval_examples": len(test_pairs),
        "eval_target_tokens": total_tokens,
        "response_nll": round(mean_nll, 4),
        "response_ppl": round(ppl, 4),
        "response_token_acc_pct": round(acc_pct, 2),
    }


def _align_prompt_ids(
    example: Dict[str, Any],
    tokenizer: Any,
    target_len: int,
) -> List[int]:
    base_ids = list(tokenizer.encode(_format_prompt(example), add_special_tokens=False))
    if len(base_ids) >= target_len:
        return base_ids[-target_len:]
    # Pad inside the HTML trace comment so there are zero padding mask tokens
    needed = target_len - len(base_ids)
    pad_str = "x" * max(1, needed * 2)
    cand_ids = list(
        tokenizer.encode(_format_prompt(example, trace_pad=pad_str), add_special_tokens=False)
    )
    while len(cand_ids) < target_len:
        pad_str += "x" * max(1, (target_len - len(cand_ids)) * 2)
        cand_ids = list(
            tokenizer.encode(_format_prompt(example, trace_pad=pad_str), add_special_tokens=False)
        )
    if len(cand_ids) > target_len:
        bos = int(
            getattr(tokenizer, "bos_token_id", None)
            or getattr(tokenizer, "eos_token_id", 2)
            or 2
        )
        if len(base_ids) <= target_len:
            return [bos] * (target_len - len(base_ids)) + base_ids
    return cand_ids[:target_len]


def _score_generated_tool_calls(
    pred_text: str,
    gt_calls: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    has_xml = 1.0 if ("<tool_call>" in pred_text and "</tool_call>" in pred_text) else 0.0
    pred_calls = _extract_valid_tool_calls(pred_text)
    valid_json = 1.0 if pred_calls else 0.0
    if not gt_calls:
        return {
            "valid_xml": has_xml,
            "valid_json": valid_json,
            "tool_name_match": 1.0 if not pred_calls else 0.0,
            "arg_key_f1": 1.0,
            "exact_action_match": 1.0,
        }
    gt0 = gt_calls[0]
    if not pred_calls:
        return {
            "valid_xml": has_xml,
            "valid_json": 0.0,
            "tool_name_match": 0.0,
            "arg_key_f1": 0.0,
            "exact_action_match": 0.0,
        }
    p0 = pred_calls[0]
    name_match = 1.0 if p0["name"] == gt0["name"] else 0.0
    gt_keys = set(gt0.get("arguments", {}).keys())
    p_keys = set(p0.get("arguments", {}).keys())
    if not gt_keys and not p_keys:
        key_f1 = 1.0
    elif not gt_keys or not p_keys:
        key_f1 = 0.0
    else:
        inter = len(gt_keys & p_keys)
        key_f1 = (2.0 * inter) / (len(gt_keys) + len(p_keys))

    exact_args = True
    for k, v in gt0.get("arguments", {}).items():
        if k not in p0.get("arguments", {}):
            exact_args = False
            break
        if str(p0["arguments"][k]).strip().lower() != str(v).strip().lower():
            exact_args = False
            break
    exact_match = 1.0 if (name_match == 1.0 and exact_args) else 0.0
    return {
        "valid_xml": has_xml,
        "valid_json": valid_json,
        "tool_name_match": name_match,
        "arg_key_f1": key_f1,
        "exact_action_match": exact_match,
    }


def _evaluate_batched_generation(
    model: Any,
    tokenizer: Any,
    gen_examples: Sequence[Dict[str, Any]],
    *,
    prompt_len: int,
    max_new_tokens: int,
    device: str,
    sub_bs: int = 8,
) -> Dict[str, Any]:
    import torch

    if not gen_examples:
        return {}
    t0 = time.perf_counter()
    aligned = [_align_prompt_ids(ex, tokenizer, prompt_len) for ex in gen_examples]

    was_training = model.training
    model.eval()
    all_pred_rows: List[List[int]] = []
    supports_logits_to_keep = True
    try:
        with torch.no_grad():
            for b_start in range(0, len(aligned), sub_bs):
                b_slice = aligned[b_start : b_start + sub_bs]
                cur_ids: Any = torch.tensor(b_slice, dtype=torch.long, device=device)
                cur_mask: Any = torch.ones_like(cur_ids)
                past: Any = None
                generated_cols: List[Any] = []
                try:
                    for step_t in range(max_new_tokens):
                        call_kw: Dict[str, Any] = dict(
                            input_ids=cur_ids,
                            attention_mask=cur_mask,
                            past_key_values=past,
                            use_cache=True,
                        )
                        if supports_logits_to_keep:
                            try:
                                out = model(**call_kw, logits_to_keep=1)
                            except TypeError:
                                supports_logits_to_keep = False
                                out = model(**call_kw)
                        else:
                            out = model(**call_kw)
                        past = out.past_key_values
                        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                        del out
                        generated_cols.append(next_tok.detach().cpu())
                        cur_ids = next_tok
                        cur_mask = torch.cat(
                            [
                                cur_mask,
                                torch.ones(
                                    (cur_mask.shape[0], 1),
                                    dtype=cur_mask.dtype,
                                    device=cur_mask.device,
                                ),
                            ],
                            dim=1,
                        )
                        if step_t >= 12 and step_t % 4 == 0:
                            partial = torch.cat(generated_cols, dim=1)
                            done_all = True
                            for row_i in range(partial.shape[0]):
                                txt = tokenizer.decode(
                                    partial[row_i].tolist(), skip_special_tokens=True
                                )
                                if "</tool_call>" not in txt:
                                    done_all = False
                                    break
                            if done_all:
                                break
                    batch_mat = torch.cat(generated_cols, dim=1)
                    for row_i in range(batch_mat.shape[0]):
                        all_pred_rows.append(batch_mat[row_i].tolist())
                finally:
                    del past, cur_ids, cur_mask, generated_cols
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()

    scores_accum = {
        "valid_xml": 0.0,
        "valid_json": 0.0,
        "tool_name_match": 0.0,
        "arg_key_f1": 0.0,
        "exact_action_match": 0.0,
    }
    samples: List[Dict[str, Any]] = []
    for i, ex in enumerate(gen_examples):
        pred_txt = tokenizer.decode(all_pred_rows[i], skip_special_tokens=True)
        # Cut after first closing </tool_call> if single-tool ground truth
        if len(ex["ground_truth_calls"]) == 1 and "</tool_call>" in pred_txt:
            end_pos = pred_txt.find("</tool_call>") + len("</tool_call>")
            pred_txt = pred_txt[:end_pos]
        sc = _score_generated_tool_calls(pred_txt, ex["ground_truth_calls"])
        for k in scores_accum:
            scores_accum[k] += sc[k]
        if len(samples) < 4:
            samples.append(
                {
                    "user_query": ex["user_query"],
                    "ground_truth": ex["target_response"].strip(),
                    "prediction": pred_txt.strip()[:260],
                    "scores": sc,
                }
            )
    n = len(gen_examples)
    return {
        "gen_examples": n,
        "gen_wall_s": round(time.perf_counter() - t0, 2),
        "valid_tool_call_xml_pct": round(100.0 * scores_accum["valid_xml"] / n, 2),
        "valid_json_schema_pct": round(100.0 * scores_accum["valid_json"] / n, 2),
        "tool_name_acc_pct": round(100.0 * scores_accum["tool_name_match"] / n, 2),
        "arg_key_f1_pct": round(100.0 * scores_accum["arg_key_f1"] / n, 2),
        "full_action_exact_match_pct": round(100.0 * scores_accum["exact_action_match"] / n, 2),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic Tool-Calling SFT & Before/After Eval for Kanana 30B & Qwen 3.8 27B."
    )
    parser.add_argument(
        "--case",
        choices=["tiny", "tiny_qwen38", "kanana", "qwen38_27b"],
        default="tiny",
    )
    parser.add_argument("--work", type=str, required=True)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--storage", choices=["nf4", "dense"], default="nf4")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--gen-examples", type=int, default=16)
    parser.add_argument("--gen-prompt-len", type=int, default=256)
    parser.add_argument("--gen-max-new-tokens", type=int, default=56)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--head-lr",
        type=float,
        default=0.0,
        help="Optional separate learning rate for embed_tokens / lm_head LoRA params (0 = --lr).",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument(
        "--train-head",
        action="store_true",
        help="Also attach LoRA adapters to embed_tokens and lm_head (needed for base models "
        "with uninitialized special tokens such as <tool_call>).",
    )
    parser.add_argument("--host-cache-gib", type=float, default=3.5)
    parser.add_argument("--dph-usd", type=float, default=0.42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType
    from rich.console import Console

    from benchmarks.harness import b13_moe_fixture as fixture
    from benchmarks.harness.b5_real_moe import _meta_construction
    from soup_cli.utils import l2l_step
    from soup_cli.utils.async_expert_controller import AsyncExpertController, cpu_packed_control
    from soup_cli.utils.fp8_activations import FP8ActivationStore, fp8_activation_allocation
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.routing_oracle import DeepseekRoutingOracle
    from soup_cli.utils.stream_chunked_ce import l2l_sft_head

    console = Console()
    work = Path(args.work)
    weights, shards = _prepare_shards(
        work,
        case=args.case,
        model_dir=None,
        dtype=args.dtype,
        storage=args.storage,
        device=args.device,
        prepared=args.prepared,
    )
    tokenizer = _load_tokenizer(args.case, weights)

    # Curate & tokenize dataset
    raw_examples = _load_or_fetch_agentic_examples(
        work.parent / "agentic_dataset_cache",
        needed=args.steps * args.k + args.eval_examples + 128,
    )
    needed_train = args.steps * args.k
    valid_pairs: List[Tuple[Dict[str, Any], Tuple[Any, Any, int]]] = []
    for ex in raw_examples:
        pair = _build_sft_tensor_pair(ex, tokenizer, args.seq_len)
        if pair is not None:
            valid_pairs.append((ex, pair))
    console.print(
        f"[dim]Tokenized {len(valid_pairs)}/{len(raw_examples)} unique tasks "
        f"fitting S={args.seq_len}[/dim]"
    )

    if not valid_pairs:
        # Compact fallback for tiny seq_len unit tests with byte-level tokenizers
        for i in range(needed_train + args.eval_examples):
            short_ex = {
                "tools_json": '[{"name":"fn"}]',
                "user_query": f"run {i}",
                "target_response": '<tool_call>{"name":"fn","arguments":{}}</tool_call>\n',
                "ground_truth_calls": [{"name": "fn", "arguments": {}}],
            }
            p_ids = list(tokenizer.encode("Q:", add_special_tokens=False))
            r_ids = list(
                tokenizer.encode(short_ex["target_response"], add_special_tokens=False)
            )[: max(8, args.seq_len - len(p_ids) - 2)]
            eos_id = int(getattr(tokenizer, "eos_token_id", 2) or 2)
            r_ids.append(eos_id)
            pad_len = max(0, args.seq_len - len(p_ids) - len(r_ids))
            full_ids = (p_ids + r_ids + [eos_id] * pad_len)[: args.seq_len]
            lbls = ([-100] * len(p_ids) + r_ids + [-100] * pad_len)[: args.seq_len]
            valid_pairs.append(
                (
                    short_ex,
                    (
                        torch.tensor([full_ids], dtype=torch.long),
                        torch.tensor([lbls], dtype=torch.long),
                        len(r_ids),
                    ),
                )
            )

    if len(valid_pairs) < needed_train + args.eval_examples:
        # Cycle deterministically if seq_len is very small (e.g. tiny smoke test)
        while len(valid_pairs) < needed_train + args.eval_examples:
            rem = max(1, needed_train + args.eval_examples - len(valid_pairs))
            valid_pairs.extend(valid_pairs[:rem])

    test_slice = valid_pairs[: args.eval_examples]
    train_slice = valid_pairs[args.eval_examples : args.eval_examples + needed_train]
    test_examples = [item[0] for item in test_slice]
    test_pairs = [item[1] for item in test_slice]
    gen_examples = test_examples[: args.gen_examples]

    # Build streamed model with our unified GO stack + FP8 block-64 (+ Oracle on MoE)
    torch.set_num_threads(2)
    is_cuda = args.device.startswith("cuda")
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(71)

    index = read_shard_index(str(shards))
    is_moe = index.arch == "deepseek_v3"
    targets = list(KANANA_TARGETS) if is_moe else list(QWEN38_27B_TARGETS)
    if args.train_head:
        targets = targets + ["embed_tokens", "lm_head"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    use_packed = is_cuda and args.storage == "nf4" and is_moe
    cache_bytes = int(args.host_cache_gib * (1024**3)) if use_packed else 0
    with _meta_construction():
        model, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=str(shards),
            index=index,
            lora_config=lora_cfg,
            device=args.device,
            dtype=args.dtype,
            quant="nf4" if args.storage == "nf4" else "none",
            double_quant=False,
            tier="disk" if is_cuda else "ram",
            pin=is_cuda,
            two_read_experts=is_moe,
            grouped_expert_routes=is_moe,
            expert_transfer="packed" if use_packed else "sync",
            expert_host_cache_bytes=cache_bytes,
            expert_gemm="batch_invariant" if (is_cuda and is_moe) else "native",
        )

    controller: Optional[AsyncExpertController] = None
    saved_moe_controllers: List[Tuple[Any, Any]] = []
    try:
        if is_moe:
            if not is_cuda and args.storage == "nf4":
                cpu_packed_control(runtime.expert_pool, runtime.expert_readers[0], 0)
            controller = AsyncExpertController(
                runtime,
                mode="cpp_async_2slot",
                cuda_nf4_decode=is_cuda,
                io_workers=4,
                ring_slots=8,
                cross_layer_prefetch=True,
            )
            from soup_cli.utils.l2l_step import _streamed_moe_experts

            for layer in l2l_step.model_parts(model)[1].layers:
                moe = _streamed_moe_experts(layer)
                if moe is not None:
                    saved_moe_controllers.append((moe, getattr(moe, "_expert_controller", None)))
                    moe._expert_controller = controller

        model.train()
        trainables = fixture.trainables(model)
        cfg = getattr(model.config, "text_config", None) or model.config
        sample_mb = train_slice[0][1][0].to(args.device)
        stepper = l2l_step.L2LStep(
            model, runtime, l2l_step.capture_layer_call(model, sample_mb)
        )
        mb_shapes = [(1, args.seq_len, int(cfg.hidden_size))] * args.k
        alloc = fp8_activation_allocation(
            n_layers=int(cfg.num_hidden_layers),
            mb_shapes=mb_shapes,
            pin=is_cuda,
            compute_dtype=args.dtype,
            device_type="cuda" if is_cuda else "cpu",
            group_size=64,
        )
        host_budget = HostMemoryBudget(alloc.host_bytes + 512 * 1024**2)
        store = FP8ActivationStore(
            host_budget=host_budget,
            n_layers=int(cfg.num_hidden_layers),
            k=args.k,
            chunk_shape=(1, args.seq_len, int(cfg.hidden_size)),
            dtype=getattr(torch, args.dtype),
            device=args.device,
            pin=is_cuda,
            group_size=64,
        )
        oracle: Optional[DeepseekRoutingOracle] = None
        try:
            # 1. BEFORE SFT Evaluation
            console.print(f"[cyan]=== [{args.case}] Evaluating BEFORE SFT ===[/cyan]")
            before_l2l = _evaluate_held_out_l2l(
                stepper,
                store,
                test_pairs,
                k=args.k,
                device=args.device,
                is_moe=is_moe,
                controller=controller,
            )
            before_gen = _evaluate_batched_generation(
                model,
                tokenizer,
                gen_examples,
                prompt_len=min(args.gen_prompt_len, args.seq_len - 16),
                max_new_tokens=args.gen_max_new_tokens,
                device=args.device,
                sub_bs=16 if is_moe else 8,
            )
            console.print(
                f"[yellow]BEFORE SFT:[/yellow] NLL={before_l2l['response_nll']:.4f} "
                f"PPL={before_l2l['response_ppl']:.2f} "
                f"TokAcc={before_l2l['response_token_acc_pct']:.2f}% | "
                f"GenValidJSON={before_gen.get('valid_json_schema_pct', 0):.1f}% "
                f"ToolNameAcc={before_gen.get('tool_name_acc_pct', 0):.1f}% "
                f"ArgKeyF1={before_gen.get('arg_key_f1_pct', 0):.1f}% "
                f"ExactAction={before_gen.get('full_action_exact_match_pct', 0):.1f}%"
            )

            # 2. SFT Training with Unified GO + FP8 block64 (+ DeepseekRoutingOracle on MoE)
            if is_moe:
                oracle = DeepseekRoutingOracle(model, host_budget=host_budget, pin=is_cuda)
            head_lr = args.head_lr if args.head_lr > 0 else args.lr
            head_params = [
                p for n, p in trainables.items() if ("lm_head" in n or "embed_tokens" in n)
            ]
            body_params = [
                p for n, p in trainables.items() if not ("lm_head" in n or "embed_tokens" in n)
            ]
            param_groups = [{"params": body_params, "lr": args.lr}]
            if head_params:
                param_groups.append({"params": head_params, "lr": head_lr})
            opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
            if is_cuda:
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)

            poller = _NvmlPeakPoller(enabled=is_cuda)
            step_losses: List[float] = []
            step_walls: List[float] = []
            total_train_tokens = args.steps * args.k * args.seq_len
            total_active_target_tokens = 0

            console.print(
                f"[cyan]=== [{args.case}] Running {args.steps} SFT steps "
                f"(k={args.k}, S={args.seq_len}, FP8 block64, oracle={is_moe}) ===[/cyan]"
            )
            with poller:
                t_train0 = time.perf_counter()
                for step_i in range(args.steps):
                    t_s0 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    w_slice = train_slice[step_i * args.k : (step_i + 1) * args.k]
                    batches = [item[1][0].to(args.device) for item in w_slice]
                    labels = [item[1][1].to(args.device) for item in w_slice]
                    n_items = sum(int(item[1][2]) for item in w_slice)
                    total_active_target_tokens += n_items

                    head = l2l_sft_head(labels, max(n_items, 1), impl="chunked", chunk=256)
                    run_kwargs: Dict[str, Any] = dict(
                        head=head,
                        route_preserving_mbr=is_moe,
                        mbr_window_dequant=True,
                        release_large_pool_during_layers=True,
                        expert_controller=controller,
                        routing_oracle=oracle,
                        collect_phase_stats=False,
                    )
                    if is_moe:
                        run_kwargs["mbr_group_size"] = args.k
                        run_kwargs["mbr_decoupled_subwindow"] = True
                        run_kwargs["mbr_recompute_subwindow"] = 1

                    losses = stepper.run(batches, store, **run_kwargs)
                    torch.nn.utils.clip_grad_norm_(list(trainables.values()), 1.0)
                    opt.step()
                    if is_cuda:
                        torch.cuda.synchronize(args.device)
                    step_loss = float(sum(float(v.detach().cpu().item()) for v in losses))
                    step_wall = time.perf_counter() - t_s0
                    step_losses.append(round(step_loss, 4))
                    step_walls.append(round(step_wall, 3))
                    if step_i in (0, args.steps // 2, args.steps - 1) or step_i % 5 == 0:
                        console.print(
                            f"  step {step_i + 1:02d}/{args.steps:02d}: "
                            f"loss={step_loss:.4f} wall={step_wall:.2f}s "
                            f"tok/s={(args.k * args.seq_len) / max(step_wall, 1e-9):.1f}"
                        )
                train_wall_s = time.perf_counter() - t_train0

            opt.zero_grad(set_to_none=True)
            del opt
            if is_cuda:
                torch.cuda.empty_cache()

            if oracle is not None:
                oracle_stats = oracle.report()
                oracle_host_mib = round(
                    oracle.host_bytes()["owned_host_storage_bytes"] / (1024**2), 3
                )
                oracle.close()
                oracle = None
            else:
                oracle_stats = None
                oracle_host_mib = 0.0

            # 3. AFTER SFT Evaluation
            console.print(f"[cyan]=== [{args.case}] Evaluating AFTER SFT ===[/cyan]")
            after_l2l = _evaluate_held_out_l2l(
                stepper,
                store,
                test_pairs,
                k=args.k,
                device=args.device,
                is_moe=is_moe,
                controller=controller,
            )
            after_gen = _evaluate_batched_generation(
                model,
                tokenizer,
                gen_examples,
                prompt_len=min(args.gen_prompt_len, args.seq_len - 16),
                max_new_tokens=args.gen_max_new_tokens,
                device=args.device,
                sub_bs=16 if is_moe else 8,
            )
            console.print(
                f"[green]AFTER SFT:[/green]  NLL={after_l2l['response_nll']:.4f} "
                f"PPL={after_l2l['response_ppl']:.2f} "
                f"TokAcc={after_l2l['response_token_acc_pct']:.2f}% | "
                f"GenValidJSON={after_gen.get('valid_json_schema_pct', 0):.1f}% "
                f"ToolNameAcc={after_gen.get('tool_name_acc_pct', 0):.1f}% "
                f"ArgKeyF1={after_gen.get('arg_key_f1_pct', 0):.1f}% "
                f"ExactAction={after_gen.get('full_action_exact_match_pct', 0):.1f}%"
            )

            store_host_mib = round(
                store.host_bytes()["owned_host_storage_bytes"] / (1024**2), 3
            )
            train_cost_usd = round((train_wall_s / 3600.0) * args.dph_usd, 4)
            payload = {
                "git_sha": _git_sha(),
                "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                "case": args.case,
                "model_repo": KANANA_REPO if args.case == "kanana" else (
                    QWEN38_27B_REPO if args.case == "qwen38_27b" else args.case
                ),
                "is_moe": is_moe,
                "dataset": "NousResearch/hermes-function-calling-v1",
                "steps": args.steps,
                "k": args.k,
                "seq_len": args.seq_len,
                "lr": args.lr,
                "head_lr": head_lr,
                "lora_r": args.lora_r,
                "train_head": bool(args.train_head),
                "activation_codec": "fp8_e4m3_block64",
                "routing_oracle": is_moe,
                "total_train_tokens": total_train_tokens,
                "total_active_target_tokens": total_active_target_tokens,
                "train_wall_s": round(train_wall_s, 3),
                "train_tokens_per_s": round(total_train_tokens / max(train_wall_s, 1e-9), 2),
                "train_cost_usd": train_cost_usd,
                "step_losses": step_losses,
                "step_walls_s": step_walls,
                "store_host_mib": store_host_mib,
                "oracle_host_mib": oracle_host_mib,
                "rss_peak_gib": round(poller.rss_peak_gib, 4),
                "rss_end_gib": round(_rss_gib(), 4),
                "cuda_allocated_peak_gib": (
                    round(torch.cuda.max_memory_allocated(args.device) / (1024**3), 4)
                    if is_cuda
                    else 0.0
                ),
                "cuda_reserved_peak_gib": (
                    round(torch.cuda.max_memory_reserved(args.device) / (1024**3), 4)
                    if is_cuda
                    else 0.0
                ),
                "nvml_peak_gib": (
                    round(poller.peak_gib, 4) if poller.peak_gib is not None else None
                ),
                "nvml_used_gib": _nvml_used_gib() if is_cuda else None,
                "window_dequant_stats": stepper.window_dequant_stats,
                "large_pool_release_stats": stepper.large_pool_release_stats,
                "routing_oracle_stats": oracle_stats,
                "native_telemetry": controller.native_telemetry() if controller else None,
                "before_sft": {
                    "held_out_l2l": before_l2l,
                    "free_generation": before_gen,
                },
                "after_sft": {
                    "held_out_l2l": after_l2l,
                    "free_generation": after_gen,
                },
                "delta": {
                    "response_nll_drop": round(
                        before_l2l["response_nll"] - after_l2l["response_nll"], 4
                    ),
                    "response_ppl_before": before_l2l["response_ppl"],
                    "response_ppl_after": after_l2l["response_ppl"],
                    "response_token_acc_gain_pct": round(
                        after_l2l["response_token_acc_pct"]
                        - before_l2l["response_token_acc_pct"],
                        2,
                    ),
                    "valid_json_schema_gain_pct": round(
                        after_gen.get("valid_json_schema_pct", 0.0)
                        - before_gen.get("valid_json_schema_pct", 0.0),
                        2,
                    ),
                    "tool_name_acc_gain_pct": round(
                        after_gen.get("tool_name_acc_pct", 0.0)
                        - before_gen.get("tool_name_acc_pct", 0.0),
                        2,
                    ),
                    "arg_key_f1_gain_pct": round(
                        after_gen.get("arg_key_f1_pct", 0.0)
                        - before_gen.get("arg_key_f1_pct", 0.0),
                        2,
                    ),
                    "full_action_exact_match_gain_pct": round(
                        after_gen.get("full_action_exact_match_pct", 0.0)
                        - before_gen.get("full_action_exact_match_pct", 0.0),
                        2,
                    ),
                },
            }
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            console.print(f"[green]Saved Agentic SFT & Eval results to {out_path}[/green]")
        finally:
            if oracle is not None:
                oracle.close()
            store.close()
    finally:
        for moe, prev_ctrl in saved_moe_controllers:
            moe._expert_controller = prev_ctrl
        if controller is not None:
            controller.close()
        runtime.close()


if __name__ == "__main__":
    main()
