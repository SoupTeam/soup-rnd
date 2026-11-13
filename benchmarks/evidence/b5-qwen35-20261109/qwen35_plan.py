# ruff: noqa: E501, E402, E731  (evidence script: sys.path set before imports, report prose lines not wrapped)
"""Header-only cache plan for Qwen3.5-35B-A3B at a pinned revision. No weight payload is read.

    python -I qwen35_plan.py CONFIG_DIR OUT.json

Builds the meta skeleton the sharder builds (NF4, single quantisation), asks it which per-layer
tensors become NF4, reads the 14 safetensors headers over HTTP Range, and sums bytes per class.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])  # repository root
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT)
MODEL = "Qwen/Qwen3.5-35B-A3B"
REV = "59d61f3ce65a6d9863b86d2e96597125219dc754"
cfgdir, out = sys.argv[1], sys.argv[2]

import torch
from huggingface_hub import HfApi, hf_hub_url

from benchmarks.harness.b6_source_plan import read_remote_header
from soup_cli.utils.layer_stream import estimate_stream_store_bytes, stream_arch_of
from soup_cli.utils.layer_stream_runtime import (
    build_meta_skeleton,
    quantised_layer_suffixes,
    shard_layout_of,
)

info = HfApi(token=False).model_info(MODEL, revision=REV, files_metadata=True)
assert info.sha == REV, info.sha
files = sorted((r for r in info.siblings if r.rfilename.endswith(".safetensors")),
               key=lambda r: r.rfilename)
source_bytes = sum(r.size for r in files)
print("shards", len(files), "source bytes", source_bytes)

config = json.load(open(cfgdir + "/config.json"))
print("model_type", config["model_type"], "text", config["text_config"]["model_type"])
from soup_cli.utils.kimi_k2 import load_stream_config

loaded = load_stream_config(cfgdir, False)
print("loaded", type(loaded).__name__, "stream_arch_of", stream_arch_of(loaded))

previous = torch.utils.deterministic.fill_uninitialized_memory
torch.utils.deterministic.fill_uninitialized_memory = False
try:
    skeleton = build_meta_skeleton(cfgdir, dtype="bfloat16", quant="nf4", double_quant=False)
finally:
    torch.utils.deterministic.fill_uninitialized_memory = previous
suffixes = sorted(quantised_layer_suffixes(skeleton))
layout = shard_layout_of(skeleton)
print("skeleton class", type(skeleton).__name__, "n_layers", layout.n_layers,
      "stack_experts", layout.stack_experts)
print("nf4 suffixes", suffixes)
skel_params = {}
for name, param in skeleton.named_parameters():
    skel_params[name] = (tuple(param.shape), str(param.dtype), type(param).__name__)
experts_in_skeleton = [n for n in skel_params if "experts" in n][:4]
print("skeleton expert params", [(n, skel_params[n]) for n in experts_in_skeleton])
del skeleton

rows = []
for r in files:
    raw, header = read_remote_header(hf_hub_url(MODEL, r.rfilename, revision=REV), r.size)
    rows.append((r.size, len(raw), header, r.rfilename, hashlib.sha256(raw).hexdigest()))

LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.(.*)")
nf4 = lambda n: (n + 1) // 2 + 4 * ((n + 63) // 64)  # 0.5 B/elem + fp32 absmax per 64
classes = {"routed_experts": 0, "nf4_dense": 0, "bf16_other_layer": 0, "embed_head_norm": 0,
           "dropped_visual": 0, "dropped_mtp": 0}
elements = dict.fromkeys(classes, 0)
per_layer_bf16 = {}
dtypes = set()
for _, _, header, _, _ in rows:
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        count = math.prod(spec["shape"])
        dtypes.add(spec["dtype"])
        match = LAYER.fullmatch(key)
        if key.startswith("model.visual."):
            classes["dropped_visual"] += count * 2
            elements["dropped_visual"] += count
        elif key.startswith("mtp."):
            classes["dropped_mtp"] += count * 2
            elements["dropped_mtp"] += count
        elif match:
            short = match.group(2)
            if short.startswith("mlp.experts."):
                classes["routed_experts"] += count * 2
                elements["routed_experts"] += count
            elif short in suffixes:
                classes["nf4_dense"] += nf4(count)
                elements["nf4_dense"] += count
            else:
                classes["bf16_other_layer"] += count * 2
                elements["bf16_other_layer"] += count
            per_layer_bf16.setdefault(int(match.group(1)), 0)
            if short.startswith("mlp.experts."):
                per_layer_bf16[int(match.group(1))] += count * 2
        else:
            classes["embed_head_norm"] += count * 2
            elements["embed_head_norm"] += count

# NF4 store bytes: routed experts stay at the base dtype (bf16) unless they are Params4bit.
experts_nf4 = "mlp.experts.gate_up_proj" in suffixes
store = (nf4(elements["routed_experts"]) if experts_nf4 else classes["routed_experts"]) \
    + classes["nf4_dense"] + classes["bf16_other_layer"] + classes["embed_head_norm"]
result = {
    "model": MODEL, "revision": REV, "n_shards": len(files), "source_bytes": source_bytes,
    "source_dtypes": sorted(dtypes), "model_type": config["model_type"],
    "text_model_type": config["text_config"]["model_type"],
    "stream_arch": stream_arch_of(loaded),"n_layers": layout.n_layers,
    "stack_experts": layout.stack_experts, "nf4_suffixes": suffixes,
    "routed_experts_are_nf4": experts_nf4,
    "bytes_by_class_store": classes, "elements_by_class": elements,
    "arithmetic_store_bytes": store,
    "arithmetic_store_bytes_if_experts_were_nf4": (
        nf4(elements["routed_experts"]) + classes["nf4_dense"] + classes["bf16_other_layer"]
        + classes["embed_head_norm"]),
    "soup_coarse_estimate_bytes": estimate_stream_store_bytes(
        source_bytes, dtype="bfloat16", quant="nf4", double_quant=False),
    "soup_coarse_estimate_double_quant_bytes": estimate_stream_store_bytes(
        source_bytes, dtype="bfloat16", quant="nf4", double_quant=True),
    "largest_layer_expert_bytes_bf16": max(per_layer_bf16.values()),
    "headers": [{"name": n, "size": s, "header_sha256": d} for s, _, _, n, d in rows],
}
json.dump(result, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in result.items() if k != "headers"}, indent=1))
