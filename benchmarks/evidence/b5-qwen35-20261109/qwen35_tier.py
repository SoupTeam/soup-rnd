# ruff: noqa: E501, E402, F401  (evidence script: sys.path set before imports, kept as run)
"""What Soup's own tier planner says about Qwen3.5-35B-A3B at a 32 GiB limit. CPU only, no weights.

    python -I qwen35_tier.py qwen35_plan.json

Feeds the header-only sizes from ``qwen35_plan.json`` to the functions ``soup train`` calls:
``estimate_stream_store_bytes`` (early check in ``stream_setup.py``) and ``build_stream_plan``
(after sharding). The 32 GiB figure is what ``total_ram_bytes()`` returns under a 32 GiB cgroup.
"""
import json
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])  # repository root
sys.path.insert(0, ROOT + "/src")

from soup_cli.utils.layer_stream import (
    PHYSICAL_RAM_TIER_HEADROOM,
    RAM_TIER_HEADROOM,
    build_stream_plan,
    estimate_stream_store_bytes,
)

plan = json.load(open(sys.argv[1]))
GIB = 2**30
limit = 32 * GIB
n_layers = plan["n_layers"]
embed = plan["bytes_by_class_store"]["embed_head_norm"]
classes = plan["bytes_by_class_store"]
layers_bf16_experts = (plan["arithmetic_store_bytes"] - embed)
layers_nf4_experts = (plan["arithmetic_store_bytes_if_experts_were_nf4"] - embed)
print(f"limit {limit} B ({limit / 1e9:.2f} GB); headroom free {RAM_TIER_HEADROOM}, "
      f"physical {PHYSICAL_RAM_TIER_HEADROOM}; physical ceiling "
      f"{limit * PHYSICAL_RAM_TIER_HEADROOM / 1e9:.2f} GB")
print(f"resident extras (embedding, lm_head, norms): {embed / 1e9:.2f} GB")

early = plan["soup_coarse_estimate_bytes"]
print(f"early check: coarse NF4 estimate {early / 1e9:.2f} GB vs free*0.7 "
      f"{limit * RAM_TIER_HEADROOM / 1e9:.2f} GB -> "
      f"{'passes' if early < limit * RAM_TIER_HEADROOM else 'refuses'}")

for label, layers in (("as built today (routed experts bf16)", layers_bf16_experts),
                      ("hypothetical, routed experts NF4", layers_nf4_experts)):
    for kind in ("nvme", "hdd"):
        try:
            result = build_stream_plan(
                arch="qwen3", n_layers=n_layers, layer_bytes=layers // n_layers,
                embed_bytes=embed, available_ram_bytes=int(limit * 0.98),
                total_ram_bytes=limit, disk_kind=kind, store_bytes=layers,
                pinned_limit_bytes=None)  # None, as stream_setup.py passes it
            verdict = f"tier={result.tier}"
        except ValueError as error:
            verdict = f"ValueError: {str(error)[:160]}"
        print(f"{label}: store {layers / 1e9:.2f} GB + extras, disk {kind}: {verdict}")
    need = (layers + embed) / PHYSICAL_RAM_TIER_HEADROOM
    print(f"{label}: physical RAM needed for the RAM tier >= {need / 1e9:.1f} GB "
          f"({need / GIB:.1f} GiB)")
