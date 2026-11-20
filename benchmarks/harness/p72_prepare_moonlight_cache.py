"""Build the pinned Moonlight per-expert NF4 cache the P3 bridge expects (PR #72 diagnostics).

Downloads the pinned revision, shards it exactly as ``b5_real_moe.py`` does (bf16, NF4, no
double quant, per_expert) and writes ``ROOT/results/cache.json`` with ``status: PASS``.
The cache is rebuilt on the rental host, not the one of the 2026-11-17 runs, so absolute
times may differ from those reports.

    python p72_prepare_moonlight_cache.py --root /workspace/p3-baseline
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from b5_real_moe import _meta_construction
from p3_route_preserving_mbr_soup_bridge import REVISION

MODEL = "moonshotai/Moonlight-16B-A3B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/p3-baseline")
    args = parser.parse_args()
    root = Path(args.root)
    source, cache, results = root / "source", root / "cache", root / "results"
    results.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    started = time.time()
    snapshot_download(
        MODEL, revision=REVISION, local_dir=str(source), max_workers=8, token=False,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.py", "*.txt"],
    )
    download_seconds = time.time() - started

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
        shard_layout_of,
    )

    started = time.time()
    with _meta_construction():
        skeleton = build_meta_skeleton(str(source), dtype="bfloat16", quant="nf4",
                                       double_quant=False)
    suffixes, layout = quantised_layer_suffixes(skeleton), shard_layout_of(skeleton)
    del skeleton
    gc.collect()
    index = shard_checkpoint(
        str(source), str(cache), dtype="bfloat16", arch="deepseek_v3", layout=layout,
        quant="nf4", quant_suffixes=suffixes, quant_device="cuda", double_quant=False,
        expert_layout="per_expert",
    )
    (results / "cache.json").write_text(
        json.dumps(
            {
                "status": "PASS", "revision": REVISION, "model": MODEL,
                "download_seconds": download_seconds,
                "shard_seconds": time.time() - started,
                "total_params": int(index.total_params),
                "note": "rebuilt for PR #72 diagnostics with the b5_real_moe shard call",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("cache ready", flush=True)


if __name__ == "__main__":
    main()
