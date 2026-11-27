"""ABBA plan of arm G with and without the expert host cache, at one window (harness only).

A is arm G with the host cache off; B is arm G with a 5 GiB host cache. Each block runs
A-B-B-A. The plan and the child argv are pure; the GPU runs go through ``mbr_measure.py run``.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

ARM = "G"
K = 8
SEQ = 64
CACHE_BYTES = 5 * 1024**3
ORDER: Tuple[int, ...] = (0, CACHE_BYTES, CACHE_BYTES, 0)
DEFAULT_LORA_TARGETS: Tuple[str, ...] = ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")


def abba_cache_plan(*, blocks: int = 5, start_block: int = 0, k: int = K,
                     seq: int = SEQ) -> List[Dict[str, Any]]:
    """A-B-B-A blocks of arm G at the window (k, seq), default k=8, seq=64. Block numbers and run
    indexes continue from ``start_block``, like ``mbr_measure.abba_plan``."""
    per_block = len(ORDER)
    plan: List[Dict[str, Any]] = []
    for block in range(start_block, start_block + blocks):
        for position, host_cache_bytes in enumerate(ORDER):
            plan.append({"arm": ARM, "k": k, "seq": seq, "block": block,
                         "index": block * per_block + position,
                         "host_cache_bytes": host_cache_bytes, "kind": "speed"})
    return plan


def abba_cache_argv(item: Dict[str, Any], *, source: str, cache: str, out: str,
                    lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS, steps: int = 4,
                    lora_r: int = 16, device: str = "auto",
                    tokenizer: Optional[str] = None) -> List[str]:
    """argv of one ``mbr_measure.py run`` child for a plan item. The flags mirror
    ``child_run_args``, which reads the cache budget from the parent namespace, not the item."""
    argv = ["run", "--arm", item["arm"], "--k", str(item["k"]), "--seq", str(item["seq"]),
            "--kind", item["kind"], "--source", source, "--cache", cache,
            "--steps", str(steps), "--out", out, "--device", device,
            "--lora-r", str(lora_r), "--lora-targets", ",".join(lora_targets)]
    if tokenizer:
        argv += ["--tokenizer", tokenizer]
    if item["host_cache_bytes"]:
        argv += ["--host-cache-bytes", str(item["host_cache_bytes"])]
    return argv


def main(argv: Any = None) -> int:  # pragma: no cover - thin CLI
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="print the plan only; GPU runs go through mbr_measure.py")
    args = parser.parse_args(argv)
    console = Console()
    for item in abba_cache_plan(blocks=args.blocks, start_block=args.start_block):
        console.print(f"index={item['index']} block={item['block']} arm={item['arm']} "
                      f"k={item['k']} seq={item['seq']} "
                      f"host_cache_bytes={item['host_cache_bytes']}", markup=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
