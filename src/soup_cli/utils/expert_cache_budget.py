"""Conservative cache budget for a new, single-disk per-expert NF4 store.

Reads only bounded JSON headers, including FP8, without memory-mapping weights.
The budget includes an additional largest layer for atomic writes and reserves
space for headers, the index, code tables and a possible tied vocabulary copy.
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Sequence

_SIZES = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4}
_EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$")
_LAYER = re.compile(r"model\.layers\.(\d+)\.")


def new_expert_cache_budget(source_files: Sequence[str]) -> int:
    """Budget bytes; refuse unsupported inputs rather than extrapolating a ratio.

    Valid for NF4 with blocksize 64 and no nested statistics. The writer checks
    codec sidecars against their declared shape and dtype before any write.
    Existing cache replacement and striping require a separate placement plan.
    """
    if not source_files:
        raise ValueError("expert cache budget requires complete source files")
    seen = set()
    layers = defaultdict(int)
    total = count = largest = expert_count = 0
    for path in source_files:
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError("truncated safetensors header")
            length = int.from_bytes(raw, "little")
            if not 2 <= length <= 64 * 2**20:
                raise ValueError("safetensors header exceeds budget reader limit")
            data = handle.read(length)
            if len(data) != length:
                raise ValueError("truncated safetensors header")
            header = json.loads(data)
            file_bytes = os.fstat(handle.fileno()).st_size
        if not isinstance(header, dict):
            raise ValueError("invalid safetensors header")
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in seen or len(key.encode("utf-8")) > 1024:
                raise ValueError("duplicate or oversized source tensor key")
            seen.add(key)
            shape, dtype = value["shape"], value["dtype"]
            if (dtype not in _SIZES or not isinstance(shape, list) or len(shape) > 8
                    or any(type(dim) is not int or dim < 0 for dim in shape)):
                raise ValueError("unsupported tensor shape or dtype for expert cache budget")
            elements = math.prod(shape)
            offsets = value["data_offsets"]
            if (len(offsets) != 2 or any(type(pos) is not int for pos in offsets)
                    or offsets[0] < 0 or offsets[1] - offsets[0] != elements * _SIZES[dtype]
                    or offsets[1] > file_bytes - length - 8):
                raise ValueError("source tensor offsets do not fit the file")
            count += 1
            if key.endswith("weight_scale_inv"):
                continue
            if _EXPERT.fullmatch(key):
                # Gate/up may be concatenated. Extra blocks bound that rounding,
                # and every source matrix gets its own metadata allowance below.
                size = (elements + 1) // 2 + 8 * ((elements + 63) // 64 + 1)
                expert_count += 1
            else:
                size = 4 * elements
                largest = max(largest, size)
            total += size
            layer = _LAYER.match(key)
            if layer:
                layers[layer[1]] += size
    if not expert_count:
        raise ValueError("source contains no supported routed experts")
    # Four stored entries per source tensor, each with generous JSON/alignment
    # allowance, plus index/temp reserve and a possible duplicated tied head.
    metadata = 4 * count * 16384
    return total + max(layers.values(), default=0) + largest + metadata + 2**30
