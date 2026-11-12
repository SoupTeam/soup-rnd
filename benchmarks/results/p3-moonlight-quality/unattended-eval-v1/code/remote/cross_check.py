"""Cross-host admissibility check (CPU only; no model, no scoring).

Exit 0: the old-host BF16 shard pairs with the new-host Oracle shard.
Exit 3: the new-host shard is admissible alone but cannot pair with the old-host shard.
Exit 1: the new-host shard itself is refused by the pairing collector.
"""
import json
import sys
from pathlib import Path

from benchmarks.harness import p3_deepseek_quality_eval as ev

PROFILE = "moonlight-16b-a3b"
old, new, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
record = {"old": old.name, "new": new.name}


def finish(code: int, **values) -> None:
    record.update(values)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))
    sys.exit(code)


try:
    ev._collect([new], ev.ARMS[1], profile_id=PROFILE)
except Exception as error:  # noqa: BLE001 - classify, never hide
    finish(1, status="NEW_SHARD_REFUSED", error=f"{type(error).__name__}: {error}")
try:
    primary, secondary = ev.pair_shards([old], [new], profile_id=PROFILE)
except Exception as error:  # noqa: BLE001
    finish(3, status="PAIR_REFUSED", error=f"{type(error).__name__}: {error}")
finish(0, status="PASS", primary_rows=len(primary), secondary_rows=len(secondary))
