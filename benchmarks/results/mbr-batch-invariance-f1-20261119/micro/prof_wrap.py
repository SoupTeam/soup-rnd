"""Run the MBR bridge under torch.profiler without touching its code.

Usage: prof_wrap.py <bridge.py> <out-prefix> [bridge args...]
Writes <out-prefix>.txt (op table, sorted by CUDA time) and <out-prefix>.counts.json.
"""
import json
import runpy
import sys

from torch.profiler import ProfilerActivity, profile

bridge, out = sys.argv[1], sys.argv[2]
sys.argv = [bridge] + sys.argv[3:]
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    try:
        runpy.run_path(bridge, run_name="__main__")
    except SystemExit:
        pass
events = prof.key_averages()
with open(out + ".txt", "w", encoding="utf-8") as fh:
    fh.write(events.table(sort_by="cuda_time_total", row_limit=35, max_name_column_width=60))
counts = {e.key: {"count": e.count, "cuda_us": round(e.device_time_total, 1),
                  "cpu_us": round(e.cpu_time_total, 1)} for e in events
          if e.key in ("aten::linear", "aten::mul", "aten::cat", "aten::slice",
                       "aten::select", "aten::mm", "aten::addmm", "aten::matmul",
                       "aten::index_put_", "aten::zeros", "aten::copy_", "aten::_to_copy")}
with open(out + ".counts.json", "w", encoding="utf-8") as fh:
    json.dump(counts, fh, indent=1)
