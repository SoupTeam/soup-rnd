# H3 measured pilot

| Arm | GPUs | Input tokens/s | Step seconds |
|---|---:|---:|---:|
| repeat-1-gpu-1 | 1 | 6865.86 | 0.5966 |
| repeat-1-gpu-2 | 2 | 7702.35 | 0.5318 |
| repeat-2-gpu-2 | 2 | 7467.19 | 0.5485 |
| repeat-2-gpu-1 | 1 | 6998.48 | 0.5853 |

Speedup(2): 1.0941; efficiency(2): 0.5471.

4/8 GPUs, 70B, optimizer/scheduler resume and quality: **unmeasured**.
Per-rank VRAM/RAM, first-step deltas and raw timings: summary.json and rank-*.json.
