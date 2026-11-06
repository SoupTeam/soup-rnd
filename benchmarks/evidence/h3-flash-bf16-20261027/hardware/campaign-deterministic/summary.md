# H3 measured pilot

| Arm | GPUs | Input tokens/s | Step seconds |
|---|---:|---:|---:|
| repeat-1-gpu-1 | 1 | 8235.82 | 0.4973 |
| repeat-1-gpu-2 | 2 | 8139.69 | 0.5032 |
| repeat-2-gpu-2 | 2 | 8065.24 | 0.5079 |
| repeat-2-gpu-1 | 1 | 8120.44 | 0.5044 |

Speedup(2): 0.9907; efficiency(2): 0.4954.

4/8 GPUs, 70B, optimizer/scheduler resume and quality: **unmeasured**.
Per-rank VRAM/RAM, first-step deltas and raw timings: summary.json and rank-*.json.
