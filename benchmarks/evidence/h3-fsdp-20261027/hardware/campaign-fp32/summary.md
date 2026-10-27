# H3 measured pilot

| Arm | GPUs | Input tokens/s | Step seconds |
|---|---:|---:|---:|
| repeat-1-gpu-1 | 1 | 4276.82 | 0.9577 |
| repeat-1-gpu-2 | 2 | 2070.65 | 1.9781 |
| repeat-2-gpu-2 | 2 | 2079.69 | 1.9695 |
| repeat-2-gpu-1 | 1 | 4156.11 | 0.9855 |

Speedup(2): 0.4922; efficiency(2): 0.2461.

4/8 GPUs, 70B, optimizer/scheduler resume and quality: **unmeasured**.
Per-rank VRAM/RAM, first-step deltas and raw timings: summary.json and rank-*.json.
