"""Post-hoc A4 G2 check, run once on a local RTX 4050 after the v1 session (result:
``posthoc-probe-result.json``): does a host read of a CUDA
float32 scalar add a 4 B block to torch's caching host allocator, as counted by
``host_memory_stats()['allocated_bytes.current' / '.peak']`` the way ``a4_probe.mode_arm``
counts it (release cache -> baseline current -> ... -> reset peak -> peak)?"""
import json
import platform
import sys

import torch


def cur() -> int:
    return int(torch.cuda.host_memory_stats()["allocated_bytes.current"])


def peak() -> int:
    return int(torch.cuda.host_memory_stats()["allocated_bytes.peak"])


def release() -> int:
    before = cur()
    torch._C._host_emptyCache()  # what release_cached_pinned_memory() calls
    return before - cur()


def case(name, body):
    torch.cuda.synchronize()
    release()
    base = cur()
    out = body()
    torch.cuda.synchronize()
    after = cur()
    row = {"case": name, "baseline": base, "after": after, "delta": after - base}
    if out:
        row.update(out)
    return row


rows = []
dev = "cuda"
# 1. the minimal test
rows.append(case("float(fp32 scalar)",
                 lambda: {"value": float(torch.ones((), device=dev, dtype=torch.float32))}))
# 2. control: same tensor, never read back to the host
def control():
    t = torch.ones((), device=dev, dtype=torch.float32)
    t.mul_(2)  # device work only, never read back to the host


rows.append(case("control: no host read", control))
# 3. block size follows sizeof(dtype): bf16 -> 2, fp64 -> 8
rows.append(case("float(bf16 scalar)",
                 lambda: {"value": float(torch.ones((), device=dev, dtype=torch.bfloat16))}))
rows.append(case("float(fp64 scalar)",
                 lambda: {"value": float(torch.ones((), device=dev, dtype=torch.float64))}))
# 4. reuse: k = 2, 3, 4 losses per step x 5 steps -> one cached block, still +4
for k in (2, 3, 4):
    def many(k=k):
        losses = [torch.rand((), device=dev, dtype=torch.float32) for _ in range(k)]
        for _ in range(5):
            [float(x) for x in losses]
    rows.append(case(f"k={k} x 5 steps of float(loss)", many))
# 5. the arm's exact order: baseline, warm-up WITHOUT a read, reset peak, measured read
def arm_order(read_in_warmup):
    def body():
        x = torch.rand((), device=dev, dtype=torch.float32)
        if read_in_warmup:
            float(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_host_memory_stats()
        at_reset = peak()
        float(x)
        return {"peak_at_reset": at_reset, "peak_after_measured_read": peak()}
    return body
for flag in (False, True):
    r = case(f"arm order, read in warm-up={flag}", arm_order(flag))
    r["peak_minus_baseline"] = r["peak_after_measured_read"] - r["baseline"]
    rows.append(r)

print(json.dumps({
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
    "platform": platform.platform(), "rows": rows}, indent=1))
