"""Gate a measurement run on AC power and healthy clocks (ticket 13, phase 2).

The card lost its 15 W limit after a power failure until a reboot, so a run is valid only when
the AC adapter is online, the enforced power limit is the normal TGP (not the collapsed 15 W)
and the SM clock under load stays at or above 1500 MHz. A ``SW Power Cap`` at a healthy clock
is the 80 W TGP limit doing its job; it is recorded as a warning, and the gate fails only when
the clock falls or the enforced limit collapses. Exit 0 when the run may start, 1 otherwise.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

AC_ONLINE = Path("/sys/class/power_supply/ACAD/online")
MIN_SM_MHZ = 1500
MIN_POWER_LIMIT_W = 50.0
SW_POWER_CAP = 0x4
LOAD_SECONDS = 10.0
WARMUP_SECONDS = 3.0


def query(fields: str) -> list[str]:
    done = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return [part.strip() for part in done.stdout.strip().split(",")]


def assess(ac: str, samples: list[int], limit: float) -> list[str]:
    """Problems that forbid a measurement run; an empty list means the card is healthy."""
    problems: list[str] = []
    if ac != "1":
        problems.append(f"AC not online: {AC_ONLINE} == {ac!r}")
    if not samples:
        problems.append("no clock samples collected")
    else:
        low = min(samples)
        if low < MIN_SM_MHZ:
            problems.append(f"SM clock {low} MHz < {MIN_SM_MHZ} MHz under load")
    if limit < MIN_POWER_LIMIT_W:
        problems.append(f"enforced power limit {limit:.1f} W < {MIN_POWER_LIMIT_W:.0f} W")
    return problems


def main() -> int:
    ac = AC_ONLINE.read_text(encoding="ascii").strip()

    import torch

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    samples: list[int] = []
    caps = 0
    started = time.monotonic()
    while time.monotonic() - started < LOAD_SECONDS:
        a @ b
        torch.cuda.synchronize()
        if time.monotonic() - started >= WARMUP_SECONDS:
            samples.append(int(query("clocks.sm")[0]))
            if int(query("clocks_throttle_reasons.active")[0], 16) & SW_POWER_CAP:
                caps += 1
    limit = float(query("enforced.power.limit")[0])
    problems = assess(ac, samples, limit)
    if samples:
        print(
            f"gate: ac={ac} sm_min={min(samples)} sm_max={max(samples)} caps={caps}/{len(samples)} "
            f"power_limit_w={limit:.1f} load_s={LOAD_SECONDS}",
            flush=True,
        )
        if caps:
            print(f"gate: warning SW Power Cap in {caps} sample(s), clocks stayed healthy",
                  file=sys.stderr, flush=True)
    if problems:
        for problem in problems:
            print(f"gate: FAIL {problem}", file=sys.stderr, flush=True)
        return 1
    print("gate: OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
