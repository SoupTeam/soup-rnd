"""Markdown tables of the B1.4 grouped-compute diagnostic, from a result JSON.

Usage: python make_tables.py vast-rtx3070/result.json
"""

import json
import sys

ORDER = ("sequential", "grouped_restore", "grouped_gemm")


def main(path: str) -> None:
    data = json.load(open(path))
    print("| Rows per expert | Arm | Kernels | Forward kernels | Backward kernels | Copies "
          "| Forward device us | Backward device us | Wall ms (median) "
          "| Bitwise equal to resident |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for bucket in data["buckets"]:
        for arm in ORDER:
            entry = bucket["arms"].get(arm)
            if entry is None:
                continue
            if entry["status"] != "ok":
                print(f"| {bucket['rows']} | {arm} | {entry['status']} | | | | | | | |")
                continue
            kernels, phases = entry["kernels"], entry["kernels"]["by_phase"]
            device = kernels["gpu_time_us_by_phase"]
            forward = phases.get("fwd", 0) + phases.get("gemm", 0) + phases.get("act", 0)
            forward_us = device.get("fwd", 0) + device.get("gemm", 0) + device.get("act", 0)
            equal = entry["bitwise_equal_to_resident"]
            verdict = "yes" if all(equal.values()) else "NO " + str(
                [name for name, same in equal.items() if not same])
            print(f"| {bucket['rows']} | {arm} | {kernels['total']} | {forward} "
                  f"| {phases.get('bwd', 0)} | {kernels['memcpy_events']} | {forward_us} "
                  f"| {device.get('bwd', 0)} | {entry['timing_ms']['total_median']:.2f} "
                  f"| {verdict} |")


if __name__ == "__main__":
    main(sys.argv[1])
