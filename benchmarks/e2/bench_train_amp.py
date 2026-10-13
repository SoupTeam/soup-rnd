#!/usr/bin/env python3
"""Run Soup benchmark with an explicit, verified AMP GradScaler initial scale.

Exploratory launcher for CUDA FP16. Not a Soup CLI public interface.
"""

from __future__ import annotations

import argparse
import math
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp-init-scale", type=float, required=True)
    args, soup_args = parser.parse_known_args()
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
        parser.error("--amp-init-scale must be finite and >0")
    if soup_args[:2] != ["bench", "train"]:
        parser.error("Only 'bench train' is supported")

    # Import this same object used by Accelerator.__init__ (single-process FP16).
    from accelerate import accelerator as accelerator_module

    original = accelerator_module.get_grad_scaler
    created = []

    def configured_grad_scaler(*positional, **kwargs):
        if created:
            raise RuntimeError("More than one GradScaler created during benchmark")
        kwargs["init_scale"] = args.amp_init_scale
        scaler = original(*positional, **kwargs)
        actual = float(scaler.get_scale())
        if actual != args.amp_init_scale:
            raise RuntimeError(
                f"GradScaler initial scale mismatch: {actual} != {args.amp_init_scale}"
            )
        created.append(scaler)
        print(f"E2_AMP_SCALER_CREATED init_scale={actual}", flush=True)
        return scaler

    accelerator_module.get_grad_scaler = configured_grad_scaler
    sys.argv = ["soup_cli", *soup_args]
    try:
        runpy.run_module("soup_cli", run_name="__main__", alter_sys=True)
    finally:
        print(f"E2_AMP_SCALERS_CREATED: {len(created)}", flush=True)
        for scaler in created:
            print(f"E2_AMP_FINAL_SCALE: {float(scaler.get_scale())}", flush=True)
        accelerator_module.get_grad_scaler = original
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one AMP GradScaler, found {len(created)}")


if __name__ == "__main__":
    main()
