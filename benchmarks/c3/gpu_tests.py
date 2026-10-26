"""Run only the C3 CUDA controls in an owned local context and preserve its identity."""

import json
from pathlib import Path


def main() -> int:
    from benchmarks.harness.l2l_box import confirm_gpu_owner, gpu_compute_pids

    before = gpu_compute_pids()
    if before != []:
        raise RuntimeError(f"GPU not observably idle: {before!r}")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires exactly one CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe = torch.empty(1, device="cuda:0")
    torch.cuda.synchronize()
    owner = confirm_gpu_owner(before, context_active=True, single_device=True)
    del probe
    Path("/data/c3/evidence/test_gpu_owner.json").write_text(json.dumps(owner, indent=2))
    if owner["method"] != "exclusive-local-CUDA-initialization":
        raise RuntimeError("GPU ownership unresolved")
    import pytest

    return pytest.main(
        [
            "-o",
            "addopts=",
            "-m",
            "gpu",
            "-q",
            "--junitxml=/data/c3/evidence/cuda_tests.xml",
            "tests/test_c3_partial_ram.py",
            "tests/test_c3_host_budget.py",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
