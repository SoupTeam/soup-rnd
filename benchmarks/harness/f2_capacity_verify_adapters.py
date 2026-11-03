"""Verify each first F1-valid adapter against its immutable originating train audit."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from rich.console import Console
from safetensors import safe_open


def tensor_bytes(value):
    import torch

    return value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapters", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cell", action="append", help="Explicit subset for an in-progress transfer"
    )
    args = parser.parse_args()
    results = []
    frozen = json.loads((args.campaign / "frozen.json").read_text())
    remote_root = next(
        name.rsplit("/", 1)[0] + "/" for name in frozen["inputs"] if name.endswith("/schedule.json")
    )
    pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.self_attn\."
        r"(q_proj|v_proj)\.lora_([AB])\.weight"
    )
    for path in sorted(args.adapters.glob("*/adapter_model.safetensors")):
        cell = path.parent.name
        if args.cell and cell not in args.cell:
            continue
        origin = json.loads((path.parent / "origin.json").read_text())
        assert origin["run"].startswith(remote_root)
        audit_path = args.campaign / origin["run"][len(remote_root) :] / "audit.json"
        raw = json.loads((audit_path.parent / "result.json").read_text())
        assert raw["status"] == "PASS" and raw["kind"] == "train"
        assert raw["validity"]["outcome"] == "ok"
        audit = json.loads(audit_path.read_text())
        assert origin["fingerprint_last"] == audit["fingerprint_last"]
        for name, expected in origin["files"].items():
            assert hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == expected
        with safe_open(path, framework="pt", device="cpu") as stream:
            tensors = {key: stream.get_tensor(key) for key in stream.keys()}
        matched = {key: pattern.fullmatch(key) for key in tensors}
        assert len(tensors) == 192 and all(matched.values()), f"Unexpected tensor set: {path}"

        def model_order(key):
            layer, projection, side = matched[key].groups()
            return int(layer), ("q_proj", "v_proj").index(projection), ("A", "B").index(side)

        digest = hashlib.sha256()
        total = 0
        b_nonzero = 0
        for key in sorted(tensors, key=model_order):
            value = tensors[key]
            assert value.isfinite().all().item(), f"Nonfinite weights: {path} {key}"
            total += value.numel()
            b_nonzero += int(".lora_B." in key and value.count_nonzero().item() > 0)
            # PEFT removes the adapter name from saved keys. Restore its default
            # name, preserving the actual Qwen module registration order.
            native_name = key.replace(".lora_A.weight", ".lora_A.default.weight")
            native_name = native_name.replace(".lora_B.weight", ".lora_B.default.weight")
            digest.update(native_name.encode("utf-8"))
            digest.update(tensor_bytes(value))
        assert total == 12_582_912 and b_nonzero == 96
        saved_fingerprint = digest.hexdigest()
        assert saved_fingerprint == audit["fingerprint_last"], (
            cell,
            saved_fingerprint,
            audit["fingerprint_last"],
        )
        results.append(
            {
                "cell": cell,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tensor_count": len(tensors),
                "parameter_count": total,
                "nonzero_lora_B_tensors": b_nonzero,
                "all_finite": True,
                "originating_train_audit": str(audit_path),
                "saved_matches_runtime_fingerprint": True,
                "fingerprint": saved_fingerprint,
            }
        )
    assert len(results) == (len(args.cell) if args.cell else 6), "Expected final adapters missing"
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    Console().print({"verified_saved_adapters": len(results), "output": str(args.output)})


if __name__ == "__main__":
    main()
