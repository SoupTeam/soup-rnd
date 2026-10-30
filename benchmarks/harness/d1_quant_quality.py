"""Bounded BF16 versus nested-NF4 pretrained quality screen; no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from rich.console import Console

MODEL = "mistralai/Mistral-7B-v0.3"
REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
DATASET = "Salesforce/wikitext"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATA_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
BLOCKS = 32
LENGTH = 512
WEIGHT_HASHES = {
    "model-00001-of-00003.safetensors":
        "2d7de8eb9bd554cccc999649fd5a3a28a44d09042879d6c1cedd13f59c86f7a8",
    "model-00002-of-00003.safetensors":
        "79a5f871a0fbb7919f263d78caff0d4d3b7a1d444552db3011ada70f48fd5ee4",
    "model-00003-of-00003.safetensors":
        "ba7ae59358f726f5f5bc5264fd8a3aec907f42e6009d7b22454007d18723616a",
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(root: Path) -> None:
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(
        MODEL, revision=REVISION,
        allow_patterns=["*.json", "model-*.safetensors", "tokenizer.model*"], max_workers=4,
    ))
    hashes = {name: digest(snapshot / name) for name in WEIGHT_HASHES}
    if hashes != WEIGHT_HASHES:
        raise RuntimeError("original checkpoint hash mismatch")
    dataset = Path(hf_hub_download(
        DATASET, DATA_FILE, repo_type="dataset", revision=DATA_REVISION,
    ))
    rows = pq.read_table(dataset, columns=["text"])["text"].to_pylist()
    text = "\n\n".join(rows)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=False)
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    # Deterministic stratification over the whole test stream, fixed before GPU execution.
    starts = [index * (len(tokens) - LENGTH) // (BLOCKS - 1) for index in range(BLOCKS)]
    if any(right - left < LENGTH for left, right in zip(starts, starts[1:])):
        raise RuntimeError("validation windows overlap")
    inputs = np.array([tokens[start:start + LENGTH] for start in starts], dtype=np.int64)
    if inputs.shape != (BLOCKS, LENGTH):
        raise RuntimeError("unexpected token shape")
    np.save(root / "inputs.npy", inputs, allow_pickle=False)
    write(root / "prepared.json", {
        "model": MODEL, "revision": REVISION, "snapshot": str(snapshot),
        "weight_hashes": hashes, "dataset": DATASET, "dataset_revision": DATA_REVISION,
        "dataset_file": DATA_FILE, "dataset_sha256": digest(dataset),
        "text_join": "two newlines between all original rows; no BOS or EOS added",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_stream_length": len(tokens), "window_starts": starts,
        "blocks": BLOCKS, "length": LENGTH, "scored_tokens": BLOCKS * (LENGTH - 1),
        "inputs_sha256": digest(root / "inputs.npy"),
    })


def evaluate(root: Path, precision: str) -> None:
    import importlib.metadata

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    torch.manual_seed(842)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    prepared = json.loads((root / "prepared.json").read_text())
    if digest(root / "inputs.npy") != prepared["inputs_sha256"]:
        raise RuntimeError("inputs changed")
    options = dict(dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
                   trust_remote_code=False, local_files_only=True)
    if precision == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(prepared["snapshot"], **options).eval()
    model.requires_grad_(False)
    quantized = [name for name, module in model.named_modules()
                 if module.__class__.__name__ == "Linear4bit"]
    if len(quantized) != (224 if precision == "nf4" else 0):
        raise RuntimeError("unexpected quantized linear coverage")
    inputs = np.load(root / "inputs.npy", allow_pickle=False)
    losses, predictions = [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, block in enumerate(inputs):
            tensor = torch.tensor(block, device="cuda").unsqueeze(0)
            logits = model(input_ids=tensor, use_cache=False).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tensor[:, 1:].reshape(-1),
                reduction="none",
            )
            if not torch.isfinite(loss).all():
                raise RuntimeError("nonfinite NLL")
            losses.append(loss.cpu().numpy())
            predictions.append(logits.argmax(-1).cpu().numpy().reshape(-1))
            del logits, loss
            Console().print({"precision": precision, "completed_blocks": index + 1})
    torch.cuda.synchronize()
    nll = np.stack(losses)
    np.savez_compressed(root / f"{precision}-tokens.npz", nll=nll,
                        predictions=np.stack(predictions))
    average = float(nll.astype(np.float64).mean())
    write(root / f"{precision}.json", {
        "precision": precision, "nll": average, "perplexity": math.exp(average),
        "scored_tokens": int(nll.size), "block_nll": nll.mean(axis=1).tolist(),
        "quantized_modules": quantized, "compute_dtype": "bfloat16", "attention": "sdpa",
        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "python": platform.python_version(), "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "bitsandbytes", "accelerate", "numpy")},
        "inputs_sha256": prepared["inputs_sha256"],
        "raw_sha256": digest(root / f"{precision}-tokens.npz"),
        "training": False, "candidate_kernel": False, "ram_streaming": False,
    })


def compare(root: Path) -> dict:
    import numpy as np

    prepared = json.loads((root / "prepared.json").read_text())
    arms = [json.loads((root / f"{arm}.json").read_text()) for arm in ("bf16", "nf4")]
    raw = []
    for arm, metadata in zip(("bf16", "nf4"), arms):
        path = root / f"{arm}-tokens.npz"
        if digest(path) != metadata["raw_sha256"]:
            raise RuntimeError("raw evidence hash mismatch")
        if metadata["inputs_sha256"] != prepared["inputs_sha256"]:
            raise RuntimeError("input identity mismatch")
        with np.load(path, allow_pickle=False) as values:
            raw.append({name: values[name].copy() for name in values.files})
        nll = raw[-1]["nll"]
        if nll.shape != (BLOCKS, LENGTH - 1) or not np.isfinite(nll).all():
            raise RuntimeError("incomplete validation coverage")
        if float(nll.astype(np.float64).mean()) != metadata["nll"]:
            raise RuntimeError("summary NLL does not reproduce")
    if arms[0]["packages"] != arms[1]["packages"] or arms[0]["gpu"] != arms[1]["gpu"]:
        raise RuntimeError("different execution environment")
    delta = arms[1]["nll"] - arms[0]["nll"]
    block_delta = (raw[1]["nll"].astype(np.float64)
                   - raw[0]["nll"].astype(np.float64)).mean(axis=1)
    return {
        "status": "DESCRIPTIVE_PRETRAINED_QUANTIZATION_SCREEN",
        "bf16_nll": arms[0]["nll"], "nf4_nll": arms[1]["nll"],
        "bf16_perplexity": arms[0]["perplexity"],
        "nf4_perplexity": arms[1]["perplexity"], "delta_nll": delta,
        "relative_perplexity_change_percent": 100 * math.expm1(delta),
        "top1_agreement_percent": 100 * float(
            (raw[0]["predictions"] == raw[1]["predictions"]).mean()),
        "scored_tokens": prepared["scored_tokens"], "block_delta_nll": block_delta.tolist(),
        "limitations": ["32 fixed English windows, context512; no quality equivalence gate",
                        "pretrained models only; no training or downstream QA benchmark",
                        "resident transformers/bitsandbytes; no D1 kernel or RAM streaming"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "bf16", "nf4", "compare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    elif args.action in ("bf16", "nf4"):
        evaluate(args.output, args.action)
    elif args.action == "compare":
        write(args.output / "comparison.json", compare(args.output))
    else:
        for action in ("prepare", "bf16", "nf4", "compare"):
            subprocess.run([sys.executable, __file__, action, "--output", str(args.output)],
                           check=True)


if __name__ == "__main__":
    main()
