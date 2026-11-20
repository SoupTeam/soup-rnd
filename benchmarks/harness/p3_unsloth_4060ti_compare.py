"""Compare Unsloth (and HF BnB CPU-offload) against Soup P3 on RTX 4060 Ti 8 GB.

Runs each arm in an isolated child process with a 20 ms NVML + RSS peak poller:
1. Positive control: Unsloth on Qwen/Qwen2.5-0.5B (proves Unsloth + Triton work on this GPU).
2. Target model (Qwen/Qwen3.8-27B or Kanana-2-30B-A3B):
   - unsloth_default: FastLanguageModel.from_pretrained(..., load_in_4bit=True)
   - unsloth_cpu_offload: FastLanguageModel with device_map="auto" + CPU offload
   - hf_bnb_cpu_offload: Transformers + BitsAndBytes NF4 with device_map="auto" + CPU offload
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

KANANA_REPO = "kakaocorp/kanana-2-30b-a3b-base-2601"
KANANA_REV = "0d31365acf656e5f9a593cacc4913e9c0d36cb1c"
QWEN38_REPO = "Qwen/Qwen3.8-27B"
QWEN38_REV = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
CONTROL_REPO = "Qwen/Qwen2.5-0.5B"


def _rss_gib() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss) / (1024**3)


class _PeakPoller:
    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.nvml_peak_gib: Optional[float] = None
        self.rss_peak_gib: float = _rss_gib()
        self.samples: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle: Any = None
        self._pynvml: Any = None

    def __enter__(self) -> "_PeakPoller":
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None
            self._handle = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = _rss_gib()
            if rss > self.rss_peak_gib:
                self.rss_peak_gib = rss
            if self._pynvml is not None and self._handle is not None:
                try:
                    info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    used = float(info.used) / (1024**3)
                    if self.nvml_peak_gib is None or used > self.nvml_peak_gib:
                        self.nvml_peak_gib = used
                except Exception:
                    pass
            self.samples += 1
            time.sleep(self.interval_s)

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def _run_worker(args: argparse.Namespace) -> None:
    import torch

    t0 = time.perf_counter()
    res: Dict[str, Any] = {
        "arm": args.arm,
        "case": args.case,
        "model_path": args.model_path,
        "k": args.k,
        "seq_len": args.seq_len,
        "lora_r": args.lora_r,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_total_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3)
            if torch.cuda.is_available()
            else 0.0
        ),
        "status": "unknown",
    }
    try:
        import transformers

        res["transformers_version"] = transformers.__version__
    except Exception:
        res["transformers_version"] = "unknown"

    try:
        import unsloth

        res["unsloth_version"] = getattr(unsloth, "__version__", "unknown")
    except Exception as exc:
        res["unsloth_version"] = f"import_error: {exc}"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    with _PeakPoller(0.02) as poller:
        try:
            if args.arm == "unsloth_default":
                from unsloth import FastLanguageModel

                model, tok = FastLanguageModel.from_pretrained(
                    model_name=args.model_path,
                    max_seq_length=args.seq_len,
                    dtype=torch.bfloat16,
                    load_in_4bit=True,
                )
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=args.lora_r,
                    lora_alpha=args.lora_r * 2,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    use_gradient_checkpointing="unsloth",
                )
            elif args.arm == "unsloth_cpu_offload":
                from unsloth import FastLanguageModel

                model, tok = FastLanguageModel.from_pretrained(
                    model_name=args.model_path,
                    max_seq_length=args.seq_len,
                    dtype=torch.bfloat16,
                    load_in_4bit=True,
                    device_map="auto",
                    max_memory={0: "7GiB", "cpu": "50GiB"},
                )
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=args.lora_r,
                    lora_alpha=args.lora_r * 2,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    use_gradient_checkpointing="unsloth",
                )
            elif args.arm == "hf_bnb_cpu_offload":
                from peft import LoraConfig, TaskType, get_peft_model
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                    max_memory={0: "7GiB", "cpu": "50GiB"},
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                model.gradient_checkpointing_enable()
                targets = (
                    ["q_a_proj", "kv_a_proj_with_mqa", "o_proj"]
                    if args.case == "kanana"
                    else ["q_proj", "k_proj", "v_proj", "o_proj"]
                )
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=args.lora_r,
                        lora_alpha=args.lora_r * 2,
                        lora_dropout=0.0,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=targets,
                    ),
                )
            else:
                raise ValueError(f"Unknown arm: {args.arm}")

            load_wall_s = time.perf_counter() - t0
            res["load_wall_s"] = round(load_wall_s, 3)

            # Run 1 optimizer step with gradient accumulation = args.k
            cfg = getattr(model.config, "text_config", None) or model.config
            vocab_size = int(getattr(cfg, "vocab_size", 32000))
            model.train()
            trainables = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainables, lr=5e-4)
            gen = torch.Generator().manual_seed(2026)
            batches = [
                torch.randint(0, vocab_size, (1, args.seq_len), generator=gen)
                for _ in range(args.k)
            ]

            torch.cuda.synchronize(0)
            t_step0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            for mb in batches:
                ids = mb.to("cuda:0")
                out = model(input_ids=ids, labels=ids, use_cache=False)
                loss = out.loss / float(args.k)
                loss.backward()
                total_loss += float(loss.detach().cpu().item())
            opt.step()
            torch.cuda.synchronize(0)
            step_wall_s = time.perf_counter() - t_step0

            res["status"] = "ok"
            res["step_wall_s"] = round(step_wall_s, 3)
            res["tokens"] = int(args.k * args.seq_len)
            res["tokens_per_s"] = round((args.k * args.seq_len) / max(step_wall_s, 1e-9), 2)
            res["loss"] = round(total_loss, 4)
            del tok
        except Exception as exc:
            res["status"] = "oom" if "out of memory" in str(exc).lower() else "error"
            res["error_type"] = type(exc).__name__
            res["error_message"] = str(exc)[:600]
            res["traceback_tail"] = traceback.format_exc()[-1200:]

    res["total_wall_s"] = round(time.perf_counter() - t0, 3)
    res["cuda_allocated_peak_gib"] = (
        round(torch.cuda.max_memory_allocated(0) / (1024**3), 4)
        if torch.cuda.is_available()
        else 0.0
    )
    res["cuda_reserved_peak_gib"] = (
        round(torch.cuda.max_memory_reserved(0) / (1024**3), 4)
        if torch.cuda.is_available()
        else 0.0
    )
    res["nvml_peak_gib"] = (
        round(poller.nvml_peak_gib, 4) if poller.nvml_peak_gib is not None else None
    )
    res["rss_peak_gib"] = round(poller.rss_peak_gib, 4)
    res["poll_samples"] = poller.samples

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")


def _download_model(case: str, work_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = work_dir / case
    target.mkdir(parents=True, exist_ok=True)
    if case == "control_05b":
        snapshot_download(CONTROL_REPO, local_dir=target, token=False, max_workers=4)
    elif case == "qwen38_27b":
        snapshot_download(
            QWEN38_REPO, revision=QWEN38_REV, local_dir=target, token=False, max_workers=4
        )
    elif case == "kanana":
        snapshot_download(
            KANANA_REPO, revision=KANANA_REV, local_dir=target, token=False, max_workers=4
        )
    else:
        raise ValueError(f"Unknown case: {case}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth vs Soup RTX 4060 Ti 8GB comparison.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--case", choices=["control_05b", "qwen38_27b", "kanana"], required=True
    )
    parser.add_argument("--work", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument(
        "--arm",
        choices=["unsloth_default", "unsloth_cpu_offload", "hf_bnb_cpu_offload"],
        default="unsloth_default",
    )
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.worker:
        _run_worker(args)
        return

    from rich.console import Console

    console = Console()
    work = Path(args.work)
    model_path = Path(args.model_path) if args.model_path else _download_model(args.case, work)
    arms: List[str] = args.arms or [args.arm]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    for arm in arms:
        arm_out = out_dir / f"{args.case}_{arm}.json"
        console.print(
            f"[cyan]=== Running {args.case} | {arm} (k={args.k}, S={args.seq_len}) ===[/cyan]"
        )
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--case",
            args.case,
            "--work",
            str(work),
            "--model-path",
            str(model_path),
            "--arm",
            arm,
            "--k",
            str(args.k),
            "--seq-len",
            str(args.seq_len),
            "--lora-r",
            str(args.lora_r),
            "--output",
            str(arm_out),
        ]
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.run(cmd, env=env, check=False)
        if arm_out.exists():
            data = json.loads(arm_out.read_text(encoding="utf-8"))
        else:
            data = {
                "arm": arm,
                "case": args.case,
                "status": "process_killed",
                "returncode": proc.returncode,
            }
        summary.append(data)
        console.print(
            f"  -> status={data.get('status')} tok/s={data.get('tokens_per_s')} "
            f"nvml_pk={data.get('nvml_peak_gib')}GiB rss_pk={data.get('rss_peak_gib')}GiB "
            f"err={data.get('error_type')}: {str(data.get('error_message', ''))[:140]}"
        )

    (out_dir / f"{args.case}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
