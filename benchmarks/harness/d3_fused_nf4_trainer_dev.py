"""Synthetic development preflight. Not a trained-model or formal performance gate."""

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(os.environ.get("D3_FUSED_TRAINER_ROOT", "/root/d3-fused-screen/trainer-development-01"))
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = Path(os.environ.get("D3_DEVELOPMENT_SOURCE", "/root/d3-fused-screen"))
sys.path.insert(0, str(SOURCE / "src"))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def main(args):
    global torch
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    metadata = json.loads((SOURCE / "DEVELOPMENT-SOURCE.json").read_text())
    for relative, expected in metadata["files"].items():
        assert hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest() == expected, relative
    case = ROOT / args.case
    assert not case.exists(), "retain prior attempt; use a new --case"
    case.mkdir()
    record = dict(
        kind="synthetic public trainer CUDA development preflight",
        formal_session_started=False,
        backend=args.backend,
        codec=args.codec,
        l2l=args.l2l,
        source_files_verified=len(metadata["files"]),
        status="running",
    )
    save(case / "status.json", record)
    wrapper = None
    try:
        checkpoint = ROOT / "synthetic-checkpoint"
        if not checkpoint.exists():
            torch.manual_seed(71)
            cfg = OlmoeConfig(
                hidden_size=2048,
                intermediate_size=1024,
                num_hidden_layers=2,
                num_attention_heads=16,
                num_key_value_heads=16,
                num_experts=4,
                num_experts_per_tok=2,
                vocab_size=64,
                max_position_embeddings=1024,
                output_router_logits=False,
                attention_dropout=0.0,
                tie_word_embeddings=True,
                eos_token_id=1,
                bos_token_id=2,
                pad_token_id=0,
            )
            native = OlmoeForCausalLM(cfg).to(dtype=torch.bfloat16)
            native.save_pretrained(checkpoint)
            raw = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0"))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="0", eos_token="1")
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )
            tokenizer.save_pretrained(checkpoint)
            del native
            gc.collect()
        config = SoupConfig(
            base=str(checkpoint),
            output=str(case / "adapter"),
            data=dict(
                train=str(case / "data.jsonl"),
                max_length=512,
                val_split=0,
                train_on_responses_only=False,
                format="sharegpt",
            ),
            training=dict(
                backend="transformers",
                stream_layers=True,
                stream_source="ram",
                batch_size=1,
                gradient_accumulation_steps=2,
                epochs=1,
                lr=0.01,
                weight_decay=0,
                max_grad_norm=1,
                warmup_ratio=0,
                scheduler="constant",
                optimizer="sgd",
                seed=19,
                data_seed=29,
                logging_steps=1,
                save_steps=100,
                packing=False,
                rewind_log=False,
                l2l=args.l2l,
                moe_aux_loss_coeff=0,
                quantization="none" if args.codec == "none" else "4bit",
                bnb_4bit_use_double_quant=args.codec == "nf4-dq",
                stream_experts=dict(
                    backend=args.backend,
                    group_width=4,
                    native_rows_from=None if args.backend == "fused_nf4" else 481,
                    cublas_library=str(
                        next(
                            Path(torch.__file__).parents[1].glob("nvidia/cu13/lib/libcublas.so.13")
                        )
                    ),
                ),
                lora=dict(r=2, alpha=4, dropout=0, target_modules=["q_proj", "v_proj"]),
            ),
        )
        save(case / "config.json", config.model_dump(mode="json"))
        rows = [
            dict(
                messages=[
                    dict(role="user", content=" ".join(str(3 + (i + j) % 60) for j in range(512)))
                ]
            )
            for i in range(3)
        ]
        (case / "data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        wrapper = SFTTrainerWrapper(config, device="cuda", report_to="none")
        wrapper.setup(dict(train=rows, val=[]))
        runtime = wrapper._stream_runtime
        initial = {
            n: p.detach().cpu().clone()
            for n, p in wrapper.model.named_parameters()
            if p.requires_grad
        }
        assert initial and all("lora_" in name for name in initial)
        assert all(len(row["input_ids"]) == 512 for row in wrapper.trainer.train_dataset)
        from safetensors.torch import save_file
        from transformers import TrainerCallback

        class Capture(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                gradients = {
                    n: p.grad.detach().cpu().clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None
                }
                assert gradients and len(gradients) == len(initial)
                assert all(torch.isfinite(p).all().item() for p in gradients.values())
                save_file(gradients, str(case / f"gradients-{state.global_step}.safetensors"))

            def on_train_end(self, args, state, control, model=None, **kwargs):
                model.eval()
                ids = torch.tensor([[3 + j % 60 for j in range(512)]], device="cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    self.logits = model(input_ids=ids).logits.float().detach().cpu().clone()
                save_file({"logits": self.logits}, str(case / "trained-logits.safetensors"))

        capture = Capture()
        wrapper.trainer.add_callback(capture)
        torch.cuda.reset_peak_memory_stats()
        result = wrapper.train()
        assert result["total_steps"] == 2, result
        final = {
            n: p.detach().cpu().clone()
            for n, p in wrapper.model.named_parameters()
            if p.requires_grad
        }
        assert all(torch.isfinite(p).all().item() for p in final.values())
        assert any(digest(initial[n]) != digest(p) for n, p in final.items())
        assert runtime.expert_resources.closed
        stats = runtime.stats()
        assert stats["expert_fetches"] > 0
        if args.backend != "sequential":
            assert (
                stats["expert_gemm"][
                    "fused_matrices" if args.backend == "fused_nf4" else "grouped_matrices"
                ]
                > 0
            ), stats
        from safetensors.torch import load_file

        artifact = case / "adapter" / "adapter_model.safetensors"
        saved = load_file(str(artifact))
        assert saved and all("lora_" in n for n in saved)
        assert all(torch.isfinite(p).all().item() for p in saved.values())
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        reload_config = config.model_copy(deep=True)
        reload_config.output = str(case / "reload")
        restored = SFTTrainerWrapper(reload_config, device="cuda", report_to="none")
        try:
            restored.setup(dict(train=rows, val=[]))
            outcome = set_peft_model_state_dict(restored.model, saved)
            assert not outcome.unexpected_keys
            state = get_peft_model_state_dict(restored.model)
            assert set(state) == set(saved)
            assert all(
                state[n].dtype == p.dtype and digest(state[n]) == digest(p)
                for n, p in saved.items()
            )
            restored.model.eval()
            ids = torch.tensor([[3 + j % 60 for j in range(512)]], device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = restored.model(input_ids=ids).logits.float()
            record["reload_comparison"] = dict(
                wanted_dtype=str(capture.logits.dtype),
                actual_dtype=str(logits.dtype),
                wanted_sha256=digest(capture.logits),
                actual_sha256=digest(logits),
            )
            assert logits.dtype == capture.logits.dtype and digest(logits) == digest(capture.logits)
        finally:
            restored._close_stream_runtime()
        record.update(
            status="passed",
            result=result,
            runtime=stats,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            adapter_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            final_parameters={
                n: dict(dtype=str(p.dtype), shape=list(p.shape), sha256=digest(p))
                for n, p in final.items()
            },
            logs=wrapper.trainer.state.log_history,
        )
        from rich.console import Console

        Console().print(json.dumps({k: record[k] for k in ("status", "backend", "codec", "l2l")}))
    except BaseException:
        record.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        if wrapper is not None:
            wrapper._close_stream_runtime()
        save(case / "status.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--backend", choices=["sequential", "fused_nf4"], required=True)
    parser.add_argument("--codec", choices=["nf4", "nf4-dq"], default="nf4")
    parser.add_argument("--l2l", action="store_true")
    main(parser.parse_args())
