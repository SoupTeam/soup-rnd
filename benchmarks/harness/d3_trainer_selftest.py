"""New launcher synthetic test only. Reject any other model bytes before forward."""

import argparse
import gc
import hashlib
import json
import os
import traceback
from pathlib import Path

os.environ.update(
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    HF_HUB_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-checkpoint", type=Path, required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    root = args.out.resolve()
    root.mkdir(exist_ok=False, parents=True)
    record = dict(
        status="preparing_synthetic",
        formal_execution=False,
        pretrained_model_executed=False,
        numerical_cells=[],
        profiles=[],
    )

    def save():
        (root / "status.json").write_text(json.dumps(record, indent=2) + "\n")

    try:
        record["source_files"] = {
            path.relative_to(source).as_posix(): dict(
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                lf_sha256=hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            )
            for base in ("src", "benchmarks/harness")
            for path in (source / base).rglob("*.py")
        }
        save()
        import torch
        from datasets import Dataset
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        from transformers import AutoConfig

        from benchmarks.harness.d3_trainer_gate import (
            correctness,
            make_config,
            public_workspace,
        )
        from benchmarks.harness.d3_trainer_gate import save as write_json
        from benchmarks.harness.d3_trainer_prepare import metadata
        from benchmarks.harness.d3_trainer_protocol import (
            CODECS,
            CORRECTNESS_MODES,
            file_hash,
            read_rule,
        )
        from benchmarks.harness.d3_trainer_series import performance
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.utils.layer_shard import resolve_shard_dir

        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        torch.use_deterministic_algorithms(True)
        checkpoint = args.synthetic_checkpoint.resolve(strict=True)
        if (
            file_hash(checkpoint / "model.safetensors")
            != "541a84fa8513d02da3381a8c2fbc83efe4e3ff89975d958f0cc3505dc91b550d"
        ):
            raise ValueError("self-test accepts only the pinned random-weight fixture")
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        if (config.num_hidden_layers, config.num_experts, config.vocab_size) != (4, 4, 64) or (
            config.tie_word_embeddings
        ):
            raise ValueError("self-test refuses a pretrained model configuration")
        record["fixture"] = {
            "layers": 4, "experts": 4, "vocab": 64, "tie_word_embeddings": False,
            "weight_sha256": file_hash(checkpoint / "model.safetensors"),
        }
        rule = read_rule()
        prepared = dict(
            workspace=str(root), codecs={}, rows_path=str(root / "rows.jsonl"), library=args.library
        )
        ids = torch.randint(3, 64, (7, 512), generator=torch.Generator().manual_seed(1729))
        labels = ids.clone()
        labels[:, ::11] = -100
        rows = [
            dict(input_ids=x.tolist(), labels=y.tolist(), attention_mask=[1] * 512)
            for x, y in zip(ids, labels)
        ]
        Path(prepared["rows_path"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
        for codec in CODECS:
            folder = root / codec
            folder.mkdir()
            alias = folder / "checkpoint"
            alias.mkdir()
            for path in checkpoint.iterdir():
                if path.is_file():
                    os.link(path, alias / path.name)
            data = folder / "dataset"
            Dataset.from_list(rows).save_to_disk(str(data))
            row = dict(
                checkpoint=str(alias),
                dataset=str(data),
                initial_adapters=str(folder / "initial-adapters.safetensors"),
                shards=resolve_shard_dir(str(alias)),
            )
            prepared["codecs"][codec] = row
            cfg = make_config(rule, prepared, folder / "setup", codec, "sequential", "ordinary")
            write_json(data / "metadata.json", metadata(cfg))
            owner = SFTTrainerWrapper(cfg, device="cuda")
            try:
                with public_workspace(prepared, folder / "setup"):
                    owner.setup(dict(train=[], val=[]))
                if not owner.model.get_input_embeddings().weight.is_meta or (
                    len(owner._stream_runtime.large_pool.specs) != 2
                ):
                    raise ValueError("untied streamed embedding/head fixture was not exercised")
                row["streamed_embedding_meta"] = True
                row["large_boundaries"] = sorted(owner._stream_runtime.large_pool.specs)
                save_file(
                    {
                        key: value.detach().cpu().contiguous().clone()
                        for key, value in get_peft_model_state_dict(owner.model).items()
                    },
                    row["initial_adapters"],
                )
            finally:
                owner._close_stream_runtime()
                del owner
                gc.collect()
                torch.cuda.empty_cache()
        write_json(root / "prepared.json", prepared)
        record.update(status="running_synthetic")
        save()
        oracles = {}
        for codec in CODECS:
            for training in ("ordinary", "l2l"):
                for mode in CORRECTNESS_MODES:
                    name = f"{codec}-{training}-{mode}"
                    folder = root / name
                    folder.mkdir()
                    record["numerical_cells"].append(dict(name=name, status="running"))
                    save()
                    oracle = oracles.get(codec)
                    result = correctness(rule, prepared, folder, codec, mode, training, oracle)
                    if mode == "native-resident" and training == "ordinary":
                        oracles[codec] = result["numerical"]
                    write_json(folder / "result.json", result)
                    record["numerical_cells"][-1]["status"] = "passed"
                    save()
                    gc.collect()
                    torch.cuda.empty_cache()
        for codec in CODECS:
            for training in ("ordinary", "l2l"):
                for mode in ("sequential", "grouped", "triton"):
                    name = f"profile-{codec}-{training}-{mode}"
                    folder = root / name
                    folder.mkdir()
                    record["profiles"].append(dict(name=name, status="running"))
                    save()
                    result = performance(
                        rule,
                        prepared,
                        folder,
                        dict(codec=codec, mode=mode, training=training, kind="profile"),
                        dict(numerical=oracles[codec]),
                    )
                    write_json(folder / "result.json", result)
                    record["profiles"][-1]["status"] = "passed"
                    save()
                    gc.collect()
                    torch.cuda.empty_cache()
        record.update(
            status="synthetic_launcher_checks_passed", numerical_count=36, profile_count=18
        )
    except BaseException:
        record.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
