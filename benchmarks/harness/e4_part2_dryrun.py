"""E4 part 2 calibration, dry run: the full path on CPU with a fake card.

    python benchmarks/harness/e4_part2_calibration.py calibrate --dry-run --out OUT \\
        --instance-id dry-1 [--fake-card a100-40] [--fake-reserved-bytes N] [--inject SPEC]

The dry run goes through the same code as a paid run: preflight card check, journal, driver
(fresh process per run, one rerun of a void), ``execute_attempt``, records, ``decide`` and
``report``. What is fake is only what a CPU cannot give: the box stamp (a fake A100 or other
card, named as fake), the GPU peers probe, and the memory numbers of F. The stages themselves
run on a tiny random OLMoE with the real stage code (run_f, run_t, run_w, the N scorer), so
every record has the schema of a real one by construction. The numbers are not results.

A dry-run record carries a ``dry_run`` block; ``dry_run_mix_problem`` keeps dry-run and real
records out of one ``--out``. ``--inject RUN_ID=KIND[@all]`` forces one of the attempt endings
a paid run can have: ``oom`` (F: the arm's OOM result; T, W, N, P: an OutOfMemoryError crash),
``kill`` (SIGKILL on Unix, TerminateProcess via SIGTERM on Windows), ``void`` (a void reason),
``fail`` (a failed result). A kind hits the first attempt only, or every attempt with ``@all``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import e4_part2_common as common
import e4_part2_quant as quant

DRY_DATASET_REVISION = "dry-run-no-dataset"
DEFAULT_FAKE_RESERVED_BYTES = 20_000_000_000  # the rule's pre-commit estimate, not a measurement
EXPERTS = 4
INJECT_KINDS = ("oom", "kill", "void", "fail")

#: name -> (nvidia-smi name, compute_cap, memory.total in MiB). Every name says it is fake.
FAKE_CARDS: Dict[str, Tuple[str, str, str]] = {
    "a100-40": ("NVIDIA A100-SXM4-40GB [dry-run fake]", "8.0", "40960"),
    "a100-80": ("NVIDIA A100-SXM4-80GB [dry-run fake]", "8.0", "81920"),
    "t4": ("Tesla T4 [dry-run fake]", "7.5", "15360"),
    "a100-odd": ("NVIDIA A100 [dry-run fake]", "8.0", "45056"),
}

TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4, num_experts=EXPERTS, num_experts_per_tok=2,
    max_position_embeddings=128, pad_token_id=1, eos_token_id=2, tie_word_embeddings=False,
)


# --------------------------------------------------------------------------
# tiny random OLMoE (also used by tests/e4_part2_support.py)
# --------------------------------------------------------------------------


def tiny_config(**overrides):
    from transformers import OlmoeConfig

    return OlmoeConfig(**{**TINY, **overrides})


def tiny_model(seed: int = 0, dtype=None, experts_implementation: str = "eager", **overrides):
    """A tiny random OLMoE (fused experts, weights drawn from the seed)."""
    import torch
    from transformers import OlmoeForCausalLM

    torch.manual_seed(seed)
    config = tiny_config(**overrides)
    config._experts_implementation = experts_implementation
    model = OlmoeForCausalLM(config)
    for param in model.parameters():  # std 0.02 init leaves the experts near-degenerate
        with torch.no_grad():
            param.copy_(torch.randn(param.shape, generator=torch.Generator().manual_seed(
                seed + hash(tuple(param.shape)) % 1000)) * 0.05)
    return model.to(dtype) if dtype is not None else model


def write_legacy_checkpoint(model, directory: Path, shards: int = 2) -> Path:
    """Save ``model`` in the original OLMoE layout: one 2-D tensor per expert projection.

    This is the layout of the real checkpoint (``experts.{e}.{gate,up,down}_proj.weight``);
    transformers 5 fuses it on load.
    """
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    inter = model.config.intermediate_size
    flat = {}
    for key, tensor in model.state_dict().items():
        if key.endswith("experts.gate_up_proj"):
            base = key[: -len("gate_up_proj")]
            for e in range(tensor.shape[0]):
                flat[f"{base}{e}.gate_proj.weight"] = tensor[e, :inter].clone()
                flat[f"{base}{e}.up_proj.weight"] = tensor[e, inter:].clone()
        elif key.endswith("experts.down_proj"):
            base = key[: -len("down_proj")]
            for e in range(tensor.shape[0]):
                flat[f"{base}{e}.down_proj.weight"] = tensor[e].clone()
        else:
            flat[key] = tensor.clone()
    names = sorted(flat)
    weight_map, total = {}, 0
    for index in range(shards):
        chunk = {k: flat[k] for k in names[index::shards]}
        fname = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        save_file(chunk, str(directory / fname), metadata={"format": "pt"})
        for k, v in chunk.items():
            weight_map[k] = fname
            total += v.numel() * v.element_size()
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
    model.config.save_pretrained(directory)
    return directory


def synthetic_examples(n: int, seed: int = 0, vocab: int = 128, eos: int = 2) -> List[dict]:
    import torch

    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        n_prompt = int(torch.randint(4, 9, (1,), generator=gen))
        n_resp = int(torch.randint(3, 8, (1,), generator=gen))
        prompt = torch.randint(3, vocab, (n_prompt,), generator=gen).tolist()
        resp = torch.randint(3, vocab, (n_resp,), generator=gen).tolist() + [eos]
        out.append({"prompt_ids": prompt, "response_ids": resp, "input_ids": prompt + resp,
                    "labels": [-100] * n_prompt + resp})
    return out


def world_dir(out: Path) -> Path:
    """Where the tiny model and its quantized files live: beside ``out``, not inside it."""
    out = Path(out)
    return out.with_name(out.name + "-dryworld")


def build_world(root: Path, log: Callable[[str], None] = lambda _m: None) -> Dict[str, Path]:
    """A tiny OLMoE checkpoint and its NF4 write-back and code files, made once."""
    import torch

    root = Path(root)
    paths = {"source": root / "source", "data": root / "data"}
    if (root / "world.ok").exists():
        return paths
    model = tiny_model(seed=6, dtype=torch.bfloat16)
    write_legacy_checkpoint(model, paths["source"])
    code = quant.run(paths["source"], paths["data"], root / "manifest.json", None, 2, EXPERTS,
                     log=log)
    if code not in (0, 2):  # 2: U2 is not compared against part 1 here
        raise RuntimeError(f"the dry-run quantization stopped with code {code}")
    (root / "world.ok").write_text("ok\n")
    return paths


def arm_dir(world: Dict[str, Path], arm: str) -> Path:
    return {"A_off": world["data"] / "writeback-dqoff", "A_on": world["data"] / "writeback-dqon",
            "A_bf16": world["source"]}[arm]


# --------------------------------------------------------------------------
# fake box
# --------------------------------------------------------------------------


def fake_stamp(card: str, instance: Optional[str], gpu_index: int = 0) -> Dict[str, Any]:
    """The keys of ``common.box_stamp`` with a fake, named card. Quiet, no GPU peers."""
    import time

    name, cap, memory = FAKE_CARDS[card]
    return {
        "unix_s": time.time(), "power_source": "N/A (hosted instance)",
        "ram_available_bytes": 64_000_000_000,
        "commit_charge": {"used_bytes": 8_000_000_000, "limit_bytes": 128_000_000_000},
        "gpu_processes": [], "gpu_clock": {"sm_clock_mhz": 1410, "reasons": []},
        "gpu_memory_used_mib": 0, "own_pid": os.getpid(), "instance_id": instance,
        "gpu_name": name, "gpu_uuid": f"GPU-DRYRUN-{card}-{instance}",
        "driver_version": "0.0 [dry-run fake]", "gpu_memory_total_mib": memory,
        "gpu_compute_cap": cap, "driver_cuda_version": "0.0",
        "torch_cuda": "dry-run (cpu)", "cuda_available": False, "unknown": {},
    }


def fake_identify_own() -> Dict[str, Any]:
    return {"own_pids": [os.getpid()], "method": "dry run: no GPU, nothing to identify"}


def dry_marker(card: str, injected: str) -> Dict[str, Any]:
    return {"fake_card": card, "inject": injected or None,
            "note": "dry run: tiny random model on CPU, fake box stamp; no number is a result"}


# --------------------------------------------------------------------------
# injections
# --------------------------------------------------------------------------


def parse_inject(spec: Optional[str]) -> Dict[str, Tuple[str, bool]]:
    """``F_A_on=oom,N_2=kill@all`` -> {run_id: (kind, every_attempt)}."""
    found: Dict[str, Tuple[str, bool]] = {}
    for item in filter(None, (spec or "").split(",")):
        run_id, _, what = item.partition("=")
        kind, _, every = what.partition("@")
        if not run_id or kind not in INJECT_KINDS or every not in ("", "all"):
            raise ValueError(f"--inject {item!r}: want RUN_ID=KIND[@all], KIND in {INJECT_KINDS}")
        found[run_id] = (kind, every == "all")
    return found


def injected_kind(spec: Dict[str, Tuple[str, bool]], run_id: str, attempt_number: int
                  ) -> Optional[str]:
    if run_id not in spec:
        return None
    kind, every = spec[run_id]
    return kind if (every or attempt_number == 1) else None


@contextmanager
def _oom_on_load(cal, enabled: bool):
    """Make the model load raise what CUDA raises when it runs out of memory."""
    if not enabled:
        yield
        return
    import torch

    original = cal.load_base

    def raising(*_a, **_k):
        raise torch.cuda.OutOfMemoryError("dry run: injected out of memory")

    cal.load_base = raising
    try:
        yield
    finally:
        cal.load_base = original


# --------------------------------------------------------------------------
# stage bodies
# --------------------------------------------------------------------------

DRY_FAKE_FIELDS = ["max_memory_allocated_bytes", "max_memory_reserved_bytes"]


def stage_body(cal, args, stage: str, arm: Optional[str], run_id: str, attempt_number: int,
               world: Dict[str, Path]) -> Dict[str, Any]:
    """One stage on the tiny world, with the real stage code. Raises or returns a result."""
    import torch

    recipe = dry_recipe(cal)
    kind = injected_kind(parse_inject(args.inject), run_id, attempt_number)
    if kind == "kill":
        os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))
    oom = kind == "oom"
    out = Path(args.out)
    torch.manual_seed(args.recipe_seed)
    smoke_rows = [r["input_ids"] for r in synthetic_examples(40, seed=3)]
    if stage in ("F", "P"):
        use_arm = "A_on" if stage == "P" else arm
        adapter_out = out / "adapters" / run_id / str(attempt_number)
        with _oom_on_load(cal, oom):
            result = cal.run_f(arm_dir(world, use_arm), use_arm, synthetic_examples(12),
                               synthetic_examples(4, 9), recipe, "cpu", 1, None, adapter_out,
                               experts_implementation="eager")
        if not result["oom"]:
            reserved = int(args.fake_reserved_bytes)
            result.update({"max_memory_reserved_bytes": reserved,
                           "max_memory_allocated_bytes": int(reserved * 0.8),
                           "dry_run_fake_fields": DRY_FAKE_FIELDS})
        result["adapter_dir"] = str(adapter_out)
        if stage == "P":
            cal.finalize_p_result(result)
            first = cal.find_record(out, f"{args.f_prefix}_A_on")
            if first is None:
                raise RuntimeError("P needs the F A_on record to compare with")
            result["comparison"] = cal.compare_repeat(first["result"], result)
    elif stage == "T":
        common.set_fp32_reference_mode()
        with _oom_on_load(cal, oom):
            bf16 = cal.load_base(arm_dir(world, "A_bf16"), "cpu", "eager")
        result = cal.run_t(quant.enumerate_units(2, EXPERTS), cal.UnitFiles(world["data"]), bf16,
                           smoke_rows, recipe, "cpu", None, per_kind=2, calls_per_unit=2,
                           min_units=1)
        result["experts_implementation_resolved"] = bf16.config._experts_implementation
        result["attention_implementation_resolved"] = bf16.config._attn_implementation
    elif stage == "W":
        with _oom_on_load(cal, oom):
            result = cal.run_w({"A_off": arm_dir(world, "A_off"),
                                "A_on": arm_dir(world, "A_on")}, smoke_rows, recipe, "cpu",
                               layers=(0, 1), rows=4, experts_implementation="eager")
    elif stage == "N":
        src = str(common.REPO_ROOT / "src")  # a paid run gets soup_cli from PYTHONPATH
        if src not in sys.path:
            sys.path.insert(0, src)
        if oom:
            with _oom_on_load(cal, True):
                cal.load_base(arm_dir(world, "A_on"), "cpu", "eager")
        result = cal.score_n(lambda _prompt: "A", {
            "experts_implementation_resolved": "eager",
            "attention_implementation_resolved": "sdpa", "device_used": "cpu"})
        result["dry_run_fake_fields"] = ["scores"]
    else:
        raise SystemExit(f"stage {stage} is not runnable")
    if kind == "void":
        result["void_reasons"] = list(result.get("void_reasons", [])) + ["dry run: injected void"]
    if kind == "fail":
        result["failed"] = True
    return result


def dry_recipe(cal):
    """The tiny recipe the tests use: a few updates of two sequences."""
    return cal.Recipe(accum=2, f_updates=3, warmup_updates=2, total_updates=8)


def cmd_run(cal, args) -> int:
    """``run`` on the dry-run box: the same ``execute_attempt`` as a paid run."""
    out = Path(args.out)
    cal.guard_out(out, dry_run=True)
    if args.stage != "F":
        args.f_prefix = cal.resolve_f_prefix(out, args.f_prefix)
    expected_instance = cal.resolve_final_instance(out) if args.stage != "F" else None
    world = build_world(world_dir(out))
    journal = common.AttemptsJournal(out / "attempts.jsonl")
    suffix = {"F": f"{args.f_prefix}_{args.arm}", "N": f"N_{args.repeat}"}.get(args.stage,
                                                                          args.stage)
    run_id = args.run_id or suffix

    def body() -> Dict[str, Any]:
        return stage_body(cal, args, args.stage, args.arm, run_id,
                          len(journal.attempts(run_id)), world)

    record = cal.execute_attempt(
        journal, run_id, args.stage, out / "runs", body, require_gpu=True,
        gpu_index=args.gpu_index, instance=args.instance_id,
        stamp=lambda _gpu, instance: fake_stamp(args.fake_card, instance),
        identify_own=fake_identify_own, expected_instance=expected_instance,
        dataset_revision=DRY_DATASET_REVISION, card_prefix=args.f_prefix,
        require_identity=True,
        dry_run=dry_marker(args.fake_card, args.inject))
    print(json.dumps({"run_id": run_id, "outcome": record["outcome"],
                      "void_reasons": record["void_reasons"]}))
    return 3 if record["outcome"] == common.OUTCOME_VOID else 0
