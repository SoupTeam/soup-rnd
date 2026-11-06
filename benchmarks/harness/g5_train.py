"""G5: one LoRA run over a MoE whose experts follow a per-expert codec map, NF4 or Q2_0.

Rule: benchmarks/gate-g5-mixed-precision-experts.md. The run is C5's
``c5_train.py``, unchanged: data, model and revision, LoRA, seeds and batch
order, training loop, task loss, gate suites and the record. Only the expert
swap differs: every expert slice goes through C5's own per-slice swap
(``c5_expert_swap._slice_errors``) with the codec the map gives it, so an
all-NF4 map gives C5's NF4 weights bit for bit, and so does all-Q2_0.

    python benchmarks/harness/g5_train.py --map SEL/maps/mixed_traffic_f25.json --seed 42 \\
        --model allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \\
        --data-dir DATA --device cuda --gradient-checkpointing \\
        --out-dir RUNS/mixed_traffic_f25-s42-a1
    python benchmarks/harness/g5_train.py --map SEL/maps/mixed_traffic_f25.json --seed 42 \\
        --fixture granite --data-dir DATA --train-limit 64 --panel-limit 32 --out-dir RUN

Every argument but ``--map`` goes to ``c5_train.py``. The arm is the map's
``arm``; its config ``codec`` is ``map:<map_sha256>``, the only config key
besides the seed where it differs from C5's arms. ``run.json`` keeps the map
under ``swap.codec_map``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_expert_swap as swap  # noqa: E402
import c5_train  # noqa: E402
import g5_select  # noqa: E402

if TYPE_CHECKING:
    import torch

#: Keys every slice of ``c5_expert_swap._slice_errors`` returns, whatever its codec.
COMMON_SUMS = ("error", "error_in_model", "weight")


def load_map(path: str | Path) -> dict[str, Any]:
    """A ``maps/<arm>.json`` from ``g5_select.py``, refused unless its hash is its codes'."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("kind") != "g5_codec_map":
        raise ValueError(f"{path} is not a G5 codec map")
    if record["arm"] not in g5_select.ARM_SPECS:
        raise ValueError(f"{path}: unknown arm {record['arm']!r}")
    if g5_select.map_sha256(record["codes"]) != record["map_sha256"]:
        raise ValueError(f"{path}: map_sha256 does not match its codes")
    return record


def map_codec(record: dict[str, Any]) -> str:
    """The config ``codec`` of a mixed arm."""
    return f"map:{record['map_sha256']}"


def apply_map(model: torch.nn.Module, codes: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Swap every expert slice to its codec in ``codes[layer][expert]``, in place.

    The same record as ``c5_expert_swap.run_arm``, with each tensor's codes and
    each expert's codec. Errors summed over a tensor cover the keys every codec
    has (``error``, ``error_in_model``).
    """
    import torch

    experts = swap.expert_tensor_names(model)
    by_module = swap.expert_module_tensor_names(model)
    params = dict(model.named_parameters())
    before = swap.fingerprints(model)
    tensors = {}
    with torch.no_grad():
        for name in experts:
            param = params[name]
            layer_codes = codes[g5_select.layer_of(name)]
            if param.dim() != 3 or param.shape[0] != len(layer_codes):
                raise ValueError(f"{name} {tuple(param.shape)} does not fit a map of "
                                 f"{len(layer_codes)} experts")
            per_expert, total = [], dict.fromkeys(COMMON_SUMS, 0.0)
            for weight, codec_name in zip(param.data, layer_codes):
                sums, n = swap._slice_errors(weight, codec_name)
                per_expert.append({"codec": codec_name, **swap._errors(sums, n)})
                for key in COMMON_SUMS:
                    total[key] += sums[key]
            tensors[name] = {"shape": list(param.shape), "dtype": str(param.dtype),
                             "codes": list(layer_codes),
                             **swap._errors(total, param.numel()), "per_expert": per_expert}
    after = swap.fingerprints(model)
    changed = [name for name in after if after[name] != before.get(name)]
    return {
        "experts": experts,
        "changed": changed,
        "tensors": tensors,
        "non_expert_fingerprints": {n: d for n, d in after.items() if n not in experts},
        "checks": {
            "expert_rules_agree": bool(experts) and sorted(experts) == sorted(by_module),
            "every_expert_changed": set(experts) <= set(changed),
            "non_expert_untouched": set(changed) <= set(experts),
        },
    }


def prepare_mixed(model: torch.nn.Module, record: dict[str, Any]) -> dict[str, Any]:
    """``c5_train.prepare_experts`` for a map: the same keys, plus the map under ``codec_map``."""
    swapped = apply_map(model, record["codes"])
    params = dict(model.named_parameters())
    expert_fingerprints = {name: swap._sha256(params[name]) for name in swapped["experts"]}
    return {
        "checks": swapped["checks"],
        "experts": swapped["experts"],
        "rel_mse": {
            name: {"error": entry["error"]["rel_mse"],
                   "error_in_model": entry["error_in_model"]["rel_mse"]}
            for name, entry in swapped["tensors"].items()
        },
        "non_expert_fingerprints": swapped["non_expert_fingerprints"],
        "expert_digest": hashlib.sha256(
            json.dumps(expert_fingerprints, sort_keys=True).encode()
        ).hexdigest(),
        "codec_map": {
            "arm": record["arm"],
            "rule": record["rule"],
            "fraction_q2_0": record["fraction_q2_0"],
            "map_sha256": record["map_sha256"],
            "stats_sha256": record["stats_sha256"],
            "calibration_sha256": record.get("calibration_sha256"),
            "q2_0_per_layer": record["q2_0_per_layer"],
            "command": record.get("command"),
            "per_expert": {name: [e["codec"] for e in entry["per_expert"]]
                           for name, entry in swapped["tensors"].items()},
            "harness": {
                "g5_train_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "g5_select_sha256": hashlib.sha256(
                    Path(g5_select.__file__).read_bytes()).hexdigest(),
                # raw and LF-normalised hashes of C5's and G5's files (g5_select.SOURCES)
                "sources": g5_select.source_digests(),
            },
        },
    }


@contextlib.contextmanager
def mixed_arm(record: dict[str, Any]) -> Iterator[str]:
    """Register the map's arm with ``c5_train`` for one run; restored on exit."""
    arm = record["arm"]
    previous = c5_train.prepare_experts

    def prepare(model: torch.nn.Module, name: str) -> dict[str, Any]:
        return prepare_mixed(model, record) if name == arm else previous(model, name)

    c5_train.ARM_CODECS[arm] = map_codec(record)
    c5_train.prepare_experts = prepare
    try:
        yield arm
    finally:
        c5_train.prepare_experts = previous
        c5_train.ARM_CODECS.pop(arm, None)


def _check_model(record: dict[str, Any], rest: list[str]) -> None:
    """The map must come from the model this run loads."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--fixture")
    parser.add_argument("--revision")
    args, _ = parser.parse_known_args(rest)
    wanted = {"id": f"fixture:{args.fixture}" if args.fixture else args.model,
              "revision": None if args.fixture else args.revision}
    if record["model"]["id"] != wanted["id"] or (
        not args.fixture and record["model"]["revision"] != wanted["revision"]
    ):
        raise SystemExit(f"the map is for {record['model']}, this run loads {wanted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--map", required=True, help="maps/<arm>.json from g5_select.py")
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if any(item == "--arm" or item.startswith("--arm=") for item in rest):
        parser.error("the arm is the map's; do not pass --arm")
    record = load_map(args.map)
    _check_model(record, rest)
    with mixed_arm(record) as arm:
        # c5_train records the argv it gets ("c5_train.py --arm ..."); the real command line,
        # with --map, goes into the record's swap.codec_map
        record["command"] = [sys.executable, *sys.argv] if argv is None else ["g5_train.py",
                                                                               *argv]
        return c5_train.main(["--arm", arm, *rest])


if __name__ == "__main__":
    sys.exit(main())
