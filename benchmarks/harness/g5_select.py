"""G5: routing statistics, a sensitivity proxy and per-expert codec maps, from the train split only.

Rule: benchmarks/gate-g5-mixed-precision-experts.md. The C5 harness is reused
unchanged: ``c5_train`` loads the model and tokenizer and encodes the rows,
``c5_expert_swap`` fingerprints tensors, ``q2_0_codec`` holds both codecs and
``c5_data`` hashes the calibration subset.

    python benchmarks/harness/g5_select.py --model allenai/OLMoE-1B-7B-0924 \\
        --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --data-dir DATA --device cuda \\
        --out-dir SEL
    python benchmarks/harness/g5_select.py --fixture granite --data-dir DATA \\
        --calibration-rows 32 --out-dir SEL

Only ``DATA/train.jsonl`` is read; the held-out panel is never opened. The
calibration set is the first ``--calibration-rows`` rows of the train split
(512 on a real model, hash fixed in the rule). On the bf16 model, before any
quantization, a forward pre-hook on every ``*Experts`` module records the
tokens' hidden states, their top-k experts and gate weights. From those:

- traffic: routed (token, slot) assignments and gate-weight mass per (layer, expert);
- sensitivity: ``d_q2_nf4[l][e] = sum over tokens t routed to e of
  g[t,e]**2 * ||f_e^Q2_0(x_t) - f_e^NF4(x_t)||**2``, the squared change of the
  MoE block output when expert e alone goes from NF4 to Q2_0, with the
  fake-quant weights the training arm would see.

Maps put ``fraction * n_experts`` experts of every layer in Q2_0 and the rest in
NF4: the lowest by traffic (assignments, then mass, then expert index) or the
lowest by ``d_q2_nf4`` (then expert index). Every map of one fraction has the
same number of Q2_0 experts in every layer, so the same bytes.

Writes ``selection.json`` and ``maps/<arm>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_data  # noqa: E402
import c5_expert_swap as swap  # noqa: E402
import c5_train  # noqa: E402
import l2l_box  # noqa: E402
import q2_0_codec as codec  # noqa: E402

if TYPE_CHECKING:
    import torch

console = Console()

RULES = ("traffic", "sensitivity")
FRACTIONS = (0.25, 0.5)
CODECS = ("nf4", "q2_0")
#: First rows of the C5 train split used for calibration on a real model, and their hash
#: (``c5_data.split_digest``; the same for the 4000- and the 2000-row split).
CALIBRATION_ROWS = 512
CALIBRATION_SHA256 = "6a62ff891333a5dc4f9386715e3424d19e64f02187ff3f1f890b5deb296e98f0"
#: Tokens per chunk of the expert forward in the sensitivity pass.
TOKEN_CHUNK = 4096
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
HARNESS_DIR = Path(__file__).resolve().parent
#: The harness files a G5 record names: the C5 code G5 runs unchanged, then G5's own.
SOURCES = ("c5_train.py", "c5_expert_swap.py", "q2_0_codec.py", "c5_data.py",
           "g5_select.py", "g5_train.py", "g5_bytes.py", "g5_verdict.py")


def source_digest(path: Path) -> dict[str, str]:
    """SHA-256 of a file as checked out, and with every CRLF read as LF.

    ``sha256`` is what C5's records hold (``harness`` in run.json). ``sha256_lf``
    is the text git stores, so it is the same in every checkout of one commit;
    the two differ only where git turned LF into CRLF (``core.autocrlf=true``).
    """
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "sha256_lf": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}


def source_digests(names: Sequence[str] = SOURCES) -> dict[str, dict[str, str]]:
    """``source_digest`` of each named file of this harness directory."""
    return {name: source_digest(HARNESS_DIR / name) for name in names}


def arm_name(rule: str, fraction: float) -> str:
    """``mixed_traffic_f25`` and the like."""
    return f"mixed_{rule}_f{round(fraction * 100):02d}"


ARMS = tuple(arm_name(rule, fraction) for fraction in FRACTIONS for rule in RULES)
ARM_SPECS = {arm_name(rule, fraction): (rule, fraction)
             for fraction in FRACTIONS for rule in RULES}


def layer_of(name: str) -> int:
    """Decoder layer index in a dotted module or parameter name."""
    match = _LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"{name!r} names no decoder layer")
    return int(match.group(1))


def ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            out[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties; None if either side is constant."""
    if len(x) != len(y):
        raise ValueError("spearman needs two sequences of one length")
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def selection_key(layer: dict[str, Any], rule: str, expert: int) -> tuple:
    """What a rule sorts experts by, lowest first; the expert index breaks ties."""
    if rule == "traffic":
        return (layer["tokens"][expert], layer["mass"][expert], expert)
    if rule == "sensitivity":
        return (layer["d_q2_nf4"][expert], expert)
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def q2_0_count(n_experts: int, fraction: float) -> int:
    """Q2_0 experts per layer; refused unless ``fraction * n_experts`` is a whole number."""
    count = fraction * n_experts
    if not 0 < fraction < 1 or count != int(count):
        raise ValueError(
            f"fraction {fraction} of {n_experts} experts is not a whole number of experts"
        )
    return int(count)


def layer_codes(layer: dict[str, Any], rule: str, fraction: float) -> list[str]:
    """The codec of every expert of one layer under ``rule`` at ``fraction``."""
    n_experts = len(layer["tokens"])
    chosen = sorted(range(n_experts), key=lambda e: selection_key(layer, rule, e))
    q2_0 = set(chosen[:q2_0_count(n_experts, fraction)])
    return ["q2_0" if e in q2_0 else "nf4" for e in range(n_experts)]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def map_sha256(codes: Sequence[Sequence[str]]) -> str:
    """SHA-256 of a map's codes, layer by layer: what the arm trains over."""
    return hashlib.sha256(_canonical([list(layer) for layer in codes])).hexdigest()


def stats_sha256(layers: Sequence[dict[str, Any]]) -> str:
    """SHA-256 of the per-layer statistics a map is derived from."""
    return hashlib.sha256(_canonical(list(layers))).hexdigest()


def build_map(layers: Sequence[dict[str, Any]], rule: str, fraction: float) -> dict[str, Any]:
    """One arm's codec map from the per-layer statistics of ``selection.json``."""
    codes = [layer_codes(layer, rule, fraction) for layer in layers]
    return {
        "kind": "g5_codec_map",
        "arm": arm_name(rule, fraction),
        "rule": rule,
        "fraction_q2_0": fraction,
        "n_layers": len(codes),
        "n_experts": len(codes[0]) if codes else 0,
        "q2_0_per_layer": [layer.count("q2_0") for layer in codes],
        "codes": codes,
        "map_sha256": map_sha256(codes),
        "stats_sha256": stats_sha256(layers),
    }


def _share(values: Sequence[float], chosen: Sequence[bool]) -> float | None:
    total = sum(values)
    return sum(v for v, c in zip(values, chosen) if c) / total if total else None


def busiest_quartile_share(tokens: Sequence[int]) -> float | None:
    """Share of a layer's routed assignments carried by its busiest quarter of experts."""
    total = sum(tokens)
    if not total:
        return None
    quarter = max(1, len(tokens) // 4)
    return sum(sorted(tokens, reverse=True)[:quarter]) / total


def correlation_report(layers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How far traffic stands in for sensitivity, per layer and over layers.

    Spearman between assignments and ``d_q2_nf4`` (total proxy error), and
    between assignments and ``d_q2_nf4`` per assignment (the part not caused by
    traffic). Per fraction: the overlap of the two rules' Q2_0 sets (Jaccard),
    and the share of traffic and of proxy error each rule moves to Q2_0.
    """
    per_layer = []
    for layer in layers:
        per_assignment = [d / t if t else 0.0 for d, t in zip(layer["d_q2_nf4"], layer["tokens"])]
        per_layer.append({
            "layer": layer["layer"],
            "spearman_tokens_d": spearman(layer["tokens"], layer["d_q2_nf4"]),
            "spearman_tokens_d_per_assignment": spearman(layer["tokens"], per_assignment),
            "busiest_quartile_traffic_share": busiest_quartile_share(layer["tokens"]),
        })
    fractions = {}
    for fraction in FRACTIONS:
        if any(fraction * len(layer["tokens"]) != int(fraction * len(layer["tokens"]))
               for layer in layers):
            continue
        inter = union = 0
        shares: dict[str, dict[str, list[float | None]]] = {
            rule: {"traffic": [], "d_q2_nf4": []} for rule in RULES
        }
        for layer in layers:
            sets = {}
            for rule in RULES:
                chosen = [code == "q2_0" for code in layer_codes(layer, rule, fraction)]
                sets[rule] = {e for e, c in enumerate(chosen) if c}
                shares[rule]["traffic"].append(_share(layer["tokens"], chosen))
                shares[rule]["d_q2_nf4"].append(_share(layer["d_q2_nf4"], chosen))
            inter += len(sets["traffic"] & sets["sensitivity"])
            union += len(sets["traffic"] | sets["sensitivity"])
        fractions[str(fraction)] = {
            "jaccard_traffic_vs_sensitivity": inter / union if union else None,
            "q2_0_share": shares,
        }
    valid = [entry["spearman_tokens_d"] for entry in per_layer
             if entry["spearman_tokens_d"] is not None]
    return {
        "per_layer": per_layer,
        "mean_spearman_tokens_d": sum(valid) / len(valid) if valid else None,
        "fractions": fractions,
    }


# --- torch side: capture, fake-quant, sensitivity --------------------------------------------


def fake_quant(weight: torch.Tensor, codec_name: str) -> torch.Tensor:
    """What C5's swap writes into one expert slice: quantize, restore, cast to its dtype."""
    restored = codec.dequantize(codec.quantize(weight, codec_name))
    return restored.to(weight.device).to(weight.dtype)


def experts_modules(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    """C5's second expert rule (class name ends in ``Experts``), keyed by decoder layer."""
    found = {}
    for name, module in model.named_modules():
        if type(module).__name__.endswith("Experts"):
            for attr in ("gate_up_proj", "down_proj", "act_fn"):
                if not hasattr(module, attr):
                    raise ValueError(f"{name} ({type(module).__name__}) has no {attr}")
            found[layer_of(name + ".")] = module
    if not found:
        raise ValueError("no module whose class name ends in Experts")
    return dict(sorted(found.items()))


def expert_forward(
    module: torch.nn.Module, expert: int, x: torch.Tensor,
    gate_up: torch.Tensor | None = None, down: torch.Tensor | None = None,
) -> torch.Tensor:
    """One expert's output for tokens ``x`` in float32, with its own or given weights."""
    from torch.nn import functional

    gate_up = module.gate_up_proj[expert] if gate_up is None else gate_up
    down = module.down_proj[expert] if down is None else down
    gate, up = functional.linear(x.float(), gate_up.float()).chunk(2, dim=-1)
    return functional.linear(module.act_fn(gate) * up, down.float())


def capture_routing(
    model: torch.nn.Module, examples: Sequence[dict[str, Any]], device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Run the model on each example alone; what every experts module received.

    Per layer: ``x`` [T, H] (model dtype), ``index`` [T, k] and ``weight`` [T, k]
    (float32), the router's choice and gate weights exactly as the experts got
    them, for every token of every example (no padding, so no pad tokens).
    """
    import torch

    modules = experts_modules(model)
    seen: dict[int, list[tuple]] = {layer: [] for layer in modules}

    def hook(layer: int):
        def capture(module, args, kwargs):
            names = ("hidden_states", "top_k_index", "top_k_weights")
            values = list(args) + [kwargs[n] for n in names[len(args):]]
            x, index, weight = values[:3]
            if index.dtype.is_floating_point or index.shape != weight.shape:
                raise ValueError(f"layer {layer}: unexpected experts call {index.shape}")
            seen[layer].append((x.detach().cpu(), index.detach().cpu(),
                                weight.detach().float().cpu()))
        return capture

    handles = [m.register_forward_pre_hook(hook(layer), with_kwargs=True)
               for layer, m in modules.items()]
    model.eval()
    try:
        with torch.no_grad():
            for example in examples:
                ids = torch.tensor([example["input_ids"]], device=device)
                model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {
        layer: {"x": torch.cat([c[0] for c in calls]),
                "index": torch.cat([c[1] for c in calls]),
                "weight": torch.cat([c[2] for c in calls])}
        for layer, calls in seen.items()
    }


def traffic(index: torch.Tensor, weight: torch.Tensor, n_experts: int) -> dict[str, list]:
    """Routed (token, slot) assignments and gate-weight mass of every expert."""
    import torch

    flat = index.reshape(-1).long()
    tokens = torch.bincount(flat, minlength=n_experts)
    mass = torch.zeros(n_experts, dtype=torch.float64).index_add_(
        0, flat, weight.reshape(-1).double())
    return {"tokens": [int(t) for t in tokens], "mass": [float(m) for m in mass]}


def layer_sensitivity(
    module: torch.nn.Module, routed: dict[str, torch.Tensor], device: str,
) -> dict[str, list[float]]:
    """Per expert, gate-weighted squared output change: Q2_0 vs NF4, Q2_0 vs bf16, NF4 vs bf16.

    ``sum_t g[t,e]**2 * ||f_a(x_t) - f_b(x_t)||**2`` over the tokens routed to
    expert e, in float64, with both codecs' fake-quant weights as C5's swap
    writes them and the expert forward in float32.
    """
    import torch

    x, index, weight = routed["x"], routed["index"], routed["weight"]
    out = {"d_q2_nf4": [], "d_q2_bf16": [], "d_nf4_bf16": []}
    with torch.no_grad():
        for expert in range(module.gate_up_proj.shape[0]):
            weights = {"bf16": (module.gate_up_proj[expert], module.down_proj[expert])}
            for name in CODECS:
                weights[name] = tuple(fake_quant(w, name) for w in weights["bf16"])
            token, slot = torch.nonzero(index == expert, as_tuple=True)
            sums = dict.fromkeys(out, 0.0)
            for start in range(0, len(token), TOKEN_CHUNK):
                part = token[start:start + TOKEN_CHUNK]
                xs = x[part].to(device)
                g = weight[part, slot[start:start + TOKEN_CHUNK]].to(device)[:, None]
                f = {name: expert_forward(module, expert, xs, *w) * g
                     for name, w in weights.items()}
                for key, (a, b) in (("d_q2_nf4", ("q2_0", "nf4")),
                                    ("d_q2_bf16", ("q2_0", "bf16")),
                                    ("d_nf4_bf16", ("nf4", "bf16"))):
                    sums[key] += (f[a] - f[b]).double().pow(2).sum().item()
            for key in out:
                out[key].append(sums[key])
    return out


def forward_check(module: torch.nn.Module, routed: dict[str, torch.Tensor], device: str,
                  tokens: int = 64) -> float:
    """Relative difference between the module's own forward and ``expert_forward`` summed
    over the routed slots, on the first ``tokens`` captured tokens of a layer."""
    import torch

    x = routed["x"][:tokens].to(device)
    index = routed["index"][:tokens].to(device)
    weight = routed["weight"][:tokens].to(device)
    with torch.no_grad():
        reference = module(x, index, weight.to(x.dtype)).float()
        mine = torch.zeros_like(reference)
        for slot in range(index.shape[1]):
            for expert in index[:, slot].unique().tolist():
                rows = index[:, slot] == expert
                mine[rows] += expert_forward(module, expert, x[rows]) * weight[rows, slot, None]
    return ((reference - mine).norm() / mine.norm().clamp_min(1e-30)).item()


#: ``forward_check`` limit: the module's bf16 forward against ``expert_forward`` in float32.
FORWARD_CHECK_LIMIT = 0.05


def select(
    model: torch.nn.Module, examples: Sequence[dict[str, Any]], device: str,
) -> dict[str, Any]:
    """Routing statistics and the sensitivity proxy of every layer, on ``model`` as given."""
    start = time.perf_counter()
    routed = capture_routing(model, examples, device)
    capture_s = time.perf_counter() - start
    modules = experts_modules(model)
    layers, checks = [], {}
    start = time.perf_counter()
    for layer, module in modules.items():
        n_experts = int(module.gate_up_proj.shape[0])
        entry = {"layer": layer, **traffic(routed[layer]["index"], routed[layer]["weight"],
                                           n_experts)}
        entry.update(layer_sensitivity(module, routed[layer], device))
        layers.append(entry)
        checks[layer] = forward_check(module, routed[layer], device)
    tokens = int(routed[next(iter(routed))]["x"].shape[0])
    top_k = int(routed[next(iter(routed))]["index"].shape[1])
    return {
        "layers": layers,
        "tokens": tokens,
        "top_k": top_k,
        "forward_check": {str(layer): value for layer, value in checks.items()},
        "phases_s": {"capture": capture_s, "sensitivity": time.perf_counter() - start},
    }


def write_maps(out_dir: Path, layers: Sequence[dict[str, Any]], extra: dict[str, Any]
               ) -> dict[str, Any]:
    """``maps/<arm>.json`` for every arm; their summary for ``selection.json``."""
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm, (rule, fraction) in ARM_SPECS.items():
        record = {**build_map(layers, rule, fraction), **extra}
        path = maps_dir / f"{arm}.json"
        path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        summary[arm] = {"file": f"maps/{arm}.json", "map_sha256": record["map_sha256"],
                        "q2_0_per_layer": record["q2_0_per_layer"]}
    return summary


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="model id or local path")
    source.add_argument("--fixture", choices=("granite",),
                        help="C5's tiny random-init granitemoe, labelled synthetic")
    parser.add_argument("--revision", help="commit to load; required with --model")
    parser.add_argument("--data-dir", required=True,
                        help="output of c5_data.py; only train.jsonl is read")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", required=True, help="writes selection.json and maps/")
    parser.add_argument("--calibration-rows", type=int,
                        help=f"first N train rows (fixture only; a real model uses "
                             f"{CALIBRATION_ROWS})")
    args = parser.parse_args(argv)
    if args.model and not args.revision:
        parser.error("--model needs --revision: the record must name the exact weights")
    if args.model and args.calibration_rows:
        parser.error("--calibration-rows is for the fixture only")
    if args.fixture and not args.calibration_rows:
        args.calibration_rows = 32
    return args


def main(argv: list[str] | None = None) -> int:
    import torch

    args = _parse(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    train_path = Path(args.data_dir) / "train.jsonl"
    rows, train_sha = c5_train.load_split(train_path, None)
    calibration = rows[:args.calibration_rows or CALIBRATION_ROWS]
    calibration_sha = c5_data.split_digest(calibration)
    if not args.fixture and (train_sha not in c5_train.TRAIN_SHA256
                             or calibration_sha != CALIBRATION_SHA256):
        raise SystemExit(f"calibration data does not match the rule: train {train_sha}, "
                         f"calibration {calibration_sha}")

    tokenizer = c5_train.load_tokenizer(args)
    examples = [c5_train.encode_example(row, tokenizer, c5_train.MAX_LENGTH)
                for row in calibration]
    start = time.perf_counter()
    model, info = c5_train.load_model(args, tokenizer)
    load_s = time.perf_counter() - start
    if info["synthetic"]:
        console.print(f"  {info['note']}")
    before = swap.fingerprints(model)
    with torch.no_grad():
        result = select(model, examples, args.device)
    unchanged = swap.fingerprints(model) == before
    layers = result.pop("layers")
    stats_sha = stats_sha256(layers)
    calibration_record = {
        "rows": len(calibration),
        "sha256": calibration_sha,
        "train_sha256": train_sha,
        "split": "train.jsonl, first rows in file order; the held-out panel is never read",
        "row_ids": [row["row"] for row in calibration],
        "tokens": result.pop("tokens"),
    }
    map_extra = {"model": {"id": info["id"], "revision": info["revision"]},
                 "synthetic": info["synthetic"], "calibration_sha256": calibration_sha}
    maps = write_maps(out_dir, layers, map_extra)
    expected = calibration_record["tokens"] * result["top_k"]
    checks = {
        "model_unchanged_by_selection": unchanged,
        "every_token_routed_top_k": all(sum(layer["tokens"]) == expected for layer in layers),
        "forward_check_within_limit": all(
            value <= FORWARD_CHECK_LIMIT for value in result["forward_check"].values()),
        "maps_equal_budget_per_fraction": all(
            len({tuple(maps[arm_name(rule, fraction)]["q2_0_per_layer"]) for rule in RULES}) == 1
            for fraction in FRACTIONS),
    }
    record = {
        "kind": "g5_selection",
        "synthetic": info["synthetic"],
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": [sys.executable, *sys.argv] if argv is None else ["g5_select.py", *argv],
        "commit": c5_train._commit(),
        "versions": c5_train._versions(),
        "device": args.device,
        "gpu": c5_train.gpu_info(args.device),
        "box_stamp": l2l_box.box_stamp(),
        "harness": {"sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "c5_train_sha256": c5_train._sha256_file(Path(c5_train.__file__)),
                    "swap_sha256": swap.harness_sha256(),
                    "codec_sha256": codec.codec_sha256(),
                    "sources": source_digests()},
        "model": info,
        "calibration": calibration_record,
        "routing": {"top_k": result.pop("top_k"), "n_layers": len(layers),
                    "n_experts": len(layers[0]["tokens"]),
                    "router": "the experts module's own inputs (top_k_index, top_k_weights)"},
        "sensitivity": {
            "proxy": "d_q2_nf4[e] = sum over tokens t routed to e of "
                     "g[t,e]^2 * ||f_e^Q2_0(x_t) - f_e^NF4(x_t)||^2, float64 sums",
            "fake_quant": "q2_0_codec quantize, dequantize, cast to the model dtype "
                          "(c5_expert_swap's per-slice swap)",
        },
        "phases_s": {"load": load_s, **result.pop("phases_s")},
        "forward_check": result.pop("forward_check"),
        "checks": checks,
        "layers": layers,
        "stats_sha256": stats_sha,
        "correlation": correlation_report(layers),
        "maps": maps,
    }
    (out_dir / "selection.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    label = "SYNTHETIC " if info["synthetic"] else ""
    corr = record["correlation"]["mean_spearman_tokens_d"]
    console.print(f"{label}selection: {len(calibration)} rows, {calibration_record['tokens']} "
                  f"tokens, calibration sha256 {calibration_sha}")
    console.print(f"  mean Spearman(traffic, d_q2_nf4) over layers: "
                  f"{'n/a' if corr is None else f'{corr:.3f}'}")
    for arm, entry in maps.items():
        console.print(f"  {arm}: map {entry['map_sha256'][:12]}, Q2_0 per layer "
                      f"{entry['q2_0_per_layer'][0]}")
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        console.print(f"[red]selection checks failed[/]: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
