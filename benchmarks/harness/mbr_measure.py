"""Measurement runner pieces for the exact-MBR campaign on Kanana (harness only, not shipped).

Pure logic that does not need a GPU: the four windows, the bit ladder, the mode guard (speed only
in normal mode, bits only in deterministic mode), OOM turned into a table row with its peaks,
MBPP-train records with a dataset hash, and the median of timed steps after warm-up. The GPU
driver calls these around ``L2LStep.run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

WINDOWS: Tuple[Tuple[int, int], ...] = ((8, 64), (8, 512), (64, 64), (64, 512))
BIT_LADDER: Tuple[int, ...] = (1, 2, 4, 8)
A0_MAX_K = 8
PEAK_KEYS = ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")
# LoRA targets of the Kanana campaign. Other models pass --lora-targets.
DEFAULT_LORA_TARGETS: Tuple[str, ...] = ("q_proj", "kv_b_proj", "o_proj")


def parse_lora_targets(text: Optional[str]) -> Tuple[str, ...]:
    """Comma-separated module suffixes from --lora-targets; the Kanana list when absent."""
    if text is None:
        return DEFAULT_LORA_TARGETS
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("--lora-targets needs at least one module name")
    return names


def parse_k_list(text: str) -> Tuple[int, ...]:
    """Comma-separated k values of a bits ladder, every one positive."""
    ks = tuple(int(part) for part in text.split(",") if part.strip())
    if not ks or any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("--ladder-ks takes positive comma-separated integers")
    return ks


def parse_arm_list(text: str) -> Tuple[str, ...]:
    """Comma-separated arm names, each a known arm."""
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    known = ("A0", "B0", "C", SEGMENTED_ARM, DECOUPLED_SEGMENTED_ARM, *LEVER_ARMS)
    if not names or any(name not in known for name in names):
        raise argparse.ArgumentTypeError(f"--arms takes names of {', '.join(known)}")
    return names


class MixedModeError(RuntimeError):
    """Raised when a measurement kind is requested in the wrong torch mode."""


def check_mode(kind: str, *, deterministic: bool) -> str:
    """The mode label for ``kind`` ("speed" or "bits"); a mixed mode is refused."""
    if kind == "speed" and deterministic:
        raise MixedModeError("speed must be measured in normal mode, not deterministic")
    if kind == "bits" and not deterministic:
        raise MixedModeError("bits must be measured in deterministic mode, not normal")
    if kind not in ("speed", "bits"):
        raise ValueError(f"unknown measurement kind {kind!r}")
    return "deterministic" if deterministic else "normal"


def arms_for(*, k: int) -> Tuple[str, ...]:
    """A0 only exists for k <= 8 (its micro-batches no longer fit above that)."""
    return ("A0", "B0", "C") if k <= A0_MAX_K else ("B0", "C")


def reference_arm(k: int) -> str:
    """The arm a bits ladder compares against at ``k``: A0 up to ``A0_MAX_K``, C above it, where
    A0 does not run (ticket 04)."""
    return "A0" if k <= A0_MAX_K else "C"


def run_guarded(run: Callable[[], Dict[str, Any]], *, peaks: Callable[[], Dict[str, Any]],
                is_oom: Callable[[BaseException], bool]) -> Dict[str, Any]:
    """Run one window; an OOM becomes ``status: OOM`` with the peaks seen, other errors raise."""
    try:
        out = dict(run())
        status = "ok"
    except Exception as error:  # noqa: BLE001 - re-raised unless it is an OOM
        if not is_oom(error):
            raise
        out, status = {"error": str(error).splitlines()[0][:200]}, "OOM"
    out.update(peaks())
    out["status"] = status
    return out


def make_row(*, arm: str, k: int, seq: int, kind: str, deterministic: bool,
             result: Dict[str, Any], dataset_sha256: str) -> Dict[str, Any]:
    """One report row; the mode is always written and checked against the kind."""
    row = {"arm": arm, "k": k, "seq": seq, "kind": kind,
           "mode": check_mode(kind, deterministic=deterministic),
           "dataset_sha256": dataset_sha256}
    row.update(result)
    row.setdefault("status", "ok")
    return row


def median_after_warmup(step_seconds: Sequence[float]) -> float:
    """Median of the steps after the first (warm) one."""
    if len(step_seconds) < 2:
        raise ValueError("need a warm-up step and at least one timed step")
    return statistics.median(step_seconds[1:])


def mbpp_records(rows: Sequence[Dict[str, Any]], tokenizer: Any, *, seq: int) -> Dict[str, Any]:
    """Right-truncated, padded token rows of MBPP text + code, and the sha256 of all of them.

    The hash covers the final padded ids and the sequence length, so any change of the text,
    the tokenizer or ``seq`` changes it."""
    pad = int(tokenizer.eos_token_id or 0)
    input_ids: List[List[int]] = []
    for row in rows:
        text = f"{str(row['text']).strip()}\n{str(row['code']).strip()}"
        ids = list(tokenizer(text, add_special_tokens=True)["input_ids"])[:seq]
        input_ids.append(ids + [pad] * (seq - len(ids)))
    digest = hashlib.sha256(json.dumps({"seq": seq, "ids": input_ids}).encode()).hexdigest()
    return {"input_ids": input_ids, "sha256": digest, "seq": seq}


def take_window(input_ids: Sequence[Sequence[int]], *, k: int) -> List[List[int]]:
    """``k`` rows, cycling through the dataset when it has fewer than ``k``."""
    return [list(input_ids[i % len(input_ids)]) for i in range(k)]


# --------------------------------------------------------------------------
# Arms, plans and verdicts
# --------------------------------------------------------------------------
def arm_options(arm: str, *, k: int) -> Dict[str, Any]:
    """``L2LStep.run`` keywords of an arm: A0 plain L2L, B0 route-preserving MBR, C B0 plus the
    ordered backward (the exact GEMM joins C with ticket 02), D C's decoupled sibling."""
    if arm not in ("A0", "B0", "C", SEGMENTED_ARM, DECOUPLED_SEGMENTED_ARM, *LEVER_ARMS):
        raise ValueError(f"unknown arm {arm!r}")
    if arm in ("A0", "B0", "C") and arm not in arms_for(k=k):
        raise ValueError(f"arm {arm} does not exist at k={k} (A0 only at k<={A0_MAX_K})")
    if arm == "A0":
        return {}
    if arm in (DECOUPLED_ARM, DECOUPLED_SEGMENTED_ARM):
        # C's step without the ordered backward; the subwindow orders it. DS is D with the
        # segmented native GEMM, the diagnostic that separates D's forward drift from its
        # backward's (ticket 17).
        return {"route_preserving_mbr": True, "mbr_group_size": k,
                "mbr_decoupled_subwindow": True}
    options: Dict[str, Any] = {"route_preserving_mbr": True, "mbr_group_size": k}
    if arm in ("C", SEGMENTED_ARM) or arm in LEVER_ARMS:
        options["mbr_ordered_backward"] = True
    return options


# Lever arms of ticket 02, each stacked on the one before: C is B0 plus the ordered backward, E
# adds the exact expert GEMM (``lt-fused``), F the async lease, G the light row gathers. They run
# the step options of C; the levers are switched outside ``L2LStep.run`` (``arm_runtime``). D is
# not part of that stack: it is C's decoupled sibling and carries its own step options.
LEVER_ARMS: Dict[str, Dict[str, Any]] = {
    "E": {"expert_gemm": "lt-fused", "async_lease": False, "fast_row_gather": False,
          "expert_loader": "async_lease"},
    "F": {"expert_gemm": "lt-fused", "async_lease": True, "fast_row_gather": False,
          "expert_loader": "async_lease"},
    "G": {"expert_gemm": "lt-fused", "async_lease": True, "fast_row_gather": True,
          "expert_loader": "async_lease"},
    # Ticket 05: the same stack as G with the expert pool fed by the compiled two-slot stream.
    "H": {"expert_gemm": "lt-fused", "async_lease": True, "fast_row_gather": True,
          "expert_loader": "cpp_async_2slot"},
    # Ticket 06: D is C's sibling, not a step of the E..H stack. Its decoupled backward runs
    # its own expert backward, so it needs C's native GEMM and C's lease (no lt-fused).
    "D": {"expert_gemm": "native", "async_lease": False, "fast_row_gather": False,
          "expert_loader": "async_lease"},
}
LEVER_PREVIOUS = {"E": "C", "F": "E", "G": "F", "H": "G"}
# The local gate of ticket 05 compares the cpp loader arm with the Python lease on the windows
# that fit 4 GiB, and asks for two of the three (the main track asks three of four).
LOADER_WINDOWS: Tuple[Tuple[int, int], ...] = ((8, 64), (8, 512), (64, 64))
LOADER_MIN_WINDOWS = 2
LOADER_PREVIOUS, LOADER_ARM = "G", "H"
LEVER_MIN_GAIN = 0.03
LEVER_MIN_WINDOWS = 3
LEVER_T_CRITICAL = 2.78  # two-sided 5% with 4 degrees of freedom (5 blocks)
VRAM_TOLERANCE_BYTES = 64 * 2**20  # reserved and NVML wobble; allocated is compared exactly

# Ticket 06: decoupled MBR. It is C's sibling: the same route-preserving step without the
# ordered backward, with ``mbr_decoupled_subwindow`` doing the ordered recompute itself.
DECOUPLED_ARM, DECOUPLED_PREVIOUS = "D", "C"
# Ticket 17: D with ``expert_gemm="segmented"``. D's forward is C's grouped native GEMM, whose
# row-count kernel choice is the forward divergence (ticket 14); segmented keeps the reference
# row count. If DS equals A0, the drift was the forward, not D's own decoupled backward. A
# diagnostic arm: it is not part of the lever stack and stays out of every ladder.
DECOUPLED_SEGMENTED_ARM = "DS"
DECOUPLED_SEGMENTED_RUNTIME: Dict[str, Any] = {"expert_gemm": "segmented", "async_lease": False,
                                               "fast_row_gather": False,
                                               "expert_loader": "async_lease"}
# The three windows that fit 4 GiB for C; (64,512) is the large-window case of the switch.
DECOUPLED_WINDOWS: Tuple[Tuple[int, int], ...] = ((8, 64), (8, 512), (64, 64))
DECOUPLED_MIN_WINDOWS = 2  # the local gate of ticket 06 asks for two of the three fitting ones

# Ticket 14 tail: the price of the exact path. S is C's ordered step with
# ``expert_gemm="segmented"`` (native cuBLAS per micro-batch), so the bit-preserving mode is
# measured against C (what segmentation costs) and against A0 (how much of C's speed is left).
SEGMENTED_ARM, SEGMENTED_PREVIOUS = "S", "C"
SEGMENTED_WINDOWS: Tuple[Tuple[int, int], ...] = ((8, 64), (8, 512), (64, 64))
SEGMENTED_RUNTIME: Dict[str, Any] = {"expert_gemm": "segmented", "async_lease": False,
                                    "fast_row_gather": False, "expert_loader": "async_lease"}


def ladder_arms(k: int) -> Tuple[str, ...]:
    """The bits ladder at ``k``: A0 with C and every lever up to k=8; C and the levers above it,
    where A0 does not run."""
    return ("A0", "C", *LEVER_ARMS) if k <= A0_MAX_K else ("C", *LEVER_ARMS)


def arm_runtime(arm: str) -> Dict[str, Any]:
    """The levers outside ``L2LStep.run``: expert GEMM mode, async lease, fast row gather, and
    which loader feeds the expert pool (``async_lease`` unless the arm is H)."""
    if arm == SEGMENTED_ARM:
        return dict(SEGMENTED_RUNTIME)
    if arm == DECOUPLED_SEGMENTED_ARM:
        return dict(DECOUPLED_SEGMENTED_RUNTIME)
    return dict(LEVER_ARMS.get(arm, {"expert_gemm": "native", "async_lease": False,
                                     "fast_row_gather": False,
                                     "expert_loader": "async_lease"}))


def _pair_speed_plan(previous: str, arm: str, windows: Sequence[Tuple[int, int]], *,
                     blocks: int) -> List[Dict[str, Any]]:
    """Normal-mode A-B-B-A blocks of two hands over ``windows``: the two-arm order of
    ``METHOD.md``, where each hand sits at mean position 2.5 so a drift of the host cancels."""
    return [{"arm": hand, "k": k, "seq": seq, "block": block, "kind": "speed",
             "deterministic": False}
            for block in range(blocks) for k, seq in windows
            for hand in (previous, arm, arm, previous)]


def loader_speed_plan(*, blocks: int = 5) -> List[Dict[str, Any]]:
    """Normal-mode G-H-H-G blocks over the three fitting windows (ticket 05).

    G is the current stack with the Python lease (A), H the same stack with ``cpp_async_2slot``
    (B). The ticket names the three-arm ``A-B-C-C-B-A``; that order needs three hands and this
    gate has two.
    """
    return _pair_speed_plan(LOADER_PREVIOUS, LOADER_ARM, LOADER_WINDOWS, blocks=blocks)


def decoupled_speed_plan(*, blocks: int = 5) -> List[Dict[str, Any]]:
    """Normal-mode C-D-D-C blocks over the three fitting windows (ticket 06).

    C is the ordered arm (A), D decoupled MBR (B): the A-B-B-A form of the ticket's
    ``A-B-C-C-B-A`` for a two-hand gate.
    """
    return _pair_speed_plan(DECOUPLED_PREVIOUS, DECOUPLED_ARM, DECOUPLED_WINDOWS, blocks=blocks)


def segmented_speed_plan(*, blocks: int = 5) -> List[Dict[str, Any]]:
    """Normal-mode A0-C-S-S-C-A0 blocks over the three fitting windows (ticket 14 tail).

    A0 is the context hand, C the ordered arm, S C with ``expert_gemm="segmented"``. The order
    is the ticket's ``A-B-C-C-B-A`` with the three hands; above ``A0_MAX_K`` A0 does not run, so
    (64,64) keeps the C-S-S-C core.
    """
    return [{"arm": arm, "k": k, "seq": seq, "block": block, "kind": "speed",
             "deterministic": False}
            for block in range(blocks) for k, seq in SEGMENTED_WINDOWS
            for arm in ("A0", "C", SEGMENTED_ARM, SEGMENTED_ARM, "C", "A0")
            if arm != "A0" or k <= A0_MAX_K]


# Ticket 16: the three candidate default hands in one campaign. C is the ordered arm, S its
# segmented sibling, E the lt-fused one. All three run the three windows that fit 4 GiB, and the
# block order is the ticket's ``A-B-C-C-B-A`` with the hands in the roles C, S, E, E, S, C, so
# every hand sits at mean position 3.5 and host drift cancels between hands.
ESC_WINDOWS: Tuple[Tuple[int, int], ...] = ((8, 64), (8, 512), (64, 64))
ESC_HANDS: Tuple[str, ...] = ("C", SEGMENTED_ARM, "E", "E", SEGMENTED_ARM, "C")


def esc_speed_plan(*, blocks: int = 5, start_block: int = 0) -> List[Dict[str, Any]]:
    """Normal-mode C-S-E-E-S-C blocks over the three fitting windows (ticket 16).

    Six hands per window, five blocks: 90 runs. A0 has no role here (and does not exist above
    k=8, where C, S and E all do), so the palindrome is whole on every window. ``start_block``
    resumes after a gate refusal: block numbers and run indexes continue the earlier numbering,
    so the records of both parts sort together.
    """
    per_block = len(ESC_WINDOWS) * len(ESC_HANDS)
    return [{"arm": arm, "k": k, "seq": seq, "block": block, "kind": "speed",
             "deterministic": False, "index": block * per_block + position}
            for block in range(start_block, start_block + blocks)
            for position, (k, seq, arm) in enumerate(
                (k, seq, arm) for k, seq in ESC_WINDOWS for arm in ESC_HANDS)]


# Ticket 17: G (E + async_lease + fast_row_gather) against E and C on the two k=8 windows.
# The palindrome A-B-C-C-B-A with the three hands in the roles C, E, G.
CEG_WINDOWS: Tuple[Tuple[int, int], ...] = ((8, 64), (8, 512))
CEG_HANDS: Tuple[str, ...] = ("C", "E", "G", "G", "E", "C")


def ceg_speed_plan(*, blocks: int = 5, start_block: int = 0) -> List[Dict[str, Any]]:
    """Normal-mode C-E-G-G-E-C blocks over the two windows (ticket 17).

    Six hands per window, five blocks: 60 runs. Every hand sits at the block centre, so a drift
    of the host cancels between hands. ``start_block`` resumes after a gate refusal, like
    ``esc_speed_plan``.
    """
    per_block = len(CEG_WINDOWS) * len(CEG_HANDS)
    return [{"arm": arm, "k": k, "seq": seq, "block": block, "kind": "speed",
             "deterministic": False, "index": block * per_block + position}
            for block in range(start_block, start_block + blocks)
            for position, (k, seq, arm) in enumerate(
                (k, seq, arm) for k, seq in CEG_WINDOWS for arm in CEG_HANDS)]


def decoupled_segmented_bit_plan(*, seq: int = 64,
                                 ks: Sequence[int] = (2,)) -> List[Dict[str, Any]]:
    """Deterministic A0-DS pairs at ``ks`` for the D+segmented diagnostic (ticket 17).

    DS keeps D's decoupled step and switches the expert GEMM to segmented, so the cell answers
    whether D's drift was its forward's grouped native GEMM. A0 is the plain reference; the
    default cell is k=2 on S=64.
    """
    return [{"arm": arm, "k": k, "seq": seq, "block": 0, "kind": "bits",
             "deterministic": True}
            for k in ks for arm in ("A0", DECOUPLED_SEGMENTED_ARM)]


def large_window_enabled(*, previous_fits: bool) -> bool:
    """The large-window switch: decoupled only where the ordered arm does not fit (D2, ticket 06).

    Where C fits, C stays the default even if D is faster; D is a reason to take the lever only
    when it removes C's OOM.
    """
    return not previous_fits


def lever_speed_plan(*, blocks: int = 1) -> List[Dict[str, Any]]:
    """Normal-mode runs C-E-F-G-G-F-E-C per window, block-major like ``speed_plan``."""
    return [{"arm": arm, "k": k, "seq": seq, "block": block, "kind": "speed",
             "deterministic": False}
            for block in range(blocks) for k, seq in WINDOWS
            for arm in ("C", "E", "F", "G", "G", "F", "E", "C")]


def lever_bit_plan(*, seq: int, ks: Sequence[int] = BIT_LADDER) -> List[Dict[str, Any]]:
    """Deterministic ladder over ``ks`` (default k = 1, 2, 4, 8) for the reference arm, C and every
    lever arm. Above ``A0_MAX_K`` the reference is C, so A0 is not planned there."""
    return [{"arm": arm, "k": k, "seq": seq, "block": 0, "kind": "bits", "deterministic": True}
            for k in ks for arm in ladder_arms(k)]


def _vram_not_worse(lever: Dict[str, Any], previous: Dict[str, Any]) -> Optional[bool]:
    """True/False for a comparable pair; None when the lever relieves an OOM of the previous."""
    if previous["status"] != "ok" and lever["status"] == "ok":
        return None
    if lever["status"] != "ok" and previous["status"] == "ok":
        return False
    if (lever["peak_allocated"] or 0) > (previous["peak_allocated"] or 0):
        return False
    return all((lever[n] or 0) <= (previous[n] or 0) + VRAM_TOLERANCE_BYTES
               for n in ("peak_reserved", "peak_nvml"))


def _paired_t(gains: Sequence[float]) -> Optional[float]:
    if len(gains) < 2:
        return None
    spread = statistics.stdev(gains)
    if spread == 0:
        return float("inf") if statistics.mean(gains) > 0 else 0.0
    return statistics.mean(gains) / (spread / len(gains) ** 0.5)


def lever_gate(rows: Sequence[Dict[str, Any]], *, lever: str, previous: str,
               windows: Sequence[Tuple[int, int]] = WINDOWS,
               min_windows: int = LEVER_MIN_WINDOWS) -> Dict[str, Any]:
    """The D1 gates of one lever against the arm before it: bits against the reference arm of each
    k (A0 at k <= 8, C above it; ladder, deterministic rows), peak VRAM in every window (an OOM of
    the previous arm that the lever removes counts as a reason to take it), and speed (at least 3%
    in at least ``min_windows`` of ``windows``; with 5 or more blocks the paired t of the per-block
    gains must exceed 2.78 too). The defaults are the main-track gate; ticket 05 passes its three
    fitting windows and two of three."""
    bit_rows = [r for r in rows if r.get("kind") == "bits"]
    verdicts = ladder_verdict(bit_rows)
    # The ladder of k <= 8 stays the gate it was; a k above it counts when it has bits rows.
    bit_ks = list(BIT_LADDER) + sorted({r["k"] for r in bit_rows if r["k"] > A0_MAX_K})
    differs = [k for k in bit_ks if verdicts.get((k, lever)) != "equal"]
    speed_rows = [r for r in rows if r.get("kind") == "speed"]
    table = summarize_rows(speed_rows)
    windows_won, relief, worse, detail = 0, [], [], {}
    for k, seq in windows:
        now, before = table.get((lever, k, seq)), table.get((previous, k, seq))
        if now is None or before is None:
            continue
        verdict = _vram_not_worse(now, before)
        if verdict is None:
            relief.append((k, seq))
        elif verdict is False:
            worse.append((k, seq))
        if now["step_seconds"] is None or before["step_seconds"] is None:
            continue
        gain = 1 - now["step_seconds"] / before["step_seconds"]
        per_block: Dict[int, Dict[str, List[float]]] = {}
        for r in speed_rows:
            if (r["k"], r["seq"]) == (k, seq) and r.get("status") == "ok" \
                    and r["arm"] in (lever, previous):
                per_block.setdefault(r.get("block", 0), {}).setdefault(r["arm"], []).append(
                    r["step_seconds"])
        gains = [1 - statistics.median(v[lever]) / statistics.median(v[previous])
                 for v in per_block.values() if lever in v and previous in v]
        t_value = _paired_t(gains) if len(gains) >= 5 else None
        won = gain >= LEVER_MIN_GAIN and (t_value is None or t_value > LEVER_T_CRITICAL)
        windows_won += int(won)
        detail[(k, seq)] = {"gain": gain, "t": t_value}
    blocks = max((len({r.get("block", 0) for r in speed_rows if r["arm"] == lever}),), default=0)
    vram_pass = not worse
    return {
        "bits": {"pass": not differs, "differs_at": differs},
        "vram": {"pass": vram_pass, "worse_in": worse, "oom_relief": relief},
        "speed": {"pass": windows_won >= min_windows or (vram_pass and bool(relief)),
                  "windows_won": windows_won, "windows": detail,
                  "needed_for_vram": bool(relief) and windows_won < min_windows,
                  "t_test": "not computed (fewer than 5 blocks)" if blocks < 5 else "computed"},
    }


def speed_plan(*, blocks: int = 5) -> List[Dict[str, Any]]:
    """Normal-mode runs in A-B-C-C-B-A order (A0 left out when k > 8, which leaves B-C-C-B).

    Block-major: block 0 covers all four windows before block 1 starts, so a run cut short by
    the budget still has every window with the same number of blocks (or one more)."""
    plan: List[Dict[str, Any]] = []
    for block in range(blocks):
        for k, seq in WINDOWS:
            arms = arms_for(k=k)
            for arm in [a for a in ("A0", "B0", "C", "C", "B0", "A0") if a in arms]:
                plan.append({"arm": arm, "k": k, "seq": seq, "block": block, "kind": "speed",
                             "deterministic": False})
    return plan


def profile_plan() -> List[Dict[str, Any]]:
    """One phase-profiled normal-mode run per arm and window, apart from the timed runs so the
    instrumented step does not touch their peaks or times."""
    return [{"arm": arm, "k": k, "seq": seq, "block": 0, "kind": "speed", "deterministic": False,
             "profile": True} for k, seq in WINDOWS for arm in arms_for(k=k)]


def bit_plan(*, seq: int, ks: Sequence[int] = BIT_LADDER) -> List[Dict[str, Any]]:
    """Deterministic-mode ladder over ``ks`` (default k = 1, 2, 4, 8) for the arms of each k."""
    return [{"arm": arm, "k": k, "seq": seq, "block": 0, "kind": "bits", "deterministic": True}
            for k in ks for arm in arms_for(k=k)]


def grad_digest(grads: Dict[str, Sequence[float]]) -> str:
    """sha256 over the gradients in name order, bit-exact on the floats (``repr`` round trip)."""
    text = json.dumps({name: [float(v).hex() for v in grads[name]] for name in sorted(grads)})
    return hashlib.sha256(text.encode()).hexdigest()


def _against_reference(rows: Sequence[Dict[str, Any]],
                       same: Callable[[Dict[str, Any], Dict[str, Any]], bool],
                       ) -> Dict[Tuple[int, str], str]:
    """``equal`` / ``differs`` per (k, arm) by ``same(base_row, row)``, where the base is the
    reference arm of that k (``reference_arm``). A row that did not finish keeps its status (OOM,
    CRASH); a k without an ok reference row says ``no A0`` (k <= 8) or ``no C`` (above it)."""
    ref = {r["k"]: r for r in rows
           if r["arm"] == reference_arm(r["k"]) and r.get("status") == "ok"}
    out: Dict[Tuple[int, str], str] = {}
    for row in rows:
        name = reference_arm(row["k"])
        if row["arm"] == name:
            continue
        base = ref.get(row["k"])
        if base is None:
            out[(row["k"], row["arm"])] = f"no {name}"
        elif row.get("status") != "ok":
            out[(row["k"], row["arm"])] = row.get("status", "?")
        else:
            out[(row["k"], row["arm"])] = "equal" if same(base, row) else "differs"
    return out


def ladder_verdict(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, str], str]:
    """Verdict per (k, arm) from the gradient digest and the loss, against the reference arm of
    the same k (A0 at k <= 8, C above it)."""
    return _against_reference(rows, lambda base, row: base["grad_sha256"] == row["grad_sha256"]
                              and base["loss"] == row["loss"])


def tensor_file_name(arm: str, k: int, seq: int) -> str:
    """Name of the saved loss and LoRA grads of one bits run, under ``<out>/tensors``."""
    return f"{arm}-k{k}-s{seq}.pt"


def bit_tensor_dir(out: Any) -> Path:
    """Folder of the saved bits tensors of a sweep or run written to ``out``."""
    return Path(out) / "tensors"


def save_bit_tensors(path: Path, *, losses: Sequence[Any], grads: Dict[str, Any]) -> None:
    """Write one bits run's loss (float64, exact) and LoRA grads (as they are) to ``path``."""
    import torch

    torch.save({"loss": torch.tensor([float(x) for x in losses], dtype=torch.float64),
                "grads": {name: grad.detach().cpu() for name, grad in grads.items()}}, path)


def tensors_equal(ref: Dict[str, Any], other: Dict[str, Any]) -> bool:
    """``torch.equal`` on the loss and on every LoRA grad; the grad names must match. A NaN
    counts as different, because ``torch.equal`` never calls NaN equal to NaN."""
    import torch

    if sorted(ref["grads"]) != sorted(other["grads"]):
        return False
    if not torch.equal(ref["loss"], other["loss"]):
        return False
    return all(torch.equal(ref["grads"][name], other["grads"][name]) for name in ref["grads"])


def tensor_ladder_verdict(rows: Sequence[Dict[str, Any]], *, tensor_dir: Path,
                          ) -> Dict[Tuple[int, str], str]:
    """Verdict per (k, arm) by ``torch.equal`` on the saved tensors, against the reference arm of
    the same k (A0 at k <= 8, C above it). The reference file of each k loads once."""
    import torch

    folder = Path(tensor_dir)
    base_cache: Dict[str, Dict[str, Any]] = {}

    def base_tensors(row: Dict[str, Any]) -> Dict[str, Any]:
        name = row["tensor_file"]
        if name not in base_cache:
            base_cache[name] = torch.load(folder / name, weights_only=True)
        return base_cache[name]

    def other_tensors(row: Dict[str, Any]) -> Dict[str, Any]:
        return torch.load(folder / row["tensor_file"], weights_only=True)

    return _against_reference(rows, lambda base, row: tensors_equal(base_tensors(base),
                                                                    other_tensors(row)))


def parse_nvml_used(samples: Sequence[str]) -> int:
    """Largest ``memory.used`` sample of ``nvidia-smi --query-gpu=memory.used`` in bytes."""
    return max(int(s.split()[0]) for s in samples) * 2**20


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    """Per (arm, k, seq): median step seconds over the ok runs, max of each peak, run count.
    A cell with an OOM keeps the OOM status and its peaks; it has no step time."""
    cells: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["arm"], row["k"], row["seq"]), []).append(row)
    out: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for key, group in cells.items():
        ok = [r for r in group if r.get("status") == "ok"]
        bad = [r for r in group if r.get("status") != "ok"]

        def peak(name: str) -> Any:
            values = [r[name] for r in group if r.get(name) is not None]
            return max(values) if values else None

        out[key] = {
            "step_seconds": statistics.median(r["step_seconds"] for r in ok) if ok else None,
            "runs": len(group), "status": bad[0]["status"] if bad else "ok",
            **{name: peak(name) for name in PEAK_KEYS}}
    return out


def markdown_table(table: Dict[Tuple[str, int, int], Dict[str, Any]]) -> str:
    """The summary table for ``benchmarks/``: seconds and GiB, OOM shown in the status column."""
    def gib(value: Any) -> str:
        return "-" if value is None else f"{value / 2**30:.2f}"

    lines = ["| arm | k | S | step s | alloc GiB | reserved GiB | NVML GiB | RAM GiB | runs "
             "| status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (arm, k, seq), c in sorted(table.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        step = "-" if c["step_seconds"] is None else f"{c['step_seconds']:.1f}"
        lines.append(f"| {arm} | {k} | {seq} | {step} | {gib(c['peak_allocated'])} | "
                     f"{gib(c['peak_reserved'])} | {gib(c['peak_nvml'])} | {gib(c['peak_ram'])} | "
                     f"{c['runs']} | {c['status']} |")
    return "\n".join(lines) + "\n"


def amdahl_min_share(*, target_gain: float, local_speedup: float) -> float:
    """Share of the critical path a phase must hold so that speeding it up by ``local_speedup``
    gives ``target_gain`` overall (0.10 = +10% throughput): p = (1 - 1/(1+g)) / (1 - 1/s)."""
    if local_speedup <= 1.0:
        raise ValueError("a local speedup must be above 1")
    return (1 - 1 / (1 + target_gain)) / (1 - 1 / local_speedup)


def phase_shares(summary: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Each phase's share of the summed phase seconds (``PhaseClock.summary()`` layout)."""
    total = sum(entry["seconds"] for entry in summary.values())
    return {name: entry["seconds"] / total for name, entry in summary.items()} if total else {}


_TRACKED = ("torch", "triton", "transformers", "peft", "bitsandbytes", "safetensors")


def environment_record(lookup: Callable[[str], Any], *, git_sha: Any, ptxas: Any,
                       alloc_conf: Optional[str] = None) -> Dict[str, Any]:
    """Tool versions, the git sha and the allocator config of a run; a missing one is
    ``unknown``, and an unset allocator config is written as ``default``."""
    return {"versions": {name: lookup(name) or "unknown" for name in _TRACKED},
            "git_sha": git_sha or "unknown", "ptxas": ptxas or "unknown",
            "alloc_conf": alloc_conf or "default"}


def direct_io_record(*, requested: bool, actual: bool) -> Dict[str, bool]:
    """O_DIRECT is requested but the reader falls back to buffered reads on some filesystems
    (overlayfs); a fallback means the run measured the page cache, so it is flagged."""
    return {"requested": requested, "actual": actual, "fallback": requested and not actual}


# --------------------------------------------------------------------------
# GPU driver: one run in one fresh process, and the sweep that spawns them
# --------------------------------------------------------------------------
def _nvml_used_now() -> int:
    import subprocess

    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return parse_nvml_used(out.split("\n")[:1])


class _NvmlSampler:
    """Peak of ``nvidia-smi`` memory.used, sampled on a thread (NVML sees the whole process,
    CUDA context included, which allocated/reserved do not)."""

    def __init__(self, interval: float = 0.5) -> None:
        import threading

        self.peak, self._stop, self._interval = 0, threading.Event(), interval
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, _nvml_used_now())
            except Exception:  # noqa: BLE001 - nvidia-smi missing or busy: skip the sample
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> "_NvmlSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()


def _is_cuda_oom(error: BaseException) -> bool:
    import torch

    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def run_one(args: Any) -> Dict[str, Any]:  # pragma: no cover - needs the model and a CUDA host
    """One window of one arm. Mirrors ``oc_arm_run.run_arm``: streamed model from a prepared
    ``per_expert`` cache, ``L2LStep.run`` over MBPP-train rows, grad digest of the LoRA grads."""
    import os
    import time

    from benchmarks.harness import b5_real_moe, l2l_mistral_stand

    check_mode(args.kind, deterministic=args.deterministic)
    if args.deterministic:
        l2l_mistral_stand.set_deterministic_env()
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, label_tokens
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    options = arm_options(args.arm, k=args.k)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if args.math_attention:  # the math SDPA backend only: the others are not deterministic
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.source)
    rows = list(load_dataset("google-research-datasets/mbpp", "full", split="train"))
    data = mbpp_records(rows, tokenizer, seq=args.seq)
    cuda = torch.cuda.is_available() and args.device != "cpu"
    device = "cuda" if cuda else "cpu"
    dtype = torch.bfloat16 if cuda else torch.float32
    b5_real_moe._reset_peak_rss()
    vram_limit = float(os.environ.get("MBR_VRAM_LIMIT_GIB", "0"))
    if cuda and vram_limit:
        # Emulates a smaller card (the ticket's 8 GiB) on a bigger one: the allocator raises an
        # OOM at the limit. NVML still counts the CUDA context, which the limit does not cover.
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, vram_limit * 2**30 / total))
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def peaks() -> Dict[str, Any]:
        return {"peak_allocated": torch.cuda.max_memory_allocated() if cuda else None,
                "peak_reserved": torch.cuda.max_memory_reserved() if cuda else None,
                "peak_nvml": sampler.peak if cuda else None,
                "peak_ram": b5_real_moe._peak_rss()}

    runtime_levers = arm_runtime(args.arm)
    gemm_state: Dict[str, Any] = {}

    def work() -> Dict[str, Any]:
        from contextlib import ExitStack

        from benchmarks.harness.p72_invariant_expert_gemm import (
            async_lease_context,
            fast_row_gather_context,
        )
        from soup_cli.utils.expert_gemm import expert_gemm_context

        # The counter wraps the module-level ``F.linear`` before the expert-GEMM context captures
        # it: ``segmented`` keeps the reference it finds and calls it once per segment, so a count
        # installed inside the context would see one call per expert visit instead of one per
        # micro-batch segment.
        counted = bool(getattr(args, "count", False))
        original_linear = torch.nn.functional.linear
        counter = None
        if counted:
            from benchmarks.harness import mbr_count

            counter = mbr_count.ExpertCounter()
            torch.nn.functional.linear = counter.wrap(original_linear)
        try:
            with ExitStack() as levers:  # entered before the model is built: class defaults
                levers.enter_context(fast_row_gather_context(runtime_levers["fast_row_gather"]))
                levers.enter_context(async_lease_context(runtime_levers["async_lease"]))
                gemm_state["plan"] = levers.enter_context(
                    expert_gemm_context(runtime_levers["expert_gemm"]))
                return build_and_run(counter)
        finally:
            torch.nn.functional.linear = original_linear

    def build_and_run(counter: Any = None) -> Dict[str, Any]:
        index = read_shard_index(args.cache)
        lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                          bias="none", target_modules=list(args.lora_targets),
                          task_type=TaskType.CAUSAL_LM)
        with b5_real_moe._meta_construction():
            model, runtime = build_streamed_model(
                model_id=args.source, shard_dir=args.cache, index=index, lora_config=lora,
                device=device, dtype="bfloat16" if cuda else "float32", buffers=2, pin=False,
                seed=0, quant="nf4" if cuda else "none", double_quant=False, tier="ram",
                read_ahead=1, two_read_experts=True, grouped_expert_routes=True,
                expert_loader=runtime_levers["expert_loader"],
                **({"expert_transfer": "packed"} if cuda else {}))
        try:
            model.train()
            ids = [torch.tensor([r], device=device) for r in take_window(data["input_ids"],
                                                                           k=args.k)]
            stepper = L2LStep(model, runtime, capture_layer_call(model, ids[0]))
            store = ActivationStore(
                n_layers=len(stepper.layers), k=args.k,
                chunk_shape=(1, args.seq, int(stepper.causal.config.hidden_size)),
                dtype=dtype, device=device, pin=cuda)
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            seconds: List[float] = []
            losses: List[Any] = []
            counted = counter is not None
            snapshots: List[Dict[str, Any]] = []
            for _ in range(args.steps):
                for p in params.values():
                    p.grad = None
                if cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                losses = stepper.run(ids, store, n_items=label_tokens(ids),
                                     labels=[i.clone() for i in ids], **options)
                if cuda:
                    torch.cuda.synchronize()
                seconds.append(time.perf_counter() - started)
                if counted:
                    snapshots.append(counter.snapshot())
                    counter.reset()
            grad_tensors = {name: p.grad for name, p in sorted(params.items())
                            if p.grad is not None}
            grads = {name: g.detach().float().flatten().cpu().tolist()
                     for name, g in grad_tensors.items()}
            tensor_name = None
            if args.kind == "bits":  # torch.equal on these, in the parent, by tensor_ladder_verdict
                tensor_name = tensor_file_name(args.arm, args.k, args.seq)
                bit_dir = bit_tensor_dir(args.out)
                bit_dir.mkdir(parents=True, exist_ok=True)
                save_bit_tensors(bit_dir / tensor_name, losses=losses, grads=grad_tensors)
            out: Dict[str, Any] = {
                "tensor_file": tensor_name,
                "attention": "math" if args.math_attention else "default",
                "expert_loader": runtime_levers["expert_loader"],
                "step_seconds": median_after_warmup(seconds), "all_step_seconds": seconds,
                "loss": float(sum(float(x) for x in losses)), "levers": runtime_levers,
                "grad_sha256": grad_digest(grads), "vram_limit_gib": vram_limit or None,
                "direct_io": direct_io_record(
                    requested=True,
                    actual=all(r.direct for r in getattr(runtime, "expert_readers", ()))),
                "environment": _environment()}
            if counted:
                from benchmarks.harness import mbr_count

                phases: Dict[str, int] = {}
                for snapshot in snapshots:
                    for phase, count in snapshot["phases"].items():
                        phases[phase] = phases.get(phase, 0) + int(count)
                out["expert_gemm_calls"] = {
                    "per_step": [int(snapshot["calls"]) for snapshot in snapshots],
                    "median": mbr_count.median_calls(snapshots),
                    "rows": mbr_count.merged_rows(snapshots),
                    "phases": phases, "steps": len(snapshots)}
            # lt-fused reports visits on the fused path and its algorithm table; segmented only
            # reports its counters, and native has no plan at all.
            out.update(gemm_plan_record(gemm_state.get("plan")))
            if getattr(args, "trace", False):
                out["trace"] = _trace_step(stepper, store, ids, options, cuda)
            if args.profile:
                out["phases"] = _profile_step(stepper, store, ids, options, cuda)
            return out
        finally:
            runtime.close()

    class _Null:
        peak = 0

    sampler: Any = _NvmlSampler() if cuda else _Null()
    with (sampler if cuda else _NullContext()):
        result = run_guarded(work, peaks=peaks, is_oom=_is_cuda_oom)
    return make_row(arm=args.arm, k=args.k, seq=args.seq, kind=args.kind,
                    deterministic=args.deterministic, result=result,
                    dataset_sha256=data["sha256"])


def gemm_plan_record(plan: Any) -> Dict[str, Any]:
    """The expert GEMM context's counters for a result row.

    ``lt-fused`` carries the fused-visit statistics and its cuBLASLt algorithm table;
    ``segmented`` is a plain ``SegmentState`` without a table, and ``native`` has no plan."""
    if plan is None:
        return {}
    record: Dict[str, Any] = {"fused_visits": dict(getattr(plan, "direct_stats", {}))}
    table = getattr(plan, "table", None)
    if table is not None:
        record["algo_table"] = dict(table.stats)
    for counter in ("linear_rows", "mul_rows", "passthrough_calls"):
        value = getattr(plan, counter, None)
        if value is not None:
            record[counter] = int(value)
    return record


def _environment() -> Dict[str, Any]:  # pragma: no cover - needs the host's tools
    import os
    import shutil
    import subprocess
    from importlib import metadata

    def lookup(name: str) -> Any:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    def run(command: List[str]) -> Any:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True,
                                  timeout=20).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    ptxas = run(["ptxas", "--version"]) if shutil.which("ptxas") else None
    return environment_record(lookup, git_sha=run(["git", "rev-parse", "HEAD"]), ptxas=ptxas,
                              alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


def _profile_step(stepper: Any, store: Any, ids: List[Any], options: Dict[str, Any],
                  cuda: bool) -> Dict[str, Any]:  # pragma: no cover - needs the model
    """One CUDA-synchronised phase-profiled step (PR #72 ``PhaseClock``). Instrumented time is
    longer than a real step, so only the shares are meaningful, not the seconds."""
    from benchmarks.harness.p72_phase_profiler import PhaseClock, install
    from soup_cli.utils.l2l_step import label_tokens

    clock = PhaseClock("cuda" if cuda else "cpu")
    with install(clock):
        stepper.run(ids, store, n_items=label_tokens(ids), labels=[i.clone() for i in ids],
                    **options)
    summary = clock.summary()
    return {"summary": summary, "shares": phase_shares(summary),
            "amdahl_min_share_2x_for_10pct": amdahl_min_share(target_gain=0.10, local_speedup=2.0)}


def _trace_step(stepper: Any, store: Any, ids: List[Any], options: Dict[str, Any],
                cuda: bool) -> Dict[str, Any]:  # pragma: no cover - needs the model
    """One step under ``torch.profiler`` with CUDA activities only: the device timeline.

    A CPU-activity trace with shapes of this step does not fit the host: 7.9 GiB of RSS at (8,64)
    and OOM-killed, while a CUDA-only profile is 5.6 GiB there. At (64,64) even the CUDA-only
    profile of the whole step was OOM-killed at about 1.1M kernel events. So no CPU activity and
    no shapes: the summary is the device timeline only (kernel and busy seconds, the exposed gap,
    launches, copies and the top kernels). The CPU side comes from py-spy, the routed-expert
    counters from ``mbr_count``. The profiled step is instrumented, so its seconds are not a speed
    number."""
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    from benchmarks.harness import mbr_trace
    from soup_cli.utils.l2l_step import label_tokens

    activities = [ProfilerActivity.CUDA] if cuda else [ProfilerActivity.CPU]
    if cuda:
        torch.cuda.synchronize()
    with profile(activities=activities) as profiler:
        started = time.perf_counter()
        stepper.run(ids, store, n_items=label_tokens(ids), labels=[i.clone() for i in ids],
                    **options)
        if cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
    return mbr_trace.step_summary(list(profiler.events()), wall)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def child_run_args(args: Any, item: Dict[str, Any]) -> List[str]:
    """argv of one ``run`` child for a plan item. Every model option of the parent is forwarded,
    so a sweep measures the model the parent was asked for."""
    argv = ["run", "--arm", item["arm"], "--k", str(item["k"]), "--seq", str(item["seq"]),
            "--kind", item["kind"], "--source", args.source, "--cache", args.cache,
            "--steps", str(args.steps), "--out", str(args.out), "--device", args.device,
            "--lora-r", str(args.lora_r), "--lora-targets", ",".join(args.lora_targets)]
    if args.tokenizer:
        argv += ["--tokenizer", args.tokenizer]
    if item["deterministic"]:
        argv.append("--deterministic")
    if item.get("profile"):
        argv.append("--profile")
    if item.get("trace"):
        argv.append("--trace")
    if getattr(args, "count", False):
        argv.append("--count")
    if getattr(args, "math_attention", False):
        argv.append("--math-attention")
    return argv


ABBA_ARMS = {"A": "A0", "B": "C"}
DEFAULT_LABEL = "RTX 3050 Laptop, real"
ARM_CHECKS: Tuple[str, ...] = ("suspend", "power", "foreign_reads")


def abba_plan(*, blocks: int = 5, start_block: int = 0) -> List[Dict[str, Any]]:
    """A0 against C in A-B-B-A blocks (``METHOD.md``, the repeats rule), on the windows where
    A0 runs (k <= ``A0_MAX_K``). Block-major, like ``speed_plan``.

    ``start_block`` resumes a pilot: block numbers and run indexes continue the numbering of
    the earlier blocks, so the records of both parts sort together."""
    from benchmarks.harness.arm_validity import block_order

    order = block_order(1)[0]
    windows = [(k, seq) for k, seq in WINDOWS if k <= A0_MAX_K]
    per_block = len(windows) * len(order)
    plan: List[Dict[str, Any]] = []
    for block in range(start_block, start_block + blocks):
        position = 0
        for k, seq in windows:
            for role in order:
                plan.append({"arm": ABBA_ARMS[role], "k": k, "seq": seq, "block": block,
                             "index": block * per_block + position,
                             "kind": "speed", "deterministic": False})
                position += 1
    return plan


def arm_watch_kwargs(args: Any, *, arm: str, round_number: int, run: int) -> Dict[str, Any]:
    """Arguments of one ``ArmWatch`` around one run. All three checks are required, and the
    model folder is the disk under test, so foreign reads are measured on that disk."""
    return {"arm": arm, "round": round_number, "run": run, "label": args.label,
            "required": ARM_CHECKS, "model_path": args.source}


def run_number(item: Dict[str, Any], position: int) -> int:
    """Zero-based index of a plan item: its plan index when it has one (resumed pilots), else its
    position in the plan."""
    return item.get("index", position)


def append_jsonl(path: Any, obj: Dict[str, Any]) -> None:
    """Append one JSON object as one line."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj) + "\n")


def sweep(args: Any, plan: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:  # pragma: no cover
    """Run each plan item in a fresh process inside an ``ArmWatch``. Appends every row to
    ``<out>/rows.jsonl`` and every arm record, void ones included, to
    ``<out>/arm_records.jsonl``."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    from benchmarks.harness.arm_validity import ArmWatch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    gate = list(getattr(args, "gate", None) or ())
    for index, item in enumerate(plan):
        if deadline is not None and time.time() > deadline:
            print(f"deadline reached, {len(rows)} of {len(plan)} runs done", flush=True)
            break
        if gate:
            checked = subprocess.run(gate, capture_output=True, text=True)
            if checked.stdout.strip():
                print(checked.stdout.strip(), flush=True)
            if checked.returncode != 0:
                # The card is not healthy (ticket 13: AC, clock, power limit): refuse the run
                # instead of writing a row the verdict would have to throw away.
                print(f"gate refused run {index + 1} of {len(plan)} "
                      f"({item['arm']} k={item['k']} S={item['seq']}): "
                      f"{checked.stderr.strip()[-300:]}", flush=True)
                break
        command = [sys.executable, __file__, *child_run_args(args, item)]
        number = run_number(item, index)
        kwargs = arm_watch_kwargs(args, arm=item["arm"], round_number=number // 2 + 1,
                                  run=number + 1)
        with ArmWatch(**kwargs) as watch:
            proc = subprocess.run(command, capture_output=True, text=True)
        line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
        row = json.loads(line) if line else {
            "arm": item["arm"], "k": item["k"], "seq": item["seq"], "kind": item["kind"],
            "mode": "deterministic" if item["deterministic"] else "normal",
            "status": "CRASH", "error": proc.stderr[-300:]}
        row["block"] = item["block"]
        row["outcome"] = watch.record["outcome"]
        rows.append(row)
        append_jsonl(out / "rows.jsonl", row)
        append_jsonl(out / "arm_records.jsonl", watch.record)
    return rows


def main(argv: Any = None) -> int:  # pragma: no cover - thin CLI
    from pathlib import Path

    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "speed", "bits", "profile", "lever-speed", "lever-bits", "abba",
                 "loader-speed", "decoupled-speed", "seg-speed", "esc-speed", "ceg-speed",
                 "ds-bits"):
        p = sub.add_parser(name)
        p.add_argument("--source", required=True)
        p.add_argument("--cache", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--steps", type=int, default=4, help="first step is warm-up")
        p.add_argument("--lora-r", type=int, default=16)
        p.add_argument("--lora-targets", type=parse_lora_targets, default=DEFAULT_LORA_TARGETS,
                       help="comma-separated module names; default: the Kanana list")
        if name != "run":
            p.add_argument("--deadline-hours", type=float, default=0,
                           help="stop launching runs after this many hours (0: no limit)")
            p.add_argument("--gate", nargs="+", default=None, metavar="CMD",
                           help="command checked before every run; a non-zero exit refuses the "
                                "run and stops the sweep (ticket 13: AC, clock, power limit)")
        p.add_argument("--tokenizer", default=None, help="default: the --source folder")
        p.add_argument("--device", choices=("auto", "cpu"), default="auto")
        p.add_argument("--label", default=DEFAULT_LABEL,
                       help="card and 'real' or 'fake', written into every arm record")
        if name == "run":
            p.add_argument("--arm", required=True)
            p.add_argument("--k", type=int, required=True)
            p.add_argument("--seq", type=int, required=True)
            p.add_argument("--kind", choices=("speed", "bits"), required=True)
            p.add_argument("--deterministic", action="store_true")
            p.add_argument("--math-attention", action="store_true",
                           help="SDPA math backend only (diagnostic for attention backward)")
            p.add_argument("--profile", action="store_true",
                           help="add one phase-profiled step (normal mode only)")
            p.add_argument("--trace", action="store_true",
                           help="one CUDA-only torch-profiler step for the device timeline, "
                                "written into the row")
            p.add_argument("--count", action="store_true",
                           help="count the routed-expert F.linear calls of every step into the row")
        if name in ("speed", "lever-speed", "loader-speed", "abba", "decoupled-speed",
                    "seg-speed", "esc-speed", "ceg-speed"):
            p.add_argument("--blocks", type=int, default=1 if name == "lever-speed" else 5)
        if name in ("abba", "esc-speed", "ceg-speed"):
            p.add_argument("--start-block", type=int, default=0,
                           help="first block number, to resume a pilot in the same folder")
        if name in ("bits", "lever-bits", "ds-bits"):
            p.add_argument("--seq", type=int, default=64)
            p.add_argument("--ladder-ks", type=parse_k_list,
                           default=(2,) if name == "ds-bits" else BIT_LADDER,
                           help="comma-separated k values (default: 1,2,4,8; 2 for ds-bits)")
            p.add_argument("--arms", type=parse_arm_list, default=None,
                           help="comma-separated arms only (default: every arm of each k)")
            p.add_argument("--math-attention", action="store_true",
                           help="SDPA math backend only in every child run")
    args = parser.parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_one(args)))
        return 0
    plan = {"speed": lambda: speed_plan(blocks=args.blocks),
            "abba": lambda: abba_plan(blocks=args.blocks, start_block=args.start_block),
            "bits": lambda: bit_plan(seq=args.seq, ks=args.ladder_ks), "profile": profile_plan,
            "lever-speed": lambda: lever_speed_plan(blocks=args.blocks),
            "loader-speed": lambda: loader_speed_plan(blocks=args.blocks),
            "decoupled-speed": lambda: decoupled_speed_plan(blocks=args.blocks),
            "seg-speed": lambda: segmented_speed_plan(blocks=args.blocks),
            "esc-speed": lambda: esc_speed_plan(blocks=args.blocks,
                                               start_block=args.start_block),
            "ceg-speed": lambda: ceg_speed_plan(blocks=args.blocks,
                                                start_block=args.start_block),
            "ds-bits": lambda: decoupled_segmented_bit_plan(seq=args.seq, ks=args.ladder_ks),
            "lever-bits": lambda: lever_bit_plan(seq=args.seq, ks=args.ladder_ks)}[args.command]()
    if args.command in ("bits", "lever-bits", "ds-bits") and args.arms is not None:
        plan = [item for item in plan if item["arm"] in args.arms]
        if not plan:
            parser.error("--arms selects no arm of this ladder")
    rows = sweep(args, plan)
    (Path(args.out) / "summary.md").write_text(markdown_table(summarize_rows(rows)),
                                                 encoding="utf-8")
    if args.command in ("bits", "lever-bits", "ds-bits"):
        digests = ladder_verdict(rows)
        verdicts = tensor_ladder_verdict(rows, tensor_dir=bit_tensor_dir(args.out))
        console = Console()
        for (k, arm), verdict in sorted(verdicts.items()):
            console.print(f"k={k} {arm}: {verdict} (digest {digests.get((k, arm))})")
        (Path(args.out) / "bits-verdict.json").write_text(
            json.dumps({f"k{k}-{arm}": v for (k, arm), v in sorted(verdicts.items())}, indent=2),
            encoding="utf-8")
    if args.command.startswith("lever-"):
        gates = {lever: lever_gate(rows, lever=lever, previous=before)
                 for lever, before in LEVER_PREVIOUS.items()}
        (Path(args.out) / "gates.json").write_text(
            json.dumps(gates, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
