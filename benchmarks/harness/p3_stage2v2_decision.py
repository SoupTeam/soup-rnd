"""Apply the Stage 2 v2 informativeness veto without changing paired statistics."""

from __future__ import annotations

import copy


def apply_informativeness_gate(
    summary: dict, *, chance: dict[str, float], floor_offset: float = 0.05
) -> dict:
    """Annotate every primary cell and veto an uninformative global verdict.

    ``summary`` must come from ``p3_stage2_statistics.summarize_pairs``, which raises
    on missing paired coverage. Explicit incomplete-coverage markers also take
    precedence here. A floor never overrides excessive-degradation evidence.
    The returned deep copy retains every original statistic, including floor cells.
    Empty primary cells or a missing benchmark chance value raise ``ValueError``.
    """
    result = copy.deepcopy(summary)
    cells = result["cells"]
    if not cells:
        raise ValueError("The primary cells must contain at least one cell")

    floor_cells = []
    for cell in cells:
        benchmark = cell["benchmark"]
        if benchmark not in chance:
            raise ValueError(f"Missing chance value for benchmark {benchmark!r}")
        chance_value = chance[benchmark]
        is_floor = cell["baseline_accuracy"] < chance_value + floor_offset
        cell["informative"] = not is_floor
        if is_floor:
            floor_cells.append(
                {
                    "benchmark": benchmark,
                    "length": cell["length"],
                    "seed": cell["seed"],
                    "baseline_accuracy": cell["baseline_accuracy"],
                    "chance": chance_value,
                }
            )

    statistical_decision = summary["decision"]
    result["statistical_decision"] = statistical_decision
    result["floor_cells"] = floor_cells
    if (
        statistical_decision == "BLOCKED_INCOMPLETE_COVERAGE"
        or result["coverage"]["complete"] is not True
    ):
        result["decision"] = "BLOCKED_INCOMPLETE_COVERAGE"
    elif statistical_decision == "EXCESSIVE_DEGRADATION":
        result["decision"] = "EXCESSIVE_DEGRADATION"
    elif floor_cells:
        result["decision"] = "INCONCLUSIVE_UNINFORMATIVE_FLOOR"
    else:
        result["decision"] = statistical_decision
    return result
