"""Render a scientific grid from the completed, recomputed D1 index."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rich.console import Console


def draw(index: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    if index["status"] != "COMPLETE_MEASUREMENTS_REVIEW_REQUIRED" or (
        index["completed_projection_cases"] != 96
    ):
        raise ValueError("a partial or invalid grid cannot become a complete result figure")
    if output.exists():
        raise FileExistsError(output)
    shapes = ((4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336))
    labels = ("q/o 4096 x 4096", "k/v 1024 x 4096", "gate/up 14336 x 4096", "down 4096 x 14336")
    sizes = (1, 8, 32, 128, 512, 2048)
    figure, axes = pyplot.subplots(2, 2, figsize=(15, 8.4))
    colormap = pyplot.get_cmap("RdYlGn").copy()
    colormap.set_bad("#dedede")
    norm = TwoSlopeNorm(vmin=-math.log(2), vcenter=0, vmax=math.log(2))
    for vertical, dtype in enumerate(("float16", "bfloat16")):
        for horizontal, pair in enumerate(("dense-native", "native-forced")):
            axis = axes[vertical, horizontal]
            cells = np.full((4, 6), np.nan)
            text = {}
            for case in index["cases"]:
                plan = case["plan"]
                if plan["kind"] != "projection" or plan["dtype"] != dtype or plan["pair"] != pair:
                    continue
                row, column = shapes.index((plan["n"], plan["k"])), sizes.index(plan["m"])
                comparison = case["comparisons"][0]
                ratio = comparison["ratio_a_over_b"]
                same = comparison["status"] == "SAME_PATH"
                if not same:
                    cells[row, column] = math.log(ratio)
                text[row, column] = f"{ratio:.3f}\n" + (
                    "same path" if same else f"n={comparison['accepted_blocks']}"
                )
            if len(text) != 24:
                raise ValueError("the expected geometry/M cell is missing or duplicated")
            image = axis.imshow(
                np.ma.masked_invalid(cells), cmap=colormap, norm=norm, aspect="auto"
            )
            for (row, column), value in text.items():
                axis.text(column, row, value, ha="center", va="center", fontsize=9)
            axis.set_xticks(range(6), sizes)
            axis.set_yticks(range(4), labels)
            axis.set_xlabel("M (tokens / flattened rows)")
            axis.set_title(dtype.upper() + ": " + pair.replace("-", " / "), fontsize=12)
    figure.suptitle("Checkpointed NF4 projection: descriptive total-step A/B ratios", fontsize=16)
    figure.subplots_adjust(left=0.15, right=0.89, top=0.90, bottom=0.15, wspace=0.43, hspace=0.42)
    color_axis = figure.add_axes((0.92, 0.23, 0.015, 0.60))
    bar = figure.colorbar(image, cax=color_axis, extend="both")
    bar.set_ticks([math.log(value) for value in (0.5, 0.75, 1, 1.5, 2)])
    bar.set_ticklabels(("0.5", "0.75", "1", "1.5", "2"))
    bar.set_label("A/B: >1 means lower observed B time")
    figure.text(
        0.15,
        0.055,
        "Median of accepted arm medians; >=5 valid ABBA blocks per cell. "
        "Grey = identical backend path.\n"
        "Frozen base + input gradient + non-reentrant checkpoint; no LoRA/optimizer. "
        "Ratios are descriptive, not significance claims.\n"
        "RTX 3090, nested NF4, stock 370 W; source " + index["source_sha"][:12],
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.index.read_text(encoding="utf-8")), args.output)
    Console().print({"figure": str(args.output)})


if __name__ == "__main__":
    main()
