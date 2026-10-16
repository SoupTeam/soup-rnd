"""Scientific coverage figure; never promote a clock-invalid cell to a result."""

import math


def draw(index, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    projections = [row for row in index["cases"] if row["plan"]["kind"] == "projection"]
    assert len(projections) == 96 and all(row["attempted_arms"] >= 20 for row in projections)
    assert not output.exists()
    shapes = ((4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336))
    sizes = (1, 8, 32, 128, 512, 2048)
    labels = ("q/o 4096 x 4096", "k/v 1024 x 4096", "gate/up 14336 x 4096", "down 4096 x 14336")
    figure, axes = plt.subplots(2, 2, figsize=(16, 9))
    norm = TwoSlopeNorm(vmin=-math.log(2), vcenter=0, vmax=math.log(2))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dedede")
    for vertical, dtype in enumerate(("float16", "bfloat16")):
        for horizontal, pair in enumerate(("dense-native", "native-forced")):
            axis = axes[vertical, horizontal]
            cells = np.full((4, 6), np.nan)
            text, invalid = {}, []
            for row in projections:
                plan = row["plan"]
                if plan["dtype"] != dtype or plan["pair"] != pair:
                    continue
                r, c = shapes.index((plan["n"], plan["k"])), sizes.index(plan["m"])
                comp = row["comparisons"][0] if row["comparisons"] else {}
                status = comp.get("status", "NOT_RUN")
                if status == "MEASURED":
                    cells[r, c] = math.log(comp["ratio_a_over_b"])
                    text[r, c] = f"{comp['ratio_a_over_b']:.3f}\nn={comp['accepted_blocks']}"
                elif status == "SAME_PATH":
                    text[r, c] = f"same path\nn={comp['accepted_blocks']}"
                else:
                    text[r, c] = "NO\nVERDICT"
                    invalid.append((r, c))
            assert len(text) == 24
            image = axis.imshow(np.ma.masked_invalid(cells), cmap=cmap, norm=norm, aspect="auto")
            for r, c in invalid:
                axis.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor="#f5c6c6",
                        edgecolor="#8b5050",
                        hatch="//",
                    )
                )
            for (r, c), value in text.items():
                axis.text(c, r, value, ha="center", va="center", fontsize=9)
            axis.set_xticks(range(6), sizes)
            axis.set_yticks(range(4), labels)
            axis.set_xlabel("M (flattened rows / tokens)")
            axis.set_title(dtype.upper() + ": " + pair.replace("-", " / "))
    figure.suptitle(
        f"D1 K projection coverage: {index['completed_projection_cases']}/96 accepted comparisons",
        fontsize=16,
    )
    figure.subplots_adjust(left=0.15, right=0.88, top=0.90, bottom=0.16, wspace=0.45, hspace=0.42)
    bar = figure.colorbar(image, cax=figure.add_axes((0.91, 0.25, 0.015, 0.58)), extend="both")
    values = (0.5, 0.75, 1, 1.5, 2)
    bar.set_ticks([math.log(value) for value in values])
    bar.set_ticklabels([str(value) for value in values])
    bar.set_label("Descriptive wall-time A/B (>1 = lower observed B time)")
    figure.text(
        0.15,
        0.055,
        "Pink hatch = NO_VERDICT; no rejected timings displayed. Grey = identical backend path.\n"
        "Accepted cells require >=5 valid fresh-process ABBA blocks plus F1 sizing. "
        "Ratios are descriptive.\n"
        "Frozen-base checkpointed projection + dX; no optimizer. "
        "Original 5% clock guard unchanged.\n"
        + index["gpu_context"]["name"]
        + "; source "
        + index["source_sha"][:12]
        + "; matrix coverage does not imply global D1 PASS.",
        fontsize=10,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)
