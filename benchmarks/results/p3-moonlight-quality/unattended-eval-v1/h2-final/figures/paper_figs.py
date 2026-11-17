"""Publication figures for the Moonlight P3 quality study, figures4papers house style.

Style follows ChenLiu-1996/figures4papers scientific-figure-making: Arial/Helvetica,
top/right spines off, frameless legends, black bar edges, semantic blue (proposed) /
red (baseline) / neutral palette, tight_layout(pad=2), PNG 300 dpi + PDF.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path(__file__).parent / "paper"
DATA = json.loads((Path(__file__).parent / "data.json").read_text(encoding="utf-8"))
MEM = json.loads((Path(__file__).parent / "mem.json").read_text(encoding="utf-8"))
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "gray": "#767676", "dark": "#272727",
    "teal": "#42949E", "violet": "#9A4D8E", "highlight": "#FFD700",
}
BASE, PROP = PALETTE["red_strong"], PALETTE["blue_main"]
BASE_L, PROP_L = PALETTE["red_2"], PALETTE["blue_secondary"]
NAME_BASE, NAME_PROP = "BF16 L2L", "FP8 block64 E4M3 + Oracle"
BENCH = {"hellaswag": "HellaSwag", "arc": "ARC", "boolq": "BoolQ", "lambada": "LAMBADA"}
plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 16,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    "xtick.major.width": 2, "ytick.major.width": 2,
})
Q = DATA["summary"]["quality"]
CELLS = sorted(Q["cells"], key=lambda c: (list(BENCH).index(c["benchmark"]), c["length"]))
MARGIN = Q["margin"] * 100


def save(fig, name, title, note):
    OUT.mkdir(exist_ok=True)
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, ha="left", fontsize=21, fontweight="bold", y=1 - 0.15 / h,
                 va="top")
    fig.text(0.01, 1 - 0.62 / h, note, fontsize=13.5, color=PALETTE["gray"], va="top",
             linespacing=1.45)
    header = 0.55 + 0.25 * (note.count("\n") + 1)
    fig.tight_layout(pad=1.0, rect=(0, 0, 1, 1 - header / h))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


# Fig 1: non-inferiority forest plot (the decision rule itself)
fig, ax = plt.subplots(figsize=(13, 10.5))
y = np.arange(len(CELLS))[::-1] * 1.0
ax.axvspan(-1.5, MARGIN, color=PALETTE["green_1"], zorder=0)
ax.axvspan(MARGIN, 1.6, color=PALETTE["red_1"], zorder=0)
ax.axvline(0, color=PALETTE["gray"], lw=1.5, zorder=1)
ax.axvline(MARGIN, color=PALETTE["red_strong"], lw=2.5, ls="--", zorder=1)
for yi, c in zip(y, CELLS):
    s, p = c["simultaneous_bounds"], c["pointwise_bounds"]
    ax.plot([100 * s["lower"], 100 * s["upper"]], [yi, yi], color=PALETTE["neutral"], lw=11,
            solid_capstyle="butt", zorder=2)
    ax.plot([100 * p["lower"], 100 * p["upper"]], [yi, yi], color=PROP, lw=3, zorder=3)
    ax.scatter(100 * c["degradation"], yi, s=110, color=PROP, edgecolor="black", lw=1.5,
               zorder=4)
    ax.text(1.62, yi, f'{100 * p["upper"]:.2f}', va="center", fontsize=14,
            color=PALETTE["dark"])
ax.text(1.62, y[0] + 0.75, "upper\n95% bound", va="bottom", fontsize=12, color=PALETTE["gray"])
ax.set_yticks(y, [f'{BENCH[c["benchmark"]]}  L{c["length"]}  (n={c["n"]:,})' for c in CELLS])
ax.set_xlim(-1.5, 1.6)
ax.set_xlabel("Accuracy degradation vs BF16 (percentage points; > 0 is worse)")
ax.text(MARGIN - 0.04, y[-1] - 0.85, "margin 1 pp", ha="right", fontsize=13,
        color=PALETTE["red_strong"])
ax.legend(handles=[
    plt.Line2D([], [], marker="o", color=PROP, mec="black", lw=3, ms=10,
               label="observed, pointwise 95%"),
    plt.Line2D([], [], color=PALETTE["neutral"], lw=11,
               label="simultaneous bound (8 cells)")],
    loc="lower left", fontsize=13)
ax.set_ylim(y[-1] - 1.1, y[0] + 1.3)
ax.set_title("FP8 accuracy loss vs BF16, by benchmark and context length", fontsize=15,
             loc="left")
save(fig, "fig1_noninferiority_forest",
     "FP8 is non-inferior to BF16 within a 1-point margin in all 8 tests",
     "Each row is one benchmark at one training context length (L = tokens). A dot right of 0 "
     "means the FP8 model scored lower.\n"
     "The test uses the blue 95% interval (one-sided alpha 0.025; its upper end is printed on "
     "the right). It must stay left of the red 1-point line.\n"
     "All 8 do, so the verdict fixed before the run is NONINFERIOR. Gray = stricter bound for "
     "testing 8 cells at once, shown for context.\n"
     "Covers these 4 benchmarks, scored with plain BF16 inference, from one training run per "
     "model (seed 23): test-set noise only.")

# Fig 2: accuracy bars per benchmark and context length (figures4papers bar style)
fig, axes = plt.subplots(1, 4, figsize=(22, 8))
for ax, bench in zip(axes, BENCH):
    cells = [c for c in CELLS if c["benchmark"] == bench]
    x = np.arange(len(cells)); w = 0.36
    b = [100 * c["baseline_accuracy"] for c in cells]
    f = [100 * c["candidate_accuracy"] for c in cells]
    ax.bar(x - w / 2, b, w, color=BASE_L, edgecolor="black", lw=2, label=NAME_BASE)
    ax.bar(x + w / 2, f, w, color=PROP_L, edgecolor="black", lw=2, label=NAME_PROP)
    for xi, bv, fv in zip(x, b, f):
        ax.text(xi - w / 2, bv + 0.12, f"{bv:.2f}", ha="center", fontsize=12)
        ax.text(xi + w / 2, fv + 0.12, f"{fv:.2f}", ha="center", fontsize=12)
    lo, hi = min(b + f), max(b + f)
    ax.set_ylim(lo - 1.6, hi + 1.0)
    ax.set_xticks(x, [f'L{c["length"]}' for c in cells])
    ax.set_title(BENCH[bench], fontsize=18)
    ax.set_ylabel("Accuracy (%)" if bench == "hellaswag" else "")
axes[-1].legend(handles=[Patch(fc=BASE_L, ec="black", lw=2, label=NAME_BASE),
                         Patch(fc=PROP_L, ec="black", lw=2, label=NAME_PROP)],
                loc="upper right", fontsize=12)
save(fig, "fig2_accuracy_bars", "Test accuracy differs by at most 0.11 points",
     "One Moonlight-16B model (mixture of experts) fine-tuned with LoRA twice for 64 steps, "
     "trained layer by layer (L2L).\n"
     "Red: activations saved in BF16. Blue: saved in 8-bit FP8 (block64 E4M3), with Oracle "
     "replaying the router's expert choice in the backward pass.\n"
     "Both scored with the same plain BF16 inference. Training context 1024 or 4096 tokens. "
     "Each panel's y axis is zoomed in.")

# Fig 3: paired outcomes. Discordant tasks only, mirrored (harms left, gains right)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})
y = np.arange(len(CELLS))[::-1]
harms = np.array([c["harms"] for c in CELLS]); gains = np.array([c["gains"] for c in CELLS])
ax.barh(y, -harms, color=BASE_L, edgecolor="black", lw=1.8, label="BF16 right, FP8 wrong")
ax.barh(y, gains, color=PROP_L, edgecolor="black", lw=1.8, label="FP8 right, BF16 wrong")
for yi, h, g, c in zip(y, harms, gains, CELLS):
    ax.text(-h - 1, yi, str(h), ha="right", va="center", fontsize=13)
    ax.text(g + 1, yi, str(g), ha="left", va="center", fontsize=13)
    ax.text(41, yi, f'{100 * (h + g) / c["n"]:.1f}%', ha="left", va="center", fontsize=13,
            color=PALETTE["gray"])
ax.axvline(0, color="black", lw=2)
ax.set_xlim(-40, 48)
ax.set_xticks([-30, -20, -10, 0, 10, 20, 30], ["30", "20", "10", "0", "10", "20", "30"])
ax.set_yticks(y, [f'{BENCH[c["benchmark"]]} L{c["length"]}' for c in CELLS])
ax.set_xlabel("Tasks whose outcome flipped")
ax.text(41, y[0] + 0.7, "flipped", fontsize=12, color=PALETTE["gray"])
ax.legend(loc="upper center", fontsize=13, ncol=2, bbox_to_anchor=(0.45, -0.13))
ax.set_title("Tasks where the two models disagree", fontsize=17, loc="left")
tot_h, tot_g = harms.sum(), gains.sum()
n_all = sum(c["n"] for c in CELLS)
agree = n_all - tot_h - tot_g
ax2.bar([0], [100 * agree / n_all], 0.55, color=PALETTE["neutral"], edgecolor="black", lw=2)
ax2.bar([1], [100 * tot_h / n_all], 0.55, color=BASE_L, edgecolor="black", lw=2)
ax2.bar([2], [100 * tot_g / n_all], 0.55, color=PROP_L, edgecolor="black", lw=2)
for xi, v in enumerate([agree, tot_h, tot_g]):
    ax2.text(xi, 100 * v / n_all + 1.5, f"{100 * v / n_all:.2f}%\n({v:,})", ha="center",
             fontsize=13)
ax2.set_xticks([0, 1, 2], ["same\noutcome", "FP8\nworse", "FP8\nbetter"])
ax2.set_ylim(0, 125); ax2.set_ylabel("Share of all paired tasks (%)")
ax2.set_title(f"All {n_all:,} test tasks", fontsize=17, loc="left")
save(fig, "fig3_paired_outcomes", "Both models get the same right/wrong result on 99.1% of tasks",
     "Left: per benchmark, how many tasks only BF16 solved (red) and how many only FP8 solved "
     "(blue). Gray % = share of tasks that differ.\n"
     "The two sides are about equal (187 vs 184), consistent with the result in Fig. 1.")

# Fig 4: secondary NLL. Per-task distribution plus mean and 95% CI
nll = DATA["nll"]; sec = {(c["benchmark"], c["length"]): c for c in DATA["summary"]["secondary_nll"]}
keys = [("gsm8k", 1024), ("gsm8k", 4096), ("mbpp", 1024), ("mbpp", 4096)]
fig, ax = plt.subplots(figsize=(13, 7))
rng = np.random.default_rng(23)
for i, key in enumerate(keys):
    d = np.array([r["nll_mean_difference"] for r in nll
                  if (r["benchmark"], r["length"]) == key])
    color = PROP if key[0] == "gsm8k" else PALETTE["teal"]
    parts = ax.violinplot(d, positions=[i], widths=0.8, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color); body.set_alpha(0.18); body.set_edgecolor(color)
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=6, color=color, alpha=0.25,
               lw=0)
    c = sec[key]; lo, hi = c["confidence_interval"]
    ax.errorbar(i, c["mean_difference"], yerr=[[c["mean_difference"] - lo],
                                               [hi - c["mean_difference"]]],
                fmt="o", color="black", ms=9, capsize=8, lw=2.5, zorder=5)
    ax.text(i + 0.1, hi + 0.0012, f'{c["mean_difference"]:+.4f}\n'
            f'({100 * c["relative_mean_change"]:+.2f}%)', ha="left", va="bottom",
            fontsize=12, fontweight="bold")
ax.axhline(0, color=PALETTE["gray"], lw=1.5)
ax.set_xticks(range(4), [f"{b.upper()}\nL{L} (n={sec[(b, L)]['n']})" for b, L in keys])
ax.set_ylabel("Δ NLL per task (nats/token, FP8 − BF16)")
lim = np.percentile(np.abs([r["nll_mean_difference"] for r in nll]), 99.5)
ax.set_ylim(-lim, lim); ax.set_xlim(-0.6, 3.9)
save(fig, "fig4_secondary_nll", "Math and code NLL moves by at most 0.22% (descriptive only)",
     "NLL measures how surprised the model is by the reference answer (lower is better). Each "
     "dot is one problem: FP8 NLL minus BF16 NLL, per-task mean.\n"
     "Black dot = mean, bars = 95% interval; bold % = change vs the BF16 mean. FP8 is "
     "slightly better on GSM8K 1024, worse on the other three.\n"
     "Three of four intervals exclude zero; the shifts are small. No margin applies and these "
     "are not part of the verdict. Y axis clipped at the 99.5th percentile.")

# Fig 5: training dynamics, 64 steps
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
for k, length in enumerate((1024, 4096)):
    ax = axes[k]
    for arm, color, name, lw, ls, alpha in (("bf16_l2l", BASE, NAME_BASE, 6, "-", 0.45),
                                            ("oracle_block64_e4m3", PROP, NAME_PROP, 1.8,
                                             "--", 1.0)):
        rows = DATA["train"][f"{arm}|{length}"]
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], color=color, lw=lw,
                ls=ls, alpha=alpha, label=name)
    ax.set_title(f"Training loss, context {length}", fontsize=17)
    ax.set_xlabel("Training step"); ax.set_ylabel("Loss" if k == 0 else "")
    ax.legend(fontsize=12, loc="upper right")
for length, ls in ((1024, "-"), (4096, "--")):
    b = np.array([r["loss"] for r in DATA["train"][f"bf16_l2l|{length}"]])
    f = np.array([r["loss"] for r in DATA["train"][f"oracle_block64_e4m3|{length}"]])
    gap = 100 * (f - b) / b
    axes[2].plot(range(1, 65), gap, ls=ls, color=PALETTE["violet"], lw=1.2, alpha=0.4)
    axes[2].plot(range(8, 65), np.convolve(gap, np.ones(8) / 8, "valid"), ls=ls,
                 color=PALETTE["violet"], lw=3, label=f"context {length}, 8-step average")
axes[2].axhline(0, color=PALETTE["gray"], lw=1.5); axes[2].set_ylim(-1, 1)
axes[2].set_title("Loss gap, FP8 vs BF16", fontsize=17)
axes[2].set_xlabel("Training step"); axes[2].set_ylabel("Difference (%)")
axes[2].legend(fontsize=12, loc="upper right")
save(fig, "fig5_training_dynamics", "Training loss is nearly identical with FP8 activations",
     "Both runs start from the same weights and see the same examples in the same order, "
     "so step-to-step jumps are shared.\n"
     "Right: the per-step loss gap is at most 0.43%; the 8-step average at most 0.19%. "
     "Mean gap: +0.02% at 1024 tokens, +0.08% at 4096. One 64-step run per model.")

# Fig 6: memory. Persistent activation store, not process peak
bf = MEM["bf16_l2l|4096"]; fp = MEM["oracle_block64_e4m3|4096"]
MiB = 1024 ** 2
fig, ax = plt.subplots(figsize=(12, 9.5))
ax.bar([0], [bf["store_host_bytes"] / MiB], 0.55, color=BASE_L, edgecolor="black", lw=2)
parts = [(fp["payload"], PROP_L, "FP8 E4M3 payload"),
         (fp["scales"], PALETTE["green_2"], "block64 scales"),
         (fp["oracle_owned_host"], PALETTE["highlight"], "Oracle routing buffer")]
bottom = 0
for value, color, label in parts:
    ax.bar([1], [value / MiB], 0.55, bottom=bottom / MiB, color=color, edgecolor="black",
           lw=2, label=label)
    bottom += value
total = bottom
ax.text(0, bf["store_host_bytes"] / MiB + 8, f'{bf["store_host_bytes"] / MiB:.0f} MiB',
        ha="center", fontsize=16)
ax.text(1, total / MiB + 8, f"{total / MiB:.0f} MiB  (−{100 * (1 - total / bf['store_host_bytes']):.1f}%)",
        ha="center", fontsize=16, color=PROP, fontweight="bold")
ax.annotate("", xy=(0.72, total / MiB + 20), xytext=(0.28, bf["store_host_bytes"] / MiB - 5),
            arrowprops={"arrowstyle": "->", "color": PALETTE["gray"], "lw": 2})
ax.set_xticks([0, 1], [NAME_BASE, NAME_PROP]); ax.set_xlim(-0.6, 1.8)
ax.set_ylabel("CPU RAM per optimizer step (MiB)"); ax.set_ylim(0, 520)
ax.legend(loc="upper right", fontsize=13)
ax.set_title("CPU memory used to store layer inputs", fontsize=16, loc="left")
save(fig, "fig6_activation_memory", "FP8 cuts the persistent layer-input store in CPU RAM by 45%",
     "Layer-by-layer training keeps each layer's input in CPU RAM until the backward pass "
     "(one optimizer step = 4096 tokens).\n"
     "In FP8 this store takes 237 MiB instead of 432 MiB (-45%), counting the routing record "
     "Oracle keeps. Same size at 1024 and 4096 tokens.\n"
     "Total activation memory was not measured. GPU peaks are unchanged; sampled peak process "
     "RAM drops 5% (4096 tokens only).")
print("done", sorted(p.name for p in OUT.glob("*.png")))
