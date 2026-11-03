"""Plot the final raw RAM accounting without inferring an unavailable kernel peak."""
import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path


def draw(samples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from rich.console import Console

    points = []
    opener = gzip.open if str(samples).endswith(".gz") else open
    with opener(samples, "rt", encoding="utf8") as handle:
        for line in handle:
            row = json.loads(line)
            points.append({key: row.get(key) for key in (
                "unix", "phase", "memory_current", "process_rss_bytes", "memory_max",
                "host_total", "swap_current", "events", "memory_peak")})
    caps = {row["memory_max"] for row in points}
    guests = {row["host_total"] for row in points}
    if len(caps) != 1 or len(guests) != 1 or None in caps or None in guests:
        raise ValueError("figure requires a stable recorded scope and guest MemTotal")
    unit = 2 ** 30
    start = points[0]["unix"]
    x = [(row["unix"] - start) / 3600 for row in points]
    current = [row["memory_current"] / unit for row in points]
    rss = [row["process_rss_bytes"] / unit for row in points]
    cap, guest = next(iter(caps)) / unit, next(iter(guests)) / unit
    colors = {"download": ("#8d99ae", "Source planning/download"),
              "preparation": ("#457b9d", "Cache/model setup + diagnostic"),
              "evaluation_before": ("#2a9d8f", "Base evaluation"),
              "training": ("#e9c46a", "Training + diagnostic"),
              "evaluation_after": ("#9c6bb1", "Adapter evaluation/save"),
              "reload": ("#69a96b", "Reload check")}
    fig, (ax, phases) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [8, 1]})
    ax.plot(x, current, color="#1f4e79", linewidth=1.1, label="Cgroup usage (includes file cache)")
    ax.plot(x, rss, color="#bf693b", linewidth=1, label="Sampled RSS sum: VM processes")
    ax.axhline(cap, color="#333333", linestyle="--", linewidth=1,
               label=f"Cgroup hard cap: {cap:.2f} GiB")
    ax.axhline(guest, color="#777777", linestyle=":", linewidth=1.2,
               label=f"Guest MemTotal: {guest:.2f} GiB")
    ax.set_ylim(0, max(cap, guest) * 1.05)
    ax.set_ylabel("GiB")
    ax.set_title("DeepSeek-V3 671B / P4000 8 GiB: RAM accounting", fontsize=14)
    ax.grid(axis="y", alpha=.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=.95)
    boundaries = [0] + [i for i in range(1, len(points))
                        if points[i]["phase"] != points[i - 1]["phase"]] + [len(points) - 1]
    for left, right in zip(boundaries, boundaries[1:]):
        phase = points[left]["phase"]
        phases.axvspan(x[left], x[right], color=colors.get(phase, ("#dddddd", "Other"))[0])
    phases.set_yticks([])
    phases.set_ylim(0, 1)
    utc = datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    phases.set_xlabel(f"Elapsed pilot hours; first sample {utc}")
    used = {row["phase"] for row in points}
    phases.legend(handles=[Patch(color=c, label=label) for key, (c, label) in colors.items()
                           if key in used], loc="upper center", bbox_to_anchor=(.5, -.8),
                  ncol=3, fontsize=8, frameon=False)
    oom = points[-1]["events"]["oom_kill"] - points[0]["events"]["oom_kill"]
    swap = max(row["swap_current"] for row in points)
    peak = "unavailable" if all(row["memory_peak"] is None for row in points) else "recorded"
    fig.text(.08, .02, f"Nominal 2 s samples; cgroup OOM-kill delta {oom}; observed swap {swap} B. "
             f"Kernel memory.peak: {peak}.\n"
             "Curves show sampled accounting, not an inferred exact peak; limits remain separate.",
             fontsize=8, color="#444444")
    fig.subplots_adjust(bottom=.24, hspace=.15, left=.08, right=.98, top=.94)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    Console().print({"samples": len(points), "maximum_sampled_cgroup_gib": max(current),
                     "maximum_sampled_rss_sum_gib": max(rss), "output": str(output)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.samples, args.output)
