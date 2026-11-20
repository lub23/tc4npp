"""Reproduce the manuscript Figure 3 layout for a selected historical period.

Input NPZ keys: years, names, values. Values are UNSMOOTHED annual TCIN,
normalized once using the full historical reference, with sites along rows.
The twenty sites retain the manuscript order (groups of 5, 5, 7 and 3).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import signal, stats


COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC"]
GROUPS = [slice(0, 5), slice(5, 10), slice(10, 17), slice(17, 20)]
STYLE = {
    "font.family": "Arial",
    "font.size": 20,
    "axes.linewidth": 0.8,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 20,
}


def period_statistics(years, names, values, start, end):
    """Apply the original annual filter within the selected period."""
    mask = (years >= start) & (years <= end)
    selected_years = years[mask]
    if not np.array_equal(selected_years, np.arange(start, end + 1)):
        raise ValueError("The input must contain every requested calendar year.")
    if len(selected_years) < 5:
        raise ValueError("At least five annual values are required.")
    raw = values[:, mask]
    window = min(9, len(selected_years) if len(selected_years) % 2 else len(selected_years) - 1)
    smoothed = signal.savgol_filter(raw, window, min(3, window - 1), axis=1)
    slopes = np.array([stats.linregress(selected_years, row).slope for row in smoothed])
    means = smoothed.mean(axis=1)
    sites, groups = [], []
    for label, selection in zip("ABCD", GROUPS):
        for i in range(selection.start, selection.stop):
            sites.append({"site": str(names[i]), "group": label,
                          "mean_tcin": float(means[i]), "slope_per_year": float(slopes[i])})
        groups.append({"group": label, "mean_tcin": float(means[selection].mean()),
                       "mean_site_sd": float(means[selection].std(ddof=0)),
                       "slope_per_year": float(slopes[selection].mean()),
                       "slope_site_sd": float(slopes[selection].std(ddof=0))})
    return {"start": start, "end": end, "window": window, "sites": sites, "groups": groups}


def pairwise_comparisons(summary):
    """Use the original Figure 3 two-sided Mann–Whitney comparisons."""
    rows = []
    for panel, key in [("slope", "slope_per_year"), ("mean", "mean_tcin")]:
        values = np.array([r[key] for r in summary["sites"]])
        for left, right in combinations(range(4), 2):
            result = stats.mannwhitneyu(values[GROUPS[left]], values[GROUPS[right]],
                                       alternative="two-sided")
            p = float(result.pvalue)
            symbol = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            rows.append({"panel": panel, "left": "ABCD"[left], "right": "ABCD"[right],
                         "U": float(result.statistic), "p": p,
                         "stars": symbol})
    return rows


def add_significance_brackets(ax, data, panel, comparisons):
    selected = [r for r in comparisons if r["panel"] == panel and r["p"] < .05]
    selected.sort(key=lambda r: abs(ord(r["left"]) - ord(r["right"])))
    low, high = min(map(np.min, data)), max(map(np.max, data))
    base = high + (high - low) * .10 if panel == "slope" else high * 1.06
    step = (high - low) * .08
    for index, row in enumerate(selected):
        left, right = ord(row["left"]) - ord("A") + 1, ord(row["right"]) - ord("A") + 1
        y = base + index * step
        top = y + step * .12
        ax.plot([left, left, right, right], [y, top, top, y], "k-", linewidth=.8)
        ax.annotate(row["stars"], xy=((left + right) / 2, top), xytext=(0, -4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=plt.rcParams["legend.fontsize"])


def add_group_values(ax, data, scale=1.0, digits=3):
    """Label mean guides with mean ± site SD and separate nearby labels."""
    means = np.array([np.mean(values) for values in data])
    site_sds = np.array([np.std(values, ddof=0) for values in data])
    fontsize = ax.yaxis.label.get_fontsize()
    formatted = [
        f"{mean * scale:.{digits}f}±{sd * scale:.{digits}f}"
        for mean, sd in zip(means, site_sds)
    ]
    label_fonts = np.array(
        [fontsize if len(text) <= 12 else 14 for text in formatted]
    )
    x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
    for position, mean, color in zip(range(1, len(data) + 1), means, COLORS):
        ax.hlines(mean, x_limits[0], position, color=color, linewidth=1,
                  linestyles=(0, (4, 3)), alpha=.7, zorder=2)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.figure.canvas.draw()
    pixels = ax.transData.transform(np.column_stack([np.zeros(len(means)), means]))[:, 1]
    order = np.argsort(pixels)
    placed = pixels[order].copy()
    minimum_gap = float(np.max(label_fonts)) * ax.figure.dpi / 72 + 5
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + minimum_gap)
    placed -= np.mean(placed - pixels[order])
    lower, upper = ax.bbox.y0 + minimum_gap / 2, ax.bbox.y1 - minimum_gap / 2
    placed += max(0, lower - placed[0])
    placed -= max(0, placed[-1] - upper)
    label_pixels = pixels.copy()
    label_pixels[order] = placed
    labels = []
    for index, (mean, text, pixel, color) in enumerate(
        zip(means, formatted, label_pixels, COLORS)
    ):
        label_y = ax.transData.inverted().transform((0, pixel))[1]
        moved = abs(pixel - ax.transData.transform((0, mean))[1]) > 1
        label_x = index + 0.30 if index == 0 else index + 0.70
        alignment = "left" if index == 0 else "right"
        labels.append(ax.annotate(
            text,
            xy=(label_x, mean), xytext=(label_x, label_y),
            xycoords="data", textcoords="data",
            ha=alignment, va="center", fontsize=label_fonts[index], color=color, zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.95, pad=.15),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=.8)))
        labels[-1].arrow_patch.set_visible(moved)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    clearance = 3 * ax.figure.dpi / 72
    for label in labels:
        bounds = label.get_bbox_patch().get_window_extent(renderer)
        obstacles = [box.get_window_extent(renderer) for box in ax.patches
                     if bounds.overlaps(box.get_window_extent(renderer))]
        if obstacles:
            drop = bounds.y1 - min(box.y0 for box in obstacles) + clearance
            x, y = ax.transData.transform(label.get_position())
            label.set_position(ax.transData.inverted().transform((x, y - drop)))
            label.arrow_patch.set_visible(True)
            ax.figure.canvas.draw()
    return labels


def plot_comparison(summary, output_prefix, seed=20261110, significance=True,
                    summary_values=True):
    """Keep Figure 3's layout, styles and group-comparison brackets."""
    slopes = np.array([row["slope_per_year"] for row in summary["sites"]])
    means = np.array([row["mean_tcin"] for row in summary["sites"]])
    group_slopes = [slopes[selection] for selection in GROUPS]
    group_means = [means[selection] for selection in GROUPS]
    comparisons = pairwise_comparisons(summary) if significance else []
    rng = np.random.RandomState(seed)
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        positions = [1, 2, 3, 4]
        for ax, data in zip((ax1, ax2), (group_slopes, group_means)):
            boxes = ax.boxplot(
                data, positions=positions, widths=0.4, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", linewidth=1.5),
                showmeans=True,
                meanprops=dict(color="black", marker="D", markerfacecolor="red",
                               markersize=8, zorder=8),
            )
            for patch, color in zip(boxes["boxes"], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.25)
            for position, data_group, color in zip(positions, data, COLORS):
                jitter = rng.uniform(-0.1, 0.1, len(data_group))
                ax.plot(position + jitter, data_group, "o", color=color, alpha=0.8,
                        markersize=6, markeredgecolor="black", markeredgewidth=0.8,
                        zorder=5)
            if significance:
                add_significance_brackets(ax, data, "slope" if ax is ax1 else "mean", comparisons)
            ax.set_xticks(positions)
            ax.set_xticklabels(["Group A", "Group B", "Group C", "Group D"],
                               fontsize=plt.rcParams["axes.labelsize"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.2, axis="y")
        ax1.set_ylabel(r"Trend Slope ($10^{-3}$yr$^{-1}$)",
                       fontsize=plt.rcParams["axes.labelsize"])
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 1e3:.1f}"))
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.set_ylabel("Mean TCIN Intensity", fontsize=plt.rcParams["axes.labelsize"])
        for ax, label in zip((ax1, ax2), ("(a)", "(b)")):
            fig.text(0.0, 1.06, label, transform=ax.transAxes, ha="left", va="top",
                     fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
        legend = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="gray",
                markeredgecolor="black", markersize=6, linestyle="None",
                label="Individual site",
            ),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
                   markersize=8, linestyle="None", label="Mean"),
            Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
            Patch(facecolor="gray", alpha=0.25, edgecolor="black", label="IQR"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=4,
                   fontsize=plt.rcParams["legend.fontsize"], frameon=False,
                   bbox_to_anchor=(0.5, -0.05))
        plt.subplots_adjust(top=0.94, bottom=0.10)
        plt.tight_layout()
        if summary_values:
            add_group_values(ax1, group_slopes, scale=1e3, digits=3)
            add_group_values(ax2, group_means, digits=4)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output_prefix.with_suffix("." + extension), dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_statistics(summary, output_prefix, source, seed, significance=True, summary_values=True):
    for key in ("sites", "groups"):
        path = output_prefix.with_name(output_prefix.name + "_" + key + ".csv")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[key][0]))
            writer.writeheader()
            writer.writerows(summary[key])
    if significance:
        rows = pairwise_comparisons(summary)
        with output_prefix.with_name(output_prefix.name + "_pairwise.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "input_name": source.name,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "start": summary["start"], "end": summary["end"],
        "normalization": "input full-record normalization retained; no subset rescaling",
        "annual_filter": {"method": "Savitzky-Golay", "window": summary["window"],
                          "polynomial_order": 3, "applied_within_selected_period": True},
        "random_seed": seed,
        "significance_annotations": significance,
        "annotation_test": "two-sided Mann-Whitney U, unadjusted pairwise site-group comparisons",
        "annotation_scope": (
            "differences between site distributions; temporal trend tests are separate"
        ),
        "regional_sd": "population SD across sites, ddof=0",
        "regional_values_on_figure": summary_values,
        "figure_value_units": "mean slope in 1e-3/year; mean TCIN dimensionless",
        "figure_value_layout": (
            "mean guides to y axis; mean ± population site SD labels; "
            "colored labels are separated from nearby boxes"
        ),
    }
    output_prefix.with_name(output_prefix.name + "_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--reference-output-prefix", type=Path,
                        help="Also render the full input period using identical styles and jitter")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--no-significance", action="store_true",
                        help="Omit the original group-comparison brackets")
    parser.add_argument("--no-summary-values", action="store_true",
                        help="Reproduce the earlier layout without numerical group summaries")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        years, names, values = archive["years"], archive["names"], archive["values"]
    if (values.shape != (20, len(years)) or len(names) != 20
            or len(set(names)) != 20 or not np.isfinite(values).all()
            or not np.all(np.diff(years) == 1)):
        raise ValueError(
            "Expected twenty ordered sites and finite annual values in consecutive years."
        )
    requests = [(args.start_year, args.end_year, args.output_prefix)]
    if args.reference_output_prefix:
        requests.append((int(years[0]), int(years[-1]), args.reference_output_prefix))
    for start, end, prefix in requests:
        summary = period_statistics(years, names, values, start, end)
        plot_comparison(summary, prefix, args.seed, significance=not args.no_significance,
                        summary_values=not args.no_summary_values)
        write_statistics(
            summary,
            prefix,
            args.input,
            args.seed,
            significance=not args.no_significance,
            summary_values=not args.no_summary_values,
        )
        print(f"Saved {start}–{end}: {prefix}")


if __name__ == "__main__":
    main()
