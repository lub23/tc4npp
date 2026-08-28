#!/usr/bin/env python3
# @File: index_analysis.py
# @Date: 2026-06-28
# @Author: Lubber
"""Calculate and visualize the tropical cyclone impact index."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import signal
from scipy.interpolate import interp1d

from src.config import NPP


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data" / "CMABSTdata"
DEFAULT_FIGURE_DIR = PACKAGE_ROOT / "figures"

NATURE_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#ECE133"]
REGION_NAMES = [
    "Northern China",
    "East China Sea",
    "South China Sea",
    "Southwestern China",
]
GROUP_LIMITS = (5, 10, 17, 20)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 20,
        "axes.linewidth": 0.8,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    }
)


@dataclass
class TrackRecord:
    """One best-track observation."""

    time: str
    intensity: int
    lat: float
    lon: float
    pressure: int
    wind: float


@dataclass
class Typhoon:
    """One tropical cyclone and its ordered track records."""

    international_id: str
    china_id: str
    name: str
    records: List[TrackRecord]
    interval_hours: int


def _parse_header(line: str) -> dict:
    return {
        "international_id": line[5:10].strip(),
        "china_id": line[20:25].strip(),
        "interval": int(line[27:29].strip()),
        "name": line[29:50].strip().split("(")[0],
    }


def _parse_record(line: str) -> TrackRecord:
    return TrackRecord(
        time=line[:10].strip(),
        intensity=int(line[11:12].strip()),
        lat=float(line[13:16].strip()) / 10.0,
        lon=float(line[17:21].strip()) / 10.0,
        pressure=int(line[22:26].strip()),
        wind=float(line[31:34].strip()),
    )


def load_typhoon_year(
    year: int,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> List[Typhoon]:
    """Parse one CMA best-track file."""
    filename = Path(data_dir) / f"CH{year}BST.txt"
    if not filename.exists():
        raise FileNotFoundError(f"Best-track file not found: {filename}")

    typhoons: List[Typhoon] = []
    header: dict | None = None
    records: List[TrackRecord] = []

    with filename.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("66666"):
                if header is not None:
                    records.append(_parse_record(line))
                continue

            if header is not None:
                typhoons.append(
                    Typhoon(
                        international_id=header["international_id"],
                        china_id=header["china_id"],
                        name=header["name"],
                        records=records,
                        interval_hours=header["interval"],
                    )
                )
            header = _parse_header(line)
            records = []

    if header is not None:
        typhoons.append(
            Typhoon(
                international_id=header["international_id"],
                china_id=header["china_id"],
                name=header["name"],
                records=records,
                interval_hours=header["interval"],
            )
        )
    return typhoons


def load_typhoons(
    start_year: int,
    end_year: int,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> List[Typhoon]:
    typhoons: List[Typhoon] = []
    for year in range(start_year, end_year + 1):
        try:
            typhoons.extend(load_typhoon_year(year, data_dir))
        except FileNotFoundError as error:
            print(f"[WARN] {error}")
    return typhoons


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    term = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return float(earth_radius * 2.0 * np.arcsin(np.sqrt(term)))


def _distance_weight(
    wind_speed: float,
    distance: float,
    method: str = "inverse",
) -> float:
    """Distance weight, with the original wind-dependent impact radius."""
    length_per_degree = 6371.0088 * np.pi / 180.0
    if wind_speed < 17.2:
        impact_radius = 2 * length_per_degree
    elif wind_speed < 24.4:
        impact_radius = 3 * length_per_degree
    elif wind_speed < 32.6:
        impact_radius = 4 * length_per_degree
    else:
        impact_radius = 5 * length_per_degree
    if distance > impact_radius:
        return 0.0

    if method == "inverse":
        return 1.0 / (distance + 1.0)
    if method == "inverse_square":
        return 1.0 / (distance + 1.0) ** 2
    if method == "exponential":
        return float(np.exp(-distance / 100.0))
    if method == "gaussian":
        return float(np.exp(-(distance**2) / length_per_degree**2))
    raise ValueError(f"Unknown distance weighting method: {method}")


def _smooth(values: np.ndarray, method: str) -> np.ndarray:
    if len(values) < 5 or method == "none":
        return values
    if method == "savgol":
        window = min(9, len(values) if len(values) % 2 else len(values) - 1)
        window = max(3, window)
        order = min(3, window - 1)
        try:
            return signal.savgol_filter(values, window, order)
        except ValueError:
            return values
    if method == "gaussian":
        return signal.gaussian_filter1d(values, 1.5)
    if method == "moving_average":
        smoothed = np.convolve(values, np.ones(5) / 5.0, mode="same")
        smoothed[0], smoothed[-1] = values[0], values[-1]
        return smoothed
    raise ValueError(f"Unknown smoothing method: {method}")


def calculate_yearly_indices(
    typhoons: Iterable[Typhoon],
    npp_data: Dict[str, dict],
    weight_method: str = "inverse",
    smoothing: str = "savgol",
    gust_threshold: float = 13.9,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    """Return raw annual TCIN and its smoothed, globally normalized form."""
    typhoons = list(typhoons)
    typhoons_by_year: dict[int, list[Typhoon]] = {}
    for typhoon in typhoons:
        year = int(typhoon.records[0].time[:4])
        typhoons_by_year.setdefault(year, []).append(typhoon)

    raw_indices: dict[str, dict[int, float]] = {name: {} for name in npp_data}
    maximum_index = 0.0
    for year, year_typhoons in typhoons_by_year.items():
        for npp_name, npp in npp_data.items():
            npp_lat = float(npp["Ordinate"])
            npp_lon = float(npp["Abscissa"])
            annual_index = 0.0
            for typhoon in year_typhoons:
                weighted_gust_sum = 0.0
                weight_sum = 0.0
                for record in typhoon.records:
                    if record.wind < gust_threshold:
                        continue
                    distance = _haversine(
                        record.lat, record.lon, npp_lat, npp_lon
                    )
                    weight = _distance_weight(record.wind, distance, weight_method)
                    weighted_gust_sum += record.wind**2 * weight
                    weight_sum += weight
                annual_index += weighted_gust_sum / weight_sum if weight_sum else 0.0
            raw_indices[npp_name][year] = annual_index
            maximum_index = max(maximum_index, annual_index)

    smoothed_indices: dict[str, dict[int, float]] = {}
    for npp_name, year_values in raw_indices.items():
        years = sorted(year_values)
        values = np.array([year_values[year] / maximum_index for year in years])
        values = _smooth(values, smoothing)
        smoothed_indices[npp_name] = dict(zip(years, values.tolist()))
    return raw_indices, smoothed_indices


def calculate_monthly_indices(
    typhoons: Iterable[Typhoon],
    npp_data: Dict[str, dict],
    weight_method: str = "inverse",
    smoothing: str = "savgol",
    gust_threshold: float = 13.9,
) -> dict[str, dict[int, dict[int, float]]]:
    """Return normalized monthly TCIN, preserving the original per-month climatology."""
    typhoons = list(typhoons)
    typhoons_by_year_month: dict[int, dict[int, list[Typhoon]]] = {}
    for typhoon in typhoons:
        record_time = typhoon.records[0].time
        year, month = int(record_time[:4]), int(record_time[4:6])
        year_data = typhoons_by_year_month.setdefault(year, {})
        year_data.setdefault(month, []).append(typhoon)

    raw_indices = {
        name: {month: {} for month in range(1, 13)} for name in npp_data
    }
    maximum_index = 0.0
    for year, month_typhoons in typhoons_by_year_month.items():
        for month in range(1, 13):
            typhoons_in_month = month_typhoons.get(month, [])
            if not typhoons_in_month:
                continue
            for npp_name, npp in npp_data.items():
                npp_lat = float(npp["Ordinate"])
                npp_lon = float(npp["Abscissa"])
                month_index = 0.0
                for typhoon in typhoons_in_month:
                    weighted_gust_sum = 0.0
                    weight_sum = 0.0
                    for record in typhoon.records:
                        if record.wind < gust_threshold:
                            continue
                        distance = _haversine(
                            record.lat, record.lon, npp_lat, npp_lon
                        )
                        weight = _distance_weight(record.wind, distance, weight_method)
                        weighted_gust_sum += record.wind**2 * weight
                        weight_sum += weight
                    month_index += weighted_gust_sum / weight_sum if weight_sum else 0.0
                normalized_index = month_index / len(typhoons_in_month)
                raw_indices[npp_name][month][year] = normalized_index
                maximum_index = max(maximum_index, normalized_index)

    smoothed_indices = {
        name: {month: {} for month in range(1, 13)} for name in npp_data
    }
    for npp_name, months in raw_indices.items():
        for month, year_values in months.items():
            if not year_values:
                continue
            years = sorted(year_values)
            values = np.array([year_values[year] / maximum_index for year in years])
            values = _smooth(values, smoothing)
            smoothed_indices[npp_name][month] = dict(zip(years, values.tolist()))
    return smoothed_indices


def _group_names(npp_names: List[str]) -> list[list[str]]:
    return [
        npp_names[start:end]
        for start, end in zip((0, *GROUP_LIMITS[:-1]), GROUP_LIMITS)
    ]


def _scientific(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    return rf"{mantissa:.2f}\times10^{{{exponent}}}"


def plot_yearly_trends(
    raw_indices: dict[str, dict[int, float]],
    smoothed_indices: dict[str, dict[int, float]],
    npp_data: Dict[str, dict],
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict:
    """Plot annual TCIN for individual sites and regional summaries."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    npp_names = list(raw_indices)
    groups = _group_names(npp_names)
    all_values = [
        value for values in smoothed_indices.values() for value in values.values()
    ]
    low, high = min(all_values), max(all_values)
    margin = (high - low) * 0.1
    y_limits = (low - margin, high + margin)

    fig = plt.figure(figsize=(16, 20))
    grid = gridspec.GridSpec(
        8, 4,
        hspace=0.12, wspace=0.10, left=0.08, right=0.98, top=0.93, bottom=0.08,
        height_ratios=[0.02, 1, 1, 1, 1, 1, 1, 1],
    )
    slopes = []
    site_number = 0
    for column, (group, region) in enumerate(zip(groups, REGION_NAMES)):
        title_ax = fig.add_subplot(grid[0, column])
        title_ax.axis("off")
        title_ax.text(0, 0.95, f"({chr(97 + column)}) {region}",
                      transform=title_ax.transAxes, ha="left", va="top",
                      fontweight="bold")

        regional = {}
        for row, npp_name in enumerate(group):
            site_number += 1
            ax = fig.add_subplot(grid[row + 1, column])
            years = sorted(smoothed_indices[npp_name])
            values = [smoothed_indices[npp_name][year] for year in years]
            for year, value in zip(years, values):
                regional.setdefault(year, []).append(value)

            try:
                slope, intercept = np.polyfit(years, values, 1)
            except (np.linalg.LinAlgError, ValueError):
                slope, intercept = 0.0, float(np.mean(values))
            slopes.append(slope)
            ax.plot(years, values, "o", color=NATURE_COLORS[0], markersize=3, alpha=0.6)
            ax.plot(years, np.poly1d((slope, intercept))(years),
                    "--", color=NATURE_COLORS[1], linewidth=1.5)
            ax.text(0.02, 0.95, f"{site_number}. {npp_data[npp_name]['en_name']}",
                    transform=ax.transAxes, ha="left", va="top")
            ax.text(0.93, 0.80, rf"$\mathrm{{k}}={_scientific(slope)}$",
                    transform=ax.transAxes, ha="right", va="top",
                    color=NATURE_COLORS[1])
            ax.set_ylim(y_limits)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            if column:
                ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_summary = fig.add_subplot(grid[7, column])
        years = sorted(regional)
        means = np.array([np.mean(regional[year]) for year in years])
        stds = np.array([np.std(regional[year]) for year in years])
        try:
            slope, intercept = np.polyfit(years, means, 1)
        except (np.linalg.LinAlgError, ValueError):
            slope, intercept = 0.0, float(np.mean(means))
        ax_summary.fill_between(years, means - stds, means + stds,
                                alpha=0.25, color=NATURE_COLORS[2], label="±1 SD")
        ax_summary.plot(years, means, "o-", color=NATURE_COLORS[0],
                        markersize=4, linewidth=1, label="Mean")
        ax_summary.plot(years, np.poly1d((slope, intercept))(years),
                        "--", color=NATURE_COLORS[1], linewidth=1.5)
        ax_summary.text(0.02, 0.95, "Mean±SD", transform=ax_summary.transAxes,
                        ha="left", va="top")
        ax_summary.text(0.93, 0.80, rf"$\mathrm{{k}}={_scientific(slope)}$",
                        transform=ax_summary.transAxes, ha="right", va="top",
                        color=NATURE_COLORS[1])
        ax_summary.set_ylim(y_limits)
        ax_summary.grid(True, alpha=0.2, linewidth=0.5)
        ax_summary.set_facecolor("#F5F5F5")
        if column:
            ax_summary.set_yticklabels([])
        ax_summary.spines["top"].set_visible(False)
        ax_summary.spines["right"].set_visible(False)

    fig.text(0.02, 0.5, "TCIN Intensity", va="center", rotation="vertical")
    fig.text(0.5, 0.05, "Year", ha="center")
    legend_ax = fig.add_subplot(grid[4, 3])
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([], [], marker="o", color="w", markerfacecolor=NATURE_COLORS[0],
               markersize=5, linestyle="None", label="Annual TCIN"),
        Line2D([], [], color=NATURE_COLORS[1], linestyle="--", linewidth=1.5,
               label="Trend line"),
    ], loc="center", frameon=False, ncol=1, labelspacing=1.2)
    legend_ax_summary = fig.add_subplot(grid[6, 3])
    legend_ax_summary.axis("off")
    legend_ax_summary.legend(handles=[
        Patch(facecolor=NATURE_COLORS[2], alpha=0.25,
              edgecolor=NATURE_COLORS[2], label="±1 SD"),
        Line2D([], [], color=NATURE_COLORS[0], marker="o", markersize=5,
               linewidth=1, label="Mean"),
    ], loc="center", frameon=False, ncol=1, labelspacing=1.2)

    output = figure_dir / "plot_yearly_trends.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return {"slopes": slopes, "mean_slope": np.mean(slopes), "std_slope": np.std(slopes)}


def plot_fourier_analysis(
    indices: dict[str, dict[int, float]],
    npp_data: Dict[str, dict],
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict:
    """Plot normalized power spectra by regional NPP group."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    npp_names = list(indices)
    groups = _group_names(npp_names)
    period_grid = np.linspace(2, 30, 200)

    fig = plt.figure(figsize=(16, 20))
    grid = gridspec.GridSpec(
        8, 4,
        hspace=0.12, wspace=0.10, left=0.08, right=0.98, top=0.955, bottom=0.10,
        height_ratios=[0.02, 1, 1, 1, 1, 1, 1, 1],
    )
    main_periods: list[float] = []
    by_npp: dict[str, list[float]] = {}
    site_number = 0

    for column, (group, region) in enumerate(zip(groups, REGION_NAMES)):
        title_ax = fig.add_subplot(grid[0, column])
        title_ax.axis("off")
        title_ax.text(0, 0.95, f"({chr(97 + column)}) {region}",
                      transform=title_ax.transAxes, ha="left", va="top",
                      fontweight="bold")
        group_spectra = []

        for row, npp_name in enumerate(group):
            site_number += 1
            ax = fig.add_subplot(grid[row + 1, column])
            years = sorted(indices[npp_name])
            values = np.array([indices[npp_name][year] for year in years])
            try:
                trend = np.poly1d(np.polyfit(np.arange(len(values)), values, 1))(
                    np.arange(len(values))
                )
            except (np.linalg.LinAlgError, ValueError):
                trend = np.full_like(values, np.mean(values))
            values = values - trend
            if np.std(values) > 0:
                values = (values - np.mean(values)) / np.std(values)

            windowed = values * np.hamming(len(values))
            padded_length = 2 ** int(np.ceil(np.log2(len(windowed) * 4)))
            spectrum = np.abs(np.fft.fft(windowed, n=padded_length)) ** 2 / padded_length
            spectrum = spectrum[: padded_length // 2][1:]
            periods = 1.0 / np.fft.fftfreq(padded_length, 1.0)[1 : padded_length // 2]
            valid = (periods >= 2) & (periods <= 35)
            periods, spectrum = periods[valid], spectrum[valid]
            if len(spectrum) and np.max(spectrum) > 0:
                spectrum /= np.max(spectrum)

            if len(periods):
                interpolated = interp1d(
                    periods, spectrum, kind="linear",
                    bounds_error=False, fill_value=0,
                )(period_grid)
                group_spectra.append(interpolated)
                peaks, _ = signal.find_peaks(spectrum, height=0.1, distance=2)
                top_peaks = sorted(peaks, key=lambda p: spectrum[p], reverse=True)[:2]
                for peak in top_peaks:
                    main_periods.append(float(periods[peak]))
                    by_npp.setdefault(npp_name, []).append(float(periods[peak]))
                    ax.plot(periods[peak], spectrum[peak], "o",
                            color=NATURE_COLORS[1], markersize=5)
                    ax.text(periods[peak], spectrum[peak] + 0.05,
                            f"{periods[peak]:.1f}", ha="center")
                ax.plot(periods, spectrum, "-", color=NATURE_COLORS[2], linewidth=1.2)
                ax.fill_between(periods, 0, spectrum,
                                color=NATURE_COLORS[2], alpha=0.2)

            ax.set_xlim(2, 30)
            ax.set_ylim(0, 1.49)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.text(0.02, 0.95, f"{site_number}. {npp_data[npp_name]['en_name']}",
                    transform=ax.transAxes, ha="left", va="top")
            if column:
                ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_summary = fig.add_subplot(grid[7, column])
        if group_spectra:
            mean_spectrum = np.mean(group_spectra, axis=0)
            std_spectrum = np.std(group_spectra, axis=0)
            for spectrum in group_spectra:
                ax_summary.plot(period_grid, spectrum, color="gray",
                                alpha=0.6, linewidth=0.7)
            ax_summary.fill_between(period_grid, mean_spectrum - std_spectrum,
                                    mean_spectrum + std_spectrum,
                                    color=NATURE_COLORS[2], alpha=0.25, label="±1 SD")
            ax_summary.plot(period_grid, mean_spectrum, color=NATURE_COLORS[0],
                            linewidth=1.5, label="Mean")
            peaks, _ = signal.find_peaks(mean_spectrum, height=0.1, distance=10)
            for peak in sorted(peaks, key=lambda p: mean_spectrum[p], reverse=True)[:2]:
                ax_summary.plot(period_grid[peak], mean_spectrum[peak], "o",
                                color=NATURE_COLORS[1], markersize=6)
                ax_summary.text(period_grid[peak], mean_spectrum[peak] + 0.05,
                                f"{period_grid[peak]:.1f}", ha="center")
        ax_summary.set_xlim(2, 30)
        ax_summary.set_ylim(0, 1.49)
        ax_summary.grid(True, alpha=0.2, linewidth=0.5)
        ax_summary.set_facecolor("#F5F5F5")
        if column:
            ax_summary.set_yticklabels([])
        ax_summary.spines["top"].set_visible(False)
        ax_summary.spines["right"].set_visible(False)

    fig.text(0.02, 0.5, "Normalized PSD", va="center", rotation="vertical")
    fig.text(0.5, 0.07, "Period (years)", ha="center")
    legend_ax = fig.add_subplot(grid[4, 3])
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([], [], color=NATURE_COLORS[2], linewidth=1.2, label="PSD"),
        Line2D([], [], marker="o", color="w", markerfacecolor=NATURE_COLORS[1],
               markersize=6, linestyle="None", label="Peak"),
    ], loc="center", frameon=False, ncol=1, labelspacing=1.2)
    legend_ax_summary = fig.add_subplot(grid[6, 3])
    legend_ax_summary.axis("off")
    legend_ax_summary.legend(handles=[
        Patch(facecolor=NATURE_COLORS[2], alpha=0.25,
              edgecolor=NATURE_COLORS[2], label="±1 SD"),
        Line2D([], [], color=NATURE_COLORS[0], linewidth=1.5, label="Mean"),
    ], loc="center", frameon=False, ncol=1, labelspacing=1.2)

    output = figure_dir / "plot_fourier_analysis_peak_months.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return {"all_periods": main_periods, "by_npp": by_npp}


def plot_monthly_trends(
    indices: dict[str, dict[int, dict[int, float]]],
    npp_data: Dict[str, dict],
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict:
    """Plot monthly TCIN climatology by regional NPP group."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    npp_names = list(indices)
    groups = _group_names(npp_names)
    months = list(range(1, 13))
    month_labels = list("JFMAMJJASOND")
    all_values = [
        value for site in indices.values()
        for month_values in site.values() for value in month_values.values()
    ]
    y_max = np.percentile(all_values, 95)
    y_limits = (0, y_max * 1.5)

    fig = plt.figure(figsize=(16, 20))
    grid = gridspec.GridSpec(
        8, 4,
        hspace=0.12, wspace=0.10, left=0.08, right=0.98, top=0.955, bottom=0.12,
        height_ratios=[0.02, 1, 1, 1, 1, 1, 1, 1],
    )
    site_number = 0
    regional_stats = {}

    for column, (group, region) in enumerate(zip(groups, REGION_NAMES)):
        title_ax = fig.add_subplot(grid[0, column])
        title_ax.axis("off")
        title_ax.text(0, 0.95, f"({chr(97 + column)}) {region}",
                      transform=title_ax.transAxes, ha="left", va="top",
                      fontweight="bold")
        site_month_means = {month: [] for month in months}
        site_values = {month: [] for month in months}

        for row, npp_name in enumerate(group):
            site_number += 1
            ax = fig.add_subplot(grid[row + 1, column])
            monthly_data = []
            for month in months:
                values = list(indices[npp_name][month].values()) or [0]
                monthly_data.append(values)
                site_values[month].extend(values)
                site_month_means[month].append(np.mean(values))

            ax.boxplot(
                monthly_data, positions=months, widths=0.6,
                patch_artist=True, showfliers=False,
                boxprops={"facecolor": NATURE_COLORS[0], "alpha": 0.3},
                medianprops={"color": NATURE_COLORS[1], "linewidth": 1.5},
                whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
            )
            ax.plot(months, [np.mean(values) for values in monthly_data],
                    "o-", color=NATURE_COLORS[2], markersize=5,
                    linewidth=1, alpha=0.7)
            ax.set_ylim(y_limits)
            ax.set_xlim(0.5, 12.5)
            ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")
            ax.text(0.02, 0.95, f"{site_number}. {npp_data[npp_name]['en_name']}",
                    transform=ax.transAxes, ha="left", va="top")
            if column:
                ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_summary = fig.add_subplot(grid[7, column])
        means = np.array([np.mean(site_month_means[month]) for month in months])
        stds = np.array([np.std(site_month_means[month]) for month in months])
        regional_stats[f"({chr(97 + column)}) {region}"] = {
            "means": means.tolist(), "stds": stds.tolist(),
        }
        ax_summary.fill_between(months, means - stds, means + stds,
                                alpha=0.25, color=NATURE_COLORS[2], label="±1 SD")
        ax_summary.plot(months, means, "o-", color=NATURE_COLORS[0],
                        markersize=4, linewidth=1, label="Mean")
        ax_summary.set_ylim(y_limits)
        ax_summary.set_xlim(0.5, 12.5)
        ax_summary.grid(True, alpha=0.2, linewidth=0.5, axis="y")
        ax_summary.set_facecolor("#F5F5F5")
        ax_summary.set_xticks(months)
        ax_summary.set_xticklabels(month_labels)
        if column:
            ax_summary.set_yticklabels([])
        ax_summary.spines["top"].set_visible(False)
        ax_summary.spines["right"].set_visible(False)

    fig.text(0.02, 0.5, "TCIN Intensity", va="center", rotation="vertical")
    fig.text(0.5, 0.09, "Month", ha="center")
    legend_ax = fig.add_subplot(grid[4, 3])
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([], [], marker="o", color="w", markerfacecolor=NATURE_COLORS[2],
               markersize=7, linestyle="None", label="Mean"),
        Line2D([], [], color=NATURE_COLORS[1], linewidth=1.5, label="Median"),
        Patch(facecolor=NATURE_COLORS[0], alpha=0.3, label="IQR"),
    ], loc="center", frameon=False, ncol=1, labelspacing=1.2)
    legend_ax_summary = fig.add_subplot(grid[6, 3])
    legend_ax_summary.axis("off")
    legend_ax_summary.legend(handles=[
        Patch(facecolor=NATURE_COLORS[2], alpha=0.25,
              edgecolor=NATURE_COLORS[2], label="±1 SD"),
        Line2D([], [], color=NATURE_COLORS[0], marker="o", markersize=5,
               linewidth=1, label="Mean"),
    ], loc="center", frameon=False, ncol=1, labelspacing=1.2)

    output = figure_dir / "plot_monthly_trends.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return regional_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate and plot TCIN")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--weight-method", default="inverse",
                        choices=["inverse", "inverse_square", "exponential", "gaussian"])
    parser.add_argument("--smoothing", default="savgol",
                        choices=["savgol", "gaussian", "moving_average", "none"])
    args = parser.parse_args()

    npp_data = NPP
    typhoons = load_typhoons(args.start_year, args.end_year, args.data_dir)
    if not typhoons:
        raise RuntimeError("No best-track data was loaded")
    yearly, yearly_smoothed = calculate_yearly_indices(
        typhoons, npp_data, args.weight_method, args.smoothing
    )
    yearly_stats = plot_yearly_trends(
        yearly, yearly_smoothed, npp_data, args.figures_dir
    )
    print(
        f"Mean TCIN trend: {yearly_stats['mean_slope']:.5f} "
        f"± {yearly_stats['std_slope']:.5f}"
    )
    plot_fourier_analysis(yearly_smoothed, npp_data, args.figures_dir)
    monthly = calculate_monthly_indices(
        typhoons, npp_data, args.weight_method, args.smoothing
    )
    plot_monthly_trends(monthly, npp_data, args.figures_dir)


if __name__ == "__main__":
    main()
