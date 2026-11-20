#!/usr/bin/env python3
# @File: index_analysis.py
# @Date: 2026-06-28
# @Author: Lubber
"""Calculate and visualize the tropical cyclone impact index."""

from __future__ import annotations

import argparse
import csv
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
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

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


def _smooth(values: np.ndarray, method: str | None) -> np.ndarray:
    if len(values) < 5 or method in (None, "none"):
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
        return gaussian_filter1d(values, 1.5)
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
            annual_index /= len(year_typhoons)
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
    smoothing: str | None = None,
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
            values = np.clip(_smooth(values, smoothing), 0.0, 1.0)
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


def _stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _fit_trend(years: np.ndarray, values: np.ndarray) -> dict:
    if np.ptp(values) == 0:
        slope, standard_error, pvalue = 0.0, 0.0, 1.0
    else:
        fitted = stats.linregress(years, values)
        slope = float(fitted.slope)
        standard_error = float(fitted.stderr)
        pvalue = float(fitted.pvalue)
    return {
        "slope": slope,
        "se": standard_error,
        "p": pvalue,
        "stars": _stars(pvalue),
    }


def _annual_records(
    years: np.ndarray,
    names: List[str],
    values: np.ndarray,
) -> list[dict]:
    records = []
    for name, site_values in zip(names, values):
        record = {
            "start": int(years[0]),
            "end": int(years[-1]),
            "series_type": "site",
            "label": str(name),
        }
        record.update(_fit_trend(years, site_values))
        records.append(record)
    for label, group in zip("ABCD", _group_names(names)):
        record = {
            "start": int(years[0]),
            "end": int(years[-1]),
            "series_type": "group",
            "label": label,
        }
        group_values = values[
            names.index(group[0]) : names.index(group[-1]) + 1
        ].mean(axis=0)
        record.update(_fit_trend(years, group_values))
        records.append(record)
    return records


def _export_annual_records(records: list[dict], prefix: Path) -> None:
    path = prefix.with_name(prefix.name + "_trends.csv")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _render_yearly_trends(
    years: np.ndarray,
    names: List[str],
    values: np.ndarray,
    records: list[dict],
    output_prefix: Path,
    display_names: List[str],
    y_limits: tuple[float, float] | None = None,
) -> None:
    by_site = {row["label"]: row for row in records if row["series_type"] == "site"}
    by_group = {row["label"]: row for row in records if row["series_type"] == "group"}
    if y_limits is None:
        low, high = float(values.min()), float(values.max())
        margin = 0.1 * (high - low) if high > low else 0.1
        y_limits = (low - margin, high + margin)

    fig = plt.figure(figsize=(16, 22.5))
    grid = gridspec.GridSpec(
        9,
        4,
        hspace=0.12,
        wspace=0.10,
        left=0.08,
        right=0.98,
        top=0.93,
        bottom=0.08,
        height_ratios=[0.02] + [1] * 8,
    )
    for column, (group, region) in enumerate(zip(_group_names(names), REGION_NAMES)):
        title = fig.add_subplot(grid[0, column])
        title.axis("off")
        title.text(
            0,
            0.95,
            f"({chr(97 + column)}) {region}",
            transform=title.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
        )
        for row, npp_name in enumerate(group, 1):
            ax = fig.add_subplot(grid[row, column])
            site_index = names.index(npp_name)
            result = by_site[npp_name]
            site_values = values[site_index]
            fitted = site_values.mean() + result["slope"] * (years - years.mean())
            ax.plot(years, site_values, "o", color=NATURE_COLORS[0], markersize=3, alpha=0.6)
            ax.plot(years, fitted, "--", color=NATURE_COLORS[1], linewidth=1.5)
            ax.text(
                0.02,
                0.95,
                f"{site_index + 1}. {display_names[site_index]}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=20,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1),
            )
            symbol = f"{result['stars']} " if result["stars"] else ""
            ax.text(
                0.94,
                0.70,
                symbol + rf"$\mathrm{{k}}={_scientific(result['slope'])}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=18,
                color=NATURE_COLORS[1],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1),
            )
            ax.set_ylim(y_limits)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            if column:
                ax.set_yticklabels([])
            ax.tick_params(axis="x", labelbottom=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_summary = fig.add_subplot(grid[8, column])
        group_start = names.index(group[0])
        group_stop = names.index(group[-1]) + 1
        means = values[group_start:group_stop].mean(axis=0)
        stds = values[group_start:group_stop].std(axis=0)
        result = by_group["ABCD"[column]]
        ax_summary.fill_between(
            years,
            means - stds,
            means + stds,
            alpha=0.25,
            color=NATURE_COLORS[2],
        )
        ax_summary.plot(
            years,
            means,
            "o-",
            color=NATURE_COLORS[0],
            markersize=4,
            linewidth=1,
            alpha=0.8,
        )
        ax_summary.plot(
            years,
            means.mean() + result["slope"] * (years - years.mean()),
            "--",
            color=NATURE_COLORS[1],
            linewidth=1.5,
        )
        ax_summary.text(
            0.02,
            0.95,
            "Mean±SD",
            transform=ax_summary.transAxes,
            ha="left",
            va="top",
            fontsize=20,
        )
        symbol = f"{result['stars']} " if result["stars"] else ""
        ax_summary.text(
            0.94,
            0.70,
            symbol + rf"$\mathrm{{k}}={_scientific(result['slope'])}$",
            transform=ax_summary.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            color=NATURE_COLORS[1],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1),
        )
        ax_summary.set_ylim(y_limits)
        ax_summary.grid(True, alpha=0.2, linewidth=0.5)
        ax_summary.set_facecolor("#F5F5F5")
        if column:
            ax_summary.set_yticklabels([])
        ax_summary.spines["top"].set_visible(False)
        ax_summary.spines["right"].set_visible(False)

    fig.text(0.02, 0.5, "TCIN Intensity", va="center", rotation="vertical", fontsize=20)
    fig.text(0.5, 0.05, "Year", ha="center", fontsize=20)
    for row, handles in [
        (
            4,
            [
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="w",
                    markerfacecolor=NATURE_COLORS[0],
                    markersize=4,
                    linestyle="None",
                    label="Annual TCIN",
                ),
                Line2D(
                    [],
                    [],
                    color=NATURE_COLORS[1],
                    linestyle="--",
                    linewidth=1.5,
                    label="Trend line",
                ),
            ],
        ),
        (
            6,
            [
                Patch(
                    facecolor=NATURE_COLORS[2],
                    alpha=0.25,
                    edgecolor=NATURE_COLORS[2],
                    label="±1 SD",
                ),
                Line2D(
                    [],
                    [],
                    color=NATURE_COLORS[0],
                    marker="o",
                    markersize=4,
                    linewidth=1,
                    label="Mean",
                ),
            ],
        ),
    ]:
        ax = fig.add_subplot(grid[row, 3])
        ax.axis("off")
        ax.legend(
            handles=handles,
            loc="center",
            fontsize=18,
            frameon=False,
            labelspacing=1.2,
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            output_prefix.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_yearly_trends(
    raw_indices: dict[str, dict[int, float]],
    smoothed_indices: dict[str, dict[int, float]],
    npp_data: Dict[str, dict],
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict:
    """Plot annual TCIN with fit significance and export the fit statistics."""
    names = list(raw_indices)
    years = np.array(sorted(raw_indices[names[0]]))
    values = np.array(
        [[smoothed_indices[name][int(year)] for year in years] for name in names]
    )
    records = _annual_records(years, names, values)
    prefix = Path(figure_dir) / "plot_yearly_trends"
    _render_yearly_trends(
        years,
        names,
        values,
        records,
        prefix,
        display_names=[npp_data[name]["en_name"] for name in names],
    )
    _export_annual_records(records, prefix)
    slopes = [row["slope"] for row in records if row["series_type"] == "site"]
    return {"slopes": slopes, "mean_slope": np.mean(slopes), "std_slope": np.std(slopes)}


def plot_annual_archive(
    input_path: str | Path,
    output_prefix: str | Path,
    npp_data: Dict[str, dict],
    start_year: int = 1949,
    end_year: int = 2024,
    subset_output_prefix: str | Path | None = None,
    subset_start_year: int = 1980,
) -> None:
    """Plot Figure 2 from an archive of unsmoothed, globally normalized annual TCIN."""
    with np.load(Path(input_path), allow_pickle=False) as archive:
        all_years = np.asarray(archive["years"], dtype=int)
        names = [str(name) for name in archive["names"]]
        all_values = np.asarray(archive["values"], dtype=float)
    if (
        all_values.shape != (20, len(all_years))
        or len(names) != 20
        or len(set(names)) != 20
        or not np.isfinite(all_values).all()
        or not np.all(np.diff(all_years) == 1)
    ):
        raise ValueError(
            "Expected twenty ordered sites and finite annual values in consecutive years."
        )

    periods = [(start_year, end_year, Path(output_prefix))]
    if subset_output_prefix is not None:
        periods.append((subset_start_year, end_year, Path(subset_output_prefix)))
    results = []
    for period_start, period_end, prefix in periods:
        selected = (all_years >= period_start) & (all_years <= period_end)
        years = all_years[selected]
        if not np.array_equal(years, np.arange(period_start, period_end + 1)):
            raise ValueError("The input must contain every requested calendar year.")
        values = all_values[:, selected]
        if len(years) >= 5:
            window = min(9, len(years) if len(years) % 2 else len(years) - 1)
            values = signal.savgol_filter(values, window, min(3, window - 1), axis=1)
        records = _annual_records(years, names, values)
        results.append((years, values, records, prefix))

    low = min(float(result[1].min()) for result in results)
    high = max(float(result[1].max()) for result in results)
    margin = 0.1 * (high - low) if high > low else 0.1
    display_names = [npp_data[name]["en_name"] for name in names]
    for years, values, records, prefix in results:
        _render_yearly_trends(
            years,
            names,
            values,
            records,
            prefix,
            display_names=display_names,
            y_limits=(low - margin, high + margin),
        )
        _export_annual_records(records, prefix)
        print(f"Saved {years[0]}-{years[-1]}: {prefix}")


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
            spectrum = (
                np.abs(np.fft.fft(windowed, n=padded_length)) ** 2
                / np.sum(np.hamming(len(values)) ** 2)
            )
            spectrum[1 : padded_length // 2] *= 2.0
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
    """Linear monthly distributions with upper-tail flags and right-hand legends."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = list(indices)
    groups = _group_names(names)
    months = np.arange(1, 13)
    month_labels = list("JFMAMJJASOND")
    data = {
        name: [list(indices[name][month].values()) or [0.0] for month in months]
        for name in names
    }
    focus_upper = 0.25
    largest_q3 = max(
        np.percentile(values, 75)
        for site in data.values()
        for values in site
    )
    focus_upper = max(
        focus_upper,
        float(np.ceil(largest_q3 * 1.1 / 0.05) * 0.05),
    )

    fig = plt.figure(figsize=(16, 22))
    grid = fig.add_gridspec(
        9,
        4,
        hspace=0.24,
        wspace=0.12,
        left=0.08,
        right=0.98,
        top=0.955,
        bottom=0.10,
        height_ratios=[0.02, 1, 1, 1, 1, 1, 1, 1, 1.1],
    )
    regional_stats = {}
    upper_summary = 0.0
    for group in groups:
        site_means = np.array(
            [[np.mean(values) for values in data[name]] for name in group]
        )
        upper_summary = max(
            upper_summary,
            float(np.max(site_means.mean(axis=0) + site_means.std(axis=0))),
        )
    summary_upper = max(0.05, np.ceil(upper_summary * 1.3 / 0.05) * 0.05)

    for column, (group, region) in enumerate(zip(groups, REGION_NAMES)):
        title = fig.add_subplot(grid[0, column])
        title.axis("off")
        title.text(
            0,
            0.95,
            f"({chr(97 + column)}) {region}",
            ha="left",
            va="top",
            transform=title.transAxes,
            fontsize=20,
        )
        for row, name in enumerate(group, 1):
            ax = fig.add_subplot(grid[row, column])
            observations = data[name]
            ax.boxplot(
                observations,
                positions=months,
                widths=0.62,
                patch_artist=True,
                whis=1.5,
                showfliers=False,
                boxprops=dict(
                    facecolor=NATURE_COLORS[0],
                    alpha=0.32,
                    linewidth=0.8,
                ),
                medianprops=dict(color="#CC6B00", linewidth=2.2, zorder=6),
                whiskerprops=dict(color=".42", linewidth=0.8),
                capprops=dict(color=".42", linewidth=0.8),
            )
            ax.plot(
                months,
                [np.mean(values) for values in observations],
                "o-",
                color=NATURE_COLORS[2],
                linewidth=1,
                markersize=4.5,
                alpha=0.85,
                zorder=5,
            )
            above = np.array(
                [max(values) > focus_upper for values in observations]
            )
            ax.scatter(
                months[above],
                np.full(above.sum(), 0.972 * focus_upper),
                marker="^",
                s=17,
                facecolor=".45",
                edgecolor="white",
                linewidth=0.35,
                zorder=7,
            )
            ax.set(
                xlim=(0.5, 12.5),
                ylim=(-0.048 * focus_upper, 1.14 * focus_upper),
            )
            ax.yaxis.set_major_locator(
                FixedLocator([0, 0.4 * focus_upper, 0.8 * focus_upper])
            )
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:g}")
            )
            ax.set_xticks(months)
            ax.set_xticklabels([])
            if column:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="both", length=2.5)
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                0.02,
                0.97,
                f"{names.index(name) + 1}. {npp_data[name]['en_name']}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=19,
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
            )

        site_means = np.array(
            [[np.mean(values) for values in data[name]] for name in group]
        )
        means = site_means.mean(axis=0)
        stds = site_means.std(axis=0)
        regional_stats[f"({chr(97 + column)}) {region}"] = {
            "means": means.tolist(),
            "stds": stds.tolist(),
        }
        ax = fig.add_subplot(grid[8, column])
        ax.fill_between(
            months,
            means - stds,
            means + stds,
            color=NATURE_COLORS[2],
            alpha=0.22,
        )
        ax.plot(
            months,
            means,
            "o-",
            color=NATURE_COLORS[0],
            markersize=4,
            linewidth=1,
        )
        ax.set(
            xlim=(0.5, 12.5),
            ylim=(0, summary_upper),
            facecolor="#F5F5F5",
        )
        ax.set_xticks(months, month_labels)
        ax.yaxis.set_major_locator(
            FixedLocator(
                [
                    0,
                    0.25 * summary_upper,
                    0.5 * summary_upper,
                    0.75 * summary_upper,
                ]
            )
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}")
        )
        if column:
            ax.tick_params(axis="y", labelleft=False)
        ax.text(
            0.02,
            0.96,
            "Mean ± site SD",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=18,
        )
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    legend = fig.add_subplot(grid[4, 3])
    legend.axis("off")
    legend.legend(
        handles=[
            Patch(facecolor=NATURE_COLORS[0], alpha=0.32, label="IQR"),
            Line2D([], [], color="#CC6B00", linewidth=2.2, label="Median"),
            Line2D(
                [],
                [],
                color=NATURE_COLORS[2],
                marker="o",
                markersize=5,
                linewidth=1,
                label="Mean",
            ),
            Line2D(
                [],
                [],
                color=".45",
                marker="^",
                markersize=6,
                linestyle="None",
                label=f"> {focus_upper:g}",
            ),
        ],
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=18,
        labelspacing=0.7,
        handlelength=1.4,
        handletextpad=0.6,
    )
    summary_legend = fig.add_subplot(grid[6, 3])
    summary_legend.axis("off")
    summary_legend.legend(
        handles=[
            Patch(facecolor=NATURE_COLORS[2], alpha=0.22, label="±1 site SD"),
            Line2D(
                [],
                [],
                color=NATURE_COLORS[0],
                marker="o",
                markersize=4,
                linewidth=1,
                label="Regional mean",
            ),
        ],
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=18,
        labelspacing=0.9,
    )
    fig.text(
        0.018,
        0.52,
        "TCIN intensity",
        rotation="vertical",
        va="center",
        fontsize=20,
    )
    fig.text(0.5, 0.064, "Month", ha="center", fontsize=20)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            figure_dir / f"plot_monthly_trends.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    return regional_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate and plot TCIN")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--weight-method",
        default="gaussian",
        choices=["inverse", "inverse_square", "exponential", "gaussian"],
    )
    parser.add_argument(
        "--smoothing",
        default="savgol",
        choices=["savgol", "gaussian", "moving_average", "none"],
    )
    parser.add_argument(
        "--monthly-smoothing",
        default="none",
        choices=["none", "savgol", "gaussian", "moving_average"],
        help="Monthly TCIN is unsmoothed by default",
    )
    parser.add_argument(
        "--monthly-map",
        action="store_true",
        help="Also generate Figure 5",
    )
    parser.add_argument(
        "--annual-input",
        type=Path,
        help="Plot Figure 2 from an NPZ archive instead of the example best tracks",
    )
    parser.add_argument(
        "--annual-output-prefix",
        type=Path,
        default=DEFAULT_FIGURE_DIR / "figure_2",
    )
    parser.add_argument("--annual-start-year", type=int, default=1949)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--subset-output-prefix", type=Path)
    parser.add_argument("--subset-start-year", type=int, default=1980)
    args = parser.parse_args()

    npp_data = NPP
    if args.annual_input is not None:
        plot_annual_archive(
            args.annual_input,
            args.annual_output_prefix,
            npp_data,
            start_year=args.annual_start_year,
            end_year=args.annual_end_year,
            subset_output_prefix=args.subset_output_prefix,
            subset_start_year=args.subset_start_year,
        )
        return

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
        typhoons, npp_data, args.weight_method, args.monthly_smoothing
    )
    plot_monthly_trends(monthly, npp_data, args.figures_dir)
    if args.monthly_map:
        from tcin.monthly_map import plot_monthly_map
        plot_monthly_map(monthly, npp_data, args.figures_dir)


if __name__ == "__main__":
    main()
