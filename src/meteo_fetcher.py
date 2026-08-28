#!/usr/bin/env python3
# @File: meteo_fetcher.py
# @Date: 2026-06-28
# @Author: Lubber
"""Download and prepare ERA5 files for Polyphemus.

The output layout is compatible with ``tcin.event``: monthly downloads are split
into hourly single-level and pressure-level daily GRIB messages and then combined
into ``ECMWF-YYYYMMDD.grb`` daily files.
"""

from __future__ import annotations

import argparse
import calendar
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence



SINGLE_LEVEL_VARIABLES = [
    "2m_temperature",
    "boundary_layer_height",
    "eastward_turbulent_surface_stress",
    "evaporation",
    "high_cloud_cover",
    "medium_cloud_cover",
    "northward_turbulent_surface_stress",
    "skin_temperature",
    "surface_sensible_heat_flux",
    "surface_solar_radiation_downwards",
    "surface_pressure",
    "convective_precipitation",
    "large_scale_precipitation",
    "volumetric_soil_water_layer_1",
]

PRESSURE_LEVEL_VARIABLES = [
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

PRESSURE_LEVELS = [
    "500", "550", "600", "650", "700", "750", "775", "800",
    "825", "850", "875", "900", "925", "950", "975", "1000",
]

HOURS = [f"{hour:02}:00" for hour in range(24)]

REGION_BOUNDS = {
    "Beijing": (41.25, 115.25, 39.25, 117.5),
    "Xian": (44.25, 97.5, 24.25, 119.5),
    "Wuhan": (40.5, 104.5, 20.5, 124.5),
    "Shenzhen": (38.5, 104.5, 18.5, 124.5),
    "China": (54, 73, 17.5, 135.5),
    "Southeast": (33, 108, 18, 123),
}


def append_file(source: Path, target: Path) -> None:
    """Append one binary file to another without loading it into memory."""
    with source.open("rb") as src, target.open("ab") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


class MeteoFetcher:
    """Fetch ERA5 data and convert it to the Polyphemus daily-file layout."""

    def __init__(self, region_name: str, output_dir: str | Path = "meteo") -> None:
        if region_name not in REGION_BOUNDS:
            options = ", ".join(sorted(REGION_BOUNDS))
            raise ValueError(f"Unknown region '{region_name}'. Available regions: {options}")

        self.region_name = region_name
        self.output_dir = Path(output_dir) / region_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        import cdsapi

        self.client = cdsapi.Client()

    @property
    def area(self) -> Sequence[float]:
        return REGION_BOUNDS[self.region_name]

    def _write_filter_rules(self) -> tuple[Path, Path]:
        """Create local grib_filter rules so paths do not depend on the caller's CWD."""
        single_rule = self.output_dir / "rules_file"
        pressure_rule = self.output_dir / "rules_file_pressure"
        single_rule.write_text('write "[centre]_[dataType]_[date]_[hour].grib";\n', encoding="utf-8")
        pressure_rule.write_text('write "[centre]_[date]_pressureLevels.grib";\n', encoding="utf-8")
        return single_rule, pressure_rule

    def _run_grib_filter(self, rule_file: Path, source: Path) -> None:
        subprocess.run(
            ["grib_filter", str(rule_file), str(source)],
            cwd=self.output_dir,
            check=True,
        )

    def _retrieve(self, dataset: str, request: dict, target: Path) -> None:
        if target.exists() and target.stat().st_size > 0:
            print(f"[SKIP] {target}")
            return
        print(f"[FETCH] {target}")
        partial_target = target.with_suffix(target.suffix + ".part")
        partial_target.unlink(missing_ok=True)
        try:
            self.client.retrieve(dataset, request, str(partial_target))
            partial_target.replace(target)
        except Exception:
            partial_target.unlink(missing_ok=True)
            raise

    def _download_single_level(self, year: int, month: int, days: list[str]) -> Path:
        target = self.output_dir / f"SingleLevel_{year}{month:02}01.grib"
        request = {
            "product_type": ["reanalysis"],
            "variable": SINGLE_LEVEL_VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02}"],
            "day": days,
            "time": HOURS,
            "data_format": "grib",
            "download_format": "unarchived",
            "area": self.area,
        }
        self._retrieve("reanalysis-era5-single-levels", request, target)
        return target

    def _download_pressure_level(self, year: int, month: int, days: list[str]) -> Path:
        target = self.output_dir / f"PressureLevel_{year}{month:02}01.grib"
        second_target = self.output_dir / f"PressureLevel_{year}{month:02}01-2.grib"

        base_request = {
            "product_type": ["reanalysis"],
            "variable": PRESSURE_LEVEL_VARIABLES,
            "pressure_level": PRESSURE_LEVELS,
            "year": [str(year)],
            "month": [f"{month:02}"],
            "time": HOURS,
            "data_format": "grib",
            "download_format": "unarchived",
            "area": self.area,
        }
        self._retrieve(
            "reanalysis-era5-pressure-levels",
            {**base_request, "day": days[:15]},
            target,
        )
        if days[15:]:
            self._retrieve(
                "reanalysis-era5-pressure-levels",
                {**base_request, "day": days[15:]},
                second_target,
            )
            append_file(second_target, target)
            second_target.unlink()
        return target

    def _month_has_daily_files(self, first_day: datetime, last_day: datetime) -> bool:
        for day in (first_day, last_day):
            if not (self.output_dir / f"ECMWF-{day.strftime('%Y%m%d')}.grb").exists():
                return False
        return True

    def _append_boundary_forecast(self, month_start: datetime, first_month: bool) -> None:
        """Add the 18 UTC forecast from the previous day to the previous daily file."""
        if first_month:
            return
        previous_day = month_start - timedelta(days=1)
        forecast_time = month_start - timedelta(hours=6)
        forecast_file = self.output_dir / f"ecmf_fc_{forecast_time:%Y%m%d}_{forecast_time.hour}.grib"
        daily_file = self.output_dir / f"ECMWF-{previous_day.strftime('%Y%m%d')}.grb"
        if forecast_file.exists() and daily_file.exists():
            append_file(forecast_file, daily_file)

    def _combine_daily_files(self, first_day: datetime, first_month: bool) -> None:
        """Combine analysis, forecast, and pressure-level messages by day."""
        self._append_boundary_forecast(first_day, first_month)
        days_in_month = calendar.monthrange(first_day.year, first_day.month)[1]

        for day_index in range(days_in_month):
            day = first_day + timedelta(days=day_index)
            pressure_file = self.output_dir / f"ecmf_{day.strftime('%Y%m%d')}_pressureLevels.grib"
            daily_file = self.output_dir / f"ECMWF-{day.strftime('%Y%m%d')}.grb"
            if not pressure_file.exists():
                raise FileNotFoundError(f"Expected pressure-level slice is missing: {pressure_file}")

            daily_file.touch()
            for hour in (-6, 6, 18):
                forecast_time = day + timedelta(hours=hour)
                forecast_file = self.output_dir / f"ecmf_fc_{forecast_time:%Y%m%d}_{forecast_time.hour}.grib"
                if not forecast_file.exists():
                    raise FileNotFoundError(f"Expected forecast slice is missing: {forecast_file}")
                append_file(forecast_file, daily_file)
                if hour != 18:
                    forecast_file.unlink()

            for hour in range(24):
                analysis_time = day + timedelta(hours=hour)
                analysis_file = self.output_dir / f"ecmf_an_{analysis_time:%Y%m%d}_{analysis_time.hour}.grib"
                if not analysis_file.exists():
                    raise FileNotFoundError(f"Expected analysis slice is missing: {analysis_file}")
                append_file(analysis_file, daily_file)
                analysis_file.unlink()

            append_file(pressure_file, daily_file)
            pressure_file.unlink()

        next_day = first_day + timedelta(days=days_in_month)
        next_daily_file = self.output_dir / f"ECMWF-{next_day.strftime('%Y%m%d')}.grb"
        next_daily_file.touch()
        next_forecast_time = next_day - timedelta(hours=6)
        next_forecast_file = self.output_dir / f"ecmf_fc_{next_forecast_time:%Y%m%d}_{next_forecast_time.hour}.grib"
        if not next_forecast_file.exists():
            raise FileNotFoundError(f"Expected month-boundary forecast is missing: {next_forecast_file}")
        append_file(next_forecast_file, next_daily_file)
        next_forecast_file.unlink()

    def fetch_by_month(self, year: int, month: int, first_month: bool = False) -> None:
        """Download and process one calendar month."""
        _, num_days = calendar.monthrange(year, month)
        first_day = datetime(year, month, 1)

        if self._month_has_daily_files(first_day, datetime(year, month, num_days)):
            print(f"[SKIP] {year}-{month:02}: daily files already exist")
            return

        days = [f"{day:02}" for day in range(1, num_days + 1)]
        single_file = self._download_single_level(year, month, days)
        pressure_file = self._download_pressure_level(year, month, days)
        single_rule, pressure_rule = self._write_filter_rules()

        self._run_grib_filter(single_rule, single_file)
        self._run_grib_filter(pressure_rule, pressure_file)
        self._combine_daily_files(first_day, first_month)

    def fetch_monthly_range(self, start_year: int, start_month: int, num_months: int) -> None:
        """Download consecutive months in chronological order."""
        current = datetime(start_year, start_month, 1)
        for month_index in range(num_months):
            self.fetch_by_month(current.year, current.month, first_month=month_index == 0)
            next_month = current.month % 12 + 1
            next_year = current.year + (current.month == 12)
            current = datetime(next_year, next_month, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare ERA5 data")
    parser.add_argument("region", choices=sorted(REGION_BOUNDS), help="Region name")
    parser.add_argument("year", type=int, help="Start year")
    parser.add_argument("month", type=int, choices=range(1, 13), help="Start month")
    parser.add_argument("months", type=int, help="Number of months")
    parser.add_argument("--output-dir", default="meteo", help="Output root directory")
    args = parser.parse_args()

    fetcher = MeteoFetcher(args.region, args.output_dir)
    fetcher.fetch_monthly_range(args.year, args.month, args.months)


if __name__ == "__main__":
    main()
