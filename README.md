# TC4NPP

TC4NPP links tropical cyclone impacts to nuclear power plant risk. It prepares
ERA5 meteorology, runs two-day Polyphemus/Polair3D dispersion scenarios,
calculates the Tropical Cyclone Impact Index (TCIN), and compares radiological
impacts across background, strong-wind, heavy-rain, and tropical-cyclone
regimes.

Polair3D is part of
[Polyphemus](https://cerea.enpc.fr/polyphemus/). Polyphemus 1.11.1 is expected
under `~/Polyphemus-1.11.1/`; adjust `POLY_DIR` and `POLAIR_DIR` in
`src/config.py` for another installation.

## Layout

```text
assets/       Custom colormap
config/       Polyphemus configuration templates
data/         Event metrics, canonical NPP metadata, CMA best-track examples, downloaded rasters
geo/          Province boundaries used by maps
src/          ERA5 acquisition and Polyphemus workflow
tc4npp/       Meteorological-regime and dispersion analysis
tcin/         TCIN calculation and impact figures
```

## Installation

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The ERA5 splitting step also requires `grib_filter` from ECMWF eccodes or
`libgrib-api-tools`.

`tc4npp.analyse` downloads LandScan 2023 population data on first use to
`data/landscan-global-2023.tif`. The download is intentionally excluded from
Git by `.gitignore`.

## Polyphemus configuration

The case configurations are stored in `config/`. Copy them into the active
Polyphemus case directory before running the workflow:

```bash
mkdir -p ~/TestCase/na-nuc
cp -a config/. ~/TestCase/na-nuc/config/
```

If you change `SCENE_NAME` in `src/config.py`, copy the templates to the
corresponding case directory instead. `src/example.cfg` is no longer used.

The nuclear power plant metadata in `data/npp.json` is the canonical source.
It is loaded by `src/config.py` and shared by both analysis packages.

## ERA5 data

Download and split monthly files into the daily `ECMWF-YYYYMMDD.grb` layout:

```bash
python -m src.meteo_fetcher Southeast 2023 1 12 --output-dir meteo
```

Copy the resulting `meteo/Southeast` directory to
`~/TestCase/na-nuc/raw_data/meteo/Southeast/`, or change `SCENE_DIR` and
`REGION_NAME` in `src/config.py`.

## Dispersion workflow

Open `src/2days.ipynb` from the repository root. It contains:

1. the full-year 2023 screening workflow, using two-day simulations with a
   48-hour release; and
2. the final paper scenarios: four events for each background, strong-wind,
   heavy-rain, and tropical-cyclone regime, each with 12, 24, 36, and 48 hour
   releases.

`RUN_FULL_YEAR` is `false` by default because the screening stage is expensive.
Set it to `true` to reproduce event selection from scratch. Run all notebook
cells from the repository root. Model output is written to the case directory
configured in `src/config.py`.

After Polyphemus output is available, create the event comparison table and
dose/deposition maps:

```bash
python -m tc4npp.analyse
```

The table is written to `figures/event_comparison_table_48h.csv`.

## TCIN and impact figures

Calculate TCIN and create annual, spectral, and monthly figures. The repository
includes 2020-2024 CMA best-track data as a small runnable example:

```bash
python -m tcin.index_analysis --start-year 2020 --end-year 2024
```

Create the final radiological impact comparison figure directly from the event
comparison table:

```bash
python -m tcin.impact_figures
```
