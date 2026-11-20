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

The default weight is Gaussian. Monthly TCIN is unsmoothed by default, while
annual trends and spectral analysis retain Savitzky-Golay smoothing. Add
`--monthly-map` to create the monthly spatial figure alongside the monthly
distributions.

Create the final radiological impact comparison figure directly from the event
comparison table:

```bash
python -m tcin.impact_figures
```

### Annual trends

`tcin.index_analysis` can also plot Figure 2 from an NPZ archive with `years`,
`names` and `values`. Values are unsmoothed annual TCIN normalized against the
full historical reference, with the twenty sites in manuscript order along rows:

```bash
python -m tcin.index_analysis \
  --annual-input /path/to/annual_raw.npz \
  --annual-output-prefix figures/figure_2
```

The figure shows all twenty sites and four regional summaries. Asterisks
preceding the fitted slope indicate conventional OLS significance at
`p < 0.05`, `p < 0.01` and `p < 0.001`. The same slopes, standard errors and
p-values are written to `figures/figure_2_trends.csv`. Use
`--subset-output-prefix` for an additional 1980-2024 plot.

### Regional comparison

`tcin.period_group_comparison` reproduces the manuscript Figure 3 layout for a
selected period, with pairwise Mann-Whitney brackets and stars:

```bash
python -m tcin.period_group_comparison \
  --input /path/to/annual_raw.npz \
  --start-year 1980 --end-year 2024 \
  --output-prefix figures/figure_3_1980_2024
```

The NPZ contains `years`, `names` and `values`: consecutive calendar years,
the twenty sites in manuscript order, and a site-by-year array of unsmoothed
annual TCIN normalized using the full historical reference. The module keeps
that normalization and applies the annual Savitzky-Golay filter separately
within the selected period. It exports PNG, PDF, SVG, descriptive site and
group CSVs, and processing metadata. Use `--reference-output-prefix` to draw
the full input record with the same style and scatter positions. The complete
historical data archive is not included in this repository.

## Citation

If you use TC4NPP in academic work, please cite the archived software release:

> TC4NPP developers (2026). *TC4NPP: Climate-informed assessment of
> non-stationary tropical cyclone exposure and radiological consequences for
> coastal nuclear power plants* (Version 1.1.0) [Computer software]. GitHub.
> https://github.com/lub23/tc4npp

Please also cite the associated manuscript when it becomes available.
