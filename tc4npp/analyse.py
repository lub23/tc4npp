# @File: analyse.py
# @Date: 2026-06-28
# @Author: Lubber
"""Analyze meteorological regimes and radiological dispersion patterns."""

import math
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
warnings.filterwarnings('ignore')
from src.config import SCENE_DIR, REGION_BOUND, GEO_DIR, NPP, SITE_NAME, SITE_LON, SITE_LAT
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def process_geo_data(filename, ords):
    # read geo data from file
    if not os.path.exists(filename):
        return None
    if filename.endswith('.asc'):
        with open(filename, 'r') as file:
            header = [next(file) for _ in range(6)]
        data = np.loadtxt(filename, skiprows=6)
        cellsize_x = cellsize_y = float(header[4].split()[1])
        nx, ny = int(header[0].split()[1]), int(header[1].split()[1])
        x_min, y_min = float(header[2].split()[1]), float(header[3].split()[1])
        x_max = x_min + nx * cellsize_x
        y_max = y_min + ny * cellsize_y
    elif filename.endswith('.tif'):
        with rasterio.open(filename) as src:
            data = src.read(1)
            transform = src.transform
            nx, ny = src.width, src.height
            x_min, y_max = transform.c, transform.f
            cellsize_x, cellsize_y = transform.a, transform.e
            x_max = x_min + cellsize_x * nx
            y_min = y_max + cellsize_y * ny
    data = np.flipud(data)
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    x_idx = np.where((x >= ords[0][0]) & (x <= ords[0][-1]))[0]
    y_idx = np.where((y >= ords[1][0]) & (y <= ords[1][-1]))[0]
    x_subset, y_subset = x[x_idx], y[y_idx]
    data_subset = data[np.ix_(y_idx, x_idx)]
    return x_subset, y_subset, data_subset


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = PACKAGE_ROOT / "figures"
COLORMAP_PATH = PACKAGE_ROOT / "assets" / "colormap.npy"
DATA_DIR = PACKAGE_ROOT / "data"
LANDSCAN_URL = "https://data.naturalcapitalalliance.stanford.edu/download/global/landscan-pop/landscan_2023.tif"
LANDSCAN_PATH = DATA_DIR / "landscan-global-2023.tif"
SPECIES = {
    "Cs137": {
        "halflife": 11000,
        "air dose coeff": 9.14e-11,
        "dep dose coeff": 1.4e-12,
        "inh dose coeff": 4.6e-9,
        "loss rate in soil": 0.00014,
    },
    "I131": {
        "halflife": 8,
        "air dose coeff": 6.08e-11,
        "dep dose coeff": 8.75e-13,
        "inh dose coeff": 7.4e-9,
        "loss rate in soil": 0,
    },
}


def ensure_population_data() -> Path:
    """Download the LandScan population raster on first use."""
    if LANDSCAN_PATH.exists() and LANDSCAN_PATH.stat().st_size > 0:
        return LANDSCAN_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = LANDSCAN_PATH.with_suffix(".tif.part")
    print(f"Downloading population data to {LANDSCAN_PATH}")
    try:
        urlretrieve(LANDSCAN_URL, partial_path)
        partial_path.replace(LANDSCAN_PATH)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return LANDSCAN_PATH
FONT_SIZES = {
    'title': 18,
    'subtitle': 18,
    'label': 16,
    'tick': 16,
    'legend': 16
}

PLOT_LON_MIN = 108.0
PLOT_LON_MAX = 123.0
PLOT_LAT_MIN = 18.0
PLOT_LAT_MAX = 33.0

class DispersionAnalyzer:
    """Calculate dispersion dose metrics and event comparisons."""
    
    def __init__(self, analysis_mode='48h', background_sample_ratio=0.2, 
                 wind_threshold=2.0, precip_threshold=5.0):
        """
        Initialize analyzer
        
        Args:
            analysis_mode: '48h' for 48-hour analysis, 'hourly' for hourly analysis
            background_sample_ratio: Ratio to sample background data (0.0-1.0)
            wind_threshold: Wind speed threshold to define background (m/s)
            precip_threshold: Precipitation threshold to define background (mm)
        """
        self.analysis_mode = analysis_mode
        self.background_sample_ratio = background_sample_ratio
        self.wind_threshold = wind_threshold
        self.precip_threshold = precip_threshold
        self.base_dir = Path(SCENE_DIR)
        

        self.config = {
            'Nx': int((REGION_BOUND[3] - REGION_BOUND[1])/0.2 + 1),
            'Ny': int((REGION_BOUND[0] - REGION_BOUND[2])/0.2 + 1),
            'Nz': 10,
            'Nt': 48,
            'Delta_t': 3600,
            'dx': 0.2,
            'dy': 0.2,
            'x_min': REGION_BOUND[1],
            'x_max': REGION_BOUND[3],
            'y_min': REGION_BOUND[2],
            'y_max': REGION_BOUND[0],
            'species': SPECIES,
            "Abscissa": SITE_LON,
            "Ordinate": SITE_LAT
        }
        
        self.npp_name = SITE_NAME
        self.total_days = 2
        

        self.grid_area = self.config['dx'] * self.config['dy'] * 111.195 * 111.195
        

        self.r_inh = 0.96
        self.reduction_factor = 0.36 # 0.2 * 1.0 + 0.8 * 0.2
        

        self.thresholds = {
            'level_1': 0.5,
        }
        

        self.meteo_data_file = PACKAGE_ROOT / "data" / "timeseries_data_2023-1.csv"
        self.load_meteorological_data()
        self.load_population_data()
        

        self.results_df = pd.DataFrame()
    
    def load_meteorological_data(self):
        """Load the cached meteorological time series."""
        if Path(self.meteo_data_file).exists():
            df = pd.read_csv(self.meteo_data_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            self.meteo_df = df
        else:
            print(f"Warning: Meteorological data file {self.meteo_data_file} not found")
            self.meteo_df = None
    
    def load_population_data(self):
        """Load population data and resample it to the simulation grid."""
        try:
            filename = ensure_population_data()
            if os.path.exists(filename):
                lon_range = np.linspace(self.config['x_min'], self.config['x_max'], self.config['Nx'])
                lat_range = np.linspace(self.config['y_min'], self.config['y_max'], self.config['Ny'])
                ords = [lon_range, lat_range]
                
                result = process_geo_data(filename, ords)
                if result is not None:
                    pop_lon, pop_lat, pop_data_raw = result
                    

                    if pop_data_raw.shape != (self.config['Nx'], self.config['Ny']):
                        print(f"Resampling population data from {pop_data_raw.shape} to ({self.config['Nx']}, {self.config['Ny']})")
                        
                        pop_lon_orig = np.linspace(pop_lon.min(), pop_lon.max(), pop_data_raw.shape[1])
                        pop_lat_orig = np.linspace(pop_lat.min(), pop_lat.max(), pop_data_raw.shape[0])
                        
                        if pop_lat_orig[0] > pop_lat_orig[-1]:
                            pop_lat_orig = pop_lat_orig[::-1]
                            pop_data_raw = np.flipud(pop_data_raw)
                        
                        interpolator = RegularGridInterpolator(
                            (pop_lat_orig, pop_lon_orig), 
                            pop_data_raw, 
                            method='linear', 
                            bounds_error=False, 
                            fill_value=0
                        )
                        
                        target_lon, target_lat = np.meshgrid(lon_range, lat_range)
                        target_points = np.column_stack([target_lat.ravel(), target_lon.ravel()])
                        pop_data_resampled = interpolator(target_points).reshape(self.config['Ny'], self.config['Nx'])
                        pop_data_resampled = np.maximum(pop_data_resampled, 0)
                        
                        self.pop_data = pop_data_resampled
                    else:
                        self.pop_data = pop_data_raw
                        
                    self.pop_lon, self.pop_lat = pop_lon, pop_lat
                    print(f"Population data loaded: {self.pop_data.shape}, range: {self.pop_data.min():.0f} - {self.pop_data.max():.0f}")
                else:
                    self.pop_data = None
            else:
                self.pop_data = None
                print(f"Population file not found: {filename}")
        except Exception as e:
            self.pop_data = None
            print(f"Error loading population data: {e}")

    def read_concentration_data(self, filepath, sizes=None):
        """Read a Fortran-order Polyphemus binary output file."""
        if not sizes:
            sizes = (self.config['Nx'], self.config['Ny'], self.config['Nz'], self.config['Nt'])
        Nx, Ny, Nt = sizes[0], sizes[1], sizes[3]
        data_n = Nx * Ny * Nt
        
        try:
            with open(filepath, 'rb') as file:
                data = np.fromfile(file, dtype=np.float32, count=data_n)
            return data.reshape((Nx, Ny, Nt), order='F')
        except FileNotFoundError:
            return None
    
    def calculate_dose_field_with_deposition(self, date_str, time_step=None, release_duration=24):
        """Calculate total dose and deposition dose fields.

        Args:
            date_str: Date string
            time_step: Specific time step or None for total
            release_duration: Release duration in hours
        """
        suffix = f"_cr{release_duration}"
        results_dir = self.base_dir / f"results_{date_str}_{self.total_days}{suffix}"
        if not results_dir.exists():
            return None, None
        
        Nx, Ny = self.config['Nx'], self.config['Ny']
        total_dose = np.zeros((Nx, Ny))
        total_dep = np.zeros((Nx, Ny))
        wet_dep_dose = np.zeros((Nx, Ny))
        for sp, sp_info in self.config['species'].items():
            air_path = results_dir / f"{self.npp_name}_{sp}_air.bin"
            dry_path = results_dir / f"{self.npp_name}_{sp}_dry.bin"
            wet_path = results_dir / f"{self.npp_name}_{sp}_wet.bin"
            
            sizes = (Nx, Ny, self.config['Nz'], self.config['Nt'])
            

            conc_air = self.read_concentration_data(air_path, sizes)
            if conc_air is not None:
                if time_step is not None:
                    conc_air_data = conc_air[:, :, time_step]
                else:
                    conc_air_data = np.sum(conc_air, axis=2)
                
                air_dose = conc_air_data * sp_info['air dose coeff'] * 1e6 * self.reduction_factor
                total_dose += air_dose
            

            for dep_path in [dry_path, wet_path]:
                conc_dep = self.read_concentration_data(dep_path, sizes)
                if conc_dep is not None:
                    if time_step is not None:
                        conc_dep_data = np.cumsum(conc_dep, axis=2)[:, :, time_step] * 3600
                    else:
                        conc_dep_data = np.sum(np.cumsum(conc_dep, axis=2), axis=2) * 3600
                    
                    dep_dose = conc_dep_data * sp_info['dep dose coeff'] * 1e6 * self.reduction_factor
                    total_dep += dep_dose
                    wet_dep_dose += dep_dose if dep_path==wet_path else 0
                    total_dose += dep_dose
        return total_dose, wet_dep_dose, total_dep

    def calculate_metrics(self, dose_array, deposition_array=None):
        """Calculate impact metrics for the threshold mask.
        
        Args:
            dose_array: Dose field with shape (Nx, Ny).
            deposition_array: Optional deposition dose field with shape (Nx, Ny).
        """
        if dose_array is None:
            return None
        
        grid_area = self.config['dx'] * self.config['dy'] * 111.195 * 111.195
        
        mask = dose_array >= self.thresholds['level_1']
        affected_area = np.sum(mask)
        area = affected_area * grid_area
        
        total_deposition = 0
        if deposition_array is not None:
            total_deposition = np.sum(deposition_array[mask])

        if affected_area == 0:
            return {
                'peak_dose': 0.0,
                'area': 0.0,
                'average_dose': 0.0,
                'total_dose_rate': 0.0,
                'collective_effective_dose_rate': 0.0,
                'total_dep_dose_rate': 0.0
            }
        
        return {
            'peak_dose': np.max(dose_array[mask]),
            'area': area,
            'average_dose': np.mean(dose_array[mask]),
            'total_dose_rate': np.sum(dose_array[mask]),
            'collective_effective_dose_rate': np.sum(dose_array[mask] * self.pop_data[mask]),
            'total_dep_dose_rate': total_deposition
        }

    def _load_event_frames(self, date_str, release_duration):
        """Load per-timestep cumulative dose frames for a single event date.
        
        Args:
            date_str: Date string like '20230727190000'
            release_duration: Release duration in hours
            
        Returns:
            (total_frames, wet_frames) each list of 48 (Nx,Ny) arrays,
            or (None, None) if data not found
        """
        suffix = f"_cr{release_duration}"
        results_dir = self.base_dir / f"results_{date_str}_{self.total_days}{suffix}"
        if not results_dir.exists():
            return None, None
        
        Nx, Ny = self.config['Nx'], self.config['Ny']
        total_frames = [np.zeros((Nx, Ny)) for _ in range(48)]
        wet_frames = [np.zeros((Nx, Ny)) for _ in range(48)]
        sizes = (Nx, Ny, self.config['Nz'], self.config['Nt'])
        
        for sp, sp_info in self.config['species'].items():
            air_path = results_dir / f"{self.npp_name}_{sp}_air.bin"
            dry_path = results_dir / f"{self.npp_name}_{sp}_dry.bin"
            wet_path = results_dir / f"{self.npp_name}_{sp}_wet.bin"
            
            conc_air = self.read_concentration_data(air_path, sizes)
            conc_dry = self.read_concentration_data(dry_path, sizes)
            conc_wet = self.read_concentration_data(wet_path, sizes)
            
            air_coeff = sp_info['air dose coeff'] * 1e6 * self.reduction_factor
            dep_coeff = sp_info['dep dose coeff'] * 1e6 * self.reduction_factor
            
            if conc_air is not None:
                air_dose_3d = conc_air * air_coeff
                for t in range(48):
                    total_frames[t] += air_dose_3d[:, :, t]
            
            for conc_dep, is_wet in [(conc_dry, False), (conc_wet, True)]:
                if conc_dep is not None:
                    dep_dose_3d = np.cumsum(conc_dep, axis=2) * 3600 * dep_coeff
                    for t in range(48):
                        total_frames[t] += dep_dose_3d[:, :, t]
                        if is_wet:
                            wet_frames[t] += dep_dose_3d[:, :, t]
        
        return total_frames, wet_frames

    def _compute_frame_metrics(self, dose_frame, wet_frame=None):
        """Compute impact metrics for a single dose frame.
        
        Args:
            dose_frame: (Nx, Ny) cumulative total dose at a given hour
            wet_frame: (Nx, Ny) or None — cumulative wet deposition dose
            
        Returns:
            dict of metric values
        """
        grid_area = self.config['dx'] * self.config['dy'] * 111.195 * 111.195
        threshold = self.thresholds['level_1']
        
        mask = dose_frame >= threshold
        n_affected = np.sum(mask)
        
        if n_affected == 0:
            return {
                'peak_dose': 0.0,
                'area': 0.0,
                'average_dose': 0.0,
                'collective_effective_dose_rate': 0.0,
                'wet_deposition_total': 0.0,
                'total_dose_rate': 0.0,
            }
        
        area = n_affected * grid_area
        peak_dose = float(np.max(dose_frame[mask]))
        average_dose = float(np.mean(dose_frame[mask]))
        total_dose_rate = float(np.sum(dose_frame[mask]))
        
        collective_dose = 0.0
        if self.pop_data is not None:
            try:
                pop = self.pop_data
                if mask.shape != pop.shape:
                    pop = pop.T
                pop_in_mask = pop[mask]
                collective_dose = float(np.sum(dose_frame[mask] * pop_in_mask))
            except Exception:
                pass
        
        wet_deposition_total = 0.0
        if wet_frame is not None and np.any(mask):
            wet_deposition_total = float(np.sum(wet_frame[mask]))
        
        return {
            'peak_dose': peak_dose,
            'area': area,
            'average_dose': average_dose,
            'collective_effective_dose_rate': collective_dose,
            'wet_deposition_total': wet_deposition_total,
            'total_dose_rate': total_dose_rate,
        }

    def _compute_area_growth_rates(self, date_str, release_duration):
        """Calculate average affected-area growth rates at 10 and 20 hours.

        Args:
            date_str: Date string like '20230727190000'
            release_duration: Release duration in hours

        Returns:
            (growth_10h, growth_20h) in km2/h, or (None, None) if data is absent.
        """
        total_frames, _ = self._load_event_frames(date_str, release_duration)
        if total_frames is None:
            return None, None

        metrics_10h = self._compute_frame_metrics(np.sum(total_frames[:10],axis=0))
        metrics_20h = self._compute_frame_metrics(np.sum(total_frames[:20],axis=0))

        area_10h = metrics_10h['area']
        area_20h = metrics_20h['area']

        growth_10h = area_10h / 10.0
        growth_20h = area_20h / 20.0
        return growth_10h, growth_20h


    def get_meteorological_data(self, date_str, time_step=0):
        """Return meteorological variables for one event start time."""
        if self.meteo_df is None:
            return {}
        date = pd.to_datetime(date_str, format='%Y%m%d')
        if time_step is None:
            time_step = 0
        target_time = date + timedelta(hours=time_step)
        if target_time in self.meteo_df.index:
            row = self.meteo_df.loc[target_time]
            return {
                'precip_48h_point': row.get('precip_48h_point', 0),
                'wind_speed_point_48h_mean': row.get('wind_speed_point_48h_mean', 0),
                'precip_hourly': row.get('precip_hourly', 0),
                'wind_speed': row.get('wind_speed', 0)
            }
        else:
            return {}

    def analyze_48h_mode(self, date_str, release_duration=48):
        """Analyze one event over the full simulation window."""
        dose_array, _, deposition_array = self.calculate_dose_field_with_deposition(date_str, None, release_duration)
        if dose_array is None:
            return None
        
        meteo_data = self.get_meteorological_data(date_str, time_step=None)
        metrics = self.calculate_metrics(dose_array, deposition_array)
        
        result = {
            'date': pd.to_datetime(date_str, format='%Y%m%d'),
            'date_str': date_str,
            'mode': '48h',
            'precip_48h_point': meteo_data.get('precip_48h_point', 0),
            'wind_speed_point_48h_mean': meteo_data.get('wind_speed_point_48h_mean', 0)
        }
        
        if metrics:
            result.update({
                'peak_dose': metrics.get('peak_dose'),
                'area': metrics.get('area'),
                'total_dose_rate': metrics.get('total_dose_rate'),
                'collective_effective_dose_rate': metrics.get('collective_effective_dose_rate'),
                'total_dep_dose_rate': metrics.get('total_dep_dose_rate'),
                'mean_deposition': metrics.get('mean_deposition'),
            })
        
        return result

    def analyze_all_dates(self, start_date='20230101', end_date='20231231'):
        """Analyze all dates with available simulation output."""
        print(f"Analyzing dates from {start_date} to {end_date} in {self.analysis_mode} mode...")
        
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        results = []
        current_date = start_dt
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y%m%d')
            results_dir = self.base_dir / f"results_{date_str}_{self.total_days}_cr48"
            
            if results_dir.exists():
                result = self.analyze_48h_mode(date_str)
                if result is not None:
                    results.append(result)
            
            current_date += timedelta(days=1)
        
        self.results_df = pd.DataFrame(results)
        print(f"Analyzed {len(results)} dates in {self.analysis_mode} mode")
        
        return self.results_df

    def analyze_selected_events(self, selected_events, release_durations, save_dir=SAVE_DIR):
        """Analyze selected meteorological regimes.
        
        Args:
            selected_events: Mapping of regime names to event start dates.
            release_durations: Release durations in hours.
            save_dir: Directory in which figures and CSV files are written.
        """
        print(f"\n{'='*50}")
        print("SELECTED EVENTS ANALYSIS")
        print(f"{'='*50}")
        
        for release_duration in release_durations:
            self.plot_concentration_maps(selected_events, save_dir, release_duration)
        

        print("\nCreating event comparison table...")
        comparison_df = self.create_event_comparison_table(selected_events,release_durations, save_dir)
        
        print(f"\nSelected events analysis complete: {len(comparison_df)} records")
        
        return comparison_df

    def create_event_comparison_table(self, selected_events,release_durations, save_dir=SAVE_DIR):
        """Create the comparison table used directly by the plotting workflow."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        comparison_data = []

        for event_type, date_list in selected_events.items():
            for date_str in date_list:
                for release_duration in release_durations:
                    dose_array, wet_dose, deposition_array = self.calculate_dose_field_with_deposition(
                        date_str, time_step=None, release_duration=release_duration)

                    if dose_array is None:
                        print(f"Skipping missing result: {date_str} cr{release_duration}")
                        continue

                    metrics = self.calculate_metrics(dose_array, deposition_array)
                    meteo_data = self.get_meteorological_data(date_str[:8], time_step=int(date_str[8:10]))

                    # Wet deposition is expressed as a percentage of total dose
                    mask = dose_array > self.thresholds['level_1']
                    wet_dep_dose_rate = np.sum(wet_dose[mask]) if np.any(mask) else 0.0
                    total_dose_rate = metrics.get('total_dose_rate', 0.0) or 0.0
                    wet_deposition_ratio = (
                        wet_dep_dose_rate / total_dose_rate * 100.0
                        if total_dose_rate > 0 else 0.0
                    )

                    growth_10h, growth_20h = self._compute_area_growth_rates(date_str, release_duration)
                    affected_area = metrics.get('area', 0.0) or 0.0
                    collective_dose = metrics.get('collective_effective_dose_rate', 0.0) or 0.0

                    event_data = {
                        'Event': event_type,
                        'Date': int(date_str[:8]),
                        'Release_Duration': release_duration,
                        'Precip': meteo_data.get('precip_48h_point', 0.0),
                        'Wind': meteo_data.get('wind_speed_point_48h_mean', 0.0),
                        'Collective_DR': collective_dose / 1e5,
                        'Peak_DR': metrics.get('peak_dose', 0.0),
                        'WR': wet_deposition_ratio,
                        'Area': affected_area / 1e4,
                        'GR10': growth_10h or 0.0,
                        'GR20': growth_20h or 0.0,
                    }
                    comparison_data.append(event_data)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(
            save_path / f'event_comparison_table_{self.analysis_mode}.csv',
            index=False,
            float_format='%.6g',
        )

        return comparison_df

    def plot_concentration_maps(self, selected_events, save_dir=SAVE_DIR, release_duration=24):
        """Plot dose and deposition maps for selected events."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        print(f"Plotting concentration maps for selected events...")
        for event_type, date_list in selected_events.items():
            self.plot_multi_concentration_map(event_type, date_list, save_path, release_duration=release_duration, plot_type='dose')
            self.plot_multi_concentration_map(event_type, date_list, save_path, release_duration=release_duration, plot_type='deposition')
        print(f"All concentration maps saved to: {save_path}")

    def plot_multi_concentration_map(self, event_type, date_list, save_path, release_duration=24, plot_type='dose'):
        """Plot several event dates on a single map canvas."""
        FONT_SIZES = {
            'title': 40,
            'subtitle': 40,
            'label': 35,
            'tick': 35,
            'legend': 35
        }

        dose_arrays = []
        dep_arrays = []

        for date_str in date_list:
            dose_a, dep_a, _ = self.calculate_dose_field_with_deposition(
                date_str, time_step=None, release_duration=release_duration)
            dose_arrays.append(dose_a)
            dep_arrays.append(dep_a)

        if plot_type == 'dose':
            label_name = 'Dose Rate (μSv/h)'
            data_arrays = dose_arrays
        else:
            label_name = 'Deposition Dose Rate (μSv/h)'
            data_arrays = dep_arrays

        if all(d is None for d in data_arrays):
            print(f"No data available for {event_type}")
            return

        try:
            cmap = mcolors.ListedColormap(np.load(COLORMAP_PATH))
        except (OSError, ValueError):
            cmap = plt.cm.plasma
            print("Custom colormap not found, using default plasma colormap")

        lon_range = np.linspace(self.config['x_min'], self.config['x_max'], self.config['Nx'])
        lat_range = np.linspace(self.config['y_min'], self.config['y_max'], self.config['Ny'])
        lon_mesh, lat_mesh = np.meshgrid(lon_range, lat_range)

        global_min = np.inf
        global_max = -np.inf
        for da in data_arrays:
            if da is not None:
                d = da.copy()
                d[d <= 0] = np.min(d[d > 0])
                d = np.log10(d)
                global_max = max(global_max, np.max(d))
                global_min = global_max - 7

        fig = plt.figure(figsize=(16 * len(date_list), 10))
        proj = ccrs.PlateCarree(central_longitude=115.0)
        axes = []
        for i in range(len(date_list)):
            ax = fig.add_subplot(1, len(date_list), i + 1, projection=proj)
            ax.set_extent([PLOT_LON_MIN, PLOT_LON_MAX, PLOT_LAT_MIN, PLOT_LAT_MAX], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = plt.matplotlib.ticker.MultipleLocator(4)
            gl.ylocator = plt.matplotlib.ticker.MultipleLocator(4)
            gl.xlabel_style = {'size': FONT_SIZES['label']}
            gl.ylabel_style = {'size': FONT_SIZES['label']}
            if i > 0:
                gl.left_labels = False
            axes.append(ax)

        for idx, (date_str, data_array) in enumerate(zip(date_list, data_arrays)):
            ax = axes[idx]
            ax.add_feature(cfeature.LAND, color='#F5F5DC', alpha=0.3)

            if data_array is None:
                continue

            data_display = data_array.copy()
            data_display[data_display <= 0] = np.min(data_display[data_display > 0])
            display_data = np.log10(data_display)

            im = ax.pcolormesh(lon_mesh, lat_mesh, display_data.T,
                                cmap=cmap, vmin=global_min, vmax=global_max,
                                transform=ccrs.PlateCarree(), shading='auto')

            threshold = self.thresholds['level_1']
            mask_level = data_array >= threshold

            if np.any(mask_level):
                ax.contour(lon_mesh, lat_mesh, data_array.T,
                        levels=[threshold], colors=['black'], linewidths=2.5,
                        transform=ccrs.PlateCarree())
                ax.contourf(lon_mesh, lat_mesh, data_array.T,
                        levels=[threshold, np.max(data_array)*2],
                        colors=['red'], alpha=0.2,
                        transform=ccrs.PlateCarree())

            try:
                province_file = os.path.join(GEO_DIR, 'province.geojson')
                if os.path.exists(province_file):
                    provinces = gpd.read_file(province_file)
                    ax.add_geometries(
                        provinces.geometry,
                        crs=ccrs.PlateCarree(),
                        facecolor='none',
                        edgecolor='gray',
                        linewidth=1,
                        zorder=10,
                    )
                else:
                    print(f"Province boundary file not found: {os.path.join(GEO_DIR, 'province.geojson')}")
            except Exception as e:
                print(f"Warning: Cannot load province boundaries: {e}")

            npp_lon = (self.config['x_min'] + self.config['x_max']) / 2
            npp_lat = (self.config['y_min'] + self.config['y_max']) / 2
            try:
                if hasattr(self, 'npp_name') and self.npp_name in NPP:
                    npp_info = NPP[self.npp_name]
                    npp_lon = float(npp_info['Abscissa'])
                    npp_lat = float(npp_info['Ordinate'])
            except:
                pass
            ax.plot(npp_lon, npp_lat, '*', color='black', markersize=20,
                transform=ccrs.PlateCarree(), zorder=12)

            time_label = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {date_str[8:10]}:00"
            ax.text(0.5, -0.1, time_label, transform=ax.transAxes,
                fontsize=FONT_SIZES['legend'],
                ha='center', va='top')

            ax.text(0.42, 0.015, "GS (2024) 0650", transform=ax.transAxes,
                fontsize=FONT_SIZES['legend']-1,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.2))

        cbar_ax = fig.add_axes([0.524, 0.25, 0.004, 0.62])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        ticks = range(int(global_min), math.ceil(global_max) )
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(x)}}}$'))
        cbar.ax.tick_params(labelsize=FONT_SIZES['label'])
        cbar.set_label(label=label_name, fontsize=FONT_SIZES['title']-1)

        plt.subplots_adjust(left=0.1, right=0.52, bottom=0.18, top=0.95, wspace=0.01)

        filename = f'{event_type}_{self.analysis_mode}_cr{release_duration}_{plot_type}_combined.png'
        filepath = save_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Combined concentration map saved: {filename}")

def main():
    """Run the dispersion analysis workflow."""
    print("=" * 60)
    print("DISPERSION ANALYSIS")
    print("=" * 60)
    

    sample_ratio = 0.276
    wind_thresh = 5.659
    precip_thresh = 1.762
    

    analyzer = DispersionAnalyzer(
        analysis_mode='48h', 
        wind_threshold=wind_thresh,
        precip_threshold=precip_thresh
    )
    SELECTED_EVENTS = {
        'BG': ["20230103170000", "20230306220000", "20230429140000", "20230917090000"],
        'SW': ["20230219080000", "20230321000000", "20230528220000", "20230704100000"],
        'HR': ["20230325040000", "20230405100000", "20230613180000", "20230816050000"],
        'TC': ["20230727150000", "20230831040000", "20230903190000", "20231006010000"]
    }
    

    print(f"\n{'='*50}")
    print("FULL YEAR ANALYSIS")
    print(f"{'='*50}")
    analyzer.analyze_all_dates(start_date='20230101', end_date='20231231')
    print(f"\nFull year analysis complete")
    

    release_durations=[12,24,36,48]
    analyzer.analyze_selected_events(SELECTED_EVENTS, release_durations=release_durations, save_dir=SAVE_DIR)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
