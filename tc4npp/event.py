# @File: event.py
# @Date: 2026-06-28
# @Author: Lubber
"""Extract ERA5 time series and plot 48-hour weather maps."""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import pygrib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.config import SCENE_DIR, GEO_DIR, SITE_NAME, SITE_LON, SITE_LAT

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FONT_SIZES = {
    'title': 40,
    'subtitle': 40,
    'label': 35,
    'legend': 35
}
SITE_RADIUS_DEG = 1

REGION_NAME = 'Southeast'
METEO_DIR = os.path.join(SCENE_DIR, f'raw_data/meteo/{REGION_NAME}/')
OUT_PLOT_DIR = PACKAGE_ROOT / "figures"
TIMESERIES_CSV = PACKAGE_ROOT / "data" / f"timeseries_data_2023-{SITE_RADIUS_DEG}.csv"


PLOT_LON_MIN = 108.0
PLOT_LON_MAX = 123.0
PLOT_LAT_MIN = 18.0
PLOT_LAT_MAX = 33.0


SELECTED_EVENTS = {
        'BG': ["20230103170000", "20230306220000", "20230429140000", "20230917090000"],
        'SW': ["20230219080000", "20230321000000", "20230528220000", "20230704100000"],
        'HR': ["20230325040000", "20230405100000", "20230613180000", "20230816050000"],
        'TC': ["20230727150000", "20230831040000", "20230903190000", "20231006010000"]
    }
def get_grib_files(start_dt: datetime, end_dt: datetime, dataset='single') -> List[str]:
    """Find daily or monthly GRIB files covering the requested period."""
    files = []
    current_dt = start_dt
    while current_dt < end_dt:
        if (end_dt - start_dt).days <= 3:
            prefix = 'ECMWF'
            filename = os.path.join(METEO_DIR, f"{prefix}-{current_dt.strftime('%Y%m%d')}.grb")
            files.append(filename)
            current_dt += timedelta(days=1)
        else:
            year_month = current_dt.strftime('%Y%m')
            prefix = 'SingleLevel' if dataset == 'single' else 'PressureLevel'
            filename = os.path.join(METEO_DIR, f"{prefix}_{year_month}01.grib")
            if filename not in files:
                files.append(filename)
            current_dt = (current_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
    return files

def region_slice_indices_from_grb(grb, lat_min, lat_max, lon_min, lon_max):
    """Return the GRIB grid indices covering the requested region."""
    lats, lons = grb.latlons()
    lat_vals = lats[:, 0]
    lon_vals = lons[0, :]
    
    if lat_vals[0] > lat_vals[-1]:
        lat_idx_min = int(np.searchsorted(lat_vals[::-1], lat_min, side='left'))
        lat_idx_max = int(np.searchsorted(lat_vals[::-1], lat_max, side='right')) - 1
        lat_idx_min, lat_idx_max = len(lat_vals)-1-lat_idx_min, len(lat_vals)-1-lat_idx_max
        if lat_idx_min > lat_idx_max:
            lat_idx_min, lat_idx_max = lat_idx_max, lat_idx_min
    else:
        lat_idx_min = int(np.searchsorted(lat_vals, lat_min, side='left'))
        lat_idx_max = int(np.searchsorted(lat_vals, lat_max, side='right')) - 1
    
    lon_idx_min = int(np.searchsorted(lon_vals, lon_min, side='left'))
    lon_idx_max = int(np.searchsorted(lon_vals, lon_max, side='right')) - 1
    
    lat_idx_min = max(0, lat_idx_min)
    lat_idx_max = min(lats.shape[0]-1, lat_idx_max)
    lon_idx_min = max(0, lon_idx_min)
    lon_idx_max = min(lons.shape[1]-1, lon_idx_max)
    
    return lat_idx_min, lat_idx_max, lon_idx_min, lon_idx_max, lats, lons


def extract_and_save_timeseries(start_date: str, end_date: str):
    """
    Extract hourly time series and save them to a CSV file.
    """
    print("[INFO] Extracting time series data from GRIB files...")
    
    lon_min_site = SITE_LON - SITE_RADIUS_DEG
    lon_max_site = SITE_LON + SITE_RADIUS_DEG
    lat_min_site = SITE_LAT - SITE_RADIUS_DEG
    lat_max_site = SITE_LAT + SITE_RADIUS_DEG
    
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    

    hourly_lsp = {}      # large-scale precipitation
    hourly_cp = {}       # convective precipitation
    hourly_precip = {}   # total precipitation
    hourly_wind_u = {}   # u component
    hourly_wind_v = {}   # v component
    hourly_wind_u_point = {}   # u component at point
    hourly_wind_v_point = {}   # v component at point
    hourly_lsp_point = {}      # large-scale precipitation at point
    hourly_cp_point = {}       # convective precipitation at point
    hourly_precip_point = {}   # total precipitation at point
    

    print("[STEP] Extracting precipitation...")
    extended_start = start_dt - timedelta(days=1)
    files = get_grib_files(extended_start, end_dt, dataset='single')
    

    for fname in files:
        if not os.path.exists(fname):
            continue
        with pygrib.open(fname) as grbs:

            try:
                lsp_msgs = grbs.select(name='Large-scale precipitation')
            except:
                lsp_msgs = []
            

            try:
                cp_msgs = grbs.select(name='Convective precipitation')
            except:
                cp_msgs = []
            

            for msg in lsp_msgs:
                actual_time = _get_precipitation_time(msg)
                if not (start_dt <= actual_time < end_dt):
                    continue
                
                try:
                    val = _extract_precipitation_value(msg, lat_min_site, lat_max_site, 
                                                     lon_min_site, lon_max_site)
                    if val is None:
                        continue
                    

                    hourly_val = val
                    
                    if actual_time not in hourly_lsp:
                        hourly_lsp[actual_time] = 0
                    hourly_lsp[actual_time] += hourly_val

                    point_val = _extract_point_precipitation_value(msg, SITE_LAT, SITE_LON)
                    if point_val is not None:
                        if actual_time not in hourly_lsp_point:
                            hourly_lsp_point[actual_time] = 0
                        hourly_lsp_point[actual_time] += point_val

                except Exception as e:
                    print(f"[WARN] Failed to process LSP message: {e}")
                    continue


            for msg in cp_msgs:
                actual_time = _get_precipitation_time(msg)
                if not (start_dt <= actual_time < end_dt):
                    continue
                
                try:
                    val = _extract_precipitation_value(msg, lat_min_site, lat_max_site, 
                                                     lon_min_site, lon_max_site)
                    if val is None:
                        continue
                    
                    hourly_val = val
                    
                    if actual_time not in hourly_cp:
                        hourly_cp[actual_time] = 0
                    hourly_cp[actual_time] += hourly_val

                    point_val = _extract_point_precipitation_value(msg, SITE_LAT, SITE_LON)
                    if point_val is not None:
                        if actual_time not in hourly_cp_point:
                            hourly_cp_point[actual_time] = 0
                        hourly_cp_point[actual_time] += point_val

                except Exception as e:
                    print(f"[WARN] Failed to process CP message: {e}")
                    continue


    all_times = set(hourly_lsp.keys()) | set(hourly_cp.keys())
    for t in all_times:
        lsp_val = hourly_lsp.get(t, 0)
        cp_val = hourly_cp.get(t, 0)
        hourly_precip[t] = lsp_val + cp_val
        lsp_point_val = hourly_lsp_point.get(t, 0)
        cp_point_val = hourly_cp_point.get(t, 0)
        hourly_precip_point[t] = lsp_point_val + cp_point_val
    

    print("[STEP] Extracting wind components...")
    files = get_grib_files(start_dt, end_dt, dataset='pressure')
    
    for fname in files:
        if not os.path.exists(fname):
            continue
        with pygrib.open(fname) as grbs:
            try:
                u_msgs = grbs.select(name='U component of wind', typeOfLevel='isobaricInhPa', level=850)
                v_msgs = grbs.select(name='V component of wind', typeOfLevel='isobaricInhPa', level=850)
            except:
                continue
            
            for u_msg, v_msg in zip(u_msgs, v_msgs):
                t = u_msg.validDate
                if not (start_dt <= t < end_dt):
                    continue
                
                try:
                    lat_i0, lat_i1, lon_j0, lon_j1, lats, lons = region_slice_indices_from_grb(
                        u_msg, lat_min_site, lat_max_site, lon_min_site, lon_max_site)
                    
                    u_data = u_msg.values[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                    v_data = v_msg.values[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                    
                    if u_data.size == 0:
                        continue
                    

                    lat_slice = lats[lat_i0:lat_i1+1, 0]
                    weights = np.cos(np.deg2rad(lat_slice))[:, None]
                    u_mean = np.nanmean(u_data * weights)
                    v_mean = np.nanmean(v_data * weights)
                    
                    hourly_wind_u[t] = u_mean
                    hourly_wind_v[t] = v_mean
                    
                    u_point = _extract_point_wind_value(u_msg, SITE_LAT, SITE_LON)
                    v_point = _extract_point_wind_value(v_msg, SITE_LAT, SITE_LON)
                    if u_point is not None and v_point is not None:
                        hourly_wind_u_point[t] = u_point
                        hourly_wind_v_point[t] = v_point
                except Exception as e:
                    print(f"[WARN] Failed to process wind data: {e}")
                    continue
    

    print(f"[INFO] Found {len(hourly_precip)} total precipitation records")
    print(f"[INFO] Found {len(hourly_lsp)} LSP records and {len(hourly_cp)} CP records")
    print(f"[INFO] Found {len(hourly_wind_u)} wind records")
    

    full_index = pd.date_range(start=start_dt, end=end_dt - timedelta(hours=1), freq='h')
    

    precip_series = pd.Series(hourly_precip).reindex(full_index, fill_value=0)
    lsp_series = pd.Series(hourly_lsp).reindex(full_index, fill_value=0)
    cp_series = pd.Series(hourly_cp).reindex(full_index, fill_value=0)
    u_series = pd.Series(hourly_wind_u).reindex(full_index, fill_value=np.nan)
    v_series = pd.Series(hourly_wind_v).reindex(full_index, fill_value=np.nan)
    u_point_series = pd.Series(hourly_wind_u_point, dtype=float).reindex(full_index, fill_value=np.nan)
    v_point_series = pd.Series(hourly_wind_v_point, dtype=float).reindex(full_index, fill_value=np.nan)
    precip_point_series = pd.Series(hourly_precip_point, dtype=float).reindex(full_index, fill_value=0)
    

    wind_speed_series = pd.Series(np.sqrt(np.asarray(u_series, dtype=float)**2 + np.asarray(v_series, dtype=float)**2))
    wind_speed_series = wind_speed_series.ffill().bfill().fillna(0)
    wind_speed_point_series = pd.Series(np.sqrt(np.asarray(u_point_series, dtype=float)**2 + np.asarray(v_point_series, dtype=float)**2))
    wind_speed_point_series = wind_speed_point_series.ffill().bfill().fillna(0)
    u_series = u_series.ffill().bfill().fillna(0)
    v_series = v_series.ffill().bfill().fillna(0)
    u_point_series = u_point_series.ffill().bfill().fillna(0)
    v_point_series = v_point_series.ffill().bfill().fillna(0)
    

    rolling_data = []
    for i in range(len(full_index)):
        current_time = full_index[i]
        

        window_end = min(i + 48, len(full_index))
        

        precip_48h = precip_series.iloc[i:window_end].sum()
        precip_48h_point = precip_point_series.iloc[i:window_end].sum()
        

        wind_48h_mean = wind_speed_series.iloc[i:window_end].mean()
        wind_u_48h_mean = u_series.iloc[i:window_end].mean()
        wind_v_48h_mean = v_series.iloc[i:window_end].mean()
        wind_u_point_48h_mean = u_point_series.iloc[i:window_end].mean()
        wind_v_point_48h_mean = v_point_series.iloc[i:window_end].mean()
        wind_speed_point_48h_mean = wind_speed_point_series.iloc[i:window_end].mean()
        

        wind_48h_max = wind_speed_series.iloc[i:window_end].max()
        max_wind_idx = wind_speed_series.iloc[i:window_end].values.argmax() + i
        max_wind_time = full_index[max_wind_idx]
        
        rolling_data.append({
            'datetime': current_time,
            'precip_48h_total': precip_48h,
            'precip_48h_point': precip_48h_point,
            'wind_48h_mean': wind_48h_mean,
            'wind_u_48h_mean': wind_u_48h_mean,
            'wind_v_48h_mean': wind_v_48h_mean,
            'wind_48h_max': wind_48h_max,
            'wind_max_time': max_wind_time,

            'precip_hourly': precip_series.iloc[i],
            'precip_hourly_point': precip_point_series.iloc[i],
            'wind_u': u_series.iloc[i],
            'wind_v': v_series.iloc[i],
            'wind_speed': wind_speed_series.iloc[i],
            'wind_u_point': u_point_series.iloc[i],
            'wind_v_point': v_point_series.iloc[i],
            'wind_speed_point': wind_speed_point_series.iloc[i],
            'wind_u_point_48h_mean': wind_u_point_48h_mean,
            'wind_v_point_48h_mean': wind_v_point_48h_mean,
            'wind_speed_point_48h_mean': wind_speed_point_48h_mean
        })
    
    df = pd.DataFrame(rolling_data)
    

    df.to_csv(TIMESERIES_CSV, index=False)
    print(f"[OK] Time series data saved to {TIMESERIES_CSV}")
    print(f"    Data shape: {df.shape}")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    print(f"    Total precipitation range: {df['precip_48h_total'].min():.1f} - {df['precip_48h_total'].max():.1f} mm")
    print(f"    Wind speed range: {df['wind_speed_point_48h_mean'].min():.1f} - {df['wind_speed_point_48h_mean'].max():.1f} m/s")
    
    return df

def _get_precipitation_time(msg):
    """Return the GRIB message valid time."""
    return msg.validDate

def _extract_precipitation_value(msg, lat_min, lat_max, lon_min, lon_max):
    """Return the area-averaged precipitation value in mm."""
    try:
        lat_i0, lat_i1, lon_j0, lon_j1, lats, lons = region_slice_indices_from_grb(
            msg, lat_min, lat_max, lon_min, lon_max)
        
        data = msg.values[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
        if data.size == 0:
            return None
        

        lat_slice = lats[lat_i0:lat_i1+1, 0]
        weights = np.cos(np.deg2rad(lat_slice))[:, None]
        val = np.nanmean(data * weights) * 1000
        
        return val
    except Exception as e:
        return None

def _extract_point_wind_value(msg, target_lat, target_lon):
    try:
        lats, lons = msg.latlons()
        lat_vals = np.unique(lats[:, 0])
        lon_vals = np.unique(lons[0, :])
        lat_idx = int(np.searchsorted(lat_vals, target_lat, side='right')) - 1
        lon_idx = int(np.searchsorted(lon_vals, target_lon, side='right')) - 1
        lat_idx = max(0, min(lat_idx, len(lat_vals) - 2))
        lon_idx = max(0, min(lon_idx, len(lon_vals) - 2))
        y1, y2 = lat_vals[lat_idx], lat_vals[lat_idx + 1]
        x1, x2 = lon_vals[lon_idx], lon_vals[lon_idx + 1]
        lat_row_1 = len(lat_vals) - 1 - lat_idx
        lat_row_2 = len(lat_vals) - 1 - (lat_idx + 1)
        v11 = msg.values[lat_row_1, lon_idx]
        v12 = msg.values[lat_row_1, lon_idx + 1]
        v21 = msg.values[lat_row_2, lon_idx]
        v22 = msg.values[lat_row_2, lon_idx + 1]
        fx = (target_lon - x1) / (x2 - x1)
        fy = (target_lat - y1) / (y2 - y1)
        return (1-fx)*(1-fy)*v11 + fx*(1-fy)*v12 + (1-fx)*fy*v21 + fx*fy*v22
    except Exception:
        return None

def _extract_point_precipitation_value(msg, target_lat, target_lon):
    try:
        lats, lons = msg.latlons()
        lat_vals = np.unique(lats[:, 0])
        lon_vals = np.unique(lons[0, :])
        lat_idx = int(np.searchsorted(lat_vals, target_lat, side='right')) - 1
        lon_idx = int(np.searchsorted(lon_vals, target_lon, side='right')) - 1
        lat_idx = max(0, min(lat_idx, len(lat_vals) - 2))
        lon_idx = max(0, min(lon_idx, len(lon_vals) - 2))
        y1, y2 = lat_vals[lat_idx], lat_vals[lat_idx + 1]
        x1, x2 = lon_vals[lon_idx], lon_vals[lon_idx + 1]
        lat_row_1 = len(lat_vals) - 1 - lat_idx
        lat_row_2 = len(lat_vals) - 1 - (lat_idx + 1)
        v11 = msg.values[lat_row_1, lon_idx]
        v12 = msg.values[lat_row_1, lon_idx + 1]
        v21 = msg.values[lat_row_2, lon_idx]
        v22 = msg.values[lat_row_2, lon_idx + 1]
        fx = (target_lon - x1) / (x2 - x1)
        fy = (target_lat - y1) / (y2 - y1)
        return ((1-fx)*(1-fy)*v11 + fx*(1-fy)*v12 + (1-fx)*fy*v21 + fx*fy*v22) * 1000
    except Exception:
        return None

def load_or_extract_timeseries(start_date: str, end_date: str):
    """
    Load the cached CSV, or extract the time series when it is unavailable.
    """
    if os.path.exists(TIMESERIES_CSV):
        print(f"[INFO] Loading time series data from {TIMESERIES_CSV}")
        df = pd.read_csv(TIMESERIES_CSV)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['wind_max_time'] = pd.to_datetime(df['wind_max_time'])
        df = df.set_index('datetime')
        

        expected_cols = ['precip_hourly', 'precip_hourly_point', 'precip_48h_total', 'precip_48h_point',
                        'wind_u', 'wind_v', 'wind_speed', 'wind_u_48h_mean', 'wind_v_48h_mean',
                        'wind_u_point', 'wind_v_point', 'wind_speed_point',
                        'wind_u_point_48h_mean', 'wind_v_point_48h_mean', 'wind_speed_point_48h_mean']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"[INFO] Missing columns {missing_cols}, re-extracting from GRIB files...")
            df = extract_and_save_timeseries(start_date, end_date)
            df = df.set_index('datetime')
    else:
        print(f"[INFO] CSV file not found, extracting from GRIB files...")
        df = extract_and_save_timeseries(start_date, end_date)
        df = df.set_index('datetime')
    
    return df



def get_48h_precipitation_2d(start_time: datetime):
    """Return the 48-hour accumulated precipitation field."""
    end_time = start_time + timedelta(hours=48)
    files = get_grib_files(start_time, end_time, dataset='single')
    total_precip = None
    lats_out = None
    lons_out = None
    
    for fname in files:
        if not os.path.exists(fname):
            continue
        with pygrib.open(fname) as grbs:
            try:
                lsp_msgs = grbs.select(name='Large-scale precipitation')
            except:
                lsp_msgs = []
            try:
                cp_msgs = grbs.select(name='Convective precipitation')
            except:
                cp_msgs = []
            
            for msg in lsp_msgs + cp_msgs:
                if start_time <= msg.validDate < end_time:
                    lat_i0, lat_i1, lon_j0, lon_j1, lats, lons = region_slice_indices_from_grb(
                        msg, PLOT_LAT_MIN, PLOT_LAT_MAX, PLOT_LON_MIN, PLOT_LON_MAX)
                    
                    data = msg.values[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                    
                    if total_precip is None:
                        total_precip = np.zeros_like(data)
                        lats_out = lats[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                        lons_out = lons[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                    
                    total_precip += data
    if total_precip is not None:
        total_precip = total_precip * 1000.0
    
    return total_precip, lats_out, lons_out

def get_wind_field_from_csv(date_str: str):
    """Return the 850-hPa wind field at the requested time."""
    target_date = pd.to_datetime(date_str, format='%Y%m%d%H%M%S')

    files = get_grib_files(target_date, target_date + timedelta(hours=1), dataset='pressure')
    
    for fname in files:
        if not os.path.exists(fname):
            continue
        with pygrib.open(fname) as grbs:
            try:
                u_msgs = grbs.select(name='U component of wind', typeOfLevel='isobaricInhPa', level=850)
                v_msgs = grbs.select(name='V component of wind', typeOfLevel='isobaricInhPa', level=850)
            except:
                continue
            
            for u_msg, v_msg in zip(u_msgs, v_msgs):
                if u_msg.validDate == target_date:
                    lat_i0, lat_i1, lon_j0, lon_j1, lats, lons = region_slice_indices_from_grb(
                        u_msg, PLOT_LAT_MIN, PLOT_LAT_MAX, PLOT_LON_MIN, PLOT_LON_MAX)
                    
                    u_data = u_msg.values[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                    v_data = v_msg.values[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                    lats_sub = lats[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                    lons_sub = lons[lat_i0:lat_i1+1, lon_j0:lon_j1+1]
                    
                    return u_data, v_data, lats_sub, lons_sub


def plot_48h_weather_multi(date_list: List[str], event_type: str, output_dir: str):
    """Plot several 48-hour weather maps side by side."""
    n_dates = len(date_list)
    start_times = [pd.to_datetime(ds, format='%Y%m%d%H%M%S') for ds in date_list]

    print(f"[PLOT] {event_type}: plotting 48h weather for {n_dates} dates side-by-side...")


    all_precip = []
    all_wind_u = []
    all_wind_v = []
    all_lats = []
    all_lons = []
    any_valid = False
    for st in start_times:
        print(f"  [DATA] {st.strftime('%Y%m%d%H')}: reading GRIB...")
        p, lats, lons = get_48h_precipitation_2d(st)
        if p is None:
            print(f"  [WARN] No precipitation data, skipping")
            all_precip.append(None)
            all_wind_u.append(None)
            all_wind_v.append(None)
            all_lats.append(None)
            all_lons.append(None)
            continue
        u, v, _, _ = get_wind_field_from_csv(st.strftime('%Y%m%d%H%M%S'))
        any_valid = True
        all_precip.append(p)
        all_wind_u.append(u)
        all_wind_v.append(v)
        all_lats.append(lats)
        all_lons.append(lons)

    if not any_valid:
        print(f"[WARN] No valid data for {event_type}, skipping")
        return

    rain_levels = [1, 10, 25, 50, 100, 150, 200, 250]
    rain_colors = ['#FFFFFF', '#B0E2FF', '#6AC5FF', "#2C7EFF", "#5A5AFF", "#8B2FFF", '#B300FF', '#FF00FF']
    rain_cmap = mcolors.ListedColormap(rain_colors[1:])
    rain_norm = mcolors.BoundaryNorm(rain_levels, rain_cmap.N)

    province_file = os.path.join(GEO_DIR, 'province.geojson')
    provinces = None
    if os.path.exists(province_file):
        provinces = gpd.read_file(province_file)

    fig = plt.figure(figsize=(16 * n_dates, 10))
    proj = ccrs.PlateCarree(central_longitude=115.0)

    cf = None
    for di in range(n_dates):
        ax = fig.add_subplot(1, n_dates, di + 1, projection=proj)
        ax.set_extent([PLOT_LON_MIN, PLOT_LON_MAX, PLOT_LAT_MIN, PLOT_LAT_MAX],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color='#F5F5DC', alpha=0.3)

        if provinces is not None:
            provinces.plot(ax=ax, facecolor='none', edgecolor='gray',
                           linewidth=1, transform=ccrs.PlateCarree(), zorder=10)


        p = all_precip[di]
        lats = all_lats[di]
        lons = all_lons[di]
        if p is not None:
            cf = ax.contourf(lons, lats, p, levels=rain_levels,
                             cmap=rain_cmap, norm=rain_norm, extend='max',
                             transform=ccrs.PlateCarree())


        u = all_wind_u[di]
        v = all_wind_v[di]
        if u is not None and lats is not None:
            ny, nx = u.shape
            skip = max(ny // 20, nx // 20)
            lon_b = lons[skip//2::skip, skip//2::skip]
            lat_b = lats[skip//2::skip, skip//2::skip]
            u_b = u[skip//2::skip, skip//2::skip]
            v_b = v[skip//2::skip, skip//2::skip]
            ax.barbs(lon_b, lat_b, u_b, v_b,
                     length=7, linewidth=1.5,
                     barb_increments=dict(half=2, full=4, flag=20),
                     barbcolor='black', flagcolor='black', alpha=0.8,
                     transform=ccrs.PlateCarree())

        ax.plot(SITE_LON, SITE_LAT, 'r*', markersize=20,
                transform=ccrs.PlateCarree(), zorder=10)


        time_label = start_times[di].strftime('%Y-%m-%d %H:%M')
        ax.text(0.5, -0.07, time_label, transform=ax.transAxes,
                fontsize=FONT_SIZES['legend'],
                ha='center', va='top')

        ax.text(0.42, 0.015, "GS (2024) 0650", transform=ax.transAxes,
                fontsize=FONT_SIZES['legend'] - 1,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.2))

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                          alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': FONT_SIZES['label']}
        gl.ylabel_style = {'size': FONT_SIZES['label']}
        gl.xlocator = plt.matplotlib.ticker.MultipleLocator(4)
        gl.ylocator = plt.matplotlib.ticker.MultipleLocator(4)
        if di > 0:
            gl.left_labels = False


    if cf is not None:
        cbar_ax = fig.add_axes([0.524, 0.25, 0.004, 0.62])
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=FONT_SIZES['label'])
        cbar.set_label('Total Precipitation (mm)', fontsize=FONT_SIZES['title'])

    plt.subplots_adjust(left=0.1, right=0.52, bottom=0.18, top=0.95, wspace=0.01)

    filename = f"{event_type}_48h_weather.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved to {filepath}")



def main():
    t0 = time.time()
    OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    start_date = '20230101'
    end_date = '20240101'
    
    print(f"[RUN] Weather analysis for {SITE_NAME}")
    print(f"Period: {start_date} - {end_date}")
    
    print("\nLoading time series data...")
    df = load_or_extract_timeseries(start_date, end_date)

    print("\nPlotting multi-date 48h weather maps (1×N layout)...")
    for event_type, date_list in SELECTED_EVENTS.items():
        try:
            plot_48h_weather_multi(date_list, event_type, OUT_PLOT_DIR)
        except Exception as e:
            print(f"[ERROR] Failed to plot multi {event_type}: {e}")
            continue
    
    t1 = time.time()
    elapsed = t1 - t0
    print(f"\n[DONE] Total time: {elapsed:.1f} s ({elapsed/60:.2f} min)")

if __name__ == '__main__':
    main()
