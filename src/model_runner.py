# -*- coding: utf-8 -*-
# @File: model_runner.py
# @Date: 2026-06-28
# @Author: Lubber
# @Description: Execute the Polair3D atmospheric dispersion model.

import os
from pathlib import Path
from datetime import datetime, timedelta
from src.config import SCENE_DIR, REGION_BOUND, ConfigHandler, POLAIR_DIR

TOTAL_RELEASES_BQ = {
    "Cs137": 2e16,
    "I131": 1e17,
}


class ModelRunner:
    """
    Handles execution of the Polair3D atmospheric dispersion model.
    """

    def __init__(self, start_time: str, total_days: int, pos_info, release_duration: int = 24):
        """
        Initialize the ModelRunner.

        Args:
            start_time (str): Start date in 'YYYYMMDDHHMMSS' format.
            total_days (int): Total days to simulate.
            pos_info: Position information dict.
            release_duration (int): Release duration in hours (12, 24, 36, or 48).
        """
        if len(start_time) == 14:
            fmt = '%Y%m%d%H%M%S'
        else:
            fmt = '%Y%m%d'
        self.start_time = datetime.strptime(start_time, fmt)
        print(self.start_time)
        self.total_days = total_days
        self.release_duration = release_duration
        self.config_dir = SCENE_DIR + 'config/'
        self.results_dir = f"results_{start_time}_{total_days}_cr{release_duration}/"
        self.pos_info = pos_info
        results_path = Path(SCENE_DIR + self.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

    def _update_model_config(self):
        """
        Update model configuration files for the simulation day.

        Args:
            current_day (datetime): Current simulation day.
        """
        # results directory
        cfg = ConfigHandler(self.config_dir + 'case.cfg')

        updates = {
            'domain': {
                'Date_min': self.start_time.strftime('%Y-%m-%d_%H-%M-%S'),
                'Nt': int(self.total_days*24*3600/float(cfg.config['domain']['Delta_t'])),
                'x_min': str(REGION_BOUND[1]),
                'Nx': str(int((REGION_BOUND[3] - REGION_BOUND[1])/0.2+1)),
                'y_min': str(REGION_BOUND[2]),
                'Ny': str(int((REGION_BOUND[0] - REGION_BOUND[2])/0.2+1))
            }
        }
        cfg.update_config(updates)

        cfg = ConfigHandler(self.config_dir + 'case-data.cfg')

        updates = {
            'meteo': {
                'Directory_data': f"data_{self.start_time.strftime('%Y%m%d%H%M%S')}_{self.total_days}",
                'Date_min': self.start_time.strftime('%Y-%m-%d_00-00-00')
            },
            'deposition': {
                'Date_min': self.start_time.strftime('%Y-%m-%d_00-00-00')
            }
        }
        cfg.update_config(updates)

        cfg = ConfigHandler(self.config_dir + 'case-saver.cfg')

        updates = {
            'save': {
                'NPP': self.pos_info['name'],
                'Directory_results': self.results_dir
            }
        }
        cfg.update_config(updates)

        cfg_path = self.config_dir + 'source.dat'
        cfg = ConfigHandler(cfg_path)
        end_day = self.start_time + timedelta(hours=self.release_duration, minutes=-1)

        source_updates = {}
        for species, total_release in TOTAL_RELEASES_BQ.items():
            rate = total_release / (self.release_duration * 3600)
            source_updates[species] = {
                'Abscissa': self.pos_info['Abscissa'],
                'Ordinate': self.pos_info['Ordinate'],
                'Date_beg': self.start_time.strftime('%Y-%m-%d_%H-%M-%S'),
                'Date_end': end_day.strftime('%Y-%m-%d_%H-%M-%S'),
                'Date_min_file': self.start_time.strftime('%Y-%m-%d_%H-%M-%S'),
                'Rate': f"{rate:.3e}"
            }

        updates = {
            'source': source_updates
        }
        cfg.update_config(updates)

    def run_model(self):
        """
        Execute the Polair3D model for each day.
        """
        print(f"Running model of {self.start_time.strftime('%Y-%m-%d_%H-%M-%S')} for {self.total_days} days, release_duation:{self.release_duration}h")
        self._update_model_config()
        # Build command
        case_cfg = self.config_dir + 'case.cfg'
        command = f"{POLAIR_DIR}polair3d {case_cfg}"
        
        # Run model
        try:
            exit_code = os.system(command)
            if exit_code != 0:
                raise RuntimeError(f"Model failed with exit code {exit_code}")
            
            print(f"Model run completed")
            
        except Exception as e:
            print(f"Error during model execution: {str(e)}")
