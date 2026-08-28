#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File: data_preprocessor.py
# @Date: 2026-06-28
# @Author: Lubber
# @Description: Preprocess meteorological and land-use data for Polyphemus.

import os
from pathlib import Path
from datetime import datetime, date, timedelta
from src.config import *

class DataPreprocessor:
    """
    Handles preprocessing of raw data for Polyphemus.
    """

    def __init__(self, start_time, total_days):
        """
        Initialize the DataPreprocessor.

        Args:
            start_time (str, YYYYMMDD or YYYYMMDDHHMMSS): date of the start time.
            total_days (int): number of days.
        """
        if len(start_time) == 14:
            fmt = '%Y%m%d%H%M%S'
        else:
            fmt = '%Y%m%d'
        self.start_time = datetime.strptime(start_time, fmt)
        self.total_days = total_days
        self.data_dir = SCENE_DIR + f'data_{start_time}_{total_days-1}/'
        self.results_dir = SCENE_DIR + f'results_{start_time}_{total_days-1}/'
        self.config_dir = SCENE_DIR + 'config/'

        self.general_cfg_path = f'{self.config_dir}general.cfg'
        self.meteo_cfg_path = f'{self.config_dir}meteo.cfg'
        self.general_cfg = ConfigHandler(self.general_cfg_path)
        self.meteo_cfg = ConfigHandler(self.meteo_cfg_path)
        updates = {
            'general': {
                'Directory_data': f'data_{self.start_time.strftime("%Y%m%d%H%M%S")}_{self.total_days-1}'
            },
            'domain': {
                'Date': f'{self.start_time.strftime("%Y-%m-%d_00-00-00")}',
                'x_min': str(REGION_BOUND[1]),
                'Nx': str(int((REGION_BOUND[3] - REGION_BOUND[1])/0.2+1)),
                'y_min': str(REGION_BOUND[2]),
                'Ny': str(int((REGION_BOUND[0] - REGION_BOUND[2])/0.2+1))
            }
        }
        self.general_cfg.update_config(updates)
        updates = {
            'paths': {
                'Database_meteo': f'raw_data/meteo/{REGION_NAME}/'
            },
            'ECMWF': {
                'Date': f'{self.start_time.strftime("%Y-%m-%d_00-00-00")}',
                'x_min': str(REGION_BOUND[1]),
                'Nx': str(int((REGION_BOUND[3] - REGION_BOUND[1])/0.25+1)),
                'y_min': str(REGION_BOUND[2]),
                'Ny': str(int((REGION_BOUND[0] - REGION_BOUND[2])/0.25+1))
            }
        }
        self.meteo_cfg.update_config(updates)
        self._create_directory()


    def _create_directory(self):
        """
        Create subdirectories within the base directory.
        """
        # data directory
        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        # meteo data directory
        meteoPath = Path(self.data_dir + "meteo")
        meteoPath.mkdir(parents=True, exist_ok=True)

        # Kz_TM data directory
        kzTMPath = Path(self.data_dir + "meteo/Kz_TM")
        kzTMPath.mkdir(parents=True, exist_ok=True)

        # dep data directory
        depPath = Path(self.data_dir + "dep")
        depPath.mkdir(parents=True, exist_ok=True)

        # emission data directory
        emissionPath = Path(self.data_dir + "emissions")
        emissionPath.mkdir(parents=True, exist_ok=True)

        # ground directory
        groundPath = Path(self.data_dir + "ground")
        groundPath.mkdir(parents=True, exist_ok=True)
        

    def handle_meteo_data(self):
        """
        Preprocess meteorological data using Polyphemus.
        """
        dep_cfg_path = f'{self.config_dir}dep.cfg'
        # dep_cfg = ConfigHandler(dep_cfg_dir)
        
        for day in range(self.total_days):
            current_day = self.start_time + timedelta(days=day)
            next_day = current_day + timedelta(days=1)
            print('Current Day:' + current_day.isoformat())
            updates = {
                'domain': {
                    'Date': f'{self.start_time.strftime("%Y-%m-%d_00-00-00")}'
                }
            }
            self.general_cfg.update_config(updates)

            try:
                # Process the meteo data
                os.system(f"{METEO_DIR}meteo_parallel {self.general_cfg_path} {self.meteo_cfg_path} {current_day.strftime('%Y-%m-%d')}")
                print(f"{METEO_DIR}meteo_parallel {self.general_cfg_path} {self.meteo_cfg_path} {current_day.strftime('%Y-%m-%d')}")
                # Estimate Kz
                os.system(f"{METEO_DIR}Kz {self.general_cfg_path} {self.meteo_cfg_path} {current_day.strftime('%Y-%m-%d')}")
                
                # Estimate Kz_TM
                os.system(f"{METEO_DIR}Kz_TM {self.general_cfg_path} {self.meteo_cfg_path} {current_day.strftime('%Y-%m-%d')} {next_day.strftime('%Y-%m-%d')}")

                # Estimate dep
                os.system(f"{DEP_DIR}dep_aerosol {self.general_cfg_path} {dep_cfg_path} {current_day.strftime('%Y-%m-%d')} {next_day.strftime('%Y-%m-%d')}")
            
            except RuntimeError as e:
                print(f"Error encountered: {e}")
                return False
        print("meteorological data has been preprocessed successfully")
        return True

    def handle_land_use_data(self):
        """
        Preprocess land use data using Polyphemus.
        """
        groundFile = Path(self.data_dir + "ground/LUC-glcf.bin")
        if not groundFile.is_file():
            os.system(f"{GROUND_DIR}luc-glcf {self.general_cfg_path} {self.config_dir}luc-glcf.cfg")
            
        # covert the landuse data
        groundZhangFile = Path(self.data_dir + "ground/LUC-glcf-zhang.bin")
        if not groundZhangFile.is_file():
            os.system(f"{GROUND_DIR}luc-convert  {self.general_cfg_path} {self.config_dir}glcf_to_zhang.cfg")

        # roughness
        roughnessFile = Path(self.data_dir + "ground/Roughness-glcf.bin")
        if not roughnessFile.is_file():
            os.system(f"{GROUND_DIR}roughness {self.general_cfg_path} {self.config_dir}roughness.cfg")
        print("land-use data has been preprocessed successfully")




if __name__ == "__main__":
    start_time = date(2023, 2, 27).strftime('%Y%m%d%H%M%S')
    total_days = 1
    data_handler = DataPreprocessor(start_time, total_days)
    data_handler.handle_land_use_data()
    data_handler.handle_meteo_data()
