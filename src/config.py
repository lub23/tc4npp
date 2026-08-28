#!/usr/bin/env python3
# @File: config.py
# @Date: 2026-06-28
# @Author: Lubber
"""Paths and configuration loading for the TC4NPP workflow."""

import json
import os
import re

# directory that saves the executable program
POLY_DIR = os.path.expanduser("~/Polyphemus-1.11.1/")
GROUND_DIR = POLY_DIR + 'preprocessing/ground/'
DEP_DIR = POLY_DIR + 'preprocessing/dep/'
METEO_DIR = POLY_DIR + 'preprocessing/meteo/'
POLAIR_DIR = POLY_DIR + 'processing/photochemistry/'

REGION_NAME = 'Southeast'
REGION_BOUND = [33, 108, 18, 123]
SCENE_NAME = 'na-nuc'
SCENE_DIR = os.path.expanduser(f"~/TestCase/{SCENE_NAME}/")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO_DIR = os.path.join(BASE_DIR, "geo")

with open(os.path.join(BASE_DIR, "data", "npp.json"), "r", encoding="utf8") as stream:
    NPP = json.load(stream)["NPP"]

SITE_NAME = "Zhangzhou"
SITE_LON = float(NPP[SITE_NAME]["Abscissa"])
SITE_LAT = float(NPP[SITE_NAME]["Ordinate"])

class ConfigHandler:
    def __init__(self, file_path: str):
        """
        initialize
        
        Args:
            param file_path: the path of cfg file
        """
        self.file_path = file_path
        self._config = {}
    
    @property
    def config(self):
        """
        parse .cfg file

        Return:
            config: dict containing config information
        """
        current_section = None
        source_list = []

        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('%'):
                    continue

                # section
                section_match = re.match(r'^\[(.+)\]$', line)
                if section_match:
                    current_section = section_match.group(1)
                    if current_section == 'source':
                        source_list.append({})
                        self._current_source_idx = len(source_list) - 1
                    else:
                        self._config[current_section] = {}
                    continue

                # parse key-value pairs
                if current_section == 'source':
                    key_value_pairs = re.findall(r'(\S+)\s*[=:]\s*([^\s#]+)', line)
                    for key, value in key_value_pairs:
                        source_list[self._current_source_idx][key] = value
                elif current_section:
                    key_value_pairs = re.findall(r'(\S+)\s*[=:]\s*([^\s#]+)', line)
                    for key, value in key_value_pairs:
                        self._config[current_section][key] = value

        source_dict = {}
        for src in source_list:
            src_id = src.get('ID', f'source_{len(source_dict)}')
            source_dict[src_id] = src
        self._config['source'] = source_dict

        self._resolve_markups()
        return self._config

    def _resolve_markups(self):
        """
        resolve markups and replace them with actual values
        """
        for section in self._config:
            for key, value in self._config[section].items():
                if '<' in value and '>' in value:
                    # substitute markups
                    for markup in re.findall(r'<([^>]+)>', value):
                        for sec in self._config:
                            if markup in self._config[sec]:
                                value = value.replace(f'<{markup}>', self._config[sec][markup])
                                break
                    self._config[section][key] = value

    def update_config(self, updates) -> None:
        """
        update the contents of the config file

        Args:
            updates: dict[str, dict[str, str]], dict with updating contents
        """
        for section, section_updates in updates.items():
            if section == 'source':
                if 'source' not in self._config:
                    self._config['source'] = {}
                for source_id, source_props in section_updates.items():
                    if source_id in self._config['source']:
                        self._config['source'][source_id].update(source_props)
                    else:
                        self._config['source'][source_id] = source_props
            else:
                if section in self._config:
                    self._config[section].update(section_updates)
                else:
                    self._config[section] = section_updates

        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        with open(self.file_path, 'w') as file:
            current_section = None
            current_source_id = None

            for line in lines:
                stripped_line = line.strip()

                if stripped_line.startswith('[') and stripped_line.endswith(']'):
                    current_section = stripped_line[1:-1]
                    if current_section == 'source':
                        current_source_id = None
                    file.write(line)
                    continue

                if current_section == 'source':
                    key_value_pairs = re.findall(r'(\S+)\s*[=:]\s*([^\s#]+)', stripped_line)
                    if key_value_pairs:
                        for key, value in key_value_pairs:
                            if key == 'ID':
                                current_source_id = value
                                if 'source' in updates and current_source_id in updates['source']:
                                    new_line = line
                                else:
                                    new_line = line
                                file.write(new_line)
                                break
                            elif current_source_id and 'source' in updates and current_source_id in updates['source']:
                                if key in updates['source'][current_source_id]:
                                    new_value = updates['source'][current_source_id][key]
                                    new_line = re.sub(
                                        rf'({key}\s*[=:]\s*)[^\s#]+',
                                        rf'\g<1>{new_value}',
                                        line
                                    )
                                    file.write(new_line)
                                    break
                                else:
                                    file.write(line)
                                    break
                            else:
                                file.write(line)
                                break
                    else:
                        file.write(line)
                elif current_section in updates:
                    key_value_pairs = re.findall(r'(\S+)\s*[=:]\s*([^\s#]+)', stripped_line)
                    if key_value_pairs:
                        new_line = line
                        for key, _ in key_value_pairs:
                            if key in updates[current_section]:
                                new_value = updates[current_section][key]
                                new_line = re.sub(
                                    rf'({key}\s*[=:]\s*)[^\s#]+',
                                    rf'\g<1>{new_value}',
                                    new_line
                                )
                        file.write(new_line)
                        continue
                    file.write(line)
                else:
                    file.write(line)
