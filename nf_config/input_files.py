import logging
import os

from .config import Config


logger = logging.getLogger('hexrd.config')


class InputConfig(Config):
    def __init__(self, cfg):
        self._cfg = cfg
        
    @property
    def detector_file(self):
        temp = self._cfg.get('input_files:detector_file')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp

    @property
    def materials_file(self):
        temp = self._cfg.get('input_files:materials_file')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp

    @property
    def grains_out_file(self):
        temp = self._cfg.get('input_files:grains_out_file')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp
