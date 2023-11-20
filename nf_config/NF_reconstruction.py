import logging
import os
from .config import Config


logger = logging.getLogger('hexrd.config')


class NF_ReconstructionConfig(Config):
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def tomography(self):
        key = self._cfg.get('NF_reconstruction:tomography:mask_filepath', None)
        if key is not None:
            parms = dict(mask_filepath='NF_reconstruction:tomography:mask_filepath',
                         vertical_motor_position='NF_reconstruction:tomography:vertical_motor_position',
                         use_single_layer='NF_reconstruction:tomography:use_single_layer')
            return parms
        else:
            return None

    @property
    def cross_sectional_dimensions(self):
        return self._cfg.get('NF_reconstruction:cross_sectional_dimensions')

    @property
    def voxel_spacing(self):
        return self._cfg.get('NF_reconstruction:voxel_spacing')

    @property
    def desired_vertical_span(self):
        return self._cfg.get('NF_reconstruction:desired_vertical_span', 
                             self._cfg.get('experiment:beam_vertical_span'))

    @property
    def beam_stop(self):
        key = self._cfg.get('NF_reconstruction:beam_stop:beam_stop_filepath', None)
        if key is not None:
            return key
        else:
            parms = [self._cfg.get('NF_reconstruction:beam_stop:beam_stop_vertical_center', 0.0),
                         self._cfg.get('NF_reconstruction:beam_stop:beam_stop_vertical_span', 0.3)]
            return parms
