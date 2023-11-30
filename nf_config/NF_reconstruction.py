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
    
    @property
    def misorientation(self):
        key = self._cfg.get(
            'experiment:misorientation:use_misorientation', False)
        if key is True:
            parms = dict(misorientation_bnd=self.get('experiment:misorientation:bound', 0.0),
                         misorientation_spacing=self.get('experiment:misorientation:spacing', 0.25))
            return parms
        else:
            return
        
    @property
    def missing_grains(self):
        key = self._cfg.get('NF_reconstruction:missing_grains', None)
        if key is not None:
            parms = dict(reconstructed_data_path = self._cfg.get('NF_reconstruction:missing_grains:reconstructed_data_path'),
                         ori_grid_spacing = self._cfg.get('NF_reconstruction:missing_grains:ori_grid_spacing'),
                         confidence_threshold = self._cfg.get('NF_reconstruction:missing_grains:confidence_threshold'),
                         low_confidence_sparsing = self._cfg.get('NF_reconstruction:missing_grains:low_confidence_sparsing'),
                         errode_free_surface = self._cfg.get('NF_reconstruction:missing_grains:errode_free_surface'),
                         coord_cutoff_scale = self._cfg.get('NF_reconstruction:missing_grains:coord_cutoff_scale'),
                         iter_cutoff = self._cfg.get('NF_reconstruction:missing_grains:iter_cutoff'),
                         re_run_and_save = self._cfg.get('NF_reconstruction:missing_grains:re_run_and_save'))
            return parms
        else:
            return None

    @property
    def centroid_serach_radius(self):
        return self._cfg.get('NF_reconstruction:centroid_serach_radius')
    
    @property
    def expand_radius_confidence_threshold(self):
        return self._cfg.get('NF_reconstruction:expand_radius_confidence_threshold')
        