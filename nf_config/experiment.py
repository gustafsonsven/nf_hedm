import logging
import os
import numpy as np

from .config import Config


logger = logging.getLogger('hexrd.config')


class ExperimentConfig(Config):
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def beam_energy(self):
        return self._cfg.get('experiment:beam_energy')
    
    @property
    def beam_vertical_span(self):
        return self._cfg.get('experiment:beam_vertical_span')
    
    @property
    def material_name(self):
        return self._cfg.get('experiment:material_name')

    @property
    def point_group_number(self):
        return self._cfg.get('experiment:point_group_number')

    @property
    def max_tth(self):
        if self._cfg.get('experiment:max_tth') == 'None':
            return None
        else:
            return self._cfg.get('experiment:max_tth', None)

    @property
    def comp_thresh(self):
        temp = self._cfg.get('experiment:comp_thresh', None)
        if temp is None:
            return temp
        elif np.logical_and(temp <= 1.0, temp > 0.0):
            return temp
        else:
            raise RuntimeError(
                'comp_thresh must be None or a number between 0 and 1')

    @property
    def chi2_thresh(self):
        temp = self._cfg.get('experiment:chi2_thresh', None)
        if temp is None:
            return temp
        elif np.logical_and(temp <= 1.0, temp > 0.0):
            return temp
        else:
            raise RuntimeError(
                'chi2_thresh must be None or a number between 0 and 1')

    @property
    def omega_start(self):
        return self._cfg.get('experiment:images_and_omegas:omega_start', 0.0)

    @property
    def omega_stop(self):
        return self._cfg.get('experiment:images_and_omegas:omega_stop', 360.0)
    
    @property
    def shift_images_in_omega(self):
        return self._cfg.get('experiment:images_and_omegas:shift_images_in_omega', 0)

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
