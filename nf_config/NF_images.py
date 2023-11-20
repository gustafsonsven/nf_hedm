import logging
import os

from .config import Config


logger = logging.getLogger('hexrd.config')

processing_methods = {
    'gaussian': dict(sigma=2.0, size=3.0),
    'dilations_only': dict(num_erosions=2, num_dilations=3)}


class NF_ImagesConfig(Config):
    def __init__(self, cfg):
        self._cfg = cfg

    @property
    def loading(self):
        return LoadingConfig(self._cfg)

    @property
    def processing(self):
        return ProcessingConfig(self._cfg)


class LoadingConfig(Config):
    @property
    def sample_raw_data_folder(self):
        return self._cfg.get('NF_images:sample_raw_data_folder')
    
    @property
    def json_and_par_starter(self):
        return self._cfg.get('NF_images:image_loading:json_and_par_starter')
    
    @property
    def vertical_motor_name(self):
        return self._cfg.get('NF_images:image_loading:vertical_motor_name')
    
    @property
    def target_vertical_position(self):
        return self._cfg.get('NF_images:image_loading:target_vertical_position')
    
    @property
    def stem(self):
        return self._cfg.get('NF_images:image_loading:stem')

    @property
    def num_digits(self):
        return self._cfg.get('NF_images:image_loading:num_digits')

    @property
    def img_start(self):
        return self._cfg.get('NF_images:image_loading:img_start')

    @property
    def nframes(self):
        return self._cfg.get('NF_images:image_loading:nframes', 1440)

class ProcessingConfig(Config):

    @property
    def omega_kernel_size(self):
        return self._cfg.get('NF_images:processing:omega_kernel_size', 50)

    @property
    def threshold(self):
        return self._cfg.get('NF_images:processing:threshold', 0)

    @property
    def dilate_omega(self):
        return self._cfg.get('NF_images:processing:dilate_omega', 0)

    @property
    def method(self):
        key = self._cfg.get('NF_images:processing:routine_choice', 0)
        small_object_size = self._cfg.get('NF_images:processing:small_object_size', 0)
        if small_object_size is None:
            small_object_size = 0
            remove_small_objects = 0
        else:
            remove_small_objects = 1
        if key == 0:
            # Gaussian blur
            sigma = self._cfg.get('NF_images:processing:sigma', 1.0)
            gaussian_binarization_threshold = self._cfg.get('NF_images:processing:gaussian_binarization_threshold', 3)
            filter_parameters = [remove_small_objects,small_object_size,key,sigma,gaussian_binarization_threshold]
        elif key == 1:
            # Erosion/Dilation Parameters
            num_errosions = self._cfg.get('NF_images:processing:num_errosions', 3)
            num_dilations = self._cfg.get('NF_images:processing:num_dilations', 2)
            erosion_dilation_binarization_threshold = self._cfg.get('NF_images:processing:erosion_dilation_binarization_threshold', 5)
            filter_parameters = [remove_small_objects,small_object_size,key,
                                num_errosions,num_dilations,erosion_dilation_binarization_threshold]
        elif key == 2:
            # Non-Local Means Cleaning Parameters
            patch_size = self._cfg.get('NF_images:processing:patch_size', 3)
            patch_distance = self._cfg.get('NF_images:processing:patch_distance', 5)
            binarization_threshold = self._cfg.get('NF_images:processing:NLM_binarization_threshold', 10)
            filter_parameters = [remove_small_objects,small_object_size,key,
                                patch_size,patch_distance,binarization_threshold]
        return filter_parameters
