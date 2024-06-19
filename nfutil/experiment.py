"""
stuff
"""
import os
import numpy as np
import multiprocessing

from numpy import typing as npt
from hexrd import instrument
from hexrd import material
from hexrd import constants
from hexrd import rotations
from hexrd import valunits
from hexrd import crystallography
import yaml

from collections import namedtuple
from numpy.typing import NDArray
from typing import Dict, Any, Tuple, List

from nf_config.nf_root import NFRootConfig


# TODO: Panel coords are not handled correctly
# TODO: Add graindata input like Don B. mentioned
# TODO: The max tth value does nothing...
# TODO: the image stack is in this class as an object - we probably don't want to pass that around...is that happening?
    # sys.getsizeof(experiment) = 56 so maybe not
# TODO: Add project through single layer
# TODO: add distortion
# TODO: use mypy on the things
# TODO: use pylint on the things

class Experiment():
    """ Class which holds all necessary precomputed information

        Attributes:
        `analysis_name`: str, the sample/analysis name
        `main_directory`: str, the directory to work in
        `output_directory`: str, where to output saved files

        `output_plot_check`: bool where True allows plotting and False supresses it 

        `ncpus`: int, number of cpus to use
        `chunk_size`: int, number of operations to be fed to each cpu

        `cross_sectional_dimensions`: float, dimensions of your real space search space in mm
        `voxel_size`: float, size of the voxel to use in the reconstruction
        `vertical_span_of_xray_beam`: numpy array of shape [2] with floats describing the top and bottom bound of the incoming x-ray beam in mm
        `vertical_span_to_reconstruct`: numpy array of shape [2] with floats describing the top and bottom bound of the region you wish to reconstruct in mm

        `grain_orientations_as_exponential_maps`: A numpy array of size [num_grains,3] of floats with each grain's orientation as an [exponential map](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)
        `grain_translation_vectors`: A numpy array of size [num_grains,3] of floats with each grain's centroid position in the lab frame in mm
        `ff_grain_remap_ids`: A numpy array of size [num_grains] of ints with each grain's cooresponding FF grain id

        `image_stack`: A numpy array of bools with shape [num_frames,detector_rows,detector_cols] where True indicates a diffracted intensity
        `omega_edges`: A numpy array of floats with shape [num_frams+1] with omega values for the start and stop position of each frame

        `omega_range`: A tuple of floats, with range in omega in which the sample was rotated during the NF scan in radians
        `omega_period`: A tuple of floats, with the period in omega space where the data was taken

        `instrument`: A HEDMInstrument with information from the detector yaml
        `detector`: A subset of the HEDMInstrument with information about the single detector panel

        `materials`: The entire contents of the materials.h5
        `planedata`: A PlaneData object containing material crystalographic information
        `material_name`: A str with the name of your material

        `beamstop_mask`: A numpy array of bools with shape [detector_rows, detector_cols] that is True where the beamstop exists

        `x_y_omega_base_values`: A numpy array of floats with shape [3] that holds the starting value of the detector x and y edges and the starting omega value
        `x_y_omega_inverse_deltas`: A numpy array of floats with shape [3] that holds 1/(the step size) along detector x, y (effectivly the pixel size), and omega space

        `mask_filepath`: A str with the path to the tomography mask
        `vertical_motor_position`: A float describing the motor value where this NF scan was taken

        # These may change so I am not describing them
        `centroid_serach_radius`: 
        `expand_radius_confidence_threshold`: 
        `misorientation_bound_rad`: 
        `misorientation_step_rad`: 
        `refine_yes_no`: 
        `reconstructed_data_path`: 
        `ori_grid_spacing`: 
        `confidence_threshold`: 
        `low_confidence_sparsing`: 
        `errode_free_surface`: 
        `coord_cutoff_scale`: 
        `iter_cutoff`: 
        `re_run_and_save`: 

    """
    def __init__(self, config: NFRootConfig):
        # Names and paths
        self.analysis_name: str = config.analysis_name
        self.main_directory: str = config.main_directory
        self.output_directory: str = config.output_directory

        # Output plot
        self.output_plot_check: bool = config.output_plot_check

        # Multiprocessing values
        self.ncpus: int = config.multiprocessing.num_cpus
        self.chunk_size: int = config.multiprocessing.chunk_size

        # Real space parameters
        self.cross_sectional_dimensions: float = config.reconstruction.cross_sectional_dimensions # TODO: Read in two values
        self.voxel_size: float = config.reconstruction.voxel_spacing
        self.vertical_span_of_xray_beam: npt.NDArray[np.float32] = config.experiment.beam_vertical_span
        self.vertical_span_to_reconstruct: npt.NDArray[np.float32] = config.reconstruction.desired_vertical_span

        # Load in the far field data
        self.grain_orientations_as_exponential_maps: npt.NDArray[np.float32] | None = None
        self.grain_translation_vectors: npt.NDArray[np.float32] | None = None
        self.grain_orientations_as_exponential_maps: npt.NDArray[np.float32] | None = None
        (   self.grain_orientations_as_exponential_maps,
            self.grain_translation_vectors,
            self.ff_grain_remap_ids
        ) = self.load_ff_data(config)

        # Load in the images and omega edges
        self.image_stack: npt.NDArray[np.bool_] | None = None
        self.omega_edges: npt.NDArray[np.float32] | None = None
        self.image_stack, self.omega_edges = self.load_images_and_omega_edges(config)

        # Omega parameters
        self.omega_range: tuple[np.float32] = (self.omega_edges[0],self.omega_edges[self.image_stack.shape[0]])  # rad
        self.omega_period: tuple[np.float32] = (self.omega_range[0], self.omega_range[0]+(2*np.pi)) # rad

        # Load the instrument and detector
        self.instrument: instrument.HEDMInstrument = self.load_instrument(config)
        self.detector: Dict[str, instrument.planar_detector.PlanarDetector] = next(iter(self.instrument.detectors.values()))

        # Load the material and plane data
        self.materials: Dict[str, material.Material] = None
        self.planedata: crystallography.PlaneData = None
        self.material_name: str = None
        self.materials, self.planedata, self.material_name = self.load_material_and_planedata(config,instrument.max_tth(self.instrument))

        # Load the beamstop
        self.beamstop_mask: npt.NDArray[np.bool_] = self.load_beamstop(config, self.detector)

        # Precompute a few things
        self.x_y_omega_base_values: npt.NDArray[np.float32] | None = None
        self.x_y_omega_inverse_deltas: npt.NDArray[np.float32] | None = None
        self.x_y_omega_base_values, self.x_y_omega_inverse_deltas = self.generate_x_y_omega_starts_and_inverse_deltas(self.detector,self.omega_edges)

        # Load the tomography
        self.mask_filepath: str | None = None
        self.vertical_motor_position: float | None = None
        self.mask_filepath, self.vertical_motor_position = self.load_tomorgraphy_mask(config)

        # Miscelanous things I would like to move elsewhere...maybe
        self.centroid_serach_radius: float | None = config.reconstruction.centroid_serach_radius
        self.expand_radius_confidence_threshold: float | None = config.reconstruction.expand_radius_confidence_threshold
        self.num_orientations_in_bag: float | None = config.reconstruction.num_orientations_in_bag

        # Currently not handled cleanly but here it is - likely will change
        misorientation_bnd = config.reconstruction.misorientation['misorientation_bnd']
        misorientation_spacing = config.reconstruction.misorientation['misorientation_spacing']
        if misorientation_bnd is None:
            self.misorientation_bound_rad = None
            self.misorientation_step_rad = None
        else:
            self.misorientation_bound_rad = misorientation_bnd*np.pi/180.
            self.misorientation_step_rad = misorientation_spacing*np.pi/180.
        self.refine_yes_no = 0
        self.reconstructed_data_path = config.reconstruction.missing_grains['reconstructed_data_path']
        self.ori_grid_spacing = config.reconstruction.missing_grains['ori_grid_spacing']
        self.confidence_threshold = config.reconstruction.missing_grains['confidence_threshold']
        self.low_confidence_sparsing = config.reconstruction.missing_grains['low_confidence_sparsing']
        self.errode_free_surface = config.reconstruction.missing_grains['errode_free_surface']
        self.coord_cutoff_scale = config.reconstruction.missing_grains['coord_cutoff_scale']
        self.iter_cutoff = config.reconstruction.missing_grains['iter_cutoff']
        self.re_run_and_save = config.reconstruction.missing_grains['re_run_and_save']

        # Generate test coordinates
        self.generate_test_coordinates()

        # Set the multiprocessing method
        self.multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'
        if multiprocessing.get_start_method() != self.multiprocessing_start_method:
            multiprocessing.set_start_method(self.multiprocessing_start_method, force=True)

    @staticmethod
    def load_ff_data(config: NFRootConfig):
        """"
        stuff
        """
        # Read in grains.out
        ff_data = np.loadtxt(config.input_files.grains_out_file)
        print(f'Grain data loaded from: {config.input_files.grains_out_file}')

        # Unpack grain data
        completeness = ff_data[:,1] # Completness
        chi2 = ff_data[:,2] # Chi^2
        grain_orientations_as_exponential_maps = ff_data[:,3:6] # Orientations
        grain_translation_vectors = ff_data[:,6:9] # Grain centroid positions

        # How many grains do we have total?
        n_grains_pre_cut = grain_orientations_as_exponential_maps.shape[0]

        # Trim grain information so that we pull only the grains that we have confidence in their reconstruction
        completness_threshold = config.experiment.comp_thresh
        chi2_threshold = config.experiment.chi2_thresh
        good_mask = np.where(np.logical_and(completeness>completness_threshold,chi2<chi2_threshold))[0]
        grain_orientations_as_exponential_maps = grain_orientations_as_exponential_maps[good_mask,:] # Orientations
        grain_translation_vectors = grain_translation_vectors[good_mask,:] # Grain centroid positions

        # Tell the user how many grains they have
        print(f'{grain_orientations_as_exponential_maps.shape[0]} grains out of a total {n_grains_pre_cut} found to satisfy completness and chi^2 thresholds.')

        return grain_orientations_as_exponential_maps, grain_translation_vectors, good_mask

    @staticmethod
    def load_images_and_omega_edges(config: NFRootConfig):
        # Grab config items # TODO: Should I swap to a classmethod with a self call?
        analysis_name = config.analysis_name
        output_directory = config.output_directory

        # Load the images
        images_filename = output_directory + os.sep + analysis_name + '_packaged_images.npy'
        if os.path.isfile(images_filename):
            # We have an image stack to load
            print(f'Images to be loaded from: {images_filename}')
            image_stack = np.load(images_filename)
            nframes = np.shape(image_stack)[0]
        # else:
            # TODO: Add old image load routine?
            # nframes = cfg.images.nframes

        # Load the omega edges
        omega_edges_filename = output_directory + os.sep + analysis_name + '_omega_edges_deg.npy'
        if os.path.isfile(omega_edges_filename):
            # Load the omega edges - first value is the starting ome position of first image's slew, last value is the end position of the final image's slew
            omega_edges = np.load(omega_edges_filename)*np.pi/180
        else:
            # Define omega edges manually
            omega_edges = np.linspace(config.experiment.omega_start, config.experiment.omega_stop, num=nframes+1)*np.pi/180

        # Shift in omega positive or negative by X number of images
        num_img_to_shift = config.experiment.shift_images_in_omega
        if num_img_to_shift > 0:
            # Moving positive omega so first image is not at zero, but further along
            # Using the mean omega step size - change if you need to
            omega_edges += num_img_to_shift*np.mean(np.gradient(omega_edges))
        elif num_img_to_shift < 0:
            # For whatever reason the multiprocessor does not like negative numbers, trim the stack
            image_stack = image_stack[np.abs(num_img_to_shift):,:,:]
            nframes = np.shape(image_stack)[0]
            omega_edges = omega_edges[:num_img_to_shift]
        else:
            # No shift needed
            ...

        return image_stack, omega_edges

    @staticmethod
    def load_instrument(config: NFRootConfig) -> instrument.HEDMInstrument:
        # TODO: This checking for a file should be done in the NFconfig
        # Load the detector data from an absolute path
        if os.path.isfile(config.input_files.detector_file):
            with open(config.input_files.detector_file, 'r') as f:
                print(f'Loading detector data from: {config.input_files.detector_file}')
                icfg = yaml.load(f, Loader=yaml.SafeLoader)
            return instrument.HEDMInstrument(instrument_config=icfg)
        elif os.path.isfile(config.main_directory + os.sep + config.input_files.detector_file):
            # Load from the main directory
            with open(config.main_directory + os.sep + config.input_files.detector_file, 'r') as f:
                print(f'Loading detector data from: {config.main_directory + os.sep + config.input_files.detector_file}')
                icfg = yaml.load(f, Loader=yaml.SafeLoader)
            return instrument.HEDMInstrument(instrument_config=icfg)
        else:
            # No idea where it is
            print('No detector file found.')

    @staticmethod
    def load_material_and_planedata(config: NFRootConfig, max_tth: float):
        # General crystallography data
        material_name = config.experiment.material_name
        beam_energy = valunits.valWUnit("beam_energy", "energy", config.experiment.beam_energy, "keV")
        beam_wavelength = constants.keVToAngstrom(beam_energy.getVal('keV'))
        dmin = valunits.valWUnit("dmin", "length",
                                0.5*beam_wavelength/np.sin(0.5*max_tth),
                                "angstrom")

        # Load the materials file
        # TODO: This checking for a file should be done in the NFconfig
        if os.path.isfile(config.input_files.materials_file):
            materials = material.load_materials_hdf5(config.input_files.materials_file, dmin=dmin,kev=beam_energy)
            print(f'{material_name} material data loaded from: {config.input_files.materials_file}')
        elif os.path.isfile(config.main_directory + os.sep + config.input_files.materials_file):
            materials = material.load_materials_hdf5(config.main_directory + os.sep + config.input_files.materials_file, dmin=dmin,kev=beam_energy)
            print(f'{material_name} material data loaded from: {config.main_directory + os.sep + config.input_files.materials_file}')
        else:
            print('No materials file found.')

        # Load material specific planedata
        planedata = materials[config.experiment.material_name].planeData

        return materials, planedata, material_name

    @staticmethod
    def load_beamstop(config: NFRootConfig, detector): #TODO: what is this object type
        if len(config.reconstruction.beam_stop) == 2:
            # Load from configuration
            beam_stop_parms = config.reconstruction.beam_stop
            # We need to make a mask out of the parameters
            beamstop_mask = np.zeros([detector.rows,detector.cols],bool)
            # What is the middle position of the beamstop
            middle_idx = int(np.floor(detector.rows/2.) + np.round(beam_stop_parms[0]/detector.pixel_size_col))
            # How thick is the beamstop
            half_width = int(beam_stop_parms[1]/detector.pixel_size_col/2)
            # Make the beamstop all the way across the image
            beamstop_mask[middle_idx - half_width:middle_idx + half_width,:] = 1
        else:
            # Load
            try:
                beamstop_mask = np.load(config.reconstruction.beam_stop)
                print(f'Loaded beam stop mask from: {config.reconstruction.beam_stop}')
            except:
                beamstop_mask = np.load(config.output_directory + os.sep + config.analysis_name + '_beamstop_mask.npy')
                print(f'Loaded beam stop mask from: {config.output_directory + os.sep + config.analysis_name + "_beamstop_mask.npy"}')

        return beamstop_mask

    @staticmethod
    def generate_x_y_omega_starts_and_inverse_deltas(detector, omega_edges: NDArray): #TODO: what is this object type
        # Parametrization for faster computation
        x_y_omega_base_values = np.array([detector.col_edge_vec[0],
                        detector.row_edge_vec[0],
                        omega_edges[0]])
        deltas = np.array([detector.col_edge_vec[1] - detector.col_edge_vec[0],
                        detector.row_edge_vec[1] - detector.row_edge_vec[0],
                        omega_edges[1] - omega_edges[0]])
        x_y_omega_inverse_deltas = 1.0/deltas

        return x_y_omega_base_values, x_y_omega_inverse_deltas

    @staticmethod
    def load_tomorgraphy_mask(config):
        #TODO: Load in the actual mask here?
        mask_filepath = config.reconstruction.tomography['mask_filepath']
        vertical_motor_position = config.reconstruction.tomography['vertical_motor_position']

        return mask_filepath, vertical_motor_position

    @staticmethod
    def generate_detector_parameters(instr,detector):
        # TODO: These are made for simulategvecs - maybe simulategvecs should be changed?
        # Sample transformation parameters
        chi = instr.chi
        sample_translation_vector = instr.tvec
        # Some detector tilt information
        # xfcapi.makeRotMatOfExpMap(tilt) = xfcapi.makeDetectorRotMat(rotations.angles_from_rmat_xyz(xfcapi.makeRotMatOfExpMap(tilt))) where tilt are directly read in from the .yaml as a exp_map 
        detector_rotation_matrix = detector.rmat # Generated by xfcapi.makeRotMatOfExpMap(tilt) where tilt are directly read in from the .yaml as a exp_map 
        detector_passive_euler_angles_lab_xyz = np.asarray(rotations.angles_from_rmat_xyz(detector_rotation_matrix)) # These are needed for xrdutil.simulateGVecs where they are converted to a rotation matrix via xfcapi.makeDetectorRotMat(detector_params[:3]) which reads in tiltAngles = [gamma_Xl, gamma_Yl, gamma_Zl] in radians
        detector_translation_vector = detector.tvec
        detector_params = np.hstack([detector_passive_euler_angles_lab_xyz, detector_translation_vector, chi, sample_translation_vector])

        return detector_params
    
    def generate_test_coordinates(self):
        if self.mask_filepath is not None:
            # Load the mask
            mask_data = np.load(self.mask_filepath)

            mask_full = mask_data['mask']
            Xs_mask = mask_data['Xs']
            Ys_mask = mask_data['Ys']+(self.vertical_motor_position)
            Zs_mask = mask_data['Zs']

            # need to think about how to handle a single layer in this context
            tomo_layer_centers = np.squeeze(Ys_mask[:, 0, 0])
            above = np.where(tomo_layer_centers >= self.vertical_span_to_reconstruct[0])
            below = np.where(tomo_layer_centers < self.vertical_span_to_reconstruct[1])

            in_bnds = np.intersect1d(above, below)

            mask = mask_full[in_bnds]
            Xs = Xs_mask[in_bnds]
            Ys = Ys_mask[in_bnds]
            Zs = Zs_mask[in_bnds]

        else:
            test_crds_full, n_crds, Xs, Ys, Zs = self.gen_nf_test_grid(self.cross_sectional_dimensions, self.vertical_span_to_reconstruct, self.voxel_size)
            mask = np.ones(Xs.shape,bool)

        # Give unique identifier to each voxel
        dims = np.shape(Xs)
        all_ids = np.zeros(dims,dtype=int)
        id = 0
        for x in np.arange(dims[0]):
            for z in np.arange(dims[1]):
                for y in np.arange(dims[2]):
                    all_ids[x,y,z] = id
                    id = id + 1

        # Generate connectivity matrix
        connectivity_list = [None] * id
        greater_array = np.ones([dims[0]+2,dims[1]+2,dims[2]+2],dtype=int)*-1
        greater_array[1:-1,1:-1,1:-1] = all_ids
        id = 0
        shape_array = np.zeros(dims,dtype=int)
        for x in np.arange(dims[0]):
            for z in np.arange(dims[1]):
                for y in np.arange(dims[2]):
                    x_idx = x+1
                    y_idx = y+1
                    z_idx = z+1
                    nearby_ids = greater_array[x_idx-1:x_idx+2,y_idx-1:y_idx+2,z_idx-1:z_idx+2].flatten()
                    nearby_ids = nearby_ids[nearby_ids != -1]
                    connectivity_list[id] = nearby_ids
                    shape_array[x,y,z] = len(nearby_ids)
                    id = id + 1

        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs
        self.all_voxel_ids = all_ids
        self.sample_mask = mask
        self.all_coordinates = np.vstack([Xs,Ys,Zs])
        self.connectivity_list = connectivity_list
        self.shape_array = shape_array

    @staticmethod
    def gen_nf_test_grid(cross_sectional_dim, v_bnds, voxel_spacing):
        Zs_list=np.arange(-cross_sectional_dim/2.+voxel_spacing/2.,cross_sectional_dim/2.,voxel_spacing)
        Xs_list=np.arange(-cross_sectional_dim/2.+voxel_spacing/2.,cross_sectional_dim/2.,voxel_spacing)

        if v_bnds[0]==v_bnds[1]:
            Xs,Ys,Zs=np.meshgrid(Xs_list,v_bnds[0],Zs_list)
        else:
            Xs,Ys,Zs=np.meshgrid(Xs_list,np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing),Zs_list)
            #note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))

        test_crds = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
        n_crds = len(test_crds)

        return test_crds, n_crds, Xs, Ys, Zs




