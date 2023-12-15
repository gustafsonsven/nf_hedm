import os
import numpy as np
import argparse
from hexrd import instrument
from hexrd import material
from hexrd import constants
from hexrd import rotations
from hexrd import valunits
from hexrd import instrument
from hexrd.crystallography import PlaneData
import yaml

from collections import namedtuple
from numpy.typing import NDArray
from typing import Dict, Any, Tuple, List

from nf_config.nf_root import NFRootConfig


class Experiment():
    """ Class which holds all necessary precomputed information

        Attributes:
            `exponential_maps`: A numpy array of size [num_grains,3] with each grain's orientation as an [exponential map](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)
            `plane_data`: A PlaneData object containing material crystalographic information
            `detector_euler_angles`: A numpy array of size [3] Passive euler angles in order xyz rotating the detector in the HEXRD lab frame
            `detector_pixel_size`: A tuple of the effective detector pixel size, [row,col] in mm where 'detector' refers to the combination of camera and magnification
            `omega_range`: Range in omega in which the sample was rotated during the NF scan in radians
            `omega_period`: Period describing the modulus of omega space in radians
            `omega_edges`: 
            `rMat_d`: 
            `tVec_d`: 
            `chi`: 
            `tVec_s`: 
            `panel_coords`: 
            `base`: 
            `inv_deltas`: 
            `clip_vals`: 
            `bsp`: 
            `mat`: 
            `material_name`: 
            `remap`: 
            `vertical_bounds`: 
            `beam_vertical_span`: 
            `cross_sectional_dimensions`: 
            `voxel_spacing`: 
            `ncpus`: 
            `chunk_size`: 
            `analysis_name`: 
            `main_directory`: 
            `output_directory`: 
            `output_plot_check`: 
            `point_group_number`: 
            `t_vec_s`: 
            `centroid_serach_radius`: 
            `expand_radius_confidence_threshold`: 
            `distortion`: 
        ...


    """
    def __init__(self,
                 config: NFRootConfig,
                 grain_orientations_as_exponential_maps: NDArray[np.float64], # [num_grains, 3]
                 plane_data: PlaneData,
                 # detector_params: NDArray[np.float64], # [10] TODO: Remove detector_params and replace with split [tilt_angles_xyzp, tVec_d, chi, tVec_s]
                 detector_passive_euler_angles_lab_xyz: NDArray[np.float64], # [3]
                 pixel_size: Tuple[float, float], # mm
                 omega_range: Tuple[float, float],
                 omega_period: Tuple[float, float],
                 omega_edges: NDArray[np.float64], # [num_frames+1]
                 rMat_d: NDArray[np.float64], # [3, 3]
                 tVec_d: NDArray[np.float64], # [3]
                 chi: float,
                 tVec_s: NDArray[np.float64], # [3]
                 panel_coords: Tuple[Tuple[float, float], Tuple[float, float]],
                 base: NDArray[np.float64], # [3]
                 inv_deltas: NDArray[np.float64], # [3]
                 clip_vals: Tuple[int, int],
                 bsp,
                 mat,
                 material_name,
                 remap,
                 vertical_bounds,
                 beam_vertical_span,
                 cross_sectional_dimensions,
                 voxel_spacing,
                 ncpus,
                 chunk_size,
                 analysis_name,
                 main_directory,
                 output_directory,
                 output_plot_check,
                 point_group_number,
                 t_vec_s,
                 centroid_serach_radius,
                 expand_radius_confidence_threshold,
                 distortion: NDArray[np.float64]=None,
    ):
        self.t_vec_s: NDArray[np.float64] = t_vec_s
        self.chi = self.load_chi(config)
        a,b,c = self.load_many_things()



        
        
        ...

    @staticmethod
    def load_instrument(yml):
        with open(yml, 'r') as f:
            icfg = yaml.load(f, Loader=yaml.FullLoader)
        return instrument.HEDMInstrument(instrument_config=icfg)

    @staticmethod
    def load_chi(config: NFRootConfig, t_vec_s: NDArray) -> float:
        # self.load_instrument
        return 0.0
    
    @staticmethod
    def load_many_things(config: NFRootConfig):
        # self.load_instrument
        return 0.0, 0.1, 0.2

# class A():
#     def __init__(self, name):
#         self.name = name

#     @staticmethod
#     def print_1():
#         print(1)
# A.print_1()


# Generate the experiment
def generate_experiment(cfg):
    PanelCoords = namedtuple('Panel_Coordinates', ('lower_left', 'upper_right'))
    analysis_name = cfg.analysis_name # The name you want all your output files to have within thier filename (relevant to the sample)
    main_directory = cfg.main_directory
    output_directory = cfg.output_directory
    output_plot_check = cfg.output_plot_check

    # reconstruction size parameters
    # TODO: make cross sectional dims changeable
    cross_sectional_dim = cfg.reconstruction.cross_sectional_dimensions # [1.0,1.0] # mm - [perpendicualr, parallel] to incoming beam direction
    voxel_spacing = cfg.reconstruction.voxel_spacing
    beam_vertical_span = cfg.experiment.beam_vertical_span
    vertical_bounds = cfg.reconstruction.desired_vertical_span
    ncpus = cfg.multiprocessing.num_cpus
    chunk_size = cfg.multiprocessing.chunk_size
    #check = cfg.multiprocessing.check
    #limit = cfg.multiprocessing.limit
    #generate = cfg.multiprocessing.generate
    #max_RAM = cfg.multiprocessing.max_RAM  # this is in bytes
    
    # Load the grains.out data
    ff_data=np.loadtxt(cfg.input_files.grains_out_file)
    # Tell the user what we are doing so they know
    print(f'Grain data loaded from: {cfg.input_files.grains_out_file}')

    # Unpack grain data
    # TODO: Add graindata input like Don B. mentioned
    completeness = ff_data[:,1] # Completness
    chi2 = ff_data[:,2] # Chi^2
    grain_orientations_as_exponential_maps = ff_data[:,3:6] # Orientations
    t_vec_s = ff_data[:,6:9] # Grain centroid positions
    # How many grains do we have total?
    n_grains_pre_cut = grain_orientations_as_exponential_maps.shape[0]

    # Trim grain information so that we pull only the grains that we want
    comp_thresh = cfg.experiment.comp_thresh
    chi2_thresh = cfg.experiment.chi2_thresh
    cut = np.where(np.logical_and(completeness>comp_thresh,chi2<chi2_thresh))[0]
    grain_orientations_as_exponential_maps = grain_orientations_as_exponential_maps[cut,:] # Orientations
    t_vec_s = t_vec_s[cut,:] # Grain centroid positions

    # Tell the user what we are doing so they know
    print(f'{grain_orientations_as_exponential_maps.shape[0]} grains out of a total {n_grains_pre_cut} found to satisfy completness and chi^2 thresholds.')

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
        omega_edges = np.linspace(cfg.experiment.omega_start, cfg.experiment.omega_stop, num=nframes+1)*np.pi/180

    # Shift in omega positive or negative by X number of images
    num_img_to_shift = cfg.experiment.shift_images_in_omega
    if num_img_to_shift > 0:
        # Moving positive omega so first image is not at zero, but further along
        # Using the mean omega step size - change if you need to
        omega_edges += num_img_to_shift*np.mean(np.gradient(omega_edges))
    elif num_img_to_shift < 0:
        # For whatever reason the multiprocessor does not like negative numbers, trim the stack
        image_stack = image_stack[np.abs(num_img_to_shift):,:,:]
        nframes = np.shape(image_stack)[0]
        omega_edges = omega_edges[:num_img_to_shift]


    # Define variables in degrees
    # Omega range is the experimental span of omega space
    omega_range = (omega_edges[0],omega_edges[nframes])  # rad
    # Omega period is the range in which your omega space lies (often 0 to 360 or -180 to 180)
    omega_period = (omega_range[0], omega_range[0]+(2*np.pi)) # rad


    # Load the detector data
    if os.path.isfile(cfg.input_files.detector_file):
        instr = load_instrument(cfg.input_files.detector_file)
        print(f'Detector data loaded from: {cfg.input_files.detector_file}')
    elif os.path.isfile(main_directory + os.sep + cfg.input_files.detector_file):
        instr = load_instrument(main_directory + os.sep + cfg.input_files.detector_file)
        print(f'Detector data loaded from: {main_directory + os.sep + cfg.input_files.detector_file}')
    else:
        print('No detector file found.')
    panel = next(iter(instr.detectors.values()))

    # Sample transformation parameters
    chi = instr.chi
    tVec_s = instr.tvec
    # Some detector tilt information
    # xfcapi.makeRotMatOfExpMap(tilt) = xfcapi.makeDetectorRotMat(rotations.angles_from_rmat_xyz(xfcapi.makeRotMatOfExpMap(tilt))) where tilt are directly read in from the .yaml as a exp_map 
    rMat_d = panel.rmat # Generated by xfcapi.makeRotMatOfExpMap(tilt) where tilt are directly read in from the .yaml as a exp_map 
    detector_passive_euler_angles_lab_xyz = np.asarray(rotations.angles_from_rmat_xyz(rMat_d)) # These are needed for xrdutil.simulateGVecs where they are converted to a rotation matrix via xfcapi.makeDetectorRotMat(detector_params[:3]) which reads in tiltAngles = [gamma_Xl, gamma_Yl, gamma_Zl] in radians
    tVec_d = panel.tvec

    # Pixel information
    row_ps = panel.pixel_size_row
    col_ps = panel.pixel_size_col
    pixel_size = (row_ps, col_ps)
    nrows = panel.rows
    ncols = panel.cols
    # Detector panel dimension information
    # panel_coords = [tuple(panel.corner_ll),
    #               tuple(panel.corner_ur)]
    panel_coords = PanelCoords(lower_left=panel.corner_ll, upper_right=panel.corner_ur)


    x_col_edges = panel.col_edge_vec
    y_row_edges = panel.row_edge_vec
    # What is the max tth possible on the detector?
    max_pixel_tth = instrument.max_tth(instr)
    # Package detector parameters
    detector_params = np.hstack([detector_passive_euler_angles_lab_xyz, tVec_d, chi, tVec_s])
    distortion = panel.distortion  # TODO: This is currently not used.

    # Parametrization for faster computation
    base = np.array([x_col_edges[0],
                     y_row_edges[0],
                     omega_edges[0]])
    deltas = np.array([x_col_edges[1] - x_col_edges[0],
                       y_row_edges[1] - y_row_edges[0],
                       omega_edges[1] - omega_edges[0]])
    inv_deltas = 1.0/deltas
    clip_vals = (ncols, nrows)

    # General crystallography data
    beam_energy = valunits.valWUnit("beam_energy", "energy", cfg.experiment.beam_energy, "keV")
    beam_wavelength = constants.keVToAngstrom(beam_energy.getVal('keV'))
    dmin = valunits.valWUnit("dmin", "length",
                             0.5*beam_wavelength/np.sin(0.5*max_pixel_tth),
                             "angstrom")

    # Load the materials file
    if os.path.isfile(cfg.input_files.materials_file):
        mats = material.load_materials_hdf5(cfg.input_files.materials_file, dmin=dmin,kev=beam_energy)
        print(f'{cfg.experiment.material_name} material data loaded from: {cfg.input_files.materials_file}')
    elif os.path.isfile(main_directory + os.sep + cfg.input_files.materials_file):
        mats = material.load_materials_hdf5(main_directory + os.sep + cfg.input_files.materials_file, dmin=dmin,kev=beam_energy)
        print(f'{cfg.experiment.material_name} material data loaded from: {main_directory + os.sep + cfg.input_files.materials_file}')
    else:
        print('No materials file found.')


    pd = mats[cfg.experiment.material_name].planeData


    # Check and set the max tth desired or use the detector value
    max_tth = cfg.experiment.max_tth
    if max_tth is not None:
         pd.tThMax = np.amax(np.radians(max_tth))
    else:
        pd.tThMax = np.amax(max_pixel_tth)

    # Pull the beamstop
    if len(cfg.reconstruction.beam_stop) == 2:
        # Load from configuration
        beam_stop_parms = cfg.reconstruction.beam_stop
        # We need to make a mask out of the parameters
        beam_stop_mask = np.zeros([nrows,ncols],bool)
        # What is the middle position of the beamstop
        middle_idx = int(np.floor(nrows/2.) + np.round(beam_stop_parms[0]/col_ps))
        # How thick is the beamstop
        half_width = int(beam_stop_parms[1]/col_ps/2)
        # Make the beamstop all the way across the image
        beam_stop_mask[middle_idx - half_width:middle_idx + half_width,:] = 1
        # Set the mask
        beam_stop_parms = beam_stop_mask
    else:
        # Load
        try:
            beam_stop_parms = np.load(cfg.reconstruction.beam_stop)
            print(f'Loaded beam stop mask from: {cfg.reconstruction.beam_stop}')
        except:
            beam_stop_parms = np.load(output_directory + os.sep + analysis_name + '_beamstop_mask.npy')
            print(f'Loaded beam stop mask from: {output_directory + os.sep + analysis_name + "_beamstop_mask.npy"}')


    # Package up the experiment
    experiment = argparse.Namespace()
    # grains related information
    experiment.grain_orientations_as_exponential_maps = grain_orientations_as_exponential_maps
    experiment.plane_data = pd
    experiment.detector_params = detector_params
    experiment.pixel_size = pixel_size
    experiment.omega_range = omega_range
    experiment.omega_period = omega_period
    experiment.x_col_edges = x_col_edges
    experiment.y_row_edges = y_row_edges
    experiment.omega_edges = omega_edges
    experiment.ncols = ncols
    experiment.nrows = nrows
    experiment.nframes = nframes  # used only in simulate...
    experiment.rMat_d = rMat_d
    experiment.tVec_d = tVec_d
    experiment.chi = chi  # note this is used to compute S... why is it needed?
    experiment.tVec_s = tVec_s
    experiment.distortion = distortion
    experiment.panel_coords = panel_coords  # used only in simulate...
    experiment.base = base
    experiment.inv_deltas = inv_deltas
    experiment.clip_vals = clip_vals
    experiment.bsp = beam_stop_parms
    experiment.mat = mats
    experiment.material_name = cfg.experiment.material_name
    experiment.remap = cut
    experiment.vertical_bounds = vertical_bounds
    experiment.beam_vertical_span = beam_vertical_span
    experiment.cross_sectional_dimensions = cross_sectional_dim
    experiment.voxel_spacing = voxel_spacing
    experiment.ncpus = ncpus
    experiment.chunk_size = chunk_size
    experiment.analysis_name = analysis_name
    experiment.main_directory = main_directory
    experiment.output_directory = output_directory
    experiment.output_plot_check = output_plot_check
    experiment.point_group_number = cfg.experiment.point_group_number
    experiment.t_vec_s = t_vec_s
    experiment.centroid_serach_radius = cfg.reconstruction.centroid_serach_radius
    experiment.expand_radius_confidence_threshold = cfg.reconstruction.expand_radius_confidence_threshold
    experiment.detector_passive_euler_angles_lab_xyz = detector_passive_euler_angles_lab_xyz

    # Tomo parameters
    if cfg.reconstruction.tomography is None:
        experiment.mask_filepath = None
        experiment.vertical_motor_position = None
    else:
        # TODO: Add project through single layer
        experiment.mask_filepath = cfg.reconstruction.tomography['mask_filepath']
        experiment.vertical_motor_position = cfg.reconstruction.tomography['vertical_motor_position']
    
    # Misorientation
    if cfg.reconstruction.misorientation:
        misorientation_bnd = cfg.reconstruction.misorientation['misorientation_bnd']
        misorientation_spacing = cfg.reconstruction.misorientation['misorientation_spacing']
        experiment.misorientation_bound_rad = misorientation_bnd*np.pi/180.
        experiment.misorientation_step_rad = misorientation_spacing*np.pi/180.
        experiment.refine_yes_no = 1
    # Tomo mask
    if cfg.reconstruction.tomography:
        experiment.mask_filepath = cfg.reconstruction.tomography['mask_filepath']
        experiment.vertical_motor_position = cfg.reconstruction.tomography['vertical_motor_position']
        experiment.use_single_layer = cfg.reconstruction.tomography['use_single_layer']
    
    if cfg.reconstruction.missing_grains:
        experiment.reconstructed_data_path = cfg.reconstruction.missing_grains['reconstructed_data_path']
        experiment.ori_grid_spacing = cfg.reconstruction.missing_grains['ori_grid_spacing']
        experiment.confidence_threshold = cfg.reconstruction.missing_grains['confidence_threshold']
        experiment.low_confidence_sparsing = cfg.reconstruction.missing_grains['low_confidence_sparsing']
        experiment.errode_free_surface = cfg.reconstruction.missing_grains['errode_free_surface']
        experiment.coord_cutoff_scale = cfg.reconstruction.missing_grains['coord_cutoff_scale']
        experiment.iter_cutoff = cfg.reconstruction.missing_grains['iter_cutoff']
        experiment.re_run_and_save = cfg.reconstruction.missing_grains['re_run_and_save']


    return experiment, image_stack

