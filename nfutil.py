"""
nf utilities 
original author: DCP
contributing author: SEG
"""
# %% ============================================================================
# IMPORTS
# ===============================================================================
# General imports
import os
import logging
import h5py
import numpy as np
import numba
import yaml
import argparse
import timeit
import contextlib
import multiprocessing
import tempfile
import shutil
import socket
import matplotlib.pyplot as plt
import math
from scipy import stats
import scipy.ndimage as img

# HEXRD Imports
from hexrd import constants
from hexrd import instrument
from hexrd import material
from hexrd import rotations
from hexrd.transforms import xfcapi
from hexrd import valunits
from hexrd import xrdutil
from hexrd.sampleOrientations import sampleRFZ

# Yaml loader
def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)

# Constants
beam = constants.beam_vec
Z_l = constants.lab_z
vInv_ref = constants.identity_6x1

# %% ============================================================================
# CONTROLLER AND MULTIPROCESSING SCAFFOLDING FUNCTIONS
# ===============================================================================
class ProcessController:
    """This is a 'controller' that provides the necessary hooks to
    track the results of the process as well as to provide clues of
    the progress of the process"""

    def __init__(self, result_handler=None, progress_observer=None, ncpus=1,
                 chunk_size=-1):
        self.rh = result_handler
        self.po = progress_observer
        self.ncpus = ncpus
        self.chunk_size = chunk_size
        self.limits = {}
        self.timing = []
        self.multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

    # progress handling -------------------------------------------------------

    def start(self, name, count):
        self.po.start(name, count)
        t = timeit.default_timer()
        self.timing.append((name, count, t))

    def finish(self, name):
        t = timeit.default_timer()
        self.po.finish()
        entry = self.timing.pop()
        assert name == entry[0]
        total = t - entry[2]
        logging.info("%s took %8.3fs (%8.6fs per item).",
                     entry[0], total, total/entry[1])

    def update(self, value):
        self.po.update(value)

    # result handler ----------------------------------------------------------

    def handle_result(self, key, value):
        logging.debug("handle_result (%(key)s)", locals())
        self.rh.handle_result(key, value)

    # value limitting ---------------------------------------------------------
    def set_limit(self, key, limit_function):
        if key in self.limits:
            logging.warn("Overwritting limit funtion for '%(key)s'", locals())

        self.limits[key] = limit_function

    def limit(self, key, value):
        try:
            value = self.limits[key](value)
        except KeyError:
            pass
        except Exception:
            logging.warn("Could not apply limit to '%(key)s'", locals())

        return value

    # configuration  ----------------------------------------------------------

    def get_process_count(self):
        return self.ncpus

    def get_chunk_size(self):
        return self.chunk_size

def null_progress_observer():
    class NullProgressObserver:
        def start(self, name, count):
            pass

        def update(self, value):
            pass

        def finish(self):
            pass

    return NullProgressObserver()

def progressbar_progress_observer():

    class ProgressBarProgressObserver:
        def start(self, name, count):
            from progressbar import ProgressBar, Percentage, Bar

            self.pbar = ProgressBar(widgets=[name, Percentage(), Bar()],
                                    maxval=count)
            self.pbar.start()

        def update(self, value):
            self.pbar.update(value)

        def finish(self):
            self.pbar.finish()

    return ProgressBarProgressObserver()

def forgetful_result_handler():
    class ForgetfulResultHandler:
        def handle_result(self, key, value):
            pass  # do nothing

    return ForgetfulResultHandler()

def saving_result_handler(filename):
    """returns a result handler that saves the resulting arrays into a file
    with name filename"""
    class SavingResultHandler:
        def __init__(self, file_name):
            self.filename = file_name
            self.arrays = {}

        def handle_result(self, key, value):
            self.arrays[key] = value

        def __del__(self):
            logging.debug("Writing arrays in %(filename)s", self.__dict__)
            try:
                np.savez_compressed(open(self.filename, "wb"), **self.arrays)
            except IOError:
                logging.error("Failed to write %(filename)s", self.__dict__)

    return SavingResultHandler(filename)

def checking_result_handler(filename):
    """returns a return handler that checks the results against a
    reference file.

    The Check will consider a FAIL either a result not present in the
    reference file (saved as a numpy savez or savez_compressed) or a
    result that differs. It will consider a PARTIAL PASS if the
    reference file has a shorter result, but the existing results
    match. A FULL PASS will happen when all existing results match

    """
    class CheckingResultHandler:
        def __init__(self, reference_file):
            """Checks the result against those save in 'reference_file'"""
            logging.info("Loading reference results from '%s'", reference_file)
            self.reference_results = np.load(open(reference_file, 'rb'))

        def handle_result(self, key, value):
            if key in ['experiment', 'image_stack']:
                return  # ignore these

            try:
                reference = self.reference_results[key]
            except KeyError as e:
                logging.warning("%(key)s: %(e)s", locals())
                reference = None

            if reference is None:
                msg = "'{0}': No reference result."
                logging.warn(msg.format(key))

            try:
                if key == "confidence":
                    reference = reference.T
                    value = value.T

                check_len = min(len(reference), len(value))
                test_passed = np.allclose(value[:check_len],
                                          reference[:check_len])

                if not test_passed:
                    msg = "'{0}': FAIL"
                    logging.warn(msg.format(key))
                    lvl = logging.WARN
                elif len(value) > check_len:
                    msg = "'{0}': PARTIAL PASS"
                    lvl = logging.WARN
                else:
                    msg = "'{0}': FULL PASS"
                    lvl = logging.INFO
                logging.log(lvl, msg.format(key))
            except Exception as e:
                msg = "%(key)s: Failure trying to check the results.\n%(e)s"
                logging.error(msg, locals())

    return CheckingResultHandler(filename)

def build_controller(check=None,generate=None,ncpus=2,chunk_size=-1,limit=None):
    # builds the controller to use based on the args

    # result handle
    try:
        progress_handler = progressbar_progress_observer()
    except ImportError:
        progress_handler = null_progress_observer()

    if check is not None:
        if generate is not None:
            logging.warn(
                "generating and checking can not happen at the same time, "
                + "going with checking")

        result_handler = checking_result_handler(check)
    elif generate is not None:
        result_handler = saving_result_handler(generate)
    else:
        result_handler = forgetful_result_handler()

    # if args.ncpus > 1 and os.name == 'nt':
    #     logging.warn("Multiprocessing on Windows is disabled for now")
    #     args.ncpus = 1

    controller = ProcessController(result_handler, progress_handler,
                                   ncpus=ncpus,
                                   chunk_size=chunk_size)
    if limit is not None:
        controller.set_limit('coords', lambda x: min(x, limit))

    return controller

def worker_init(id_state, id_exp):
    """process initialization function. This function is only used when the
    child processes are spawned (instead of forked). When using the fork model
    of multiprocessing the data is just inherited in process memory."""
    import joblib

    global _mp_state
    state = joblib.load(id_state)
    experiment = joblib.load(id_exp)
    _mp_state = state + (experiment,)

def set_multiprocessing_method(multiprocessing_start_method):
    # Set multiprocessing method if not already done
    if multiprocessing.get_start_method() != multiprocessing_start_method:
        multiprocessing.set_start_method(multiprocessing_start_method)

@contextlib.contextmanager
def multiprocessing_pool(ncpus, state):
    """function that handles the initialization of multiprocessing. It handles
    properly the use of spawned vs forked multiprocessing. The multiprocessing
    can be either 'fork' or 'spawn', with 'spawn' being required in non-fork
    platforms (like Windows) and 'fork' being preferred on fork platforms due
    to its efficiency.
    """
    # state = ( chunk_size,
    #           image_stack,
    #           angles,
    #           precomp,
    #           coords,
    #           experiment )
    
    if multiprocessing.get_start_method() == 'fork':
        # Use FORK multiprocessing.

        # All read-only data can be inherited in the process. So we "pass" it
        # as a global that the child process will be able to see. At the end of
        # theprocessing the global is removed.
        global _mp_state
        _mp_state = state
        pool = multiprocessing.Pool(ncpus)
        yield pool
        del (_mp_state)
    else:
        # Use SPAWN multiprocessing.

        # As we can not inherit process data, all the required data is
        # serialized into a temporary directory using joblib. The
        # multiprocessing pool will have the "worker_init" as initialization
        # function that takes the key for the serialized data, which will be
        # used to load the parameter memory into the spawn process (also using
        # joblib). In theory, joblib uses memmap for arrays if they are not
        # compressed, so no compression is used for the bigger arrays.
        import joblib
        tmp_dir = tempfile.mkdtemp(suffix='-nf-grand-loop')
        try:
            # dumb dumping doesn't seem to work very well.. do something ad-hoc
            logging.info('Using "%s" as temporary directory.', tmp_dir)

            id_exp = joblib.dump(state[-1],
                                 os.path.join(tmp_dir,
                                              'grand-loop-experiment.gz'),
                                 compress=True)
            id_state = joblib.dump(state[:-1],
                                   os.path.join(tmp_dir, 'grand-loop-data'))
            pool = multiprocessing.Pool(ncpus, worker_init,
                                        (id_state[0], id_exp[0]))
            yield pool
        finally:
            logging.info('Deleting "%s".', tmp_dir)
            shutil.rmtree(tmp_dir)

# %% ============================================================================
# NUMBA FUNCTIONS
# ===============================================================================
# Check the image stack for signal
@numba.njit(nogil=True, cache=True)
def quant_and_clip_confidence(coords, angles, image,
                               base, inv_deltas, clip_vals, bsp, ome_edges):
    """quantize and clip the parametric coordinates in coords + angles

    coords - (..., 2) array: input 2d parametric coordinates
    angles - (...) array: additional dimension for coordinates
    base   - (3,) array: base value for quantization (for each dimension)
    inv_deltas - (3,) array: inverse of the quantum size (for each dimension)
    clip_vals - (2,) array: clip size (only applied to coords dimensions)
    bsp - (2,) array: beam stop vertical position and width in mm
    ome_edges - (...): list of omega edges

    clipping is performed on ranges [0, clip_vals[0]] for x and
    [0, clip_vals[1]] for y

    returns an array with the quantized coordinates, with coordinates
    falling outside the clip zone filtered out.

    """
    # Added binary serach function with Zack Singer's help - SEG 06/23/2023
    def find_target_index(array, target):
        global_index = 0
        while len(array) > 1:
            index = len(array) // 2
            if array[index] == target:
                return global_index + index
            elif array[index] > target:
                array = array[:index]
            else:
                global_index = global_index + index
                array = array[index:]
        return global_index

    # Main function
    count = len(coords)

    in_sensor = 0
    matches = 0
    for i in range(count):
        xf = coords[i, 0]
        yf = coords[i, 1]

        # does not count intensity which is covered by the beamstop dcp 5.13.21
        if abs(yf-bsp[0]) < (bsp[1]/2.):
            continue

        xf = np.floor((xf - base[0]) * inv_deltas[0])
        if not xf >= 0.0:
            continue
        if not xf < clip_vals[0]:
            continue

        yf = np.floor((yf - base[1]) * inv_deltas[1])

        if not yf >= 0.0:
            continue
        if not yf < clip_vals[1]:
            continue
        
        # CHANGE SEG 6/22/2023 and 10/03/2023 - Put in a binary serach of the omega edges
        ome_pos = angles[i]
        # While bisect left is nominally faster - it does not work with numba
        # Bisect left returns the index, j, such that all ome_edges[0:j] < ome_pos
        # zf = bisect.bisect_left(ome_edges, ome_pos) - 1
        # This method is faster when combined with numba
        zf = find_target_index(ome_edges, ome_pos)
        in_sensor += 1

        x, y, z = int(xf), int(yf), int(zf)

        if image[z, y, x]:
            matches += 1

    return 0 if in_sensor == 0 else float(matches)/float(in_sensor)

# %% ============================================================================
# PROCESSOR FUNCTIONS
# ===============================================================================
def test_single_orientation_at_single_coordinate(experiment,image_stack,coord_to_test,orientation_data_to_test,refine_yes_no=0):
    """
        Goal: 

        Input:
            
        Output:

    """
    if refine_yes_no == 0:
        # No refinement needed - just test the orientation 
        # Grab some experiment data
        tD = experiment.tVec_d # Detector X,Y,Z translation (mm)
        rD = experiment.rMat_d # Detector rotation matrix (rad)
        tS = experiment.tVec_s # Sample X,Y,Z translation (mm)
        base = experiment.base # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
        inv_deltas = experiment.inv_deltas # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
        clip_vals = experiment.clip_vals # Number of pixels along X,Y [mm,mm]
        bsp = experiment.bsp # Beam stop parameters [vertical center,width] [mm,mm]
        ome_edges = experiment.ome_edges # Omega start stop positions for each frame in image stack
        # Unpack the precomputed orientation data
        exp_map, angles, rMat_ss, gvec_cs, rMat_c = orientation_data_to_test
        # Find where those g-vectors intercept the detector from our coordinate point
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, np.squeeze(rMat_c), tD, tS, coord_to_test)
        # Check xy detector positions and omega value to see if intensity exisits
        confidence = quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                        base, inv_deltas, clip_vals, bsp, ome_edges)
        # Return the orienation and its confidence
    elif refine_yes_no == 1:
        # Refinement needed
        # Grab some experiment data
        plane_data = experiment.plane_data # Packaged information about the material and HKLs
        detector_params = experiment.detector_params # Detector tilts, position, as well as stage position and chi [?,mm,mm,chi]
        pixel_size = experiment.pixel_size # Pixel size (mm)
        ome_range = experiment.ome_range # Start and stop omega position of image stack (rad)
        ome_period = experiment.ome_period # Defined omega period for HEXRD to work in (rad)
        tD = experiment.tVec_d # Detector X,Y,Z translation (mm)
        rD = experiment.rMat_d # Detector rotation matrix (rad)
        tS = experiment.tVec_s # Sample X,Y,Z translation (mm)
        base = experiment.base # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
        inv_deltas = experiment.inv_deltas # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
        clip_vals = experiment.clip_vals # Number of pixels along X,Y [mm,mm]
        bsp = experiment.bsp # Beam stop parameters [vertical center,width] [mm,mm]
        ome_edges = experiment.ome_edges # Omega start stop positions for each frame in image stack
        panel_dims_expanded = [(-10, -10), (10, 10)] # Pixels near the edge of the detector to avoid
        ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.]) # Assume grain is unstrained 

        # Define misorientation grid
        mis_amt = experiment.misorientation_bound_rad # This is the amount of misorientation allowed on one side of the original orientation
        spacing = experiment.misorientation_step_rad # This is the spacing between orientations
        ori_pts = np.arange(-mis_amt, (mis_amt+(spacing*0.999)),spacing) # Create a linup of the orientations to go on either side
        XsO, YsO, ZsO = np.meshgrid(ori_pts, ori_pts, ori_pts) # Make that 3D
        grid0 = np.vstack([XsO.flatten(), YsO.flatten(), ZsO.flatten()]).T # Re-arange

        # Add misorientation to the trial exp_map
        exp_map_to_test, angles, rMat_ss, gvec_cs, rMat_c = orientation_data_to_test
        all_exp_maps = grid0 + np.r_[exp_map_to_test] # Define all sub orientations around the single orientation

        # Initialize an array to hold the confidence values
        n_oris = ori_pts.shape[0]**3
        all_confidence = np.zeros(n_oris)

        # Check each orientation for its confidence at the coordinate point
        for i in np.arange(n_oris):
            # Grab orientation information
            exp_map = all_exp_maps[i,:]
            # Transform exp_map to rotation matrix
            rMat_c = xfcapi.makeRotMatOfExpMap(exp_map.T)
            # Define all parameters for the orientation (strain and orientation)
            gparams = np.hstack([exp_map, ref_gparams])
            # Simulate the the diffraction events
            sim_results = xrdutil.simulateGVecs(plane_data,detector_params,gparams,panel_dims=panel_dims_expanded,
                                                pixel_pitch=pixel_size,ome_range=ome_range,ome_period=ome_period,
                                                distortion=None)
            # Pull just the angles for each g-vector
            all_angles = sim_results[2]
            # Calculate the sample rotation matrix
            rMat_ss = xfcapi.make_sample_rmat(experiment.chi, all_angles[:, 2])
            # Convert the angles to g-vectors
            gvec_cs = xfcapi.anglesToGVec(all_angles, rMat_c)
            # Find where those g-vectors intercept the detector from our coordinate point
            det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rMat_c, tD, tS, coord_to_test)
            # Check xy detector positions and omega value to see if intensity exisits
            all_confidence[i] = quant_and_clip_confidence(det_xy, all_angles[:, 2], image_stack,
                                            base, inv_deltas, clip_vals, bsp, ome_edges)

        # Find the index of the max confidence
        idx = np.where(all_confidence == np.max(all_confidence))[0][0] # Grab just the first instance if there is a tie

        # What is the hightest confidence orientation and what is its confidence
        exp_map = all_exp_maps[idx,:]
        confidence = all_confidence[idx]
    else:
        print('ERROR: refine_yes_no must be 0 or 1')
    
    # Ensure output is the correct size
    if len(np.shape(exp_map)) == 1: exp_map = np.expand_dims(exp_map,0)
    if len(np.shape(confidence)) == 0: confidence = np.expand_dims(confidence,0)
    return exp_map, confidence

def test_single_orientation_at_many_coordinates(experiment,image_stack,coords_to_test,orientation_data_to_test,refine_yes_no=0):
    """
        Goal: 
            Test a single orientation at a large number of coordinate points to check the 
                confidence the orientation exists at each coordinate point.  
        Input:
            
        Output:
            - Numpy array of size [# of coordinate points] containing a confidence value
                for each point.  
    """
    
    # Grab some experiment data
    tD = experiment.tVec_d # Detector X,Y,Z translation (mm)
    rD = experiment.rMat_d # Detector rotation matrix (rad)
    tS = experiment.tVec_s # Sample X,Y,Z translation (mm)
    base = experiment.base # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
    inv_deltas = experiment.inv_deltas # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
    clip_vals = experiment.clip_vals # Number of pixels along X,Y [mm,mm]
    bsp = experiment.bsp # Beam stop parameters [vertical center,width] [mm,mm]
    ome_edges = experiment.ome_edges # Omega start stop positions for each frame in image stack

    # Grab orientation information
    exp_map, angles, rMat_ss, gvec_cs, rMat_c = orientation_data_to_test

    # How many coordinate points do we have to test?
    n_coords = np.shape(coords_to_test)[0]

    # Initialize the confidence array to be returned
    all_exp_maps = np.zeros([n_coords,3])
    all_confidence = np.zeros(n_coords)

    # Check each orientation at the coordinate point
    for i in np.arange(n_coords):
        # What is our coordinate?
        coord_to_test = coords_to_test[i,:]
        # Find intercept point of each g-vector on the detector
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, np.squeeze(rMat_c), tD, tS, coord_to_test) # Convert angles to xy detector positions
        # Check detector positions and omega values to see if intensity exisits
        all_confidence[i] = quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                        base, inv_deltas, clip_vals, bsp, ome_edges)
        if refine_yes_no == 0:
            all_exp_maps[i] = exp_map
        elif refine_yes_no == 1:
            all_exp_maps[i], all_confidence[i] = test_single_orientation_at_single_coordinate(experiment,image_stack,coord_to_test,exp_map,refine_yes_no)
        else:
            print('ERROR: refine_yes_no must be 0 or 1')
        
        
    # Return the confidence value at each coordinate point
    return all_exp_maps, all_confidence

def test_many_orientations_at_single_coordinate(experiment,image_stack,coord_to_test,orientation_data_to_test,refine_yes_no=0):
    """
        Goal: 
            Test many orientations against a single coordinate to determine which orientation is the best fit.  If desired
                this function can call the refinement funciton such that it produces the best orientation.  
        Input:
            
        Output:
    """
    
    # Grab some experiment data
    tD = experiment.tVec_d # Detector X,Y,Z translation (mm)
    rD = experiment.rMat_d # Detector rotation matrix (rad)
    tS = experiment.tVec_s # Sample X,Y,Z translation (mm)
    base = experiment.base # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
    inv_deltas = experiment.inv_deltas # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
    clip_vals = experiment.clip_vals # Number of pixels along X,Y [mm,mm]
    bsp = experiment.bsp # Beam stop parameters [vertical center,width] [mm,mm]
    ome_edges = experiment.ome_edges # Omega start stop positions for each frame in image stack

    # Unpack the orientation information
    all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c = orientation_data_to_test

    # How many orientations do we have?
    n_oris = np.shape(all_exp_maps)[0]

    # Initialize the confidence array to be returned
    all_confidence = np.zeros(n_oris)

    # Check each orientation at the coordinate point
    for i in np.arange(n_oris):
        # Grab orientation information
        angles = all_angles[i]
        rMat_ss = all_rMat_ss[i]
        gvec_cs = all_gvec_cs[i]
        rMat_c = all_rMat_c[i]
        # Find intercept point of each g-vector on the detector
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rMat_c, tD, tS, coord_to_test) # Convert angles to xy detector positions
        # Check detector positions and omega values to see if intensity exisits
        all_confidence[i] = quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                        base, inv_deltas, clip_vals, bsp, ome_edges)

    # Find the index of the max confidence
    idx = np.where(all_confidence == np.max(all_confidence))[0][0] # Grab just the first instance if there is a tie

    # What is the hightest confidence orientation and what is its confidence
    exp_map = all_exp_maps[idx]
    confidence = all_confidence[idx]

    # Refine that orientation if we want
    if refine_yes_no == 1:
        # Refine the orientation
        exp_map, confidence = test_single_orientation_at_single_coordinate(experiment,image_stack,coord_to_test,exp_map,refine_yes_no)
    
    return exp_map, confidence, idx

def test_many_orientations_at_many_coordinates(experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no=0,start=0,stop=0):
    """
        Goal: 
            This is a multiprocessing splitter
        Input:
            
        Output:
    """
    # Check which mode we are in.  
    # If there are more coords than oris - we have chunked coords - run test_many_orientations_at_single_coordinate
    # If there are more oris than coords - we have chunked oris - run test_single_orientation_at_many_coordinates
    # How many orientations?
    n_oris = np.shape(orientation_data_to_test[0])[0]
    # How many coordinate points
    n_coords = np.shape(coordinates_to_test)[0]
    if n_coords >= n_oris:
        # We have chunked the coordinates
        coordinates_to_test = coordinates_to_test[start:stop]
        n_coords = np.shape(coordinates_to_test)[0]
        # Initalize arrays
        all_exp_maps = np.zeros([n_coords,3])
        all_confidence = np.zeros(n_coords)
        all_idx = np.zeros(n_coords)
        # Loop over the coordinates
        for i in np.arange(n_coords):
            coord_to_test = coordinates_to_test[i,:]
            if n_oris == 1:
                exp_map, confidence = test_single_orientation_at_single_coordinate(experiment,image_stack,coord_to_test,orientation_data_to_test,refine_yes_no=refine_yes_no)
                idx = 0
            else:
                exp_map, confidence, idx = test_many_orientations_at_single_coordinate(experiment,image_stack,coord_to_test,orientation_data_to_test,refine_yes_no=refine_yes_no)
            all_exp_maps[i,:] = exp_map
            all_confidence[i] = confidence
            all_idx[i] = idx
    else:
        # We have chunked the orientations
        # Unpack the orientation information
        test_exp_maps, test_angles, test_rMat_ss, test_gvec_cs, test_rMat_c = orientation_data_to_test
        test_exp_maps = test_exp_maps[start:stop]
        test_angles = test_angles[start:stop]
        test_rMat_ss = test_rMat_ss[start:stop]
        test_gvec_cs = test_gvec_cs[start:stop]
        test_rMat_c = test_rMat_c[start:stop]
        # How many orientations?
        n_oris = np.shape(test_exp_maps)[0]
        # Initalize arrays
        all_exp_maps = np.zeros([n_coords,3])
        all_confidence = np.zeros(n_coords)
        all_idx = np.zeros(n_coords)
        # Loop over the coordinates
        for i in np.arange(n_oris):
            orientation_data_to_test = [test_exp_maps[i], test_angles[i], test_rMat_ss[i], test_gvec_cs[i], test_rMat_c[i]]
            if n_coords == 1:
                exp_map, confidence = test_single_orientation_at_single_coordinate(experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no=refine_yes_no)
            else:
                exp_map, confidence = test_single_orientation_at_many_coordinates(experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no=refine_yes_no)
            # Replace any which are better than before
            to_replace = confidence > all_confidence
            all_exp_maps[to_replace,:] = exp_map[to_replace]
            all_confidence[to_replace] = confidence[to_replace]
            all_idx[to_replace] = i

    return all_exp_maps, all_confidence, all_idx, start, stop

def precompute_diffraction_data_of_single_orientation(experiment,exp_map):
    """
        Goal: 
            Read in one orientations and pre-compute all needed diffraction information on one CPU.
        Input:
            experiment: Packaged information holding experimental details
            controller: Packaged information holding multiprocessing details
            exp_maps_to_compute: exponential maps of each orientation to process
                Must of shape of (3) or (n_oris,3)
        Output:
            all_exp_maps: exponential maps of each orientation 
                Will be of shape (n_oris,3)
            all_angles: list of eta, theta, omega data for each diffraction event of each orientation
                Will be of length = n_oris
            all_rMat_ss: list of sample rotation matrices for each diffraction event of each orientation
                Will be of length = n_oris
            all_gvec_cs: list of g-vectors for each diffraction event for each grain
                Will be of length = n_oris
            all_rMat_c: rotation matrices of each orientation 
                Will be of shape (n_oris,3,3)
    """
    # Handle incoming size of exp_map
    exp_map = np.squeeze(exp_map)
    # Grab some experiment data
    plane_data = experiment.plane_data # Packaged information about the material and HKLs
    detector_params = experiment.detector_params # Detector tilts, position, as well as stage position and chi [?,mm,mm,chi]
    pixel_size = experiment.pixel_size # Pixel size (mm)
    ome_range = experiment.ome_range # Start and stop omega position of image stack (rad)
    ome_period = experiment.ome_period # Defined omega period for HEXRD to work in (rad)
    panel_dims_expanded = [(-10, -10), (10, 10)] # Pixels near the edge of the detector to avoid
    ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.]) # Assume grain is unstrained

    # Transform exp_map to rotation matrix
    rMat_c = xfcapi.makeRotMatOfExpMap(exp_map)
    # Define all parameters for the orientation (strain and orientation)
    gparams = np.hstack([exp_map, ref_gparams])
    # Simulate the the diffraction events
    sim_results = xrdutil.simulateGVecs(plane_data,detector_params,gparams,panel_dims=panel_dims_expanded,
                                        pixel_pitch=pixel_size,ome_range=ome_range,ome_period=ome_period,
                                        distortion=None)
    # Pull just the angles for each g-vector
    angles = sim_results[2]
    # Calculate the sample rotation matrix
    rMat_ss = xfcapi.make_sample_rmat(experiment.chi, angles[:, 2])
    # Convert the angles to g-vectors
    gvec_cs = xfcapi.anglesToGVec(angles, rMat_c=rMat_c)
    # Handle arrays not being the correct size if we have only one orientation
    if len(np.shape(exp_map)) == 1: exp_map = np.expand_dims(exp_map,0)
    if len(np.shape(rMat_c)) == 2: rMat_c = np.expand_dims(rMat_c,0)
    # Return precomputed data
    return exp_map, angles, rMat_ss, gvec_cs, rMat_c

def precompute_diffraction_data_of_many_orientations(experiment,exp_maps,start=0,stop=0):
    """
        Goal: 
            Read in many orientations and pre-compute all needed diffraction information on one CPU.
        Input:
            experiment: Packaged information holding experimental details
            controller: Packaged information holding multiprocessing details
            exp_maps_to_compute: exponential maps of each orientation to process
                Must of shape of (n_oris,3)
        Output:
            all_exp_maps: exponential maps of each orientation 
                Will be of shape (n_oris,3)
            all_angles: list of eta, theta, omega data for each diffraction event of each orientation
                Will be of length = n_oris
            all_rMat_ss: list of sample rotation matrices for each diffraction event of each orientation
                Will be of length = n_oris
            all_gvec_cs: list of g-vectors for each diffraction event for each grain
                Will be of length = n_oris
            all_rMat_c: rotation matrices of each orientation 
                Will be of shape (n_oris,3,3)
    """

    # Grab some experiment data
    plane_data = experiment.plane_data # Packaged information about the material and HKLs
    detector_params = experiment.detector_params # Detector tilts, position, as well as stage position and chi [?,mm,mm,chi]
    pixel_size = experiment.pixel_size # Pixel size (mm)
    ome_range = experiment.ome_range # Start and stop omega position of image stack (rad)
    ome_period = experiment.ome_period # Defined omega period for HEXRD to work in (rad)
    panel_dims_expanded = [(-10, -10), (10, 10)] # Pixels near the edge of the detector to avoid
    ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.]) # Assume grain is unstrained

    # How many orientations do we have?
    if start == 0 and stop == 0:
        # We are not in multiprocessing mode
        n_oris = np.shape(exp_maps)[0]
    else: 
        # We are in multiprocessing mode and need to pull only some of the exp_maps
        exp_maps = exp_maps[start:stop,:]
        n_oris = np.shape(exp_maps)[0]

    # Initialize lists - these must be lists since some data is of a different size for each orientations
    all_exp_maps = [None] * n_oris
    all_angles = [None] * n_oris
    all_rMat_ss = [None] * n_oris
    all_gvec_cs = [None] * n_oris
    all_rMat_c = [None] * n_oris
    # Loop through the orientations and precompute information
    for i in np.arange(n_oris):
        # What orientation are we looking at?
        exp_map = exp_maps[i]
        # Transform exp_map to rotation matrix
        rMat_c = xfcapi.makeRotMatOfExpMap(exp_map.T)
        # Define all parameters for the orientation (strain and orientation)
        gparams = np.hstack([exp_map, ref_gparams])
        # Simulate the the diffraction events
        sim_results = xrdutil.simulateGVecs(plane_data,detector_params,gparams,panel_dims=panel_dims_expanded,
                                            pixel_pitch=pixel_size,ome_range=ome_range,ome_period=ome_period,
                                            distortion=None)
        # Pull just the angles for each g-vector
        angles = sim_results[2]
        # Calculate the sample rotation matrix
        rMat_ss = xfcapi.make_sample_rmat(experiment.chi, angles[:, 2])
        # Convert the angles to g-vectors
        gvec_cs = xfcapi.anglesToGVec(angles, rMat_c=rMat_c)
        # Drop data into arrays
        all_exp_maps[i] = exp_map
        all_angles[i] = angles
        all_rMat_ss[i] = rMat_ss
        all_gvec_cs[i] = gvec_cs
        all_rMat_c[i] = rMat_c

    # Return precomputed data
    return all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c, start, stop

# %% ============================================================================
# MUTI-PROCESSOR HANDLER FUNCTIONS
# ===============================================================================
def precompute_diffraction_data(experiment,controller,exp_maps_to_precompute):
    """
        Goal: 
            Read in at least one orientation and pre-compute all needed diffraction information
                on as many CPUs as desired.
        Input:
            experiment: Packaged information holding experimental details
            controller: Packaged information holding multiprocessing details
            exp_maps_to_compute: exponential maps of each orientation to process
                Must of shape of either (3) or (n_oris,3)
        Output:
            all_exp_maps: exponential maps of each orientation 
                Will be of shape (n_oris,3)
            all_angles: list of eta, theta, omega data for each diffraction event of each orientation
                Will be of length = n_oris
            all_rMat_ss: list of sample rotation matrices for each diffraction event of each orientation
                Will be of length = n_oris
            all_gvec_cs: list of g-vectors for each diffraction event for each grain
                Will be of length = n_oris
            all_rMat_c: rotation matrices of each orientation 
                Will be of shape (n_oris,3,3)
    """
    # How many orientations?
    if len(np.shape(exp_maps_to_precompute)) == 1: exp_maps_to_precompute = np.expand_dims(exp_maps_to_precompute,0)
    n_oris = np.shape(exp_maps_to_precompute)[0]
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Are we dealing with one orientation or many?
    if n_oris == 1:
        # Single orientation, precompute the information
        # Tell the user what we are doing
        print('Precomputing diffraction data for 1 orientation on 1 CPU.')
        all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c = \
            precompute_diffraction_data_of_single_orientation(experiment,exp_maps_to_precompute)
    else:
        # Many orientations, how many cores are we dealing with?
        if ncpus == 1:
            # Many orientations, single CPU
            # Tell the user what we are doing
            print(f'Precomputing diffraction data for {n_oris} orientations on 1 CPU.')
            all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c, start, stop = \
                precompute_diffraction_data_of_many_orientations(experiment,exp_maps_to_precompute)
        elif ncpus > 1:
            # Many orientations, many CPUs
            # Define the chunk size
            chunk_size = controller.get_chunk_size()
            if chunk_size == -1:
                chunk_size = int(np.ceil(n_oris/ncpus))
            # Tell the user what we are doing
            print(f'Precomputing diffraction data for {n_oris} orientations on {ncpus} CPUs.')
            # Create chunking
            num_chunks = int(np.ceil(n_oris/chunk_size))
            chunks = np.arange(num_chunks)
            starts = np.zeros(num_chunks,dtype=int)
            stops = np.zeros(num_chunks,dtype=int)
            for i in np.arange(num_chunks):
                starts[i] = i*chunk_size
                stops[i] = i*chunk_size + chunk_size
                if stops[i] >= n_oris:
                    stops[i] = n_oris
            # Tell the user about the chunking
            print(f'There are {num_chunks} chunks with {chunk_size} orientations in each chunk.')
            # Initialize arrays to drop the precomputed data
            all_exp_maps = [None] * n_oris
            all_angles = [None] * n_oris
            all_rMat_ss = [None] * n_oris
            all_gvec_cs = [None] * n_oris
            all_rMat_c = [None] * n_oris
            # Package all inputs to the distributor function
            state = (starts,stops,experiment,exp_maps_to_precompute)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            with multiprocessing_pool(ncpus,state) as pool:
                for vals1, vals2, vals3, vals4, vals5, start, stop in pool.imap_unordered(precompute_diffraction_data_distributor,chunks):
                    # Grab the data as each CPU drops it
                    all_exp_maps[start:stop] = vals1
                    all_angles[start:stop] = vals2
                    all_rMat_ss[start:stop] = vals3
                    all_gvec_cs[start:stop] = vals4
                    all_rMat_c[start:stop] = vals5
                    # Clean up
                    del vals1, vals2, vals3, vals4, vals5, start, stop
            # Final cleanup
            pool.close()
            pool.join()
        else:
            print('Number of CPUs must be 1 or greater.')

    # Handle arrays not being the correct size if we have only one orientation
    if len(np.shape(all_exp_maps)) == 1: all_exp_maps = np.expand_dims(all_exp_maps,0)
    if len(np.shape(all_rMat_c)) == 2: all_rMat_c = np.expand_dims(all_rMat_c,0)
    # Say we are done and return precomputed data
    print('Done precomputing orientation data.')
    return all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c

def test_orientations_at_coordinates(experiment,controller,image_stack,orientation_data_to_test,coordinates_to_test,refine_yes_no=0):
    """
        Goal: 
            
        Input:
            
        Output:

    """
    # How many orientations?
    n_oris = np.shape(orientation_data_to_test[0])[0]
    # How many coordinate points
    if len(np.shape(coordinates_to_test)) == 1: coordinates_to_test = np.expand_dims(coordinates_to_test,0)
    n_coords = np.shape(coordinates_to_test)[0]
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Start a timer
    t0 = timeit.default_timer()
    # What senario do we have?
    if ncpus == 1 or (n_oris == 1 and n_coords == 1):
        # Single processor variant
        # Tell the user what we are doing
        print(f'Testing {n_oris} orientations at {n_coords} coordinate points on 1 CPU.')
        # Treat entire coordinate array as a single chunk and run
        all_exp_maps, all_confidence, all_idx, start, stop = test_many_orientations_at_many_coordinates(experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no=refine_yes_no,start=0,stop=n_coords)
    else:
        if n_coords > n_oris:
            # Fastest to loop over orientations for single processor or chunk up coordinates for multiprocessor
            # Tell the user what we are doing
            print(f'Testing {n_oris} orientations at {n_coords} coordinate points on {ncpus} CPUs.')
            # Define the chunk size
            chunk_size = controller.get_chunk_size()
            if chunk_size == -1:
                chunk_size = int(np.ceil(n_coords/ncpus))
            # Create chunking
            num_chunks = int(np.ceil(n_coords/chunk_size))
            chunks = np.arange(num_chunks)
            starts = np.zeros(num_chunks,dtype=int)
            stops = np.zeros(num_chunks,dtype=int)
            for i in np.arange(num_chunks):
                starts[i] = i*chunk_size
                stops[i] = i*chunk_size + chunk_size
                if stops[i] >= n_coords:
                    stops[i] = n_coords
            # Tell the user about the chunking
            print(f'There are {num_chunks} chunks with {chunk_size} coordinate points for each of the {n_oris} orientation.')
            # Initialize arrays to drop the exp_map and confidence
            all_exp_maps = np.zeros([n_coords,3])
            all_confidence = np.zeros(n_coords)
            all_idx = np.zeros(n_coords)
            # Unpack the orientation data
            # Package all inputs to the distributor function
            state = (starts,stops,experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            with multiprocessing_pool(ncpus,state) as pool:
                for vals1, vals2, vals3, start, stop in pool.imap_unordered(test_many_orientations_at_many_coordinates_distributor,chunks):
                    # Grab the data as each CPU drops it
                    all_exp_maps[start:stop] = vals1
                    all_confidence[start:stop] = vals2
                    all_idx[start:stop] = vals3
                    # Clean up
                    del vals1, vals2, vals3, start, stop
            # Final cleanup
            pool.close()
            pool.join()
        else:
            # Fastest to loop over coordinates for single processor or chunk up orientations for multiprocessor
            # Tell the user what we are doing
            print(f'Testing {n_oris} orientations at {n_coords} coordinate points on {ncpus} CPUs.')
            # Define the chunk size
            chunk_size = controller.get_chunk_size()
            if chunk_size == -1:
                chunk_size = int(np.ceil(n_oris/ncpus))
            # Create chunking
            num_chunks = int(np.ceil(n_oris/chunk_size))
            chunks = np.arange(num_chunks)
            starts = np.zeros(num_chunks,dtype=int)
            stops = np.zeros(num_chunks,dtype=int)
            for i in np.arange(num_chunks):
                starts[i] = i*chunk_size
                stops[i] = i*chunk_size + chunk_size
                if stops[i] >= n_oris:
                    stops[i] = n_oris
            # Tell the user about the chunking
            print(f'Each CPU will be given {num_chunks} chunks with {chunk_size} orientations for each of the {n_coords} coordinate points.')
            # Initialize arrays to drop the exp_map and confidence
            all_exp_maps = np.zeros([n_coords,3])
            all_confidence = np.zeros(n_coords)
            all_idx = np.zeros(n_coords)
            # Package all inputs to the distributor function
            state = (starts,stops,experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            chunk_num = 0
            with multiprocessing_pool(ncpus,state) as pool:
                for vals1, vals2, vals3, start, stop in pool.imap_unordered(test_many_orientations_at_many_coordinates_distributor,chunks):
                    # Grab the data as each CPU drops it
                    # Replace any which are better than before
                    to_replace = vals2 > all_confidence
                    all_exp_maps[to_replace,:] = vals1[to_replace]
                    all_confidence[to_replace] = vals2[to_replace]
                    all_idx[to_replace] = vals3[to_replace]
                    # Clean up
                    chunk_num = chunk_num + 1
                    del vals1, vals2, vals3, start, stop
            # Final cleanup
            pool.close()
            pool.join()
    # How long did it take?
    t1 = timeit.default_timer()
    print(f'All done.  Took {np.round((t1-t0)/60)} minutes.')
    return all_exp_maps, all_confidence, all_idx.astype(int)

# %% ============================================================================
# MULTI-PROCESSOR DISTRIBUTOR FUNCTIONS
# ===============================================================================
def precompute_diffraction_data_distributor(chunk):
    # Where are we pulling data from within the lists?
    starts = _mp_state[0]
    stops = _mp_state[1]
    return precompute_diffraction_data_of_many_orientations(*_mp_state[2:], start=starts[chunk], stop=stops[chunk])

def test_many_orientations_at_many_coordinates_distributor(chunk):
    # Where are we pulling data from within the lists?
    starts = _mp_state[0]
    stops = _mp_state[1]
    return test_many_orientations_at_many_coordinates(*_mp_state[2:], start=starts[chunk], stop=stops[chunk])

# %% ============================================================================
# TEST GRID GENERATION FUNCTIONS
# ===============================================================================
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

def gen_nf_test_grid_vertical(cross_sectional_dim, v_bnds, voxel_spacing):

    # This is designed only for calibration of a single slice along the z direction
    # So the Z thickness will be one voxel
    Zs_list=np.arange(-voxel_spacing/2.,voxel_spacing/2,voxel_spacing)
    Xs_list=np.arange(-cross_sectional_dim/2.+voxel_spacing/2.,cross_sectional_dim/2.,voxel_spacing)


    if v_bnds[0]==v_bnds[1]:
        Xs,Ys,Zs=np.meshgrid(Xs_list,v_bnds[0],Zs_list)
    else:
        Xs,Ys,Zs=np.meshgrid(Xs_list,np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing),Zs_list)
        #note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))



    test_crds = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds)


    return test_crds, n_crds, Xs, Ys, Zs

def generate_test_coordinates(cross_sectional_dim, v_bnds, voxel_spacing,
                              mask_data_file=None,mask_vert_offset=0.0):
    if mask_data_file is not None:
        # Load the mask
        mask_data = np.load(mask_data_file)

        mask_full = mask_data['mask']
        Xs_mask = mask_data['Xs']
        Ys_mask = mask_data['Ys']-(mask_vert_offset)
        Zs_mask = mask_data['Zs']
        voxel_spacing = mask_data['voxel_spacing']

        # need to think about how to handle a single layer in this context
        tomo_layer_centers = np.squeeze(Ys_mask[:, 0, 0])
        above = np.where(tomo_layer_centers >= v_bnds[0])
        below = np.where(tomo_layer_centers < v_bnds[1])

        in_bnds = np.intersect1d(above, below)

        mask = mask_full[in_bnds]
        Xs = Xs_mask[in_bnds]
        Ys = Ys_mask[in_bnds]
        Zs = Zs_mask[in_bnds]

        test_crds_full = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
        n_crds = len(test_crds_full)
        
        to_use = np.squeeze(np.where(mask.flatten()))
    else:
        test_crds_full, n_crds, Xs, Ys, Zs = gen_nf_test_grid(
            cross_sectional_dim, v_bnds, voxel_spacing)
        to_use = np.arange(len(test_crds_full))
        mask = np.ones(Xs.shape)

    test_coordinates = test_crds_full[to_use, :]
    return Xs, Ys, Zs, mask, test_coordinates

# %% ============================================================================
# DATA COLLECTOR FUNCTIONS
# ===============================================================================
# Generate the experiment
def generate_experiment(grain_out_file,det_file,mat_file, mat_name, max_tth, comp_thresh, chi2_thresh,omega_edges_deg, 
                       beam_stop_parms, misorientation_bnd=0.0, misorientation_spacing=0.25,v_bnds = [-0.05, 0.05],
                       cross_sectional_dim):
    # Load the grains.out data
    ff_data=np.loadtxt(grain_out_file)
    # Tell the user what we are doing so they know
    print(f'Grain data loaded from: {grain_out_file}')

    # Unpack grain data
    completeness = ff_data[:,1] # Completness
    chi2 = ff_data[:,2] # Chi^2
    exp_maps = ff_data[:,3:6] # Orientations
    t_vec_ds = ff_data[:,6:9] # Grain centroid positions
    # How many grains do we have total?
    n_grains_pre_cut = exp_maps.shape[0]

    # Trim grain information so that we pull only the grains that we want
    cut = np.where(np.logical_and(completeness>comp_thresh,chi2<chi2_thresh))[0]
    exp_maps = exp_maps[cut,:] # Orientations
    t_vec_ds = t_vec_ds[cut,:] # Grain centroid positions

    # How many grains do we have after the cull?
    n_grains = exp_maps.shape[0]
    # Tell the user what we are doing so they know
    print(f'{n_grains} grains out of a total {n_grains_pre_cut} found to satisfy completness and chi^2 thresholds.')

    # How many frames do we have?
    nframes = np.shape(omega_edges_deg)[0] - 1
    # Define variables in degrees
    # Omega range is the experimental span of omega space
    ome_range_deg = [(omega_edges_deg[0],omega_edges_deg[nframes])]  # Degrees
    # Omega period is the range in which your omega space lies (often 0 to 360 or -180 to 180)
    ome_period_deg = (ome_range_deg[0][0], ome_range_deg[0][0]+360.) # Degrees
    # Define variables in radians
    ome_period = (ome_period_deg[0]*np.pi/180.,ome_period_deg[1]*np.pi/180.)
    ome_range = [(ome_range_deg[0][0]*np.pi/180.,ome_range_deg[0][1]*np.pi/180.)]
    # Define omega edges in radians - First value is the ome start position of frame one, last value is the ome end position of final frame
    ome_edges = omega_edges_deg*np.pi/180

    # Load the detector data
    instr = load_instrument(det_file)
    print(f'Detector data loaded from: {det_file}')
    panel = next(iter(instr.detectors.values()))
    # Sample transformation parameters
    chi = instr.chi
    tVec_s = instr.tvec
    # Detector transformation parameters
    rMat_d = panel.rmat
    tilt_angles_xyzp = np.asarray(rotations.angles_from_rmat_xyz(rMat_d))
    tVec_d = panel.tvec
    # Pixel information
    row_ps = panel.pixel_size_row
    col_ps = panel.pixel_size_col
    pixel_size = (row_ps, col_ps)
    nrows = panel.rows
    ncols = panel.cols
    # Detector panel dimension information
    panel_dims = [tuple(panel.corner_ll),
                  tuple(panel.corner_ur)]
    x_col_edges = panel.col_edge_vec
    y_row_edges = panel.row_edge_vec
    # What is the max tth possible on the detector?
    max_pixel_tth = instrument.max_tth(instr)
    # Package detector parameters
    detector_params = np.hstack([tilt_angles_xyzp, tVec_d, chi, tVec_s])
    distortion = panel.distortion  # TODO: This is currently not used.

    # Parametrization for faster computation
    base = np.array([x_col_edges[0],
                     y_row_edges[0],
                     ome_edges[0]])
    deltas = np.array([x_col_edges[1] - x_col_edges[0],
                       y_row_edges[1] - y_row_edges[0],
                       ome_edges[1] - ome_edges[0]])
    inv_deltas = 1.0/deltas
    clip_vals = np.array([ncols, nrows])

    # General crystallography data
    beam_energy = valunits.valWUnit("beam_energy", "energy", instr.beam_energy, "keV")
    beam_wavelength = constants.keVToAngstrom(beam_energy.getVal('keV'))
    dmin = valunits.valWUnit("dmin", "length",
                             0.5*beam_wavelength/np.sin(0.5*max_pixel_tth),
                             "angstrom")

    # Load the materials file
    mats = material.load_materials_hdf5(mat_file, dmin=dmin,kev=beam_energy)
    print(f'{mat_name} material data loaded from: {mat_file}')
    pd = mats[mat_name].planeData
    # Check and set the max tth desired or use the detector value
    if max_tth is not None:
         pd.tThMax = np.amax(np.radians(max_tth))
    else:
        pd.tThMax = np.amax(max_pixel_tth)

    # Package up the experiment
    experiment = argparse.Namespace()
    # grains related information
    experiment.n_grains = n_grains
    experiment.exp_maps = exp_maps
    experiment.plane_data = pd
    experiment.detector_params = detector_params
    experiment.pixel_size = pixel_size
    experiment.ome_range = ome_range
    experiment.ome_period = ome_period
    experiment.x_col_edges = x_col_edges
    experiment.y_row_edges = y_row_edges
    experiment.ome_edges = ome_edges
    experiment.ncols = ncols
    experiment.nrows = nrows
    experiment.nframes = nframes  # used only in simulate...
    experiment.rMat_d = rMat_d
    experiment.tVec_d = tVec_d
    experiment.chi = chi  # note this is used to compute S... why is it needed?
    experiment.tVec_s = tVec_s
    experiment.distortion = distortion
    experiment.panel_dims = panel_dims  # used only in simulate...
    experiment.base = base
    experiment.inv_deltas = inv_deltas
    experiment.clip_vals = clip_vals
    experiment.bsp = beam_stop_parms
    experiment.mat = mats
    experiment.misorientation_bound_rad = misorientation_bnd*np.pi/180.
    experiment.misorientation_step_rad = misorientation_spacing*np.pi/180.
    experiment.remap = cut
    experiment.vertical_bounds = v_bnds
    experiment.cross_sectional_dimensions = cross_sectional_dim
    experiemnt.voxel_spacing = voxel_spacing

    return experiment

# Raw data processor
def process_raw_data(raw_confidence,raw_idx,volume_dims,mask=None,id_remap=None):
    # Assign the confidence to the correct voxel
    confidence_map = np.zeros(volume_dims)
    if mask is not None:
        confidence_map[mask] = raw_confidence
    else:
        confidence_map[:] = raw_confidence

    # Apply remap if there is one
    if id_remap is not None:
        mapped_idx = id_remap[raw_idx]
    else:
        mapped_idx = raw_idx

    # Assign the indexing to the correct voxel
    grain_map = np.zeros(volume_dims)
    if mask is not None:
        grain_map[mask] = mapped_idx
    else:
        grain_map[:] = mapped_idx

    return grain_map, confidence_map

# %% ============================================================================
# DATA WRITER FUNCTIONS
# ===============================================================================
# Simple data in, h5 out
def write_to_h5(file_dir,file_name,data_array,data_name):

    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!

    file_string = os.path.join(file_dir,file_name) + '.h5'
    hf = h5py.File(file_string, 'a')
    hf.require_dataset(data_name,np.shape(data_array),dtype=data_array.dtype)
    data = hf[data_name]
    data[...] = data_array
    hf.close()

# Writes an xdmf how Paraview requires
def xmdf_writer(file_dir,file_name):
    
    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!
    
    hf = h5py.File(os.path.join(file_dir,file_name)+'.h5','r')
    k = list(hf.keys())
    totalsets = len(k)
    # What are the datatypes
    datatypes = np.empty([totalsets], dtype=object)
    databytes = np.zeros([totalsets],dtype=np.int8)
    dims = np.ones([totalsets,4], dtype=int) # note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))
    for i in np.arange(totalsets):
        datatypes[i] = hf[k[i]].dtype
        databytes[i] = hf[k[i]].dtype.itemsize
        s = np.shape(hf[k[i]])
        if len(s) > 4:
            print('An array has greater than 4 dimensions - this writer cannot handle that')
            hf.close()
        dims[i,0:len(s)] = s
        
    hf.close()

    filename = os.path.join(file_dir,file_name) + '.xdmf'
    f = open(filename, 'w')

    # Header for xml file
    f.write('<?xml version="1.0"?>\n')
    f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd"[]>\n')
    f.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n')
    f.write(' <Domain>\n')
    f.write('\n')
    f.write('  <Grid Name="Cell Data" GridType="Uniform">\n')
    f.write('    <Topology TopologyType="3DCoRectMesh" Dimensions="' + str(dims[0,0]+1) + ' ' + str(dims[0,1]+1) + ' ' + str(dims[0,2]+1) + '"></Topology>\n')
    f.write('    <Geometry Type="ORIGIN_DXDYDZ">\n')
    f.write('      <!-- Origin -->\n')
    f.write('      <DataItem Format="XML" Dimensions="3">0 0 0</DataItem>\n')
    f.write('      <!-- DxDyDz (Spacing/Resolution)-->\n')
    f.write('      <DataItem Format="XML" Dimensions="3">1 1 1</DataItem>\n')
    f.write('    </Geometry>\n')
    f.write('\n')


    for i in np.arange(totalsets):
        if dims[i,3] == 1:
            f.write('      <Attribute Name="' + k[i] + '" AttributeType="Scalar" Center="Cell">\n')
            f.write('      <DataItem Format="HDF" Dimensions="' + str(dims[i,0]) + ' ' + str(dims[i,1]) + ' ' + str(dims[i,2]) + ' ' + str(dims[i,3]) + '" NumberType="' + str(datatypes[i]) + '" Precision="' + str(databytes[i]) + '" >\n')
            f.write('       ' + file_name + '.h5:/' + k[i] + '\n')
            f.write('      </DataItem>\n')
            f.write('       </Attribute>\n')
            f.write('\n')
        elif dims[i,3] == 3:
            f.write('      <Attribute Name="' + k[i] + '" AttributeType="Vector" Center="Cell">\n')
            f.write('      <DataItem Format="HDF" Dimensions="' + str(dims[i,0]) + ' ' + str(dims[i,1]) + ' ' + str(dims[i,2]) + ' ' + str(dims[i,3]) + '" NumberType="' + str(datatypes[i]) + '" Precision="' + str(databytes[i]) + '" >\n')
            f.write('       ' + file_name + '.h5:/' + k[i] + '\n')
            f.write('      </DataItem>\n')
            f.write('       </Attribute>\n')
            f.write('\n')
        else:
            print('Wrong array size of ' + str(dims[i,:]) + ' in array ' + str(i))

    # End the xmf file
    f.write('  </Grid>\n')
    f.write(' </Domain>\n')
    f.write('</Xdmf>\n')

    f.close()

# Data writer as either .npz or .h5 (wiht no xdmf)
def save_nf_data(save_dir,save_stem,grain_map,confidence_map,Xs,Ys,Zs,ori_list,tomo_mask=None,id_remap=None,save_type=['npz']):
    
    # H5 functionality added by SEG 8/3/2023
    
    print('Saving grain map data...')
    if id_remap is not None:
        if save_type[0] == 'hdf5':
            print('Writing HDF5 data...')
            save_string = save_dir+save_stem + '_grain_map_data.h5'
            hf = h5py.File(save_string, 'w')
            hf.create_dataset('grain_map', data=grain_map)
            hf.create_dataset('confidence', data=confidence_map)
            hf.create_dataset('Xs', data=Xs)
            hf.create_dataset('Ys', data=Ys)
            hf.create_dataset('Zs', data=Zs)
            hf.create_dataset('id_remap',data=id_remap)
            if tomo_mask is not None:
                hf.create_dataset('tomo_mask', data=tomo_mask)
            hf.close()

        elif save_type[0] == 'npz':
            print('Writing NPZ data...')
            if tomo_mask is not None:
                np.savez(save_dir+save_stem+'_grain_map_data.npz',grain_map=grain_map,confidence_map=confidence_map,Xs=Xs,Ys=Ys,Zs=Zs,ori_list=ori_list,id_remap=id_remap,tomo_mask=tomo_mask)
            else:
                np.savez(save_dir+save_stem+'_grain_map_data.npz',grain_map=grain_map,confidence_map=confidence_map,Xs=Xs,Ys=Ys,Zs=Zs,ori_list=ori_list,id_remap=id_remap)

    else:
        if save_type[0] == 'hdf5':
            print('Writing HDF5 data...')
            save_string = save_dir+save_stem + '_grain_map_data.h5'
            hf = h5py.File(save_string, 'w')
            hf.create_dataset('grain_map', data=grain_map)
            hf.create_dataset('confidence', data=confidence_map)
            hf.create_dataset('Xs', data=Xs)
            hf.create_dataset('Ys', data=Ys)
            hf.create_dataset('Zs', data=Zs)
            if tomo_mask is not None:
                hf.create_dataset('tomo_mask', data=tomo_mask)
            hf.close()

        elif save_type[0] == 'npz':
            print('Writing NPZ data...')
            if tomo_mask is not None:
                np.savez(save_dir+save_stem+'_grain_map_data.npz',grain_map=grain_map,confidence_map=confidence_map,Xs=Xs,Ys=Ys,Zs=Zs,ori_list=ori_list,tomo_mask=tomo_mask)
            else:
                np.savez(save_dir+save_stem+'_grain_map_data.npz',grain_map=grain_map,confidence_map=confidence_map,Xs=Xs,Ys=Ys,Zs=Zs,ori_list=ori_list)

# Saves the general NF output in a Paraview interpretable format
def save_nf_data_for_paraview(file_dir,file_stem,grain_map,confidence_map,Xs,Ys,Zs,ori_list,mat,tomo_mask=None,id_remap=None):
    
    # !!!!!!!!!!!!!!!!!!!!!
    # The below function has not been unit tested - use at your own risk
    # !!!!!!!!!!!!!!!!!!!!!
    
    print('Writing HDF5 data...')
    write_to_h5(file_dir,file_stem + '_grain_map_data',np.transpose(np.transpose(confidence_map,[1,0,2]),[2,1,0]),'confidence')
    write_to_h5(file_dir,file_stem + '_grain_map_data',np.transpose(np.transpose(grain_map,[1,0,2]),[2,1,0]),'grain_map')
    write_to_h5(file_dir,file_stem + '_grain_map_data',np.transpose(np.transpose(Xs,[1,0,2]),[2,1,0]),'Xs')
    write_to_h5(file_dir,file_stem + '_grain_map_data',np.transpose(np.transpose(Ys,[1,0,2]),[2,1,0]),'Ys')
    write_to_h5(file_dir,file_stem + '_grain_map_data',np.transpose(np.transpose(Zs,[1,0,2]),[2,1,0]),'Zs')
    if tomo_mask is not None:
        write_to_h5(file_dir,file_stem + '_grain_map_data',np.transpose(np.transpose(tomo_mask,[1,0,2]),[2,1,0]),'tomo_mask')
    rgb_image = generate_ori_map(grain_map, ori_list,mat,id_remap)# From unitcel the color is in hsl format
    write_to_h5(file_dir,file_stem + '_grain_map_data',np.transpose(np.transpose(rgb_image,[1,0,2,3]),[2,1,0,3]),'IPF_010')
    print('Writing XDMF...')
    xmdf_writer(file_dir,file_stem + '_grain_map_data')
    print('All done writing.')

# %% ============================================================================
# DATA PLOTTERS
# ===============================================================================
# Pulls IPF colors for each grain
def generate_ori_map(grain_map, exp_maps,mat,id_remap=None):
    # Init
    n_grains=len(exp_maps)
    if np.shape(grain_map)[0] == 1:
        grains_map_thin=np.squeeze(grain_map)
        rgb_image = np.zeros([grains_map_thin.shape[0], grains_map_thin.shape[1], 3], dtype='float32')
        # Colormapping
        for ii in np.arange(n_grains):
            if id_remap is not None:
                this_grain = np.where(np.squeeze(grains_map_thin) == id_remap[ii])
            else:
                this_grain = np.where(np.squeeze(grains_map_thin) == ii)
            if np.sum(this_grain) > 0:
                ori = exp_maps[ii, :]
                rmats = xfcapi.makeRotMatOfExpMap(ori)
                rgb = mat.unitcell.color_orientations(
                    rmats, ref_dir=np.array([0., 1., 0.]))
                rgb_image[this_grain[0], this_grain[1], 0] = rgb[0][0]
                rgb_image[this_grain[0], this_grain[1], 1] = rgb[0][1]
                rgb_image[this_grain[0], this_grain[1], 2] = rgb[0][2]
        # Redimension
        rgb_image = np.expand_dims(rgb_image,0)
    else:
        rgb_image = np.zeros(
        [grain_map.shape[0], grain_map.shape[1], grain_map.shape[2], 3], dtype='float32')
        # Colormapping
        for ii in np.arange(n_grains):
            if id_remap is not None:
                this_grain = np.where(np.squeeze(grain_map) == id_remap[ii])
            else:
                this_grain = np.where(np.squeeze(grain_map) == ii)
            if np.sum(this_grain) > 0:
                ori = exp_maps[ii, :]
                rmats = xfcapi.makeRotMatOfExpMap(ori)
                rgb = mat.unitcell.color_orientations(
                    rmats, ref_dir=np.array([0., 1., 0.]))
                rgb_image[this_grain[0], this_grain[1], this_grain[2], 0] = rgb[0][0]
                rgb_image[this_grain[0], this_grain[1], this_grain[2], 1] = rgb[0][1]
                rgb_image[this_grain[0], this_grain[1], this_grain[2], 2] = rgb[0][2]


    return rgb_image

# An IPF and confidence map plotter
def plot_ori_map(grain_map, confidence_map, Xs, Zs, exp_maps, 
                 layer_no,mat,id_remap=None, conf_thresh=None):
    # Init
    grains_plot=np.squeeze(grain_map[layer_no,:,:])
    conf_plot=np.squeeze(confidence_map[layer_no,:,:])
    n_grains=len(exp_maps)
    rgb_image = np.zeros(
        [grains_plot.shape[0], grains_plot.shape[1], 3], dtype='float32')
    # Color mapping
    for ii in np.arange(n_grains):
        if id_remap is not None:
            this_grain = np.where(np.squeeze(grains_plot) == id_remap[ii])
        else:
            this_grain = np.where(np.squeeze(grains_plot) == ii)
        if np.sum(this_grain) > 0:
            ori = exp_maps[ii, :]
            rmats = xfcapi.makeRotMatOfExpMap(ori)
            rgb = mat.unitcell.color_orientations(
                rmats, ref_dir=np.array([0., 1., 0.]))
            
            rgb_image[this_grain[0], this_grain[1], 0] = rgb[0][0]
            rgb_image[this_grain[0], this_grain[1], 1] = rgb[0][1]
            rgb_image[this_grain[0], this_grain[1], 2] = rgb[0][2]
    # Define axes
    num_markers = 5
    x_axis = Xs[0,:,0] # This is the vertical axis
    no_axis = np.linspace(0,np.shape(x_axis)[0],num=num_markers)
    x_axis = np.linspace(x_axis[0],x_axis[-1],num=num_markers)
    z_axis = Zs[0,0,:] # This is the horizontal axis
    z_axis = np.linspace(z_axis[0],z_axis[-1],num=num_markers)
    # Plot
    if conf_thresh is not None:
        # Apply masking
        mask = conf_plot > conf_thresh
        rgb_image[:,:,0] = np.multiply(rgb_image[:,:,0],mask)
        rgb_image[:,:,1] = np.multiply(rgb_image[:,:,1],mask)
        rgb_image[:,:,2] = np.multiply(rgb_image[:,:,2],mask)
        conf_plot = np.multiply(conf_plot,mask)
        grains_plot = np.multiply(grains_plot,mask).astype(int)
        # Start Figure
        fig, axs = plt.subplots(2,2,constrained_layout=True)
        fig.suptitle('Layer %d' % layer_no)
        # Plot IPF
        ax1 = axs[0,0].imshow(rgb_image,interpolation='none')
        axs[0,0].title.set_text('IPF')
        axs[0,0].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[0,0].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[0,0].set_xlabel('Z Position')
        axs[0,0].set_ylabel('X Position')
        # Plot Grain Map
        ax2 = axs[0,1].imshow(grains_plot,interpolation='none',cmap='hot')
        axs[0,1].title.set_text('Grain Map')
        axs[0,1].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[0,1].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[0,1].set_xlabel('Z Position')
        axs[0,1].set_ylabel('X Position')
        plt.colorbar(ax2)
        # Plot Confidence
        ax3 = axs[1,0].imshow(conf_plot,interpolation='none',cmap='bone')
        axs[1,0].title.set_text('Confidence')
        axs[1,0].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[1,0].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[1,0].set_xlabel('Z Position')
        axs[1,0].set_ylabel('X Position')
        plt.colorbar(ax3)
        # Plot Filler Plot
        ax4 = axs[1,1].imshow(np.zeros(np.shape(conf_plot)),interpolation='none')
        axs[1,1].title.set_text('Filler')
        axs[1,1].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[1,1].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[1,1].set_xlabel('Z Position')
        axs[1,1].set_ylabel('X Position')
        # Wrap up
        plt.show()
    else:
        # Start Figure
        fig, axs = plt.subplots(2,2,constrained_layout=True)
        fig.suptitle('Layer %d' % layer_no)
        # Plot IPF
        ax1 = axs[0,0].imshow(rgb_image,interpolation='none')
        axs[0,0].title.set_text('IPF')
        axs[0,0].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[0,0].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[0,0].set_xlabel('Z Position')
        axs[0,0].set_ylabel('X Position')
        # Plot Grain Map
        ax2 = axs[0,1].imshow(grains_plot,interpolation='none',cmap='hot')
        axs[0,1].title.set_text('Grain Map')
        axs[0,1].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[0,1].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[0,1].set_xlabel('Z Position')
        axs[0,1].set_ylabel('X Position')
        plt.colorbar(ax2)
        # Plot Confidence
        ax3 = axs[1,0].imshow(conf_plot,interpolation='none',cmap='bone')
        axs[1,0].title.set_text('Confidence')
        axs[1,0].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[1,0].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[1,0].set_xlabel('Z Position')
        axs[1,0].set_ylabel('X Position')
        plt.colorbar(ax3)
        # Plot Filler Plot
        ax4 = axs[1,1].imshow(np.zeros(np.shape(conf_plot)),interpolation='none')
        axs[1,1].title.set_text('Filler')
        axs[1,1].set_xticks(no_axis,labels=np.round(x_axis,2), rotation='vertical')
        axs[1,1].set_yticks(no_axis,labels=np.round(z_axis,2), rotation='horizontal')
        axs[1,1].set_xlabel('Z Position')
        axs[1,1].set_ylabel('X Position')
        # Wrap up
        plt.show()

# %% ============================================================================
# DIFFRACTION VOLUME STITCHERS
# ===============================================================================
# Stich individual diffraction volumes
def stitch_nf_diffraction_volumes(output_dir,output_stem,paths,material, 
                                  offsets, ori_tol=0.0, overlap=0, save_h5=0, 
                                  use_mask=0, average_orientation=0, save_npz=0, remove_small_grains_under=0,
                                  voxel_size=0.005):
    '''
    Author: seg246
    This function stiches multiple NF diffraction volumes:

    Inputs:
        paths: .npz file locations 
            size: length number of diffraction volumes
        offsets: separation of diffraction volumes 
            size: length number of diffraction volumes
            These are the motor positions, thus they are likely inverted such that offset[0] will be the,
            top most diffraction volume; however, the actual value will be the smallest motor position
        ori_tol: orientation tolerance to merge adjacet grains
            size: single valued, in degrees
        overlap: overlap of diffraction volumes
            size: single valued, in voxels along stacking direction
            example: if you have 10 micron overlap, and your voxel size is 5 micron, overlap=2
    Assumptions:
        - stacking direction will be the shortest of the three normal directions
        - grains that should be merged, are touching voxel to voxel (no gap)
        - merging of overlap can be done by a simple confidence check
        - grain maps have the same dimensions
    Outputs:
        - .npz and .h5 (paraview readable) files containting all merged, voxeleated 
            data: grain_map (new), confidence, X, Y, Z, grain_map (old), ramsz postion
        - the .h5 will have orientation colors (IPF) with 010 reference orientation
        - a grains.out file with the new, reduced, grain_ids and orientations (no other info)
        - grain_ids have been reorded based on grain volume
    '''

    print('Loading data.')
    # Some data lists initialization
    exp_map_list = []
    grain_map_list = []
    conf_map_list = []
    Xs_list = []
    Ys_list = []
    Zs_list = []
    nf_to_ff_id_map = []
    if use_mask == 1:
        masks = []
    
    # Load data into lists
    for i, p in enumerate(paths):
        nf_recon = np.load(p)
        grain_map_list.append(np.flip(nf_recon['grain_map'],0))
        conf_map_list.append(np.flip(nf_recon['confidence_map'],0))
        exp_map_list.append(nf_recon['ori_list'])
        Xs_list.append(np.flip(nf_recon['Xs'],0))
        Ys_list.append(np.flip(nf_recon['Ys'],0) - offsets[i])
        Zs_list.append(np.flip(nf_recon['Zs'],0))
        nf_to_ff_id_map.append(nf_recon['id_remap'])
        if use_mask == 1:
            masks.append(np.flip(nf_recon['tomo_mask'],0))

    # What are the dimensions of our arrays?
    dims = np.shape(grain_map_list[0]) # Since numpy dimensions are strange this is Y X Z

    # Initialize the merged data arrays
    num_vols = i + 1
    grain_map_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.int32)
    confidence_map_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    Xs_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    Ys_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    Zs_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    vertical_position_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.float32)
    diffraction_volume = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.int16)
    if use_mask == 1:
        mask_full = np.zeros(((dims[0]-overlap*2)*num_vols,dims[1],dims[2]),dtype=np.int8)

    print('Data Loaded.')
    print('Merging diffraction volumes.')

    # Run the merge
    if overlap != 0:
        # There is an overlap, use confidence to define which voxel to pull at each overlap
        # First assume there is no overlap
        # Where are we putting the first diffraction volume?
        start = 0
        stop = dims[0] - overlap*2
        for vol in np.arange(num_vols):
            grain_map_full[start:stop,:,:] = grain_map_list[vol][overlap:dims[0]-overlap,:,:]
            confidence_map_full[start:stop,:,:] = conf_map_list[vol][overlap:dims[0]-overlap,:,:]
            Xs_full[start:stop,:,:] = Xs_list[vol][overlap:dims[0]-overlap,:,:]
            Ys_full[start:stop,:,:] = Ys_list[vol][overlap:dims[0]-overlap,:,:]
            Zs_full[start:stop,:,:] = Zs_list[vol][overlap:dims[0]-overlap,:,:]
            vertical_position_full[start:stop,:,:] = offsets[vol]
            diffraction_volume[start:stop,:,:] = vol
            if use_mask == 1:
                mask_full[start:stop,:,:] = masks[vol][overlap:dims[0]-overlap,:,:]
            start = start + dims[0] - overlap*2
            stop = stop + dims[0] - overlap*2
        
        # Now handle the overlap regions
        # Where are they?  They are the overlap voxels on either side of the volume division,
        # where the division-overlap region has to be checked against the first overlap voxels
        # of the next region, and a similar number of voxels into that region need to be checked
        # against the final voxels of the first
        # We have number of diffraction volumes - 1 overlap chunks
        division_line = dims[0] - overlap*2 # This is actually the first index of the next diffraction volume
        for overlap_region in np.arange(num_vols-1):
            # Handle one side
            # This will be true at voxels which need replacing
            replace_voxels = np.signbit(confidence_map_full[division_line-overlap:division_line,:,:]-conf_map_list[overlap_region+1][0:overlap,:,:])
            grain_map_full[division_line-overlap:division_line,:,:][replace_voxels] = grain_map_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            confidence_map_full[division_line-overlap:division_line,:,:][replace_voxels] = conf_map_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            Xs_full[division_line-overlap:division_line,:,:][replace_voxels] = Xs_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            Ys_full[division_line-overlap:division_line,:,:][replace_voxels] = Ys_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            Zs_full[division_line-overlap:division_line,:,:][replace_voxels] = Zs_list[overlap_region+1][0:overlap,:,:][replace_voxels]
            vertical_position_full[division_line-overlap:division_line,:,:][replace_voxels] = offsets[overlap_region+1]
            diffraction_volume[division_line-overlap:division_line,:,:][replace_voxels] = diffraction_volume[overlap_region+1][0:overlap,:,:][replace_voxels]
            if use_mask == 1:
                mask_full[division_line-overlap:division_line,:,:][replace_voxels] = masks[overlap_region+1][0:overlap,:,:][replace_voxels]
            division_line = division_line + dims[0] - overlap*2

    else:
        # There is no overlap, scans will be stacked directly
        # Where are we putting the first diffraction volume?
        start = 0
        stop = dims[0] - overlap*2
        for vol in np.arange(num_vols):
            grain_map_full[start:stop,:,:] = grain_map_list[vol][overlap:dims[0]-overlap,:,:]
            confidence_map_full[start:stop,:,:] = conf_map_list[vol][overlap:dims[0]-overlap,:,:]
            Xs_full[start:stop,:,:] = Xs_list[vol][overlap:dims[0]-overlap,:,:]
            Ys_full[start:stop,:,:] = Ys_list[vol][overlap:dims[0]-overlap,:,:]
            Zs_full[start:stop,:,:] = Zs_list[vol][overlap:dims[0]-overlap,:,:]
            vertical_position_full[start:stop,:,:] = offsets[vol]
            diffraction_volume[start:stop,:,:] = [vol]
            if use_mask == 1:
                mask_full[start:stop,:,:] = masks[vol][overlap:dims[0]-overlap,:,:]
            start = start + dims[0] - overlap*2
            stop = stop + dims[0] - overlap*2

    print('Diffraction volumes merged.')
    if ori_tol > 0:
        print('Voxelating data.')

        # Voxelate the data
        dims = np.shape(grain_map_full)
        numel = np.prod(dims)
        id_array = np.arange(numel).reshape(dims).astype(int)
        ori_array = np.zeros([dims[0],dims[1],dims[2],3])
        for y in np.arange(dims[0]):
            for x in np.arange(dims[1]):
                for z in np.arange(dims[2]):
                    id = grain_map_full[y,x,z]
                    if id != -1:
                        which_scan = np.where(offsets.astype(np.float32) == vertical_position_full[y,x,z])[0][0]
                        idx = np.where(nf_to_ff_id_map[which_scan] == id)
                        ori_array[y,x,z,:] = exp_map_list[which_scan][idx]
        id_array[mask_full == 0] = -1
        print('Data voxeleated.')
        print('Merging grains.')

        # Grain merge methodology: calculate the misorientation of all voxels in volume and threshold.  
            # Detect only the blob in which the current voxel is located - call that the grain.  
            # Remove those grain's voxels from the test area and move on.  Do until we have nothing left.  
        grain_identification_complete = np.zeros(dims,dtype=bool)
        grain_map_merged = np.zeros(dims,dtype=np.int32)
        grain_map_merged[:] = -2 # Define everything to -2
        ori_map_merged = np.zeros([dims[0],dims[1],dims[2],3],dtype=np.float32)
        if use_mask == 1:
            grain_identification_complete[mask_full == 0] = True
            grain_map_merged[mask_full == 0] = -1
        grain = 0 # Starting grain id
        for y in np.arange(dims[0]):
            for x in np.arange(dims[1]):
                for z in np.arange(dims[2]):
                    if grain_identification_complete[y,x,z] == False: # Should not need to check mask...
                        ori = ori_array[y,x,z,:]
                        test_ids = id_array[~grain_identification_complete]
                        test_orientations = ori_array[~grain_identification_complete]

                        # Check misorientation
                        grain_quats = np.atleast_2d(rotations.quatOfExpMap(ori)).T
                        test_quats = np.atleast_2d(rotations.quatOfExpMap(test_orientations.T))
                        if np.shape(test_quats)[0] == 1:
                            [misorientations, a] = rotations.misorientation(grain_quats,test_quats.T)
                        else:
                            [misorientations, a] = rotations.misorientation(grain_quats,test_quats)
                        idx_to_merge = misorientations < np.radians(ori_tol)
                        ids_to_merge = test_ids[idx_to_merge]
                        voxels_to_merge = np.isin(id_array,ids_to_merge)

                        # Check for connectivity
                        labeled_array, num_features = img.label(voxels_to_merge,structure=np.ones([3,3,3]))
                        if num_features > 1:
                            print('This grain has more at least one non-connected region.')
                            print('Only the region where the current voxel of interest resides will be labeled as this grain.')
                            where = labeled_array[y,x,z]
                            voxels_to_merge = labeled_array == where
                        else:
                            voxels_to_merge = labeled_array.astype('bool')

                        # Average the orientation
                        if average_orientation == 1:
                            # This has not been implemented since we currently only work with grain averaged
                            # orientations.  There may be a slight difference between nf scans but it *should*
                            # be negligible for the model HEDM materials we work with.  
                            print('Orientation averaging not implemented.  Useing single orientation.')
                            print('Implement orientation averaging if you need it.')
                            new_ori = ori
                        else:
                            new_ori = ori
                        
                        # We should not geneate any overlap but let's double check
                        if np.sum(grain_identification_complete[voxels_to_merge]) > 0:
                            print('We are trying to put a grain where one already exists...')

                        # Reasign stuff
                        grain_map_merged[voxels_to_merge] = grain
                        ori_map_merged[voxels_to_merge] = new_ori
                        grain_identification_complete[voxels_to_merge] = True

                        # Cleanup
                        print('Grain ' + str(grain) + ' identified.')
                        grain = grain + 1
        print('Done with initial merging.')

        # A check
        if np.sum(np.isin(grain_map_merged,-2)) != 0:
            print('Looks like not every voxel was merged...')
        
        num_grains = grain # That +1 above handles grain 0
        new_ids = np.arange(num_grains)
        new_sizes = np.zeros(num_grains)
        new_oris = np.zeros([num_grains,3])
        for grain in new_ids:
            new_sizes[grain] = np.sum(grain_map_merged == grain)
            new_oris[grain,:] = ori_map_merged[grain_map_merged == grain][0]

        print('Grains merged.')

        print('Reassigning grain ids.')

        # Reorder the grains IDs such that the largest grain is grain 0
        final_ids = np.zeros(num_grains)
        final_sizes = np.zeros(num_grains)
        final_orientations = np.zeros([num_grains,3])
        idx = np.argsort(-new_sizes)
        final_grain_map = np.zeros(dims,np.int32)
        final_grain_map[:] = -2 # Define everything to -2
        if use_mask == 1:
            final_grain_map[mask_full == 0] = -1
        for grain in new_ids:
            final_grain_map[grain_map_merged == new_ids[idx[grain]]] = grain
            final_ids[grain] = grain
            final_orientations[grain,:] = new_oris[idx[grain]]
            final_sizes[grain] = new_sizes[idx[grain]]

        # A check
        if np.sum(np.isin(final_grain_map,-2)) != 0:
            print('Looks like not every voxel was merged...')

        print('Grain ids reassigned.')

        # Remove small grains if remove_small_grains_under is greater than zero
        if remove_small_grains_under > 0:
            print('Removeing grains smaller than ' + str(remove_small_grains_under) + ' voxels')
            if use_mask == 1:
                print('Note that any voxels removed will only have the grain ID changed, the confidence value will not be touched')
                grain_idx_to_keep = final_sizes>=remove_small_grains_under
                working_ids = final_ids[grain_idx_to_keep].astype(int)
                working_oris = final_orientations[grain_idx_to_keep]
                ids_to_remove = final_ids[~grain_idx_to_keep]
                working_grain_map = np.copy(final_grain_map)
                working_grain_map[working_grain_map>=ids_to_remove[0]] = -2
                print('Removing, please hold...')
                for y in np.arange(dims[0]):
                    for x in np.arange(dims[1]):
                        for z in np.arange(dims[2]):
                            if working_grain_map[y,x,z] == -2:
                                mask = np.zeros(np.shape(working_grain_map))
                                mask[y,x,z] = 1
                                mask = img.binary_dilation(mask,structure=np.ones([3,3,3]))
                                mask[mask_full == 0] = 0
                                mask[y,x,z] = 0
                                m,c = stats.mode(working_grain_map[mask], axis=None, keepdims=False)
                                working_grain_map[y,x,z] = m
                
                print('Done Removing.')

                # Quick double check
                if np.sum(working_ids-np.unique(working_grain_map)[1:]) != 0:
                    print('Something went wrong with the small grain removal.')

                # Recheck sizes
                num_grains = np.shape(working_ids) # That +1 above handles grain 0
                working_sizes = np.zeros(num_grains)
                for grain in working_ids:
                    working_sizes[grain] = np.sum(working_grain_map == grain)

                print('Reassigning grain ids a final time.')

                # Reorder the grains IDs such that the largest grain is grain 0
                final_ids = np.zeros(num_grains)
                final_sizes = np.zeros(num_grains)
                final_orientations = np.zeros([num_grains[0],3])
                idx = np.argsort(-working_sizes)
                final_grain_map = np.zeros(dims,np.int32)
                if use_mask == 1:
                    final_grain_map[mask_full == 0] = -1
                for grain in working_ids:
                    final_grain_map[working_grain_map == working_ids[idx[grain]]] = grain
                    final_ids[grain] = grain
                    final_orientations[grain,:] = working_oris[idx[grain]]
                    final_sizes[grain] = working_sizes[idx[grain]]

                print('Grain ids reassigned.')
            else:
                print('You are not using a mask.  A mask is needed to define sample boundaries and correctly judge grain size.')

            print('Done removing grains smaller than ' + str(remove_small_grains_under) + ' voxels')

        # Plot histogram of grain sizes
        fig, axs = plt.subplots(2,2,constrained_layout=True)
        fig.suptitle('Grain Size Statistics')
        # Plot number of voxels in all grains
        ax1 = axs[0,0].hist(final_sizes,25)
        axs[0,0].title.set_text('Histogram of Grain \nSizes in Voxels')
        axs[0,0].set_xlabel('Number of Voxels')
        axs[0,0].set_ylabel('Frequency')
        # Plot number of voxels of just the small grains
        ax2 = axs[0,1].hist(final_sizes[final_sizes<10],25)
        axs[0,1].title.set_text('Histogram of Grain \nSizes in Voxels (smaller grains)')
        axs[0,1].set_xlabel('Number of Voxels')
        axs[0,1].set_ylabel('Frequency')
        # Plot equivalent grain diameters for all grains
        ax3 = axs[1,0].hist(np.multiply(final_sizes,6/math.pi*(voxel_size*1000)**3)**(1/3),25)
        axs[1,0].title.set_text('Histogram of Equivalent \nGrain Diameters (smaller grains)')
        axs[1,0].set_xlabel('Equivalent Grain Diameter (microns)')
        axs[1,0].set_ylabel('Frequency')
        # Plot equivalent grain diameters for the small grains
        ax4 = axs[1,1].hist(np.multiply(final_sizes[final_sizes<10],6/math.pi*(voxel_size*1000)**3)**(1/3),25)
        axs[1,1].title.set_text('Histogram of Equivalent \nGrain Diameters (smaller grains)')
        axs[1,1].set_xlabel('Equivalent Grain Diameter (microns)')
        axs[1,1].set_ylabel('Frequency')
        # Wrap up
        plt.tight_layout()
        plt.show()

    else:
        print('TODO')
        final_grain_map = grain_map_full
        final_orientations = np.reshape(exp_map_list,(np.shape(exp_map_list)[0]*np.shape(exp_map_list)[1],np.shape(exp_map_list)[2]))
    print('Writing Data (If save_h5 == 1).')

    # Currently everything is upside down
    final_grain_map = np.flip(final_grain_map,0)
    confidence_map_full = np.flip(confidence_map_full,0)
    Xs_full = np.flip(Xs_full,0)
    Ys_full = np.flip(Ys_full,0)
    Zs_full = np.flip(Zs_full,0)
    diffraction_volume = np.flip(diffraction_volume,0)
    vertical_position_full = np.flip(vertical_position_full,0)
    if use_mask == 1:
        mask_full = np.flip(mask_full,0)

    # Save stuff
    if save_h5 == 1:
        if use_mask == 0:
            save_nf_data_for_paraview(output_dir,output_stem,final_grain_map,confidence_map_full,
                                            Xs_full,Ys_full,Zs_full,final_orientations,
                                            material,tomo_mask=None,id_remap=None)
        else:
            save_nf_data_for_paraview(output_dir,output_stem,final_grain_map,confidence_map_full,
                                Xs_full,Ys_full,Zs_full,final_orientations,
                                material,tomo_mask=mask_full,id_remap=None)
    if save_npz == 1:
        print('Writing NPZ data...')
        if use_mask != 0:
            np.savez(output_dir+output_stem+'_merged_grain_map_data.npz',grain_map=final_grain_map,confidence_map=confidence_map_full,Xs=Xs_full,Ys=Ys_full,Zs=Zs_full,ori_list=final_orientations,id_remap=np.unique(final_grain_map),tomo_mask=mask_full,diffraction_volume=diffraction_volume,vertical_position_full=vertical_position_full)
        else:
            np.savez(output_dir+output_stem+'_merged_grain_map_data.npz',grain_map=final_grain_map,confidence_map=confidence_map_full,Xs=Xs_full,Ys=Ys_full,Zs=Zs_full,ori_list=final_orientations,id_remap=np.unique(final_grain_map),diffraction_volume=diffraction_volume,vertical_position_full=vertical_position_full,tomo_mask=None)


    return final_orientations

# %% ============================================================================
# MISSING GRAINS FUNCTIONS
# ===============================================================================
# Find low confidence regions within the diffraction volume
def generate_low_confidence_test_coordinates(starting_reconstruction,confidence_threshold=0.5,how_sparse=0,errode_free_surface=1):
    # Data path must point to a npz save of either a merged volume or single volume
    # Within this file there MUST be:
        # confidence, X, Y, Z, and mask
        # If you do not have a mask, make one.  

    # Load the data
    confidence_map = starting_reconstruction['confidence_map']
    mask = starting_reconstruction['tomo_mask']
    Xs = starting_reconstruction['Xs']
    Ys = starting_reconstruction['Ys']
    Zs = starting_reconstruction['Zs']

    # Create a id array
    dims = np.shape(confidence_map)
    id_array = np.zeros(dims)
    id = 1
    for y in np.arange(dims[0]):
        for x in np.arange(dims[1]):
            for z in np.arange(dims[2]):
                id_array[y,x,z] = id
                id = id + 1

    # Remove the free surface
    if errode_free_surface == 1:
        # Errode all 2D blobs - will get free surface and interior blobs
        temp = np.copy(mask)
        for i in np.arange(np.shape(mask)[0]):
            temp[i,:,:] = img.binary_erosion(mask[i,:,:])
        mask = temp

    # Find the regions of low confidence
    low_confidence_map = confidence_map<confidence_threshold
    low_confidence_map[mask == 0] = 0

    # Create sparse grid mask
    # Make all even indices 1 and odd indices 0
    sparse_mask = np.ones(np.shape(low_confidence_map))
    if how_sparse == 0:
        # Not sparce at all
        print('Returning all voxels under the confidence threshold')
    elif how_sparse == 1:
        # Some sparcing applied
        sparse_mask[::2,:,:] = 0
        sparse_mask[:,::2,:] = 0
        sparse_mask[:,:,::2] = 0
        sparse_mask = img.binary_dilation(sparse_mask,structure=np.array([[[1,0,1],[0, 0, 0],[1,0,1]],
                                                                    [[0,0,0],[0, 1, 0],[0,0,0]],
                                                                    [[1,0,1],[0, 0, 0],[1,0,1]]]))
        print('Returning most voxels under the confidence threshold')
    elif how_sparse == 2:
        # More sparcing applied
        sparse_mask[::2,:,:] = 0
        sparse_mask[:,::2,:] = 0
        sparse_mask[:,:,::2] = 0
        print('Returning some voxels under the confidence threshold')

    # Scatter shot low confidence map
    point_map = np.logical_and(low_confidence_map,sparse_mask)

    # What are these poistions in the lab coordinate system?
    Xs_positions = Xs[point_map]
    Ys_positions = Ys[point_map]
    Zs_positions = Zs[point_map]
    ids = id_array[point_map]
    test_coordiates = np.vstack([Xs_positions, Ys_positions, Zs_positions]).T

    print(f'{np.shape(test_coordiates)[0]} test coordinates found')
    return test_coordiates, ids

# Sample orientation space
def uniform_fundamental_zone_sampling(point_group_number,average_angular_spacing_in_deg=3.0):
    # Below is the list of all of the point groups, in order by thier Schoenfiles notation
    # The list starts with 1 so a point_group_number = 5 will return you c2h

    # SYM_GL_PG = {
    # 'c1': '1a',  # only identity rotation
    # 'ci': '1h',  # only inversion operation
    # 'c2': '1c',  # 2-fold rotation about z
    # 'cs': '1j',
    # 'c2h': '2ch',
    # 'd2': '2bc',
    # 'c2v': '2bj',
    # 'd2h': '3bch',
    # 'c4': '1g',
    # 's4': '1m',
    # 'c4h': '2gh',
    # 'd4': '2cg',
    # 'c4v': '2gj',
    # 'd2d': '2cm',
    # 'd4h': '3cgh',
    # 'c3': '1n',
    # 's6': '2hn',
    # 'd3': '2en',
    # 'c3v': '2kn ',
    # 'd3d': '3fhn',
    # 'c6': '2bn',
    # 'c3h': '2in',
    # 'c6h': '3bhn',
    # 'd6': '3ben',
    # 'c6v': '3bkn',
    # 'd3h': '3ikn',
    # 'd6h': '4benh',
    # 't': '2cd',
    # 'th': '3cdh',
    # 'o': '2dg',
    # 'td': '2dm',
    # 'oh': '3dgh'
    # }

    # Sample the fundamental zone
    s = sampleRFZ(point_group_number,average_angular_spacing=average_angular_spacing_in_deg)
    print(f"Sampled {np.shape(s.orientations)[0]} orientations within fundamental region of point group {point_group_number} with {average_angular_spacing_in_deg} degree step size")
    return s.orientations

# %% ============================================================================
# CALIBRATION FUNCTIONS
# ===============================================================================
def translation_calibration(experiment,calibration_parameters,):

    # NOTE: If you change anything in your detector file you will need to re-run
    # the experiment loading cell above

    # Iterator will not touch the tilts, only the x,y,z translation

    # Detector parameters
    # xtilt = experiment.detector_params[0] # exponeitial map
    # ytilt = experiment.detector_params[1] # exponential map
    # ztilt = experiment.detector_params[2] # exponential map
    # x_cen = experiment.detector_params[3] # mm
    # y_cen = experiment.detector_params[4] # mm - NB if detector is centered on the beam this should not change from 0
    # distance = experiment.detector_params[5] # mm

    # Define test range - 
        # all units in mm
        # the range will be +- the 1st value about the number in the detector .yml file - choose positive values
        # the second value defines how many steps to use - odd values make this nice and clean - 1 will leave it untouched
    x_cen_values = [0.5,21]
    y_cen_values = [0.01,1]
    z_values = [0.5,21]

    # What is the full diffraction volume height?
    experiment.vertical_bounds = [-0.05, 0.05] # mm

    # Grab original values
    working_experiment = copy.deepcopy(experiment)

    # Define multiprocssing
    multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

    # Reset test grid if we ran it already
    test_crds_full, n_crds, Xs, Ys, Zs = gen_nf_test_grid(
        cross_sectional_dim, experiment.vertical_bounds, voxel_spacing)
    to_use = np.arange(len(test_crds_full))
    test_crds = test_crds_full[to_use, :]

    # Z ITERATION --------------------------------------------------------------------------------

    # Check if we are iterating this variable
    if z_values[1] != 1:
        z_conf_to_plot = np.zeros(z_values[1])
        i = 0
        z_space = np.linspace(working_experiment.detector_params[5]-z_values[0],
                            working_experiment.detector_params[5]+z_values[0],z_values[1])
        for z in z_space:
            # Change experiment
            print(z)
            working_experiment.detector_params[5] = z
            working_experiment.tVec_d[2] = z
            working_experiment.rMat_d = makeRotMatOfExpMap(working_experiment.detector_params[0:3])

            # Define controller
            controller = nfutil.build_controller(
                ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)
            
            # Define RAM groups
            if n_groups == 0:
                raw_confidence = nfutil.test_orientations(
                    image_stack, working_experiment, test_crds, controller, multiprocessing_start_method)
                del controller
                raw_confidence_full = np.zeros(
                    [len(working_experiment.exp_maps), len(test_crds_full)])
                for ii in np.arange(raw_confidence_full.shape[0]):
                    raw_confidence_full[ii, to_use] = raw_confidence[ii, :]
            # Remap 
            grain_map, confidence_map = nfutil.process_raw_confidence(
            raw_confidence_full, Xs.shape, id_remap=experiment.remap, min_thresh=0.0)
            
            # Pull the sum of the confidence map
            z_conf_to_plot[i] = np.sum(confidence_map)
            i = i+1
        
        # Where was the center found?  
        # Weighted sum - does not work great
        #a = z_space; b = z_conf_to_plot - np.min(z_conf_to_plot); b = b/np.sum(b); working_z = np.sum(np.multiply(a,b)) 
        # Take the max - It's simple but will not throw any fits if we do not have a nice curve like a fitter might
        working_z = z_space[np.where(z_conf_to_plot == np.max(z_conf_to_plot))]

        # Set working_experiment to hold the correct z_value
        working_experiment.detector_params[5] = working_z
        working_experiment.tVec_d[2] = working_z

        # Plot the detector distance curve
        fig1 = plt.figure()
        plt.plot(z_space,z_conf_to_plot)
        plt.plot([working_z,working_z],[np.min(z_conf_to_plot),np.max(z_conf_to_plot)])
        plt.title('Detector Distance Iteration ' + str(iter))
        plt.show(block=False)

        # Run at the updated position and plot
        # Define controller
        controller = nfutil.build_controller(
            ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)
        # Define RAM groups
        if n_groups == 0:
            raw_confidence = nfutil.test_orientations(
                image_stack, working_experiment, test_crds, controller, multiprocessing_start_method)
            del controller
            raw_confidence_full = np.zeros(
                [len(working_experiment.exp_maps), len(test_crds_full)])
            for ii in np.arange(raw_confidence_full.shape[0]):
                raw_confidence_full[ii, to_use] = raw_confidence[ii, :]
        # Remap 
        grain_map, confidence_map = nfutil.process_raw_confidence(
        raw_confidence_full, Xs.shape, id_remap=experiment.remap, min_thresh=0.0)


        # Plot the new confidence map
        fig2= plt.figure()
        plt.imshow(confidence_map[0,:,:],clim=[0,1])
        plt.title('Detector Distance Confidence ' + str(iter))
        plt.show(block=False)
    else:
        working_z = working_experiment.detector_params[5]

    # X ITERATION --------------------------------------------------------------------------------  

    # Check if we are iterating this variable
    if x_cen_values[1] != 1:
        x_conf_to_plot = np.zeros(x_cen_values[1])
        i = 0
        x_space = np.linspace(working_experiment.detector_params[3]-x_cen_values[0],
                            working_experiment.detector_params[3]+x_cen_values[0],x_cen_values[1])
        for x in x_space:
            # Change experiment
            print(x)
            working_experiment.detector_params[3] = x
            working_experiment.tVec_d[0] = x
            working_experiment.rMat_d = makeRotMatOfExpMap(working_experiment.detector_params[0:3])

            # Define controller
            controller = nfutil.build_controller(
                ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)
            
            # Define RAM groups
            if n_groups == 0:
                raw_confidence = nfutil.test_orientations(
                    image_stack, working_experiment, test_crds, controller, multiprocessing_start_method)
                del controller
                raw_confidence_full = np.zeros(
                    [len(working_experiment.exp_maps), len(test_crds_full)])
                for ii in np.arange(raw_confidence_full.shape[0]):
                    raw_confidence_full[ii, to_use] = raw_confidence[ii, :]
            # Remap 
            grain_map, confidence_map = nfutil.process_raw_confidence(
            raw_confidence_full, Xs.shape, id_remap=experiment.remap, min_thresh=0.0)

            # Pull the sum of the confidence map
            x_conf_to_plot[i] = np.sum(confidence_map)
            i = i+1
        
        # Where was the center found?  
        # Weighted sum - does not work great
        #a = x_space; b = x_conf_to_plot - np.min(x_conf_to_plot); b = b/np.sum(b); working_x = np.sum(np.multiply(a,b)) 
        # Take the max - It's simple but will not throw any fits if we do not have a nice curve like a fitter might
        working_x = x_space[np.where(x_conf_to_plot == np.max(x_conf_to_plot))]

        # Set working_experiment to hold the correct x_value
        working_experiment.detector_params[3] = working_x
        working_experiment.tVec_d[0] = working_x
        
        # Plot the x center curve
        fig3 = plt.figure()
        plt.plot(x_space,x_conf_to_plot)
        plt.plot([working_x,working_x],[np.min(x_conf_to_plot),np.max(x_conf_to_plot)])
        plt.title('X Center Iteration ' + str(iter))
        plt.show(block=False)

        # Run at the updated position and plot
        # Define controller
        controller = nfutil.build_controller(
            ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)
        # Define RAM groups
        if n_groups == 0:
            raw_confidence = nfutil.test_orientations(
                image_stack, working_experiment, test_crds, controller, multiprocessing_start_method)
            del controller
            raw_confidence_full = np.zeros(
                [len(working_experiment.exp_maps), len(test_crds_full)])
            for ii in np.arange(raw_confidence_full.shape[0]):
                raw_confidence_full[ii, to_use] = raw_confidence[ii, :]
        # Remap 
        grain_map, confidence_map = nfutil.process_raw_confidence(
        raw_confidence_full, Xs.shape, id_remap=experiment.remap, min_thresh=0.0)

        # Plot the new confidence map
        fig4= plt.figure()
        plt.imshow(confidence_map[0,:,:],clim=[0,1])
        plt.title('X Center Confidence ' + str(iter))
        plt.show(block=False)
    else:
        working_x = working_experiment.detector_params[3]

    # Y ITERATION --------------------------------------------------------------------------------
    # Check if we are iterating this variable
    if y_cen_values[1] != 1:
        y_conf_to_plot = np.zeros(y_cen_values[1])
        i = 0
        y_space = np.linspace(working_experiment.detector_params[4]-y_cen_values[0],
                            working_experiment.detector_params[4]+y_cen_values[0],y_cen_values[1])
        
        # Create a vertical test grid
        test_crds_full, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid_vertical(
            cross_sectional_dim, v_bnds_all, voxel_spacing)
        to_use = np.arange(len(test_crds_full))
        test_crds = test_crds_full[to_use, :]
        for y in y_space:
            # Change experiment
            print(y)
            working_experiment.detector_params[4] = y
            working_experiment.tVec_d[1] = y
            working_experiment.rMat_d = makeRotMatOfExpMap(working_experiment.detector_params[0:3])

            # Define controller
            controller = nfutil.build_controller(
                ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)
            
            # Define RAM groups
            if n_groups == 0:
                raw_confidence = nfutil.test_orientations(
                    image_stack, working_experiment, test_crds, controller, multiprocessing_start_method)
                del controller
                raw_confidence_full = np.zeros(
                    [len(working_experiment.exp_maps), len(test_crds_full)])
                for ii in np.arange(raw_confidence_full.shape[0]):
                    raw_confidence_full[ii, to_use] = raw_confidence[ii, :]
            # Remap 
            grain_map, confidence_map = nfutil.process_raw_confidence(
            raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map, min_thresh=0.0)
            
            # Pull the sum of the confidence map
            y_conf_to_plot[i] = np.sum(confidence_map)
            i = i+1
        
        # Where was the center found?  
        # Weighted sum - does not work great
        #a = y_space; b = y_conf_to_plot - np.min(y_conf_to_plot); b = b/np.sum(b); working_y = np.sum(np.multiply(a,b)) 
        # Take the max - It's simple but will not throw any fits if we do not have a nice curve like a fitter might
        working_y = y_space[np.where(y_conf_to_plot == np.max(y_conf_to_plot))]

        # Set working_experiment to hold the correct y_value
        working_experiment.detector_params[4] = working_y
        working_experiment.tVec_d[1] = working_y

        # Plot the x center curve
        fig5 = plt.figure()
        plt.plot(y_space,y_conf_to_plot)
        plt.plot([working_y,working_y],[np.min(y_conf_to_plot),np.max(y_conf_to_plot)])
        plt.title('Y Center Iteration ' + str(iter))
        plt.show(block=False)

        # Run at the updated position and plot
        # Define controller
        controller = nfutil.build_controller(
            ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)
        # Define RAM groups
        if n_groups == 0:
            raw_confidence = nfutil.test_orientations(
                image_stack, working_experiment, test_crds, controller, multiprocessing_start_method)
            del controller
            raw_confidence_full = np.zeros(
                [len(working_experiment.exp_maps), len(test_crds_full)])
            for ii in np.arange(raw_confidence_full.shape[0]):
                raw_confidence_full[ii, to_use] = raw_confidence[ii, :]
        # Remap 
        grain_map, confidence_map = nfutil.process_raw_confidence(
        raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map, min_thresh=0.0)

        # Plot the new confidence map
        fig6= plt.figure()
        plt.imshow(confidence_map[:,:,0],clim=[0,1])
        plt.title('Y Center Confidence ' + str(iter))
        plt.show(block=False)
    else:
        working_y = working_experiment.detector_params[4]
    # Print out our new varibles with precision of 100 nm
    print('Thew new detector distace was found at : ' + str(np.round(working_z,4)) +'\n'+
        'Thew new x center was found at : ' + str(np.round(working_x,4)) +'\n'+
        'Thew new y center was found at : ' + str(np.round(working_y,4)) +'\n'+
        'If you like these, change and save a new detector file and reload it to create a new '+
        'experiment.  You can then reduce the search bounds for the iterator.')




