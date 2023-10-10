"""
nf utilities 
original author: DCP
"""
# %% ============================================================================
# Imports
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
import copy
import matplotlib.pyplot as plt
import math
from scipy import stats
# import bisect # The function that calls this does not work with numba

# HEXRD Imports
from hexrd import constants
from hexrd import instrument
from hexrd import material
from hexrd import rotations
from hexrd.transforms import xfcapi
from hexrd import valunits
from hexrd import xrdutil
from hexrd.sampleOrientations import sampleRFZ

# Multiprocessing imports
hostname = socket.gethostname()
USE_MPI = False
rank = 0
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    USE_MPI = world_size > 1
    logging.info(f'{rank=} {world_size=} {hostname=}')
except ImportError:
    logging.warning(f'mpi4py failed to load on {hostname=}. MPI is disabled.')
    pass

# Image processing imports
from skimage.morphology import dilation as ski_dilation
import scipy.ndimage as img
import skimage.filters as filters
try:
    import imageio as imgio
except(ImportError):
    from skimage import io as imgio

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
# Scaffolding Functions
# ===============================================================================
class ProcessController:
    """This is a 'controller' that provides the necessary hooks to
    track the results of the process as well as to provide clues of
    the progress of the process"""

    def __init__(self, result_handler=None, progress_observer=None, ncpus=1,
                 chunk_size=100):
        self.rh = result_handler
        self.po = progress_observer
        self.ncpus = ncpus
        self.chunk_size = chunk_size
        self.limits = {}
        self.timing = []

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

# %% ============================================================================
# Numba functions
# ===============================================================================
# Check the image stack for signal
@numba.njit(nogil=True, cache=True)
def _quant_and_clip_confidence(coords, angles, image,
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
# Orientation Testing
# ===============================================================================
def test_orientations(image_stack, experiment, test_crds, controller,multiprocessing_start_method='fork'):
    """grand loop precomputing the grown image stack

    image-stack -- is the dilated image stack to be tested against.

    experiment  -- A bunch of experiment related parameters.

    test_crds  -- Coordinates to test orientations on, units mm.


    controller  -- An external object implementing the hooks to notify progress
                   as well as figuring out what to do with results.
    """
    
    # extract some information needed =========================================
    # number of grains, number of coords (maybe limited by call), projection
    # function to use, chunk size to use if multiprocessing and the number
    # of cpus.
    n_grains = experiment.n_grains
    chunk_size = controller.get_chunk_size()
    ncpus = controller.get_process_count()
    exp_maps = np.atleast_2d(experiment.exp_maps)
    n_coords = controller.limit('coords', len(test_crds))

    # Check the chunk size
    if chunk_size == -1:
        chunk_size = int(np.ceil(n_coords/ncpus))
    # precompute per-grain stuff ==============================================

    all_angles, all_rMat_ss, all_gvec_cs = \
    precompute_orientation_information_main_loop(exp_maps,experiment,controller,multiprocessing_start_method)
    print(f'Done precomputing orientation information.')

    # Divide coords by ranks
    (offset, size) = get_offset_size(n_coords)

    # grand loop ==============================================================
    # The near field simulation 'grand loop'. Where the bulk of computing is
    # performed. We are looking for a confidence matrix that has a n_grains
    chunks = range(offset, offset+size, chunk_size)

    subprocess = 'grand_loop'
    controller.start(subprocess, n_coords)
    finished = 0
    ncpus = min(ncpus, len(chunks))


    logging.info(f'For {rank=}, {offset=}, {size=}, {chunks=}, {len(chunks)=}, {ncpus=}')
    logging.info('Checking confidence for %d coords, %d grains.',
                n_coords, n_grains)
        
    confidence = np.empty((n_grains, size))
    if ncpus > 1:

        global _multiprocessing_start_method
        _multiprocessing_start_method = multiprocessing_start_method

        logging.info('Running multiprocess %d processes (%s)',
                    ncpus, _multiprocessing_start_method)
        with grand_loop_pool(ncpus=ncpus,
                             state=(chunk_size,
                                    image_stack,
                                    all_angles, all_rMat_ss, all_gvec_cs,
                                    test_crds, experiment)) as pool:
            for rslice, rvalues in pool.imap_unordered(multiproc_inner_loop,
                                                       chunks):
                count = rvalues.shape[1]
                # We need to adjust this slice for the offset
                rslice = slice(rslice.start - offset, rslice.stop - offset)
                confidence[:, rslice] = rvalues
                finished += count
                controller.update(finished)
    else:
        logging.info('Running in a single process')
        for chunk_start in chunks:
            chunk_stop = min(n_coords, chunk_start+chunk_size)
            rslice, rvalues = _grand_loop_inner(
                image_stack, all_angles,
                all_rMat_ss, all_gvec_cs, test_crds, experiment,
                start=chunk_start,
                stop=chunk_stop
            )
            count = rvalues.shape[1]
            # We need to adjust this slice for the offset
            rslice = slice(rslice.start - offset, rslice.stop - offset)
            confidence[:, rslice] = rvalues
            finished += count
            controller.update(finished)

    controller.finish(subprocess)

    # Now gather result to rank 0
    if USE_MPI:
        gather_confidence(controller, confidence, n_grains, n_coords)
    else:
        controller.handle_result("confidence", confidence)


    return confidence

def evaluate_diffraction_angles(experiment, controller=None):
    """Uses simulateGVecs to generate the angles used per each grain.
    returns a list containg one array per grain.

    experiment -- a bag of experiment values, including the grains specs
                  and other required parameters.
    """
    # extract required data from experiment
    exp_maps = experiment.exp_maps
    # Quick size check
    if len(np.shape(exp_maps)) == 1:
        exp_maps = np.expand_dims(exp_maps,axis=0)
    n_grains = experiment.n_grains
    plane_data = experiment.plane_data
    detector_params = experiment.detector_params
    pixel_size = experiment.pixel_size
    ome_range = experiment.ome_range
    ome_period = experiment.ome_period

    panel_dims_expanded = [(-10, -10), (10, 10)]
    subprocess = 'evaluate diffraction angles'
    pbar = controller.start(subprocess, n_grains)
    all_angles = []
    ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])
    for i in np.arange(n_grains):
        exp_map = exp_maps[i,:]
        gparams = np.hstack([exp_map, ref_gparams])
        sim_results = xrdutil.simulateGVecs(plane_data,
                                            detector_params,
                                            gparams,
                                            panel_dims=panel_dims_expanded,
                                            pixel_pitch=pixel_size,
                                            ome_range=ome_range,
                                            ome_period=ome_period,
                                            distortion=None)
        all_angles.append(sim_results[2])
        controller.update(i + 1)
        pass
    controller.finish(subprocess)

    return all_angles

def _grand_loop_inner(image_stack, angles, all_rMat_ss, all_gvec_cs,
                      coords, experiment, start=0, stop=None):
    """Actual simulation code for a chunk of data. It will be used both,
    in single processor and multiprocessor cases. Chunking is performed
    on the coords.

    image_stack -- the image stack from the sensors
    angles -- the angles (grains) to test
    coords -- all the coords to test
    precomp -- (gvec_cs, rmat_ss) precomputed for each grain
    experiment -- bag with experiment parameters
    start -- chunk start offset
    stop -- chunk end offset
    """

    t = timeit.default_timer()
    n_coords = len(coords)
    n_angles = len(angles)
    # experiment geometric layout parameters
    rD = experiment.rMat_d
    rCn = experiment.rMat_c
    tD = experiment.tVec_d
    tS = experiment.tVec_s

    # Quick size check
    if len(np.shape(rCn)) == 2:
        rCn = np.expand_dims(rCn,axis=0)

    # experiment panel related configuration
    base = experiment.base
    inv_deltas = experiment.inv_deltas
    clip_vals = experiment.clip_vals
    distortion = experiment.distortion
    bsp = experiment.bsp #beam stop vertical center and width
    ome_edges = experiment.ome_edges

    _to_detector = xfcapi.gvec_to_xy
    # _to_detector = _gvec_to_detector_array
    stop = min(stop, n_coords) if stop is not None else n_coords

    # FIXME: distortion hanlding is broken!
    distortion_fn = None
    if distortion is not None and len(distortion > 0):
        distortion_fn, distortion_args = distortion

    acc_detector = 0.0
    acc_distortion = 0.0
    acc_quant_clip = 0.0
    confidence = np.zeros((n_angles, stop-start))
    grains = 0
    crds = 0

    if distortion_fn is None:
        for igrn in range(n_angles):
            angs = angles[igrn]
            rC = rCn[igrn]
            #gvec_cs, rMat_ss = precomp[igrn]
            rMat_ss = all_rMat_ss[igrn]
            gvec_cs = all_gvec_cs[igrn]
            grains += 1
            for icrd in range(start, stop):
                t0 = timeit.default_timer()
                det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rC, tD, tS, coords[icrd])
                t1 = timeit.default_timer()
                c = _quant_and_clip_confidence(det_xy, angs[:, 2], image_stack,
                                               base, inv_deltas, clip_vals, bsp, ome_edges)
                t2 = timeit.default_timer()
                acc_detector += t1 - t0
                acc_quant_clip += t2 - t1
                crds += 1
                confidence[igrn, icrd - start] = c
    else:
        for igrn in range(n_angles):
            angs = angles[igrn]
            rC = rCn[igrn]
            #gvec_cs, rMat_ss = precomp[igrn]
            rMat_ss = all_rMat_ss[igrn]
            gvec_cs = all_gvec_cs[igrn]
            grains += 1
            for icrd in range(start, stop):
                t0 = timeit.default_timer()
                tmp_xys = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rC, tD, tS, coords[icrd])
                t1 = timeit.default_timer()
                det_xy = distortion_fn(tmp_xys, distortion_args, invert=True)
                t2 = timeit.default_timer()
                c = _quant_and_clip_confidence(det_xy, angs[:, 2], image_stack,
                                               base, inv_deltas, clip_vals,bsp, ome_edges)
                t3 = timeit.default_timer()
                acc_detector += t1 - t0
                acc_distortion += t2 - t1
                acc_quant_clip += t3 - t2
                crds += 1
                confidence[igrn, icrd - start] = c

    t = timeit.default_timer() - t
    return slice(start, stop), confidence

def generate_test_grid(low, top, samples):
    """generates a test grid of coordinates"""
    cvec_s = np.linspace(low, top, samples)
    Xs, Ys, Zs = np.meshgrid(cvec_s, cvec_s, cvec_s)
    return np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T

def precompute_orientation_information_main_loop(exp_maps,experiment,controller,multiprocessing_start_method):
    # Define multithreading
    global _multiprocessing_start_method
    _multiprocessing_start_method = multiprocessing_start_method
    # How many orientations?
    n_oris = np.shape(exp_maps)[0]
    # CPU count - pull from controller
    ncpus = controller.get_process_count()
    if ncpus == 1:
        # Rip it in
        all_angs, all_rMat_ss, all_gvec_cs, start, stop = precompute_orientation_information(exp_maps,experiment,0,n_oris)
    elif ncpus > 1:
        # Chunk size - make it here ------ alternative is chunk_size = controller.get_chunk_size()
        chunk_size = controller.get_chunk_size()
        if chunk_size == -1:
            chunk_size = int(np.ceil(n_oris/ncpus))
        # Start controller
        subprocess = 'simulate_loop'
        controller.start(subprocess, n_oris)
        finished = 0
        # How many chunks do we need?
        num_chunks = int(np.ceil(n_oris/chunk_size))
        chunks = np.arange(num_chunks)
        # Initialize arrays to drop the simulated information
        all_angs = [None] * n_oris
        all_rMat_ss = [None] * n_oris
        all_gvec_cs = [None] * n_oris 
        # Create chunking
        starts = np.zeros(num_chunks,dtype=int)
        stops = np.zeros(num_chunks,dtype=int)
        for i in np.arange(num_chunks):
            starts[i] = i*chunk_size
            stops[i] = i*chunk_size + chunk_size
            if stops[i] >= n_oris:
                stops[i] = n_oris
        # Tell user what we are doing
        print(f"Processing {n_oris} orientations with {num_chunks} chunks of size {chunk_size} on {ncpus} CPUs.")
        # Start the loop
        with grand_loop_pool(ncpus=ncpus,state=(starts,stops,exp_maps,experiment)) as pool:
            for vals1, vals2, vals3, start, stop in pool.imap_unordered(precompute_orientation_information_distributor,chunks):
                # Grab the data as each CPU drops it
                count = stop-start
                all_angs[start:stop] = vals1
                all_rMat_ss[start:stop] = vals2
                all_gvec_cs[start:stop] = vals3
                # Update the controller
                finished += count
                controller.update(finished)
                # Clean up
                del vals1, vals2, vals3, start, stop
        # More cleanup
        pool.close()
        pool.join()
    else:
        print('Number of CPUs must be 1 or greater.')

    # Return stuff
    return all_angs, all_rMat_ss, all_gvec_cs

def precompute_orientation_information(exp_maps,experiment,start,stop):
    # Grab some experiment data
    plane_data = experiment.plane_data
    detector_params = experiment.detector_params
    pixel_size = experiment.pixel_size
    ome_range = experiment.ome_range
    ome_period = experiment.ome_period
    chi = experiment.chi
    panel_dims_expanded = [(-10, -10), (10, 10)]
    # Grain data - assuming no strain
    ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])
    # How many grains?
    num_grains = stop-start
    # Define arrays to dump data
    all_angs = [None] * num_grains
    all_rMat_ss = [None] * num_grains
    all_gvec_cs = [None] * num_grains
    # Pull just the exp_maps that we need
    exp_maps = exp_maps[start:stop,:]
    for i, exp_map in enumerate(exp_maps):
        rMat_c = xfcapi.makeRotMatOfExpMap(exp_map.T) # Convert to rotation matrix
        gparams = np.hstack([exp_map, ref_gparams]) # Stack grain data into single variable
        # Simulate the angles for each of these orientations
        sim_results = xrdutil.simulateGVecs(plane_data,detector_params,gparams,panel_dims=panel_dims_expanded,
                                            pixel_pitch=pixel_size,ome_range=ome_range,ome_period=ome_period,
                                            distortion=None)
        angs = sim_results[2] # Grab just the angles 
        rMat_ss = xfcapi.make_sample_rmat(chi, angs[:, 2]) # Calculate the sample rotation matrix
        gvec_cs = xfcapi.anglesToGVec(angs, rMat_c=rMat_c) # Calcutate the g_vectors
        # Drop the data in
        all_angs[i] = angs
        all_rMat_ss[i] = rMat_ss
        all_gvec_cs[i] = gvec_cs
    # Return
    return all_angs, all_rMat_ss, all_gvec_cs, start, stop

def find_single_orientation(image_stack, experiment, test_crd, test_exp_maps, test_angles, test_rMat_ss, test_gvec_cs, misorientation_bnd, misorientation_spacing, confidence_threshold):
    logging.disable()
    # Method:
        # Go to each coordinate point and grab its non-refined orientation
        # Blow up a misorientation spacing around this orientation
        # Test those orientations
        # Find the best and return that value

    # Grab some experiment data
    rD = experiment.rMat_d
    tD = experiment.tVec_d
    tS = experiment.tVec_s
    base = experiment.base
    inv_deltas = experiment.inv_deltas
    clip_vals = experiment.clip_vals
    bsp = experiment.bsp
    ome_edges = experiment.ome_edges

    # Check the confidence of each orientation
    n_oris = np.shape(test_exp_maps)[0]
    confidence = np.zeros((n_oris, 1)) # Place for confidence values for each individaul sub orientation

    # Loop over each orientation
    t0 = timeit.default_timer()
    for i, exp_map in enumerate(test_exp_maps):
        all_angles = test_angles[i]
        rMat_ss = test_rMat_ss[i]
        gvec_cs = test_gvec_cs[i]
        # rMat_c = rotations.rotMatOfExpMap(exp_map.T) # Convert to rotation matrix
        rMat_c = xfcapi.makeRotMatOfExpMap(exp_map.T) # Much faster
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rMat_c, tD, tS, test_crd) # Convert angles to xy detector positions
        # Check xy detector positions and omega value to see if intensity exisits
        confidence[i] = _quant_and_clip_confidence(det_xy, all_angles[:, 2], image_stack,
                                        base, inv_deltas, clip_vals, bsp, ome_edges)
        if confidence[i] > confidence_threshold:
            # We found one!
            print(f'Found orientation {np.round(i/n_oris,2)*100}% through the orientation list.')
            break
    if i == len(exp_map)-1:
        print(f'Did not find an orientaiton above threshold, attempting refinement on best guess.')

    # Unpack confidence and determine the orientation
    confidence = np.squeeze(confidence)
    where = np.where(confidence == np.max(confidence))
    new_exp_map = test_exp_maps[where[0],:].squeeze() # Grab just one 
    if len(np.shape(new_exp_map)) == 2:
        # Not sure why I have to do this...
        new_exp_map = new_exp_map[0,:].squeeze()
        print('Strange')
        print(np.shape(test_exp_maps))
        print(np.shape(test_exp_maps[where[0],:].squeeze()))
        print(where[0])

    # Refine that orientation
    refined_exp_map, refined_conf = refine_single_orientation(image_stack, experiment, test_crd, new_exp_map, misorientation_bnd, misorientation_spacing)
    print(f'Refined confidence of final orientation is {np.round(refined_conf,2)}.')
    return refined_exp_map,refined_conf

def test_single_coordinate_main_loop(image_stack, experiment, test_crd, test_exp_maps, test_angles, 
                                        test_rMat_ss, test_gvec_cs, misorientation_bnd, misorientation_spacing
                                        ,controller,multiprocessing_start_method):
    # Define multithreading
    global _multiprocessing_start_method
    _multiprocessing_start_method = multiprocessing_start_method
    # Check the confidence of each orientation
    n_oris = np.shape(test_exp_maps)[0]
    # CPU count - pull from controller
    ncpus = controller.get_process_count()
    # Use multiprocessing if needed
    if ncpus == 1:
        # Rip it in
        confidence, start, stop = test_single_coordinate(test_angles,test_rMat_ss,test_gvec_cs,
                                        test_exp_maps,image_stack,experiment,test_crd,0,n_oris)
    elif ncpus > 1:
        # Chunk size - make it here ------ alternative is chunk_size = controller.get_chunk_size()
        chunk_size = controller.get_chunk_size()
        if chunk_size == -1:
            chunk_size = int(np.ceil(n_oris/ncpus))
        # Start controller
        subprocess = 'test_coordinate'
        controller.start(subprocess, n_oris)
        finished = 0
        # How many chunks do we need?
        num_chunks = int(np.ceil(n_oris/chunk_size))
        chunks = np.arange(num_chunks)
        # Initialize arrays to fill
        confidence = np.zeros(n_oris)
        # Create chunking
        starts = np.zeros(num_chunks,dtype=int)
        stops = np.zeros(num_chunks,dtype=int)
        for i in np.arange(num_chunks):
            starts[i] = i*chunk_size
            stops[i] = i*chunk_size + chunk_size
            if stops[i] >= n_oris:
                stops[i] = n_oris
        # Tell user what we are doing
        print(f"Processing {n_oris} orientations with {num_chunks} chunks of size {chunk_size} on {ncpus} CPUs.")
        # Start up the pool
        with grand_loop_pool(ncpus=ncpus,state=(starts,stops,test_angles,test_rMat_ss,test_gvec_cs,
                                            test_exp_maps,image_stack,experiment,test_crd)) as pool:
            for values, start, stop in pool.imap_unordered(test_single_coordinate_distributor,chunks):
                # Grab data as each CPU drops it
                count = np.shape(values)[0]
                confidence[start:stop] = values.squeeze()
                # Update controller
                finished += count
                controller.update(finished)
                # Clean up
                del values, start, stop
        # More cleanup
        pool.close()
        pool.join() 
    else:
        print('Must use 1 or more CPUs')
    
    # Unpack confidence and determine the orientation
    confidence = np.squeeze(confidence)
    where = np.where(confidence == np.max(confidence))
    new_exp_map = test_exp_maps[where[0],:].squeeze() # Grab just one
    if len(np.shape(new_exp_map)) == 2:
        # Not sure why I have to do this...
        new_exp_map = new_exp_map[0,:].squeeze()
        # print('Strange')
        # print(np.shape(test_exp_maps))
        # print(np.shape(test_exp_maps[where[0],:].squeeze()))
        # print(where[0])

    # Refine that orientation
    refined_exp_map, refined_conf = refine_single_coordinate(image_stack, experiment, test_crd, new_exp_map, misorientation_bnd, misorientation_spacing)
    return refined_exp_map,refined_conf

def test_single_coordinate(test_angles,test_rMat_ss,test_gvec_cs,
                                        test_exp_maps,image_stack,experiment,test_crd,start,stop):
    # Grab some experiment data
    rD = experiment.rMat_d
    tD = experiment.tVec_d
    tS = experiment.tVec_s
    base = experiment.base
    inv_deltas = experiment.inv_deltas
    clip_vals = experiment.clip_vals
    bsp = experiment.bsp
    ome_edges = experiment.ome_edges
    # Initialize
    conf = np.zeros([stop-start,1])
    # Loop through the orientations given to us
    for i in np.arange(stop-start):
        # Grab the orinetation data
        all_angles = test_angles[i+start]
        rMat_ss = test_rMat_ss[i+start]
        gvec_cs = test_gvec_cs[i+start]
        exp_map = test_exp_maps[i+start,:]
        # Calculations
        rMat_c = xfcapi.makeRotMatOfExpMap(exp_map.T) # Convert to rotation matrix
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rMat_c, tD, tS, test_crd) # Convert angles to xy detector positions
        # Check xy detector positions and omega value to see if intensity exisits
        conf[i] = _quant_and_clip_confidence(det_xy, all_angles[:, 2], image_stack,
                                        base, inv_deltas, clip_vals, bsp, ome_edges)
    # Return
    return conf, start, stop

def refine_single_coordinate(image_stack, experiment, test_crd, exp_map, misorientation_bnd, misorientation_spacing):
    logging.disable()
    # Method:
        # Go to each coordinate point and grab its non-refined orientation
        # Blow up a misorientation spacing around this orientation
        # Test those orientations
        # Find the best and return that value

    # Grab some experiment data
    plane_data = experiment.plane_data
    detector_params = experiment.detector_params
    pixel_size = experiment.pixel_size
    ome_range = experiment.ome_range
    ome_period = experiment.ome_period
    rD = experiment.rMat_d
    tD = experiment.tVec_d
    tS = experiment.tVec_s
    base = experiment.base
    inv_deltas = experiment.inv_deltas
    clip_vals = experiment.clip_vals
    bsp = experiment.bsp
    ome_edges = experiment.ome_edges
    panel_dims_expanded = [(-10, -10), (10, 10)]
    ref_gparams = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])

    # Define misorientation grid
    mis_amt = misorientation_bnd*np.pi/180.
    spacing = misorientation_spacing*np.pi/180.
    ori_pts = np.arange(-mis_amt, (mis_amt+(spacing*0.999)),spacing)
    n_grains = ori_pts.shape[0]**3
    XsO, YsO, ZsO = np.meshgrid(ori_pts, ori_pts, ori_pts)
    grid0 = np.vstack([XsO.flatten(), YsO.flatten(), ZsO.flatten()]).T

    # Add Misorientation
    exp_maps = grid0+np.r_[exp_map] # Define all sub orientations around the single orientation
    all_rMat_c = rotations.rotMatOfExpMap(exp_maps.T) # Convert to rotation matrix
    #all_rMat_c = xfcapi.makeRotMatOfExpMap(exp_maps.T) # Much faster but can't use this since it is only for single exp maps at a time
    # Check the confidence of each g vector
    confidence = np.zeros((n_grains, 1)) # Place for confidence values for each individaul sub orientation
    # Loop over each sub orientation
    for i, exp_map in enumerate(exp_maps):
        gparams = np.hstack([exp_map, ref_gparams]) # Stack grain data into single variable
        # Simulate the angles for each of these orientations
        sim_results = xrdutil.simulateGVecs(plane_data,detector_params,gparams,panel_dims=panel_dims_expanded,
                                            pixel_pitch=pixel_size,ome_range=ome_range,ome_period=ome_period,
                                            distortion=None)
        all_angles = sim_results[2] # Grab just the angles 
        rMat_ss = xfcapi.make_sample_rmat(experiment.chi, all_angles[:, 2]) # Calculate the sample rotation matrix
        # gvec_cs = _anglesToGVec(all_angles, rMat_ss, all_rMat_c[i]) # Convert to g vectors
        gvec_cs = xfcapi.anglesToGVec(all_angles, rMat_c=all_rMat_c[i]) # Much faster
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, all_rMat_c[i], tD, tS, test_crd) # Convert angles to xy detector positions
        # Check xy detector positions and omega value to see if intensity exisits
        confidence[i] = _quant_and_clip_confidence(det_xy, all_angles[:, 2], image_stack,
                                        base, inv_deltas, clip_vals, bsp, ome_edges)
        # anglesToGVec and Quant_and_clip are the thing that take about 50% of time each
        
    # Unpack confidence and determine refined orientation
    confidence = np.squeeze(confidence)
    where = np.where(confidence == np.max(confidence))
    new_exp_map = exp_maps[where[0],:].squeeze() # Grab just one
    new_conf = np.max(confidence)

    return new_exp_map, new_conf

def gather_confidence(controller, confidence, n_grains, n_coords):
    if rank == 0:
        global_confidence = np.empty(n_grains * n_coords, dtype=np.float64)
    else:
        global_confidence = None

    # Calculate the send buffer sizes
    coords_per_rank = n_coords // world_size
    send_counts = np.full(world_size, coords_per_rank * n_grains)
    send_counts[-1] = (n_coords - (coords_per_rank * (world_size-1))) * n_grains

    if rank == 0:
        # Time how long it takes to perform the MPI gather
        controller.start('gather_confidence', 1)

    # Transpose so the data will be more easily re-shaped into its final shape
    # Must be flattened as well so the underlying data is modified...
    comm.Gatherv(confidence.T.flatten(), (global_confidence, send_counts), root=0)
    if rank == 0:
        controller.finish('gather_confidence')
        confidence = global_confidence.reshape(n_coords, n_grains).T
        controller.handle_result("confidence", confidence)

# %% ============================================================================
# Some multiprocessing
# ===============================================================================
# The parallellized part of test_orientations uses some big arrays as part of
# the state that needs to be communicated to the spawn processes.
#
# On fork platforms, take advantage of process memory inheritance.
#
# On non fork platforms, rely on joblib dumping the state to disk and loading
# back in the target processes, pickling only the minimal information to load
# state back. Pickling the big arrays directly was causing memory errors and
# would be less efficient in memory (as joblib memmaps by default the big
# arrays, meaning they may be shared between processes).
def get_offset_size(n_coords):
    offset = 0
    size = n_coords
    if USE_MPI:
        coords_per_rank = n_coords // world_size
        offset = rank * coords_per_rank

        size = coords_per_rank
        if rank == world_size - 1:
            size = n_coords - offset

    return (offset, size)

# Controller for multiprocessing
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

def multiproc_inner_loop(chunk):
    """function to use in multiprocessing that computes the simulation over the
    task's alloted chunk of data"""

    chunk_size = _mp_state[0]
    n_coords = len(_mp_state[5])

    (offset, size) = get_offset_size(n_coords)

    chunk_stop = min(offset+size, chunk+chunk_size)
    return _grand_loop_inner(*_mp_state[1:], start=chunk, stop=chunk_stop)

def precompute_orientation_information_distributor(chunk_num):
    # Distributor function
    starts = _mp_state[0]
    stops = _mp_state[1]
    return precompute_orientation_information(*_mp_state[2:], start=starts[chunk_num], stop=stops[chunk_num])

def test_single_coordinate_distributor(chunk_num):
    # Distributor function
    starts = _mp_state[0]
    stops = _mp_state[1]
    return test_single_coordinate(*_mp_state[2:], start=starts[chunk_num], stop=stops[chunk_num])

def worker_init(id_state, id_exp):
    """process initialization function. This function is only used when the
    child processes are spawned (instead of forked). When using the fork model
    of multiprocessing the data is just inherited in process memory."""
    import joblib

    global _mp_state
    state = joblib.load(id_state)
    experiment = joblib.load(id_exp)
    _mp_state = state + (experiment,)

@contextlib.contextmanager
def grand_loop_pool(ncpus, state):
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
    global _multiprocessing_start_method

    try:
        multiprocessing.set_start_method(_multiprocessing_start_method)
    except:
        print('Multiprocessing context already set')

    if _multiprocessing_start_method == 'fork':
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
# Test grid generation
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

def gen_nf_test_grid_tomo(x_dim_pnts, z_dim_pnts, v_bnds, voxel_spacing):

    if v_bnds[0]==v_bnds[1]:
        Xs,Ys,Zs=np.meshgrid(np.arange(x_dim_pnts),v_bnds[0],np.arange(z_dim_pnts))
    else:
        Xs,Ys,Zs=np.meshgrid(np.arange(x_dim_pnts),np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing),np.arange(z_dim_pnts))
        #note numpy shaping of arrays is goofy, returns(length(y),length(x),length(z))


    Zs=(Zs-(z_dim_pnts/2))*voxel_spacing
    Xs=(Xs-(x_dim_pnts/2))*voxel_spacing


    test_crds = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds)

    return test_crds, n_crds, Xs, Ys, Zs

# %% ============================================================================
# Image Processing
# ===============================================================================
# Old image dilation
def get_dilated_image_stack(image_stack, experiment, controller,
                            cache_file='gold_cubes_dilated.npy'):

    try:
        dilated_image_stack = np.load(cache_file, mmap_mode='r',
                                      allow_pickle=False)
    except Exception:
        dilated_image_stack = dilate_image_stack(image_stack, experiment,
                                                 controller)
        np.save(cache_file, dilated_image_stack)

    return dilated_image_stack
# Old image dilation
def dilate_image_stack(image_stack, experiment, controller):
    # first, perform image dilation ===========================================
    # perform image dilation (using scikit_image dilation)
    subprocess = 'dilate image_stack'
    dilation_shape = np.ones((2*experiment.row_dilation + 1,
                              2*experiment.col_dilation + 1),
                             dtype=np.uint8)
    image_stack_dilated = np.empty_like(image_stack)
    dilated = np.empty(
        (image_stack.shape[-2], image_stack.shape[-1] << 3),
        dtype=bool
    )
    n_images = len(image_stack)
    controller.start(subprocess, n_images)
    for i_image in range(n_images):
        to_dilate = np.unpackbits(image_stack[i_image], axis=-1)
        ski_dilation(to_dilate, dilation_shape,
                     out=dilated)
        image_stack_dilated[i_image] = np.packbits(dilated, axis=-1)
        controller.update(i_image + 1)
    controller.finish(subprocess)

    return image_stack_dilated
# Old darkfiled generation
def gen_nf_dark(data_folder,img_nums,num_for_dark,nrows,ncols,dark_type='median',stem='nf_',num_digits=5,ext='.tif'):

    dark_stack=np.zeros([num_for_dark,nrows,ncols])

    print('Loading data for dark generation...')
    for ii in np.arange(num_for_dark):
        print('Image #: ' + str(ii))
        dark_stack[ii,:,:]=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)
        #image_stack[ii,:,:]=np.flipud(tmp_img>threshold)

    if dark_type=='median':
        print('making median...')
        dark=np.median(dark_stack,axis=0)
    elif dark_type=='min':
        print('making min...')
        dark=np.min(dark_stack,axis=0)

    return dark
# Old image cleaner
def gen_nf_cleaned_image_stack(data_folder,img_nums,dark,nrows,ncols, \
                               process_type='gaussian',process_args=[4.5,5], \
                               threshold=1.5,ome_dilation_iter=1,stem='nf_', \
                               num_digits=5,ext='.tif'):

    image_stack=np.zeros([img_nums.shape[0],nrows,ncols],dtype=bool)

    print('Loading and Cleaning Images...')


    if process_type=='gaussian':
        sigma=process_args[0]
        size=process_args[1].astype(int) #needs to be int

        for ii in np.arange(img_nums.shape[0]):
            print('Image #: ' + str(ii))
            tmp_img=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
            #image procesing
            tmp_img = filters.gaussian(tmp_img, sigma=sigma)

            tmp_img = img.morphology.grey_closing(tmp_img,size=(size,size))

            binary_img = img.morphology.binary_fill_holes(tmp_img>threshold)
            image_stack[ii,:,:]=binary_img
            plt.imshow(binary_img,interpolation='none')

    else:

        num_erosions=process_args[0]
        num_dilations=process_args[1]


        for ii in np.arange(img_nums.shape[0]):
            print('Image #: ' + str(ii))
            tmp_img=imgio.imread(data_folder+'%s'%(stem)+str(img_nums[ii]).zfill(num_digits)+ext)-dark
            #image procesing
            image_stack[ii,:,:]=img.morphology.binary_erosion(tmp_img>threshold,iterations=num_erosions)
            image_stack[ii,:,:]=img.morphology.binary_dilation(image_stack[ii,:,:],iterations=num_dilations)


    #%A final dilation that includes omega
    print('Final Dilation Including Omega....')
    image_stack=img.morphology.binary_dilation(image_stack,iterations=ome_dilation_iter)


    return image_stack

# %% ============================================================================
# Data processors and savers
# ===============================================================================
def gen_trial_exp_data(grain_out_file,det_file,mat_file, mat_name, max_tth, comp_thresh, chi2_thresh,omega_edges_deg, 
                       beam_stop_parms, misorientation_bnd=0.0, misorientation_spacing=0.25):


    print('Loading Grain Data...')
    #gen_grain_data
    ff_data=np.loadtxt(grain_out_file)

    #ff_data=np.atleast_2d(ff_data[2,:])

    exp_maps=ff_data[:,3:6]
    t_vec_ds=ff_data[:,6:9]

    #
    completeness=ff_data[:,1]

    chi2=ff_data[:,2]

    n_grains=exp_maps.shape[0]

    rMat_c = rotations.rotMatOfExpMap(exp_maps.T)

    cut=np.where(np.logical_and(completeness>comp_thresh,chi2<chi2_thresh))[0]
    exp_maps=exp_maps[cut,:]
    t_vec_ds=t_vec_ds[cut,:]
    chi2=chi2[cut]


    # Add Misorientation
    mis_amt=misorientation_bnd*np.pi/180.
    spacing=misorientation_spacing*np.pi/180.

    mis_steps = int(misorientation_bnd/misorientation_spacing)

    ori_pts = np.arange(-mis_amt, (mis_amt+(spacing*0.999)),spacing)
    num_ori_grid_pts=ori_pts.shape[0]**3
    num_oris=exp_maps.shape[0]


    XsO, YsO, ZsO = np.meshgrid(ori_pts, ori_pts, ori_pts)

    grid0 = np.vstack([XsO.flatten(), YsO.flatten(), ZsO.flatten()]).T


    exp_maps_expanded=np.zeros([num_ori_grid_pts*num_oris,3])
    t_vec_ds_expanded=np.zeros([num_ori_grid_pts*num_oris,3])


    for ii in np.arange(num_oris):
        pts_to_use=np.arange(num_ori_grid_pts)+ii*num_ori_grid_pts
        exp_maps_expanded[pts_to_use,:]=grid0+np.r_[exp_maps[ii,:] ]
        t_vec_ds_expanded[pts_to_use,:]=np.r_[t_vec_ds[ii,:] ]


    exp_maps=exp_maps_expanded
    t_vec_ds=t_vec_ds_expanded

    n_grains=exp_maps.shape[0]

    rMat_c = rotations.rotMatOfExpMap(exp_maps.T)


    print('Loading Instrument Data...')
    # CHANGES FROM SEG 05/31/2023
    # OLD ---------
    # ome_period_deg=(ome_range_deg[0][0], (ome_range_deg[0][0]+360.)) #degrees
    # ome_step_deg=(ome_range_deg[0][1]-ome_range_deg[0][0])/nframes #degrees
    # ome_period = (ome_period_deg[0]*np.pi/180.,ome_period_deg[1]*np.pi/180.)
    # ome_range = [(ome_range_deg[0][0]*np.pi/180.,ome_range_deg[0][1]*np.pi/180.)]
    # ome_step = ome_step_deg*np.pi/180.
    # ome_edges = np.arange(nframes+1)*ome_step+ome_range[0][0]#fixed 2/26/17
    # NEW ---------
    
    # How many frames do we have?
    nframes = np.shape(omega_edges_deg)[0]-1
    
    # Define variables in degrees
    # Omega range is the experimental span of omega space
    ome_range_deg = [(omega_edges_deg[0],omega_edges_deg[nframes])]  # degrees
    # Omega period is the range in which your omega space lies (often 0 to 360 or -180 to 180)
    ome_period_deg = (ome_range_deg[0][0], ome_range_deg[0][0]+360.) #degrees
    
    # Define variables in radians
    ome_period = (ome_period_deg[0]*np.pi/180.,ome_period_deg[1]*np.pi/180.)
    ome_range = [(ome_range_deg[0][0]*np.pi/180.,ome_range_deg[0][1]*np.pi/180.)]

    # Define omega edges - First value is the ome start position of frame one, last value is the ome end position of final frame
    ome_edges = omega_edges_deg*np.pi/180
    
    # END CHANGES --------
    
    instr=load_instrument(det_file)
    panel = next(iter(instr.detectors.values()))  # !!! there is only 1

        # tranform paramters
    #   Sample
    chi = instr.chi
    tVec_s = instr.tvec
    #   Detector
    rMat_d = panel.rmat
    tilt_angles_xyzp = np.asarray(rotations.angles_from_rmat_xyz(rMat_d))
    tVec_d = panel.tvec

    # pixels
    row_ps = panel.pixel_size_row
    col_ps = panel.pixel_size_col
    pixel_size = (row_ps, col_ps)
    nrows = panel.rows
    ncols = panel.cols

    # panel dimensions
    panel_dims = [tuple(panel.corner_ll),
                  tuple(panel.corner_ur)]

    x_col_edges = panel.col_edge_vec
    y_row_edges = panel.row_edge_vec
    rx, ry = np.meshgrid(x_col_edges, y_row_edges)

    max_pixel_tth = instrument.max_tth(instr)

    detector_params = np.hstack([tilt_angles_xyzp, tVec_d, chi, tVec_s])
    distortion = panel.distortion  # !!! must be None for now

    # a different parametrization for the sensor
    # (makes for faster quantization)
    base = np.array([x_col_edges[0],
                     y_row_edges[0],
                     ome_edges[0]])
    deltas = np.array([x_col_edges[1] - x_col_edges[0],
                       y_row_edges[1] - y_row_edges[0],
                       ome_edges[1] - ome_edges[0]])
    inv_deltas = 1.0/deltas
    clip_vals = np.array([ncols, nrows])

    # # dilation
    # max_diameter = np.sqrt(3)*0.005
    # row_dilation = int(np.ceil(0.5 * max_diameter/row_ps))
    # col_dilation = int(np.ceil(0.5 * max_diameter/col_ps))



    print('Loading Materials Data...')
    # crystallography data
    beam_energy = valunits.valWUnit("beam_energy", "energy", instr.beam_energy, "keV")
    beam_wavelength = constants.keVToAngstrom(beam_energy.getVal('keV'))
    dmin = valunits.valWUnit("dmin", "length",
                             0.5*beam_wavelength/np.sin(0.5*max_pixel_tth),
                             "angstrom")

    # material loading
    mats = material.load_materials_hdf5(mat_file, dmin=dmin,kev=beam_energy)
    pd = mats[mat_name].planeData

    if max_tth is not None:
         pd.tThMax = np.amax(np.radians(max_tth))
    else:
        pd.tThMax = np.amax(max_pixel_tth)



    print('Final Assembly...')
    experiment = argparse.Namespace()
    # grains related information
    experiment.n_grains = n_grains  # this can be derived from other values...
    experiment.rMat_c = rMat_c  # n_grains rotation matrices (one per grain)
    experiment.exp_maps = exp_maps  # n_grains exp_maps (one per grain)

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
    experiment.rMat_c = rMat_c
    # ns.row_dilation = 0 #done beforehand row_dilation, disabled
    # experiemnt.col_dilation = 0 #col_dilation
    experiment.distortion = distortion
    experiment.panel_dims = panel_dims  # used only in simulate...
    experiment.base = base
    experiment.inv_deltas = inv_deltas
    experiment.clip_vals = clip_vals
    experiment.bsp = beam_stop_parms
    experiment.mat = mats


    if mis_steps ==0:
        nf_to_ff_id_map = cut
    else:
        nf_to_ff_id_map=np.tile(cut,3**3*mis_steps)

    return experiment, nf_to_ff_id_map

def process_raw_confidence(raw_confidence,vol_shape=None,id_remap=None,min_thresh=0.0):

    print('Compiling Confidence Map...')
    if vol_shape == None:
        confidence_map=np.max(raw_confidence,axis=0)
        grain_map=np.argmax(raw_confidence,axis=0)
    else:
        confidence_map=np.max(raw_confidence,axis=0).reshape(vol_shape)
        grain_map=np.argmax(raw_confidence,axis=0).reshape(vol_shape)

    #fix grain indexing
    not_indexed=np.where(confidence_map<=min_thresh)
    grain_map[not_indexed] =-1

    if id_remap is not None:
        max_grain_no=np.max(grain_map)
        grain_map_copy=copy.copy(grain_map)
        print('Remapping grain ids to ff...')
        for ii in np.arange(max_grain_no+1): # SEG CHANGE 8/28/2023: Added +1, prior version left the max grain ID the same resulting in a double grain
            this_grain=np.where(grain_map==ii)
            grain_map_copy[this_grain]=id_remap[ii]
        grain_map=grain_map_copy

    return grain_map.astype(int), confidence_map

# Saves the raw confidence map - this is likely a very large file
def save_raw_confidence(save_dir,save_stem,raw_confidence,id_remap=None):
    print('Saving raw confidence, might take a while...')
    if id_remap is not None:
        np.savez(save_dir+save_stem+'_raw_confidence.npz',raw_confidence=raw_confidence,id_remap=id_remap)
    else:
        np.savez(save_dir+save_stem+'_raw_confidence.npz',raw_confidence=raw_confidence)

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
# Data plotters
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
# Diffraction volume stitchers
# ===============================================================================
# A grain map output which handles both overlap and merging grains together
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

# An older grain map output which does not handle overlap or merging grain together
def output_grain_map(data_location,data_stems,output_stem,vol_spacing,top_down=True,save_type=['npz']):

    num_scans=len(data_stems)

    confidence_maps=[None]*num_scans
    grain_maps=[None]*num_scans
    Xss=[None]*num_scans
    Yss=[None]*num_scans
    Zss=[None]*num_scans

    if len(vol_spacing)==1:
        vol_shifts=np.arange(0,vol_spacing[0]*num_scans+1e-12,vol_spacing[0])
    else:
        vol_shifts=vol_spacing


    for ii in np.arange(num_scans):
        print('Loading Volume %d ....'%(ii))
        conf_data=np.load(os.path.join(data_location,data_stems[ii]+'_grain_map_data.npz'))

        confidence_maps[ii]=conf_data['confidence_map']
        grain_maps[ii]=conf_data['grain_map']
        Xss[ii]=conf_data['Xs']
        Yss[ii]=conf_data['Ys']
        Zss[ii]=conf_data['Zs']

    #assumes all volumes to be the same size
    num_layers=grain_maps[0].shape[0]

    total_layers=num_layers*num_scans

    num_rows=grain_maps[0].shape[1]
    num_cols=grain_maps[0].shape[2]

    grain_map_stitched=np.zeros((total_layers,num_rows,num_cols))
    confidence_stitched=np.zeros((total_layers,num_rows,num_cols))
    Xs_stitched=np.zeros((total_layers,num_rows,num_cols))
    Ys_stitched=np.zeros((total_layers,num_rows,num_cols))
    Zs_stitched=np.zeros((total_layers,num_rows,num_cols))


    for ii in np.arange(num_scans):
        if top_down==True:
            grain_map_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=grain_maps[num_scans-1-ii]
            confidence_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=confidence_maps[num_scans-1-ii]
            Xs_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=\
                Xss[num_scans-1-ii]
            Zs_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=\
                Zss[num_scans-1-ii]
            Ys_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=Yss[num_scans-1-ii]+vol_shifts[ii]
        else:

            grain_map_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=grain_maps[ii]
            confidence_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=confidence_maps[ii]
            Xs_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=Xss[ii]
            Zs_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=Zss[ii]
            Ys_stitched[((ii)*num_layers):((ii)*num_layers+num_layers),:,:]=Yss[ii]+vol_shifts[ii]

    for ii in np.arange(len(save_type)):

        if save_type[ii] == 'hdf5':

            print('Writing HDF5 data...')

            hf = h5py.File(output_stem + '_assembled.h5', 'w')
            hf.create_dataset('grain_map', data=grain_map_stitched)
            hf.create_dataset('confidence', data=confidence_stitched)
            hf.create_dataset('Xs', data=Xs_stitched)
            hf.create_dataset('Ys', data=Ys_stitched)
            hf.create_dataset('Zs', data=Zs_stitched)

        elif save_type[ii]=='npz':

            print('Writing NPZ data...')

            np.savez(output_stem + '_assembled.npz',\
             grain_map=grain_map_stitched,confidence=confidence_stitched,
             Xs=Xs_stitched,Ys=Ys_stitched,Zs=Zs_stitched)

        elif save_type[ii]=='vtk':


            print('Writing VTK data...')
            # VTK Dump
            Xslist=Xs_stitched[:,:,:].ravel()
            Yslist=Ys_stitched[:,:,:].ravel()
            Zslist=Zs_stitched[:,:,:].ravel()

            grainlist=grain_map_stitched[:,:,:].ravel()
            conflist=confidence_stitched[:,:,:].ravel()

            num_pts=Xslist.shape[0]
            num_cells=(total_layers-1)*(num_rows-1)*(num_cols-1)

            f = open(os.path.join(output_stem +'_assembled.vtk'), 'w')


            f.write('# vtk DataFile Version 3.0\n')
            f.write('grainmap Data\n')
            f.write('ASCII\n')
            f.write('DATASET UNSTRUCTURED_GRID\n')
            f.write('POINTS %d double\n' % (num_pts))

            for i in np.arange(num_pts):
                f.write('%e %e %e \n' %(Xslist[i],Yslist[i],Zslist[i]))

            scale2=num_cols*num_rows
            scale1=num_cols

            f.write('CELLS %d %d\n' % (num_cells, 9*num_cells))
            for k in np.arange(Xs_stitched.shape[0]-1):
                for j in np.arange(Xs_stitched.shape[1]-1):
                    for i in np.arange(Xs_stitched.shape[2]-1):
                        base=scale2*k+scale1*j+i
                        p1=base
                        p2=base+1
                        p3=base+1+scale1
                        p4=base+scale1
                        p5=base+scale2
                        p6=base+scale2+1
                        p7=base+scale2+scale1+1
                        p8=base+scale2+scale1

                        f.write('8 %d %d %d %d %d %d %d %d \n' \
                                %(p1,p2,p3,p4,p5,p6,p7,p8))


            f.write('CELL_TYPES %d \n' % (num_cells))
            for i in np.arange(num_cells):
                f.write('12 \n')

            f.write('POINT_DATA %d \n' % (num_pts))
            f.write('SCALARS grain_id int \n')
            f.write('LOOKUP_TABLE default \n')
            for i in np.arange(num_pts):
                f.write('%d \n' %(grainlist[i]))

            f.write('FIELD FieldData 1 \n' )
            f.write('confidence 1 %d float \n' % (num_pts))
            for i in np.arange(num_pts):
                f.write('%e \n' %(conflist[i]))


            f.close()

        else:
            print('Not a valid save option, npz, vtk, or hdf5 allowed.')

    return grain_map_stitched, confidence_stitched, Xs_stitched, Ys_stitched, \
            Zs_stitched

# %% ============================================================================
# Calibration
# ===============================================================================
# An old calibration routine
def scan_detector_parm(image_stack, experiment,test_crds,controller,parm_to_opt,parm_range,slice_shape,ang='deg'):
    #0-distance
    #1-x center
    #2-y center
    #3-xtilt
    #4-ytilt
    #5-ztilt

    parm_vector=np.arange(parm_range[0],parm_range[1]+1e-6,(parm_range[1]-parm_range[0])/parm_range[2])


    if parm_to_opt>2 and ang=='deg':
        parm_vector=parm_vector*np.pi/180.

    multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

    #current detector parameters, note the value for the actively optimized parameters will be ignored
    distance=experiment.detector_params[5]#mm
    x_cen=experiment.detector_params[3]#mm
    y_cen=experiment.detector_params[4]#mm
    xtilt=experiment.detector_params[0]
    ytilt=experiment.detector_params[1]
    ztilt=experiment.detector_params[2]
    ome_range=copy.copy(experiment.ome_range)
    ome_period=copy.copy(experiment.ome_period)
    ome_edges=copy.copy(experiment.ome_edges)

    num_parm_pts=len(parm_vector)

    trial_data=np.zeros([num_parm_pts,slice_shape[0],slice_shape[1]])

    tmp_td=copy.copy(experiment.tVec_d)
    for jj in np.arange(num_parm_pts):
        print('cycle %d of %d'%(jj+1,num_parm_pts))

        #overwrite translation vector components
        if parm_to_opt==0:
            tmp_td[2]=parm_vector[jj]
        if parm_to_opt==1:
            tmp_td[0]=parm_vector[jj]
        if parm_to_opt==2:
            tmp_td[1]=parm_vector[jj]
        if  parm_to_opt==3:
            rMat_d_tmp=xfcapi.makeDetectorRotMat([parm_vector[jj],ytilt,ztilt])
        elif parm_to_opt==4:
            rMat_d_tmp=xfcapi.makeDetectorRotMat([xtilt,parm_vector[jj],ztilt])
        elif parm_to_opt==5:
            rMat_d_tmp=xfcapi.makeDetectorRotMat([xtilt,ytilt,parm_vector[jj]])
        else:
            rMat_d_tmp=xfcapi.makeDetectorRotMat([xtilt,ytilt,ztilt])

        experiment.rMat_d = rMat_d_tmp
        experiment.tVec_d = tmp_td

        if parm_to_opt==6:

            experiment.ome_range=[(ome_range[0][0]-parm_vector[jj],ome_range[0][1]-parm_vector[jj])]
            experiment.ome_period=(ome_period[0]-parm_vector[jj],ome_period[1]-parm_vector[jj])
            experiment.ome_edges=np.array(ome_edges-parm_vector[jj])
            experiment.base[2]=experiment.ome_edges[0]

            # print(experiment.ome_range)
            # print(experiment.ome_period)
            # print(experiment.ome_edges)
            # print(experiment.base)

        conf=test_orientations(image_stack, experiment,test_crds,controller, \
                               multiprocessing_start_method)


        trial_data[jj]=np.max(conf,axis=0).reshape(slice_shape)

    return trial_data, parm_vector

# %% ============================================================================
# Missing Grains Utility Functions
# ===============================================================================
# Function to find regions of low confidence with a single or merged volume
def find_low_confidence_centroids(reconstructed_data_path,output_dir,output_stem,num_diffraction_volumes=1,confidence_threshold=0.5,min_size=5,centroids_or_sparse_grid=0):
    # Data path must point to a npz save of either a merged volume or single volume
    # Within this file there MUST be:
        # confidence, X, Y, Z, and mask
        # If you do not have a mask, make one.  

    # Load the data
    reconstruction = np.load(reconstructed_data_path)
    confidence_map = reconstruction['confidence_map']
    mask = reconstruction['tomo_mask']
    Xs = reconstruction['Xs']
    if num_diffraction_volumes > 1:
        Ys = reconstruction['Ys'] + reconstruction['vertical_position_full']
        diffraction_volume = reconstruction['diffraction_volume']
    else:
        Ys = reconstruction['Ys']
    Zs = reconstruction['Zs']

    # Find the regions of low confidence
    low_confidence_map = confidence_map<confidence_threshold
    low_confidence_map[mask == 0] = 0
    low_confidence_map[mask == 1] = 1

    if centroids_or_sparse_grid == 1:
        # Find our low confidence regions
        labeled_array, num_features = img.label(low_confidence_map,structure=np.ones([3,3,3]))

        # Find centroids 
        centroids = img.center_of_mass(low_confidence_map, labeled_array, np.unique(labeled_array))
        # Removed the first centroid which is for the high confidence region
        centroids = centroids[1:]

        point_map = np.zeros(np.shape(confidence_map)).astype(bool)
        count = 1
        for i in range(1,len(centroids)): # This is actually the ID - starts at one
            where = len(np.where(labeled_array==i)[0]) # How many pixels?
            if where >= min_size: # Size threshold
                # Flag the centroid pixel
                point_map[np.rint(centroids[i][0]).astype('int'),np.rint(centroids[i][1]).astype('int'), np.rint(centroids[i][2]).astype('int')] = True
                print(str(count) + ' total centroids populated to test for a missing grain.')
                count = count + 1

    elif centroids_or_sparse_grid == 0:
        # Create sparse grid mask
        # Make all even indices 1 and odd indices 0
        sparse_mask = np.ones(np.shape(low_confidence_map))
        sparse_mask[::2,:,:] = 0
        sparse_mask[:,::2,:] = 0
        sparse_mask[:,:,::2] = 0
        sparse_mask = img.binary_dilation(sparse_mask,structure=np.array([[[1,0,1],[0, 0, 0],[1,0,1]],
                                                                 [[0,0,0],[0, 1, 0],[0,0,0]],
                                                                 [[1,0,1],[0, 0, 0],[1,0,1]]]))

        # Scatter shot low confidence map
        point_map = np.logical_and(low_confidence_map,sparse_mask)

    # What are these poistions in the lab coordinate system?
    Xs_positions = Xs[point_map]
    Ys_positions = Ys[point_map]
    Zs_positions = Zs[point_map]

    # Generate and save the test coordinates
    if num_diffraction_volumes > 1:
        which_vols = diffraction_volume[point_map]
        for vol in np.arange(num_diffraction_volumes):
            test_coordiates = np.vstack([Xs_positions[which_vols==vol], Ys_positions[which_vols==vol], Zs_positions[which_vols==vol]]).T
            save_str = output_dir + output_stem + '_diffraction_vol_%d.npy' % (vol+1)
            np.save(save_str,test_coordiates)
    else:
        test_coordiates = np.vstack([Xs_positions, Ys_positions, Zs_positions]).T
        save_str = output_dir + output_stem + '.npy'
        np.save(save_str,test_coordiates)
    
    print('Test coordinates saved.')

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

# %%



