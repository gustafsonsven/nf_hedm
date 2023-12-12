import numba
import numpy as np
import math
import bisect
import scipy
import timeit
import skimage

from hexrd.transforms import xfcapi
from hexrd import xrdutil
from hexrd import rotations


import timeit
import logging
import os
import multiprocessing
import tempfile
import shutil
import contextlib
import numpy as np









#from .process_controller import set_multiprocessing_method, multiprocessing_pool, _mp_state
from .data_io import _load_images

# %% ============================================================================
# NUMBA FUNCTIONS
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
    #@profile
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
        # if len(bsp) == 2: # Added this flag for handling if we have a mask type beamstop - SEG 10/28/2023
        #     if abs(yf-bsp[0]) < (bsp[1]/2.):
        #         continue
        
        xf = np.floor((xf - base[0]) * inv_deltas[0])
        #xf = math.floor((coords[i, 0] - base[0]) * inv_deltas[0])

        #if not (0.0 <= xf < clip_vals[0]): continue

        if not xf >= 0.0:
            continue
        if not xf < clip_vals[0]:
            continue

        #yf = math.floor((coords[i, 1] - base[1]) * inv_deltas[1])
        yf = np.floor((yf - base[1]) * inv_deltas[1])


        #if not (0.0 <= yf < clip_vals[1]): continue

        if not yf >= 0.0:
            continue
        if not yf < clip_vals[1]:
            continue
        
        # Adding 2D 'beamstop' mask functionality to handle the 2x objective lens + scinitaltor issues - SEG 10/28/2023
        # The beamstop parameter is now the shape of a single image
        # The beamstop mask is TRUE on the beamstop/past the edge of scintilator
        # Comment out the top bsp function
        # if len(bsp) > 2:
        if bsp[int(yf), int(xf)]: continue

        # CHANGE SEG 6/22/2023 and 10/03/2023 - Put in a binary serach of the omega edges
        #ome_pos = angles[i]
        # While bisect left is nominally faster - it does not work with numba
        # Bisect left returns the index, j, such that all ome_edges[0:j] < ome_pos
        # zf = bisect.bisect_left(ome_edges, ome_pos) - 1
        # This method is faster when combined with numba
        #zf = find_target_index(ome_edges, angles[i])
        zf = find_target_index(ome_edges, angles[i])
        in_sensor += 1

        x, y, z = int(xf), int(yf), int(zf)

        if image[z, y, x]:
            matches += 1

    return 0 if in_sensor == 0 else float(matches)/float(in_sensor)

# %% ============================================================================
# PROCESSOR FUNCTIONS
# ===============================================================================
#@profile
def _test_single_orientation_at_single_coordinate(experiment,image_stack,coord_to_test,orientation_data_to_test,refine_yes_no=0):
    """
        Goal: 

        Input:
        Output:

    """
    if refine_yes_no == 0:
        # No refinement needed - just test the orientation
        # Unpack the precomputed orientation data
        exp_map, angles, rMat_ss, gvec_cs, rMat_c = orientation_data_to_test
        # Grab some experiment data
        tD = experiment.tVec_d # Detector X,Y,Z translation (mm)
        rD = experiment.rMat_d # Detector rotation matrix (rad)
        tS = experiment.tVec_s # Sample X,Y,Z translation (mm)
        base = experiment.base # Physical position of (0,0) pixel at omega = 0 [X,Y,omega] = [mm,mm,rad]
        inv_deltas = experiment.inv_deltas # 1 over step size along X,Y,omega in image stack [1/mm,1/mm/,1/rad]
        clip_vals = experiment.clip_vals # Number of pixels along X,Y [mm,mm]
        bsp = experiment.bsp # Beam stop parameters [vertical center,width] [mm,mm]
        ome_edges = experiment.ome_edges # Omega start stop positions for each frame in image stack
        # Find where those g-vectors intercept the detector from our coordinate point
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, np.squeeze(rMat_c), tD, tS, coord_to_test)
        # Check xy detector positions and omega value to see if intensity exisits
        confidence = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                        base, inv_deltas, clip_vals, bsp, ome_edges)
        # Return the orienation and its confidence
        misorientation = 0
    elif refine_yes_no == 1:
        # Refinement needed
        # Unpack the precomputed orientation data
        original_exp_map = orientation_data_to_test[0]
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
        all_exp_maps = grid0 + np.r_[original_exp_map] # Define all sub orientations around the single orientation

        # Initialize an array to hold the confidence values
        n_oris = ori_pts.shape[0]**3
        all_confidence = np.zeros(n_oris)

        # Check each orientation for its confidence at the coordinate point
        for i in np.arange(n_oris):
            # Grab orientation information
            exp_map = all_exp_maps[i,:]
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
            # Find where those g-vectors intercept the detector from our coordinate point
            det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rMat_c, tD, tS, coord_to_test)
            # Check xy detector positions and omega value to see if intensity exisits
            all_confidence[i] = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                            base, inv_deltas, clip_vals, bsp, ome_edges)
            
        # Find the index of the max confidence
        idx = np.where(all_confidence == np.max(all_confidence))[0][0] # Grab just the first instance if there is a tie

        # What is the hightest confidence orientation and what is its confidence
        exp_map = all_exp_maps[idx,:]
        confidence = all_confidence[idx]

        # What is the misorientation, in degrees between this exp_map and the original?
        original_quats = np.atleast_2d(rotations.quatOfExpMap(original_exp_map))
        refined_quats = np.atleast_2d(rotations.quatOfExpMap(exp_map))
        [misorientation, a] = rotations.misorientation(original_quats.T,refined_quats.T) # In radians
        misorientation = np.degrees(misorientation)
    
    # Ensure output is the correct size
    if len(np.shape(exp_map)) == 1: exp_map = np.expand_dims(exp_map,0)
    if len(np.shape(confidence)) == 0: confidence = np.expand_dims(confidence,0)

    return exp_map, confidence, misorientation

#@profile
def _test_single_orientation_at_many_coordinates(experiment,image_stack,coords_to_test,orientation_data_to_test):
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
    all_confidence = np.zeros(n_coords)
    all_exp_maps = np.zeros([n_coords,3])

    # Check each orientation at the coordinate point
    for i in np.arange(n_coords):
        # What is our coordinate?
        coord_to_test = coords_to_test[i,:]
        # Find intercept point of each g-vector on the detector
        det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, np.squeeze(rMat_c), tD, tS, coord_to_test) # Convert angles to xy detector positions
        # Check detector positions and omega values to see if intensity exisits
        all_confidence[i] = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                        base, inv_deltas, clip_vals, bsp, ome_edges)
        all_exp_maps[i,:] = exp_map

    # Return the confidence value at each coordinate point
    return all_exp_maps, all_confidence

#@profile
def _test_many_orientations_at_single_coordinate(experiment,image_stack,coord_to_test,orientation_data_to_test):
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
        # Check for centroid position
        if np.linalg.norm(coord_to_test - experiment.t_vec_s[i,:]) <= experiment.centroid_serach_radius:
            # We are close enough, test the orientation
            # Grab orientation information
            angles = all_angles[i]
            rMat_ss = all_rMat_ss[i]
            gvec_cs = all_gvec_cs[i]
            rMat_c = all_rMat_c[i]
            # Find intercept point of each g-vector on the detector
            det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rMat_c, tD, tS, coord_to_test) # Convert angles to xy detector positions
            # Check detector positions and omega values to see if intensity exisits
            all_confidence[i] = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                            base, inv_deltas, clip_vals, bsp, ome_edges)
        else:
            # Not close enough ignore
            all_confidence[i] = 0

    if np.max(all_confidence) < experiment.expand_radius_confidence_threshold:
        # We did not find the grain we needed close enough, open up the serach bounds
        # Check each orientation at the coordinate point
        for i in np.arange(n_oris):
            # We are close enough, test the orientation
            # Grab orientation information
            angles = all_angles[i]
            rMat_ss = all_rMat_ss[i]
            gvec_cs = all_gvec_cs[i]
            rMat_c = all_rMat_c[i]
            # Find intercept point of each g-vector on the detector
            det_xy = xfcapi.gvec_to_xy(gvec_cs, rD, rMat_ss, rMat_c, tD, tS, coord_to_test) # Convert angles to xy detector positions
            # Check detector positions and omega values to see if intensity exisits
            test_confidence = _quant_and_clip_confidence(det_xy, angles[:, 2], image_stack,
                                            base, inv_deltas, clip_vals, bsp, ome_edges)
            if (test_confidence - all_confidence[i]) > 0.05: # Harcode!
                all_confidence[i] = test_confidence


    # Find the index of the max confidence
    idx = np.where(all_confidence == np.max(all_confidence))[0][0] # Grab just the first instance if there is a tie

    # What is the hightest confidence orientation and what is its confidence
    exp_map = all_exp_maps[idx]
    confidence = all_confidence[idx]

    return exp_map, confidence, idx

#@profile
def _test_many_orientations_at_many_coordinates(experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no=0,start=0,stop=0):
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
        all_idx = np.zeros(n_coords,int)
        all_misorientation = np.zeros(n_coords)
        # Loop over the coordinates
        for i in np.arange(n_coords):
            coord_to_test = coordinates_to_test[i,:]
            if n_oris == 1:
                # There is only one orientation, don't check centroid position
                exp_map, confidence, dummy = _test_single_orientation_at_single_coordinate(experiment,image_stack,coord_to_test,orientation_data_to_test)
                idx = 0
            else:
                exp_map, confidence, idx = _test_many_orientations_at_single_coordinate(experiment,image_stack,coord_to_test,orientation_data_to_test)
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
        all_idx = np.zeros(n_coords,int)
        all_misorientation = np.zeros(n_coords)
        # Loop over the coordinates
        for i in np.arange(n_oris):
            orientation_data_to_test = [test_exp_maps[i], test_angles[i], test_rMat_ss[i], test_gvec_cs[i], test_rMat_c[i]]
            if n_coords == 1:
                # Check for centroid position
                if np.linalg.norm(coord_to_test - experiment.t_vec_s[start+i,:]) <= experiment.centroid_serach_radius:
                    # We are close enough, run the test
                    exp_maps, confidence, dummy = _test_single_orientation_at_single_coordinate(experiment,image_stack,coordinates_to_test,orientation_data_to_test)
                else:
                    # Not close enough, ignore and move on
                    exp_maps = test_exp_maps[i]
                    confidence = 0
                    dummy = 0
            else:
                # Create mask of which coords to test
                test_these_coords = np.linalg.norm(coordinates_to_test - experiment.t_vec_s[start+i,:],axis=1) <= experiment.centroid_serach_radius
                coordinates_to_test_full = coordinates_to_test
                coordinates_to_test = coordinates_to_test[test_these_coords]
                exp_maps, confidence = _test_single_orientation_at_many_coordinates(experiment,image_stack,coordinates_to_test,orientation_data_to_test)
                exp_maps_full = np.zeros([n_coords,3])
                confidence_full = np.zeros(n_coords)
                exp_maps_full[test_these_coords] = exp_maps
                confidence_full[test_these_coords] = confidence
                exp_maps = exp_maps_full
                confidence = confidence_full
                coordinates_to_test = coordinates_to_test_full
            # Replace any which are better than before
            to_replace = confidence > all_confidence
            all_exp_maps[to_replace,:] = exp_maps[to_replace]
            all_confidence[to_replace] = confidence[to_replace]
            all_idx[to_replace] = start + i
    
    # Refine if we need to, we have one orientation per coordinate point
    if refine_yes_no == 1:
        # Not sure if it is faster to pull the orientation info or compute it
        for i in np.arange(n_coords):
            coord_to_test = coordinates_to_test[i,:]
            single_orientation_data_to_test = _precompute_diffraction_data_of_single_orientation(experiment,all_exp_maps[i])
            all_exp_maps[i], all_confidence[i], all_misorientation[i] = _test_single_orientation_at_single_coordinate(experiment,image_stack,coord_to_test,single_orientation_data_to_test,refine_yes_no=1)

    return all_exp_maps, all_confidence, all_idx, all_misorientation, start, stop

#@profile
def _precompute_diffraction_data_of_single_orientation(experiment,exp_map):
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

#@profile
def _precompute_diffraction_data_of_many_orientations(experiment,exp_maps,start=0,stop=0):
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
# USER FACEING MUTI-PROCESSOR HANDLER FUNCTIONS
# ===============================================================================
#@profile
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
    # Start a timer
    t0 = timeit.default_timer()
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
            _precompute_diffraction_data_of_single_orientation(experiment,exp_maps_to_precompute)
    else:
        # Many orientations, how many cores are we dealing with?
        if ncpus == 1:
            # Many orientations, single CPU
            # Tell the user what we are doing
            print(f'Precomputing diffraction data for {n_oris} orientations on 1 CPU.')
            all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c, start, stop = \
                _precompute_diffraction_data_of_many_orientations(experiment,exp_maps_to_precompute)
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
                for vals1, vals2, vals3, vals4, vals5, start, stop in pool.imap_unordered(_precompute_diffraction_data_distributor,chunks):
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
    # How long did it take?
    t1 = timeit.default_timer()
    elapsed = t1-t0
    if elapsed < 60.0:
        print(f'Completed {n_oris} orientation precomputations in {np.round(elapsed,1)} seconds ({elapsed/n_oris} seconds per precomputation).')
    else:
        print(f'Completed {n_oris} orientation precomputations in {np.round(elapsed/60,1)} minutes ({elapsed/n_oris} seconds per precomputation).')

    # Handle arrays not being the correct size if we have only one orientation
    if len(np.shape(all_exp_maps)) == 1: all_exp_maps = np.expand_dims(all_exp_maps,0)
    if len(np.shape(all_rMat_c)) == 2: all_rMat_c = np.expand_dims(all_rMat_c,0)
    # Say we are done and return precomputed data
    print('Done precomputing orientation data.')
    return all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c

#@profile
def test_orientations_at_coordinates(experiment,controller,image_stack,orientation_data_to_test,coordinates_to_test,refine_yes_no=0,return_misorientation=0):
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
    # Shoot a warning to the user if they are running with refinment
    if refine_yes_no == 1:
        mis_amt = experiment.misorientation_bound_rad # This is the amount of misorientation allowed on one side of the original orientation
        spacing = experiment.misorientation_step_rad # This is the spacing between orientations
        ori_pts = np.arange(-mis_amt, (mis_amt+(spacing*0.999)),spacing) # Create a linup of the orientations to go on either side
        n_oris_refine = ori_pts.shape[0]**3
        print('Since you are refining ')

    # What senario do we have?
    if ncpus == 1 or (n_oris == 1 and n_coords == 1):
        # Single processor variant
        # Tell the user what we are doing
        print(f'Testing {n_oris} orientations at {n_coords} coordinate points on 1 CPU.')
        # Treat entire coordinate array as a single chunk and run
        all_exp_maps, all_confidence, all_idx, all_misorientation, start, stop = _test_many_orientations_at_many_coordinates(experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no=refine_yes_no,start=0,stop=n_coords)
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
            print(f'The {ncpus} CPUs will tackle {num_chunks} chunks with {chunk_size} coordinate points to test against the {n_oris} orientations.')
            # Initialize arrays to drop the exp_map and confidence
            all_exp_maps = np.zeros([n_coords,3])
            all_confidence = np.zeros(n_coords)
            all_idx = np.zeros(n_coords)
            all_misorientation = np.zeros(n_coords)
            # Unpack the orientation data
            # Package all inputs to the distributor function
            state = (starts,stops,experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            with multiprocessing_pool(ncpus,state) as pool:
                for vals1, vals2, vals3, vals4, start, stop in pool.imap_unordered(_test_many_orientations_at_many_coordinates_distributor,chunks):
                    # Grab the data as each CPU drops it
                    all_exp_maps[start:stop] = vals1
                    all_confidence[start:stop] = vals2
                    all_idx[start:stop] = vals3
                    all_misorientation[start:stop] = vals4
                    # Clean up
                    del vals1, vals2, vals3, vals4, start, stop
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
            print(f'The {ncpus} CPUs will tackle {num_chunks} chunks with {chunk_size} orientations to test at the {n_coords} coordinate points.')
            # Initialize arrays to drop the exp_map and confidence
            all_exp_maps = np.zeros([n_coords,3])
            all_confidence = np.zeros(n_coords)
            all_idx = np.zeros(n_coords)
            all_misorientation = np.zeros(n_coords)
            # Package all inputs to the distributor function
            state = (starts,stops,experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            chunk_num = 0
            with multiprocessing_pool(ncpus,state) as pool:
                for vals1, vals2, vals3, vals4, start, stop in pool.imap_unordered(_test_many_orientations_at_many_coordinates_distributor,chunks):
                    # Grab the data as each CPU drops it
                    # Replace any which are better than before
                    to_replace = vals2 > all_confidence
                    all_exp_maps[to_replace,:] = vals1[to_replace]
                    all_confidence[to_replace] = vals2[to_replace]
                    all_idx[to_replace] = vals3[to_replace]
                    all_misorientation[to_replace] = vals4[to_replace]
                    # Clean up
                    chunk_num = chunk_num + 1
                    del vals1, vals2, vals3, vals4, start, stop
            # Final cleanup
            pool.close()
            pool.join()

            # We may have coordinates which have not found thier grain if the serach radius was too small
            # Flag those we need to look at
            coords_to_retest = all_confidence < experiment.expand_radius_confidence_threshold
            n_coords_to_retest = np.sum(coords_to_retest)
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
            print(f'The {ncpus} CPUs will tackle {num_chunks} chunks with {chunk_size} orientations to re-test at the {n_coords_to_retest} coordinate points.')
            # Initialize arrays to drop the exp_map and confidence
            retest_exp_maps = np.zeros([n_coords_to_retest,3])
            retest_confidence = np.zeros(n_coords_to_retest)
            retest_idx = np.zeros(n_coords_to_retest)
            retest_misorientation = np.zeros(n_coords_to_retest)
            # Generate a working_experiment with a very large centroid_serach_radius
            working_experiment = experiment
            working_experiment.centroid_serach_radius = 1000 #mm
            # Package all inputs to the distributor function
            state = (starts,stops,working_experiment,image_stack,coordinates_to_test,orientation_data_to_test,refine_yes_no)
            # Start the multiprocessing loop
            set_multiprocessing_method(controller.multiprocessing_start_method)
            chunk_num = 0
            with multiprocessing_pool(ncpus,state) as pool:
                for vals1, vals2, vals3, vals4, start, stop in pool.imap_unordered(_test_many_orientations_at_many_coordinates_distributor,chunks):
                    # Grab the data as each CPU drops it
                    # Replace any which are better than before
                    to_replace = vals2 > retest_confidence
                    retest_exp_maps[to_replace,:] = vals1[to_replace]
                    retest_confidence[to_replace] = vals2[to_replace]
                    retest_idx[to_replace] = vals3[to_replace]
                    retest_misorientation[to_replace] = vals4[to_replace]
                    # Clean up
                    chunk_num = chunk_num + 1
                    del vals1, vals2, vals3, vals4, start, stop
            # Final cleanup
            pool.close()
            pool.join()
            # Make sure the orientations are better by enough
            much_beter = (retest_confidence - all_confidence[coords_to_retest]) > 0.05 # Hardcode!
            all_exp_maps[coords_to_retest[much_beter]] = retest_exp_maps[much_beter]
            all_confidence[coords_to_retest[much_beter]] = retest_confidence[much_beter]
            all_idx[coords_to_retest[much_beter]] = retest_idx[much_beter]
            all_misorientation[coords_to_retest[much_beter]] = retest_misorientation[much_beter]







    # How long did it take?
    t1 = timeit.default_timer()
    elapsed = t1-t0
    if elapsed < 60.0:
        print(f'Completed {n_oris*n_coords} orientation/coordinate tests in {np.round(elapsed,1)} seconds ({elapsed/n_oris/n_coords} seconds per test).')
    else:
        print(f'Completed {n_oris*n_coords} orientation/coordinate tests in {np.round(elapsed/60,1)} minutes ({elapsed/n_oris/n_coords} seconds per test).')

    if return_misorientation == 1:
        # Spit out the misorientation as well
        return all_exp_maps, all_confidence, all_idx.astype(int), all_misorientation
    else:
        return all_exp_maps, all_confidence, all_idx.astype(int)

def load_all_images(filenames,controller):
    """
        Goal: 
            
        Input:
            
        Output:

    """
    # Start a timer
    t0 = timeit.default_timer()
    # How many images to load
    n_imgs = len(filenames)
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Grab in formation about the files
    quick_image = skimage.io.imread(filenames[0])
    image_shape = np.shape(quick_image)
    image_dtype = quick_image.dtype
    # Single process or multi-thread?
    if ncpus == 1:
        # Just go ahead and load the images
        print(f'Loading {n_imgs} images with a single CPU.')
        raw_image_stack, start, stop = _load_images(filenames,image_shape,image_dtype,0,n_imgs)
    else:
        # Generate the blank image stack
        raw_image_stack = np.zeros([n_imgs,image_shape[0],image_shape[1]],image_dtype)
        # Define the chunk size
        chunk_size = controller.get_chunk_size()
        if chunk_size == -1:
            chunk_size = int(np.ceil(n_imgs/ncpus))
        # Create chunking
        num_chunks = int(np.ceil(n_imgs/chunk_size))
        chunks = np.arange(num_chunks)
        starts = np.zeros(num_chunks,dtype=int)
        stops = np.zeros(num_chunks,dtype=int)
        for i in np.arange(num_chunks):
            starts[i] = i*chunk_size
            stops[i] = i*chunk_size + chunk_size
            if stops[i] >= n_imgs:
                stops[i] = n_imgs
        print(f'Loading {n_imgs} images with {ncpus} CPUs and {num_chunks} chunks of size {chunk_size}.')
        # Package all inputs to the distributor function
        state = (starts,stops,filenames,image_shape,image_dtype)
        # Start the multiprocessing loop
        set_multiprocessing_method(controller.multiprocessing_start_method)
        with multiprocessing_pool(ncpus,state) as pool:
            for vals1, start, stop in pool.imap_unordered(_load_images_distributor,chunks):
                # Grab the data as each CPU drops it
                raw_image_stack[start:stop,:,:] = vals1
                # Clean up
                del vals1, start, stop

    # How long did it take?
    t1 = timeit.default_timer()
    elapsed = t1-t0
    if elapsed < 60.0:
        print(f'Loaded {n_imgs} images in {np.round(elapsed,1)} seconds ({elapsed/n_imgs} seconds per image).')
    else:
        print(f'Loaded {n_imgs} images in {np.round(elapsed/60,1)} minutes ({elapsed/n_imgs} seconds per image).')

    return raw_image_stack

def remove_median_darkfields(raw_image_stack,controller,configuration):
    """
        Goal: 
            
        Input:
            
        Output:

    """
    # Start a timer
    t0 = timeit.default_timer()
    # How many slices are we dealing with
    n_slices = np.shape(raw_image_stack)[2]
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Pull configuration data
    median_size_through_omega = configuration.images.processing.omega_kernel_size
    global_threshold = configuration.images.processing.threshold
    # Single process or multi-thread?
    if ncpus == 1:
        # Just go ahead and load the images
        print(f'Subtracting dynamic darkfield from {n_slices} slices with a single CPU.')
        cleaned_image_stack, start, stop = _remove_dynamic_median(raw_image_stack,median_size_through_omega,0,n_slices)
    else:
        # Generate the blank image stack
        cleaned_image_stack = np.zeros(np.shape(raw_image_stack),raw_image_stack.dtype)
        # Define the chunk size
        chunk_size = controller.get_chunk_size()
        if chunk_size == -1:
            chunk_size = int(np.ceil(n_slices/ncpus))
        # Create chunking
        num_chunks = int(np.ceil(n_slices/chunk_size))
        chunks = np.arange(num_chunks)
        starts = np.zeros(num_chunks,dtype=int)
        stops = np.zeros(num_chunks,dtype=int)
        for i in np.arange(num_chunks):
            starts[i] = i*chunk_size
            stops[i] = i*chunk_size + chunk_size
            if stops[i] >= n_slices:
                stops[i] = n_slices
        print(f'Subtracting dynamic darkfield from {n_slices} slices with {ncpus} CPUs and {num_chunks} chunks of size {chunk_size}.')
        # Package all inputs to the distributor function
        state = (starts,stops,raw_image_stack,median_size_through_omega)
        # Start the multiprocessing loop
        set_multiprocessing_method(controller.multiprocessing_start_method)
        with multiprocessing_pool(ncpus,state) as pool:
            for vals1, start, stop in pool.imap_unordered(_remove_median_darkfield_distributor,chunks):
                # Grab the data as each CPU drops it
                cleaned_image_stack[:,:,start:stop] = vals1
                # Clean up
                del vals1, start, stop

    # Remove the global threshold
    print('Dynamic darkfield generated and subtracted.')
    print(f'Subtracting global threshold of {global_threshold}.')
    mask = cleaned_image_stack<=global_threshold
    cleaned_image_stack[mask] = 0
    cleaned_image_stack[~mask] = cleaned_image_stack[~mask] - global_threshold

    # How long did it take?
    t1 = timeit.default_timer()
    elapsed = t1-t0
    if elapsed < 60.0:
        print(f'Subtracted dynamic darkfield and global threshold from {n_slices} slices in {np.round(elapsed,1)} seconds ({elapsed/n_slices} seconds per slice).')
    else:
        print(f'Subtracted dynamic darkfield and global threshold from {n_slices} slices in {np.round(elapsed/60,1)} minutes ({elapsed/n_slices} seconds per slice).')

    return cleaned_image_stack

def filter_and_binarize_images(cleaned_image_stack,controller,filter_parameters):
    """
        Goal: 
            
        Input:
            
        Output:

    """
    # Start a timer
    t0 = timeit.default_timer()
    # How many slices are we dealing with
    n_images = np.shape(cleaned_image_stack)[0]
    # How many CPUs?
    ncpus = controller.get_process_count()
    # Create some text
    cleanup_text = ['Gaussian Cleanup','Errosion/Dilation Cleanup','Non-Local Means Cleanup']
    small_objects_text = ['no Filtering of Small Objects','Filtering of Small Objects']
    # Filter Parameters information
        # filter_parameters[0] - if 1, remove small objects, if 0 do nothing
        # filter_parameters[1] - what size of small objects to remove
        # filter_parameters[2] - which cleanup to use
        # filter_parameters[3:] - cleanup parameters
    # Single process or multi-thread?
    if ncpus == 1:
        # Just go ahead and load the images
        print(f'Cleaning {n_images} images with {cleanup_text[filter_parameters[2]]} and {small_objects_text[filter_parameters[0]]} on a single CPU.')
        binarized_image_stack, start, stop = _filter_and_binarize_image(cleaned_image_stack,filter_parameters,0,n_images)
    else:
        # Generate the blank image stack
        binarized_image_stack = np.zeros(np.shape(cleaned_image_stack),bool)
        # Define the chunk size
        chunk_size = controller.get_chunk_size()
        if chunk_size == -1:
            chunk_size = int(np.ceil(n_images/ncpus))
        # Create chunking
        num_chunks = int(np.ceil(n_images/chunk_size))
        chunks = np.arange(num_chunks)
        starts = np.zeros(num_chunks,dtype=int)
        stops = np.zeros(num_chunks,dtype=int)
        for i in np.arange(num_chunks):
            starts[i] = i*chunk_size
            stops[i] = i*chunk_size + chunk_size
            if stops[i] >= n_images:
                stops[i] = n_images
        print(f'Cleaning {n_images} images with {cleanup_text[filter_parameters[2]]} and {small_objects_text[filter_parameters[0]]}\n\
              with {ncpus} CPUs and {num_chunks} chunks of size {chunk_size}')
        # Package all inputs to the distributor function
        state = (starts,stops,cleaned_image_stack,filter_parameters)
        # Start the multiprocessing loop
        set_multiprocessing_method(controller.multiprocessing_start_method)
        with multiprocessing_pool(ncpus,state) as pool:
            for vals1, start, stop in pool.imap_unordered(_filter_and_binarize_images_distributor,chunks):
                # Grab the data as each CPU drops it
                binarized_image_stack[start:stop,:,:] = vals1
                # Clean up
                del vals1, start, stop

    # How long did it take?
    t1 = timeit.default_timer()
    elapsed = t1-t0
    if elapsed < 60.0:
        print(f'Filtered {n_images} images in {np.round(elapsed,1)} seconds ({elapsed/n_images} seconds per image).')
    else:
        print(f'Filtered {n_images} images in {np.round(elapsed/60,1)} minutes ({elapsed/n_images} seconds per image).')

    return binarized_image_stack

# %% ============================================================================
# MULTI-PROCESSOR DISTRIBUTOR FUNCTIONS
# ===============================================================================
#@profile
def _precompute_diffraction_data_distributor(chunk):
    # Where are we pulling data from within the lists?
    starts = _mp_state[0]
    stops = _mp_state[1]
    return _precompute_diffraction_data_of_many_orientations(*_mp_state[2:], start=starts[chunk], stop=stops[chunk])

#@profile
def _test_many_orientations_at_many_coordinates_distributor(chunk):
    # Where are we pulling data from within the lists?
    starts = _mp_state[0]
    stops = _mp_state[1]
    return _test_many_orientations_at_many_coordinates(*_mp_state[2:], start=starts[chunk], stop=stops[chunk])

def _load_images_distributor(chunk):
    # Where are we pulling data from within the lists?
    starts = _mp_state[0]
    stops = _mp_state[1]
    return _load_images(*_mp_state[2:], start=starts[chunk], stop=stops[chunk])

def _remove_median_darkfield_distributor(chunk):
    # Where are we pulling data from within the lists?
    starts = _mp_state[0]
    stops = _mp_state[1]
    return _remove_dynamic_median(*_mp_state[2:], start=starts[chunk], stop=stops[chunk])

def _filter_and_binarize_images_distributor(chunk):    
    # Where are we pulling data from within the lists?
    starts = _mp_state[0]
    stops = _mp_state[1]
    return _filter_and_binarize_image(*_mp_state[2:], start=starts[chunk], stop=stops[chunk])


# %% ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ===============================================================================

# Dynamic median function
def _remove_dynamic_median(raw_image_stack,median_size_through_omega=25,start=0,stop=0):
    # How many slices?
    n_slices = stop - start
    # Create a new variable with just want we need so we are not pinning raw_image_stack so much
    cleaned_slices = np.copy(raw_image_stack[:,:,start:stop])
    for slice in np.arange(n_slices):
        # Grab a slice at a specific through the raw image stack (does not really matter which axis)
        raw_slice = cleaned_slices[:, :, slice]
        # Calculate a moving median along omega at each 
        raw_slice_dark = scipy.ndimage.median_filter(raw_slice, size=[median_size_through_omega, 1])
        # Update the slice and handle negatives
        new_slice = raw_slice - raw_slice_dark
        new_slice[raw_slice_dark>=raw_slice] = 0
        # Place in slices
        cleaned_slices[:,:,slice] = new_slice
    # Return the things
    return cleaned_slices, start, stop

# Image binarization
def _filter_and_binarize_image(cleaned_image_stack,filter_parameters,start,stop):
    # Grab a chunk of the image stack
    working_image_stack = np.copy(cleaned_image_stack[start:stop,:,:])
    # Create a binarized image stack
    binarized_image_stack = np.zeros(np.shape(working_image_stack),bool)
    # How many images?
    n_images = np.shape(working_image_stack)[0]
    # Filter Parameters information
    # filter_parameters[0] - if 1, remove small objects, if 0 do nothing
    # filter_parameters[1] - what size of small objects to remove
    # filter_parameters[2] - which cleanup to use
    # filter_parameters[3:] - cleanup parameters
    # What filter are we using?
    which_filter = filter_parameters[2]
    if which_filter == 0:
        # Gaussian cleanup
        # Grab parameters
        [sigma,threshold] = filter_parameters[3:]
        for i in np.arange(n_images):
            # Grab the image
            img = working_image_stack[i, :, :]
            # Filter
            img = skimage.filters.gaussian(img, sigma=sigma,preserve_range=True)
            # Threshold and put into binary stack
            binarized_image_stack[i,:,:] = img > threshold
    elif which_filter == 1:
        # Errosion/dilation cleanup
        # Grab parameters
        [errosions,dilations,threshold] = filter_parameters[3:]
        for i in np.arange(n_images):
            # Grab the image
            img = working_image_stack[i, :, :]
            # Binarize
            img_binary = img > threshold
            # Errode then dilate
            img_binary = scipy.ndimage.binary_erosion(img_binary, iterations=errosions)
            img_binary = scipy.ndimage.binary_dilation(img_binary, iterations=dilations)
            # Toss into binary stack
            binarized_image_stack[i,:,:] = img_binary
    elif which_filter == 2:
        # Non-local means cleanup
        # Grab parameters
        [patch_size,patch_distance,threshold] = filter_parameters[3:]
        for i in np.arange(n_images):
            # Grab the image
            img = working_image_stack[i, :, :]
            # Estimage the per-slice sigma
            s_est = skimage.restoration.estimate_sigma(img)
            # Run non-local_means
            img = skimage.restoration.denoise_nl_means(img, sigma=s_est, h=0.8 * s_est, patch_size=patch_size, patch_distance=patch_distance, preserve_range = True)
            # Binarize and throw into new stack
            binarized_image_stack[i,:,:] = img > threshold

    # Are we removing small features?
    remove_small_features = filter_parameters[0]
    if remove_small_features == 1:
        for i in np.arange(n_images):
            binarized_image_stack[i, :, :] = skimage.morphology.remove_small_objects(binarized_image_stack[i, :, :],filter_parameters[1],connectivity=1)

    # Return the things
    return binarized_image_stack, start, stop

# Dilation through omega
def dilate_image_stack(binarized_image_stack,dilate_omega):
    if dilate_omega > 0:
        # Start a timer
        t0 = timeit.default_timer()
        # Tell the user
        print('Dilating image stack.')
        dilated_image_stack = scipy.ndimage.binary_dilation(binarized_image_stack, iterations=dilate_omega)
        # How long did it take?
        t1 = timeit.default_timer()
        elapsed = t1-t0
        if elapsed < 60.0:
            print(f'Dilated image stack in {np.round(elapsed,1)} seconds.')
        else:
            print(f'Dilated image stack in {np.round(elapsed/60,1)} minutes.')

        # Return the thing
        return dilated_image_stack
    else:
        print('No dilation asked for, returning binarized image stack.')
        return binarized_image_stack


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


# We don't know why these need to be here...
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

def build_controller(configuration):
    # builds the controller to use based on the args
    ncpus = configuration.multiprocessing.num_cpus
    chunk_size = configuration.multiprocessing.chunk_size
    check = configuration.multiprocessing.check
    generate = configuration.multiprocessing.generate
    limit = configuration.multiprocessing.limit
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


