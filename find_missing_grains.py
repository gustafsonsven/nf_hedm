#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
contributing authors: dcp5303, ken38, seg246
"""
"""
NOTES:
    - The reference frame used in this script is the HEXRD frame
        - X points right if facing towards the x-ray detector (downstream)
        - Y points up (against gravity)
        - Z points upstream (away from the detector)
        - X,Y,Z center is where the beam intercepts the rotation axis

    - The standard reconstruction must be done ahead of time WITH A MASK
    - Don't use a really small voxel size - 0.005 is fine (unless you have very 
        small grains and very good data)
    - The goal of this script is to find grains within the volume that FF is 
        struggling to find - FIND AS MANY GRAINS AS YOU CAN WITH FF
        The more low confidence you have in your grain map the longer this will take
    - Even with relativly few low confidence regions, this script will still take 
        several hours to complete.  
    - This script will output a grains.out file with the original grains (those found
        in the input reconstruction) and the new, found grains appended to the list.
    - If you decide to save and re-run it will use all the new grains (and the old)
    - Grains found in NF should be pushed back to FF to attempt a fit - FF will provide
        a more refined orientation.  

"""
# %% Imports - NO CHANGES NEEDED
#==============================================================================
# General Imports
import numpy as np
import multiprocessing as mp
import os

# Hexrd imports
from hexrd.transforms import xfcapi
import nfutil as nfutil
import timeit
from hexrd import rotations
from hexrd import constants
from hexrd import instrument

# Matplotlib
# This is to allow interactivity of inline plots in your gui
# the import ipywidgets as widgets line is not needed - however, you do need to run a pip install ipywidgets
# the import ipympl line is not needed - however, you do need to run a pip install ipympl
#import ipywidgets as widgets
#import ipympl 
import matplotlib
# The next line is formatted correctly, no matter what your IDE says
#%matplotlib widget
#%matplotlib inline
import matplotlib.pyplot as plt

#==============================================================================
# %% Paths - CAN BE EDITED
#==============================================================================
# Working directory
# Should be of the form: '/nfs/chess/aux/reduced_data/cycles/[cycle ID]/[beamline]/BTR/sample'
main_dir = '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/shanks-3731-a/ti-13-exsitu/nf/2'
# Detector file (retiga, manta,...)
det_file = main_dir + '/retiga.yml'

#==============================================================================
# %% Materials File - CAN BE EDITED
#==============================================================================
# Materials file - from HEXRDGUI (MAKE SURE YOUR HKLS ARE DEFINED CORRECTLY FOR YOUR MATERIAL)
mat_file = main_dir + '/materials.h5'

# Material name in materials.h5 file from HEXRGUI
mat_name = 'ti7al'
max_tth = None  # degrees, if None is input max tth will be set by the geometry
# NOTE: Again, make sure the HKLs are set correctly in the materials file that you loaded
    # If you set max_tth to 20 degrees, but you only have HKLs out to 15 degrees selected
    # then you will only use the selected HKLs out to 15 degrees

# Point group number - Check out the nfutil for more - (Ti-alpha (Hexagonal) = 27) (Nickel (FCC) = 32)
pt_gr_num = 27

#==============================================================================
# %% Output information - CAN BE EDITED
#==============================================================================
# Where do you want to drop any output files
output_dir = main_dir + '/output/'
# What was the stem you used during image creation via nf_multithreaded_image_processing?
image_stem = 'ti-13-exsitu_layer_2'
# How do you want your outputs to be named?
output_stem = 'ti-13-exsitu_layer_2_with_missing_grains'

#==============================================================================
# %% Grains.out File - CAN BE EDITED
#==============================================================================
# Location of grains.out file from far field
# This actually does not get used directly, but it does need to point to a grain.out
grain_out_file = '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/shanks-3731-a/ti-13-exsitu/nf/merged_2023_09_13.out'

#==============================================================================
# %% Orginal Reconstruction - CAN BE EDITED
#==============================================================================
# Load in the output from the basic search
reconstructed_data_path = '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/shanks-3731-a/ti-13-exsitu/nf/2/output/ti-13-exsitu_layer_2_grain_map_data.npz'
starting_reconstruction = np.load(reconstructed_data_path)

# Beam stop details
beam_stop_y_cen = 0.0  # mm, measured from the origin of the detector paramters
beam_stop_width = 0.2  # mm, width of the beam stop vertically

#==============================================================================
# %% Grain Serach details - CAN BE EDITED
#==============================================================================
# Multiprocessing
ncpus = 128 #mp.cpu_count() - 10 # use as many CPUs as are available
chunk_size = -1 # -1 will use np.ceil(num_operations/ncpus)

# Orientation grid spacing?
# The grid spacing must be sufficently full to populate the fundamental region, I suggest 1.0 deg
ori_grid_spacing = 1.0 # Spacing between orientations used in serach - in degrees - 1.0 is good
misorientation_bnd = 0.6  # Refinement bounds on found orientation - in degrees - about half of your ori_grid_spacing
misorientation_spacing = 0.1  # Step size for orientation refinement - in degrees - orientation resolution of NF is about 0.1 deg

# What is our confidence threshold to be applied to the original reconstruction (check this out ahead of time)
confidence_threshold = 0.6 # Anything less than this will be looked at for new grains
low_confidence_sparsing = 0 # 0 will take all voxels, 1 will take about half, 2 about a quarter - uses a sparse regular grid - Up to you, 0 has the best chance of finding all missing grains
errode_free_surface = 1 # NF is poor at the surface already, removes the free surface of your tomo mask from the search space - 1 is suggested

# Define a cutoff value for when to switch to a brute force
coord_cutoff_scale = 0.15 # Once we only have coord_cutoff_scale*100 % voxels left to look at we transition to brute search of each voxel - its quicker - 0.15 is good
iter_cutoff = 10 # If we don't find a grain after iter_cutoff iterations we break and go straight to searching all voxels - 10 is good

# Re-rerun and save reconstruction
re_run_and_save = 1 # If 1 this will re-run the full reconstruction with all grains and save an npz + paraview file

#==============================================================================
# %% Load the Images - NO CHANGES NEEDED
#==============================================================================
print('Loading Images.')
# Load the cleaned image stack from the first script
image_stack = np.load(output_dir + os.sep + image_stem + '_binarized_images.npy')
print('Images loaded.')

# Load the omega edges - first value is the starting ome position of first image's slew, last value is the end position of the final image's slew
omega_edges_deg = np.load(output_dir + os.sep + image_stem + '_omega_edges_deg.npy')

# Shift in omega positive or negative by X number of images
num_img_to_shift = -2 # Postive moves positive omega, negative moves negative omega, must be integer 
if num_img_to_shift > 0:
    # Moving positive omega so first image is not at zero, but further along
    # Using the mean omega step size - change if you need to
    omega_edges_deg = omega_edges_deg + num_img_to_shift*np.mean(np.gradient(omega_edges_deg))
elif num_img_to_shift < 0:
    # For whatever reason the multiprocessor does not like negative numbers, trim the stack
    image_stack = image_stack[np.abs(num_img_to_shift):,:,:]
    omega_edges_deg = omega_edges_deg[:num_img_to_shift]


#==============================================================================
# %% Load the Experiment - NO CHANGES NEEDED
#==============================================================================
# Make beamstop
beam_stop_parms = np.array([beam_stop_y_cen, beam_stop_width])

# These are here so that they are not changed
comp_thresh = 0
chi2_thresh = 1
check = None
limit = None
generate = None
# Generate the experiment
experiment, nf_to_ff_id_map = \
    nfutil.gen_trial_exp_data(grain_out_file,det_file,mat_file, mat_name, max_tth, 
                              comp_thresh, chi2_thresh,omega_edges_deg, beam_stop_parms, 
                              misorientation_bnd=0.0, misorientation_spacing=0.25)

#==============================================================================
# %% GENERATE ORIENTATIONS TO TEST - NO CHANGES NEEDED
#==============================================================================
# Create a regular orientation grid
quats = np.transpose(nfutil.uniform_fundamental_zone_sampling(pt_gr_num,average_angular_spacing_in_deg=ori_grid_spacing))
n_grains = quats.shape[1]

# Convert to rotation matrices and exponential maps
exp_maps = np.zeros([quats.shape[1],3])
for i in range(0,quats.shape[1]):
    phi = 2*np.arccos(quats[0,i])
    n = xfcapi.unitRowVector(quats[1:,i])
    exp_maps[i,:] = phi*n

# Define multiprocessing details
controller = nfutil.build_controller(ncpus=ncpus, chunk_size=chunk_size, check=check, 
                                        generate=generate, limit=limit)
multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

# Precompute all relevant orientation data for each orientaiton
print(f'Precomputing orientation information.')
# This can get very RAM heavy
all_angles, all_rMat_ss, all_gvec_cs = \
    nfutil.precompute_orientation_information_main_loop(exp_maps,experiment,controller,multiprocessing_start_method)
print(f'Done precomputing orientation information.')

#==============================================================================
# %% Generate Test Coordinates - NO CHANGES NEEDED
#==============================================================================
# Initialize some arrays
original_confidence = starting_reconstruction['confidence_map']
original_exp_maps = starting_reconstruction['ori_list']
new_exp_maps = original_exp_maps
mask = starting_reconstruction['tomo_mask']
voxels_to_check = original_confidence<confidence_threshold
voxels_to_check[mask == 0] = 0

# Grab the sparsest array of the low confidence points
test_coordinates, ids = \
    nfutil.generate_low_confidence_test_coordinates(starting_reconstruction,confidence_threshold,
                                                    how_sparse=low_confidence_sparsing,errode_free_surface=errode_free_surface)

# Print a warning
print(f"We will be testing {np.shape(test_coordinates)[0]} coordinates against {n_grains} orientations.")
# All depends on how much you want to wait - I don't suggest more than a couple thousand coordinate points
# Check your reconstruction ahead of time, if you are missing more than 5-10% of your reconstruction 
    # then take a look at your FF indexing and try to improve it

#==============================================================================
# %% GRAIN SEARCH LOOP - NO CHANGES NEEDED
#==============================================================================
# Initialize
new_grains = 0
t0 = timeit.default_timer() # Start a timer
print(f'Searching {np.shape(test_coordinates)[0]} coordinates.')
# Define the cutoff value for when to switch to a brute force
coord_cutoff = np.shape(test_coordinates)[0] * coord_cutoff_scale
# Define a counter of not finding a grain 
no_grain_here_count = 0 # If we hit this too many times in a row, we probably don't have large grains left
# Start while loop
while np.shape(test_coordinates)[0] > coord_cutoff: 
    # Initialize
    t1 = timeit.default_timer() # Start a timer
    print('-------------------------------------------------')
    print('Searching for a grain.')
    print('-------------------------------------------------')

    # Grab a test coordinate
    # Using a random coordinate to avoid sampling the edges before filling in middle holes
    idx = int(np.floor(np.random.uniform(low = 0, high = np.shape(test_coordinates)[0])))
    test_coord = test_coordinates[idx,:]
    id = ids[idx]

    # Test a single coordinate and refine orientation
    refined_exp_map,refined_conf = nfutil.test_single_coordinate_main_loop(image_stack, experiment, test_coord, 
                                                                exp_maps, all_angles, all_rMat_ss, all_gvec_cs, 
                                                                misorientation_bnd, misorientation_spacing
                                                                ,controller,multiprocessing_start_method)
    if refined_exp_map.ndim == 2:
        refined_exp_map = refined_exp_map[0,:]
    else:
        refined_exp_map = refined_exp_map

    print(f'Orientaiton determined at this voxel with {np.round(refined_conf*100)}% confidence.')
    print(f"Took {np.round(timeit.default_timer() - t1)} seconds to search for a grain at this voxel.")
    # Check and see if we want to look elsewhere in the reconstruction for this orientation
    if refined_conf < confidence_threshold:
        print('No grain found at this voxel, moving to the next.')
        # Reset our test_coordinates
        test_coordinates = test_coordinates[ids != id]
        ids = ids[ids != id]
        no_grain_here_count = no_grain_here_count + 1
        print(f'We have not found a grain for {no_grain_here_count} iterations, if we hit {iter_cutoff} we will break.')
    else:
        # Nice, we found one!
        new_grains = new_grains + 1
        print(f'Found a grain!  That makes {new_grains} so far.')
        print(f'Searching the other low confidence voxels to see if this orientation is anywhere else.')
        # Add the orientaiton to the stack of orientaitons
        new_exp_maps = np.vstack([new_exp_maps,refined_exp_map])
        # Let's see if this orientation is anywhere else
        # Push new information to the experiment
        experiment.n_grains = 1
        experiment.rMat_c = rotations.rotMatOfExpMap(refined_exp_map.T)
        experiment.exp_maps = refined_exp_map
        raw_confidence = nfutil.test_orientations(image_stack, experiment, test_coordinates,
                                                controller,multiprocessing_start_method)
        raw_confidence = raw_confidence.squeeze()
        good_ids = ids[raw_confidence > confidence_threshold]
        print(f'This orientation was found at {np.sum(raw_confidence > confidence_threshold)} total voxels.')
        # Reset our test_coordinates
        test_coordinates = test_coordinates[raw_confidence < confidence_threshold]
        ids = ids[raw_confidence < confidence_threshold]

        print('Saving the current grains.out.')
        # Save the grains.out in case you want it before the script is done
        gw = instrument.GrainDataWriter(
            os.path.join(output_dir, output_stem+'.out')
        )
        for gid, ori in enumerate(new_exp_maps):
            grain_params = np.hstack([ori, constants.zeros_3, constants.identity_6x1])
            gw.dump_grain(gid, 1., 0., grain_params)
        gw.close()

        no_grain_here_count = 0
    
    if no_grain_here_count == iter_cutoff:
        # We likely don't have any large grains left - let's brute force it
        break

    print(f'We have {np.shape(test_coordinates)[0]} low confidence coordinates left to test.')

    # Time check
    print(f"Took {np.round((timeit.default_timer() - t0)/60.)} minutes so far.")
print(f'Found {new_grains} with random serach.  Wrapping up last chunk of coordinates with a brute search and refinement.')
#==============================================================================
# %% Brute force the rest - NO CHANGES NEEDED
#==============================================================================
# At this point, it is statistically likely that we found most of the grains
# The startup and shutdown cost of the multiprocessing is not time cheap so we will 
    # go ahead and brute force the rest of the coordinates in one go to check for any
    # remaining grains
experiment.n_grains = np.shape(exp_maps)[0]
experiment.rMat_c = rotations.rotMatOfExpMap(exp_maps.T)
experiment.exp_maps = exp_maps
print(f'Starting serach with {experiment.n_grains} orientations on {np.shape(test_coordinates)[0]} spatial coordinates.')
raw_confidence = nfutil.test_orientations(image_stack, experiment, test_coordinates,
                                        controller,multiprocessing_start_method)
print(f'Search done, entering refinement on orientations with greater than {confidence_threshold} confidence.')
print(f"Took {np.round((timeit.default_timer() - t0)/60.)} minutes so far.")
#==============================================================================
# %% Unpack and Refine - NO CHANGES NEEDED
#==============================================================================
# Find the best quaternion
best_exp_maps = np.zeros([test_coordinates.shape[0],3])
best_quats = np.zeros([4,test_coordinates.shape[0]])
best_conf = np.zeros([1,test_coordinates.shape[0]])
for i in range(0,raw_confidence.shape[1]):
    where = np.where(raw_confidence[:,i] == np.max(raw_confidence[:,i]))
    best_exp_maps[i,:] = exp_maps[where[0][0],:]
    best_quats[:,i] = quats[:,where[0][0]]
    best_conf[0,i] = np.max(raw_confidence[:,i])
    print(f'Refining Point {i} of {raw_confidence.shape[1]}')
    exp_map = exp_maps[where[0][0],:]
    test_crd = test_coordinates[i,:]
    new_exp_map, new_conf = nfutil.refine_single_coordinate(image_stack, experiment, test_crd, exp_map, misorientation_bnd, misorientation_spacing)
    if new_exp_map.ndim == 2:
        best_exp_maps[i,:] = new_exp_map[0,:]
        best_quats[:,i] = rotations.quatOfExpMap(new_exp_map[0,:])
        best_conf[0,i] = new_conf
    else:
        best_exp_maps[i,:] = new_exp_map
        best_quats[:,i] = rotations.quatOfExpMap(new_exp_map)
        best_conf[0,i] = new_conf
print(f"Took {np.round((timeit.default_timer() - t0)/60.)} minutes so far.")
#==============================================================================
# %% Check for similar orientations - NO CHANGES NEEDED
#==============================================================================
print('Checking for similar orientations.')
# Initialize
idx = best_conf>confidence_threshold
working_quats = best_quats[:,idx.squeeze()]
final_quats = np.zeros(np.shape(working_quats))
count = 0
# Run through to test misorientation
while working_quats is not None:
    if np.shape(working_quats)[1] == 0:
        working_quats = None
    else:
        # Check misorientation
        grain_quats = np.atleast_2d(working_quats[:,0]).T
        test_quats = np.atleast_2d(working_quats)
        if np.shape(test_quats)[0] == 1:
            [misorientations, a] = rotations.misorientation(grain_quats,test_quats.T)
        else:
            [misorientations, a] = rotations.misorientation(grain_quats,test_quats)
        # Which are the same
        idx_to_merge = misorientations < np.radians(0.1)
        # Remove them and add a single orientation to the list
        final_quats[:,count] = working_quats[:,0]
        working_quats = np.delete(working_quats,idx_to_merge,1)
        count = count + 1

# Trim the list
final_quats = final_quats[:,:count-1]
# Convert to exp maps
final_exp_maps = rotations.expMapOfQuat(final_quats)
print(f'Found {count} additional grains during brute force search.')

#==============================================================================
# %% Save new grains.out - NO CHANGES NEEDED
#==============================================================================
print('Saving a final grains.out.')
final_exp_maps = np.vstack([new_exp_maps,np.transpose(final_exp_maps)])
# Get original exp_maps
gw = instrument.GrainDataWriter(
    os.path.join(output_dir, output_stem+'.out')
)
for gid, ori in enumerate(final_exp_maps):
    grain_params = np.hstack([ori, constants.zeros_3, constants.identity_6x1])
    gw.dump_grain(gid, 1., 0., grain_params)
gw.close()
print('Done.')

#==============================================================================
# %% Re-Run Reconstruction and Save - NO CHANGES NEEDED
#==============================================================================
if re_run_and_save == 1:
    # Generate the experiment
    grain_out_file = os.path.join(output_dir, output_stem+'.out')
    experiment, nf_to_ff_id_map = nfutil.gen_trial_exp_data(grain_out_file, det_file,
                                                            mat_file, mat_name, max_tth,
                                                            comp_thresh, chi2_thresh,
                                                            omega_edges_deg,beam_stop_parms,misorientation_bnd=0.0,
                                                            misorientation_spacing=0.25)
    
    # Grab the mask and coordinates
    Xs = starting_reconstruction['Xs']
    Ys = starting_reconstruction['Ys']
    Zs = starting_reconstruction['Zs']
    test_crds_full = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    to_use = np.where(mask.flatten())[0]
    test_crds = test_crds_full[to_use, :]

    # Run the reconstruction
    raw_confidence = nfutil.test_orientations(image_stack, experiment, test_crds,
                                        controller,multiprocessing_start_method)
    raw_confidence_full = np.zeros([len(experiment.exp_maps), len(test_crds_full)])
    for ii in np.arange(raw_confidence_full.shape[0]):
        raw_confidence_full[ii, to_use] = raw_confidence[ii, :]
    
    # Process the confidence array
    grain_map, confidence_map = nfutil.process_raw_confidence(raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map)
    # Save and NPZ
    nfutil.save_nf_data(output_dir, output_stem, grain_map, confidence_map,
                    Xs, Ys, Zs, experiment.exp_maps, tomo_mask=mask, id_remap=nf_to_ff_id_map,
                    save_type=['npz']) # Can be npz or hdf5
    # Save a H5 for paraview
    nfutil.save_nf_data_for_paraview(output_dir,output_stem,grain_map,confidence_map,Xs,Ys,Zs,
                                experiment.exp_maps,experiment.mat[mat_name], tomo_mask=mask,
                                id_remap=nf_to_ff_id_map)




