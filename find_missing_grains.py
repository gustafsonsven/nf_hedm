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

    - Note that HEXRD works with grain orientations which are defined from CRYSTAL TO SAMPLE 
        specifically as v_samp = R_cry_to_samp * v_cry
    - Here is a description pulled from the xf.py script within HEXRD.  (COB is change of basis). 
    
        gVec_c : numpy.ndarray
            (3, n) array of n reciprocal lattice vectors in the CRYSTAL FRAME.
        rMat_s : numpy.ndarray
            (3, 3) array, the COB taking SAMPLE FRAME components to LAB FRAME. 
        rMat_c : numpy.ndarray
            (3, 3) array, the COB taking CRYSTAL FRAME components to SAMPLE FRAME.
    
        # form unit reciprocal lattice vectors in lab frame (w/o translation)
        gVec_l = np.dot(rMat_s, np.dot(rMat_c, unitVector(gVec_c)))
    
    The line above is the important one.  It is the coordinate transformation of the g vector 
        from the crystal frame to the lab frame.  rMat_c is the orientation matrix that we have 
        outputted into the grain.out files from HEXRD (though it is expressed in axis angle form).  
        Note that rMat_c is defined from the crystal to the sample frame, but more specifically it transforms a 
        vector in the crystal frame into the sample frame as: gVec_s = np.dot(rMat_c,gVec_c).  Note that 
        np.dot in python is the same as rMat_c*gVec_c in matlab (a pre-multiplication).  Recall 
        of course that gVec_c here is a column vector (tall – 3x1 in both python and matlab) as is gVec_s.

"""
# %% ==========================================================================
# IMPORTS - DO NOT CHANGE
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
# The next lines are formatted correctly, no matter what your IDE says
# For inline, interactive plots (if you use these, make sure to run a plt.close() to prevent crashing)
%matplotlib widget
# For inline, non-interactive plots
# %matplotlib inline
# For pop out, interactive plots (cannot be used with an SSH tunnel)
# %matplotlib qt
import matplotlib.pyplot as plt

# %% ==============================================================================
# FILES TO LOAD -CAN BE EDITED
# ==============================================================================
configuration_filename = '/nfs/chess/user/seg246/software/development/nf_config.yml'
# %% ==========================================================================
# UNIQUE USER INFORMATION - CAN BE EDITED
# =============================================================================
# Where is the original grain map?
reconstructed_data_path = '/your/path/here/sample_reconstruction.npz'

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
iter_cutoff = 10 # If we don't find a grain after iter_cutoff iterations we break and go straight to searching all voxels - 10 is good but you may need more if you have a poor reconstruction

# Re-rerun and save reconstruction
re_run_and_save = 1 # If 1 this will re-run the full reconstruction with all grains and save an npz + paraview file
# %% ==========================================================================
# LOAD IMAGES AND EXPERIMENT - DO NOT EDIT
# =============================================================================
# Generate the experiment
experiment, image_stack = nfutil.generate_experiment(config_fname)
controller = nfutil.build_controller(ncpus=experiment.ncpus, chunk_size=experiment.chunk_size, check=None, generate=None, limit=None)

# %% ==========================================================================
# GENERATE ORIENTATIONS TO TEST - DO NOT CHANGE
#==============================================================================
# Create a regular orientation grid
quats = np.transpose(nfutil.uniform_fundamental_zone_sampling(experiment.point_group_number,average_angular_spacing_in_deg=ori_grid_spacing))
n_grains = quats.shape[1]

# Convert to rotation matrices and exponential maps
exp_maps_to_precompute = np.zeros([quats.shape[1],3])
for i in range(0,quats.shape[1]):
    phi = 2*np.arccos(quats[0,i])
    n = xfcapi.unitRowVector(quats[1:,i])
    exp_maps_to_precompute[i,:] = phi*n

# Precompute all relevant orientation data for each orientaiton
# This can get very RAM heavy
orientation_data_to_test = \
    nfutil.precompute_diffraction_data(experiment,controller,exp_maps_to_precompute)

# %% ==========================================================================
# GENREATE TEST COORDINATES - DO NOT CHANGE
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

# %% ==========================================================================
# FIND MISSING GRAINS SMARTLY - DO NOT CHANGE
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
count = 0
while count < 2:#np.shape(test_coordinates)[0] > coord_cutoff:
    count = count+1
    # Initialize
    t1 = timeit.default_timer() # Start a timer
    print('-------------------------------------------------')
    print('Searching for a grain.')
    print('-------------------------------------------------')

    # Grab a test coordinate
    # Using a random coordinate to avoid sampling the edges before filling in middle holes
    idx = int(np.floor(np.random.uniform(low = 0, high = np.shape(test_coordinates)[0])))
    coordinate_to_test = test_coordinates[idx,:]
    id = ids[idx]

    # Test a single coordinate and refine orientation
    refined_exp_map,refined_conf,refined_idx = nfutil.test_orientations_at_coordinates(experiment,controller,image_stack,orientation_data_to_test,coordinate_to_test,refine_yes_no=1)
    print(f'Orientaiton determined at this voxel with {np.round(refined_conf[0]*100)}% confidence.')

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
        single_orientation_data_to_test = nfutil.precompute_diffraction_data(experiment,controller,refined_exp_map)
        exp_maps,confidence,idx = nfutil.test_orientations_at_coordinates(experiment,controller,image_stack,single_orientation_data_to_test,test_coordinates)
        print(f'This orientation was found at {np.sum(confidence > confidence_threshold)} total voxels.')
        
        # Reset our test_coordinates
        test_coordinates = test_coordinates[confidence < confidence_threshold]
        ids = ids[confidence < confidence_threshold]

        print('Saving the current grains.out.')
        # Save the grains.out in case you want it before the script is done
        gw = instrument.GrainDataWriter(
            os.path.join(output_directory, output_stem+'.out')
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
# %% ==========================================================================
# BRUTE FORCE REMAINING VOXELS - DO NOT CHANGE
#==============================================================================
# At this point, it is statistically likely that we found most of the grains
# The startup and shutdown cost of the multiprocessing is not time cheap so we will 
    # go ahead and brute force the rest of the coordinates in one go to check for any
    # remaining grains
print(f'Starting serach with {n_grains} orientations on {np.shape(test_coordinates)[0]} spatial coordinates.')
refined_exp_maps,refined_confidence,refined_idx = nfutil.test_orientations_at_coordinates(experiment,controller,image_stack,orientation_data_to_test,test_coordinates,refine_yes_no=1)
print(f'Search done, entering refinement on orientations with greater than {confidence_threshold} confidence.')
print(f"Took {np.round((timeit.default_timer() - t0)/60.)} minutes so far.")

# %% ==========================================================================
# CULL DUPLICATE ORIENTATIONS - DO NOT CHANGE
#==============================================================================
print('Checking for similar orientations.')
# Initialize
idx = refined_idx[refined_confidence>confidence_threshold]
working_quats = quats[:,idx.squeeze()]
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
        idx_to_merge = misorientations < np.radians(misorientation_spacing)
        # Remove them and add a single orientation to the list
        final_quats[:,count] = working_quats[:,0]
        working_quats = np.delete(working_quats,idx_to_merge,1)
        count = count + 1

# Trim the list
final_quats = final_quats[:,:count-1]
# Convert to exp maps
final_exp_maps = rotations.expMapOfQuat(final_quats)
print(f'Found {count} additional grains during brute force search.')

# %% ==========================================================================
# SAVE A FINAL GRAINS.OUT - DO NOT CHANGE
#==============================================================================
print('Saving a final grains.out.')
final_exp_maps = np.vstack([new_exp_maps,np.transpose(final_exp_maps)])
# Get original exp_maps
gw = instrument.GrainDataWriter(
    os.path.join(output_directory, output_stem+'.out')
)
for gid, ori in enumerate(final_exp_maps):
    grain_params = np.hstack([ori, constants.zeros_3, constants.identity_6x1])
    gw.dump_grain(gid, 1., 0., grain_params)
gw.close()
print('Done.')

# %% ==========================================================================
# RE-RUN RECONSTRUCTION AND SAVE OUTPUTS - DO NOT CHANGE
#==============================================================================
if re_run_and_save == 1:
    print('Re-running reconstruction.')
    # Generate the experiment
    grain_out_file = os.path.join(output_directory, output_stem+'.out')
    experiment = nfutil.generate_experiment(grains_out_filepath, detector_filepath, materials_filepath, material_name, 
                                            max_tth,completness_threshold, chi2_threshold,omega_edges_deg,
                                            beam_stop_parms,voxel_spacing,vertical_bounds,cross_sectional_dim=cross_sectional_dimensions)
    controller = nfutil.build_controller(ncpus=ncpus, chunk_size=chunk_size, check=None, generate=None, limit=None)
    # Generate space
    Xs, Ys, Zs, mask, test_coordinates = nfutil.generate_test_coordinates(experiment.cross_sectional_dimensions, experiment.vertical_bounds, voxel_spacing,mask_data_file=mask_filepath,mask_vert_offset=mask_vertical_offset)
    # Precompute
    precomputed_orientation_data = nfutil.precompute_diffraction_data(experiment,controller,experiment.exp_maps)
    # Run the search
    raw_exp_maps, raw_confidence, raw_idx = nfutil.test_orientations_at_coordinates(experiment,controller,image_stack,precomputed_orientation_data,test_coordinates,refine_yes_no=0)
    # Process the output
    grain_map, confidence_map = nfutil.process_raw_data(raw_confidence,raw_idx,Xs.shape,mask=mask,id_remap=experiment.remap)
    # Save npz
    nfutil.save_nf_data(output_directory, output_stem, grain_map, confidence_map,
                        Xs, Ys, Zs, experiment.exp_maps, tomo_mask=mask, id_remap=experiment.remap,
                        save_type=['npz']) # Can be npz or hdf5
    # Save paraview h5
    nfutil.save_nf_data_for_paraview(output_directory,output_stem,grain_map,confidence_map,Xs,Ys,Zs,
                                experiment.exp_maps,experiment.mat[material_name], tomo_mask=mask,
                                id_remap=experiment.remap)





# %%
