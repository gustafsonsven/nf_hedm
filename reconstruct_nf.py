#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
original author: dcp5303
edits made by: seg246
"""
"""
NOTES:
    - The reference frame used in this script is the HEXRD frame
        - X points right if facing towards the x-ray detector (downstream)
        - Y points up (against gravity)
        - Z points upstream (away from the detector)
        - X,Y,Z center is where the beam intercepts the rotation axis

    - The tomography mask input is a binarized array of the entire tomography volume
    - Y (vertical) calibration is only needed if your detector center was not
        placed at the center of the beam (usually the case)
    - Z distance is around 6 mm normally (11 mm if the furnace is in)
    - X distance is the same as from your tomography reconstruction
        If you have a RAMS sample then it is usually less than 0.1 mm
    - You voxel size should not be less than your pixel size - you cannot claim such resolution
    - The IPF color plotting has not be unit tested and as such it should not be used for 
        anything but general debugging and initial visualization (currently)
    - This reconstruction alogorithm only produces a grain averaged microstructure
        If your sample has high dislocation content it will not do well
    - For choosing the HKLs, it is advised to draw out the rough geometry and run 
        the numbers to see which HKLs will hit the detector at the front and 
        back of the sample - refine as you calibration z
    - If FF did not find a grain, it will show up as a low confidence region in
        the NF reconstruction
    - Your images have already been processed and binarized with a prior script


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

# %% Imports - NO CHANGES NEEDED
#==============================================================================
# General Imports
import numpy as np
import multiprocessing as mp
import os
import psutil
import copy
from scipy.stats import norm
import copy

# Hexrd imports
from hexrd.transforms.xfcapi import makeRotMatOfExpMap
import nfutil as nfutil
import importlib
importlib.reload(nfutil)
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
#should be of the form: '/nfs/chess/aux/reduced_data/cycles/[cycle ID]/[beamline]/BTR/sample'
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

#==============================================================================
# %% Output information - CAN BE EDITED
#==============================================================================
# Where do you want to drop any output files
output_dir = main_dir + '/output/'
# What was the stem you used during image creation via nf_multithreaded_image_processing?
image_stem = 'ti-13-exsitu_layer_2'
# How do you want your outputs to be named?
output_stem = 'ti-13-exsitu_layer_2_merged'

#==============================================================================
# %% Grains.out File - CAN BE EDITED
#==============================================================================
# Location of grains.out file from far field
grain_out_file = '/nfs/chess/aux/cycles/2023-2/id3a/shanks-3731-a/reduced_data/ti-13-exsitu/ff/output/7/grains.out'
grain_out_file = '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/shanks-3731-a/ti-13-exsitu/nf/merged_2023_09_13.out'

# Completness threshold - grains with completness GREATER than this value will be used
comp_thresh = 0.25 # 0.5 is a good place to start

# Chi^2 threshold - grains with Chi^2 LESS than this value will be used
chi2_thresh = 0.005  # 0.005 is a good place to stay at unless you have good reason to change it

#==============================================================================
# %% Tomography Mask and Geometry - CAN BE EDITED
#==============================================================================
# Use a mask or not
use_mask = True #True
# Mask location, used if use_mask=True
mask_data_file = '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/shanks-3731-a/ti-13-exsitu/tomo//coarse_tomo_mask.npz'
# Vertical offset: this is generally the difference in y motor positions 
# between the tomo and nf layer (tomo_motor_z-nf_motor_z), needed for registry
mask_vert_offset = -(-0.315) # mm

# If no tomography is used (use_mask=False) we will generate a square test grid
# Cross sectional to reconstruct (should be at least 20%-30% over sample width)
cross_sectional_dim = 1.3 # dia in mm
voxel_spacing = 0.005 # in mm, voxel spacing for the near field reconstruction

#==============================================================================
# %% Other details - CAN BE EDITED
#==============================================================================

# Vertical (y) reconstruction voxel bounds in mm, ALWAYS USED REGARDLESS OF TOMOGRAPHY
# If bounds are equal, a single layer is produced
# Suggestion: set v_bounds to cover exactly the voxel_spacing when calibrating
v_bnds = [-0.05, 0.05] # mm 

# Beam stop details
beam_stop_y_cen = 0.0  # mm, measured from the origin of the detector paramters
beam_stop_width = 0.2  # mm, width of the beam stop vertically

# Multiprocessing and RAM parameters
check = None
limit = None
generate = None
ncpus = 128 #mp.cpu_count() - 10 #use as many CPUs as are available
chunk_size = -1 # Don't mess with unless you know what you're doing
RAM_set = True  # if True, manually set max amount of ram
max_RAM = 200 # only used if RAM_set is true. in GB

# Command line flag (Set to True if you are running from the command line)
# This will inhibit plotting and calibration
command_line = True

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
#reconstruction with misorientation included, for many grains, this will quickly
#make the reconstruction size unmanagable
misorientation_bnd = 0.0  # degrees
misorientation_spacing = 0.25  # degrees

# Make beamstop
beam_stop_parms = np.array([beam_stop_y_cen, beam_stop_width])

# Generate the experiment
experiment, nf_to_ff_id_map = nfutil.gen_trial_exp_data(grain_out_file, det_file,
                                                        mat_file, mat_name, max_tth,
                                                        comp_thresh, chi2_thresh,
                                                        omega_edges_deg,beam_stop_parms,misorientation_bnd,
                                                        misorientation_spacing)

#==============================================================================
# %% LOAD / GENERATE TEST DATA  - NO CHANGES NEEDED
#==============================================================================
if use_mask:
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
    
    #sub_space = (Xs<0.1) & (Xs>-0.1) & (Zs<0.1) & (Zs>-0.1)
    #mask = mask*sub_space
    to_use = np.where(mask.flatten())[0]
else:
    test_crds_full, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid(
        cross_sectional_dim, v_bnds, voxel_spacing)
    to_use = np.arange(len(test_crds_full))

test_crds = test_crds_full[to_use, :]


#==============================================================================
# %% Testing Ground
#==============================================================================
importlib.reload(nfutil)

#precomputed_orientation_data = nfutil.precompute_diffraction_data_of_single_orientation(experiment,experiment.exp_maps[0,:])
#exp_map, angles, rMat_ss, gvec_cs, rMat_c = precomputed_orientation_data

#precomputed_orientation_data = nfutil.precompute_diffraction_data_of_many_orientations(experiment,experiment.exp_maps)
#exp_map, angles, rMat_ss, gvec_cs, rMat_c, start, stop = precomputed_orientation_data
ncpus = 128
controller = nfutil.build_controller(
    ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)

precomputed_orientation_data = nfutil.precompute_diffraction_data(experiment,controller,experiment.exp_maps)
# all_exp_maps, all_angles, all_rMat_ss, all_gvec_cs, all_rMat_c = precomputed_orientation_data
#==============================================================================
# %% Testing Ground 2
#==============================================================================
importlib.reload(nfutil)
ncpus = 128
controller = nfutil.build_controller(
    ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)

all_exp_maps, all_confidence, all_idx = nfutil.test_orientations_at_coordinates(experiment,controller,image_stack,precomputed_orientation_data,test_crds,refine_yes_no=0)


#==============================================================================
# %% RAM Splitting - NO CHANGES NEEDED
#==============================================================================

if RAM_set is True:
    RAM = max_RAM * 1e9
else:
    RAM = psutil.virtual_memory().available  # in GB
#  # turn into number of bytes

RAM_to_use = 0.75 * RAM

n_oris = len(nf_to_ff_id_map)
n_voxels = len(test_crds)

bits_for_arrays = 64*n_oris*n_voxels + 192 * \
    n_voxels  # bits raw conf + bits voxel positions
bytes_for_array = bits_for_arrays/8.

n_groups = int(np.floor(bytes_for_array/RAM_to_use))  # number of full groups
# Break it up into groups if needed
if n_groups != 0:
    leftover_voxels = np.mod(n_voxels, n_groups)

    print('Splitting data into %d groups with %d leftover voxels' %
        (int(n_groups), int(leftover_voxels)))


    grouped_voxels = n_voxels - leftover_voxels

    voxels_per_group = grouped_voxels/n_groups
else:
    print('No splitting needed with alloted RAM')
#==============================================================================
# %% Multiprocessing Controller - NO CHANGES NEEDED
#==============================================================================
# assume that if os has fork, it will be used by multiprocessing.
# note that on python > 3.4 we could use multiprocessing get_start_method and
# set_start_method for a cleaner implementation of this functionality.
# This breaks on Windows since windows os cannot do fork
multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'
controller = nfutil.build_controller(
    ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)

#==============================================================================
# %% TEST ORIENTATIONS - NO CHANGES NEEDED
#==============================================================================
print('Testing Orientations...')

if n_groups == 0:
    raw_confidence = nfutil.test_orientations(
        image_stack, experiment, test_crds, controller, multiprocessing_start_method)

    del controller

    raw_confidence_full = np.zeros(
        [len(experiment.exp_maps), len(test_crds_full)])

    for ii in np.arange(raw_confidence_full.shape[0]):
        raw_confidence_full[ii, to_use] = raw_confidence[ii, :]

else:
    grain_map_list = np.zeros(n_voxels)
    confidence_map_list = np.zeros(n_voxels)

    # test voxels in groups
    for group in range(int(n_groups)):
        voxels_to_test = test_crds[int(
            group) * int(voxels_per_group):int(group + 1) * int(voxels_per_group), :]
        print('Calculating group %d' % group)
        raw_confidence = nfutil.test_orientations(
            image_stack, experiment, voxels_to_test, controller, multiprocessing_start_method)
        print('Calculated raw confidence group %d' % group)
        grain_map_group_list, confidence_map_group_list = nfutil.process_raw_confidence(
            raw_confidence, id_remap=nf_to_ff_id_map, min_thresh=0.0)

        grain_map_list[int(
            group) * int(voxels_per_group):int(group + 1) * int(voxels_per_group)] = grain_map_group_list

        confidence_map_list[int(
            group) * int(voxels_per_group):int(group + 1) * int(voxels_per_group)] = confidence_map_group_list
        # change group from abcd (what sort of variable name is that...)
        del raw_confidence

    if leftover_voxels > 0:
        #now for the leftover voxels
        voxels_to_test = test_crds[int(
            n_groups) * int(voxels_per_group):, :]
        raw_confidence = nfutil.test_orientations(
            image_stack, experiment, voxels_to_test, controller, multiprocessing_start_method)
        grain_map_group_list, confidence_map_group_list = nfutil.process_raw_confidence(
            raw_confidence, id_remap=nf_to_ff_id_map, min_thresh=0.0)

        grain_map_list[int(
            n_groups) * int(voxels_per_group):] = grain_map_group_list

        confidence_map_list[int(
            n_groups) * int(voxels_per_group):] = confidence_map_group_list
    
    #fix so that chunking will work with tomography
    grain_map_list_full=np.zeros(Xs.shape[0]*Xs.shape[1]*Xs.shape[2])
    confidence_map_list_full=np.zeros(Xs.shape[0]*Xs.shape[1]*Xs.shape[2])
    
    for jj in np.arange(len(to_use)):
        grain_map_list_full[to_use[jj]]=grain_map_list[jj]
        confidence_map_list_full[to_use[jj]]=confidence_map_list[jj]
    
    #reshape them
    grain_map = grain_map_list_full.reshape(Xs.shape)
    confidence_map = confidence_map_list_full.reshape(Xs.shape)

    del controller

#==============================================================================
# %% POST PROCESS - NO CHANGES NEEDED
#==============================================================================
#note all masking is already handled by not evaluating specific points
grain_map, confidence_map = nfutil.process_raw_confidence(
    raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map, min_thresh=0.0)

#==============================================================================
# %% Show Images - CAN BE EDITED
#==============================================================================
if command_line == False:
    layer_num = 10 # Which layer in Y?
    conf_thresh = 0.6 # If set to None no threshold is used
    nfutil.plot_ori_map(grain_map, confidence_map, Xs, Zs, experiment.exp_maps, 
                        layer_num,experiment.mat[mat_name],nf_to_ff_id_map,conf_thresh)
    # Quick note - nfutil assumes that the IPF reference vector is [0 1 0]
    # Print out the average and max confidence
    print('The average confidence map value is: ' + str(np.mean(confidence_map)) +'\n'+
        'The maximum confidence map value is : ' + str(np.max(confidence_map)))

#==============================================================================
# %% SAVE PROCESSED GRAIN MAP DATA - CAN BE EDITED
#==============================================================================
nfutil.save_nf_data(output_dir, output_stem, grain_map, confidence_map,
                    Xs, Ys, Zs, experiment.exp_maps, tomo_mask=mask, id_remap=nf_to_ff_id_map,
                    save_type=['npz']) # Can be npz or hdf5

#==============================================================================
# %% SAVE PROCESSED GRAIN MAP DATA WITH IPF COLORS - CAN BE EDITED
#==============================================================================
nfutil.save_nf_data_for_paraview(output_dir,output_stem,grain_map,confidence_map,Xs,Ys,Zs,
                             experiment.exp_maps,experiment.mat[mat_name], tomo_mask=mask,
                             id_remap=nf_to_ff_id_map)
# Quick note - nfutil assumes that the IPF reference vector is [0 1 0]

#==============================================================================
# %% Calibration Iterator - CAN BE EDITED
#==============================================================================
if command_line == False:
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
    v_bnds_all = [-0.05, 0.05] # mm

    # Grab original values
    working_experiment = copy.deepcopy(experiment)

    # Define multiprocssing
    multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

    # Reset test grid if we ran it already
    test_crds_full, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid(
        cross_sectional_dim, v_bnds, voxel_spacing)
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
            raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map, min_thresh=0.0)
            
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
        raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map, min_thresh=0.0)


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
            raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map, min_thresh=0.0)

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
        raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map, min_thresh=0.0)

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







# # %% ============================================================================
# # Find Missing Grains
# # ===============================================================================
# # Function to find regions of low confidence with a single or merged volume
# def find_low_confidence_centroids(npz_data_path,output_dir,output_stem,num_diffraction_volumes=1,confidence_threshold=0.5):
#     # Data path must point to a npz save of either a merged volume or single volume
#     # Within this file there MUST be:
#         # confidence, X, Y, Z, and mask
#         # If you do not have a mask, make one.  
    
#     # Load the data
#     np.load(npz_data_path)

