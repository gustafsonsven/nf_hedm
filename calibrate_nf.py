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
command_line = False

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
experiment = nfutil.generate_experiment(grain_out_file, det_file, mat_file, mat_name, 
                                        max_tth,comp_thresh, chi2_thresh,omega_edges_deg,
                                        beam_stop_parms,
                                        misorientation_bnd=misorientation_bnd,
                                        misorientation_spacing=misorientation_spacing,
                                        v_bnds=v_bnds,cross_sectional_dim=cross_sectional_dim,voxel_spacing=voxel_spacing)
controller = nfutil.build_controller(
    ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)
#==============================================================================
# %% LOAD MASK / GENERATE TEST COORDINATES  - NO CHANGES NEEDED
#==============================================================================
Xs, Ys, Zs, mask, test_coordinates = nfutil.generate_test_coordinates(cross_sectional_dim, experiment.vertical_bounds, voxel_spacing,mask_data_file,mask_vert_offset)

#==============================================================================
# %% PRECOMPUTE ORIENTATION DATA
#==============================================================================
precomputed_orientation_data = nfutil.precompute_diffraction_data(experiment,controller,experiment.exp_maps)

#==============================================================================
# %% TEST ORIENTATIONS AND PROCESS OUTPUT
#==============================================================================
raw_exp_maps, raw_confidence, raw_idx = nfutil.test_orientations_at_coordinates(experiment,controller,image_stack,precomputed_orientation_data,test_coordinates,refine_yes_no=0)
grain_map, confidence_map = nfutil.process_raw_data(raw_confidence,raw_idx,Xs.shape,mask=mask,id_remap=experiment.remap)

#==============================================================================
# %% Show Images - CAN BE EDITED
#==============================================================================
if command_line == False:
    layer_num = 0 # Which layer in Y?
    conf_thresh = 0.0 # If set to None no threshold is used
    nfutil.plot_ori_map(grain_map, confidence_map, Xs, Zs, experiment.exp_maps, 
                        layer_num,experiment.mat[mat_name],experiment.remap,conf_thresh)
    # Quick note - nfutil assumes that the IPF reference vector is [0 1 0]
    # Print out the average and max confidence
    print('The average confidence map value is: ' + str(np.mean(confidence_map)) +'\n'+
        'The maximum confidence map value is : ' + str(np.max(confidence_map)))

#==============================================================================
# %% SAVE PROCESSED GRAIN MAP DATA - CAN BE EDITED
#==============================================================================
nfutil.save_nf_data(output_dir, output_stem, grain_map, confidence_map,
                    Xs, Ys, Zs, experiment.exp_maps, tomo_mask=mask, id_remap=experiment.remap,
                    save_type=['npz']) # Can be npz or hdf5

#==============================================================================
# %% SAVE PROCESSED GRAIN MAP DATA WITH IPF COLORS - CAN BE EDITED
#==============================================================================
nfutil.save_nf_data_for_paraview(output_dir,output_stem,grain_map,confidence_map,Xs,Ys,Zs,
                             experiment.exp_maps,experiment.mat[mat_name], tomo_mask=mask,
                             id_remap=experiment.remap)
# Quick note - nfutil assumes that the IPF reference vector is [0 1 0]

