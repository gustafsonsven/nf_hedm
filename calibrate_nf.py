#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
original author: dcp5303
contributing author: seg246
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
    - Z distance is around 6-7 mm normally (11 mm if the furnace is in)
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
# %% ==========================================================================
# IMPORTS - DO NOT CHANGE
#==============================================================================
# General Imports
import numpy as np
import multiprocessing as mp
import os

# Hexrd imports
import nfutil as nfutil

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

# %% ==========================================================================
# COMMON USER INFORMATION - CAN BE EDITED
# =============================================================================
# Working directory - could be of the form: '/nfs/chess/aux/reduced_data/cycles/[cycle ID]/[beamline]/BTR/sample/YOUR FAVORITE BOOKKEEPING STRUCTURE'
working_directory = '/your/path/here'

# Where do you want to drop any output files
output_directory = working_directory + '/output/'
output_stem = 'sample_1_name' # Something relevant to your sample

# Detector file (retiga, manta,...)
detector_filepath = working_directory + '/manta.yml'

# Materials file - from HEXRDGUI (MAKE SURE YOUR HKLS ARE DEFINED CORRECTLY FOR YOUR MATERIAL)
materials_filepath = working_directory + '/materials.h5'

# Material name in materials.h5 file from HEXRGUI
material_name = 'in718'
max_tth = None  # degrees, if None is input max tth will be set by the geometry
# NOTE: Again, make sure the HKLs are set correctly in the materials file that you loaded
    # If you set max_tth to 20 degrees, but you only have HKLs out to 15 degrees selected
    # then you will only use the selected HKLs out to 15 degrees

# What was the stem you used during image creation via nf_multithreaded_image_processing?
image_stem = 'sample_1_images'
num_img_to_shift = 0 # Postive moves positive omega, negative moves negative omega, must be integer (if nothing was wrong with your metadata this should be 0)

# Grains.out information
grains_out_filepath = '/your/path/here/grains.out'
# Completness threshold - grains with completness GREATER than this value will be used
completness_threshold = 0.25 # 0.5 is a good place to start
# Chi^2 threshold - grains with Chi^2 LESS than this value will be used
chi2_threshold = 0.005  # 0.005 is a good place to stay at unless you have good reason to change it

# Tomorgraphy mask information
# Mask location
mask_filepath = None # If you have no mask set mask_filepath = None
# Vertical offset: this is generally the difference in y motor positions between the tomo and nf layer (tomo_motor_z-nf_motor_z), needed for registry
mask_vertical_offset = 0.0 # mm

# If no tomography is used (use_mask=False) we will generate a square test grid
# Cross sectional to reconstruct (should be at least 20%-30% over sample width)
cross_sectional_dimensions = 1.3 # Side length of the cross sectional region to probe (mm)
voxel_spacing = 0.005 # in mm, voxel spacing for the near field reconstruction

# Vertical (y) reconstruction voxel bounds in mm, ALWAYS USED REGARDLESS OF TOMOGRAPHY
# A single layer is produced if, for example, vertical_bounds = [-0.0025, 0.0025] with a 0.005 voxel size
vertical_bounds = [-0.06, 0.06] # mm 

# Beam stop details
use_beam_stop_mask = 1 # If 1, this will ignore the next two parameters and load the mask made by the raw_to_binary_nf_image_processor.py
beam_stop_y_cen = 0.0  # mm, measured from the origin of the detector paramters
beam_stop_width = 0.1  # mm, width of the beam stop vertically

# Multiprocessing and RAM parameters
ncpus = 128 # mp.cpu_count() - 10 # Use as many CPUs as are available
chunk_size = -1 # Use -1 if you wish automatic chunk_size calculation

# %% ==========================================================================
# LOAD IMAGES AND EXPERIMENT - DO NOT CHANGE
#==============================================================================
print('Loading the image stack...')
# Load the cleaned image stack from the first script
image_stack = np.load(output_directory + os.sep + image_stem + '_binarized_images.npy')
# Load the omega edges - first value is the starting ome position of first image's slew, last value is the end position of the final image's slew
omega_edges_deg = np.load(output_directory + os.sep + image_stem + '_omega_edges_deg.npy')
# Load/make the beamstop where 1s indicate non-intensity-counting pixels
if use_beam_stop_mask == 1:
    # Load from file
    beam_stop_parms = np.load(output_directory + os.sep + image_stem + '_beamstop_mask.npy')
else:
    # Generate
    beam_stop_parms = np.array([beam_stop_y_cen, beam_stop_width])

# Shift in omega positive or negative by X number of images
if num_img_to_shift > 0:
    # Moving positive omega so first image is not at zero, but further along
    # Using the mean omega step size - change if you need to
    omega_edges_deg = omega_edges_deg + num_img_to_shift*np.mean(np.gradient(omega_edges_deg))
elif num_img_to_shift < 0:
    # For whatever reason the multiprocessor does not like negative numbers, trim the stack
    image_stack = image_stack[np.abs(num_img_to_shift):,:,:]
    omega_edges_deg = omega_edges_deg[:num_img_to_shift]
print('Image stack loaded.')

# Generate the experiment
experiment = nfutil.generate_experiment(grains_out_filepath, detector_filepath, materials_filepath, material_name, 
                                        max_tth,completness_threshold, chi2_threshold,omega_edges_deg,
                                        beam_stop_parms,voxel_spacing,vertical_bounds,cross_sectional_dim=cross_sectional_dimensions)

# Make the controller
controller = nfutil.build_controller(ncpus=ncpus, chunk_size=chunk_size, check=None, generate=None, limit=None)

# %% ==========================================================================
# CALIBRATE THE TRANSLATIONS - CAN BE EDITED
#==============================================================================
parameter = 3 # 0=X, 1=Y, 2=Z, 3=RX, 4=RY, 5=RZ, 6=chi
start = -2 # mm for translations, degrees for rotations
stop = 2 # mm for translations, degrees for rotations
steps = 21
calibration_parameters = [parameter,steps,start,stop]
experiment = nfutil.calibrate_parameter(experiment,controller,image_stack,calibration_parameters)


# %%
