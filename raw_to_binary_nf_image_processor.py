#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contributing authors: dcp5303, ken38, seg246, Austin Gerlt, Simon Mason
"""
"""
    A few notes:
        - Depending on how many images you have this script can eat up a lot of RAM
        - Be sure to double check the meta data that is being read in to ensure it 
            is correct from the .par files and your logbook
        - Change the intensities in the plotting cells to scale to the data you have
        - Take time to double check that the spots you want are being binarized 
            correctly
        - In general, gaussian filtering is the easiest and often the most robust,
            just be careful not to blur your images
        - Non-local means filtering takes more time to optimize the parameters,
            but can do a very good job of detecting the spot edges
        - Errosion/dilation is there if the others fail
        - It is generally suggested to dilate through omega (only one frame on 
            either side) as this will handle any small omega differences between
            FF and NF
        - Removing small features may be needed for the gaussian filter as that
            function can blur hot pixels/noise to look like a very small spot.  
            Be very careful not to remove spots from grains which are small so 
            it is suggested to do some quick math on how many pixels to set as 
            the threshold in comarison to your grain size.  
"""
# %% ==========================================================================
# IMPORTS - DO NOT CHANGE
# =============================================================================
# Gneral imports
import numpy as np
import os

# HEXRD Imports
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
# USER INFORMATION - CAN BE EDITED
# =============================================================================
# Paths and names
# Raw directory - could be of the form: '/nfs/chess/raw/[cycle ID]/[beamline]/BTR/sample'
nf_raw_folder = '/nfs/chess/raw/current/id3a/pagan-3807-a/in718ln'
json_and_par_starter = 'id3a-rams2_nf*'
# Working directory - could be of the form: '/nfs/chess/aux/reduced_data/cycles/[cycle ID]/[beamline]/BTR/sample/YOUR FAVORITE BOOKKEEPING STRUCTURE'
working_directory = '/nfs/chess/user/seg246/software/development_working_space'
output_dir = working_directory + '/output' # Folder must exist already
output_stem = 'test_2x_objective_with_scintillator_layer_3'

# Metadata Paramters
target_zheight = -0.25 # At what z height was this NF taken?  This defines what set of scans is grabbed from metadata.  
zheight_motor_name = 'ramsz' # What is the name of the z motor used?

# Multiprocessing
ncpus = 128 # mp.cpu_count() - 10 # Use as many CPUs as are available
chunk_size = -1 # Use -1 if you wish automatic chunk_size calculation

# Darfield Generation
median_size_through_omega = 50 # For dynamic 10 degs worth of images in omega makes sense

# Do you want any small objects removed from the binarized images
remove_small_binary_objects = 1 # If 0 it will not remove small features, if 1 it will
close_size = 200 # Number of pixels

# Choose a routine
cleaning_routine = 0 # 0 = gaussian blurring, 1 = dilation/errosion, 2 = non-local means
if cleaning_routine == 0:
    # Gaussian Cleaning Parameters
    sigma = 2.0 # Larger values begin to rapidly blur images
    binarization_threshold = 3.0 # Images are floats when this is applied
    filter_parameters = [remove_small_binary_objects,close_size,cleaning_routine,
                         sigma,binarization_threshold]
elif cleaning_routine == 1:
    # Erosion/Dilation Parameters
    num_errosions = 3 # General practice, num_errosions > num_dilations
    num_dilations = 2
    binarization_threshold = 5 # Images are uints when this is applied
    filter_parameters = [remove_small_binary_objects,close_size,cleaning_routine,
                        num_errosions,num_dilations,binarization_threshold]
elif cleaning_routine == 2:
    # Non-Local Means Cleaning Parameters
    patch_size = 3
    patch_distance = 5
    binarization_threshold = 10 # Images are floats when this is applied
    filter_parameters = [remove_small_binary_objects,close_size,cleaning_routine,
                        patch_size,patch_distance,binarization_threshold]

# Flags
save_omegas_and_image_stack = True # Will save the binarized image stack and an omega list as an .npy
suppress_plots = False # Will plot debug images
dilate_omega = True # Dilate the binarized image stack in omega

# %% ==========================================================================
# METADATA HANDLING - DO NOT EDIT
# =============================================================================
# Comb the nf folder for metadata files (.json and .par) and compile them
all_meta = nfutil.skim_metadata(nf_raw_folder + os.sep + json_and_par_starter)

# Find the folders associated with this z_height 
unique_zheights = np.sort(all_meta[zheight_motor_name].unique())
meta = all_meta[np.round(all_meta[zheight_motor_name],5) == target_zheight]

# Grab the array of per-frame omega values and file locations
filenames,num_imgs = nfutil.skim_image_locations(meta, nf_raw_folder)

# Generate the omega edges from the .par file information
omegas,omega_edges_deg = nfutil.generate_omega_edges(meta,num_imgs)

# %% ===========================================================================
# LOAD IMAGES - DO NOT EDIT
# ==============================================================================
# Load all of the images
controller = nfutil.build_controller(ncpus=ncpus, chunk_size=chunk_size, check=None, generate=None, limit=None)
raw_image_stack = nfutil.load_all_images(filenames[0:10],controller)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if not suppress_plots:
    img_num = 100
    fig = plt.figure()
    plt.title('Raw Image: ' + str(img_num))
    plt.imshow(raw_image_stack[img_num,:,:],interpolation='none',clim=[30, 100],cmap='bone')
    plt.show(block=False)
# %% ===========================================================================
# INTENSITY CHECK - DO NOT EDIT
# ==============================================================================
if not suppress_plots:
    summed_image_int = np.sum(np.sum(raw_image_stack,axis=1),axis=1)
    plt.figure()
    plt.scatter(np.arange(0,np.shape(omegas)[0],1),summed_image_int)
    plt.ylim(0,np.max(summed_image_int))
    plt.title('Image Intensity vs Image Number')
    plt.xlabel('Image Number')
    plt.ylabel('Summed Intensity')
    plt.show(block=False)

    # What is the median intensity, how many are well below that and what could be the expected confidence drop
    median_int = np.median(summed_image_int)
    num_bad_images = np.sum(summed_image_int < median_int*0.75)
    print('There are potentially ' + str(num_bad_images) + ' images with poor intensity.')
    print('Potential confidence maximum around ' + str(np.round(1 - num_bad_images/np.shape(omegas)[0],2)))

# %% ===========================================================================
# MEDIAN DARKFIELD REMOVAL - DO NOT EDIT
# ==============================================================================
# Perform median darkfield subtraction
cleaned_image_stack = nfutil.remove_median_darkfields(raw_image_stack,controller,median_size_through_omega)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if not suppress_plots:
    fig, axs = plt.subplots(1,2)
    img_num = 100
    axs[0].imshow(raw_image_stack[img_num,:,:],interpolation='none',clim=[0, 80],cmap='bone')
    axs[1].imshow(cleaned_image_stack[img_num,:,:],interpolation='none',clim=[0, 50],cmap='bone')
    axs[0].title.set_text('Raw Image: ' + str(img_num))
    axs[1].title.set_text('Cleaned Image: ' + str(img_num))
    plt.show(block=False)
# %% ===========================================================================
# IMAGE CLEANING AND BINARIZATION - DO NOT EDIT
# ==============================================================================
# Perform image filtering, small object removal, and binarization
binarized_image_stack = nfutil.filter_and_binarize_images(cleaned_image_stack,controller,filter_parameters)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if not suppress_plots:
    fig, axs = plt.subplots(1,2)
    img_num = 100
    axs[0].imshow(cleaned_image_stack[img_num,:,:],interpolation='none',clim=[0, 50],cmap='bone')
    axs[1].imshow(binarized_image_stack[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    axs[0].title.set_text('Cleaned Image: ' + str(img_num))
    axs[1].title.set_text('Binarized Image: ' + str(img_num))
    plt.show(block=False)

# %% ==========================================================================
# OMEGA DILATION - DO NOT EDIT
# =============================================================================
# Dilate the image stack in omega
if dilate_omega == 1:
    dilated_image_stack = nfutil.dilate_image_stack(binarized_image_stack)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if suppress_plots == False and dilate_omega == 1:
    fig, axs = plt.subplots(1,2)
    img_num = 100
    axs[0].imshow(binarized_image_stack[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    axs[1].imshow(dilated_image_stack[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    axs[0].title.set_text('Binarized Image: ' + str(img_num))
    axs[1].title.set_text('Dilated Image: ' + str(img_num))
    plt.show(block=False)

# %% ==========================================================================
# SAVING - DO NOT EDIT
# =============================================================================
if save_omegas_and_image_stack:
    if dilate_omega == 1:
        np.save(output_dir + os.sep + output_stem + '_binarized_images.npy', dilated_image_stack)
    else:
        np.save(output_dir + os.sep + output_stem + '_binarized_images.npy', binarized_image_stack)
    np.save(output_dir + os.sep + output_stem + '_omega_edges_deg.npy', omega_edges_deg)
print("Done saving")

# %% ==========================================================================
# MAKE A SCINTILATOR/BEAMSTOP MASK - DO NOT EDIT
# =============================================================================
num_img_for_median = 250
binarization_threshold = 20
errosions = 10
dilations = 10
feature_size_to_remove = 10000
beamstop_mask = nfutil.make_beamstop_mask(raw_image_stack,num_img_for_median,binarization_threshold,errosions,dilations,feature_size_to_remove)

plt.figure()
plt.imshow(beamstop_mask,interpolation=None)
plt.show()

np.save(output_dir + os.sep + output_stem + '_beamstop_mask.npy', beamstop_mask)





# %%
