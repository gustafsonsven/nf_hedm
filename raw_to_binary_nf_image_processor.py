#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:05:28 2023
original authors: Austin Gerlt, Simon Mason
edited by: seg246
"""
# =============================================================================
# %% Imports
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
# The next line is formatted correctly, no matter what your IDE says
# %matplotlib widget
# %matplotlib inline
import matplotlib.pyplot as plt

# =============================================================================
# %% User Input - Can be edited
# =============================================================================
# Paths and names
nf_raw_folder = '/nfs/chess/raw/2023-3/id3a/gustafson-1-a/c103-1-nf'
json_and_par_starter = 'id3a-rams2_nf*'
main_dir = '/nfs/chess/aux/cycles/2023-3/id3a/gustafson-1-a/reduced_data/c103-1-nf/reconstructions/1' #working directory
output_dir = main_dir + '/output' #must exist
output_stem = 'c103-1-nf_layer_1'

# Metadata Paramters
target_zheight = -0.05 # At what z height was this NF taken?  This defines what set of scans is grabbed from metadata.  
zheight_motor_name = 'ramsz' # What is the name of the z motor used?

# Multiprocessing
ncpus = 128 #mp.cpu_count() - 10 # Use as many CPUs as are available
chunk_size = -1 # Use -1 if you wish automatic chunk_size calculation

# Darfield Generation
median_size_through_omega = 50 # For dynamic XX makes sense, for static use 250

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
use_dynamic_median = True # If set to false will use a universial dark generated from first 'num_images_to_use' images

print('User Inputs Read')
# =============================================================================
# %% Collect Metadata and file locations - DO NOT EDIT
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

print('Metadata Read')
# ==============================================================================
# %% Load Images - DO NOT EDIT
# ==============================================================================
# Load all of the images
controller = nfutil.build_controller(ncpus=ncpus, chunk_size=chunk_size, check=None, generate=None, limit=None)
raw_image_stack = nfutil.load_all_images(filenames,controller)

# ==============================================================================
# %% Plot the Raw Images - Can be edited
# ==============================================================================
if not suppress_plots:
    img_num = 100
    fig = plt.figure()
    plt.title('Raw Image: ' + str(img_num))
    plt.imshow(raw_image_stack[img_num,:,:],interpolation='none',clim=[30, 100],cmap='bone')
    plt.show(block=False)
# ==============================================================================
# %% Plot out the summed intensity for each image as a function of omega
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

# ==============================================================================
# %% Remove Median Darkfield - DO NOT EDIT
# ==============================================================================
# Perform medial darkfield subtraction
cleaned_image_stack = nfutil.remove_median_darkfields(raw_image_stack,controller,median_size_through_omega)

# ==============================================================================
# %% Plot the Darkfield Removed Images - Can be edited
# ==============================================================================
if not suppress_plots:
    fig, axs = plt.subplots(1,2)
    img_num = 100
    axs[0].imshow(raw_image_stack[img_num,:,:],interpolation='none',clim=[30, 50],cmap='bone')
    axs[1].imshow(cleaned_image_stack[img_num,:,:],interpolation='none',clim=[0, 20],cmap='bone')
    plt.show(block=False)
# ==============================================================================
# %% Clean images with whichever cleaner you desire - DO NOT EDIT
# ==============================================================================
binarized_image_stack = nfutil.filter_and_binarize_images(cleaned_image_stack,controller,filter_parameters)

# ==============================================================================
# %% Plot the Images Against Thresholds - Can be edited
# ==============================================================================
if not suppress_plots:
    fig, axs = plt.subplots(1,2)
    img_num = 100
    axs[0].imshow(cleaned_image_stack[img_num,:,:],interpolation='none',clim=[0, 10],cmap='bone')
    axs[1].imshow(binarized_image_stack[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    plt.show(block=False)

# =============================================================================
# %% Dialte the Images in Omega - DO NOT EDIT
# =============================================================================
if dilate_omega == 1:
    dilated_image_stack = nfutil.dilate_image_stack(binarized_image_stack)

# ==============================================================================
# %% Plot the Binarized, Dilated Images - Can be edited
# ==============================================================================
if not suppress_plots:
    fig, axs = plt.subplots(1,2)
    img_num = 100
    axs[0].imshow(binarized_image_stack[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    axs[1].imshow(dilated_image_stack[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    plt.show(block=False)

# =============================================================================
# %% Save the Binarized Image Stack - DO NOT EDIT
# =============================================================================
if save_omegas_and_image_stack:
    if dilate_omega == 1:
        np.save(output_dir + os.sep + output_stem + '_binarized_images.npy', dilated_image_stack)
    else:
        np.save(output_dir + os.sep + output_stem + '_binarized_images.npy', binarized_image_stack)
    np.save(output_dir + os.sep + output_stem + '_omega_edges_deg.npy', omega_edges_deg)
print("Done saving")



































# %%
