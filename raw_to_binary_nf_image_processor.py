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
import nf_config
import importlib
importlib.reload(nfutil) # This reloads the file if you made changes to it

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
# %% ===========================================================================
# USER INPUT - CAN BE EDITED
# ==============================================================================
# What is the file path to the configuration file?
configuration_filepath = '/nfs/chess/user/seg246/software/development/nf_config.yml'

# %% ===========================================================================
# LOAD CONFIGURATION - DO NOT EDIT
# ==============================================================================
# Go ahead and load the configuration
configuration = nf_config.open_file(configuration_filepath)[0]

# Comb the nf folder for metadata files (.json and .par) and compile them
all_meta = nfutil.skim_metadata(configuration)

# Find the folders associated with this z_height 
unique_zheights = np.sort(all_meta[configuration.images.loading.vertical_motor_name].unique())
meta = all_meta[np.round(all_meta[configuration.images.loading.vertical_motor_name],5) == configuration.images.loading.target_vertical_position]

# Manually downselect if needed
meta = meta[:4]

# Grab the array of per-frame omega values and file locations
filenames,num_imgs = nfutil.skim_image_locations(meta, configuration.images.loading.sample_raw_data_folder)

# Generate the omega edges from the .par file information
omegas,omega_edges_deg = nfutil.generate_omega_edges(meta,num_imgs)

# %% ===========================================================================
# LOAD IMAGES - DO NOT EDIT
# ==============================================================================
# Load all of the images
controller = nfutil.build_controller(configuration)
raw_image_stack = nfutil.load_all_images(filenames,controller)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if configuration.output_plot_check:
    img_num = 100
    fig = plt.figure()
    plt.title('Raw Image: ' + str(img_num))
    plt.imshow(raw_image_stack[img_num,:,:],interpolation='none',clim=[30, 50],cmap='bone')
    plt.show(block=False)
# %% ===========================================================================
# INTENSITY CHECK - DO NOT EDIT
# ==============================================================================
if configuration.output_plot_check:
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
cleaned_image_stack = nfutil.remove_median_darkfields(raw_image_stack,controller,configuration)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if configuration.output_plot_check:
    fig, axs = plt.subplots(1,2)
    img_num = 100
    axs[0].imshow(raw_image_stack[img_num,:,:],interpolation='none',clim=[0, 50],cmap='bone')
    axs[1].imshow(cleaned_image_stack[img_num,:,:],interpolation='none',clim=[0, 40],cmap='bone')
    axs[0].title.set_text('Raw Image: ' + str(img_num))
    axs[1].title.set_text('Cleaned Image: ' + str(img_num))
    plt.show(block=False)
# %% ===========================================================================
# IMAGE CLEANING AND BINARIZATION - DO NOT EDIT
# ==============================================================================
# Perform image filtering, small object removal, and binarization
binarized_image_stack = nfutil.filter_and_binarize_images(cleaned_image_stack,controller,configuration.images.processing.method)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if configuration.output_plot_check:
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
dilated_image_stack = nfutil.dilate_image_stack(binarized_image_stack,configuration.images.processing.dilate_omega)

# %% ===========================================================================
# PLOTTING - CAN BE EDITED
# ==============================================================================
if configuration.output_plot_check == True and configuration.images.processing.dilate_omega > 0:
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
print(f'Saving image stack and omega edges to: {configuration.output_directory}')
nfutil.save_image_stack(configuration,dilated_image_stack,omega_edges_deg)


# %% ==========================================================================
# MAKE A SCINTILATOR/BEAMSTOP MASK - DO NOT EDIT
# =============================================================================
# This is only needed when using the Multilayer optic or the 2x lens
# If neither of these apply, skip this
num_img_for_median = 50
binarization_threshold = 20
errosions = 10
dilations = 10
feature_size_to_remove = 10000
beamstop_mask = nfutil.make_beamstop_mask(raw_image_stack,num_img_for_median,binarization_threshold,errosions,dilations,feature_size_to_remove)

plt.figure()
plt.imshow(beamstop_mask,interpolation=None,clim=[0,20])
plt.show()

print(f'Saving beam stop mask to: {configuration.output_directory}')
np.save(configuration.output_directory + os.sep + configuration.analysis_name + '_beamstop_mask.npy', beamstop_mask)


# %%
