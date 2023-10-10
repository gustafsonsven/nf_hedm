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
import pandas as pd
import os
import glob
import json
import re
import time
import multiprocessing as mp

# Image processing Imports
from skimage import io, filters
from skimage.morphology import remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import binary_erosion, binary_dilation
from scipy import ndimage

# Multithreading libraries
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

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
nf_raw_folder = '/nfs/chess/raw/2023-2/id3a/shanks-3731-a/ti-13-exsitu'
json_and_par_starter = 'id3a-rams2_nf*'
main_dir = '/nfs/chess/aux/cycles/2023-2/id3a/shanks-3731-a/reduced_data/ti-13-exsitu/nf/2' #working directory
output_dir = main_dir + '/output' #must exist
output_stem = 'ti-13-exsitu_layer_2'

# Metadata Paramters
target_zheight = -0.315 # At what z height was this NF taken?  This defines what set of scans is grabbed from metadata.  
zheight_motor_name = 'ramsz' # What is the name of the z motor used?

# How many cpus to use?
ncpus = 128 #mp.cpu_count() - 10

# Darfield Generation
num_images_to_use = 25 # For dynamic XX makes sense, for static use 250

# Choose a routine
cleaning_routine = 2 # 1 = non-local means, 2 = gaussian blurring, 3 = dilation/errosion
if cleaning_routine == 1:
    # Non-Local Means Cleaning Parameters
    binarization_threshold = 10 # Images are floats when this is applied
    patch_size = 3
    patch_distance = 5

elif cleaning_routine == 2:
    # Gaussian Cleaning Parameters
    binarization_threshold = 2.0 # Images are floats when this is applied
    sigma = 2.0 # Larger values begin to rapidly blur images

elif cleaning_routine == 3:
    # Erosion/Dilation Parameters
    binarization_threshold = 5 # Images are uints when this is applied
    num_errosions = 3 # General practice, num_errosions > num_dilations
    num_dilations = 2

# Do you want any small objects removed from the binarized images
remove_small_binary_objects = 1
close_size = 200

# Flags
save_omegas_and_image_stack = True # Will save the binarized image stack and an omega list as an .npy
suppress_plots = True # Will plot debug images
dilate_omega = True # Dilate the binarized image stack in omega
use_dynamic_median = True # If set to false will use a universial dark generated from first 'num_images_to_use' images

print('User Inputs Read')

# =============================================================================
# %% Metadata Skimmers - DO NOT EDIT
# =============================================================================
# Metadata skimmer function
def skim_metadata(raw_folder, output_dict=False):
    """
    skims all the .josn and .par files in a folder, and returns a concacted
    pandas DataFrame object with duplicates removed. If Dataframe=False, will
    return the same thing but as a dictionary of dictionaries.
    NOTE: uses Pandas Dataframes because some data is int, some float, some
    string. Pandas auto-parses dtypes per-column, and also has
    dictionary-like indexing.
    """
    # Grab all the nf json files, assert they both exist and have par pairs
    f_jsons = glob.glob(raw_folder + "*json")
    assert len(f_jsons) > 0, "No .jsons found in {}".format(nf_raw_folder)
    f_par = [x[:-4] + "par" for x in f_jsons]
    assert np.all([os.path.isfile(x) for x in f_par]), "missing .par files"
    
    # Read in headers from jsons
    headers = [json.load(open(j, "r")).values() for j in f_jsons]
    
    # Read in headers from each json and data from each par as Dataframes
    df_list = [
        pd.read_csv(p, names=h, delim_whitespace=True, comment="#")
        for h, p in zip(headers, f_par)
    ]
    
    # Concactionate into a single dataframe and delete duplicate columns
    meta_df_dups = pd.concat(df_list, axis=1)
    meta_df = meta_df_dups.loc[:, ~meta_df_dups.columns.duplicated()].copy()
    if output_dict:
        # convert to dict of dicts if requested
        return dict(zip(meta_df.keys(), [x[1].to_list() for x in meta_df.iteritems()]))
    # else, just return
    return meta_df

# Omega locator function
def ome_from_df(meta_df,num_imgs_per_scan):
    """ takes in a dataframe generated from the metadata using "skim_metadata",
    and returns a numpy array of the omega data (ie, what frames represent
    which omega angle in the results)"""
    start = meta_df["ome_start_real"].to_numpy() # ome_start_real is the starting omega position of the first good frame's omega slew
    stop = meta_df["ome_end_real"].to_numpy() # ome_end_read is the final omega position of the last good frame's omega slew
    steps = num_imgs_per_scan
    num_frames_anticipated = meta_df['nframes_real'].to_numpy()
    # This *should* not be needed if meta data was written correctly
    if np.sum(num_frames_anticipated-steps) != 0:
        print('Editing omega stop postion from the metadata to deal with fewer images')
        print('CHECK YOUR OMEGA EDGES AND STEP SIZE WITH np.gradient(omega_edges_deg)')
        stop = stop - (stop/num_frames_anticipated)*(num_frames_anticipated-steps)
    scan = meta_df["SCAN_N"].to_numpy() 
    # Find the omega start positions for each frame
    lines = [np.linspace(a, b - (b - a) / c, c)
             for a, b, c in zip(start, stop, steps)]
    omes = np.hstack([x for y, x in sorted(zip(scan, lines))])
    omega_edges_deg = np.append(omes,stop[np.shape(stop)[0]-1])
    return omes, omega_edges_deg

# Image file locations
def image_locs_from_df(meta_df, raw_folder):
    """ takes in a dataframe generated from the metadata using "skim_metadata"
    plus the near_field folder locations, and returns a list of image locations
    """
    scan = meta_df['SCAN_N'].to_numpy()
    first = meta_df['goodstart'].to_numpy()
    num_frames_anticipated = meta_df['nframes_real'].to_numpy()
    num_imgs_per_scan = np.zeros(len(scan), dtype=int)

    files = []
    flag = 0
    for i in range(len(scan)):
        all_files = glob.glob(raw_folder + os.sep + str(scan[i]) + "/nf/*.tif")
        all_names = [x.split(os.sep)[-1] for x in all_files]
        all_ids_list = [int(re.findall("([0-9]+)", x)[0]) for x in all_names]
        all_ids = np.array(all_ids_list)
        good_ids = (all_ids >= first[i]) * (all_ids <= first[i]+num_frames_anticipated[i]-1)
        files.append([x for x, y in sorted(zip(all_files, good_ids)) if y])
        if sum(good_ids) != num_frames_anticipated[i]:
            flag = 1
            print('There are ' + str(sum(good_ids)) + ' images in scan ' + str(scan[i]) +\
                  ' when ' + str(num_frames_anticipated[i]) + ' were expected')
        num_imgs_per_scan[i] = sum(good_ids)
    
    if flag == 1:
        print("HEY, LISTEN!  There was an unexpected number of images within at least one scan folder.\n\
This code will proceed with the shortened number of images and will assume that\n\
the first image is still the 'goodstart' as defined in the par file.")
            
    # flatten the list of lists
    files = [item for sub in files for item in sub]
    # sanity check
    s_id = np.array(
        [int(re.findall("([0-9]+)", x.split(os.sep)[-3])[0]) for x in files]
    )
    f_id = np.array(
        [int(re.findall("([0-9]+)", x.split(os.sep)[-1])[0]) for x in files]
    )
    assert np.all(s_id[1:] - s_id[:-1] >= 0), "folders are out of order"
    assert np.all(f_id[1:] - f_id[:-1] >= 0), "files are out of order"

    return files, num_imgs_per_scan

print('Metadata Functions Created')
# =============================================================================
# %% Image Processing - DO NOT EDIT
# =============================================================================
# Image reader
def img_in(slice_id, filename):
    """ This just wraps io.imread, but allows the slice_id to be an unused
    input. this makes reordering the threaded data easier"""
    return io.imread(filename)

# Dynamic median function
def dynamic_median(z, median_distance=15):
    """subtracts a local median darkfield from a slice using plus-or-minus a
    number of frames in the omega equal to "median_distance"./home/seg246
    Note, this is NOT a per-image process, it's taking a line from each frame,
    and comparing adjacent frames to each other to allow for multithreading"""
    plate = raw_cube[:, :, z]
    local_dark = ndimage.median_filter(plate, size=[median_distance, 1])
    new_plate = plate-local_dark
    new_plate[local_dark >= plate] = 0
    return new_plate

# Static median function
def static_median(z, nth):
    """subtracts a median darkfield background, where the darkfield is
    calculated using every "nth" frame.
    Note, this is NOT a per-image process, it's taking a line from each frame,
    and comparing adjacent frames to each other to allow for multithreading"""
    plate = raw_cube[:, :, z]
    darkfield = np.median(plate[::nth, :], axis=0)
    # fix this. it makex negatives real big bc uint overflow.
    new_plate = plate-darkfield
    new_plate[darkfield >= plate] = 0
    return new_plate

# Guassian cleanup
def per_image_gaussian_cleanup(slice_id, thresh=1, sigma=2, size=3):
    """Verbaitm copy of the gaussian cleanup used in hexrd.grainmap.nfutil,
    just in a function that is easier to multithread"""
    img = img_cube[slice_id, :, :]
    img = filters.gaussian(img, sigma=sigma,preserve_range=True)
    img_binary = img > thresh
    return slice_id, img, img_binary

# Errosion/dilation cleanup
def per_image_erosion_cleanup(slice_id, thresh=1, erosions=4, dilations=5):
    # Cleanup images using 
    img = img_cube[slice_id, :, :]
    img_binary = img > thresh
    img_binary = binary_erosion(img_binary, iterations=erosions)
    img_binary = binary_dilation(img_binary, iterations=dilations)
    return slice_id, img, img_binary

# Non-local means cleaup
def per_image_nl_means_cleanup(slice_id, thresh=1, patch_size=5, patch_distance=5):
    """
    Simon Mason figured out this function, comments are adopedet from him.
    Takes in a slice id, then loads and processes it in a thread-friendly
    way
    """
    img = img_cube[slice_id, :, :]
    # Estimage the per-slice sigma
    s_est = estimate_sigma(img)
    # Run non-local_means
    img = denoise_nl_means(
        img, sigma=s_est, h=0.8 * s_est, patch_size=patch_size, patch_distance=patch_distance,
        preserve_range = True)
    img_binary = img > thresh
    return slice_id, img, img_binary

# Remove small objects function
def per_image_remove_small_objects(slice_id, close_size=10):
    """
    Area closing - remove small features of the binary images under a specifc number
    of pixels
    """
    # Non-local means on dark field subtracted images
    img = bin_cube[slice_id, :, :]
    img_closed = remove_small_objects(img,close_size,connectivity=1)
    return slice_id, img, img_closed

print('Image Processing Functions Created')
# =============================================================================
# %% Collect Metadata and file locations - DO NOT EDIT
# =============================================================================
# comb the nf folder for metadata files and compile them
all_meta = skim_metadata(nf_raw_folder + os.sep + json_and_par_starter)
# find the unique z_height requests, parse out just the a specific layer
unique_zheights = np.sort(all_meta[zheight_motor_name].unique())
meta = all_meta[all_meta[zheight_motor_name] == target_zheight]
# get the array of per-frame omega values and file locations
f_imgs,num_imgs = image_locs_from_df(meta, nf_raw_folder)
omegas,omega_edges_deg = ome_from_df(meta,num_imgs)
# create an Image Collection, which is basically a hexrd.ImageSeries but better
# NOTE: if you are a hexrd traditionalist and prefer imageseries, you'll
# want to change this line accordingly
img_collection = io.imread_collection(f_imgs)
# Sanity checkall_meta
assert len(omegas) == len(img_collection), """mismatch between expected and actual filecount"""

print('Metadata Read')
# ==============================================================================
# %% Load Images - DO NOT EDIT
# ==============================================================================
# Define the global img_cube where all darkfield-removed images reside
global raw_cube, img_cube, bin_cube

# Load the very first image just to get the dimensions
# img_collection = img_collection[:100] # this is the line to change if you want to do a truncated test
y, z = img_collection[0].shape
raw_cube = np.zeros([len(img_collection), y, z], dtype=np.uint16) # Raw images
img_cube = np.zeros([len(img_collection), y, z], dtype=np.uint16) # Dark field subtracted images
cln_cube = np.zeros([len(img_collection), y, z], dtype=np.uint16) # Processed images
# sum_cube = np.zeros([len(img_collection), y, z], dtype=np.uint16)
bin_cube = np.zeros([len(img_collection), y, z], dtype=bool) # Binarized images
clo_cube = np.zeros([len(img_collection), y, z], dtype=bool) # Area closed images

# Load all of the images
print("loading images...")
with ThreadPoolExecutor(ncpus) as executor:
    tic = time.time()
    id_and_file = [x for x in enumerate(img_collection.files)]
    futures = {executor.submit(img_in, x[0], x[1]): x for x in id_and_file}
    i = 0
    for future in as_completed(futures):
        data = future.result()
        inputs = futures.pop(future)  # this also stops memory leaks
        slice_id = inputs[0]
        raw_cube[slice_id] = data
        i += 1
        print("{} of {} images loaded".format(i, raw_cube.shape[0]))
        del inputs, slice_id, data
    tocA = time.time() - tic
    print(tocA)
    executor.shutdown()

print("Raw Images Loaded")

# ==============================================================================
# %% Plot the Raw Images - Can be edited
# ==============================================================================
if not suppress_plots:
    img_num = 100
    fig = plt.figure()
    plt.title('Raw Image: ' + str(img_num))
    plt.imshow(raw_cube[img_num,:,:],interpolation='none',clim=[10, 30],cmap='bone')
    plt.show(block=False)

# ==============================================================================
# %% Remove Median Darkfield - DO NOT EDIT
# ==============================================================================
# Perform medial darkfield subtraction
with ThreadPoolExecutor(ncpus) as executor:
    tic = time.time()
    zz = np.arange(raw_cube.shape[2])
    if use_dynamic_median:
        # Futures code for dynamic median filter
        futures = {executor.submit(dynamic_median, z, num_images_to_use): z for z in zz}
        print("Starting dynamic median dark field subtraction...(it could take a few minutes)")
    else:
        # Futures code for static median filter
        futures = {executor.submit(static_median, z, num_images_to_use): z for z in zz}
        print("Starting static median dark field subtraction...(it could take a few minutes)")
    i = 0
    for future in as_completed(futures):
        new_plate = future.result()
        z = futures.pop(future)  # this also stops memory leaks
        img_cube[:, :, z] = new_plate
        i += 1
        if i % 25 == 0 or i == zz[-1]:
            print("{} of {} y-slices filtered".format(i, raw_cube.shape[2]))
        del z, new_plate

    tocB = time.time() - tic
    print(tocB)
    executor.shutdown()

print("Dark field subtraction is done")

# ==============================================================================
# %% Plot out the summed intensity for each image as a function of omega
# ==============================================================================
if not suppress_plots:
    summed_image_int = np.sum(np.sum(raw_cube,axis=1),axis=1)
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
# %% Plot the Darkfield Removed Images - Can be edited
# ==============================================================================
if not suppress_plots:
    img_num = 100
    fig = plt.figure()
    plt.title('Image with Darkfield Removed: ' + str(img_num))
    plt.imshow(img_cube[img_num,:,:],interpolation='none',clim=[0, 10],cmap='bone')
    plt.show(block=False)

# ==============================================================================
# %% Clean images with whichever cleaner you desire - DO NOT EDIT
# ==============================================================================
executor = ThreadPoolExecutor(ncpus)
#xx = np.arange(img_cube.shape[0])
xx = np.arange(img_cube.shape[0])

# Define the cleanup routine
if cleaning_routine == 1:
    print("Using Non-Local Means Cleanup")
    futures = {executor.submit(per_image_nl_means_cleanup,x, binarization_threshold, patch_size, patch_distance): x for x in xx}
elif cleaning_routine == 2:
    print("Using Gaussian Cleanup")
    futures = {executor.submit(per_image_gaussian_cleanup, x, binarization_threshold, sigma): x for x in xx}
elif cleaning_routine == 3:
    print("Using Errosion/Dilation Cleanup")
    futures = {executor.submit(per_image_erosion_cleanup, x, binarization_threshold, num_errosions, num_dilations): x for x in xx}
elif cleaning_routine == 4:
    print('Using Homebrewed Cleanup')
    futures = {executor.submit(per_image_seg_cleanup,x, ed_thresh=1, g_thresh = 1, err=1,
                           dil=1, sigma = 2.5): x for x in xx}

# Run the routine
print("starting denoising and binarization...")
i = 0
tic = time.time()
for future in as_completed(futures):
    x, cleaned_slice, binarized_slice = future.result()
    cln_cube[x, :, :] = cleaned_slice
    bin_cube[x, :, :] = binarized_slice
    inputs = futures.pop(future)  # this stops memory leaks
    i += 1
    if i % 25 == 0 or i == xx[-1]:
        print("{} of {} layers cleaned".format(i, img_cube.shape[0]))
    del inputs, x, cleaned_slice, binarized_slice  # this is just a cleanup
tocC = time.time() - tic
print(tocC)
executor.shutdown()

print("Image Cleanup is done")
# ==============================================================================
# %% Run a binary area closure if desired - DO NOT EDIT
# ==============================================================================
executor = ThreadPoolExecutor(ncpus)
#xx = np.arange(img_cube.shape[0])
xx = np.arange(img_cube.shape[0])

# Define the cleanup routine
if remove_small_binary_objects == 1:
    print("Removing small binary objects")
    futures = {executor.submit(per_image_remove_small_objects,x, close_size): x for x in xx}
    # Run the routine
    i = 0
    tic = time.time()
    for future in as_completed(futures):
        x, binarized_slice, closed_slice = future.result()
        clo_cube[x, :, :] = closed_slice
        inputs = futures.pop(future)  # this stops memory leaks
        i += 1
        if i % 25 == 0 or i == xx[-1]:
            print("{} of {} layers done".format(i, img_cube.shape[0]))
        del inputs, x, closed_slice, binarized_slice  # this is just a cleanup
    tocC = time.time() - tic
    print(tocC)
    executor.shutdown() 

    print("Small objects have been removed")

# ==============================================================================
# %% Plot the Images Against Thresholds - Can be edited
# ==============================================================================
if not suppress_plots:
    fig, axs = plt.subplots(2,2)
    img_num = 100
    inten = binarization_threshold
    axs[0,0].imshow(img_cube[img_num,:,:],interpolation='none',clim=[0, inten],cmap='bone')
    axs[0,1].imshow(cln_cube[img_num,:,:],interpolation='none',clim=[0, inten],cmap='bone')
    axs[1,0].imshow(bin_cube[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    axs[1,1].imshow(clo_cube[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    plt.show(block=False)

# =============================================================================
# %% Dialte the Images in Omega - DO NOT EDIT
# =============================================================================
if dilate_omega:
    print("dilating in omega (this will take a few minutes)...")
    if remove_small_binary_objects == 1:
        dilated_cube = binary_dilation(clo_cube, iterations=1)
    else:
        dilated_cube = binary_dilation(bin_cube, iterations=1)
else:
    if remove_small_binary_objects == 1:
        dilated_cube = clo_cube
    else:
        dilated_cube = bin_cube

print("Dilation (or not doing it) is done")
# ==============================================================================
# %% Plot the Binarized, Dilated Images - Can be edited
# ==============================================================================
if not suppress_plots:
    fig, axs = plt.subplots(2,2)
    img_num = 99
    inten = binarization_threshold
    axs[0,0].imshow(img_cube[img_num,:,:],interpolation='none',clim=[0, inten],cmap='bone')
    axs[0,1].imshow(cln_cube[img_num,:,:],interpolation='none',clim=[0, inten],cmap='bone')
    axs[1,0].imshow(clo_cube[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    axs[1,1].imshow(dilated_cube[img_num,:,:],interpolation='none',clim=[0, 1],cmap='bone')
    plt.show(block=False)

# =============================================================================
# %% Save the Binarized Image Stack - DO NOT EDIT
# =============================================================================
if save_omegas_and_image_stack:
    np.save(output_dir + os.sep + output_stem + '_binarized_images.npy', dilated_cube)
    np.save(output_dir + os.sep + output_stem + '_omega_edges_deg.npy', omega_edges_deg)
print("Done saving")



































# %%
