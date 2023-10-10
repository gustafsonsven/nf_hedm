# Convert from a Nexus file (.h5) to a .npz
# %% ================================================================================
# IMPORTS - DO NOT EDIT
#====================================================================================
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_bregman
from skimage.morphology import remove_small_objects
from skimage.transform import resize
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
import time
from hexrd.grainmap import nfutil_SEG as nfutil
import os
from nexusformat.nexus import nxload,nxsetconfig

# Multithreading libraries
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import multiprocessing as mp

# Matplotlib
# This is to allow interactivity of inline plots in your gui
# the import ipywidgets as widgets line is not needed - however, you do need to run a 
# pip install ipywidgets the import ipympl line is not needed - however, you do need 
# to run a pip install ipympl #import ipywidgets as widgets #import ipympl 
import matplotlib
# The next line is formatted correctly, no matter what your IDE says
%matplotlib widget
import matplotlib.pyplot as plt

# %% ================================================================================
# USER INPUT - CAN BE EDITED
#====================================================================================
# Where are we working?
working_dir = '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/shanks-3731-a/ti-13-exsitu/tomo/'
output_dir = working_dir
# Reconstructed data filepath
CHAP_file_path = working_dir + 'ti-13-exsitu_reconstructed_tomo.nxs' # Nexsus file with reconstructed tomo from CHAP
# Output file name?
output_stem = 'ti-13-exsitu'
# DBR string
dbr_str = 'shanks-3731-a'

# Detector image size in pixels
image_size = [2048,2048] # horizontal, vertical
# What is the desired cross section of the output array - should match that used in the NF reconstruction
desired_cross_section = 1.3 # mm
# Detector pixel size (not effective, the pre-optics value)
pixel_size = 0.0074 # mm
# Lens magnification
lens_magnification = 5.0
# Voxel size - to match that of NF
desired_voxel_size = 0.005 # mm
# Where is the horizontal center of the detector relative to the origin?
h_center = 0 # mm # This can be pulled from the NF detector calibration

# Non-Local Means Cleaning Parameters
patch_size = 1
patch_distance = 3

# Multipllicative value to apply to image thresholding
scaling = 0.25 # Trial and error found 0.25 is pretty good

ncpus = 128 # mp.cpu_count()

# %% ================================================================================
# LOAD DATA - DO NOT EDIT
#====================================================================================
# Load the main file
nxroot = nxload(CHAP_file_path)
nxsetconfig(memory=100000) # Needed since some of the files are fairly large
# The tomo data reads in with increasing indices proceeding along -Z,Y,X in the lab frame
raw_tomo_data = np.asarray(nxroot[dbr_str]['reconstructed_data']['data']['reconstructed_data'])
# What are the bounds in the vertical lab direction?
vertical_bounds = np.asarray(nxroot[dbr_str]['reduced_data']['img_row_bounds'])
# Grab the offset of the rotation axis
horizontal_offset = int(np.round((float(nxroot[dbr_str]['reconstructed_data']['center_offsets'][0])+
                                  float(nxroot[dbr_str]['reconstructed_data']['center_offsets'][1]))/2))
# Grab the X and Y bounds if the reconstruction was cropped in CHAP
x_bounds = np.asarray(nxroot[dbr_str]['reconstructed_data']['x_bounds'])
y_bounds = np.asarray(nxroot[dbr_str]['reconstructed_data']['y_bounds'])

# %% ================================================================================
# MANIPULATE ARRAY INTO THE LAB FRAME - DO NOT EDIT
#====================================================================================
# Define full array extent
full_array = np.zeros([np.shape(raw_tomo_data)[0],image_size[0],image_size[0]])

# Place the cropped reconstruction into the full array
full_array[:,y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]] = raw_tomo_data

# Manipulate into an XYZ lab frame
full_array = np.transpose(full_array,[2,1,0])
full_array = np.flip(full_array,2)

# Plot it real quick
slice_number = 100
plt.figure()
plt.imshow(full_array[:,:,slice_number])
plt.title('Raw Tomo Reconstruction')
plt.ylabel('X Position (pixels)')
plt.xlabel('Y Position (pixels)')
plt.show(block=False)
# %% ================================================================================
# DEFINE SPATIAL ARRAYS AND CROP ARRAY TO THE DESIRED SIZE - DO NOT EDIT
#====================================================================================
# Voxel size?
current_voxel_size = pixel_size/lens_magnification # mm

# What is the height of the reconstruciton
vertical_height = np.shape(full_array)[2] * current_voxel_size # mm

# Create position lists in the lab frame about zero
Xs_list = np.arange(-desired_cross_section/2.+current_voxel_size/2.,desired_cross_section/2.,current_voxel_size)
Ys_list = np.arange(-desired_cross_section/2.+current_voxel_size/2.,desired_cross_section/2,current_voxel_size)
Zs_list = np.arange(-vertical_height/2.+current_voxel_size/2.,vertical_height/2.,current_voxel_size)

# Vertical shift (Z)
# Are we offset from that center?
current_center_pixel = np.floor((vertical_bounds[1]+vertical_bounds[0])/2).astype(int)
# How far to shift?  # If negative our mask center is below the v_center, if postive it is above
vertical_shift = (current_center_pixel - image_size[1]/2)*current_voxel_size # mm 
# Shift
Zs_list = Zs_list + h_center + vertical_shift

# Crop data to desired size
if (np.shape(Xs_list)[0]/2)%2 == 0:
    start = int(image_size[0]/2 - np.round(np.shape(Xs_list)[0]/2))
    stop = int(image_size[0]/2 + np.round(np.shape(Xs_list)[0]/2))
else:
    start = int(image_size[0]/2 - np.round(np.shape(Xs_list)[0]/2) - 1)
    stop = int(image_size[0]/2 + np.round(np.shape(Xs_list)[0]/2))
cropped_data = full_array[start:stop,start:stop,:]

# Plot it real quick
plt.figure()
plt.imshow(cropped_data[:,:,200])
plt.title('Cropped Tomo Reconstruction')
plt.ylabel('X Position (pixels)')
plt.xlabel('Y Position (pixels)')
plt.show(block=False)

# %% ================================================================================
# SINGLE SLICE IMAGE PROCESSING CHECKER - CAN BE EDITED
#====================================================================================
# What slice number to look at - check over the entire range
slice_num = 200
# Pull that slice
raw_img = cropped_data[:,:,slice_num]

# Estimage the per-slice sigma
s_est = estimate_sigma(raw_img)
# Run non-local_means
img = denoise_nl_means(
    raw_img, sigma=s_est, h=0.8 * s_est, patch_size=patch_size, patch_distance=patch_distance,
    preserve_range = True)
# Denoise - this helps get rid of rough bits on the edges
img = denoise_tv_bregman(img,1)

# Binarize
threshold = (np.mean(img)+np.std(img)*3)*scaling # This has been found to be the most robust
mask = img>threshold

# Dilate, close holes, erode, then remove small voxels
mask = binary_dilation(mask)
mask = binary_dilation(mask)
mask = binary_fill_holes(mask)
mask = binary_erosion(mask)
mask = binary_erosion(mask)
mask = remove_small_objects(mask,10000)

fig, axs = plt.subplots(1,2)
axs[0].imshow(img,clim=[-threshold*2,threshold*2],interpolation='none')
axs[0].title.set_text('Layer %d Raw Image' % slice_num)
axs[1].imshow(mask,interpolation='none',clim=[0, 1],cmap='bone')
axs[1].title.set_text('Layer %d Processed Image' % slice_num)
plt.show(block=False)


# %% ================================================================================
# ALL SLICES IMAGE PROCESSING - DO NOT EDIT
#====================================================================================
def per_slice_raw_to_binary(slice_num,scaling=0.25):
    # Pull img
    raw_img = cropped_data[:,:,slice_num]
    # Estimage the per-slice sigma
    s_est = estimate_sigma(raw_img)
    # Run non-local_means
    img = denoise_nl_means(
        raw_img, sigma=s_est, h=0.8 * s_est, patch_size=patch_size, patch_distance=patch_distance,
        preserve_range = True)
    # Denoise - this helps get rid of rough bits on the edges
    img = denoise_tv_bregman(img,1)
    # Binarize
    threshold = (np.mean(img)+np.std(img)*3)*scaling
    bin_img = img>threshold
    # Dilate, close holes, erode, then remove small voxels
    bin_img = binary_dilation(bin_img)
    bin_img = binary_dilation(bin_img)
    bin_img = binary_fill_holes(bin_img)
    bin_img = binary_erosion(bin_img)
    bin_img = binary_erosion(bin_img)
    bin_img = remove_small_objects(bin_img,10000) # This is specifically very large to remove everything but the central blob
    return slice_num, bin_img

# Define mask array
full_mask = np.zeros(np.shape(cropped_data),dtype=np.uint8)

# Startup the processor and run
executor = ThreadPoolExecutor(ncpus)
xx = np.arange(np.shape(cropped_data)[2])

# Define the cleanup routine
futures = {executor.submit(per_slice_raw_to_binary,x, scaling=scaling): x for x in xx}

# Run the routine
print("Starting denoising and binarization.")
i = 0
tic = time.time()
for future in as_completed(futures):
    x, binarized_slice = future.result()
    full_mask[:, :, x] = binarized_slice
    inputs = futures.pop(future)  # this stops memory leaks
    i += 1
    if i % 25 == 0 or i == xx[-1]:
        print("{} of {} layers cleaned".format(i, np.shape(cropped_data)[2]))
    del inputs, x, binarized_slice  # this is just a cleanup
tocC = time.time() - tic
print(tocC)
executor.shutdown()

print("Masking is done.")

# %% ================================================================================
# PLOTTING - CAN BE EDITED
#====================================================================================
slice_num = 649
raw_img = cropped_data[:,:,slice_num]
mask_img = full_mask[:,:,slice_num]
threshold = np.max(raw_img)*scaling

fig, axs = plt.subplots(1,2)
axs[0].imshow(raw_img,clim=[-threshold*2,threshold*2],interpolation='none')
axs[0].title.set_text('Layer %d Raw Image' % slice_num)
axs[1].imshow(mask_img,interpolation='none',clim=[0, 1],cmap='bone')
axs[1].title.set_text('Layer %d Processed Image' % slice_num)
plt.show(block=False)

# %% ================================================================================
# SAVE THE FULL MASK FOR VISUALIZATION - DO NOT EDIT
#====================================================================================
# This may be a little large to open in Paraview
# You can always keep going to produce the coarsened mask to look at

# Save as an h5 with xdmf for Paraview
nfutil.write_to_h5(output_stem,output_stem+'_full_mask',full_mask,'tomo_mask')
nfutil.xmdf_writer(output_stem,output_stem+'_full_mask')

# Save as an .npz for loading
np.savez(os.path.join(output_stem,output_stem+'_full_mask') + '.npz',tomo_mask=full_mask)

# %% ================================================================================
# RESIZE ARRAY TO MATCH NF - DO NOT EDIT
#====================================================================================
# What should the dimensions be?
new_dims = np.round(np.divide([desired_cross_section,desired_cross_section,vertical_height],desired_voxel_size)).astype(np.int32)

# Coarsen the mask
coarse_mask = resize(full_mask,new_dims,preserve_range=True,anti_aliasing=True)
coarse_mask = coarse_mask > 0.5

# Plot
original_slice_number = 200
new_slice_number = int(original_slice_number/np.shape(full_mask)[2]*new_dims[2])
fig, axs = plt.subplots(1,2)
axs[0].imshow(full_mask[:,:,original_slice_number],interpolation='none',clim=[0, 1],cmap='bone')
axs[0].title.set_text('Layer %d Original Mask' % 0)
axs[1].imshow(coarse_mask[:,:,new_slice_number],interpolation='none',clim=[0, 1],cmap='bone')
axs[1].title.set_text('Layer %d Coarse Mask' % 0)
plt.show(block=False)

# %% ================================================================================
# TRANSFORM INTO THE HEXRD FRAME - DO NOT EDIT
#====================================================================================
# What is our current frame?  X,Y,Z
# What is this in the HEXRD frame?  X,-Z,Y

# Initialize
coarse_mask_hexrd = np.copy(coarse_mask)

# Flip Z and y
coarse_mask_hexrd = np.transpose(coarse_mask_hexrd,[0,2,1])

original_slice_number = 200
new_slice_number = int(original_slice_number/np.shape(full_mask)[2]*new_dims[2])
fig, axs = plt.subplots(1,2)
axs[0].imshow(coarse_mask[:,:,new_slice_number],interpolation='none',clim=[0, 1],cmap='bone')
axs[0].title.set_text('Layer %d Shifted Mask' % 0)
axs[1].imshow(coarse_mask_hexrd[:,new_slice_number,:],interpolation='none',clim=[0, 1],cmap='bone')
axs[1].title.set_text('Layer %d Corse Mask' % 0)
plt.show(block=False)
# %% ================================================================================
# CREATE NEW POSITION ARRAYS - DO NOT EDIT
#====================================================================================
X_hexrd_list = resize(Xs_list,[new_dims[0],1],preserve_range=True,anti_aliasing=True)
Y_hexrd_list = resize(Zs_list,[new_dims[2],1],preserve_range=True,anti_aliasing=True)
Z_hexrd_list = resize(Ys_list,[new_dims[1],1],preserve_range=True,anti_aliasing=True)
X_hexrd,Y_hexrd,Z_hexrd=np.meshgrid(X_hexrd_list,Y_hexrd_list,Z_hexrd_list,indexing='ij')


# %% ================================================================================
# Save mask for use later and for visualization
#====================================================================================
save_name = output_stem + '_' + str(desired_voxel_size) + '_tomo_mask'
# Paraview reads in data as XYZ so transpose the two when writing
# Save as an h5 with xdmf for Paraview
nfutil.write_to_h5(output_dir,save_name,np.transpose(coarse_mask_hexrd,[2,1,0]),'tomo_mask')
nfutil.write_to_h5(output_dir,save_name,np.transpose(X_hexrd,[2,1,0]),'Xh')
nfutil.write_to_h5(output_dir,save_name,np.transpose(Y_hexrd,[2,1,0]),'Yh')
nfutil.write_to_h5(output_dir,save_name,np.transpose(Z_hexrd,[2,1,0]),'Zh')
nfutil.xmdf_writer(output_dir,save_name)

# Save as an .npz for loading
np.savez(os.path.join(output_dir,save_name) + '.npz',mask=np.transpose(coarse_mask_hexrd,[1,0,2]),
         Xs=np.transpose(X_hexrd,[1,0,2]),Ys=np.transpose(Y_hexrd,[1,0,2]),
         Zs=np.transpose(Z_hexrd,[1,0,2]),voxel_spacing=0.005)




# %% 








