#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 2023

@author: seg246
"""

"""
Goal: Using nf_stich_grains_functions_DJS_SEG.py as a function set, streamline the
workflow for running NF/FF with multiple diffraction volumes

Step 1: Create one unified grains.out to feed back into NF
    This, in general, helps to grab grains which sit at the boundary between volumes
        which may have not been picked up in every FF indexing.  It also produces a
        unified grains.out.  
Step 2: Assuming NF was ran with this unified grains.out
    This fills in those gaps at the boundary if grains were missed between indexings.  
Step 3: Merge the NF volumes into one grain map and run a basic cleanup
    This is an intermediate grain map where we have not yet serached for grains.  
        Here, since the previous step may have ended up with a single grain being 
        labeled with multiple grain IDs (if orientations are simialr enough), run
        a cleanup that isloates similarly oriented grain volumes
Step 4: Create a mask with low confidence regions?  Blob it?  Paraview?
Step 5: ????
Step 6: Profit
"""

# %% ============================================================================
# Imports
# ===============================================================================
import numpy as np
import os
import hexrd.grainmap.nfutil_SEG as nfutil
import importlib
importlib.reload(nfutil) # This reloads the file if you made changes to it

# Matplotlib
# This is to allow interactivity of inline plots in your gui
# the import ipywidgets as widgets line is not needed - however, you do need to run a pip install ipywidgets
# the import ipympl line is not needed - however, you do need to run a pip install ipympl
#import ipywidgets as widgets
#import ipympl 
import matplotlib
# The next line is formatted correctly, no matter what your IDE says
%matplotlib widget
import matplotlib.pyplot as plt

from hexrd import material
from hexrd import valunits
from hexrd import constants
from hexrd import instrument
#from hexrd.cli.fit_grains import write_results
import yaml

def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)

# %% ============================================================================
# User Inputs
# ===============================================================================
# Directory and file setup may be user specific, change as needed to work with your directly setup
working_dir = '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/shanks-3731-a/ti-13-exsitu/nf/'
# Naming of each layer to read in - these need to be in the output folders of each folder containing each layer
# The below two variables I concatate into start + folder_id + end
individual_nf_layer_stem_start = 'ti-13-exsitu_layer_'
individual_nf_layer_stem_end = '_merged_grain_map_data.npz'
folder_ids = [1,2,3,4,5,6,7,8,9,10] # Folder ids for each separate nf layer (there is an output folder within each - this works for my folder construction

# You can manually make the paths below if you want!

"""
My Folder Construction

/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/nygren-3738-a/ti7al-9-1/nf
    1
        output
            ti7al-9-1_1_secondary_indexing_grain_map_data.npz
    2
        output
            ti7al-9-1_2_secondary_indexing_grain_map_data.npz
    3
        output
            ti7al-9-1_3_secondary_indexing_grain_map_data.npz
    4
        output
            ti7al-9-1_4_secondary_indexing_grain_map_data.npz

"""

# How are the scans spatially separated
# Use ramsz positions
offsets = np.linspace(-0.405,0.405,10) # What were your ramsz positions, the last number is the total number of layers

# Material and detector info
det_file = working_dir + '1/retiga.yml'
mat_file = working_dir + '1/materials.h5'
energy = 41.991
mat_name = 'ti7al'

# Parameters
orientation_tolerance = 0.25 # Deg
layer_overlap = 0 # In voxels so 2 means I have two voxels worth of overlap on either size a single layer

# Output names
output_dir = working_dir
output_stem = 'merged_2023_09_14' # What do you want to call the output h5?
merged_grains_out_name = 'no_merged_2023_09_14.out'

# Some flags
save_h5 = 1 # If zero the funciton will only return the exp_maps, if 1 it will also save a paraview readable h5
use_mask = 1 # If you have a mask make a 1, else 0

# %% ============================================================================
# Function Call
# ===============================================================================
# Generate data path names
paths = []
for scan in np.arange(len(folder_ids)):
    paths.append(os.path.join(working_dir,str(folder_ids[scan]),'output',individual_nf_layer_stem_start+
                              str(folder_ids[scan])+individual_nf_layer_stem_end))
    
# # Manually make paths if you need to
# paths =['/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/nygren-3738-a/ti7al-9-1/nf/1/output/ti7al-9-1_1_secondary_indexing_grain_map_data.npz',
#         '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/nygren-3738-a/ti7al-9-1/nf/2/output/ti7al-9-1_2_secondary_indexing_grain_map_data.npz',
#         '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/nygren-3738-a/ti7al-9-1/nf/3/output/ti7al-9-1_3_secondary_indexing_grain_map_data.npz',
#         '/nfs/chess/aux/reduced_data/cycles/2023-2/id3a/nygren-3738-a/ti7al-9-1/nf/4/output/ti7al-9-1_4_secondary_indexing_grain_map_data.npz']

# Initialize some details
beam_energy = valunits.valWUnit("beam_energy", "energy", energy, "keV")
beam_wavelength = constants.keVToAngstrom(beam_energy.getVal('keV'))
instr=load_instrument(det_file)
max_pixel_tth = instrument.max_tth(instr)
dmin = valunits.valWUnit("dmin", "length",
                            0.5*beam_wavelength/np.sin(0.5*max_pixel_tth),
                            "angstrom")
mats = material.load_materials_hdf5(mat_file, dmin=dmin,kev=beam_energy)
mat = mats[mat_name]
# Run the stitching and output an h5
exp_maps = nfutil.stitch_nf_diffraction_volumes(output_dir,output_stem,paths,mat,
                               offsets, ori_tol=0.25, overlap=layer_overlap, save_h5=save_h5,
                               use_mask=1,average_orientation=0,remove_small_grains_under=3,voxel_size=0.005,save_npz=1)
print('All Done!')

# %% ============================================================================
# Save a merged grains.out
# ===============================================================================
gw = instrument.GrainDataWriter(
    os.path.join(working_dir, merged_grains_out_name)
)
for gid, ori in enumerate(exp_maps):
    grain_params = np.hstack([ori, constants.zeros_3, constants.identity_6x1])
    gw.dump_grain(gid, 1., 0., grain_params)
gw.close()



















# %%
