#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
original author: seg246
"""

"""
    Notes:
        - 
"""

# %% ============================================================================
# IMPORTS - DO NOT EDIT
# ===============================================================================
import numpy as np
import os

# HEXRD imports
from hexrd import material
from hexrd import valunits
from hexrd import constants
from hexrd import instrument
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

# File loader imports
import yaml
def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)

# %% ============================================================================
# USER INFORMATION - CAN BE EDITED
# ===============================================================================
# Working directory - could be of the form: '/nfs/chess/aux/reduced_data/cycles/[cycle ID]/[beamline]/BTR/sample/YOUR FAVORITE BOOKKEEPING STRUCTURE'
working_dir = '/your/path/here'

# Naming of each layer to read in - these need to be in the output folders of each folder containing each layer
# The below two variables I concatate into start + folder_id + end
individual_nf_layer_stem_start = 'stem_start_'
individual_nf_layer_stem_end = '_stem_finish.npz'
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
det_file = working_dir + '/1/retiga.yml'
mat_file = working_dir + '/1/materials.h5'
energy = 41.991
mat_name = 'ti7al'

# Parameters
orientation_tolerance = 0.25 # Deg
layer_overlap = 0 # In voxels so 2 means you have two voxels worth of overlap on either size a single layer

# Output names
output_dir = working_dir
output_stem = 'merged_sample_1_name' # What do you want to call the output h5?

# Some flags
save_h5 = 1 # If zero the funciton will only return the exp_maps, if 1 it will also save a paraview readable h5
use_mask = 1 # If you have a mask make a 1, else 0
save_grains_out = 1 # Do you want a grains.out file made?
single_or_multiple_grains_out_files = 0 # If 0, a single grains.out for the volume with save, if 1 a grains.out for each diffraction volume will print 
# %% ============================================================================
# PATHS PREP - CAN BE EDITIED (if you need to manually input file paths)
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

# %% ============================================================================
# VARIABLE PREP - DO NOT EDIT
# ===============================================================================
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

# %% ============================================================================
# FUNCTION CALL - DO NOT EDIT
# ===============================================================================
# Run the stitching and output an h5
exp_maps = nfutil.stitch_nf_diffraction_volumes(output_dir,output_stem,paths,mat,
                               offsets, ori_tol=0.25, overlap=layer_overlap, save_h5=save_h5,
                               use_mask=1,average_orientation=0,remove_small_grains_under=2,
                               voxel_size=0.005,save_npz=1,save_grains_out=save_grains_out,
                               single_or_multiple_grains_out_files=0)


















# %%
