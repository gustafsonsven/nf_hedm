"""
contributing authors: dcp5303, ken38, seg246, Austin Gerlt, Simon Mason
"""
# %% ============================================================================
# IMPORTS
# ===============================================================================
# General imports
import os
import logging
import h5py
import numpy as np
import numba
import yaml
import argparse
import timeit
import contextlib
import multiprocessing
import tempfile
import shutil
import math
import scipy
import skimage
import copy
import glob
import json
import pandas as pd
import re
from collections import defaultdict

# HEXRD Imports
from hexrd import constants
from hexrd import instrument
from hexrd import material
from hexrd import rotations
from hexrd.transforms import xfcapi
from hexrd import valunits
from hexrd import xrdutil
from hexrd.sampleOrientations import sampleRFZ
import nf_config

# Matplotlib
# This is to allow interactivity of inline plots in your gui
# the import ipywidgets as widgets line is not needed - however, you do need to run a pip install ipywidgets
# the import ipympl line is not needed - however, you do need to run a pip install ipympl
#import ipywidgets as widgets
#import ipympl 
import matplotlib
# The next lines are formatted correctly, no matter what your IDE says
# For inline, interactive plots (if you use these, make sure to run a plt.close() to prevent crashing)
# %matplotlib widget
# For inline, non-interactive plots
# %matplotlib inline
# For pop out, interactive plots (cannot be used with an SSH tunnel)
# %matplotlib qt
import matplotlib.pyplot as plt

# Yaml loader
def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)

# Constants
beam = constants.beam_vec
Z_l = constants.lab_z
vInv_ref = constants.identity_6x1

# This is here for grabbing when needed in other scripts
# import importlib
# importlib.reload(nfutil) # This reloads the file if you made changes to it

from src.process_controller import ProcessController







