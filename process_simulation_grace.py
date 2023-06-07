#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:48:41 2022

@author: drounce
"""

# Built-in libraries
import collections
import glob
import os
import pickle
import shutil
import time
import zipfile
# External libraries
import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
import geopandas
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.ndimage import generic_filter
from scipy.ndimage import uniform_filter
import xarray as xr
# Local libraries
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

# ----- Processing options -----
option_process_binned2grid = True           # Process binned data to grid

# ----- Parameters -----
#regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
regions = [1]

normyear = 2015

# GCMs and RCP scenarios
gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
             'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']


time_start = time.time()

#%%
binned_fp_prefix = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-ind/'
oggm_gdirs_fp_prefix = '/Users/drounce/Documents/HiMAT/oggm_gdirs/per_glacier-all/'

for reg in regions:
    for rcp in rcps:
        for gcm_name in gcm_names:
            
            netcdf_fp = binned_fp_prefix + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
            
            netcdf_fns = []
            for i in os.listdir(netcdf_fp):
                if i.endswith('binned.nc'):
                    netcdf_fns.append(i)
            netcdf_fns = sorted(netcdf_fns)
            
            for netcdf_fn in netcdf_fns:
                ds = xr.open_dataset(netcdf_fp + netcdf_fn)
                
                #%%
                # Load DEM
                rgiid = ds.RGIId.values[0]
                
                oggm_gdirs_fp = oggm_gdirs_fp_prefix + rgiid[0:8] + '/' + rgiid[0:11] + '/' + rgiid + '/'
                dem_fn = 'dem.tif'
                
                #%%
                
                assert 1==0, 'here'
                    
