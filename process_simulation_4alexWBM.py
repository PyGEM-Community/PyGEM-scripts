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


regions = [1,2,3,4,5,7,8,9,10,12,13,14,15,16,17,18]
rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # used to get initial mass
#regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # used to get initial mass

era5_zipped_fp_prefix = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/_zipped/'
era5_zipped_fp_prefix_updated = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/_zipped_era5/'

mass_fp_prefix = '/Users/drounce/Documents/HiMAT/spc_ultee/sims_aggregated/mass_annual/'

year_start = 2000

for reg in regions:
    era5_zipped_fp = era5_zipped_fp_prefix + str(reg).zfill(2) + '/stats/'
    era5_zipped_fn = 'ERA5__stats.zip'
    
    unzipped_fp =era5_zipped_fp + 'ERA5/'
    if not os.path.exists(unzipped_fp):
        os.makedirs(unzipped_fp)
    
#    # Unzip filepath
#    with zipfile.ZipFile(era5_zipped_fp + era5_zipped_fn, 'r') as zip_ref:
#        zip_ref.extractall(unzipped_fp)
        
    glac_fns = []
    for i in os.listdir(unzipped_fp):
        if i.endswith('.nc'):
            glac_fns.append(i)
    glac_fns = sorted(glac_fns)
    
    #%% 
    # Get initial mass
    mass_fp = mass_fp_prefix + str(reg).zfill(2) + '/'

    for nrcp, rcp in enumerate(rcps):        
        mass_fns = []
        mass_fns_int = []
        for i in os.listdir(mass_fp):
            if rcp in i:
                mass_fns.append(i)
                mass_fns_int.append(int(i.split('-')[-2]))
        mass_fns = [x for _,x in sorted(zip(mass_fns_int, mass_fns))]
    
        ds_reg = None
        for nfn, ds_fn in enumerate(mass_fns):
            print(ds_fn)
            ds_batch = xr.open_dataset(mass_fp + ds_fn)
            
            if ds_reg is None:
                ds_reg = ds_batch
            else:
                ds_reg = xr.concat([ds_reg, ds_batch], dim='glacier')
        
        if nrcp == 0:
            rgiids_list = list(ds_reg.RGIId.values)
            reg_glac_mass_init_all = np.mean(ds_reg.glac_mass_annual.values[:,:,0], axis=0)[:,np.newaxis]
        else:
            reg_glac_mass_init_all = np.concatenate((reg_glac_mass_init_all, np.mean(ds_reg.glac_mass_annual.values[:,:,0], axis=0)[:,np.newaxis]), axis=1)
        
    reg_glac_mass_init = reg_glac_mass_init_all.mean(1)
    reg_glac_vol_init = reg_glac_mass_init / 900
    
    rgiid_vol_init_dict = dict(zip(rgiids_list, list(reg_glac_vol_init)))
        
    year_idx = None
    glacno_all = []
    reg_masschg_2000_2019 = 0
    for nglac, glac_fn in enumerate(glac_fns):
        
        if nglac%100 == 0:
            print(glac_fn)
            
        updated_fp = unzipped_fp.replace('ERA5','ERA5-updated')
        if not os.path.exists(updated_fp):
            os.makedirs(updated_fp)
            
        if not os.path.exists(updated_fp + glac_fn):
#        for batman in [0]:
#        if glac_fn.startswith('6.00475'):
##            print('\n\nswitch back\n\n')
        
            ds = xr.open_dataset(unzipped_fp + glac_fn)
            glacno = glac_fn.split('_')[0]
            glacno_all.append(glacno)
            
            if int(glacno.split('.')[0]) < 10:
                rgiid = 'RGI60-0' + glacno
            else:
                rgiid = 'RGI60-' + glacno
            
            if rgiid in rgiids_list:
                glac_vol_init = rgiid_vol_init_dict[rgiid]
                
                if year_idx is None:
                    years = ds.year.values
                    year_idx = list(years).index(year_start)
        
                # Create glacier volume assuming a constant glacier area and thus adding/subtracting mass gain/lost
                glac_vol = np.zeros(years.shape) # annual vol
                glac_massbaltotal_annual = ds.glac_massbaltotal_monthly.values[0,:].reshape(-1,12).sum(axis=1)
                glac_vol_chg_annual = glac_massbaltotal_annual * 1000 / 900  # annual vol chg
                glac_vol_init = rgiid_vol_init_dict[rgiid]  # initial vol
        
                glac_vol[year_idx] = glac_vol_init
                
                # Subtract annual change to grow/shrink glacier as needed prior to initial glacier volume
                glac_vol[0:year_idx] = glac_vol_init - np.cumsum(glac_vol_chg_annual[:year_idx][::-1])[::-1]
                
                # Add annual change after initial time
                glac_vol[year_idx+1:] = glac_vol_init + np.cumsum(glac_vol_chg_annual[year_idx:])
                
                reg_masschg_2000_2019 += glac_massbaltotal_annual[year_idx:].sum()

                ds.glac_volume_annual.values[0,:] = glac_vol
                
                ds.to_netcdf(updated_fp + glac_fn)
                
    # Check regional mass change from 2000-2019
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no=glacno_all)
    area_all_m2 = main_glac_rgi_all.Area.sum()*1e6
    print('Reg', reg, 'MB (m w.e. yr-1):', reg_masschg_2000_2019/area_all_m2/20)
        
    # Zip directory
    if not os.path.exists(era5_zipped_fp_prefix_updated):
        os.makedirs(era5_zipped_fp_prefix_updated)
    shutil.make_archive(era5_zipped_fp_prefix_updated + str(reg).zfill(2) + '_ERA5_1979_2019', 'zip', updated_fp)
                
            