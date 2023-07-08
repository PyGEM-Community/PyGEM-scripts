#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export subset of climate data for land-terminating and tidewater glacier tests
"""

import os
import sys
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
# Local libraries
try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

#%%
main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no = pygem_prms.glac_no,
        rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
        rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
        include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater, 
        min_glac_area_km2=pygem_prms.min_glac_area_km2)

subset_fp = pygem_prms.output_filepath + '/subset_ERA5_data/'
if not os.path.exists(subset_fp):
    os.makedirs(subset_fp)
    
fns = [pygem_prms.era5_elev_fn, pygem_prms.era5_temp_fn, pygem_prms.era5_tempstd_fn, pygem_prms.era5_prec_fn, pygem_prms.era5_lr_fn]
for fn in fns:
    ds = xr.open_dataset(pygem_prms.era5_fp + fn)
    
    
    # Find nearest indices
    lat_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lat_colname].values[:,np.newaxis] - 
                          ds.variables['latitude'][:].values).argmin(axis=1))
    lon_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lon_colname].values[:,np.newaxis] - 
                          ds.variables['longitude'][:].values).argmin(axis=1))
    
    latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
    latlon_nearidx_unique = list(set(latlon_nearidx))
    
    ds_subset = ds.isel(latitude=lat_nearidx, longitude=lon_nearidx)
    ds_subset.to_netcdf(subset_fp + fn)
    
    
#%%
# Need to produce orog, pr, tas
name = 'CESM2'
scenario = 'ssp245'

# Variable names
temp_vn = 'tas'
prec_vn = 'pr'
elev_vn = 'orog'
lat_vn = 'lat'
lon_vn = 'lon'
time_vn = 'time'
# Variable filenames
temp_fn = name + '_' + scenario + '_r1i1p1f1_' + temp_vn + '.nc'
prec_fn = name + '_' + scenario + '_r1i1p1f1_' + prec_vn + '.nc'
elev_fn = name + '_' + elev_vn + '.nc'
# Variable filepaths
var_fp = pygem_prms.cmip6_fp_prefix + name + '/'
fx_fp = pygem_prms.cmip6_fp_prefix + name + '/'
# Extra information
timestep = pygem_prms.timestep
rgi_lat_colname=pygem_prms.rgi_lat_colname
rgi_lon_colname=pygem_prms.rgi_lon_colname

vns = ['temp', 'prec', 'elev']
for vn in vns:
    if vn == 'temp':
        gcm_fp = var_fp
        gcm_fn = temp_fn
    elif vn == 'prec':
        gcm_fp = var_fp
        gcm_fn = prec_fn
    elif vn == 'elev':
        gcm_fp = fx_fp
        gcm_fn = elev_fn
        
    ds = xr.open_dataset(gcm_fp + gcm_fn)
    
    # Find nearest indices
    lat_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lat_colname].values[:,np.newaxis] - 
                          ds.variables[lat_vn][:].values).argmin(axis=1))
    lon_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lon_colname].values[:,np.newaxis] - 
                          ds.variables[lon_vn][:].values).argmin(axis=1))
    
    latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
    latlon_nearidx_unique = list(set(latlon_nearidx))
    
    ds_subset = ds.isel(lat=lat_nearidx, lon=lon_nearidx)
    
    subset_fp = pygem_prms.output_filepath + '/subset_cmip6/' + name + '/'
    if not os.path.exists(subset_fp):
        os.makedirs(subset_fp)

    ds_subset.to_netcdf(subset_fp + gcm_fn)
