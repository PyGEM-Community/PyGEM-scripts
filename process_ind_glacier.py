#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 23:41:02 2022

@author: drounce
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

regions = [1]
glacnos = ['1.00570']

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

netcdf_fp_prefix = '/Users/drounce/Documents/HiMAT/spc_backup/nsidc/glacier_stats_nsidc_compliant/'
netcdf_fp_prefix_runoff = '/Users/drounce/Documents/HiMAT/spc_ultee/sims_aggregated/runoff_monthly/'
csv_fp = '/Users/drounce/Documents/HiMAT/spc_backup/matt_nolan/'

process_mass_change = True
process_runoff = False

if process_mass_change:
    for region in regions:
        for scenario in scenarios:
            
            netcdf_fp = netcdf_fp_prefix + 'mass_annual/' + str(region).zfill(2) + '/'
            netcdf_fn = 'R' + str(region).zfill(2) + '_glac_mass_annual_c2_ba1_50sets_2000_2100-' + scenario + '.nc'
            ds = xr.open_dataset(netcdf_fp + netcdf_fn)
                
            for glacno in glacnos:
                rgiid_reg, rgiid_no = glacno.split('.')[0], glacno.split('.')[1]
                rgiid = 'RGI60-' + str(rgiid_reg).zfill(2) + '.' + rgiid_no

                rgiids_all = list(ds.RGIId.values)
                ds_idx = rgiids_all.index(rgiid)
                
                glac_mass_annual = ds.glac_mass_annual.values[:,ds_idx,:]
                glac_mass_annual_df = pd.DataFrame(glac_mass_annual, index=list(ds.Climate_Model.values), columns=list(ds.year.values))

                if not os.path.exists(csv_fp):
                    os.makedirs(csv_fp)
                csv_fn = rgiid + '_glac_mass_annual_kg_' + scenario + '.csv'
                
                glac_mass_annual_df.to_csv(csv_fp + csv_fn)
                
if process_runoff:
    for region in regions:
        for scenario in scenarios:
            
            netcdf_fp = netcdf_fp_prefix_runoff + str(region).zfill(2) + '/'
            
            for glacno in glacnos:
                rgiid_reg, rgiid_no = glacno.split('.')[0], glacno.split('.')[1]
                rgiid = 'RGI60-' + str(rgiid_reg).zfill(2) + '.' + rgiid_no
                
                batch_low = str(int(np.floor(int(rgiid_no)/1000)*1000+1))
                batch_high = str(int(np.ceil(int(rgiid_no)/1000)*1000))
                
                netcdf_fn = ('R' + str(region).zfill(2) + '_runoff_monthly_c2_ba1_1set_2000_2100-' + scenario + 
                             '-Batch-' + batch_low + '-' + batch_high + '.nc')
                ds = xr.open_dataset(netcdf_fp + netcdf_fn)
                
                rgiids_all = list(ds.RGIId.values)
                ds_idx = rgiids_all.index(rgiid)
                
                runoff_monthly = ds.glac_runoff_monthly.values[:,ds_idx,:]
                runoff_monthly_df = pd.DataFrame(runoff_monthly, index=list(ds.Climate_Model.values), columns=list(ds.time.values))

                if not os.path.exists(csv_fp):
                    os.makedirs(csv_fp)
                csv_fn = rgiid + '_glac_runoff_monthly_m3_' + scenario + '.csv'
                
                runoff_monthly_df.to_csv(csv_fp + csv_fn)